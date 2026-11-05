"""Explicit inference precision; no shared patches or global backend settings."""
from pathlib import Path
import platform

import torch

from wtpsplit.extract import PyTorchWrapper


def cpu_bf16_support():
    """Conservative native-instruction check for the supported Linux x86 path.

    Flags do not prove a particular kernel is selected or a workload is faster.
    Do not silently time software emulation as native BF16 support.
    """
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64', 'i686'):
        return dict(supported=False, reason='Native CPU BF16 detection currently supports Linux x86 only', flags=[])
    try:
        rows = [set(line.split(':', 1)[1].split()) for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('flags')]
    except OSError:
        rows = []
    flags = sorted(set.intersection(*rows)) if rows else []
    supported = bool({'avx512_bf16', 'amx_bf16'} & set(flags))
    return dict(supported=supported, flags=flags,
                reason='Native BF16 instruction flags present' if supported else 'Native CPU BF16 instructions unavailable')


_FP8_E4M3_MAX = 448.0


def gpu_fp8_support(device='cuda'):
    """Check native Ada Lovelace (SM89, e.g. L40S) or Hopper (SM90, e.g. H100) FP8 Tensor Core support."""
    target = torch.device('cuda' if device is None else device)
    if target.type != 'cuda' or not torch.cuda.is_available():
        return dict(supported=False, capability=(0, 0), gpu='',
                    reason='CUDA device required for FP8 Tensor Core inference')
    cap = torch.cuda.get_device_capability(target)
    gpu_name = torch.cuda.get_device_name(target)
    has_e4m3 = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, '_scaled_mm')
    supported = bool(has_e4m3 and cap >= (8, 9))
    reason = (
        f'Native FP8 (float8_e4m3fn) Tensor Cores supported on {gpu_name} (SM{cap[0]}{cap[1]})'
        if supported
        else f'GPU {gpu_name} (SM{cap[0]}{cap[1]}) lacks native SM89+ FP8 Tensor Cores'
    )
    return dict(supported=supported, capability=cap, gpu=gpu_name, reason=reason)


class FP8DynamicLinear(torch.nn.Module):
    """Row-wise (`per-token` `[M, 1]` × `per-output-channel` `[N, 1]`) dynamic `float8_e4m3fn` linear layer."""

    def __init__(self, weight_fp8: torch.Tensor, scale_w: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.out_features, self.in_features = int(weight_fp8.size(0)), int(weight_fp8.size(1))
        self.register_buffer('weight_fp8', weight_fp8.contiguous())
        self.register_buffer('scale_w', scale_w.reshape(self.out_features, 1).to(dtype=torch.float32))
        self.register_buffer('_unit_scale', torch.ones(1, dtype=torch.float32))
        if bias is not None:
            self.register_buffer('bias', bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_weight_bias(cls, weight: torch.Tensor, bias: torch.Tensor | None) -> 'FP8DynamicLinear':
        with torch.no_grad():
            w_f32 = weight.detach().float()
            w_amax = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale_w = (w_amax / _FP8_E4M3_MAX).to(dtype=torch.float32)
            w_fp8 = (w_f32 / scale_w).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            b_out = bias.detach().clone() if bias is not None else None
            return cls(w_fp8, scale_w, b_out)

    @classmethod
    def from_float(cls, linear: torch.nn.Linear) -> 'FP8DynamicLinear':
        return cls.from_weight_bias(linear.weight, linear.bias)

    @classmethod
    def from_qkv(cls, q_lin: torch.nn.Linear, k_lin: torch.nn.Linear, v_lin: torch.nn.Linear) -> 'FP8DynamicLinear':
        with torch.no_grad():
            w_cat = torch.cat([q_lin.weight.detach(), k_lin.weight.detach(), v_lin.weight.detach()], dim=0)
            b_cat = (
                torch.cat([q_lin.bias.detach(), k_lin.bias.detach(), v_lin.bias.detach()], dim=0)
                if (q_lin.bias is not None and k_lin.bias is not None and v_lin.bias is not None)
                else None
            )
            return cls.from_weight_bias(w_cat, b_cat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        out_dt = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
        x_2d = x.reshape(-1, self.in_features)
        m_rows = int(x_2d.size(0))
        if m_rows == 0:
            return torch.empty((*orig_shape[:-1], self.out_features), dtype=out_dt, device=x.device)

        x_f32 = x_2d.float()
        x_amax = x_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale_x = (x_amax / _FP8_E4M3_MAX).to(dtype=torch.float32)
        x_fp8 = (x_f32 / scale_x).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        if (
            x.is_cuda
            and hasattr(torch, '_scaled_mm')
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
        ):
            rem = m_rows % 16
            x_mm = torch.nn.functional.pad(x_fp8, (0, 0, 0, 16 - rem)) if rem != 0 else x_fp8
            w_col_major = self.weight_fp8.contiguous().t()
            mm_dt = torch.float16 if out_dt == torch.float16 else torch.bfloat16
            raw_2d = torch._scaled_mm(
                x_mm,
                w_col_major,
                scale_a=self._unit_scale,
                scale_b=self._unit_scale,
                out_dtype=mm_dt,
            )
            if isinstance(raw_2d, tuple):
                raw_2d = raw_2d[0]
            if rem != 0:
                raw_2d = raw_2d[:m_rows]
            out_2d = raw_2d.float() * (scale_x * self.scale_w.t())
            if self.bias is not None:
                out_2d.add_(self.bias.float())
            return out_2d.to(dtype=out_dt).reshape(*orig_shape[:-1], self.out_features)

        x_deq = x_fp8.to(dtype=torch.float32) * scale_x
        w_deq = self.weight_fp8.to(dtype=torch.float32) * self.scale_w
        b_f32 = self.bias.to(dtype=torch.float32) if self.bias is not None else None
        out_f32 = torch.nn.functional.linear(x_deq, w_deq, b_f32)
        return out_f32.to(dtype=out_dt).reshape(*orig_shape[:-1], self.out_features)


def quantize_fp8_roberta(model: torch.nn.Module) -> int:
    """Convert XLM-R encoder Linear modules to `FP8DynamicLinear` and pre-fuse QKV projections."""
    roberta = getattr(model, 'roberta', None)
    if roberta is None or not hasattr(roberta, 'encoder'):
        raise ValueError('FP8 quantization requires an XLM-R roberta encoder')
    converted = 0
    for layer_module in roberta.encoder.layer:
        self_attn = layer_module.attention.self
        if (
            isinstance(self_attn.query, torch.nn.Linear)
            and isinstance(self_attn.key, torch.nn.Linear)
            and isinstance(self_attn.value, torch.nn.Linear)
        ):
            self_attn.__dict__['_fused_fp8_qkv'] = FP8DynamicLinear.from_qkv(
                self_attn.query, self_attn.key, self_attn.value
            )
            self_attn.query = FP8DynamicLinear.from_float(self_attn.query)
            self_attn.key = FP8DynamicLinear.from_float(self_attn.key)
            self_attn.value = FP8DynamicLinear.from_float(self_attn.value)
            converted += 3
        if isinstance(layer_module.attention.output.dense, torch.nn.Linear):
            layer_module.attention.output.dense = FP8DynamicLinear.from_float(layer_module.attention.output.dense)
            converted += 1
        if isinstance(layer_module.intermediate.dense, torch.nn.Linear):
            layer_module.intermediate.dense = FP8DynamicLinear.from_float(layer_module.intermediate.dense)
            converted += 1
        if isinstance(layer_module.output.dense, torch.nn.Linear):
            layer_module.output.dense = FP8DynamicLinear.from_float(layer_module.output.dense)
            converted += 1
    return converted


def validate_precision(precision, *, device, quantization, ort_providers, compile, from_pretrained_kwargs):
    """Reject unsupported/conflicting choices before loading checkpoint files."""
    if precision not in (None, 'fp32', 'fp16', 'bf16', 'fp8'):
        raise ValueError("precision must be None, 'fp32', 'fp16', 'bf16', or 'fp8'")
    if precision is None:
        return None
    if ort_providers is not None:
        raise ValueError('precision configures PyTorch inference; ONNX precision is determined by the exported graph')
    if quantization is not None:
        raise ValueError('Choose precision or INT8 quantization, not both')
    if any(k in (from_pretrained_kwargs or {}) for k in ('dtype', 'torch_dtype')):
        raise ValueError('Choose precision or a dtype in from_pretrained_kwargs, not both')
    target = torch.device('cpu' if device is None else device)
    if precision == 'fp32':
        return None
    if compile is not False:
        raise ValueError('FP16/BF16/FP8 inference precision is not yet supported with compile')
    if precision == 'fp8':
        support = gpu_fp8_support(target)
        if not support['supported']:
            raise ValueError(f"precision='fp8' requires native Ada/Hopper FP8 Tensor Cores: {support['reason']}")
        return support
    if precision == 'fp16':
        if target.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError("precision='fp16' requires an available CUDA device; use precision='fp32' on CPU")
        # Resolve the requested index before downloading/loading any weights.
        torch.cuda.get_device_properties(target)
        return dict(gpu=torch.cuda.get_device_name(target))
    if target.type != 'cpu':
        raise ValueError("precision='bf16' currently supports CPU autocast only; use precision='fp16' on CUDA")
    support = cpu_bf16_support()
    if not support['supported']:
        raise ValueError(f"precision='bf16' requires native CPU BF16 support: {support['reason']}; use FP32 or CPU INT8")
    return support


class PrecisionInferenceWrapper(PyTorchWrapper):
    """GPU half weights or CPU BF16 autocast, with FP32 output/stitching."""

    def __init__(self, model, precision, hardware):
        super().__init__(model, autocast_device='cpu' if precision == 'bf16' else 'cuda',
                         autocast_dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
                         autocast_enabled=precision == 'bf16',
                         output_float32=True)
        self.precision = precision
        self.precision_info = dict(precision=precision, hardware=hardware, output_dtype='torch.float32',
                                   autocast=precision == 'bf16', scope='Inference only; source checkpoint unchanged')
        self._refresh_info()

    def _refresh_info(self):
        param = next(self.model.parameters())
        self.precision_info.update(device=str(param.device), weight_dtype=str(param.dtype),
                                   compute_dtype='torch.float16' if self.precision == 'fp16' else 'torch.bfloat16')
        if self.precision == 'fp16':
            self.precision_info['hardware']['gpu'] = torch.cuda.get_device_name(param.device)

    def __call__(self, *args, **kwargs):
        param = next(self.model.parameters())
        expected_device = 'cuda' if self.precision == 'fp16' else 'cpu'
        expected_dtype = torch.float16 if self.precision == 'fp16' else torch.float32
        if param.device.type != expected_device or param.dtype != expected_dtype or self.model.training:
            raise ValueError('Inference precision state changed; reload with the desired device/precision and adapt in FP32')
        return super().__call__(*args, **kwargs)

    def to(self, device=None, *, dtype=None, **kwargs):
        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError('Specify dtype once')
            dtype, device = device, None
        expected = torch.float16 if self.precision == 'fp16' else torch.float32
        target = next(self.model.parameters()).device if device is None else torch.device(device)
        expected_device = 'cuda' if self.precision == 'fp16' else 'cpu'
        if target.type != expected_device or dtype not in (None, expected):
            raise ValueError('Changing inference precision/device type requires reloading the original checkpoint')
        self.model.to(device=target, dtype=expected, **kwargs)
        self._refresh_info()
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None):
        target = 'cuda' if device is None else f'cuda:{device}' if isinstance(device, int) else device
        return self.to(target)

    def half(self):
        return self.to(dtype=torch.float16)

    def float(self):
        return self.to(dtype=torch.float32)

    def bfloat16(self):
        raise ValueError('CPU BF16 autocast keeps weights FP32; select inference precision at load time')

    def train(self, mode=True):
        if mode:
            raise ValueError('FP16/BF16 modes are inference-only; adapt/train the original model in FP32')
        self.model.eval()
        return self

    def eval(self):
        return self.train(False)


class FP8InferenceWrapper(PyTorchWrapper):
    """Per-tensor dynamic `float8_e4m3fn` encoder linear layers (`torch._scaled_mm` on Ada/Hopper Tensor Cores)."""

    def __init__(self, model, hardware, linear_modules: int):
        param = next(model.parameters())
        super().__init__(
            model,
            autocast_device=param.device.type if param.device.type in ('cpu', 'cuda') else None,
            autocast_dtype=torch.float16,
            autocast_enabled=False,
            output_float32=True,
        )
        self.precision = 'fp8'
        self.precision_info = dict(
            precision='fp8',
            hardware=hardware,
            device=str(param.device),
            weight_dtype='torch.float8_e4m3fn',
            compute_dtype='torch.float8_e4m3fn',
            output_dtype='torch.float32',
            linear_modules=linear_modules,
            autocast=False,
            scope='Inference only; encoder Linears in float8_e4m3fn, embeddings/norms/heads in float16/float32',
        )

    def train(self, mode=True):
        if mode:
            raise ValueError('FP8 mode is inference-only; adapt/train the original model in FP32')
        self.model.eval()
        return self

    def eval(self):
        return self.train(False)


def configure_precision(model, precision, hardware):
    """Call after FP32 checkpoint loading, LoRA merging and device placement."""
    if precision == 'fp8':
        if model.config.model_type != 'xlm-token' or getattr(model.config, 'experimental_token_compression', False):
            raise ValueError('FP8 precision currently supports ordinary XLM-R SaT token/character heads only')
        param = next(model.parameters())
        if param.device.type == 'cuda':
            model.half()
        else:
            model.float()
        model.eval()
        count = quantize_fp8_roberta(model)
        if not count:
            raise ValueError('No encoder Linear modules found to convert to FP8')
        return FP8InferenceWrapper(model, hardware, count)
    if precision in ('fp16', 'bf16'):
        if model.config.model_type != 'xlm-token' or getattr(model.config, 'experimental_token_compression', False):
            raise ValueError('FP16/BF16 precision currently supports ordinary XLM-R SaT token/character heads only')
        expected_device = 'cuda' if precision == 'fp16' else 'cpu'
        if next(model.parameters()).device.type != expected_device:
            raise ValueError(f'Inference precision {precision} requires {expected_device} placement; specify device explicitly')
        model.half() if precision == 'fp16' else model.float()
        model.eval()
        return PrecisionInferenceWrapper(model, precision, hardware)
    wrapper = PyTorchWrapper(model)
    if precision == 'fp32':
        param = next(model.parameters())
        wrapper = PyTorchWrapper(model, autocast_device=param.device.type if param.device.type in ('cpu', 'cuda') else None,
                                 autocast_enabled=False, output_float32=True)
        wrapper.precision_info = dict(precision='fp32', device=str(param.device), weight_dtype=str(param.dtype),
                                      compute_dtype='torch.float32', output_dtype='torch.float32', autocast=False)
    return wrapper
