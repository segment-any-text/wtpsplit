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


def validate_precision(precision, *, device, quantization, ort_providers, compile, from_pretrained_kwargs):
    """Reject unsupported/conflicting choices before loading checkpoint files."""
    if precision not in (None, 'fp32', 'fp16', 'bf16'):
        raise ValueError("precision must be None, 'fp32', 'fp16', or 'bf16'")
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
        raise ValueError('FP16/BF16 inference precision is not yet supported with compile')
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


def configure_precision(model, precision, hardware):
    """Call after FP32 checkpoint loading, LoRA merging and device placement."""
    if precision in ('fp16', 'bf16'):
        if model.config.model_type != 'xlm-token':
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
