"""Experimental, in-memory CPU quantization for XLM-R SaT inference."""

import gc
import time

import torch

from wtpsplit.extract import PyTorchWrapper


class Int8CPUWrapper(PyTorchWrapper):
    """Keep packed INT8 weights on CPU and the floating-point remainder in FP32."""

    def to(self, device=None, *, dtype=None, **kwargs):
        if device is not None and (isinstance(device, torch.dtype) or torch.device(device).type != 'cpu'):
            raise ValueError("INT8 SaT supports CPU FP32 inputs only; reload without quantization for another device/dtype")
        if dtype not in (None, torch.float32):
            raise ValueError("INT8 SaT keeps its floating-point remainder in FP32; half/bfloat16 conversion is unsupported")
        return self

    def cpu(self):
        return self

    def cuda(self, *args, **kwargs):
        raise ValueError("INT8 SaT is CPU-only; reload without quantization to use CUDA")

    def half(self):
        raise ValueError("INT8 SaT keeps its floating-point remainder in FP32")

    def bfloat16(self):
        raise ValueError("INT8 SaT keeps its floating-point remainder in FP32")

    def train(self, mode=True):
        if mode:
            raise ValueError("INT8 SaT is inference-only; adapt/train the FP32 model first")
        self.model.eval()
        return self

    def save_pretrained(self, *args, **kwargs):
        raise ValueError("INT8 SaT is converted at load time; keep the original checkpoint and reload with quantization='int8'")


def _own_float_storage(model):
    """Release checkpoint views, including biases hidden in packed Linears.

    After quantizing the vocabulary, small floating tensors can keep the entire
    source mapping resident. Copy the small remainder without inspecting OS
    mappings, so this also works with sharded checkpoints and on non-Linux CPUs.
    Do not use this on Linear-only models: their FP32 vocabulary is still useful.
    """
    copied = 0
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point():
                tensor.data = tensor.detach().clone()
                copied += tensor.numel() * tensor.element_size()
        for module in model.modules():
            if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
                bias = module.bias()
                if bias is not None:
                    module.set_weight_bias(module.weight(), bias.detach().clone())
                    copied += bias.numel() * bias.element_size()
    return copied


def quantize_cpu(model, mode="linear_int8"):
    """Convert vocabulary and/or encoder Linears after any LoRA merging.

    SaT passes its selected mode explicitly; this helper's historical default
    stays Linear-only for frozen research producers that import it directly.
    Vocabulary quantization uses row-wise unsigned INT8 weights; encoder Linears
    use dynamic signed INT8. Norms, positions, classifier and character head stay
    FP32. No global engine/thread settings or on-disk weights are changed.
    """
    if mode not in ("linear_int8", "embedding_int8", "combined_int8"):
        raise ValueError(f"Unknown CPU quantization mode: {mode!r}")
    if model.config.model_type != 'xlm-token':
        raise ValueError("Experimental INT8 currently supports ordinary XLM-R SaT token/character heads only")
    engine = torch.backends.quantized.engine
    if engine == 'none':
        raise RuntimeError("This PyTorch build has no enabled quantized CPU engine")
    model.cpu().float().eval()
    quantize_linears = mode != "embedding_int8"
    quantize_embeddings = mode != "linear_int8"
    count = sum(isinstance(module, torch.nn.Linear) for module in model.roberta.modules()) if quantize_linears else 0
    if quantize_linears and not count:
        raise ValueError("No encoder Linear modules found to quantize")
    started = time.perf_counter()
    if quantize_linears:
        torch.ao.quantization.quantize_dynamic(model.roberta, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    if quantize_embeddings:
        embedding = model.roberta.embeddings.word_embeddings
        embedding.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
        model.roberta.embeddings.word_embeddings = torch.ao.nn.quantized.Embedding.from_float(embedding)
        del embedding  # Do not retain the original FP32 vocabulary through the memory fix.
    private_bytes = _own_float_storage(model) if quantize_embeddings else 0
    # Discarded conversion modules can form unreachable Python cycles that
    # still own checkpoint-backed tensors. Detaching live tensors alone is not
    # sufficient. Collect once at load time, without changing GC settings.
    collected = gc.collect() if quantize_embeddings else 0
    wrapper = Int8CPUWrapper(model, autocast_device="cpu", autocast_enabled=False, output_float32=True)
    wrapper.quantization_info = {
        'method': {'linear_int8': 'dynamic-int8-encoder-linear',
                   'embedding_int8': 'rowwise-int8-vocabulary',
                   'combined_int8': 'rowwise-int8-vocabulary+dynamic-int8-encoder-linear'}[mode],
        'mode': mode, 'engine': engine, 'linear_modules': count,
        'vocabulary_quantized': quantize_embeddings,
        'private_float_bytes': private_bytes,
        'gc_collected_objects': collected,
        'checkpoint_views_detached': quantize_embeddings,
        'conversion_seconds': time.perf_counter() - started, 'torch_version': torch.__version__,
        'scope': f'{mode}; positions, token types, norms, classifier and character head FP32',
    }
    return wrapper
