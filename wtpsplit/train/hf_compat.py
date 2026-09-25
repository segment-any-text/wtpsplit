"""TPU / XLA checks compatible with transformers 4.29 through 5.x.

``is_torch_xla_available`` exists from transformers 4.39. The training pin
``transformers==4.29.2`` only has ``is_torch_tpu_available``. Transformers 5.0
dropped ``is_torch_tpu_available``.
"""

try:
    from transformers.utils.import_utils import is_torch_xla_available
except ImportError:  # transformers < 4.39
    is_torch_xla_available = None

try:
    from transformers.utils.import_utils import is_torch_tpu_available as _hf_is_torch_tpu_available
except ImportError:  # transformers 5.0
    _hf_is_torch_tpu_available = None


def is_torch_tpu_available(check_device: bool = True) -> bool:
    """Same idea as the old ``transformers.trainer.is_torch_tpu_available``.

    - ``check_device=False``: ``torch_xla`` is importable (optional imports).
    - ``check_device=True`` (default): current process is on a TPU.
    """
    if is_torch_xla_available is not None:
        if check_device:
            return is_torch_xla_available(check_is_tpu=True)
        return is_torch_xla_available()
    if _hf_is_torch_tpu_available is not None:
        return _hf_is_torch_tpu_available(check_device=check_device)
    return False
