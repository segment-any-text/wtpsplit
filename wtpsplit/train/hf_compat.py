"""TPU / XLA checks compatible with **transformers 4.x and 5.x**.

v5 removed ``is_torch_tpu_available`` from ``transformers.trainer``; ``is_torch_xla_available``
exists on both and matches the old behaviour.
"""

from transformers.utils.import_utils import is_torch_xla_available


def is_torch_tpu_available(check_device: bool = True) -> bool:
    """Same idea as the old ``transformers.trainer.is_torch_tpu_available``.

    - ``check_device=False``: ``torch_xla`` is importable (optional imports).
    - ``check_device=True`` (default): current process is on a TPU.
    """
    if check_device:
        return is_torch_xla_available(check_is_tpu=True)
    return is_torch_xla_available()
