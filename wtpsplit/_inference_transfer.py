"""Instance-owned pinned input buffers, bounded independently of model weights."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from numpy.typing import NDArray


class PinnedInputs:
    """Reuse at most ``max_bytes`` of host buffers on an explicit CUDA path.

    The wrapper returns CPU logits before the next call, so transfers have
    completed before a buffer is reused. This object is not shared across callers.
    """

    def __init__(self, max_bytes: int = 64 * 2**20) -> None:
        self.max_bytes = max_bytes
        self._buffers: OrderedDict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor] = OrderedDict()
        self._bytes = 0

    def tensor(self, name: str, value: NDArray[np.generic] | torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
        size = tensor.numel() * tensor.element_size()
        if device.type != "cuda" or tensor.device.type != "cpu" or tensor.numel() < 1024 or size > self.max_bytes:
            return tensor.to(device)
        key = name, tuple(tensor.shape), tensor.dtype
        if key not in self._buffers:
            while self._buffers and self._bytes + size > self.max_bytes:
                _, evicted = self._buffers.popitem(last=False)
                self._bytes -= evicted.numel() * evicted.element_size()
            self._buffers[key] = torch.empty_like(tensor, pin_memory=True)
            self._bytes += size
        pinned = self._buffers[key]
        self._buffers.move_to_end(key)
        pinned.copy_(tensor)
        return pinned.to(device, non_blocking=True)

    def clear(self) -> None:
        """Release this session's host buffers after its final blocking return."""
        self._buffers.clear()
        self._bytes = 0
