"""Instance-owned pinned input buffers, bounded independently of model weights."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


class PinnedInputs:
    """Reuse at most ``max_bytes`` of double-buffered pinned host buffers on a CUDA path.

    Maintains a 2-slot ring (`slot 0`, `slot 1`) per `(name, shape, dtype)` synchronized
    via lightweight per-slot `torch.cuda.Event` markers recorded immediately after each
    `non_blocking=True` H2D copy. This guarantees that CPU preparation of batch `b + 1`
    never overwrites batch `b`'s pinned buffer while its H2D DMA is still in flight, even
    when GPU-to-CPU logit transfers are deferred across multiple batches.
    """

    def __init__(self, max_bytes: int = 64 * 2**20, slots: int = 2) -> None:
        self.max_bytes = max_bytes
        self.slots = max(1, int(slots))
        self._buffers: OrderedDict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor] = OrderedDict()
        self._events: dict[tuple[str, tuple[int, ...], torch.dtype], Any] = {}
        self._slot_cursor: dict[tuple[str, tuple[int, ...], torch.dtype], int] = {}
        self._bytes = 0

    def tensor(self, name: str, value: NDArray[np.generic] | torch.Tensor, device: torch.device) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
        size = tensor.numel() * tensor.element_size()
        if device.type != "cuda" or tensor.device.type != "cpu" or tensor.numel() < 1024 or size > self.max_bytes:
            return tensor.to(device)
        shape_tuple = tuple(tensor.shape)
        base_key = (name, shape_tuple, tensor.dtype)
        slot = self._slot_cursor.get(base_key, 0)
        self._slot_cursor[base_key] = (slot + 1) % self.slots
        slot_name = name if slot == 0 else f"{name}#{slot}"
        key = (slot_name, shape_tuple, tensor.dtype)
        if key not in self._buffers:
            while self._buffers and self._bytes + size > self.max_bytes:
                evicted_key, evicted = self._buffers.popitem(last=False)
                evicted_ev = self._events.pop(evicted_key, None)
                if evicted_ev is not None:
                    evicted_ev.synchronize()
                self._bytes -= evicted.numel() * evicted.element_size()
            self._buffers[key] = torch.empty_like(tensor, pin_memory=True)
            self._bytes += size
        else:
            prev_ev = self._events.get(key)
            if prev_ev is not None:
                prev_ev.synchronize()
        pinned = self._buffers[key]
        self._buffers.move_to_end(key)
        pinned.copy_(tensor)
        out = pinned.to(device, non_blocking=True)
        if torch.cuda.is_available():
            ev = self._events.get(key)
            if ev is None:
                ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
                self._events[key] = ev
            ev.record(torch.cuda.current_stream(device))
        return out

    def clear(self) -> None:
        """Release this session's host buffers after its final blocking return."""
        for ev in self._events.values():
            ev.synchronize()
        self._events.clear()
        self._slot_cursor.clear()
        self._buffers.clear()
        self._bytes = 0
