"""Instance-owned pinned input buffers, bounded independently of model weights."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from math import prod
from types import TracebackType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


Layout = dict[str, tuple[torch.dtype, int, tuple[int, ...]]]
_INTEGER_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)


def _pack(values: Mapping[str, torch.Tensor]) -> tuple[dict[torch.dtype, torch.Tensor], Layout]:
    """Pack CPU fields by dtype without changing their bytes or element order."""
    groups: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for key, tensor in values.items():
        if tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise ValueError('packing requires contiguous CPU fields')
        groups.setdefault(tensor.dtype, []).append((key, tensor))
    flats: dict[torch.dtype, torch.Tensor] = {}
    layout: Layout = {}
    for dtype, fields in groups.items():
        if dtype in _INTEGER_DTYPES:
            flats[dtype] = torch.from_numpy(np.concatenate([value.numpy().reshape(-1) for _, value in fields]))
        else:
            flats[dtype] = torch.cat([value.reshape(-1) for _, value in fields])
        offset = 0
        for key, value in fields:
            layout[key] = (dtype, offset, tuple(value.shape))
            offset += value.numel()
    return flats, layout


def _unpack(flat: Mapping[torch.dtype, torch.Tensor], layout: Layout) -> dict[str, torch.Tensor]:
    """Restore dtype-grouped fields as views at their recorded offsets/shapes."""
    return {key: flat[dtype][offset:offset + prod(shape)].view(shape)
            for key, (dtype, offset, shape) in layout.items()}


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

    def tensor(self, name: str, value: NDArray[np.generic] | torch.Tensor, device: torch.device, *, serial_integer_copy: bool = False) -> torch.Tensor:
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
        if serial_integer_copy:
            # Metadata consists of integer/bool bytes. NumPy avoids launching
            # an OpenMP copy team that competes with background preparation.
            np.copyto(pinned.numpy(), tensor.numpy(), casting='no')
        else:
            pinned.copy_(tensor)
        out = pinned.to(device, non_blocking=True)
        if torch.cuda.is_available():
            ev = self._events.get(key)
            if ev is None:
                ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
                self._events[key] = ev
            ev.record(torch.cuda.current_stream(device))
        return out

    def tensors(self, name: str, values: Mapping[str, NDArray[np.generic] | torch.Tensor],
                device: torch.device) -> dict[str, torch.Tensor]:
        """Copy contiguous CPU metadata in one pinned transfer per dtype.

        Returned views retain every field's shape, dtype and element order. GPU
        and strided fields keep the existing per-field path. The ring owns the
        packed host storage until its transfer event completes; values are
        copied afresh on every call, never looked up by document or request.
        """
        result: dict[str, torch.Tensor] = {}
        eligible: dict[str, torch.Tensor] = {}
        for key, value in values.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
            if device.type != 'cuda' or tensor.device.type != 'cpu' or not tensor.is_contiguous():
                result[key] = self.tensor(name + '/' + key, tensor, device)
            else:
                eligible[key] = tensor
        flats, layout = _pack(eligible)
        transferred = {dtype: self.tensor(name + '/' + str(dtype), value, device,
                                          serial_integer_copy=dtype in _INTEGER_DTYPES)
                       for dtype, value in flats.items()}
        result.update(_unpack(transferred, layout))
        return result

    def __enter__(self) -> PinnedInputs:
        """Own metadata buffers for one extraction request."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.clear()

    def clear(self) -> None:
        """Release this session's host buffers after its final blocking return."""
        for ev in self._events.values():
            ev.synchronize()
        self._events.clear()
        self._slot_cursor.clear()
        self._buffers.clear()
        self._bytes = 0
