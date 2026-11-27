"""Device overlap-add for packed subword windows."""
from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

_TRITON_AVAILABLE = find_spec("triton") is not None
_TILE = 128


def subword_stitch_tiles(
    locations: NDArray[np.int32], offsets: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Partition flat output into tiles whose covering windows form one row range."""
    docs, starts, ends = locations.T
    left, right = offsets[docs] + starts, offsets[docs] + ends
    edges = np.union1d(left, right)
    if edges.size < 2:
        return np.empty((0, 4), dtype=np.int64)
    starts, stops = edges[:-1], edges[1:]
    first = np.searchsorted(right, starts, side="right")
    last = np.searchsorted(left, starts, side="right")
    sizes = (stops - starts + _TILE - 1) // _TILE
    intervals = np.repeat(np.arange(len(sizes)), sizes)
    within = np.arange(len(intervals)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    starts = starts[intervals] + within * _TILE
    stops = np.minimum(stops[intervals], starts + _TILE)
    return np.column_stack((starts, stops, first[intervals], last[intervals]))


class SubwordStitcher:
    """Prepare compressed window geometry and launch one ordered overlap-add."""

    def __init__(
        self, locations: NDArray[np.int32], lengths: Sequence[int], block_size: int,
        weighting: Literal["uniform", "hat"], device: Any,
    ) -> None:
        self.offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(lengths, dtype=np.int64, out=self.offsets[1:])
        self.window_lengths = locations[:, 2] - locations[:, 1]
        # Optional width bucketing changes the order across documents only.
        # Each long document's windows have equal width and retain their order.
        order = np.argsort(locations[:, 0], kind="stable")
        locations = locations[order]
        docs, starts, _ = locations.T
        # Long documents have reference-style additions, even where just one
        # window covers a position. Short documents directly cast their logits.
        long = np.asarray(lengths)[docs] > block_size
        left = self.offsets[docs] + starts
        self.windows = self._upload(np.column_stack((left, order, long, starts == 0)), device)
        self.tiles = self._upload(subword_stitch_tiles(locations, self.offsets), device)
        self.weights = None
        if weighting == "hat":
            from wtpsplit._host_kernels import subword_hat_weights

            self.weights = self._upload(subword_hat_weights(block_size), device)

    @staticmethod
    def _upload(array: NDArray[Any], device: Any) -> torch.Tensor:
        import torch

        tensor = torch.from_numpy(array)
        return tensor.to(device, non_blocking=True)

    def validate_widths(self, batches: Sequence[torch.Tensor], *, cls_prefix: int) -> None:
        """Check each row against its own width before the padding copy."""
        row = 0
        for batch in batches:
            end = row + len(batch)
            if batch.shape[1] < cls_prefix + int(self.window_lengths[row:end].max(initial=0)):
                raise ValueError("Forward logits are shorter than their window")
            row = end
        if row != len(self.window_lengths):
            raise ValueError("Wrong number of forward windows")

    def merge(self, batches: Sequence[torch.Tensor], *, cls_prefix: int) -> torch.Tensor:
        """Enqueue the kernel after the forwards on their stream."""
        import torch

        from wtpsplit._subword_stitching_triton import overlap_add

        first = batches[0]
        labels = first.shape[-1]
        result = torch.empty((int(self.offsets[-1]), labels), dtype=torch.float16, device=first.device)
        if self.tiles.shape[0]:
            width = max(batch.shape[1] for batch in batches)
            if any(batch.shape[1] != width for batch in batches):
                raw = torch.empty((sum(map(len, batches)), width, labels), device=first.device, dtype=first.dtype)
                start = 0
                for batch in batches:
                    raw[start:start + len(batch), :batch.shape[1]].copy_(batch)
                    start += len(batch)
            else:
                raw = torch.cat(tuple(batches), dim=0) if len(batches) > 1 else first.contiguous()
            overlap_add[(self.tiles.shape[0], labels)](
                raw, self.windows, self.tiles, self.weights, result,
                raw.shape[1], labels, cls_prefix, self.weights is not None, _TILE,
                num_warps=4, enable_fp_fusion=False,
            )
        return result

    def copy_to_host(self, result: torch.Tensor) -> list[NDArray[np.float16]]:
        """Copy the normalized logits once and return views for each document."""
        values: NDArray[np.float16] = result.cpu().numpy()
        return [values[start:end] for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True)]


def available() -> bool:
    return _TRITON_AVAILABLE


def supports_device(device: Any) -> bool:
    import torch

    return (
        available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) == (12, 0)
    )
