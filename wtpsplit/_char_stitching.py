"""Ordered uniform character-window fusion on device.

``UniformStitcher.add`` consumes one model batch; ``finish`` copies the complete
float16 result into pinned host storage once. Overlaps add in reference window
order in float32, without CUDA atomic reductions. All label columns and uncovered
positions are retained. Non-uniform weighting stays with the host implementation.
"""
from collections.abc import Mapping, Sequence

from wtpsplit._inference_transfer import PinnedInputs
from wtpsplit._char_decisions import CharacterDecisionRequest, CharacterDecisions, pack_decisions
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch


def _increment_sorted_counts(counts: NDArray[np.int32], ordered: NDArray[np.int64]) -> None:
    """Add one per sorted destination, preserving int32 modular addition.

    ``ordered`` contains nondecreasing destination indices. Each destination is
    updated once by its run length; no floating-point accumulation is involved.
    Empty windows contribute no indices and leave the counts unchanged.
    """
    if ordered.size == 0:
        return
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    increments = np.diff(np.r_[starts, ordered.size]).astype(np.int32)
    counts[ordered[starts]] += increments


class UniformStitcher:
    """Request-local accumulator; owns no streams, workers or persistent buffers."""

    def __init__(self, lengths: Sequence[int], labels: int, device: Any,
                 *, transfers: PinnedInputs | None = None) -> None:
        self.offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        self.values = torch.zeros((int(self.offsets[-1]), labels), dtype=torch.float32, device=device)
        self.counts = np.zeros(int(self.offsets[-1]), dtype=np.int32)
        self.finished = False
        self.transfers = transfers

    def add(self, windows: Sequence[Mapping[str, Any]], logits: torch.Tensor) -> None:
        """Accumulate valid characters, preserving each position's addition order."""
        if self.finished:
            raise RuntimeError('Cannot add windows after finish')
        targets = []
        sources = []
        width = logits.shape[1]
        for row, window in enumerate(windows):
            start, end = int(window['char_start']), int(window['char_end'])
            doc = int(window['text_index'])
            if not 0 <= start <= end <= self.offsets[doc + 1] - self.offsets[doc]:
                raise ValueError('Window lies outside its document')
            targets.append(np.arange(start, end, dtype=np.int64) + self.offsets[doc])
            sources.append(np.arange(end - start, dtype=np.int64) + row * width)
        if not targets:
            return
        target = np.concatenate(targets)
        source = np.concatenate(sources)
        # Stable destination order gives the reference rank of every contribution.
        order = np.argsort(target, kind='stable')
        ordered = target[order]
        indices = np.arange(len(order), dtype=np.int64)
        first = np.maximum.accumulate(np.where(np.r_[True, ordered[1:] != ordered[:-1]], indices, 0))
        ranks = np.empty_like(order)
        ranks[order] = indices - first
        _increment_sorted_counts(self.counts, ordered)
        rank_count = int(ranks.max(initial=-1)) + 1
        metadata: dict[str, NDArray[np.generic] | torch.Tensor] = {}
        for rank in range(rank_count):
            keep = ranks == rank
            metadata[f'dest/{rank}'] = target[keep]
            metadata[f'src/{rank}'] = source[keep]
        if self.transfers is None:
            transferred = {key: torch.as_tensor(value, device=logits.device) for key, value in metadata.items()}
        else:
            transferred = self.transfers.tensors('stitch', metadata, logits.device)
        flat = logits.reshape(-1, logits.shape[-1])
        for rank in range(rank_count):
            dest = transferred[f'dest/{rank}']
            src = transferred[f'src/{rank}']
            # Distinct destinations within a rank: deterministic, no atomic sum.
            self.values.index_copy_(0, dest, self.values[dest] + flat[src].float())

    def _normalize(self) -> torch.Tensor:
        """Produce the same rounded merged logits as the reference host path."""
        if self.finished:
            raise RuntimeError('finish may be called only once')
        self.finished = True
        counts = (torch.as_tensor(self.counts, device=self.values.device) if self.transfers is None
                  else self.transfers.tensor('stitch/counts', self.counts, self.values.device))
        self.values /= counts.clamp_min(1)[:, None].float()
        self.values.masked_fill_(counts[:, None] == 0, -12.0)
        return self.values.to(torch.float16)

    @staticmethod
    def _to_host(result: torch.Tensor) -> Any:
        """Complete one asynchronous pinned transfer before exposing numpy data."""
        if result.is_cuda:
            host = torch.empty(result.shape, dtype=result.dtype, pin_memory=True)
            host.copy_(result, non_blocking=True)
            # The numpy consumer cannot read an in-flight pinned transfer.
            event_factory: Any = torch.cuda.Event
            event = event_factory()
            event.record(torch.cuda.current_stream(result.device))
            event.synchronize()
        else:
            host = result.cpu()
        return host.numpy()

    def finish(self) -> list[NDArray[np.float16]]:
        """Normalize and copy every position and label for probability consumers."""
        array = self._to_host(self._normalize())
        return [array[start:end] for start, end in zip(self.offsets[:-1], self.offsets[1:])]

    def finish_decisions(self, request: CharacterDecisionRequest) -> list[CharacterDecisions]:
        """Copy only packed split decisions; probability outputs use finish instead."""
        from wtpsplit.utils import Constants

        logits = self._normalize()[:, Constants.NEWLINE_INDEX]
        packed = self._to_host(pack_decisions(logits, request))
        mask = np.unpackbits(packed, bitorder='little', count=int(self.offsets[-1]))
        return [CharacterDecisions(mask[start:end]) for start, end in zip(self.offsets[:-1], self.offsets[1:])]
