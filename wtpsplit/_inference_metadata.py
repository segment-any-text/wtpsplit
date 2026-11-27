"""Typed, request-local host metadata passed explicitly to native SaT models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch


@dataclass(frozen=True)
class MaskMetadata:
    """Own exact host patterns; model shape/device/lookahead complete the cache key."""

    padding_pattern: bytes | None = None
    document_pattern: tuple[str, bytes] | None = None
    all_ones: bool = False

    @classmethod
    def padding(cls, values: NDArray[np.generic]) -> MaskMetadata:
        """Snapshot nonzero semantics without reading device memory."""
        return cls(padding_pattern=(values != 0).tobytes(), all_ones=bool((values == 1).all()))

    @classmethod
    def documents(cls, values: NDArray[np.generic]) -> MaskMetadata:
        """Snapshot document identities, including their representation dtype."""
        return cls(document_pattern=(str(values.dtype), values.tobytes()))


@dataclass(frozen=True)
class GateState:
    """One batch's already-synchronized gate, consumed once by its wrapper."""

    logits: torch.Tensor
    threshold: float
    active: bool
