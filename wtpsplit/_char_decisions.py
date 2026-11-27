"""Exact internal split decisions; probability APIs retain full character logits.

Merge and float16-round logits before thresholding. One bit per character is
copied; host decoding uses the existing sentence ownership rules. No positions
or label columns are dropped from predict_proba or extraction/logging callers.
"""
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CharacterDecisionRequest:
    """Smallest passing float16 logit, or None when no logit may pass."""

    minimum_logit: float | None


@dataclass(frozen=True)
class CharacterDecisions:
    """A document's character decisions, without materialized probabilities."""

    mask: NDArray[np.uint8]

    def positions(self) -> NDArray[np.intp]:
        """Return positive character indices in the reference decoding order."""
        return np.flatnonzero(self.mask)


def pack_decisions(logits: 'torch.Tensor', request: CharacterDecisionRequest) -> 'torch.Tensor':
    """Pack rounded-logit comparisons little-endian, zero-padding the final byte."""
    import torch
    minimum = request.minimum_logit
    rounded = logits.to(torch.float16)
    if minimum is None or (math.isinf(minimum) and minimum > 0):
        mask = torch.zeros(rounded.shape, dtype=torch.uint8, device=rounded.device)
    else:
        mask = (rounded >= minimum).to(torch.uint8)
    mask = torch.nn.functional.pad(mask.reshape(-1), (0, (-mask.numel()) % 8))
    shifts = torch.arange(8, dtype=torch.uint8, device=mask.device)
    return (mask.reshape(-1, 8) << shifts).sum(dim=-1).to(torch.uint8)
