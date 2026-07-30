"""Loss helpers shared by SaT token-classification backbones."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_binary_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    *,
    balance_classes: bool = False,
) -> torch.Tensor:
    """Binary cross-entropy for a single-logit token classifier.

    Hugging Face token-classification models normally apply multiclass cross-entropy.
    SaT Stage 3 (formerly: 2) instead emits one boundary logit and labels tokens with 0/1, so the
    one-class cross-entropy path would reject every positive label.
    """

    if logits.shape[-1] != 1:
        raise ValueError(f"Expected one boundary logit per token, got shape {tuple(logits.shape)}")

    keep = labels.ne(ignore_index)
    if not keep.any():
        return logits.sum() * 0.0

    kept_labels = labels[keep].to(logits.dtype)
    pos_weight = None
    if balance_classes:
        positives = kept_labels.sum()
        negatives = kept_labels.numel() - positives
        if positives > 0:
            pos_weight = (negatives / positives).detach()

    return F.binary_cross_entropy_with_logits(
        logits.squeeze(-1)[keep],
        kept_labels,
        pos_weight=pos_weight,
    )
