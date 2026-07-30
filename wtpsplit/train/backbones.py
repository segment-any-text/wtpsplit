"""Map a checkpoint to the SaT config/model pair that can train on it.

SaT wraps its backbones so they support limited lookahead, so loading a checkpoint with
`AutoModelForTokenClassification` would silently return the stock transformers class and
drop the lookahead constraint. This resolves the correct wrapper from the checkpoint's
`model_type` instead, so training scripts do not have to branch on the backbone.
"""

from __future__ import annotations

from transformers import AutoConfig

from wtpsplit.model_registry import get_sat_backbone_types

__all__ = ["BACKBONE_BY_MODEL_TYPE", "DEFAULT_TOKENIZER_BY_MODEL_TYPE", "resolve_backbone"]

BACKBONE_BY_MODEL_TYPE = get_sat_backbone_types()

DEFAULT_TOKENIZER_BY_MODEL_TYPE = {
    "xlm-roberta": "facebookAI/xlm-roberta-base",
    "xlm-token": "facebookAI/xlm-roberta-base",
    "modernbert": "jhu-clsp/mmBERT-base",
    "modernbert-token": "jhu-clsp/mmBERT-base",
}


def resolve_backbone(model_name_or_path: str) -> tuple[type, type, str | None]:
    """Return `(config_class, model_class, default_tokenizer)` for a checkpoint."""
    model_type = AutoConfig.from_pretrained(model_name_or_path).model_type
    if model_type not in BACKBONE_BY_MODEL_TYPE:
        raise ValueError(
            f"Unsupported backbone {model_type!r} for {model_name_or_path!r}. Known: {sorted(BACKBONE_BY_MODEL_TYPE)}"
        )
    config_class, model_class = BACKBONE_BY_MODEL_TYPE[model_type]
    return config_class, model_class, DEFAULT_TOKENIZER_BY_MODEL_TYPE.get(model_type)
