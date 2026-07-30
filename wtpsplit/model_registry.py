"""Single internal boundary for Hugging Face Auto* registrations.

The rest of wtpsplit loads checkpoints through this module rather than relying
on registration side effects spread across configuration and model files.
Imports remain lazy so ONNX users can register configs without importing torch
model implementations.
"""

from __future__ import annotations

from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

_SAT_CONFIGS_REGISTERED = False
_SAT_MODELS_REGISTERED = False
_LEGACY_CONFIGS_REGISTERED = False
_LEGACY_MODELS_REGISTERED = False


def resolve_effective_lookahead(config, *, supports_split_layers: bool = False) -> int | None:
    """Return the per-layer lookahead budget shared by all SaT backbones."""
    lookahead = getattr(config, "lookahead", None)
    if lookahead is None:
        return None

    split_layers = getattr(config, "lookahead_split_layers", None)
    if split_layers is not None:
        if not supports_split_layers:
            raise ValueError(
                "lookahead_split_layers is not supported by this backbone; it requires per-layer-index attention masks."
            )
        if not 0 < split_layers <= config.num_hidden_layers:
            raise ValueError(
                "lookahead_split_layers must be between 1 and "
                f"num_hidden_layers ({config.num_hidden_layers}), got {split_layers}"
            )
        divisor = split_layers
    else:
        divisor = config.num_hidden_layers

    if lookahead % divisor != 0:
        raise ValueError(f"lookahead ({lookahead}) must be divisible by the active layer count ({divisor})")
    return lookahead // divisor


def get_sat_backbone_types() -> dict[str, tuple[type, type]]:
    """Return the canonical config/model wrappers for inference and training."""
    from wtpsplit.configs import SubwordModernBertConfig, SubwordXLMConfig
    from wtpsplit.models import SubwordXLMForTokenClassification
    from wtpsplit.models_modernbert import SubwordModernBertForTokenClassification

    return {
        "xlm-roberta": (SubwordXLMConfig, SubwordXLMForTokenClassification),
        "xlm-token": (SubwordXLMConfig, SubwordXLMForTokenClassification),
        "modernbert": (SubwordModernBertConfig, SubwordModernBertForTokenClassification),
        "modernbert-token": (SubwordModernBertConfig, SubwordModernBertForTokenClassification),
    }


def register_sat_configs() -> None:
    global _SAT_CONFIGS_REGISTERED
    if _SAT_CONFIGS_REGISTERED:
        return
    _SAT_CONFIGS_REGISTERED = True

    try:
        from wtpsplit.configs import SubwordModernBertConfig, SubwordXLMConfig

        AutoConfig.register("xlm-token", SubwordXLMConfig)
        AutoConfig.register("modernbert-token", SubwordModernBertConfig)
    except Exception:
        _SAT_CONFIGS_REGISTERED = False
        raise


def register_sat_models() -> None:
    global _SAT_MODELS_REGISTERED
    if _SAT_MODELS_REGISTERED:
        return
    _SAT_MODELS_REGISTERED = True

    try:
        register_sat_configs()
        from wtpsplit.models import SubwordXLMForTokenClassification, SubwordXLMRobertaModel

        backbones = get_sat_backbone_types()
        SubwordXLMConfig, _ = backbones["xlm-token"]
        SubwordModernBertConfig, SubwordModernBertForTokenClassification = backbones["modernbert-token"]
        AutoModel.register(SubwordXLMConfig, SubwordXLMRobertaModel)
        AutoModelForTokenClassification.register(SubwordXLMConfig, SubwordXLMForTokenClassification)
        AutoModelForTokenClassification.register(
            SubwordModernBertConfig,
            SubwordModernBertForTokenClassification,
        )
    except Exception:
        _SAT_MODELS_REGISTERED = False
        raise


def register_legacy_configs() -> None:
    global _LEGACY_CONFIGS_REGISTERED
    if _LEGACY_CONFIGS_REGISTERED:
        return
    _LEGACY_CONFIGS_REGISTERED = True

    try:
        from wtpsplit.legacy.configs import BertCharConfig, LACanineConfig

        AutoConfig.register("bert-char", BertCharConfig)
        AutoConfig.register("la-canine", LACanineConfig)
    except Exception:
        _LEGACY_CONFIGS_REGISTERED = False
        raise


def register_legacy_models() -> None:
    global _LEGACY_MODELS_REGISTERED
    if _LEGACY_MODELS_REGISTERED:
        return
    _LEGACY_MODELS_REGISTERED = True

    try:
        register_legacy_configs()
        from wtpsplit.legacy.configs import BertCharConfig, LACanineConfig
        from wtpsplit.legacy.models import (
            BertCharForTokenClassification,
            BertCharModel,
            LACanineForTokenClassification,
            LACanineModel,
        )

        AutoModel.register(LACanineConfig, LACanineModel)
        AutoModelForTokenClassification.register(LACanineConfig, LACanineForTokenClassification)
        AutoModel.register(BertCharConfig, BertCharModel)
        AutoModelForTokenClassification.register(BertCharConfig, BertCharForTokenClassification)
    except Exception:
        _LEGACY_MODELS_REGISTERED = False
        raise
