# Model backend boundary

wtpsplit supports three checkpoint families through one private integration
boundary: `wtpsplit.model_registry`.

## Registration contract

Inference code calls one of four idempotent functions:

- `register_sat_configs()` for ONNX SaT configuration loading
- `register_sat_models()` for PyTorch SaT loading
- `register_legacy_configs()` for ONNX WtP configuration loading
- `register_legacy_models()` for PyTorch WtP loading

No inference path depends on scattered `AutoConfig` or `AutoModel` mutations.
Historical imports of `wtpsplit.configs`, `wtpsplit.models`, and their legacy
counterparts still trigger the same registration through this boundary.

## Vendored-code audit

`wtpsplit/models.py` retains only the XLM-R code needed to change attention
semantics:

- `SubwordXLMRobertaModel` constructs the limited-lookahead mask and injects it
  before the encoder.
- `SubwordXLMRobertaEncoder` can replace that mask at
  `lookahead_split_layers`, which upstream XLM-R does not expose.
- `_get_head_mask` and `get_extended_attention_mask` support that custom
  encoder path.

The token-classification head is no longer copied: it inherits the upstream
Transformers implementation and only swaps in the custom backbone. Removing
the remaining model/encoder copies would require an upstream per-layer
attention-mask hook; replacing them with calls to undocumented internals would
be less stable.

`wtpsplit/models_modernbert.py` vendors no model or encoder implementation. It
uses the public ModernBERT mask-dictionary path and overrides only mask
construction.

XLM-R retains `lookahead_split_layers` for checkpoints that switch to a
strict causal mask above a specified layer. ModernBERT rejects that option
explicitly because its mask dictionary is keyed by attention type, not layer
index. No *current* ModernBERT configuration uses it.

The CANINE and character-BERT implementations are required solely to load old
WtP checkpoints. They are under `wtpsplit.legacy` and are not imported by the
SaT configuration-only path.

## Upgrade policy

Imports from `transformers.models.*.modeling_*` are confined to the backend
modules. The non-blocking prerelease CI job imports all custom models and runs
hub-free XLM-R, ModernBERT, and legacy registry tests so upstream signature
changes are visible before a stable Transformers release.
