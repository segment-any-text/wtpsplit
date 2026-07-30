"""ModernBERT-backed backbone for SaT 2 (mmSaT).

The XLM-R backbone implements limited lookahead by materialising a dense
`[batch, 1, seq, seq]` additive mask (`get_extended_attention_mask` in `models.py`) and
threading it through a vendored copy of the RoBERTa encoder.

ModernBERT needs neither. `ModernBertModel.forward` accepts `attention_mask` as a dict
keyed by layer type, and `transformers.masking_utils` composes mask predicates via
`and_mask_function`, dispatching to FlexAttention, SDPA or eager as configured. Limited
lookahead therefore reduces to a single predicate and this module vendors nothing.

Semantics match the XLM-R implementation exactly: query position `i` attends to every
past key and to at most `lookahead` future keys, i.e. `kv_idx <= q_idx + lookahead`,
which is what `torch.tril(ones, diagonal=lookahead)` produces.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertForTokenClassification,
)

from wtpsplit._training_loss import masked_binary_token_loss
from wtpsplit.char_head import CharacterResolutionHead, CharInputs
from wtpsplit.configs import SubwordModernBertConfig
from wtpsplit.model_registry import resolve_effective_lookahead

__all__ = [
    "SubwordModernBertForTokenClassification",
    "build_lookahead_attention_masks",
    "lookahead_mask_function",
    "resolve_effective_lookahead",
]


def lookahead_mask_function(lookahead: int) -> Callable:
    """Predicate allowing all past keys and at most `lookahead` future keys.

    Equivalent to `torch.tril(torch.ones(L, L), diagonal=lookahead)`, which is how the
    XLM-R backbone expresses the same constraint.
    """

    def inner(batch_idx, head_idx, q_idx, kv_idx):
        return kv_idx <= q_idx + lookahead

    return inner


def build_lookahead_attention_masks(
    config,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    lookahead: int,
) -> dict[str, object]:
    """Mask dict for `ModernBertModel.forward`, with lookahead applied to both layer types.

    Returns a `BlockMask` per entry under FlexAttention and a dense tensor under
    SDPA/eager; `masking_utils` picks based on `config._attn_implementation`.
    """
    # The helpers read only batch size, sequence length, dtype and device off this.
    proxy_embeds = torch.empty((batch_size, seq_len, 1), dtype=dtype, device=device)
    and_mask = lookahead_mask_function(lookahead)
    mask_kwargs = {
        "config": config,
        "inputs_embeds": proxy_embeds,
        "attention_mask": attention_mask,
        "and_mask_function": and_mask,
    }
    return {
        "full_attention": create_bidirectional_mask(**mask_kwargs),
        "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
    }


class SubwordModernBertForTokenClassification(ModernBertForTokenClassification):
    """ModernBERT token classifier with SaT's limited-lookahead constraint.

    Drop-in counterpart to `SubwordXLMForTokenClassification`. With
    `config.lookahead=None` this is plain ModernBERT and the mask path is untouched.
    """

    config_class = SubwordModernBertConfig

    def __init__(self, config: SubwordModernBertConfig):
        # `from_pretrained` applies overrides such as a reduced `num_hidden_layers` after
        # the config is constructed, so realign `layer_types` before the layers are built.
        if hasattr(config, "_resync_layer_types"):
            config._resync_layer_types()
        super().__init__(config)
        self.effective_lookahead = resolve_effective_lookahead(config)
        self.character_head = (
            CharacterResolutionHead(config.hidden_size, num_labels=config.num_labels)
            if config.use_character_head
            else None
        )
        self.post_init()
        if self.character_head is not None and config.character_head_init == "identity":
            self.character_head.reset_to_identity()

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        """Keep identity semantics when adding a character head to a token checkpoint."""
        head = self.character_head
        missing = None
        if head is not None:
            missing = (
                not getattr(head.delta.weight, "_is_hf_initialized", False),
                not getattr(head.delta.bias, "_is_hf_initialized", False),
                not getattr(head.suppression.weight, "_is_hf_initialized", False),
            )

        super()._initialize_missing_keys(is_quantized)

        if head is not None and missing is not None and self.config.character_head_init == "identity":
            head.reset_missing_to_identity(
                delta_weight=missing[0],
                delta_bias=missing[1],
                suppression=missing[2],
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, object] | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        char_to_token: torch.Tensor | None = None,
        char_is_token_final: torch.Tensor | None = None,
        char_position_in_token: torch.Tensor | None = None,
        char_hashes: torch.Tensor | None = None,
        char_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        # A dict means a caller already built per-layer-type masks; leave it alone.
        if self.effective_lookahead is not None and not isinstance(attention_mask, dict):
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
                device, dtype = inputs_embeds.device, inputs_embeds.dtype
            else:
                batch_size, seq_len = input_ids.shape[:2]
                device, dtype = input_ids.device, self.dtype
            attention_mask = build_lookahead_attention_masks(
                config=self.config,
                attention_mask=attention_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                lookahead=self.effective_lookahead,
            )

        if self.character_head is not None:
            char_tensors = (
                char_to_token,
                char_is_token_final,
                char_position_in_token,
                char_hashes,
                char_mask,
            )
            if any(tensor is None for tensor in char_tensors):
                raise ValueError("Character-head models require all five `char_*` input tensors.")

            return_dict = kwargs.pop("return_dict", self.config.use_return_dict)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                **kwargs,
            )
            token_hidden = outputs.last_hidden_state
            token_logits = self.classifier(self.drop(self.head(token_hidden)))
            char_inputs = CharInputs(
                char_to_token=char_to_token,
                is_token_final=char_is_token_final,
                position_in_token=char_position_in_token,
                hashes=char_hashes,
                mask=char_mask,
            )
            logits = self.character_head(token_hidden, token_logits, char_inputs)
            loss = (
                masked_binary_token_loss(
                    logits,
                    labels,
                    balance_classes=self.config.balance_character_loss,
                )
                if labels is not None
                else None
            )
            result = TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return result if return_dict else result.to_tuple()

        binary_labels = labels if labels is not None and self.config.num_labels == 1 else None
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=None if binary_labels is not None else labels,
            **kwargs,
        )
        if binary_labels is None:
            return outputs

        loss = masked_binary_token_loss(outputs.logits if return_dict else outputs[0], binary_labels)
        if not return_dict:
            return (loss,) + outputs
        outputs.loss = loss
        return outputs
