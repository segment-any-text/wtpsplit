from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.models.xlm_roberta import XLMRobertaForTokenClassification, XLMRobertaModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaEmbeddings,
    XLMRobertaLayer,
    XLMRobertaPooler,
)

from wtpsplit._training_loss import masked_binary_token_loss
from wtpsplit.char_head import CharacterResolutionHead, CharInputs
from wtpsplit.configs import SubwordXLMConfig
from wtpsplit.model_registry import resolve_effective_lookahead


def _get_head_mask(
    head_mask: Optional[Tensor],
    num_hidden_layers: int,
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tensor]:
    """Expand a one- or two-dimensional head mask to encoder shape."""
    if head_mask is None:
        return None
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if head_mask.dim() != 5:
        raise ValueError(f"head_mask.dim() != 5, got {head_mask.dim()}")
    if dtype is not None:
        head_mask = head_mask.to(dtype=dtype)
    return head_mask


class SubwordXLMForTokenClassification(XLMRobertaForTokenClassification):
    config_class = SubwordXLMConfig

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = SubwordXLMRobertaModel(config, add_pooling_layer=False)
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
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        char_to_token=None,
        char_is_token_final=None,
        char_position_in_token=None,
        char_hashes=None,
        char_mask=None,
        hashed_ids=None,
        language_ids=None,
        **kwargs,
    ):
        """Run token-level SaT or the optional character-resolution boundary head."""
        if self.character_head is None:
            if labels is None or self.config.num_labels != 1:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    **kwargs,
                )

            return_dict = kwargs.get("return_dict", self.config.use_return_dict)
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=None,
                **kwargs,
            )
            loss = masked_binary_token_loss(outputs.logits if return_dict else outputs[0], labels)
            if not return_dict:
                return (loss,) + outputs
            outputs.loss = loss
            return outputs

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
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        token_hidden = outputs.last_hidden_state
        token_logits = self.classifier(self.dropout(token_hidden))
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


class SubwordXLMRobertaModel(XLMRobertaModel):
    config_class = SubwordXLMConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->XLMRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = SubwordXLMRobertaEncoder(config)
        self.lookahead_split_layers = config.lookahead_split_layers
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None
        self.effective_lookahead = resolve_effective_lookahead(config, supports_split_layers=True)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = _get_head_mask(head_mask, self.config.num_hidden_layers, dtype=getattr(self, "dtype", None))

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            self.config, attention_mask, input_shape, self.effective_lookahead, device, self.dtype
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            init_attention_mask=attention_mask,
            dtype=self.dtype,
            cache_position=cache_position,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


def get_extended_attention_mask(
    config,
    attention_mask: Tensor,
    input_shape: Tuple[int],
    lookahead: Optional[int] = None,
    device: torch.device = None,
    dtype: torch.float = None,
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """

    # if not (attention_mask.dim() == 2 and config.is_decoder):
    # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
    # if device is not None:
    #     warnings.warn(
    #         "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
    #     )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]

    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if config.is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask
            )
        if lookahead is not None:
            # lookahead mask of shape [batch_size, 1, seq_length, seq_length]
            # the current token should attend to the next `lookahead` tokens
            # the current token should not attend to the previous `lookahead` tokens
            _, seq_length = attention_mask.shape
            # Create a lookahead mask
            lookahead_mask = torch.tril(torch.ones(seq_length, seq_length), diagonal=lookahead, out=None).to(
                attention_mask.device
            )
            # Combine the attention mask with the lookahead mask
            extended_attention_mask = attention_mask[:, None, None, :] * lookahead_mask
        else:
            # [batch, 1, seq, seq] for transformers 5 SDPA; semantics
            # equivalent to [batch, 1, 1, seq] broadcast (mask depends only on key positions).
            extended_attention_mask = attention_mask[:, None, None, :].expand(-1, 1, attention_mask.size(1), -1)
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with Roberta->XLMRoberta
class SubwordXLMRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        init_attention_mask: Optional[torch.Tensor] = None,
        dtype: torch.float = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        next_decoder_cache = () if use_cache else None

        if cache_position is None:
            _pkv_len = past_key_values[0][0].shape[2] if past_key_values else 0
            _seq_len = hidden_states.size(1)
            cache_position = torch.arange(_pkv_len, _pkv_len + _seq_len, device=hidden_states.device, dtype=torch.long)

        for i, layer_module in enumerate(self.layer):
            # MODIFIED: if lookahead_split_layers is given, use causal mask starting from that layer
            if self.config.lookahead_split_layers is not None:
                if i == self.config.lookahead_split_layers:
                    attention_mask = get_extended_attention_mask(
                        self.config, init_attention_mask, init_attention_mask.shape, 0, hidden_states.device, dtype
                    )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values else None

            layer_kwargs = dict(
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                head_mask=layer_head_mask,
                past_key_values=past_key_value,
                cache_position=cache_position,
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module, **fwd_kwargs):
                    def custom_forward(*inputs):
                        extra = {
                            k: v
                            for k, v in fwd_kwargs.items()
                            if k not in ("attention_mask", "encoder_hidden_states", "encoder_attention_mask")
                        }
                        return module(
                            inputs[0],
                            attention_mask=inputs[1],
                            encoder_hidden_states=inputs[2],
                            encoder_attention_mask=inputs[3],
                            **extra,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module, **layer_kwargs),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, **layer_kwargs)

            # transformers 5 XLMRobertaLayer returns the hidden-state tensor directly.
            hidden_states = layer_outputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# if __name__ == "__main__":
#     # test XLM
#     from transformers import AutoTokenizer

#     model_str = "xlm-roberta-base"
#     config = SubwordXLMConfig.from_pretrained(model_str)
#     config.num_labels = 4
#     config.num_hidden_layers = 12
#     config.lookahead = 48
#     config.lookahead_split_layers = 6
#     backbone = SubwordXLMForTokenClassification.from_pretrained(model_str, config=config)
#     print(summary(backbone, depth=4))

#     # some sample input
#     text = "A sentence. Now we move on. And on and this is the last sentence. Now, we are starting to move on to the next sentence. This is the last sentence."
#     tokenizer = AutoTokenizer.from_pretrained(model_str)

#     tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False, pad_to_multiple_of=512, padding=True)
#     from tokenizers import AddedToken

#     tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
#     print(tokenizer.tokenize(text))
#     print(tokenizer.encode(text))
#     print(tokens)

#     # forward pass
#     print(backbone(**tokens))


_LEGACY_MODELS = {
    "LACanineEmbeddings",
    "LACanineSelfAttention",
    "LACanineAttention",
    "LACanineOutput",
    "LACanineLayer",
    "LACanineEncoder",
    "CausalCharactersToMolecules",
    "LACanineModel",
    "LACanineForTokenClassification",
    "BertCharModel",
    "BertCharForTokenClassification",
}
_LEGACY_CONFIGS = {"BertCharConfig", "LACanineConfig"}


def __getattr__(name: str):
    if name in _LEGACY_MODELS:
        from wtpsplit.legacy import models as legacy_models

        return getattr(legacy_models, name)
    if name in _LEGACY_CONFIGS:
        from wtpsplit.legacy import configs as legacy_configs

        return getattr(legacy_configs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Preserve the historical ``import wtpsplit.models`` registration behavior while
# keeping all Auto* mutations behind one documented internal boundary.
from wtpsplit.models_modernbert import SubwordModernBertForTokenClassification  # noqa: E402,F401
from wtpsplit.model_registry import register_sat_models  # noqa: E402

register_sat_models()
