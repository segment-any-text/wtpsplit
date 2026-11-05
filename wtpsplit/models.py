from typing import Any, List, Optional, Sequence, Tuple, Union, cast

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

from wtpsplit._training_loss import boundary_loss_balancing, masked_binary_token_loss
from wtpsplit.char_head import CharacterResolutionHead, CharInputs
from wtpsplit.configs import SubwordXLMConfig
from wtpsplit.model_registry import resolve_effective_lookahead


_LOOKAHEAD_MASK_CACHE: dict[tuple[torch.device, torch.dtype, Optional[int]], Tensor] = {}
_MAX_LOOKAHEAD_CACHE_ENTRIES = 32
_DEFAULT_LOOKAHEAD_ALLOC_LEN = 514
_FLEX_BLOCK_MASK_CACHE: dict[tuple[torch.device, int, Optional[int]], Any] = {}
_MAX_FLEX_CACHE_ENTRIES = 32


def _get_cached_flex_block_mask(
    seq_length: int,
    lookahead: Optional[int],
    device: torch.device,
) -> Any:
    """Return cached broadcastable ``BlockMask`` (`B=None, H=None`) on ``device``."""
    from torch.nn.attention.flex_attention import create_block_mask

    key = (device, seq_length, None if lookahead is None else int(lookahead))
    cached = _FLEX_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    if lookahead is None:
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= 0,
            B=None,
            H=None,
            Q_LEN=seq_length,
            KV_LEN=seq_length,
            device=device,
        )
    else:
        la = int(lookahead)
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: kv_idx <= q_idx + la,
            B=None,
            H=None,
            Q_LEN=seq_length,
            KV_LEN=seq_length,
            device=device,
        )
    if len(_FLEX_BLOCK_MASK_CACHE) >= _MAX_FLEX_CACHE_ENTRIES and key not in _FLEX_BLOCK_MASK_CACHE:
        _FLEX_BLOCK_MASK_CACHE.pop(next(iter(_FLEX_BLOCK_MASK_CACHE)))
    _FLEX_BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


def _get_cached_additive_mask(
    seq_length: int,
    lookahead: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return cached additive attention bias on ``device`` for ``lookahead`` (or ``None``)."""
    key = (device, dtype, None if lookahead is None else int(lookahead))
    cached = _LOOKAHEAD_MASK_CACHE.get(key)
    if cached is not None and cached.size(-1) >= seq_length:
        if lookahead is None:
            return cached[:, :, :, :seq_length]
        return cached[:, :, :seq_length, :seq_length]

    alloc_len = max(seq_length, _DEFAULT_LOOKAHEAD_ALLOC_LEN)
    if lookahead is None:
        additive = torch.zeros((1, 1, 1, alloc_len), device=device, dtype=dtype) * torch.finfo(dtype).min
    else:
        one_zero = (
            torch.ones((alloc_len, alloc_len), device=device, dtype=dtype)
            .tril_(diagonal=lookahead)
            .view(1, 1, alloc_len, alloc_len)
        )
        additive = (1.0 - one_zero) * torch.finfo(dtype).min
    if len(_LOOKAHEAD_MASK_CACHE) >= _MAX_LOOKAHEAD_CACHE_ENTRIES and key not in _LOOKAHEAD_MASK_CACHE:
        _LOOKAHEAD_MASK_CACHE.pop(next(iter(_LOOKAHEAD_MASK_CACHE)))
    _LOOKAHEAD_MASK_CACHE[key] = additive
    if lookahead is None:
        return additive[:, :, :, :seq_length]
    return additive[:, :, :seq_length, :seq_length]


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
            CharacterResolutionHead(
                config.hidden_size, num_labels=config.num_labels,
                project_tokens_first=config.character_head_project_tokens_first,
            )
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

    def encode_tokens(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Run the subword backbone and token classifier without requiring `char_*` inputs."""
        kwargs.pop("return_dict", None)
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        token_hidden: Tensor = outputs.last_hidden_state
        token_logits: Tensor = self.classifier(self.dropout(token_hidden))
        return token_hidden, token_logits

    def forward_features(self, **inputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the encoder once and expose the character head's distillation taps."""
        head = self.character_head
        if head is None or not head.project_tokens_first:
            raise ValueError("feature distillation requires a project_tokens_first character head")
        token_inputs = {k: v for k, v in inputs.items() if not k.startswith("char_")}
        hidden, token_logits = self.encode_tokens(**token_inputs)
        chars = CharInputs(
            char_to_token=inputs["char_to_token"], is_token_final=inputs["char_is_token_final"],
            position_in_token=inputs["char_position_in_token"], hashes=inputs["char_hashes"],
            mask=inputs["char_mask"],
        )
        projected = torch.nn.functional.linear(hidden, head.project.weight[:, :hidden.shape[-1]])
        logits, trunk = head.forward_bottleneck(projected, token_logits, chars, hidden.shape[-1])
        return logits, {"token_hidden": hidden, "projected_tokens": projected,
                        "token_logits": token_logits, "char_trunk": trunk}

    def encode_tokens_varlen(
        self,
        packed_input_ids: Tensor,
        packed_position_ids: Tensor,
        doc_ids: Tensor,
        *,
        doc_runs: Optional[list[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the subword backbone and token classifier on a 1D packed sequence ``[1, N_valid]``."""
        n_valid = packed_input_ids.size(1)
        device = packed_input_ids.device
        dtype = self.dtype
        if n_valid == 0:
            empty_h = torch.empty((1, 0, self.config.hidden_size), device=device, dtype=dtype)
            empty_l = torch.empty((1, 0, self.config.num_labels), device=device, dtype=dtype)
            return empty_h, empty_l

        emb = self.roberta.embeddings
        if (
            not self.training
            and not torch.is_grad_enabled()
            and not torch.jit.is_tracing()
            and not packed_input_ids.is_meta
            and getattr(emb, "position_embedding_type", "absolute") == "absolute"
        ):
            out_emb: Tensor = emb.word_embeddings(packed_input_ids)
            if hasattr(emb, "token_type_embeddings"):
                out_emb.add_(emb.token_type_embeddings.weight[0].view(1, 1, -1))
            out_emb.add_(emb.position_embeddings(packed_position_ids))
            embedding_output: Tensor = emb.LayerNorm(out_emb)
        else:
            embedding_output = emb(
                input_ids=packed_input_ids,
                position_ids=packed_position_ids,
            )

        eff_la = self.roberta.effective_lookahead
        attn_impl = getattr(self.config, "_attn_implementation", None)
        split_layer = self.config.lookahead_split_layers

        if (
            doc_runs is not None
            and attn_impl != "flex_attention"
            and not self.training
            and not torch.is_grad_enabled()
            and not torch.jit.is_tracing()
        ):
            hidden_states: Tensor = embedding_output
            cur_la = eff_la
            for i, layer_module in enumerate(self.roberta.encoder.layer):
                if split_layer is not None and i == split_layer:
                    cur_la = 0
                self_attn = layer_module.attention.self
                num_heads = self_attn.num_attention_heads
                head_dim = self_attn.attention_head_size
                q_all, k_all, v_all = _project_qkv_fused(self_attn, hidden_states)
                attn_ctx = torch.empty_like(q_all)
                for t_start, t_end, run_len, run_count in doc_runs:
                    q_r = q_all[:, t_start:t_end, :].view(run_count, run_len, num_heads, head_dim).transpose(1, 2)
                    k_r = k_all[:, t_start:t_end, :].view(run_count, run_len, num_heads, head_dim).transpose(1, 2)
                    v_r = v_all[:, t_start:t_end, :].view(run_count, run_len, num_heads, head_dim).transpose(1, 2)
                    if cur_la is None:
                        ctx_r = torch.nn.functional.scaled_dot_product_attention(
                            q_r, k_r, v_r, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                    elif cur_la == 0:
                        ctx_r = torch.nn.functional.scaled_dot_product_attention(
                            q_r, k_r, v_r, attn_mask=None, dropout_p=0.0, is_causal=True
                        )
                    else:
                        mask_r = _get_cached_additive_mask(run_len, cur_la, device, q_r.dtype).expand(
                            run_count, 1, run_len, run_len
                        )
                        ctx_r = torch.nn.functional.scaled_dot_product_attention(
                            q_r, k_r, v_r, attn_mask=mask_r, dropout_p=0.0, is_causal=False
                        )
                    attn_ctx[:, t_start:t_end, :] = ctx_r.transpose(1, 2).reshape(1, t_end - t_start, -1)
                hidden_states = _run_ffn_residual_norm(layer_module, attn_ctx, hidden_states)
            token_logits_runs: Tensor = self.classifier(self.dropout(hidden_states))
            return hidden_states, token_logits_runs

        def build_varlen_mask(la: Optional[int]) -> Tensor:
            if attn_impl == "flex_attention" and (la is None or la >= 0):
                from torch.nn.attention.flex_attention import create_block_mask

                if la is None:
                    return cast(
                        Tensor,
                        create_block_mask(
                            lambda b, h, q_idx, kv_idx: doc_ids[q_idx] == doc_ids[kv_idx],
                            B=1,
                            H=None,
                            Q_LEN=n_valid,
                            KV_LEN=n_valid,
                            device=device,
                        ),
                    )
                la_int = int(la)
                return cast(
                    Tensor,
                    create_block_mask(
                        lambda b, h, q_idx, kv_idx: (doc_ids[q_idx] == doc_ids[kv_idx])
                        & (kv_idx <= q_idx + la_int),
                        B=1,
                        H=None,
                        Q_LEN=n_valid,
                        KV_LEN=n_valid,
                        device=device,
                    ),
                )
            same_doc = doc_ids[:, None] == doc_ids[None, :]
            if la is not None:
                idx = torch.arange(n_valid, device=device)
                valid_pair = same_doc & (idx[None, :] <= idx[:, None] + int(la))
            else:
                valid_pair = same_doc
            min_val = torch.finfo(dtype).min
            neg_zero = torch.zeros((), device=device, dtype=dtype) * min_val
            min_tensor = torch.tensor(min_val, device=device, dtype=dtype)
            return torch.where(valid_pair, neg_zero, min_tensor).view(1, 1, n_valid, n_valid)

        attn_mask = build_varlen_mask(eff_la)
        hidden_states: Tensor = embedding_output
        split_layer = self.config.lookahead_split_layers
        for i, layer_module in enumerate(self.roberta.encoder.layer):
            if split_layer is not None and i == split_layer:
                attn_mask = build_varlen_mask(0)
            hidden_states = layer_module(hidden_states, attention_mask=attn_mask)

        token_logits: Tensor = self.classifier(self.dropout(hidden_states))
        return hidden_states, token_logits

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
            loss = masked_binary_token_loss(
                outputs.logits if return_dict else outputs[0], labels,
                balance_classes=boundary_loss_balancing(self.config, character_head=False),
            )
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
                balance_classes=boundary_loss_balancing(self.config, character_head=True),
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
            attention_mask._sat_all_ones = True  # type: ignore[attr-defined]

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

        emb = self.embeddings
        p_idx = getattr(emb, "padding_idx", None)
        can_fast_embed = (
            input_ids is not None
            and inputs_embeds is None
            and token_type_ids is None
            and position_ids is None
            and past_key_values_length == 0
            and p_idx is not None
            and getattr(emb, "position_embedding_type", "absolute") == "absolute"
            and not self.training
            and not torch.is_grad_enabled()
            and not torch.jit.is_tracing()
            and not input_ids.is_meta
            and (p_idx + 1 + seq_length) <= emb.position_embeddings.num_embeddings
        )
        if can_fast_embed:
            p0 = int(p_idx) + 1
            out_emb: torch.Tensor = emb.word_embeddings(input_ids)
            out_emb.add_(emb.token_type_embeddings.weight[0].view(1, 1, -1))
            is_unpadded_ids = (
                seq_length > 0
                and (
                    (
                        device.type != "cpu"
                        and bool(getattr(attention_mask, "_sat_all_ones", False))
                    )
                    or (
                        device.type == "cpu"
                        and bool((input_ids[:, -1] != p_idx).all())
                        and bool((input_ids != p_idx).all())
                    )
                )
            )
            if is_unpadded_ids:
                out_emb.add_(emb.position_embeddings.weight[p0 : p0 + seq_length].unsqueeze(0))
            else:
                mask_ne = input_ids.ne(p_idx).int()
                pos_ids = (torch.cumsum(mask_ne, dim=1).type_as(mask_ne) * mask_ne).long() + int(p_idx)
                out_emb.add_(emb.position_embeddings(pos_ids))
            embedding_output = emb.LayerNorm(out_emb)
        else:
            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

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


@torch._dynamo.disable  # type: ignore[misc]
def get_extended_attention_mask(
    config: Any,
    attention_mask: Tensor,
    input_shape: Sequence[int],
    lookahead: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
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
    resolved_dtype = (
        dtype
        if dtype is not None
        else (attention_mask.dtype if attention_mask.is_floating_point() else torch.float32)
    )
    min_val = torch.finfo(resolved_dtype).min
    mask_device = attention_mask.device

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :].to(dtype=resolved_dtype)
        return (1.0 - extended_attention_mask) * min_val

    if attention_mask.dim() == 2:
        batch_size, seq_length = attention_mask.shape
        if config.is_decoder and lookahead is None:
            decoder_mask = cast(
                Tensor,
                ModuleUtilsMixin.create_extended_attention_mask_for_decoder(  # type: ignore[no-untyped-call]
                    input_shape, attention_mask
                ),
            )
            decoder_mask = decoder_mask.to(dtype=resolved_dtype)
            return (1.0 - decoder_mask) * min_val

        if torch.jit.is_tracing():
            if lookahead is not None:
                lookahead_mask = torch.tril(
                    torch.ones((seq_length, seq_length), device=mask_device, dtype=resolved_dtype),
                    diagonal=lookahead,
                )
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=resolved_dtype) * lookahead_mask
                return (1.0 - extended_attention_mask) * min_val
            mask_4d = attention_mask[:, None, None, :].to(dtype=resolved_dtype)
            inv_mask = (1.0 - mask_4d) * min_val
            return inv_mask.expand(batch_size, 1, seq_length, seq_length)

        is_all_ones = bool(getattr(attention_mask, "_sat_all_ones", False)) or (
            mask_device.type == "cpu"
            and seq_length > 0
            and bool((attention_mask[:, -1] == 1).all())
            and bool((attention_mask == 1).all())
        )
        if (
            getattr(config, "_attn_implementation", None) == "flex_attention"
            and seq_length > 0
            and (lookahead is None or lookahead >= 0)
            and (is_all_ones or bool((attention_mask[:, 0] != 0).all()))
        ):
            if is_all_ones:
                return cast(Tensor, _get_cached_flex_block_mask(seq_length, lookahead, mask_device))
            from torch.nn.attention.flex_attention import create_block_mask

            pad_bool = attention_mask.to(dtype=torch.bool)
            if lookahead is None:
                return cast(
                    Tensor,
                    create_block_mask(
                        lambda b, h, q_idx, kv_idx: pad_bool[b, kv_idx],
                        B=batch_size,
                        H=None,
                        Q_LEN=seq_length,
                        KV_LEN=seq_length,
                        device=mask_device,
                    ),
                )
            la = int(lookahead)
            return cast(
                Tensor,
                create_block_mask(
                    lambda b, h, q_idx, kv_idx: (kv_idx <= q_idx + la) & pad_bool[b, kv_idx],
                    B=batch_size,
                    H=None,
                    Q_LEN=seq_length,
                    KV_LEN=seq_length,
                    device=mask_device,
                ),
            )
        if lookahead is not None:
            # Lookahead mask of shape [batch_size, 1, seq_length, seq_length]:
            # query token i attends to key tokens j <= i + lookahead.
            additive_la = _get_cached_additive_mask(
                seq_length, lookahead, mask_device, resolved_dtype
            )
            if is_all_ones:
                return additive_la.expand(batch_size, 1, seq_length, seq_length)
            inv_pad = (1.0 - attention_mask[:, None, None, :].to(dtype=resolved_dtype)) * min_val
            return torch.minimum(additive_la, inv_pad)

        # [batch, 1, seq, seq] view for transformers 5 SDPA: compute additive mask on
        # [batch, 1, 1, seq] (512x smaller in memory) before zero-copy .expand().
        if is_all_ones:
            additive_zero = _get_cached_additive_mask(
                seq_length, None, mask_device, resolved_dtype
            )
            return additive_zero.expand(batch_size, 1, seq_length, seq_length)
        mask_4d = attention_mask[:, None, None, :].to(dtype=resolved_dtype)
        inv_mask = (1.0 - mask_4d) * min_val
        return inv_mask.expand(batch_size, 1, seq_length, seq_length)

    raise ValueError(
        f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
    )


def _project_qkv_fused(self_attn: Any, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Project `(Q, K, V)` in a single `[..., 3 * D]` GEMM with version/pointer-tracked weight caching."""
    qw, qb = self_attn.query.weight, self_attn.query.bias
    kw, kb = self_attn.key.weight, self_attn.key.bias
    vw, vb = self_attn.value.weight, self_attn.value.bias
    key = (
        qw._version,
        kw._version,
        vw._version,
        qb._version if qb is not None else -1,
        kb._version if kb is not None else -1,
        vb._version if vb is not None else -1,
        qw.data_ptr(),
        kw.data_ptr(),
        vw.data_ptr(),
        qb.data_ptr() if qb is not None else 0,
        kb.data_ptr() if kb is not None else 0,
        vb.data_ptr() if vb is not None else 0,
        qw.dtype,
        qw.device,
    )
    if self_attn.__dict__.get("_qkv_cache_key") != key:
        w_qkv = torch.cat([qw, kw, vw], dim=0)
        b_qkv = (
            torch.cat([qb, kb, vb], dim=0)
            if (qb is not None and kb is not None and vb is not None)
            else None
        )
        self_attn.__dict__["_qkv_cache_key"] = key
        self_attn.__dict__["_qkv_cache_val"] = (w_qkv, b_qkv)
    w_qkv, b_qkv = self_attn.__dict__["_qkv_cache_val"]
    if b_qkv is None and (qb is not None or kb is not None or vb is not None):
        return self_attn.query(hidden_states), self_attn.key(hidden_states), self_attn.value(hidden_states)
    qkv: Tensor = torch.nn.functional.linear(hidden_states, w_qkv, b_qkv)
    q_t, k_t, v_t = qkv.chunk(3, dim=-1)
    return q_t, k_t, v_t


def _run_ffn_residual_norm(layer_module: Any, attn_ctx: Tensor, hidden_states: Tensor) -> Tensor:
    """Execute attention output projection + LayerNorm + FFN + LayerNorm with in-place GELU and residuals."""
    attn_proj: Tensor = layer_module.attention.output.dense(attn_ctx)
    if attn_proj.dtype == hidden_states.dtype:
        attn_proj.add_(hidden_states)
    else:
        attn_proj = attn_proj + hidden_states
    attn_out: Tensor = layer_module.attention.output.LayerNorm(attn_proj)

    inter_dense: Tensor = layer_module.intermediate.dense(attn_out)
    act_fn = layer_module.intermediate.intermediate_act_fn
    if type(act_fn).__name__ == "GELUActivation" and not inter_dense.is_meta:
        inter_out: Tensor = getattr(torch._C._nn, "gelu_")(inter_dense)
    else:
        inter_out = act_fn(inter_dense)

    ffn_proj: Tensor = layer_module.output.dense(inter_out)
    if ffn_proj.dtype == attn_out.dtype:
        ffn_proj.add_(attn_out)
    else:
        ffn_proj = ffn_proj + attn_out
    out_norm: Tensor = layer_module.output.LayerNorm(ffn_proj)
    return out_norm


# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with Roberta->XLMRoberta
class SubwordXLMRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        compression = getattr(config, "experimental_token_compression", None)
        if compression is not None:
            from wtpsplit.experiment.token_compression import TokenCompression
            if config.num_hidden_layers < 2:
                raise ValueError("Compression needs at least two encoder layers")
            self.token_compressor = TokenCompression(config.hidden_size, compression)

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
        if hasattr(self, "token_compressor"):
            from wtpsplit.experiment.token_compression import compressed_forward
            if output_attentions or encoder_hidden_states is not None or past_key_values is not None or use_cache:
                raise ValueError("Compression pilot supports encoder-only inference without attention outputs/cache")
            return compressed_forward(self, hidden_states, attention_mask, head_mask,
                                      init_attention_mask, output_hidden_states, return_dict)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        next_decoder_cache = () if use_cache else None

        if cache_position is None and (past_key_values is not None or use_cache):
            _pkv_len = past_key_values[0][0].shape[2] if past_key_values else 0
            _seq_len = hidden_states.size(1)
            cache_position = torch.arange(_pkv_len, _pkv_len + _seq_len, device=hidden_states.device, dtype=torch.long)

        can_fast_encoder = (
            not self.training
            and not torch.is_grad_enabled()
            and not torch.jit.is_tracing()
            and not hidden_states.is_meta
            and not output_attentions
            and not output_hidden_states
            and head_mask is None
            and encoder_hidden_states is None
            and past_key_values is None
            and not use_cache
            and getattr(self.config, "_attn_implementation", "sdpa") in (None, "sdpa")
            and not getattr(self.config, "add_cross_attention", False)
            and getattr(self.config, "position_embedding_type", "absolute") == "absolute"
        )
        if can_fast_encoder:
            bsz, seq_len, _ = hidden_states.shape
            is_unpadded_cpu = (
                hidden_states.device.type == "cpu"
                and init_attention_mask is not None
                and (
                    bool(getattr(init_attention_mask, "_sat_all_ones", False))
                    or (init_attention_mask.numel() > 0 and bool((init_attention_mask == 1).all()))
                )
            )
            cur_la = resolve_effective_lookahead(self.config, supports_split_layers=True)
            split_layer = self.config.lookahead_split_layers
            for i, layer_module in enumerate(self.layer):
                if split_layer is not None and i == split_layer:
                    cur_la = 0
                    if not is_unpadded_cpu and init_attention_mask is not None:
                        attention_mask = get_extended_attention_mask(
                            self.config, init_attention_mask, init_attention_mask.shape, 0, hidden_states.device, dtype
                        )
                self_attn = layer_module.attention.self
                num_heads = self_attn.num_attention_heads
                head_dim = self_attn.attention_head_size
                q_t, k_t, v_t = _project_qkv_fused(self_attn, hidden_states)
                q_t = q_t.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                k_t = k_t.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                v_t = v_t.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                if is_unpadded_cpu and cur_la is None:
                    ctx_t = torch.nn.functional.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                elif is_unpadded_cpu and cur_la == 0:
                    ctx_t = torch.nn.functional.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                else:
                    ctx_t = torch.nn.functional.scaled_dot_product_attention(
                        q_t, k_t, v_t, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
                attn_ctx = ctx_t.transpose(1, 2).reshape(bsz, seq_len, -1)
                hidden_states = _run_ffn_residual_norm(layer_module, attn_ctx, hidden_states)

            if not return_dict:
                return (hidden_states,)
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )

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
