import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from transformers import AutoModel, AutoModelForTokenClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertEncoder, BertForTokenClassification, BertModel, BertPooler
from transformers.models.canine.modeling_canine import (
    _PRIMES,
    ACT2FN,
    BaseModelOutput,
    CanineAttention,
    CanineEmbeddings,
    CanineEncoder,
    CanineForTokenClassification,
    CanineIntermediate,
    CanineLayer,
    CanineModel,
    CanineModelOutputWithPooling,
    CanineOutput,
    CaninePooler,
    CanineSelfAttention,
    CanineSelfOutput,
    CharactersToMolecules,
    ConvProjection,
    TokenClassifierOutput,
)
from transformers.models.xlm_roberta import (
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaEmbeddings,
    XLMRobertaEncoder,
    XLMRobertaPooler,
)

from wtpsplit.configs import BertCharConfig, LACanineConfig, SubwordXLMConfig
from wtpsplit.utils import Constants


# added n-gram representations
class LACanineEmbeddings(CanineEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        self.ngram_order = getattr(config, "ngram_order", 1)
        if self.ngram_order > 1:
            raise NotImplementedError("n-gram representations are not implemented.")

        shard_embedding_size = config.hidden_size // config.num_hash_functions
        for j in range(2, self.ngram_order + 1):
            for i in range(config.num_hash_functions):
                name = f"HashBucketCodepointEmbedder_{i}_{j}"
                setattr(
                    self,
                    name,
                    nn.Embedding(config.num_hash_buckets, shard_embedding_size),
                )

    def _embed_hash_buckets(self, input_ids=None, hashed_ids=None):
        embedding_size = self.config.hidden_size
        num_hashes = self.config.num_hash_functions
        num_buckets = self.config.num_hash_buckets

        assert (input_ids is None) + (
            hashed_ids is None
        ) == 1, "Either `input_ids` or `hashed_ids` must be provided (and not both!)."

        """Converts IDs (e.g. codepoints) into embeddings via multiple hashing."""
        if embedding_size % num_hashes != 0:
            raise ValueError(f"Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0")

        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        embedding_shards = []
        for i in range(num_hashes):
            name = f"HashBucketCodepointEmbedder_{i}"

            hash_ids = ((input_ids + 1) * _PRIMES[i]) % num_buckets if hashed_ids is None else hashed_ids[:, :, i]

            shard_embeddings = getattr(self, name)(hash_ids)
            embedding_shards.append(shard_embeddings)

        return torch.cat(embedding_shards, dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        hashed_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        elif hashed_ids is not None:
            input_shape = hashed_ids.size()[:-1]
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(input_ids, hashed_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LACanineSelfAttention(CanineSelfAttention):
    def __init__(self, config):
        super(CanineSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.lookahead = config.lookahead
        self.lookahead_block_size = config.lookahead_block_size

        if self.lookahead is not None:
            max_positions = config.max_position_embeddings
            assert max_positions % self.lookahead_block_size == 0

            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        (
                            max_positions // self.lookahead_block_size,
                            max_positions // self.lookahead_block_size,
                        ),
                        dtype=torch.uint8,
                    ),
                    diagonal=self.lookahead,
                )
                .repeat_interleave(self.lookahead_block_size, dim=1)
                .repeat_interleave(self.lookahead_block_size, dim=0)
                .view(
                    1,
                    1,
                    max_positions,
                    max_positions,
                ),
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        from_tensor: torch.Tensor,
        to_tensor: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mixed_query_layer = self.query(from_tensor)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        key_layer = self.transpose_for_scores(self.key(to_tensor))
        value_layer = self.transpose_for_scores(self.value(to_tensor))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = from_tensor.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=from_tensor.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=from_tensor.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                # if attention_mask is 3D, do the following:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                attention_mask = (1.0 - attention_mask.to(attention_scores.dtype)) * torch.finfo(
                    attention_scores.dtype
                ).min

            # Apply the attention mask (precomputed for all layers in CanineModel forward() function)
            attention_scores = attention_scores + attention_mask

        if self.lookahead is not None:
            query_length, key_length = query_layer.size(-2), key_layer.size(-2)
            causal_mask = self.bias[
                :,
                :,
                key_length - query_length : key_length,
                :key_length,
            ].to(torch.bool)
            mask_value = torch.finfo(attention_scores.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attention_scores.dtype).to(attention_scores.device)
            attention_scores = torch.where(
                causal_mask,
                attention_scores,
                mask_value,
            )

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LACanineAttention(CanineAttention):
    """
    Additional arguments related to local attention:

        - **local** (`bool`, *optional*, defaults to `False`) -- Whether to apply local attention.
        - **always_attend_to_first_position** (`bool`, *optional*, defaults to `False`) -- Should all blocks be able to
          attend
        to the `to_tensor`'s first position (e.g. a [CLS] position)? - **first_position_attends_to_all** (`bool`,
        *optional*, defaults to `False`) -- Should the *from_tensor*'s first position be able to attend to all
        positions within the *from_tensor*? - **attend_from_chunk_width** (`int`, *optional*, defaults to 128) -- The
        width of each block-wise chunk in `from_tensor`. - **attend_from_chunk_stride** (`int`, *optional*, defaults to
        128) -- The number of elements to skip when moving to the next block in `from_tensor`. -
        **attend_to_chunk_width** (`int`, *optional*, defaults to 128) -- The width of each block-wise chunk in
        *to_tensor*. - **attend_to_chunk_stride** (`int`, *optional*, defaults to 128) -- The number of elements to
        skip when moving to the next block in `to_tensor`.
    """

    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position: bool = False,
        first_position_attends_to_all: bool = False,
        attend_from_chunk_width: int = 128,
        attend_from_chunk_stride: int = 128,
        attend_to_chunk_width: int = 128,
        attend_to_chunk_stride: int = 128,
    ):
        super(CanineAttention, self).__init__()
        self.self = LACanineSelfAttention(config)
        self.output = CanineSelfOutput(config)
        self.pruned_heads = set()

        # additional arguments related to local attention
        self.local = local
        if attend_from_chunk_width < attend_from_chunk_stride:
            raise ValueError(
                "`attend_from_chunk_width` < `attend_from_chunk_stride` would cause sequence positions to get skipped."
            )
        if attend_to_chunk_width < attend_to_chunk_stride:
            raise ValueError(
                "`attend_to_chunk_width` < `attend_to_chunk_stride`would cause sequence positions to get skipped."
            )
        self.always_attend_to_first_position = always_attend_to_first_position
        self.first_position_attends_to_all = first_position_attends_to_all
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride


class LACanineOutput(CanineOutput):
    def __init__(self, config):
        super().__init__(config)
        self.bottleneck_size = config.hidden_size // config.bottleneck_factor
        self.hidden_size = config.hidden_size

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.language_adapter = config.language_adapter

        if self.language_adapter in {"on", "shared"}:
            # language adapter init
            dummy_ff_downs = [nn.Linear(self.hidden_size, self.bottleneck_size) for _ in range(config.n_languages)]
            dummy_ff_ups = [nn.Linear(self.bottleneck_size, self.hidden_size) for _ in range(config.n_languages)]

            self.lang_ff_down_weights = nn.Embedding(
                config.n_languages,
                (self.hidden_size * self.bottleneck_size),
                _weight=torch.stack([l.weight.data.T.reshape(-1) for l in dummy_ff_downs]),
            )
            self.lang_ff_up_weights = nn.Embedding(
                config.n_languages,
                (self.bottleneck_size * self.hidden_size),
                _weight=torch.stack([l.weight.data.T.reshape(-1) for l in dummy_ff_ups]),
            )
            self.lang_ff_down_biases = nn.Embedding(
                config.n_languages,
                self.bottleneck_size,
                _weight=torch.stack([l.bias.data for l in dummy_ff_downs]),
            )
            self.lang_ff_up_biases = nn.Embedding(
                config.n_languages,
                self.hidden_size,
                _weight=torch.stack([l.bias.data for l in dummy_ff_ups]),
            )

    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        input_tensor: torch.FloatTensor,
        language_ids: Optional[torch.LongTensor],
    ) -> torch.FloatTensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if self.language_adapter == "off":
            return hidden_states
        elif self.language_adapter == "on":
            ff_down_weights = self.lang_ff_down_weights(language_ids)
            ff_up_weights = self.lang_ff_up_weights(language_ids)
            ff_down_biases = self.lang_ff_down_biases(language_ids)
            ff_up_biases = self.lang_ff_up_biases(language_ids)
        elif self.language_adapter == "shared":
            ff_down_weights = self.lang_ff_down_weights.weight.mean(0, keepdim=True)
            ff_up_weights = self.lang_ff_up_weights.weight.mean(0, keepdim=True)
            ff_down_biases = self.lang_ff_down_biases.weight.mean(0, keepdim=True)
            ff_up_biases = self.lang_ff_up_biases.weight.mean(0, keepdim=True)

        ff_down_weights = ff_down_weights.view(-1, self.hidden_size, self.bottleneck_size)
        ff_up_weights = ff_down_weights.view(-1, self.bottleneck_size, self.hidden_size)

        down = self.act_fn(torch.matmul(hidden_states, ff_down_weights) + ff_down_biases.unsqueeze(1))
        up = torch.matmul(down, ff_up_weights) + ff_up_biases.unsqueeze(1)

        hidden_states = self.LayerNorm(hidden_states + up)

        return hidden_states


class LACanineLayer(CanineLayer):
    def __init__(
        self,
        config,
        local,
        always_attend_to_first_position,
        first_position_attends_to_all,
        attend_from_chunk_width,
        attend_from_chunk_stride,
        attend_to_chunk_width,
        attend_to_chunk_stride,
    ):
        super(CanineLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        assert self.chunk_size_feed_forward == 0
        self.seq_len_dim = 1
        self.attention = LACanineAttention(
            config,
            local,
            always_attend_to_first_position,
            first_position_attends_to_all,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride,
        )
        self.intermediate = CanineIntermediate(config)
        self.output = LACanineOutput(config)

    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        language_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, language_ids)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        raise NotImplementedError()


class LACanineEncoder(CanineEncoder):
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position=False,
        first_position_attends_to_all=False,
        attend_from_chunk_width=128,
        attend_from_chunk_stride=128,
        attend_to_chunk_width=128,
        attend_to_chunk_stride=128,
    ):
        super(CanineEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                LACanineLayer(
                    config,
                    local,
                    always_attend_to_first_position,
                    first_position_attends_to_all,
                    attend_from_chunk_width,
                    attend_from_chunk_stride,
                    attend_to_chunk_width,
                    attend_to_chunk_stride,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        language_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    language_ids,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    language_ids,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CausalCharactersToMolecules(CharactersToMolecules):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:
        # MODIFIED: no cls encoding!
        assert char_encoding.shape[1] % self.conv.stride[0] == 0

        # char_encoding has shape [batch, char_seq, hidden_size]
        # We transpose it to be [batch, hidden_size, char_seq]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)
        downsampled = self.activation(downsampled)

        result = downsampled

        result = self.LayerNorm(result)

        return result


class LACanineModel(CanineModel):
    config_class = LACanineConfig

    def __init__(self, config, add_pooling_layer=True):
        super(CanineModel, self).__init__(config)
        self.config = config
        final_shallow_config = copy.deepcopy(config)
        final_shallow_config.num_hidden_layers = 1

        initial_shallow_config = copy.deepcopy(config)
        initial_shallow_config.num_hidden_layers = 1

        if config.lookahead is not None:
            initial_shallow_config.lookahead = config.lookahead
            initial_shallow_config.lookahead_block_size = config.downsampling_rate

            # the ConvProjection must not leak future context (ks = 1)
            final_shallow_config.lookahead_block_size = config.downsampling_rate
            final_shallow_config.lookahead = 0

            config.lookahead = 0

        self.char_embeddings = LACanineEmbeddings(config)
        # shallow/low-dim transformer encoder to get a initial character encoding
        self.initial_char_encoder = LACanineEncoder(
            initial_shallow_config,
            local=True,
            always_attend_to_first_position=False,
            first_position_attends_to_all=False,
            attend_from_chunk_width=config.local_transformer_stride,
            attend_from_chunk_stride=config.local_transformer_stride,
            attend_to_chunk_width=config.local_transformer_stride,
            attend_to_chunk_stride=config.local_transformer_stride,
        )
        self.chars_to_molecules = CausalCharactersToMolecules(config)
        # deep transformer encoder
        self.encoder = LACanineEncoder(config)
        self.projection = ConvProjection(config)
        # shallow/low-dim transformer encoder to get a final character encoding
        self.final_char_encoder = LACanineEncoder(final_shallow_config)

        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""

        rate = self.config.downsampling_rate

        return torch.repeat_interleave(molecules, repeats=rate, dim=-2)

    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # if broadcasted, use MaxPool1d to just pool the last dimension
        if char_attention_mask.shape[2] == 1:
            return torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(
                char_attention_mask.squeeze(2)
            ).unsqueeze(2)

        return torch.nn.MaxPool2d(kernel_size=downsampling_rate, stride=downsampling_rate)(char_attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        language_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CanineModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        extended_molecule_attention_mask = self._downsample_attention_mask(
            extended_attention_mask, downsampling_rate=self.config.downsampling_rate
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # `input_char_embeddings`: shape (batch_size, char_seq, char_dim)
        input_char_embeddings = self.char_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Contextualize character embeddings using shallow Transformer.
        # We use a 3D attention mask for the local attention.
        # `input_char_encoding`: shape (batch_size, char_seq_len, char_dim)
        if attention_mask.ndim == 2:
            char_attention_mask = self._create_3d_attention_mask_from_input_mask(
                input_ids or inputs_embeds, attention_mask
            )
        else:
            char_attention_mask = attention_mask
        init_chars_encoder_outputs = self.initial_char_encoder(
            input_char_embeddings,
            language_ids=language_ids,
            attention_mask=char_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        input_char_encoding = init_chars_encoder_outputs.last_hidden_state

        # Downsample chars to molecules.
        # The following lines have dimensions: [batch, molecule_seq, molecule_dim].
        # In this transformation, we change the dimensionality from `char_dim` to
        # `molecule_dim`, but do *NOT* add a resnet connection. Instead, we rely on
        # the resnet connections (a) from the final char transformer stack back into
        # the original char transformer stack and (b) the resnet connections from
        # the final char transformer stack back into the deep BERT stack of
        # molecules.
        #
        # Empirically, it is critical to use a powerful enough transformation here:
        # mean pooling causes training to diverge with huge gradient norms in
        # this region of the model; using a convolution here resolves this issue. From
        # this, it seems that molecules and characters require a very different
        # feature space; intuitively, this makes sense.
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)

        # Deep BERT encoder
        # `molecule_sequence_output`: shape (batch_size, mol_seq_len, mol_dim)
        encoder_outputs = self.encoder(
            init_molecule_encoding,
            language_ids=language_ids,
            attention_mask=extended_molecule_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        molecule_sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(molecule_sequence_output) if self.pooler is not None else None

        # Upsample molecules back to characters.
        # `repeated_molecules`: shape (batch_size, char_seq_len, mol_hidden_size)
        repeated_molecules = self._repeat_molecules(molecule_sequence_output, char_seq_length=input_shape[-1])

        # Concatenate representations (contextualized char embeddings and repeated molecules):
        # `concat`: shape [batch_size, char_seq_len, molecule_hidden_size+char_hidden_final]
        concat = torch.cat([input_char_encoding, repeated_molecules], dim=-1)

        # Project representation dimension back to hidden_size
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        sequence_output = self.projection(concat)

        # Apply final shallow Transformer
        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size])
        final_chars_encoder_outputs = self.final_char_encoder(
            sequence_output,
            language_ids=language_ids,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = final_chars_encoder_outputs.last_hidden_state

        if output_hidden_states:
            deep_encoder_hidden_states = encoder_outputs.hidden_states if return_dict else encoder_outputs[1]
            all_hidden_states = (
                all_hidden_states
                + init_chars_encoder_outputs.hidden_states
                + deep_encoder_hidden_states
                + final_chars_encoder_outputs.hidden_states
            )

        if output_attentions:
            deep_encoder_self_attentions = encoder_outputs.attentions if return_dict else encoder_outputs[-1]
            all_self_attentions = (
                all_self_attentions
                + init_chars_encoder_outputs.attentions
                + deep_encoder_self_attentions
                + final_chars_encoder_outputs.attentions
            )

        if not return_dict:
            output = (sequence_output, pooled_output)
            output += tuple(v for v in [all_hidden_states, all_self_attentions] if v is not None)
            return output

        return CanineModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LACanineForTokenClassification(CanineForTokenClassification):
    config_class = LACanineConfig

    def __init__(self, config):
        super(CanineForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.canine = LACanineModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def set_language_adapters(self, lang):
        lang_index = Constants.LANG_CODE_TO_INDEX[lang]

        for module in self.modules():
            if isinstance(module, LACanineOutput):
                module.lang_ff_down_biases.weight = nn.Parameter(module.lang_ff_down_biases.weight[[lang_index]])
                module.lang_ff_down_weights.weight = nn.Parameter(module.lang_ff_down_weights.weight[[lang_index]])
                module.lang_ff_up_biases.weight = nn.Parameter(module.lang_ff_up_biases.weight[[lang_index]])
                module.lang_ff_up_weights.weight = nn.Parameter(module.lang_ff_up_weights.weight[[lang_index]])
                module.language_adapter = "shared"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        language_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hashed_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        inputs_embeds = self.canine.char_embeddings._embed_hash_buckets(
            input_ids=input_ids,
            hashed_ids=hashed_ids,
        )
        input_ids = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.canine(
            input_ids,
            language_ids=language_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertCharModel(BertModel):
    config_class = BertCharConfig

    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = LACanineEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings


class BertCharForTokenClassification(BertForTokenClassification):
    config_class = BertCharConfig

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertCharModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hashed_ids: Optional[torch.Tensor] = None,
        language_ids=None,
        return_dict: Optional[bool] = None,
    ):
        inputs_embeds = self.bert.embeddings._embed_hash_buckets(
            input_ids=input_ids,
            hashed_ids=hashed_ids,
        )
        input_ids = None

        return super().forward(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )


class SubwordXLMForTokenClassification(XLMRobertaForTokenClassification):
    config_class = SubwordXLMConfig

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SubwordXLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hashed_ids: Optional[torch.Tensor] = None,
        language_ids=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SubwordXLMRobertaModel(XLMRobertaModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->XLMRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)

        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None
        self.effective_lookahead = (
            config.lookahead // config.num_hidden_layers if config.lookahead is not None else None
        )

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
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.effective_lookahead
        )

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
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
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
        self,
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
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            if lookahead:
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
                extended_attention_mask = attention_mask[:, None, None, :]
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


AutoModel.register(LACanineConfig, LACanineModel)
AutoModelForTokenClassification.register(LACanineConfig, LACanineForTokenClassification)

AutoModel.register(BertCharConfig, BertCharModel)
AutoModelForTokenClassification.register(BertCharConfig, BertCharForTokenClassification)

AutoModel.register(SubwordXLMConfig, SubwordXLMForTokenClassification)
AutoModelForTokenClassification.register(SubwordXLMConfig, SubwordXLMForTokenClassification)

if __name__ == "__main__":
    # test XLM
    from transformers import AutoConfig, AutoTokenizer

    model_str = "xlm-roberta-base"
    config = AutoConfig.from_pretrained(model_str)
    config.num_labels = 4
    config.num_hidden_layers = 1
    backbone = SubwordXLMForTokenClassification.from_pretrained(model_str, config=config)
    print(summary(backbone, depth=4))

    # some sample input
    text = "A sentence. Now we move on. And on and this is the last sentence. Now, we are starting to move on to the next sentence. This is the last sentence."
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False, pad_to_multiple_of=512, padding=True)
    from tokenizers import AddedToken

    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    print(tokenizer.tokenize(text))
    print(tokenizer.encode(text))
    print(tokens)
    
    # forward pass
    print(backbone(**tokens))
