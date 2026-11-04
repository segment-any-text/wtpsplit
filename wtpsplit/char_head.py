"""Character-resolution boundary head.

SaT predicts one logit per subword token and `token_to_char_probs` then assigns it to the
*last character* of that token, leaving every other character at `-inf`. A boundary that
falls elsewhere is therefore unreachable at any threshold, which caps recall independently
of training.

This head removes the cap on both sides, turning the backbone swap from a trade into a
strict improvement. It predicts a logit at *every* character position from the covering
token's hidden state plus character-local features, so the reachable set is the whole
string and the ceiling is 1.0 by construction.

Behaviour-preserving initialisation. At initialisation the head reproduces the
existing last-character mapping exactly: the residual path is zero-initialised and a
learned bias suppresses non-final characters. An existing SaT checkpoint can therefore be
warm-started and fine-tuned rather than retrained.

The head is enabled 3 with `use_character_head=true` (in stage 3?). Both supported backbone
wrappers attach it from their serialized config, and `train_SM.py` dynamically constructs
the per-character inputs and padded targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from transformers import default_data_collator

from wtpsplit.utils import PRIMES

__all__ = [
    "CharacterResolutionHead",
    "CharacterDataCollator",
    "CharInputs",
    "build_char_inputs",
    "char_boundary_labels",
    "char_boundary_label_batch",
]

# Large enough to zero out a character after a sigmoid, small enough to stay finite in
# fp16 so the head can be trained in mixed precision (`-inf` would produce NaNs).
SUPPRESSION_BIAS = 12.0


@dataclass
class CharInputs:
    """Per-character indexing into the token sequence.

    All tensors are `[batch, num_chars]` except `hashes`, which is
    `[batch, num_chars, num_hashes]`.
    """

    char_to_token: torch.Tensor  # index of the covering token; 0 where padded
    is_token_final: torch.Tensor  # 1 where the character ends its token
    position_in_token: torch.Tensor  # offset of the character within its token, clamped
    hashes: torch.Tensor  # CANINE-style hashed codepoints
    mask: torch.Tensor  # 1 for real characters


def build_char_inputs(
    texts: list[str],
    offset_mappings: list[list[tuple[int, int]]],
    *,
    num_hashes: int = 8,
    num_buckets: int = 8192,
    max_position_in_token: int = 15,
    device: torch.device | None = None,
) -> CharInputs:
    """Map characters onto the tokens covering them.

    `offset_mappings` is what a fast tokenizer returns with `return_offsets_mapping=True`,
    already stripped of special tokens. Characters covered by no token (which a tokenizer
    can produce for stripped whitespace) point at token 0 and are suppressed by `mask`.
    """
    batch = len(texts)
    width = max((len(t) for t in texts), default=0)

    char_to_token = np.zeros((batch, width), dtype=np.int64)
    is_token_final = np.zeros((batch, width), dtype=np.int64)
    position_in_token = np.zeros((batch, width), dtype=np.int64)
    ordinals = np.zeros((batch, width), dtype=np.int64)
    mask = np.zeros((batch, width), dtype=np.int64)

    for i, (text, offsets) in enumerate(zip(texts, offset_mappings)):
        length = len(text)
        mask[i, :length] = 1
        ordinals[i, :length] = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)

        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            end = min(end, length)
            char_to_token[i, start:end] = token_index
            position_in_token[i, start:end] = np.minimum(np.arange(end - start), max_position_in_token)
            is_token_final[i, end - 1] = 1

    # Same hashing scheme as the CANINE-style char models in `wtpsplit.utils.hash_encode`,
    # vectorised over the batch.
    primes = np.array(PRIMES[:num_hashes], dtype=np.int64)
    hashes = ((ordinals[..., None] + 1) * primes) % num_buckets

    def tensor(array: NDArray[Any]) -> torch.Tensor:
        return torch.as_tensor(array, device=device)

    return CharInputs(
        char_to_token=tensor(char_to_token),
        is_token_final=tensor(is_token_final),
        position_in_token=tensor(position_in_token),
        hashes=tensor(hashes),
        mask=tensor(mask),
    )


def char_boundary_labels(sentences: list[str], separator: str) -> NDArray[np.int64]:
    """Per-character boundary targets for a document built from gold sentences.

    A label of 1 marks the **last content character** of each sentence, which is the
    position `indices_to_sentences` needs in order to reconstruct that split. This is the
    same convention `train_SM.py` already computes internally as
    `start_position + len(sentence) - 1` before downsampling it onto tokens — the
    character positions are available, they are simply discarded today.

    The final sentence gets no label: `indices_to_sentences` always appends the tail, so
    a boundary there is free for every model and training on it teaches nothing.
    """
    text_length = sum(len(s) for s in sentences) + len(separator) * max(len(sentences) - 1, 0)
    labels = np.zeros(text_length, dtype=np.int64)

    position = 0
    for index, sentence in enumerate(sentences):
        end = position + len(sentence)
        if index < len(sentences) - 1:
            labels[end - 1] = 1
        position = end + len(separator)
    return labels


def char_boundary_label_batch(
    documents: list[list[str]],
    separators: list[str],
    *,
    width: int | None = None,
    ignore_index: int = -100,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Padded `[batch, num_chars]` targets, with padding set to `ignore_index`."""
    per_document = [char_boundary_labels(s, sep) for s, sep in zip(documents, separators)]
    if width is None:
        width = max((len(labels) for labels in per_document), default=0)

    batch = np.full((len(per_document), width), ignore_index, dtype=np.int64)
    for i, labels in enumerate(per_document):
        length = min(len(labels), width)
        batch[i, :length] = labels[:length]
    return torch.as_tensor(batch, device=device)


class CharacterDataCollator:
    """Dynamically pad character features while token blocks remain fixed-width."""

    def __init__(self, special_tokens_before: int = 1):
        self.special_tokens_before = special_tokens_before

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        texts = [feature.pop("text") for feature in features]
        offsets = [feature.pop("offset_mapping") for feature in features]
        labels = [feature.pop("labels") for feature in features]
        batch: dict[str, torch.Tensor] = default_data_collator(features)

        char_inputs = build_char_inputs(texts, offsets)
        batch["char_to_token"] = char_inputs.char_to_token + self.special_tokens_before
        batch["char_is_token_final"] = char_inputs.is_token_final
        batch["char_position_in_token"] = char_inputs.position_in_token
        batch["char_hashes"] = char_inputs.hashes
        batch["char_mask"] = char_inputs.mask

        width = char_inputs.mask.shape[1]
        padded_labels = torch.full((len(labels), width), -100, dtype=torch.long)
        for index, values in enumerate(labels):
            padded_labels[index, : len(values)] = torch.as_tensor(values)
        batch["labels"] = padded_labels
        return batch


class CharacterResolutionHead(nn.Module):
    """Per-character boundary logits from per-token hidden states.

    The output is

        logit(c) = token_logit(t(c)) + bias(is_final(c)) + delta(c)

    where `delta` is a small convolutional network over character features. At
    initialisation `delta` is exactly zero and `bias` is `[-SUPPRESSION_BIAS, 0]`, so the
    head reproduces the current last-character behaviour and training only has to learn
    where to *relax* it.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 1,
        *,
        num_hashes: int = 8,
        num_buckets: int = 8192,
        char_embedding_size: int = 64,
        bottleneck_size: int = 128,
        kernel_size: int = 5,
        max_position_in_token: int = 15,
        project_tokens_first: bool = False,
        gate_logit_threshold: float | None = None,
    ):
        super().__init__()
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.max_position_in_token = max_position_in_token
        self.project_tokens_first = project_tokens_first
        self.gate_logit_threshold = gate_logit_threshold
        self.register_buffer(
            "_primes",
            torch.tensor(PRIMES[:num_hashes], dtype=torch.int64),
            persistent=False,
        )

        self.char_embeddings = nn.Embedding(num_buckets, char_embedding_size)
        self.position_embeddings = nn.Embedding(max_position_in_token + 1, char_embedding_size)
        self.final_embeddings = nn.Embedding(2, char_embedding_size)

        self.project = nn.Linear(hidden_size + char_embedding_size, bottleneck_size)
        self.activation = nn.GELU()
        # Depthwise-separable conv so neighbouring characters can disambiguate which
        # position inside a token carries the boundary, at negligible cost.
        self.context = nn.Conv1d(
            bottleneck_size,
            bottleneck_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=bottleneck_size,
        )
        self.delta = nn.Linear(bottleneck_size, num_labels)

        # `bias[1]` (token-final) stays at 0; `bias[0]` suppresses every other character.
        self.suppression = nn.Embedding(2, num_labels)

        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        """Initialise so the head reproduces `token_to_char_probs` exactly."""
        self.reset_missing_to_identity(
            delta_weight=True,
            delta_bias=True,
            suppression=True,
        )

    def reset_missing_to_identity(
        self,
        *,
        delta_weight: bool,
        delta_bias: bool,
        suppression: bool,
    ) -> None:
        """Restore identity values for components absent from a loaded checkpoint."""
        if delta_weight:
            nn.init.zeros_(self.delta.weight)
        if delta_bias:
            nn.init.zeros_(self.delta.bias)
        with torch.no_grad():
            if suppression:
                self.suppression.weight[0].fill_(-SUPPRESSION_BIAS)
                self.suppression.weight[1].fill_(0.0)

    def _depthwise(self, hidden: torch.Tensor) -> torch.Tensor:
        """`self.context` applied as shifted multiply-adds instead of `nn.Conv1d`.

        Mathematically identical and uses the same parameters, so checkpoints stay
        interchangeable with a standard depthwise `Conv1d`. PyTorch's depthwise kernel is
        pathologically slow on CPU for long sequences — it measured 1277 ms of a 1298 ms
        head, against 21 ms for everything else — whereas `kernel_size` shifted
        elementwise products are trivial.
        """
        x = hidden.transpose(1, 2)  # [B, C, T]
        kernel_size = self.context.kernel_size[0]
        padding = kernel_size // 2
        padded = nn.functional.pad(x, (padding, padding))
        weight = self.context.weight.squeeze(1)  # [C, k]

        length = x.size(-1)
        bias = cast(torch.Tensor, self.context.bias)
        out = bias.view(1, -1, 1).expand_as(x).clone()
        for offset in range(kernel_size):
            out = out + weight[:, offset].view(1, -1, 1) * padded[:, :, offset : offset + length]
        return out.transpose(1, 2)

    def _compute_hashes(self, ordinals: torch.Tensor) -> torch.Tensor:
        """Expand codepoint ordinals `[...]` into CANINE hashes `[..., num_hashes]` on-device."""
        primes = cast(torch.Tensor, self._primes).to(device=ordinals.device)
        return ((ordinals.to(torch.int64).unsqueeze(-1) + 1) * primes) % self.num_buckets

    def _resolve_hashes(self, hashes_or_ordinals: torch.Tensor) -> torch.Tensor:
        """Accept either 3D `[B, C, num_hashes]` hashes or 2D `[B, C]` codepoint ordinals."""
        if hashes_or_ordinals.ndim == 2:
            return self._compute_hashes(hashes_or_ordinals)
        return hashes_or_ordinals

    def _sparse_delta(
        self,
        token_hidden: torch.Tensor,  # [B, T, D]
        token_logits: torch.Tensor,  # [B, T, num_labels]
        char_inputs: CharInputs,
        *,
        gate_logit_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Compute `(tgt_b, tgt_c, delta, tgt_flat_tok, tgt_is_final)` on gated tokens only."""
        batch_size, num_chars = char_inputs.char_to_token.shape
        num_tokens = token_hidden.size(1)

        tok_pass = token_logits.max(dim=-1).values > gate_logit_threshold  # [B, T]
        target_chars = tok_pass.gather(1, char_inputs.char_to_token) & (char_inputs.mask != 0)  # [B, C]
        if not target_chars.any():
            return None

        kernel_size = self.context.kernel_size[0]
        padding = kernel_size // 2
        padded_target = nn.functional.pad(target_chars, (padding, padding), value=False)
        padded_halo = padded_target.clone()
        for shift in range(1, padding + 1):
            padded_halo[:, :-shift] |= padded_target[:, shift:]
            padded_halo[:, shift:] |= padded_target[:, :-shift]

        halo_b, halo_c_pad = padded_halo.nonzero(as_tuple=True)
        halo_c = halo_c_pad - padding
        in_bounds = (halo_c >= 0) & (halo_c < num_chars)
        safe_c = halo_c.clamp(0, num_chars - 1)

        c2t_sub = char_inputs.char_to_token[halo_b, safe_c]
        pos_sub = char_inputs.position_in_token[halo_b, safe_c]
        fin_sub = char_inputs.is_token_final[halo_b, safe_c]

        flat_tok_idx = halo_b * num_tokens + c2t_sub

        if char_inputs.hashes.ndim == 2:
            halo_hashes = self._compute_hashes(char_inputs.hashes[halo_b, safe_c])
        else:
            halo_hashes = char_inputs.hashes[halo_b, safe_c]
        c_feat = (
            self.char_embeddings(halo_hashes).sum(dim=-2)
            + self.position_embeddings(pos_sub)
            + self.final_embeddings(fin_sub)
        )

        if self.project_tokens_first:
            token_width = token_hidden.size(-1)
            total_tok_slots = batch_size * num_tokens
            tok_used = torch.zeros(total_tok_slots, dtype=torch.bool, device=token_hidden.device)
            tok_used[flat_tok_idx] = True
            uniq_tok = tok_used.nonzero(as_tuple=True)[0]
            tok_map = torch.zeros(total_tok_slots, dtype=torch.int64, device=token_hidden.device)
            tok_map[uniq_tok] = torch.arange(len(uniq_tok), device=token_hidden.device)
            tok_proj = nn.functional.linear(
                token_hidden.reshape(total_tok_slots, -1)[uniq_tok],
                self.project.weight[:, :token_width],
            )
            g_proj = tok_proj[tok_map[flat_tok_idx]]
            c_proj = nn.functional.linear(c_feat, self.project.weight[:, token_width:], self.project.bias)
            pre_conv = self.activation(g_proj + c_proj)
        else:
            g_hid = token_hidden.reshape(batch_size * num_tokens, -1)[flat_tok_idx]
            pre_conv = self.activation(self.project(torch.cat([g_hid, c_feat], dim=-1)))

        pre_conv = pre_conv.masked_fill(~in_bounds.unsqueeze(-1), 0.0)

        h_idx = padded_target[halo_b, halo_c_pad].nonzero(as_tuple=True)[0]
        weight = self.context.weight.squeeze(1)  # [C_Neck, k]
        bias = cast(torch.Tensor, self.context.bias)
        conv_out = bias.unsqueeze(0).expand(len(h_idx), -1).clone()
        for offset in range(kernel_size):
            conv_out = conv_out + pre_conv[h_idx + (offset - padding)] * weight[:, offset]
        delta: torch.Tensor = self.delta(conv_out)
        return halo_b[h_idx], halo_c[h_idx], delta, flat_tok_idx[h_idx], fin_sub[h_idx]

    def forward_sparse_compact(
        self,
        token_hidden: torch.Tensor,  # [B, T, D]
        token_logits: torch.Tensor,  # [B, T, num_labels]
        char_inputs: CharInputs,
        *,
        gate_logit_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(tgt_b, tgt_c, tgt_logits)` for gated characters only without `[B, C]` output allocation."""
        batch_size, num_chars = char_inputs.char_to_token.shape
        num_tokens = token_hidden.size(1)
        if batch_size == 0 or num_chars == 0:
            empty_idx = torch.empty(0, dtype=torch.int64, device=token_hidden.device)
            empty_log = torch.empty(
                (0, token_logits.size(-1)),
                dtype=token_logits.dtype,
                device=token_logits.device,
            )
            return empty_idx, empty_idx, empty_log
        if num_tokens == 0:
            raise ValueError("token_hidden must have at least one token when num_chars > 0")

        res = self._sparse_delta(
            token_hidden,
            token_logits,
            char_inputs,
            gate_logit_threshold=gate_logit_threshold,
        )
        if res is None:
            empty_idx = torch.empty(0, dtype=torch.int64, device=token_hidden.device)
            empty_log = torch.empty(
                (0, token_logits.size(-1)),
                dtype=token_logits.dtype,
                device=token_logits.device,
            )
            return empty_idx, empty_idx, empty_log

        tgt_b, tgt_c, delta, tgt_flat_tok, tgt_is_final = res
        g_log = token_logits.reshape(batch_size * num_tokens, -1)[tgt_flat_tok]
        tgt_logits: torch.Tensor = g_log + self.suppression(tgt_is_final) + delta
        return tgt_b, tgt_c, tgt_logits

    def forward_sparse(
        self,
        token_hidden: torch.Tensor,  # [B, T, D]
        token_logits: torch.Tensor,  # [B, T, num_labels]
        char_inputs: CharInputs,
        *,
        gate_logit_threshold: float,
    ) -> torch.Tensor:
        """Coarse-gated halo-compacted sparse evaluation.

        Evaluates `delta(c)` only on characters belonging to tokens where
        `max_l token_logits[b, t, l] > gate_logit_threshold` (using a `+-padding` character
        halo so the depthwise convolution sees the exact local neighbourhood), while
        leaving non-gated tokens at `token_logit(t(c)) + suppression(is_final(c))`.
        On gated characters, outputs match dense `forward` up to float32 GEMM tile
        accumulation order (measured `<= 1e-6` across the 10-language benchmark).
        """
        batch_size, num_chars = char_inputs.char_to_token.shape
        num_tokens = token_hidden.size(1)
        if batch_size == 0 or num_chars == 0:
            return torch.empty(
                (batch_size, num_chars, token_logits.size(-1)),
                dtype=token_logits.dtype,
                device=token_logits.device,
            )
        if num_tokens == 0:
            raise ValueError("token_hidden must have at least one token when num_chars > 0")

        index = char_inputs.char_to_token.unsqueeze(-1)
        gathered_logits = token_logits.gather(1, index.expand(-1, -1, token_logits.size(-1)))
        logits: torch.Tensor = gathered_logits + self.suppression(char_inputs.is_token_final)

        res = self._sparse_delta(
            token_hidden,
            token_logits,
            char_inputs,
            gate_logit_threshold=gate_logit_threshold,
        )
        if res is not None:
            tgt_b, tgt_c, delta, _, _ = res
            logits[tgt_b, tgt_c] = logits[tgt_b, tgt_c] + delta
        return logits.masked_fill(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)

    def forward(
        self,
        token_hidden: torch.Tensor,  # [B, T, D]
        token_logits: torch.Tensor,  # [B, T, num_labels]
        char_inputs: CharInputs,
        *,
        gate_logit_threshold: float | None = None,
    ) -> torch.Tensor:
        """Returns per-character logits `[B, C, num_labels]`."""
        eff_gate = gate_logit_threshold if gate_logit_threshold is not None else self.gate_logit_threshold
        if eff_gate is not None and not self.training:
            return self.forward_sparse(
                token_hidden,
                token_logits,
                char_inputs,
                gate_logit_threshold=eff_gate,
            )
        if self.project_tokens_first:
            return self._forward_project_tokens(token_hidden, token_logits, char_inputs)
        index = char_inputs.char_to_token.unsqueeze(-1)

        gathered_hidden = token_hidden.gather(1, index.expand(-1, -1, token_hidden.size(-1)))
        gathered_logits = token_logits.gather(1, index.expand(-1, -1, token_logits.size(-1)))

        hashes = self._resolve_hashes(char_inputs.hashes)
        char_features = (
            self.char_embeddings(hashes).sum(dim=-2)
            + self.position_embeddings(char_inputs.position_in_token)
            + self.final_embeddings(char_inputs.is_token_final)
        )

        hidden = self.activation(self.project(torch.cat([gathered_hidden, char_features], dim=-1)))
        hidden = self._depthwise(hidden)
        delta: torch.Tensor = self.delta(hidden)

        logits: torch.Tensor = gathered_logits + self.suppression(char_inputs.is_token_final) + delta
        return logits.masked_fill(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)

    def _forward_project_tokens(
        self,
        token_hidden: torch.Tensor,
        token_logits: torch.Tensor,
        char_inputs: CharInputs,
    ) -> torch.Tensor:
        """Gather bottleneck-width states instead of full backbone-width states.

        Linear projection commutes with gathering. Splitting the token and
        character contributions preserves parameters and real-arithmetic
        semantics, but changes mixed-precision rounding. This is opt-in so
        existing checkpoints retain their original execution path.
        """
        hidden = self.projected_features(token_hidden, char_inputs)
        index = char_inputs.char_to_token.unsqueeze(-1)
        gathered_logits = token_logits.gather(1, index.expand(-1, -1, token_logits.shape[-1]))
        delta: torch.Tensor = self.delta(hidden)
        logits: torch.Tensor = gathered_logits + self.suppression(char_inputs.is_token_final) + delta
        return logits.masked_fill(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)

    def projected_features(self, token_hidden: torch.Tensor, char_inputs: CharInputs) -> torch.Tensor:
        """Shared character trunk, before task-specific residual and suppression.

        Uses the opt-in project-before-gather path. No new parameters or state-dict
        names: exported sentence models remain ordinary CharacterResolutionHeads.
        """
        index = char_inputs.char_to_token.unsqueeze(-1)
        token_width = token_hidden.shape[-1]
        projected_tokens = nn.functional.linear(token_hidden, self.project.weight[:, :token_width])
        gathered_projection = projected_tokens.gather(1, index.expand(-1, -1, projected_tokens.shape[-1]))
        hashes = self._resolve_hashes(char_inputs.hashes)
        char_features = (
            self.char_embeddings(hashes).sum(dim=-2)
            + self.position_embeddings(char_inputs.position_in_token)
            + self.final_embeddings(char_inputs.is_token_final)
        )
        projected_chars = nn.functional.linear(
            char_features, self.project.weight[:, token_width:], self.project.bias
        )
        hidden = self.activation(gathered_projection + projected_chars)
        return self._depthwise(hidden)
