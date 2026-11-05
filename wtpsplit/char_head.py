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
_PRIMES_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


@torch._dynamo.disable  # type: ignore[untyped-decorator]
def _get_device_primes_impl(device: torch.device, num_hashes: int) -> torch.Tensor:
    key = (device, num_hashes)
    cached = _PRIMES_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(PRIMES[:num_hashes], dtype=torch.int64, device=device)
        _PRIMES_CACHE[key] = cached
    return cached


def _get_device_primes(device: torch.device, num_hashes: int) -> torch.Tensor:
    return cast(torch.Tensor, _get_device_primes_impl(device, num_hashes))


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
        fuse_depthwise_delta: bool = False,
    ):
        super().__init__()
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.max_position_in_token = max_position_in_token
        self.project_tokens_first = project_tokens_first
        self.gate_logit_threshold = gate_logit_threshold
        self.fuse_depthwise_delta = fuse_depthwise_delta
        self._residue_table_cache: tuple[int, int, torch.dtype, torch.device, int, int, torch.Tensor] | None = None
        self._fused_dw_delta_cache: tuple[Any, ...] | None = None
        self._proj_char_tables_cache: tuple[Any, ...] | None = None
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
            primes_buf = cast(torch.Tensor, self._primes)
            if not primes_buf.is_meta:
                primes_buf.copy_(_get_device_primes(primes_buf.device, self.num_hashes))
            if suppression:
                self.suppression.weight[0].fill_(-SUPPRESSION_BIAS)
                self.suppression.weight[1].fill_(0.0)

    def _depthwise(self, hidden: torch.Tensor) -> torch.Tensor:
        """`self.context` applied as shifted multiply-adds instead of `nn.Conv1d`.

        Mathematically identical and uses the same parameters, so checkpoints stay
        interchangeable with a standard depthwise `Conv1d`. Operates directly in the
        native `[B, T, C]` channel-last layout so neither `F.pad` nor `self.delta`
        pays a `[B, T, C] <-> [B, C, T]` transpose copy.
        """
        kernel_size = self.context.kernel_size[0]
        padding = kernel_size // 2
        length = hidden.size(1)
        padded = nn.functional.pad(hidden, (0, 0, padding, padding))
        weight = self.context.weight.squeeze(1)  # [C, k]

        bias = cast(torch.Tensor, self.context.bias)
        out = bias.view(1, 1, -1).expand_as(hidden).clone()
        can_inplace = (
            not self.training
            and not torch.is_grad_enabled()
            and not bool(getattr(torch.jit, "is_tracing")())
        )
        if can_inplace:
            for offset in range(kernel_size):
                out.add_(padded[:, offset : offset + length, :] * weight[:, offset].view(1, 1, -1))
        else:
            for offset in range(kernel_size):
                out = out + padded[:, offset : offset + length, :] * weight[:, offset].view(1, 1, -1)
        return out

    @torch._dynamo.disable  # type: ignore[untyped-decorator]
    def _get_fused_dw_delta_params_impl(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(w_fused, b_fused)` composing `self.delta` (`[L, C]`) and `self.context` (`[C, 1, K]`)."""
        cw = self.context.weight
        cb = cast(torch.Tensor, self.context.bias)
        dw = self.delta.weight
        db = cast(torch.Tensor, self.delta.bias)
        key = (
            cw._version,
            cw.data_ptr(),
            cb._version,
            cb.data_ptr(),
            dw._version,
            dw.data_ptr(),
            db._version,
            db.data_ptr(),
            dw.dtype,
            dw.device,
        )
        cached = self._fused_dw_delta_cache
        if cached is not None and cached[0] == key:
            return cast(tuple[torch.Tensor, torch.Tensor], cached[1])
        k_size = self.context.kernel_size[0]
        num_labels, c_dim = dw.shape
        w_conv = cw.squeeze(1)  # [C, K]
        w_fused = (dw.unsqueeze(1) * w_conv.t().unsqueeze(0)).reshape(num_labels * k_size, c_dim).contiguous()
        b_fused = db + torch.mv(dw, cb)
        pair = (w_fused, b_fused)
        self._fused_dw_delta_cache = (key, pair)
        return pair

    @torch._dynamo.disable  # type: ignore[untyped-decorator]
    def _get_projected_char_tables_impl(
        self, token_width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(w_tok_contig, e_res_proj, e_pf_proj)` pre-projecting `char_embeddings` and `(pos, final)` into bottleneck space."""
        cew = self.char_embeddings.weight
        pew = self.position_embeddings.weight
        few = self.final_embeddings.weight
        pw = self.project.weight
        pb = cast(torch.Tensor, self.project.bias)
        key = (
            cew._version,
            cew.data_ptr(),
            pew._version,
            pew.data_ptr(),
            few._version,
            few.data_ptr(),
            pw._version,
            pw.data_ptr(),
            pb._version,
            pb.data_ptr(),
            pw.dtype,
            pw.device,
            self.num_hashes,
            self.num_buckets,
            token_width,
        )
        cached = self._proj_char_tables_cache
        if cached is not None and cached[0] == key:
            return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], cached[1])
        w_tok = pw[:, :token_width].contiguous()
        w_char = pw[:, token_width:].contiguous()
        e_res = self._get_residue_embedding_table()
        e_res_proj = nn.functional.linear(e_res, w_char).contiguous()
        e_pf = (pew[:, None, :] + few[None, :, :]).reshape(-1, pew.size(-1))
        e_pf_proj = nn.functional.linear(e_pf, w_char, pb).contiguous()
        triple = (w_tok, e_res_proj, e_pf_proj)
        self._proj_char_tables_cache = (key, triple)
        return triple

    def _fused_depthwise_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate `self.delta(self._depthwise(hidden))` via a single `[B, T, C] @ [C, L*K]` GEMM."""
        k_size = self.context.kernel_size[0]
        pad = k_size // 2
        b_sz, seq_len, _ = hidden.shape
        num_labels = self.delta.out_features
        w_fused, b_fused = cast(
            tuple[torch.Tensor, torch.Tensor],
            self._get_fused_dw_delta_params_impl(),
        )
        taps = nn.functional.linear(hidden, w_fused).view(b_sz, seq_len, num_labels, k_size)
        padded_taps = nn.functional.pad(taps, (0, 0, 0, 0, pad, pad))
        out = b_fused.view(1, 1, num_labels).expand(b_sz, seq_len, num_labels).clone()
        for offset in range(k_size):
            out.add_(padded_taps[:, offset : offset + seq_len, :, offset])
        return out

    def _compute_hashes(self, ordinals: torch.Tensor) -> torch.Tensor:
        """Expand codepoint ordinals `[...]` into CANINE hashes `[..., num_hashes]` on-device."""
        primes = _get_device_primes(ordinals.device, self.num_hashes)
        return ((ordinals.to(torch.int64).unsqueeze(-1) + 1) * primes) % self.num_buckets

    def _resolve_hashes(self, hashes_or_ordinals: torch.Tensor) -> torch.Tensor:
        """Accept either 3D `[B, C, num_hashes]` hashes or 2D `[B, C]` codepoint ordinals."""
        if hashes_or_ordinals.ndim == 2:
            return self._compute_hashes(hashes_or_ordinals)
        return hashes_or_ordinals

    @torch._dynamo.disable  # type: ignore[untyped-decorator]
    def _get_residue_embedding_table_impl(self) -> torch.Tensor:
        """Return pre-summed `[num_buckets, char_embedding_size]` residue embedding table.

        Because `((c + 1) * p_k) % M == (((c + 1) % M) * p_k) % M` for every `c >= -1`,
        the 8-hash tuple depends only on the residue `r = (c + 1) % M` in `0..M-1`.
        Pre-summing the 8 hash embeddings for each of the `M` residues reduces `[B, C]`
        ordinal hashing + `[B, C, 8, D]` 4D embedding + `.sum(dim=-2)` to a single 2D
        `F.embedding` lookup while remaining 100% bit-exact (`0.0` ulp difference).
        """
        weight = self.char_embeddings.weight
        ver = weight._version
        ptr = weight.data_ptr()
        dt = weight.dtype
        dev = weight.device
        nh = self.num_hashes
        nb = self.num_buckets
        cached = self._residue_table_cache
        if (
            cached is not None
            and cached[0] == ver
            and cached[1] == ptr
            and cached[2] == dt
            and cached[3] == dev
            and cached[4] == nh
            and cached[5] == nb
        ):
            return cached[6]
        primes = _get_device_primes(dev, nh)
        r = torch.arange(nb, dtype=torch.int64, device=dev).unsqueeze(-1)
        lut = (r * primes) % nb
        table: torch.Tensor = nn.functional.embedding_bag(lut, weight, mode="sum")
        self._residue_table_cache = (ver, ptr, dt, dev, nh, nb, table)
        return table

    def _get_residue_embedding_table(self) -> torch.Tensor:
        return cast(torch.Tensor, self._get_residue_embedding_table_impl())

    def _embed_char_hashes(
        self,
        hashes_or_ordinals: torch.Tensor,
        *,
        is_ordinals: bool | None = None,
    ) -> torch.Tensor:
        """Sum `self.char_embeddings` across hashes without materializing a 4D `[B, C, H, D]` tensor."""
        if is_ordinals is None:
            is_ordinals = hashes_or_ordinals.ndim == 2
        if bool(getattr(torch.jit, "is_tracing")()):
            hashes = self._compute_hashes(hashes_or_ordinals) if is_ordinals else hashes_or_ordinals
            out_trace: torch.Tensor = self.char_embeddings(hashes).sum(dim=-2)
            return out_trace
        if is_ordinals:
            if (
                not self.training
                and not torch.is_grad_enabled()
                and not hashes_or_ordinals.is_meta
                and not self.char_embeddings.weight.is_meta
            ):
                nb = self.num_buckets
                if (nb & (nb - 1)) == 0:
                    residues = (hashes_or_ordinals + 1) & (nb - 1)
                else:
                    residues = torch.remainder(hashes_or_ordinals + 1, nb)
                return nn.functional.embedding(residues, self._get_residue_embedding_table())
            hashes = self._compute_hashes(hashes_or_ordinals)
        else:
            hashes = hashes_or_ordinals
        if hashes.numel() == 0:
            out_empty: torch.Tensor = self.char_embeddings(hashes).sum(dim=-2)
            return out_empty
        flat_hashes = hashes.reshape(-1, hashes.size(-1))
        bag_out: torch.Tensor = nn.functional.embedding_bag(
            flat_hashes, self.char_embeddings.weight, mode="sum"
        )
        return bag_out.reshape(*hashes.shape[:-1], -1)

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

        can_fuse_lut = (
            self.project_tokens_first
            and self.fuse_depthwise_delta
            and not self.training
            and not torch.is_grad_enabled()
            and not bool(getattr(torch.jit, "is_tracing")())
            and char_inputs.hashes.ndim == 2
            and not token_hidden.is_meta
        )
        if can_fuse_lut:
            token_width = token_hidden.size(-1)
            w_tok, e_res_proj, e_pf_proj = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                self._get_projected_char_tables_impl(token_width),
            )
            total_tok_slots = batch_size * num_tokens
            tok_used = torch.zeros(total_tok_slots, dtype=torch.bool, device=token_hidden.device)
            tok_used[flat_tok_idx] = True
            uniq_tok = tok_used.nonzero(as_tuple=True)[0]
            tok_map = torch.zeros(total_tok_slots, dtype=torch.int64, device=token_hidden.device)
            tok_map[uniq_tok] = torch.arange(len(uniq_tok), device=token_hidden.device)
            tok_proj = nn.functional.linear(
                token_hidden.reshape(total_tok_slots, -1)[uniq_tok],
                w_tok,
            )
            g_proj = tok_proj[tok_map[flat_tok_idx]]
            nb = self.num_buckets
            halo_ord = char_inputs.hashes[halo_b, safe_c]
            residues = (halo_ord + 1) & (nb - 1) if (nb & (nb - 1)) == 0 else torch.remainder(halo_ord + 1, nb)
            pf_idx = (pos_sub << 1) | fin_sub
            g_proj.add_(nn.functional.embedding(residues, e_res_proj))
            g_proj.add_(nn.functional.embedding(pf_idx, e_pf_proj))
            pre_conv = self.activation(g_proj)
        else:
            c_feat = self._embed_char_hashes(
                char_inputs.hashes[halo_b, safe_c],
                is_ordinals=(char_inputs.hashes.ndim == 2),
            )
            if not self.training and not torch.is_grad_enabled() and not bool(getattr(torch.jit, "is_tracing")()):
                c_feat.add_(self.position_embeddings(pos_sub)).add_(self.final_embeddings(fin_sub))
            else:
                c_feat = c_feat + self.position_embeddings(pos_sub) + self.final_embeddings(fin_sub)

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
                g_proj.add_(c_proj)
                pre_conv = self.activation(g_proj)
            else:
                g_hid = token_hidden.reshape(batch_size * num_tokens, -1)[flat_tok_idx]
                pre_conv = self.activation(self.project(torch.cat([g_hid, c_feat], dim=-1)))

        pre_conv.masked_fill_(~in_bounds.unsqueeze(-1), 0.0)

        h_idx = padded_target[halo_b, halo_c_pad].nonzero(as_tuple=True)[0]
        if self.fuse_depthwise_delta and not self.training and not bool(getattr(torch.jit, "is_tracing")()):
            num_labels = self.delta.out_features
            w_fused, b_fused = cast(
                tuple[torch.Tensor, torch.Tensor],
                self._get_fused_dw_delta_params_impl(),
            )
            taps = nn.functional.linear(pre_conv, w_fused).view(len(pre_conv), num_labels, kernel_size)
            delta = b_fused.unsqueeze(0).expand(len(h_idx), num_labels).clone()
            for offset in range(kernel_size):
                delta.add_(taps[h_idx + (offset - padding), :, offset])
        else:
            weight = self.context.weight.squeeze(1)  # [C_Neck, k]
            bias = cast(torch.Tensor, self.context.bias)
            conv_out = bias.unsqueeze(0).expand(len(h_idx), -1).clone()
            for offset in range(kernel_size):
                conv_out.add_(pre_conv[h_idx + (offset - padding)] * weight[:, offset])
            delta = self.delta(conv_out)
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
        logits: torch.Tensor = token_logits.gather(1, index.expand(-1, -1, token_logits.size(-1)))
        logits.add_(self.suppression(char_inputs.is_token_final))

        res = self._sparse_delta(
            token_hidden,
            token_logits,
            char_inputs,
            gate_logit_threshold=gate_logit_threshold,
        )
        if res is not None:
            tgt_b, tgt_c, delta, _, _ = res
            logits[tgt_b, tgt_c] = logits[tgt_b, tgt_c] + delta
        return logits.masked_fill_(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)

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

        can_inplace = (
            not self.training
            and not torch.is_grad_enabled()
            and not bool(getattr(torch.jit, "is_tracing")())
        )
        char_features = self._embed_char_hashes(char_inputs.hashes)
        if can_inplace:
            char_features.add_(self.position_embeddings(char_inputs.position_in_token)).add_(
                self.final_embeddings(char_inputs.is_token_final)
            )
        else:
            char_features = (
                char_features
                + self.position_embeddings(char_inputs.position_in_token)
                + self.final_embeddings(char_inputs.is_token_final)
            )

        hidden = self.activation(self.project(torch.cat([gathered_hidden, char_features], dim=-1)))
        if self.fuse_depthwise_delta and can_inplace:
            delta = self._fused_depthwise_delta(hidden)
        else:
            hidden = self._depthwise(hidden)
            delta = self.delta(hidden)

        if can_inplace:
            gathered_logits.add_(self.suppression(char_inputs.is_token_final)).add_(delta)
            return gathered_logits.masked_fill_(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)
        logits: torch.Tensor = gathered_logits + self.suppression(char_inputs.is_token_final) + delta
        return logits.masked_fill(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)

    def _projected_pre_conv(self, token_hidden: torch.Tensor, char_inputs: CharInputs) -> torch.Tensor:
        index = char_inputs.char_to_token.unsqueeze(-1)
        token_width = token_hidden.shape[-1]
        can_inplace = (
            not self.training
            and not torch.is_grad_enabled()
            and not bool(getattr(torch.jit, "is_tracing")())
        )
        if (
            self.fuse_depthwise_delta
            and can_inplace
            and char_inputs.hashes.ndim == 2
            and not token_hidden.is_meta
        ):
            w_tok, e_res_proj, e_pf_proj = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                self._get_projected_char_tables_impl(token_width),
            )
            projected_tokens = nn.functional.linear(token_hidden, w_tok)
            gathered_projection = projected_tokens.gather(1, index.expand(-1, -1, projected_tokens.shape[-1]))
            nb = self.num_buckets
            ordinals = char_inputs.hashes
            residues = (ordinals + 1) & (nb - 1) if (nb & (nb - 1)) == 0 else torch.remainder(ordinals + 1, nb)
            pf_idx = (char_inputs.position_in_token << 1) | char_inputs.is_token_final
            gathered_projection.add_(nn.functional.embedding(residues, e_res_proj))
            gathered_projection.add_(nn.functional.embedding(pf_idx, e_pf_proj))
            out_fused: torch.Tensor = self.activation(gathered_projection)
            return out_fused

        projected_tokens = nn.functional.linear(token_hidden, self.project.weight[:, :token_width])
        gathered_projection = projected_tokens.gather(1, index.expand(-1, -1, projected_tokens.shape[-1]))
        char_features = self._embed_char_hashes(char_inputs.hashes)
        if can_inplace:
            char_features.add_(self.position_embeddings(char_inputs.position_in_token)).add_(
                self.final_embeddings(char_inputs.is_token_final)
            )
            projected_chars = nn.functional.linear(
                char_features, self.project.weight[:, token_width:], self.project.bias
            )
            gathered_projection.add_(projected_chars)
            out_inplace: torch.Tensor = self.activation(gathered_projection)
            return out_inplace

        char_features = (
            char_features
            + self.position_embeddings(char_inputs.position_in_token)
            + self.final_embeddings(char_inputs.is_token_final)
        )
        projected_chars = nn.functional.linear(
            char_features, self.project.weight[:, token_width:], self.project.bias
        )
        out: torch.Tensor = self.activation(gathered_projection + projected_chars)
        return out

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
        index = char_inputs.char_to_token.unsqueeze(-1)
        gathered_logits = token_logits.gather(1, index.expand(-1, -1, token_logits.shape[-1]))
        can_inplace = (
            not self.training
            and not torch.is_grad_enabled()
            and not bool(getattr(torch.jit, "is_tracing")())
        )
        if self.fuse_depthwise_delta and can_inplace:
            pre_conv = self._projected_pre_conv(token_hidden, char_inputs)
            delta = self._fused_depthwise_delta(pre_conv)
        else:
            hidden = self.projected_features(token_hidden, char_inputs)
            delta = self.delta(hidden)
        if can_inplace:
            gathered_logits.add_(self.suppression(char_inputs.is_token_final)).add_(delta)
            return gathered_logits.masked_fill_(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)
        logits: torch.Tensor = gathered_logits + self.suppression(char_inputs.is_token_final) + delta
        return logits.masked_fill(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)

    def projected_features(self, token_hidden: torch.Tensor, char_inputs: CharInputs) -> torch.Tensor:
        """Shared character trunk, before task-specific residual and suppression.

        Uses the opt-in project-before-gather path. No new parameters or state-dict
        names: exported sentence models remain ordinary CharacterResolutionHeads.
        """
        return self._depthwise(self._projected_pre_conv(token_hidden, char_inputs))
