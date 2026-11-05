"""Exact, opt-in request-local character preprocessing.

Preserve all legacy input tensors, including uncovered characters, padding hashes,
overlapping-token last-writer semantics, and token-final flags from earlier spans.
Nonmonotonic/malformed offsets fall back to the original implementation.
"""
import math

import numpy as np
import torch

from wtpsplit._host_kernels import GigaPackedDoc, can_use_rust_encode_batch, fast_tokenize_subword
from wtpsplit.char_head import CharInputs, build_char_inputs as reference_builder
from wtpsplit.extract import _window_weights, max_content_block_size
from wtpsplit.utils import PRIMES


FIELDS = ("char_to_token", "is_token_final", "position_in_token", "hashes", "mask")
MAX_DEFERRED_D2H_BYTES = 64 * 1024 * 1024


def build_char_inputs_vectorized(texts, offset_mappings, *, num_hashes=8,
                                 num_buckets=8192, max_position_in_token=15, device=None,
                                 return_ordinals=False):
    batch = len(texts)
    width = max(map(len, texts), default=0)
    mapping = np.zeros((batch, width), dtype=np.int64)
    final = np.zeros_like(mapping)
    position = np.zeros_like(mapping)
    ordinals = np.zeros((batch, width), dtype=np.int32 if return_ordinals else np.int64)
    mask = np.zeros_like(mapping)
    for row, (text, raw) in enumerate(zip(texts, offset_mappings)):
        length = len(text)
        offsets = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
        # Match the original's clipping, empty-span treatment, and errors via
        # fallback for unusual inputs; don't redefine their semantics here.
        valid = offsets[:, 1] > offsets[:, 0]
        indices = np.flatnonzero(valid)
        spans = offsets[valid].copy()
        if len(spans):
            spans[:, 1] = np.minimum(spans[:, 1], length)
        safe = not len(spans) or (
            np.all(spans[:, 0] >= 0) and np.all(spans[:, 0] < spans[:, 1])
            and np.all(np.diff(spans[:, 0]) >= 0) and np.all(np.diff(spans[:, 1]) >= 0))
        if not safe:
            return reference_builder(texts, offset_mappings, num_hashes=num_hashes,
                                     num_buckets=num_buckets, max_position_in_token=max_position_in_token,
                                     device=device)
        mask[row, :length] = 1
        u32_ord = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        ordinals[row, :length] = u32_ord.astype(ordinals.dtype, copy=False)
        if not len(spans):
            continue
        chars = np.arange(length)
        # With monotone ends, the last start <= c is also the last covering
        # token, if any. This retains duplicate/overlapping offset semantics.
        owner = np.searchsorted(spans[:, 0], chars, side="right") - 1
        safe_owner = np.maximum(owner, 0)
        covered = (owner >= 0) & (chars < spans[safe_owner, 1])
        selected = chars[covered]
        mapping[row, selected] = indices[owner[covered]]
        position[row, selected] = np.minimum(
            selected - spans[owner[covered], 0], max_position_in_token)
        # The reference never clears an earlier token's final flag when a
        # later overlapping token overwrites its mapping.
        final[row, spans[:, 1] - 1] = 1
    if return_ordinals:
        hashes = ordinals
    else:
        primes = np.asarray(PRIMES[:num_hashes], dtype=np.int64)
        hashes = ((ordinals[..., None] + 1) * primes) % num_buckets
    return CharInputs(**dict(zip(FIELDS, [torch.as_tensor(a, device=device)
                         for a in (mapping, final, position, hashes, mask)], strict=True)))


class DocumentFeatures:
    """Reuse only unambiguously sliceable offsets; fallback preserves the rest."""
    def __init__(self, texts, offsets, *, use_ordinals=False, max_position_in_token=15,
                 num_hashes=8, num_buckets=8192):
        texts_list = list(texts)
        offsets_list = list(offsets)
        if len(texts_list) != len(offsets_list):
            raise ValueError("texts and offsets must have the same length")
        n_docs = len(texts_list)
        self.rows = [None] * n_docs
        self.covered = [None] * n_docs
        self.edges = [None] * n_docs
        self.span_pairs = [None] * n_docs
        self._spans = None
        self.fallback_documents = 0
        self.use_ordinals = bool(use_ordinals)
        if n_docs == 0:
            return

        primes = None if self.use_ordinals else np.asarray(PRIMES[:num_hashes], dtype=np.int64)
        giga_seen = {}
        dup_targets = []
        unique_texts = []
        unique_records = []

        for d, (text, raw) in enumerate(zip(texts_list, offsets_list)):
            if isinstance(raw, GigaPackedDoc):
                prev_d = giga_seen.get(text)
                if prev_d is not None:
                    prev_raw = offsets_list[prev_d]
                    if raw is prev_raw or np.array_equal(raw.packed, prev_raw.packed):
                        dup_targets.append((d, prev_d))
                        continue
                else:
                    giga_seen[text] = d
            unique_texts.append(text)
            unique_records.append((d, text, raw))

        full_u32 = np.frombuffer("".join(unique_texts).encode("utf-32-le"), dtype=np.uint32)
        if self.use_ordinals:
            full_hashes = full_u32.astype(np.int32, copy=False)
        else:
            full_hashes = ((full_u32.astype(np.int64)[:, None] + 1) * primes) % num_buckets

        safe_records = []
        s0_shifted = []
        s1_shifted = []
        char_cursor = 0
        safe_char_cursor = 0
        safe_tok_cursor = 0
        any_invalid_spans = False
        _one_i64 = np.int64(1)

        for d, text, raw in unique_records:
            length = len(text)
            c_start = char_cursor
            char_cursor += length

            if isinstance(raw, GigaPackedDoc):
                packed = raw.packed
                n_tok = len(packed)
                if n_tok == 0:
                    empty_i64 = np.empty(0, dtype=np.int64)
                    self.span_pairs[d] = (empty_i64, empty_i64)
                    self.fallback_documents += 1
                    continue
                ends = np.cumsum(packed[:, 1], dtype=np.int64) - 1
                starts = ends - packed[:, 2]
                self.span_pairs[d] = (starts, ends)
                if not raw.has_zero_delta and not bool(np.any(packed[:, 2] <= 0)):
                    if int(starts[0]) < 0 or int(ends[-1]) > length or (
                        n_tok > 1 and not bool(np.all(ends[1:] >= ends[:-1]))
                    ):
                        self.fallback_documents += 1
                        continue
                    s0 = starts
                    s1 = ends
                    all_valid = True
                    val_idx = None
                    self.edges[d] = (ends, starts)
                else:
                    valid = ends > starts
                    all_valid = bool(np.all(valid))
                    s0 = starts if all_valid else starts[valid]
                    s1 = ends if all_valid else ends[valid]
                    safe = (
                        len(s0) > 0
                        and int(s0[0]) >= 0
                        and int(s1[-1]) <= length
                        and bool(np.all(s0[1:] >= s0[:-1]))
                        and bool(np.all(s1[1:] >= s1[:-1]))
                    )
                    if not safe:
                        self.fallback_documents += 1
                        continue
                    if all_valid:
                        self.edges[d] = (s1, s0)
                        val_idx = None
                    else:
                        any_invalid_spans = True
                        prefix_ends = np.maximum.accumulate(np.where(valid, ends, -1))
                        suffix_starts = np.minimum.accumulate(
                            np.where(valid, starts, length + 1)[::-1]
                        )[::-1]
                        self.edges[d] = (prefix_ends, suffix_starts)
                        val_idx = np.flatnonzero(valid)
            else:
                if hasattr(raw, "offsets"):
                    all_spans = np.asarray(raw.offsets, dtype=np.int64).reshape(-1, 2)
                else:
                    all_spans = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
                self.span_pairs[d] = (all_spans[:, 0], all_spans[:, 1])
                valid = all_spans[:, 1] > all_spans[:, 0]
                all_valid = bool(np.all(valid))
                spans = all_spans if all_valid else all_spans[valid]
                s0 = spans[:, 0]
                s1 = spans[:, 1]
                safe = (
                    len(spans) > 0
                    and int(s0[0]) >= 0
                    and int(s1[-1]) <= length
                    and bool(np.all(s0[1:] >= s0[:-1]))
                    and bool(np.all(s1[1:] >= s1[:-1]))
                )
                if not safe:
                    self.fallback_documents += 1
                    continue
                if all_valid:
                    self.edges[d] = (s1, s0)
                    val_idx = None
                else:
                    any_invalid_spans = True
                    prefix_ends = np.maximum.accumulate(np.where(valid, all_spans[:, 1], -1))
                    suffix_starts = np.minimum.accumulate(
                        np.where(valid, all_spans[:, 0], length + 1)[::-1]
                    )[::-1]
                    self.edges[d] = (prefix_ends, suffix_starts)
                    val_idx = np.flatnonzero(valid)

            n_tok = len(s0)
            s0_shifted.append(s0 + safe_char_cursor if safe_char_cursor else s0)
            s1_shifted.append(s1 + safe_char_cursor if safe_char_cursor else s1)
            safe_records.append(
                (d, length, c_start, safe_char_cursor, safe_char_cursor + length, safe_tok_cursor, n_tok, val_idx)
            )
            safe_char_cursor += length
            safe_tok_cursor += n_tok

        if safe_records:
            g_s0 = s0_shifted[0] if len(s0_shifted) == 1 else np.concatenate(s0_shifted)
            g_s1 = s1_shifted[0] if len(s1_shifted) == 1 else np.concatenate(s1_shifted)
            if len(g_s0) > 1 and not bool(np.all(g_s0[1:] >= g_s0[:-1])):
                filtered_records = []
                s0_shifted_f = []
                s1_shifted_f = []
                sc_cur = 0
                st_cur = 0
                for rec, s0_u, s1_u in zip(safe_records, s0_shifted, s1_shifted):
                    d, length, c_start, old_sc, _, _, n_tok, val_idx = rec
                    s0_raw = s0_u - old_sc if old_sc else s0_u
                    s1_raw = s1_u - old_sc if old_sc else s1_u
                    if n_tok > 1 and not bool(np.all(s0_raw[1:] >= s0_raw[:-1])):
                        self.edges[d] = None
                        self.fallback_documents += 1
                        continue
                    s0_shifted_f.append(s0_raw + sc_cur if sc_cur else s0_raw)
                    s1_shifted_f.append(s1_raw + sc_cur if sc_cur else s1_raw)
                    filtered_records.append(
                        (d, length, c_start, sc_cur, sc_cur + length, st_cur, n_tok, val_idx)
                    )
                    sc_cur += length
                    st_cur += n_tok
                safe_records = filtered_records
                safe_char_cursor = sc_cur
                if safe_records:
                    g_s0 = s0_shifted_f[0] if len(s0_shifted_f) == 1 else np.concatenate(s0_shifted_f)
                    g_s1 = s1_shifted_f[0] if len(s1_shifted_f) == 1 else np.concatenate(s1_shifted_f)

        if safe_records:
            n_total_tok = len(g_s0)
            g_chars = np.arange(safe_char_cursor, dtype=np.int64)
            # Fast O(N_chars) linear-time gap/span repeat when spans are disjoint (no overlapping tokens)
            if not any_invalid_spans and n_total_tok > 0:
                gaps = np.empty(n_total_tok + 1, dtype=np.int64)
                gaps[0] = g_s0[0]
                if n_total_tok > 1:
                    gaps[1:-1] = g_s0[1:] - g_s1[:-1]
                gaps[-1] = safe_char_cursor - g_s1[-1]
                is_disjoint = bool(np.all(gaps >= 0))
            else:
                is_disjoint = False

            if is_disjoint:
                span_lens = g_s1 - g_s0
                run_lens = np.empty(2 * n_total_tok + 1, dtype=np.int64)
                run_lens[0::2] = gaps
                run_lens[1::2] = span_lens
                run_id = np.repeat(np.arange(2 * n_total_tok + 1, dtype=np.int64), run_lens)
                g_covered = (run_id & 1) == 1
                safe_owner = np.minimum(run_id >> 1, n_total_tok - 1)
                g_owner = safe_owner
                g_pos = np.where(g_covered, np.minimum(g_chars - g_s0[safe_owner], max_position_in_token), 0)
            else:
                g_owner = np.searchsorted(g_s0, g_chars, side="right") - 1
                safe_owner = np.maximum(g_owner, 0)
                g_covered = (g_owner >= 0) & (g_chars < g_s1[safe_owner])
                g_pos = np.where(g_covered, np.minimum(g_chars - g_s0[safe_owner], max_position_in_token), 0)

            g_final = np.zeros(safe_char_cursor, dtype=np.int64)
            g_final[g_s1 - 1] = 1

            if not any_invalid_spans:
                if len(safe_records) == 1:
                    g_mapping = np.where(g_covered, g_owner, 0)
                else:
                    tok_bases = np.repeat(
                        np.fromiter((r[5] for r in safe_records), dtype=np.int64, count=len(safe_records)),
                        np.fromiter((r[1] for r in safe_records), dtype=np.int64, count=len(safe_records)),
                    )
                    g_mapping = np.where(g_covered, g_owner - tok_bases, 0)
            else:
                g_mapping = np.zeros(safe_char_cursor, dtype=np.int64)

            for d, length, c_start, sc_start, sc_end, st_start, _n_tok, val_idx in safe_records:
                cov_slice = g_covered[sc_start:sc_end]
                if any_invalid_spans:
                    local_owner = g_owner[sc_start:sc_end][cov_slice] - st_start
                    g_mapping[sc_start:sc_end][cov_slice] = (
                        local_owner if val_idx is None else val_idx[local_owner]
                    )
                self.rows[d] = {
                    "char_to_token": g_mapping[sc_start:sc_end],
                    "is_token_final": g_final[sc_start:sc_end],
                    "position_in_token": g_pos[sc_start:sc_end],
                    "hashes": full_hashes[c_start : c_start + length],
                    "mask": np.broadcast_to(_one_i64, (length,)),
                }
                self.covered[d] = cov_slice

        for d, prev_d in dup_targets:
            self.span_pairs[d] = self.span_pairs[prev_d]
            self.rows[d] = self.rows[prev_d]
            self.covered[d] = self.covered[prev_d]
            self.edges[d] = self.edges[prev_d]
            if self.rows[prev_d] is None:
                self.fallback_documents += 1

    @property
    def spans(self):
        if self._spans is None:
            self._spans = [
                np.column_stack(pair) if pair is not None else np.empty((0, 2), dtype=np.int64)
                for pair in self.span_pairs
            ]
        return self._spans

    def can_slice(self, index, token_start, token_end, char_start, char_end):
        if self.rows[index] is None:
            return False
        prefix,suffix = self.edges[index]
        # An excluded token touching the slice would change last-writer mapping
        # or final flags. Recompute that window, while reusing interior windows.
        return not ((token_start and prefix[token_start-1] > char_start)
                    or (token_end < len(suffix) and suffix[token_end] < char_end))

    def batch(self, windows, *, token_offset=0, skip_hashes=False):
        width = max(w["char_end"] - w["char_start"] for w in windows)
        use_ord = self.use_ordinals and all(
            w.get("reuse", self.rows[w["text_index"]] is not None) for w in windows
        )
        n_win = len(windows)
        t_off = int(token_offset)
        if skip_hashes and all(w.get("reuse", self.rows[w["text_index"]] is not None) for w in windows):
            c2t_arr = np.empty((n_win, width), dtype=np.int64)
            fin_arr = np.empty((n_win, width), dtype=np.int64)
            msk_arr = np.empty((n_win, width), dtype=np.int64)
            for i, window in enumerate(windows):
                index, start, end = window["text_index"], window["char_start"], window["char_end"]
                length = end - start
                if length < width:
                    c2t_arr[i, length:] = t_off
                    fin_arr[i, length:] = 0
                    msk_arr[i, length:] = 0
                source = self.rows[index]
                fin_arr[i, :length] = source["is_token_final"][start:end]
                msk_arr[i, :length] = 1
                shift = t_off - window["token_start"]
                c2t_arr[i, :length] = np.where(
                    self.covered[index][start:end], source["char_to_token"][start:end] + shift, t_off
                )
            dummy_pos = torch.zeros((0, 0), dtype=torch.int64)
            dummy_hash = torch.zeros((0, 0), dtype=torch.int32 if use_ord else torch.int64)
            return CharInputs(
                char_to_token=torch.from_numpy(c2t_arr),
                is_token_final=torch.from_numpy(fin_arr),
                position_in_token=dummy_pos,
                hashes=dummy_hash,
                mask=torch.from_numpy(msk_arr),
            )

        arrays = {
            "char_to_token": np.empty((n_win, width), dtype=np.int64),
            "is_token_final": np.empty((n_win, width), dtype=np.int64),
            "position_in_token": np.empty((n_win, width), dtype=np.int64),
            "mask": np.empty((n_win, width), dtype=np.int64),
        }
        pad_hash_row = None
        if use_ord:
            arrays["hashes"] = np.empty((n_win, width), dtype=np.int32)
        else:
            pad_hash_row = np.asarray(PRIMES[:8], dtype=np.int64) % 8192
            arrays["hashes"] = np.empty((n_win, width, 8), dtype=np.int64)
        for i, window in enumerate(windows):
            index, start, end = window["text_index"], window["char_start"], window["char_end"]
            length = end - start
            if length < width:
                arrays["char_to_token"][i, length:] = t_off
                arrays["is_token_final"][i, length:] = 0
                arrays["position_in_token"][i, length:] = 0
                arrays["mask"][i, length:] = 0
                if use_ord:
                    arrays["hashes"][i, length:] = 0
                else:
                    arrays["hashes"][i, length:] = pad_hash_row
            source = self.rows[index]
            if not window.get("reuse", source is not None):
                fallback = build_char_inputs_vectorized([window["text"]], [window["offsets"]])
                for key in FIELDS:
                    val = getattr(fallback, key)[0].numpy()
                    arrays[key][i, :length] = val + t_off if (key == "char_to_token" and t_off != 0) else val
            else:
                arrays["is_token_final"][i, :length] = source["is_token_final"][start:end]
                arrays["position_in_token"][i, :length] = source["position_in_token"][start:end]
                if not use_ord and source["hashes"].ndim == 1:
                    ord_slice = source["hashes"][start:end].astype(np.int64)
                    primes = np.asarray(PRIMES[:8], dtype=np.int64)
                    arrays["hashes"][i, :length] = ((ord_slice[:, None] + 1) * primes) % 8192
                else:
                    arrays["hashes"][i, :length] = source["hashes"][start:end]
                arrays["mask"][i, :length] = 1
                shift = t_off - window["token_start"]
                arrays["char_to_token"][i, :length] = np.where(
                    self.covered[index][start:end], source["char_to_token"][start:end] + shift, t_off)
        return CharInputs(**{k: torch.from_numpy(v) for k,v in arrays.items()})


_DEFAULT_DOCUMENT_FEATURES = DocumentFeatures


def extract_character_head_exact(texts, model, *, stride, max_block_size, batch_size,
                                 pad_last_batch=False, weighting="uniform", verbose=False, tokenizer,
                                 trim_padding=True, trim_subword_padding=False, pack_sequences=False):
    """Reference windowing/model/stitching, with request-local document features.

Preserves reference padding, batch layout and fusion unless trim_padding is enabled.
No document features or predictions are retained between requests.
"""
    if stride < 1:
        raise ValueError(f"`stride` must be at least 1, got {stride}.")
    block = min(max_block_size, max_content_block_size(model.config))
    if block < 1:
        raise ValueError("No room for content tokens")
    can_use_two_stage = (
        DocumentFeatures is _DEFAULT_DOCUMENT_FEATURES
        and type(model).__name__ == "PyTorchWrapper"
        and hasattr(model, "encode_tokens")
        and hasattr(model, "run_character_head")
        and getattr(getattr(model, "model", None), "character_head", None) is not None
        and not getattr(getattr(model, "model", None), "_forward_pre_hooks", None)
        and not getattr(getattr(model, "model", None), "_forward_hooks", None)
        and getattr(model, "pinned_inputs", None) is None
        and "__call__" not in getattr(model, "__dict__", {})
    )
    is_ort_wrapper = (
        DocumentFeatures is _DEFAULT_DOCUMENT_FEATURES
        and type(model).__name__ == "SaTORTWrapper"
        and "__call__" not in getattr(model, "__dict__", {})
    )
    use_ordinals = can_use_two_stage or (
        is_ort_wrapper and getattr(model, "input_ranks", {}).get("char_hashes") == 2
    )
    if (can_use_two_stage or is_ort_wrapper) and len(texts) > 0 and can_use_rust_encode_batch(tokenizer):
        enc_batch = fast_tokenize_subword(tokenizer, texts)
        encoded = enc_batch
        token_ids = enc_batch.token_arrays
        doc_items = [enc_batch.offsets_or_encoding(i) for i in range(len(texts))]
        cached = DocumentFeatures(texts, doc_items, use_ordinals=use_ordinals)
        offsets = None
    else:
        encoded = tokenizer(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
        token_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        cached = (
            DocumentFeatures(texts, offsets, use_ordinals=use_ordinals)
            if (can_use_two_stage or is_ort_wrapper)
            else DocumentFeatures(texts, offsets)
        )
    windows = []
    doc_win_counts = [0] * len(texts)
    has_span_pairs = hasattr(cached, "span_pairs")
    has_spans = (not has_span_pairs) and hasattr(cached, "spans")
    for text_index, text in enumerate(texts):
        ids = token_ids[text_index]
        n_ids = len(ids)
        if n_ids == 0:
            continue
        doc_span_pair = cached.span_pairs[text_index] if has_span_pairs else None
        doc_spans = (cached.spans[text_index] if has_spans else None) if doc_span_pair is None else None
        mapping = None if (doc_span_pair is not None or doc_spans is not None) else (
            offsets[text_index] if offsets is not None else encoded.offset_mapping[text_index]
        )
        for token_start in range(0, n_ids, stride):
            token_end = token_start + block
            done = token_end >= n_ids
            if done:
                token_end = n_ids
                token_start = max(token_end - block, 0)
            if doc_span_pair is not None:
                char_start = 0 if token_start == 0 else int(doc_span_pair[0][token_start])
                char_end = len(text) if token_end == n_ids else int(doc_span_pair[1][token_end - 1])
            elif doc_spans is not None:
                char_start = 0 if token_start == 0 else int(doc_spans[token_start, 0])
                char_end = len(text) if token_end == n_ids else int(doc_spans[token_end - 1, 1])
            else:
                char_start = 0 if token_start == 0 else mapping[token_start][0]
                char_end = len(text) if token_end == n_ids else mapping[token_end - 1][1]
            reuse = cached.can_slice(text_index, token_start, token_end, char_start, char_end)
            if not reuse and mapping is None:
                mapping = offsets[text_index] if offsets is not None else encoded.offset_mapping[text_index]
            # Only fallback documents need repeated Python offset conversion.
            relative = ([(max(s-char_start,0), min(e-char_start,char_end-char_start))
                         for s,e in mapping[token_start:token_end]] if not reuse else None)
            windows.append(dict(text_index=text_index, token_start=token_start,
                                char_start=char_start, char_end=char_end,
                                text=text[char_start:char_end] if relative is not None else None,
                                offsets=relative, reuse=reuse, input_ids=ids[token_start:token_end]))
            doc_win_counts[text_index] += 1
            if done:
                break
    if trim_subword_padding and len(windows) > batch_size:
        win_tok_lens = [len(w["input_ids"]) for w in windows]
        if any(win_tok_lens[i] > win_tok_lens[i + 1] for i in range(len(win_tok_lens) - 1)):
            windows.sort(key=lambda w: len(w["input_ids"]))

    num_labels = model.config.num_labels
    is_uniform = weighting == "uniform"
    _zero_f32 = np.float32(0.0)
    out_f16 = [None] * len(texts)
    logits = [
        None if (is_uniform and doc_win_counts[i] <= 1) else np.zeros((len(t), num_labels), dtype=np.float32)
        for i, t in enumerate(texts)
    ]
    counts = [
        None if (is_uniform or doc_win_counts[i] == 0) else np.zeros(len(t), dtype=np.float32)
        for i, t in enumerate(texts)
    ]
    diffs = [
        np.zeros(len(t) + 1, dtype=np.int32) if (is_uniform and doc_win_counts[i] > 1) else None
        for i, t in enumerate(texts)
    ]
    n_batches = math.ceil(len(windows) / batch_size)
    batches = range(n_batches)
    if verbose:
        from tqdm.auto import tqdm
        batches = tqdm(batches)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    is_default_df = type(cached) is _DEFAULT_DOCUMENT_FEATURES
    can_pack_char_varlen = (
        pack_sequences
        and can_use_two_stage
        and not pad_last_batch
        and hasattr(model, "encode_tokens_varlen_batch")
        and hasattr(getattr(model, "model", None), "encode_tokens_varlen")
    )
    if can_use_two_stage:
        gpu_batches = []
        batch_windows = []
        accum_d2h_bytes = 0

        def _flush_char_gpu_batches():
            nonlocal accum_d2h_bytes
            for b_win, logits_dev in zip(batch_windows, gpu_batches):
                if is_uniform and all(doc_win_counts[w["text_index"]] == 1 for w in b_win):
                    values_f16 = (logits_dev + 0.0).to(torch.float16).cpu().numpy()
                    for row, window in enumerate(b_win):
                        ti, start, end = window["text_index"], window["char_start"], window["char_end"]
                        length = end - start
                        doc_len = len(texts[ti])
                        if length == doc_len:
                            out_f16[ti] = values_f16[row, :length]
                        else:
                            res = np.full((doc_len, num_labels), -12.0, dtype=np.float16)
                            res[start:end] = values_f16[row, :length]
                            out_f16[ti] = res
                    continue

                values = logits_dev.cpu().numpy()
                for row, window in enumerate(b_win):
                    ti, start, end = window["text_index"], window["char_start"], window["char_end"]
                    length = end-start
                    if is_uniform:
                        if doc_win_counts[ti] == 1:
                            doc_len = len(texts[ti])
                            if length == doc_len:
                                out_f16[ti] = (values[row, :length] + _zero_f32).astype(np.float16)
                            else:
                                res = np.full((doc_len, num_labels), -12.0, dtype=np.float16)
                                res[start:end] = (values[row, :length] + _zero_f32).astype(np.float16)
                                out_f16[ti] = res
                        else:
                            logits[ti][start:end] += values[row, :length]
                            diffs[ti][start] += 1
                            diffs[ti][end] -= 1
                    else:
                        weights = _window_weights(length, weighting)
                        logits[ti][start:end] += values[row, :length] * weights[:, None]
                        counts[ti][start:end] += weights
            gpu_batches.clear()
            batch_windows.clear()
            accum_d2h_bytes = 0

        for bi in batches:
            batch = windows[bi*batch_size:(bi+1)*batch_size]
            real = len(batch)
            b_wl = np.fromiter((len(w["input_ids"]) for w in batch), dtype=np.int64, count=real)
            max_wl = int(b_wl.max()) if real > 0 else 0
            min_wl = int(b_wl.min()) if real > 0 else 0
            token_width = (max_wl + 2) if trim_padding else (block + 2)
            ids = np.full((real, token_width), pad_id, dtype=np.int64)
            attention = np.zeros_like(ids, dtype=np.float32)
            for row, window in enumerate(batch):
                win_ids = window["input_ids"]
                n_win = len(win_ids)
                ids[row, 0] = cls_id
                ids[row, 1 : n_win + 1] = win_ids
                ids[row, n_win + 1] = sep_id
                attention[row, : n_win + 2] = 1
            if real < batch_size and pad_last_batch:
                missing = batch_size-real
                ids = np.pad(ids, ((0,missing),(0,0)), constant_values=pad_id)
                attention = np.pad(attention, ((0,missing),(0,0)))
            if can_pack_char_varlen and min_wl < max_wl:
                token_hidden, token_logits = model.encode_tokens_varlen_batch(ids, b_wl)
            else:
                token_hidden, token_logits = model.encode_tokens(
                    attention_mask=attention,
                    input_ids=ids,
                )
            char_head_obj = getattr(getattr(model, "model", None), "character_head", None)
            eff_gate = getattr(char_head_obj, "gate_logit_threshold", None) if char_head_obj is not None else None
            skip_hashes = (
                is_default_df
                and eff_gate is not None
                and not getattr(char_head_obj, "training", False)
                and not (real < batch_size and pad_last_batch)
                and not bool((token_logits.max(dim=-1).values > eff_gate).any())
            )
            if is_default_df:
                inputs = cached.batch(batch, token_offset=1, skip_hashes=skip_hashes)
            else:
                inputs = cached.batch(batch)
                inputs.char_to_token = inputs.char_to_token + 1
            if real < batch_size and pad_last_batch:
                missing = batch_size-real
                for key in FIELDS:
                    value = getattr(inputs, key)
                    pad_val = -1 if (key == "hashes" and value.ndim == 2) else 0
                    setattr(inputs, key, torch.nn.functional.pad(value, (0,)*(2*(value.ndim-1))+(0,missing), value=pad_val))
            char_logits_t = model.run_character_head(token_hidden, token_logits, inputs)[:real]
            gpu_batches.append(char_logits_t)
            batch_windows.append(batch)
            accum_d2h_bytes += int(char_logits_t.numel() * char_logits_t.element_size())
            del char_logits_t, token_hidden, token_logits
            if accum_d2h_bytes >= MAX_DEFERRED_D2H_BYTES:
                _flush_char_gpu_batches()
        _flush_char_gpu_batches()
    else:
        for bi in batches:
            batch = windows[bi*batch_size:(bi+1)*batch_size]
            real = len(batch)
            token_width = block+2
            if trim_padding:
                token_width = max(len(window["input_ids"]) + 2 for window in batch)
            ids = np.full((real, token_width), pad_id, dtype=np.int64)
            attention = np.zeros_like(ids, dtype=np.float32)
            for row, window in enumerate(batch):
                win_ids = window["input_ids"]
                n_win = len(win_ids)
                ids[row, 0] = cls_id
                ids[row, 1 : n_win + 1] = win_ids
                ids[row, n_win + 1] = sep_id
                attention[row, : n_win + 2] = 1
            if is_default_df:
                inputs = cached.batch(batch, token_offset=1)
            else:
                inputs = cached.batch(batch)
                inputs.char_to_token = inputs.char_to_token + 1
            if real < batch_size and pad_last_batch:
                missing = batch_size-real
                ids = np.pad(ids, ((0,missing),(0,0)), constant_values=pad_id)
                attention = np.pad(attention, ((0,missing),(0,0)))
                for key in FIELDS:
                    value = getattr(inputs, key)
                    pad_val = -1 if (key == "hashes" and value.ndim == 2) else 0
                    setattr(inputs, key, torch.nn.functional.pad(value, (0,)*(2*(value.ndim-1))+(0,missing), value=pad_val))
            values = model(input_ids=ids, attention_mask=attention,
                           char_to_token=inputs.char_to_token, char_is_token_final=inputs.is_token_final,
                           char_position_in_token=inputs.position_in_token,
                           char_hashes=inputs.hashes, char_mask=inputs.mask)["logits"][:real]
            for row, window in enumerate(batch):
                ti, start, end = window["text_index"], window["char_start"], window["char_end"]
                length = end-start
                if is_uniform:
                    if doc_win_counts[ti] == 1:
                        doc_len = len(texts[ti])
                        if length == doc_len:
                            out_f16[ti] = (values[row, :length] + _zero_f32).astype(np.float16)
                        else:
                            res = np.full((doc_len, num_labels), -12.0, dtype=np.float16)
                            res[start:end] = (values[row, :length] + _zero_f32).astype(np.float16)
                            out_f16[ti] = res
                    else:
                        logits[ti][start:end] += values[row, :length]
                        diffs[ti][start] += 1
                        diffs[ti][end] -= 1
                else:
                    weights = _window_weights(length, weighting)
                    logits[ti][start:end] += values[row, :length] * weights[:, None]
                    counts[ti][start:end] += weights
    for i, text in enumerate(texts):
        if out_f16[i] is not None:
            continue
        if doc_win_counts[i] == 0:
            out_f16[i] = (
                np.full((len(text), num_labels), -12.0, dtype=np.float16)
                if text
                else np.empty((0, num_labels), dtype=np.float16)
            )
        else:
            if text:
                if is_uniform:
                    cnt = np.cumsum(diffs[i][:-1], dtype=np.int32)
                    if int(cnt.min()) > 0:
                        logits[i] /= cnt.astype(np.float32)[:, None]
                    else:
                        covered = cnt > 0
                        logits[i][covered] /= cnt[covered, None].astype(np.float32)
                        logits[i][~covered] = -12.0
                else:
                    covered = counts[i] > 0
                    logits[i][covered] /= counts[i][covered, None]
                    logits[i][~covered] = -12.0
            out_f16[i] = logits[i].astype(np.float16)
    return out_f16, None, tokenizer, encoded
