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
        self.rows = []
        self.covered = []
        self.edges = []
        self.spans = []
        self.fallback_documents = 0
        self.use_ordinals = bool(use_ordinals)
        primes = None if self.use_ordinals else np.asarray(PRIMES[:num_hashes], dtype=np.int64)
        for text, raw in zip(texts, offsets, strict=True):
            length = len(text)
            if isinstance(raw, GigaPackedDoc):
                packed = raw.packed
                if len(packed) == 0:
                    all_spans = np.empty((0, 2), dtype=np.int64)
                else:
                    ends = np.cumsum(packed[:, 1], dtype=np.int64) - 1
                    starts = ends - packed[:, 2]
                    all_spans = np.empty((len(packed), 2), dtype=np.int64)
                    all_spans[:, 0] = starts
                    all_spans[:, 1] = ends
            elif hasattr(raw, "offsets"):
                all_spans = np.asarray(raw.offsets, dtype=np.int64).reshape(-1, 2)
            else:
                all_spans = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
            self.spans.append(all_spans)
            valid = all_spans[:, 1] > all_spans[:, 0]
            spans = all_spans[valid]
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
                self.rows.append(None)
                self.covered.append(None)
                self.edges.append(None)
                self.fallback_documents += 1
                continue
            indices = np.flatnonzero(valid)
            mapping = np.zeros(length, dtype=np.int64)
            final = np.zeros(length, dtype=np.int64)
            position = np.zeros(length, dtype=np.int64)
            u32_ord = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
            mask = np.ones(length, dtype=np.int64)
            chars = np.arange(length, dtype=np.int64)
            owner = np.searchsorted(s0, chars, side="right") - 1
            safe_owner = np.maximum(owner, 0)
            covered = (owner >= 0) & (chars < s1[safe_owner])
            sel = chars[covered]
            own_cov = owner[covered]
            mapping[sel] = indices[own_cov]
            position[sel] = np.minimum(sel - s0[own_cov], max_position_in_token)
            final[s1 - 1] = 1
            if self.use_ordinals:
                hashes = u32_ord.astype(np.int32, copy=False)
            else:
                ord_i64 = u32_ord.astype(np.int64)
                hashes = ((ord_i64[:, None] + 1) * primes) % num_buckets
            self.rows.append({
                "char_to_token": mapping,
                "is_token_final": final,
                "position_in_token": position,
                "hashes": hashes,
                "mask": mask,
            })
            self.covered.append(covered)
            if bool(np.all(valid)):
                prefix_ends = s1
                suffix_starts = s0
            else:
                prefix_ends = np.maximum.accumulate(np.where(valid, all_spans[:, 1], -1))
                suffix_starts = np.minimum.accumulate(np.where(valid, all_spans[:, 0], length + 1)[::-1])[::-1]
            self.edges.append((prefix_ends, suffix_starts))

    def can_slice(self, index, token_start, token_end, char_start, char_end):
        if self.rows[index] is None:
            return False
        prefix,suffix = self.edges[index]
        # An excluded token touching the slice would change last-writer mapping
        # or final flags. Recompute that window, while reusing interior windows.
        return not ((token_start and prefix[token_start-1] > char_start)
                    or (token_end < len(suffix) and suffix[token_end] < char_end))

    def batch(self, windows):
        width = max(w["char_end"] - w["char_start"] for w in windows)
        use_ord = self.use_ordinals and all(
            w.get("reuse", self.rows[w["text_index"]] is not None) for w in windows
        )
        arrays = {key: np.zeros((len(windows), width), dtype=np.int64)
                  for key in FIELDS if key != "hashes"}
        if use_ord:
            arrays["hashes"] = np.zeros((len(windows), width), dtype=np.int32)
        else:
            arrays["hashes"] = np.broadcast_to(np.asarray(PRIMES[:8], dtype=np.int64) % 8192,
                                               (len(windows), width, 8)).copy()
        for i, window in enumerate(windows):
            index, start, end = window["text_index"], window["char_start"], window["char_end"]
            length = end - start
            source = self.rows[index]
            if not window.get("reuse", source is not None):
                fallback = build_char_inputs_vectorized([window["text"]], [window["offsets"]])
                for key in FIELDS:
                    arrays[key][i, :length] = getattr(fallback, key)[0].numpy()
            else:
                for key in FIELDS:
                    if key == "hashes" and not use_ord and source["hashes"].ndim == 1:
                        ord_slice = source["hashes"][start:end].astype(np.int64)
                        primes = np.asarray(PRIMES[:8], dtype=np.int64)
                        arrays["hashes"][i, :length] = ((ord_slice[:, None] + 1) * primes) % 8192
                    else:
                        arrays[key][i, :length] = source[key][start:end]
                arrays["char_to_token"][i, :length] = np.where(
                    self.covered[index][start:end], source["char_to_token"][start:end] - window["token_start"], 0)
        return CharInputs(**{k: torch.from_numpy(v) for k,v in arrays.items()})


_DEFAULT_DOCUMENT_FEATURES = DocumentFeatures


def extract_character_head_exact(texts, model, *, stride, max_block_size, batch_size,
                                 pad_last_batch=False, weighting="uniform", verbose=False, tokenizer,
                                 trim_padding=True):
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
    for text_index, text in enumerate(texts):
        ids = token_ids[text_index]
        n_ids = len(ids)
        if n_ids == 0:
            continue
        has_spans = hasattr(cached, "spans")
        doc_spans = cached.spans[text_index] if has_spans else None
        mapping = None if has_spans else (offsets[text_index] if offsets is not None else encoded.offset_mapping[text_index])
        ids_list = ids.tolist() if isinstance(ids, np.ndarray) else ids
        for token_start in range(0, n_ids, stride):
            token_end = token_start + block
            done = token_end >= n_ids
            if done:
                token_end = n_ids
                token_start = max(token_end - block, 0)
            if doc_spans is not None:
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
                                offsets=relative, reuse=reuse, input_ids=ids_list[token_start:token_end]))
            if done:
                break
    logits = [np.zeros((len(t), model.config.num_labels), dtype=np.float32) for t in texts]
    counts = [np.zeros(len(t), dtype=np.float32) for t in texts]
    n_batches = math.ceil(len(windows) / batch_size)
    batches = range(n_batches)
    if verbose:
        from tqdm.auto import tqdm
        batches = tqdm(batches)
    if can_use_two_stage:
        gpu_batches = []
        batch_windows = []
        for bi in batches:
            batch = windows[bi*batch_size:(bi+1)*batch_size]
            real = len(batch)
            token_width = block+2
            if trim_padding:
                token_width = max(len(window["input_ids"]) + 2 for window in batch)
            ids = np.full((real, token_width), tokenizer.pad_token_id, dtype=np.int64)
            attention = np.zeros_like(ids, dtype=np.float32)
            for row, window in enumerate(batch):
                seq = [tokenizer.cls_token_id, *window["input_ids"], tokenizer.sep_token_id]
                ids[row, :len(seq)] = seq
                attention[row, :len(seq)] = 1
            inputs = cached.batch(batch)
            inputs.char_to_token = inputs.char_to_token + 1
            if real < batch_size and pad_last_batch:
                missing = batch_size-real
                ids = np.pad(ids, ((0,missing),(0,0)), constant_values=tokenizer.pad_token_id)
                attention = np.pad(attention, ((0,missing),(0,0)))
                for key in FIELDS:
                    value = getattr(inputs, key)
                    pad_val = -1 if (key == "hashes" and value.ndim == 2) else 0
                    setattr(inputs, key, torch.nn.functional.pad(value, (0,)*(2*(value.ndim-1))+(0,missing), value=pad_val))
            token_hidden, token_logits = model.encode_tokens(
                attention_mask=attention,
                input_ids=ids,
            )
            gpu_batches.append(model.run_character_head(token_hidden, token_logits, inputs)[:real])
            batch_windows.append(batch)
        for batch, logits_dev in zip(batch_windows, gpu_batches):
            values = logits_dev.cpu().numpy()
            for row, window in enumerate(batch):
                ti, start, end = window["text_index"], window["char_start"], window["char_end"]
                length = end-start
                weights = _window_weights(length, weighting)
                logits[ti][start:end] += values[row, :length] * weights[:, None]
                counts[ti][start:end] += weights
    else:
        for bi in batches:
            batch = windows[bi*batch_size:(bi+1)*batch_size]
            real = len(batch)
            token_width = block+2
            if trim_padding:
                token_width = max(len(window["input_ids"]) + 2 for window in batch)
            ids = np.full((real, token_width), tokenizer.pad_token_id, dtype=np.int64)
            attention = np.zeros_like(ids, dtype=np.float32)
            for row, window in enumerate(batch):
                seq = [tokenizer.cls_token_id, *window["input_ids"], tokenizer.sep_token_id]
                ids[row, :len(seq)] = seq
                attention[row, :len(seq)] = 1
            inputs = cached.batch(batch)
            inputs.char_to_token = inputs.char_to_token + 1
            if real < batch_size and pad_last_batch:
                missing = batch_size-real
                ids = np.pad(ids, ((0,missing),(0,0)), constant_values=tokenizer.pad_token_id)
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
                weights = _window_weights(length, weighting)
                logits[ti][start:end] += values[row, :length] * weights[:, None]
                counts[ti][start:end] += weights
    for i, text in enumerate(texts):
        if text:
            covered = counts[i] > 0
            logits[i][covered] /= counts[i][covered, None]
            logits[i][~covered] = -12.0
    return [a.astype(np.float16) for a in logits], None, tokenizer, encoded
