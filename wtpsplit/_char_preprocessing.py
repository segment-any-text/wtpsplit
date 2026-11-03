"""Exact, opt-in request-local character preprocessing.

Preserve all legacy input tensors, including uncovered characters, padding hashes,
overlapping-token last-writer semantics, and token-final flags from earlier spans.
Nonmonotonic/malformed offsets fall back to the original implementation.
"""
import math

import numpy as np
import torch

from wtpsplit.char_head import CharInputs, build_char_inputs as reference_builder
from wtpsplit.extract import _window_weights, max_content_block_size
from wtpsplit.utils import PRIMES


FIELDS = ("char_to_token", "is_token_final", "position_in_token", "hashes", "mask")


def build_char_inputs_vectorized(texts, offset_mappings, *, num_hashes=8,
                                 num_buckets=8192, max_position_in_token=15, device=None):
    batch = len(texts)
    width = max(map(len, texts), default=0)
    mapping = np.zeros((batch, width), dtype=np.int64)
    final = np.zeros_like(mapping)
    position = np.zeros_like(mapping)
    ordinals = np.zeros_like(mapping)
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
        ordinals[row, :length] = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
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
    primes = np.asarray(PRIMES[:num_hashes], dtype=np.int64)
    hashes = ((ordinals[..., None] + 1) * primes) % num_buckets
    return CharInputs(**dict(zip(FIELDS, [torch.as_tensor(a, device=device)
                         for a in (mapping, final, position, hashes, mask)], strict=True)))


class DocumentFeatures:
    """Reuse only unambiguously sliceable offsets; fallback preserves the rest."""
    def __init__(self, texts, offsets):
        self.rows = []
        self.covered = []
        self.edges = []
        self.fallback_documents = 0
        for text, raw in zip(texts, offsets, strict=True):
            all_spans = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
            valid = all_spans[:, 1] > all_spans[:, 0]
            spans = all_spans[valid]
            safe = (len(spans) > 0 and np.all(spans[:, 0] >= 0)
                    and np.all(spans[:, 1] <= len(text))
                    and np.all(spans[:, 1] > spans[:, 0])
                    and np.all(np.diff(spans[:, 0]) >= 0)
                    and np.all(np.diff(spans[:, 1]) >= 0))
            if not safe:
                self.rows.append(None)
                self.covered.append(None)
                self.edges.append(None)
                self.fallback_documents += 1
                continue
            row = build_char_inputs_vectorized([text], [raw])
            arrays = {key: getattr(row, key)[0].numpy() for key in FIELDS}
            chars = np.arange(len(text))
            owner = np.searchsorted(spans[:, 0], chars, side="right") - 1
            covered = (owner >= 0) & (chars < spans[np.maximum(owner, 0), 1])
            self.rows.append(arrays)
            self.covered.append(covered)
            prefix_ends = np.maximum.accumulate(np.where(valid, all_spans[:, 1], -1))
            suffix_starts = np.minimum.accumulate(np.where(valid, all_spans[:, 0], len(text)+1)[::-1])[::-1]
            self.edges.append((prefix_ends,suffix_starts))

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
        arrays = {key: np.zeros((len(windows), width), dtype=np.int64)
                  for key in FIELDS if key != "hashes"}
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
                    arrays[key][i, :length] = source[key][start:end]
                arrays["char_to_token"][i, :length] = np.where(
                    self.covered[index][start:end], source["char_to_token"][start:end] - window["token_start"], 0)
        return CharInputs(**{k: torch.from_numpy(v) for k,v in arrays.items()})


def extract_character_head_exact(texts, model, *, stride, max_block_size, batch_size,
                                 pad_last_batch=False, weighting="uniform", verbose=False, tokenizer,
                                 trim_padding=True):
    """Reference windowing/model/stitching, with request-local document features.

Preserves reference padding, batch layout and fusion unless trim_padding is enabled.
No document features or predictions are retained between requests.
"""
    if stride < 1:
        raise ValueError(f"`stride` must be at least 1, got {stride}.")
    encoded = tokenizer(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    token_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    block = min(max_block_size, max_content_block_size(model.config))
    if block < 1:
        raise ValueError("No room for content tokens")
    cached = DocumentFeatures(texts, offsets)
    windows = []
    for text_index, (text, ids, mapping) in enumerate(zip(texts, token_ids, offsets)):
        for token_start in range(0, len(ids), stride):
            token_end = token_start + block
            done = token_end >= len(ids)
            if done:
                token_end = len(ids)
                token_start = max(token_end - block, 0)
            char_start = 0 if token_start == 0 else mapping[token_start][0]
            char_end = len(text) if token_end == len(ids) else mapping[token_end - 1][1]
            reuse = cached.can_slice(text_index, token_start, token_end, char_start, char_end)
            # Only fallback documents need repeated Python offset conversion.
            relative = ([(max(s-char_start,0), min(e-char_start,char_end-char_start))
                         for s,e in mapping[token_start:token_end]] if not reuse else None)
            windows.append(dict(text_index=text_index, token_start=token_start,
                                char_start=char_start, char_end=char_end,
                                text=text[char_start:char_end] if relative is not None else None,
                                offsets=relative, reuse=reuse, input_ids=ids[token_start:token_end]))
            if done:
                break
    logits = [np.zeros((len(t), model.config.num_labels), dtype=np.float32) for t in texts]
    counts = [np.zeros(len(t), dtype=np.float32) for t in texts]
    batches = range(math.ceil(len(windows) / batch_size))
    if verbose:
        from tqdm.auto import tqdm
        batches = tqdm(batches)
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
                setattr(inputs, key, torch.nn.functional.pad(value, (0,)*(2*(value.ndim-1))+(0,missing)))
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
