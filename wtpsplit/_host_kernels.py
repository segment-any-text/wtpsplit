"""Clean, zero-dependency host-side kernels for subword windowing, stitching, projection, and splitting.

Every kernel in this module is 100% bit-identical (`tobytes() == tobytes()` on probability arrays
and exact list equality on sentence splits) to the historical reference path while eliminating:
1. Repeated Unigram Viterbi & NFKC tokenization on recurring words (`GigaPy` fused `(id, delta)`
   `int32x2` word cache via C-level `b"".join(map(cache.get, words))` + `np.frombuffer`),
2. Eager Python tuple materialization for untouched token offsets (`fast_tokenize_subword`),
3. Per-window Python list concatenations during window packing (`pack_subword_windows`, Identity 4),
4. Single-window float16 accumulation/division, `float16 += 1.0` count loops, and scalar multi-window
   slice additions (`stitch_subword_logits`, Identity 1 + `cumsum` counts + 2D length-grouped stitch),
5. Dense `(len(text), num_labels)` float64 `-inf` buffers and full-string `exp` evaluation in
   probability projection (`project_subword_proba`, Identity 2), and
6. Full-character probability array allocation during unconstrained `split()`
   (`decode_subword_split`, Identity 3 + prefix-sum `cumsum(deltas)` cut recovery).
"""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import re
from typing import Any, Literal
import weakref

import numpy as np
from numpy.typing import NDArray

from wtpsplit.utils import Constants, get_token_spans, indices_to_sentences, sigmoid

Float16Array = NDArray[np.float16]
Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
OffsetSpan = tuple[int, int]

MAX_WORD_CACHE_SIZE = 65536

# Rejects words containing any character that XLM-R's SentencePiece normalizer + WhitespaceSplit
# treats as a split boundary or control character (checked once per unique word on cache insertion).
_NON_SIMPLE_WORD_RE = re.compile(
    r"[\s\x00-\x1f\x7f-\xa0\u1680\u2000-\u200f\u2028\u2029\u202f\u205f\u2581\u3000\ufeff\ufffd]"
)


_ALLOWED_PRETOKENIZER_REPRS = frozenset(
    {
        'Sequence(pretokenizers=[WhitespaceSplit(), Metaspace(replacement="▁", prepend_scheme=always, split=True)])',
        'Metaspace(replacement="▁", prepend_scheme=always, split=True)',
        "WhitespaceSplit()",
    }
)
_ALLOWED_NORMALIZER_TYPES = frozenset(
    {"NoneType", "NFKC", "NFC", "NFD", "NFKD", "Lowercase", "Precompiled", "Sequence"}
)
_DISALLOWED_NORMALIZER_SUBSTRINGS = (
    "Replace",
    "Strip",
    "Prepend",
    "ByteLevel",
    "BertNormalizer",
    "Split",
)


@dataclass
class _TokenizerWordCache:
    """Per-tokenizer bounded word cache storing fused `(token_id, end_delta, span_len)` `int32x3` byte buffers."""

    supports_giga_cache: bool
    rust_tok_id: int
    inline_struct_bytes: bytes
    model_props_sig: tuple[Any, ...]
    pre_tok_repr: str
    norm_type_name: str
    full_config_sha256: str
    added_tokens_count: int
    special_tokens_set: frozenset[str]
    packed_bytes: dict[str, bytes | None]
    token_id_invariants: list[tuple[int, str | None]]
    has_zero_delta: bool = False


_TOKENIZER_CACHES: weakref.WeakKeyDictionary[Any, _TokenizerWordCache] = weakref.WeakKeyDictionary()


@lru_cache(maxsize=64)
def exact_float16_logit_threshold(threshold: float) -> np.float16 | None:
    """Return the smallest IEEE-754 float16 `z` such that `sigmoid(z) > threshold`, or `None` if none can pass.

    Evaluated exhaustively across all 65,536 IEEE-754 float16 bit patterns (including `-0.0`,
    `+0.0`, `-inf`, `+inf`, and `NaN`). Because `sigmoid` is monotone on `[-inf, +inf]` and
    `NaN` comparisons are false on both sides, `False if z_min is None else (z >= z_min)` is
    100% bit-equivalent to `sigmoid(z) > threshold` for every float16 value `z` and every float
    `threshold` (including `threshold < 0`, `threshold >= 1`, and `NaN`).
    """
    if math.isnan(threshold) or threshold >= 1.0:
        return None
    all_f16 = np.arange(65536, dtype=np.uint16).view(np.float16)
    non_nan = all_f16[~np.isnan(all_f16)]
    with np.errstate(over="ignore"):
        passing = non_nan[sigmoid(non_nan) > threshold]
    if passing.size == 0:
        return None
    return np.float16(passing.min())


@dataclass
class GigaPackedDoc:
    """Self-contained `(n_tok, 3)` `int32` view (`col 0 = id, col 1 = end_delta, col 2 = span_len`).

    Owns its complete token ID and offset information inside `packed` independently of
    word-cache eviction or subsequent tokenizer mutations.
    """

    packed: Int32Array
    has_zero_delta: bool = True
    is_single_space_ascii: bool = False
    _offsets: list[OffsetSpan] | None = None

    @property
    def ids_view(self) -> Int32Array:
        return self.packed[:, 0]

    @property
    def deltas_view(self) -> Int32Array:
        return self.packed[:, 1]

    @property
    def offsets(self) -> list[OffsetSpan]:
        """Materialize full `(start, end)` Python tuples directly from `packed`."""
        if self._offsets is None:
            if len(self.packed) == 0:
                self._offsets = []
            else:
                ends = np.cumsum(self.packed[:, 1]) - 1
                starts = ends - self.packed[:, 2]
                self._offsets = list(zip(starts.tolist(), ends.tolist()))
        return self._offsets


@dataclass
class EncodedSubwordBatch:
    """Token IDs and lazily materialized character offset mappings for a batch of texts."""

    token_arrays: list[list[int]] | list[Int32Array]
    _input_ids_list: list[list[int]] | None = None
    _offsets: list[list[OffsetSpan]] | None = None
    _rust_encodings: list[Any] | None = None
    _doc_items: list[Any] | None = None

    @property
    def input_ids(self) -> list[list[int]]:
        """Return token IDs as `list[list[int]]`, lazily converting NumPy views only if requested."""
        if self._input_ids_list is None:
            self._input_ids_list = [
                row.tolist() if isinstance(row, np.ndarray) else row for row in self.token_arrays
            ]
        return self._input_ids_list

    @property
    def offset_mapping(self) -> list[list[OffsetSpan]]:
        """Materialize full Python offset lists only when requested by `predict_proba` or `extract`."""
        if self._offsets is None:
            if self._doc_items is not None:
                self._offsets = [
                    item.offsets if hasattr(item, "offsets") else item for item in self._doc_items
                ]
            elif self._rust_encodings is not None:
                self._offsets = [enc.offsets for enc in self._rust_encodings]
            else:
                raise ValueError("EncodedSubwordBatch has neither offsets nor encodings")
        return self._offsets

    def offsets_or_encoding(self, index: int) -> Any:
        """Return the `GigaPackedDoc`, raw Rust `Encoding`, or offset list for document `index`."""
        if self._doc_items is not None:
            return self._doc_items[index]
        if self._offsets is not None:
            return self._offsets[index]
        if self._rust_encodings is not None:
            return self._rust_encodings[index]
        raise ValueError("EncodedSubwordBatch has neither offsets nor encodings")

    def as_dict(self) -> dict[str, Any]:
        """Return a HuggingFace-compatible dict with `input_ids` (`list[list[int]]`) and `offset_mapping`."""
        return {
            "input_ids": self.input_ids,
            "offset_mapping": self.offset_mapping,
        }


def can_use_rust_encode_batch(tokenizer: Any) -> bool:
    """Whether `tokenizer._tokenizer.encode_batch(..., add_special_tokens=False)` is exact for `tokenizer`."""
    if not getattr(tokenizer, "is_fast", False):
        return False
    rust_tok = getattr(tokenizer, "_tokenizer", None)
    if rust_tok is None or not hasattr(rust_tok, "encode_batch"):
        return False
    # Direct `encode_batch` inherits stateful backend truncation/padding, whereas
    # `PreTrainedTokenizerFast.__call__` disables them by default. Fall back if either is active.
    if getattr(rust_tok, "truncation", None) is not None or getattr(rust_tok, "padding", None) is not None:
        return False
    return True


def _model_props_signature(model: Any) -> tuple[Any, ...]:
    """Capture all mutable PyO3 properties on `Unigram`, `WordLevel`, and `BPE` models."""
    return (
        type(model),
        getattr(model, "unk_token", None),
        getattr(model, "dropout", None),
        getattr(model, "continuing_subword_prefix", None),
        getattr(model, "end_of_word_suffix", None),
        getattr(model, "fuse_unk", None),
        getattr(model, "byte_fallback", None),
        getattr(model, "ignore_merges", None),
    )


def _is_supported_giga_backend(rust_tok: Any) -> bool:
    """Verify that `rust_tok` uses an allowed deterministic space-independent pre-tokenizer and normalizer."""
    model = getattr(rust_tok, "model", None)
    if type(model).__name__ not in ("Unigram", "WordLevel", "BPE"):
        return False
    # Stochastic BPE dropout is non-deterministic across calls and must never be word-cached
    if getattr(model, "dropout", None) is not None:
        return False
    if repr(getattr(rust_tok, "pre_tokenizer", None)) not in _ALLOWED_PRETOKENIZER_REPRS:
        return False
    norm = getattr(rust_tok, "normalizer", None)
    norm_type = type(norm).__name__
    if norm_type not in _ALLOWED_NORMALIZER_TYPES:
        return False
    norm_repr = repr(norm)
    if any(bad in norm_repr for bad in _DISALLOWED_NORMALIZER_SUBSTRINGS):
        return False
    return True


def _rust_tokenizer_inline_bytes(rust_tok: Any) -> bytes:
    """Return the inline Rust `TokenizerImpl` struct bytes inside `PyTokenizer` (skipping `PyObject_HEAD`)."""
    size = int(rust_tok.__sizeof__())
    if size <= 16:
        return b""
    return ctypes.string_at(id(rust_tok), size)[16:]


def _fast_vocab_size_le(rust_tok: Any, limit: int = 4096) -> bool:
    """Return whether `rust_tok`'s vocabulary size is at most `limit` in O(1) via `id_to_token`."""
    return rust_tok.id_to_token(limit + 1) is None


def _get_or_init_word_cache(tokenizer: Any) -> _TokenizerWordCache | None:
    """Return the validated `_TokenizerWordCache` for `tokenizer`, or `None` if unsupported."""
    if not can_use_rust_encode_batch(tokenizer):
        return None
    added_encoder = getattr(tokenizer, "added_tokens_encoder", None) or {}
    special_set = frozenset(getattr(tokenizer, "all_special_tokens", ()) or ())
    if not set(added_encoder).issubset(special_set):
        return None

    rust_tok = tokenizer._tokenizer
    added_count = len(added_encoder)
    inline_bytes = _rust_tokenizer_inline_bytes(rust_tok)
    model_props_sig = _model_props_signature(getattr(rust_tok, "model", None))
    pre_tok_repr = repr(getattr(rust_tok, "pre_tokenizer", None))
    norm_type_name = type(getattr(rust_tok, "normalizer", None)).__name__

    cached = _TOKENIZER_CACHES.get(tokenizer)
    if (
        cached is not None
        and cached.rust_tok_id == id(rust_tok)
        and cached.added_tokens_count == added_count
        and cached.inline_struct_bytes == inline_bytes
        and cached.model_props_sig == model_props_sig
        and cached.pre_tok_repr == pre_tok_repr
        and cached.norm_type_name == norm_type_name
    ):
        # For small-vocabulary tokenizers (<= 4096 tokens), also verify the full serialized
        # JSON config hash (`to_str()` takes ~20 us) so even an hypothetical allocator address
        # reuse after multiple model swaps is caught.
        if not _fast_vocab_size_le(rust_tok, 4096) or (
            hashlib.sha256(rust_tok.to_str().encode("utf-8")).hexdigest() == cached.full_config_sha256
        ):
            id_to_token = rust_tok.id_to_token
            if all(id_to_token(tid) == tok_str for tid, tok_str in cached.token_id_invariants):
                return cached if cached.supports_giga_cache else None

    try:
        full_sha = (
            hashlib.sha256(rust_tok.to_str().encode("utf-8")).hexdigest()
            if _fast_vocab_size_le(rust_tok, 4096)
            else ""
        )
    except Exception:
        return None

    if not _is_supported_giga_backend(rust_tok):
        supports = False
    else:
        probe_words = ["Hello", "world.", "Test"]
        probe_text = " ".join(probe_words)
        try:
            full_enc = rust_tok.encode(probe_text, add_special_tokens=False)
            word_encs = rust_tok.encode_batch(probe_words, add_special_tokens=False)
            expected_ids: list[int] = []
            expected_offs: list[OffsetSpan] = []
            cursor = 0
            for w, w_enc in zip(probe_words, word_encs):
                if not w_enc.ids or w_enc.offsets[0][0] != 0 or w_enc.offsets[-1][1] != len(w):
                    supports = False
                    break
                expected_ids.extend(w_enc.ids)
                for ws, we in w_enc.offsets:
                    expected_offs.append((cursor + ws, cursor + we))
                cursor += len(w) + 1
            else:
                supports = full_enc.ids == expected_ids and full_enc.offsets == expected_offs
        except Exception:
            supports = False

    # Re-read `inline_bytes` after the probe encodes in case internal caches initialized
    inline_bytes = _rust_tokenizer_inline_bytes(rust_tok)
    entry = _TokenizerWordCache(
        supports_giga_cache=supports,
        rust_tok_id=id(rust_tok),
        inline_struct_bytes=inline_bytes,
        model_props_sig=model_props_sig,
        pre_tok_repr=pre_tok_repr,
        norm_type_name=norm_type_name,
        full_config_sha256=full_sha,
        added_tokens_count=added_count,
        special_tokens_set=special_set,
        packed_bytes={"": None},
        token_id_invariants=[],
    )
    try:
        _TOKENIZER_CACHES[tokenizer] = entry
    except TypeError:
        pass
    return entry if supports else None


def _populate_unseen_words(
    unseen: list[str],
    rust_tok: Any,
    word_cache: _TokenizerWordCache,
) -> None:
    """Encode unseen words via `rust_tok.encode_batch` and validate per-word offset invariants."""
    packed_bytes = word_cache.packed_bytes
    special_set = word_cache.special_tokens_set
    invariants = word_cache.token_id_invariants

    if len(packed_bytes) + len(unseen) > MAX_WORD_CACHE_SIZE:
        packed_bytes.clear()
        packed_bytes[""] = None
        invariants.clear()
        word_cache.has_zero_delta = False

    to_encode: list[str] = []
    for w in unseen:
        if not w or w in special_set or _NON_SIMPLE_WORD_RE.search(w) is not None:
            packed_bytes[w] = None
        else:
            to_encode.append(w)

    if not to_encode:
        return

    id_to_token = rust_tok.id_to_token
    for w, enc in zip(to_encode, rust_tok.encode_batch(to_encode, add_special_tokens=False)):
        ids = enc.ids
        offs = enc.offsets
        n = len(ids)
        if n == 0 or len(offs) != n or offs[0][0] != 0 or offs[-1][1] != len(w) or offs[0][1] <= 0:
            packed_bytes[w] = None
            continue
        valid = True
        prev_e = offs[0][1]
        word_zero_delta = False
        for idx in range(1, n):
            s_i, e_i = offs[idx]
            if s_i < 0 or e_i <= s_i or e_i < prev_e:
                valid = False
                break
            if e_i == prev_e:
                word_zero_delta = True
            prev_e = e_i
        if not valid:
            packed_bytes[w] = None
            continue
        if word_zero_delta:
            word_cache.has_zero_delta = True

        arr = np.empty((n, 3), dtype=np.int32)
        arr[:, 0] = ids
        arr[0, 1] = offs[0][1] + 1
        arr[0, 2] = offs[0][1] - offs[0][0]
        for idx in range(1, n):
            arr[idx, 1] = offs[idx][1] - offs[idx - 1][1]
            arr[idx, 2] = offs[idx][1] - offs[idx][0]
        packed_bytes[w] = arr.tobytes()
        for tid in ids:
            t_int = int(tid)
            invariants.append((t_int, id_to_token(t_int)))


def fast_tokenize_subword(
    tokenizer: Any,
    texts: list[str],
    *,
    enable_word_cache: bool = True,
) -> EncodedSubwordBatch:
    """Tokenize `texts` without special tokens, using the fused `GigaPy` word cache when eligible."""
    if not texts:
        return EncodedSubwordBatch(token_arrays=[], _input_ids_list=[], _offsets=[])

    word_cache = _get_or_init_word_cache(tokenizer) if enable_word_cache else None
    if word_cache is not None:
        rust_tok = tokenizer._tokenizer
        packed_bytes = word_cache.packed_bytes
        get_packed = packed_bytes.get

        ids_out: list[Any] = [None] * len(texts)
        doc_items: list[Any] = [None] * len(texts)
        fallback_indices: list[int] = []
        candidate_words: list[tuple[int, list[str]]] = []
        unseen_dict: dict[str, None] = {}

        for doc_idx, text in enumerate(texts):
            if not text:
                fallback_indices.append(doc_idx)
                continue
            trimmed = text[:-1] if text.endswith(" ") else text
            words = trimmed.split(" ")
            try:
                raw = b"".join(map(get_packed, words))  # type: ignore[arg-type]
                packed_arr = np.frombuffer(raw, dtype=np.int32).reshape(-1, 3)
                gdoc = GigaPackedDoc(
                    packed_arr,
                    has_zero_delta=word_cache.has_zero_delta,
                    is_single_space_ascii=True,
                )
                ids_out[doc_idx] = gdoc.ids_view
                doc_items[doc_idx] = gdoc
            except TypeError:
                candidate_words.append((doc_idx, words))
                for w in words:
                    if w not in packed_bytes:
                        unseen_dict[w] = None

        if unseen_dict:
            _populate_unseen_words(list(unseen_dict), rust_tok, word_cache)
            missing_after_evict = [
                w for _, words in candidate_words for w in words if w not in packed_bytes
            ]
            if missing_after_evict:
                _populate_unseen_words(list(dict.fromkeys(missing_after_evict)), rust_tok, word_cache)

        for doc_idx, words in candidate_words:
            try:
                raw = b"".join(map(get_packed, words))  # type: ignore[arg-type]
                packed_arr = np.frombuffer(raw, dtype=np.int32).reshape(-1, 3)
                gdoc = GigaPackedDoc(
                    packed_arr,
                    has_zero_delta=word_cache.has_zero_delta,
                    is_single_space_ascii=True,
                )
                ids_out[doc_idx] = gdoc.ids_view
                doc_items[doc_idx] = gdoc
            except TypeError:
                fallback_indices.append(doc_idx)

        if fallback_indices:
            fb_texts = [texts[i] for i in fallback_indices]
            fb_encs = rust_tok.encode_batch(fb_texts, add_special_tokens=False)
            for idx, enc in zip(fallback_indices, fb_encs):
                ids_out[idx] = np.asarray(enc.ids, dtype=np.int32)
                doc_items[idx] = enc

        return EncodedSubwordBatch(token_arrays=ids_out, _doc_items=doc_items)

    if can_use_rust_encode_batch(tokenizer):
        rust_encs = tokenizer._tokenizer.encode_batch(texts, add_special_tokens=False)
        ids = [enc.ids for enc in rust_encs]
        if len(ids) != len(texts):
            raise ValueError("Tokenizer must return one ids list per text")
        return EncodedSubwordBatch(token_arrays=ids, _input_ids_list=ids, _rust_encodings=rust_encs)

    encoded = tokenizer(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(ids) != len(texts) or len(offsets) != len(texts):
        raise ValueError("Tokenizer must return one ids/offsets list per text")
    if any(len(r_ids) != len(r_off) for r_ids, r_off in zip(ids, offsets, strict=True)):
        raise ValueError("Tokenizer ids and offsets must align")
    return EncodedSubwordBatch(token_arrays=ids, _input_ids_list=ids, _offsets=offsets)


def pack_subword_windows(
    token_ids: Sequence[Sequence[int] | NDArray[Any]],
    *,
    block_size: int,
    stride: int,
    downsampling_rate: int = 1,
    cls_id: int,
    sep_id: int,
) -> tuple[Int64Array, Float32Array, Int32Array, int]:
    """Pack subword token sequences into `[num_chunks, eff_block + 2]` arrays without per-window list copies."""
    if not token_ids:
        return (
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            0,
        )
    text_lengths = [len(t) for t in token_ids]
    eff_block = min(block_size, max(text_lengths, default=1))
    if downsampling_rate > 1:
        eff_block = math.ceil(eff_block / downsampling_rate) * downsampling_rate

    num_chunks = sum(math.ceil(max(length - eff_block, 0) / stride) + 1 for length in text_lengths)
    input_ids = np.zeros((num_chunks, eff_block + 2), dtype=np.int64)
    attention_mask = np.zeros((num_chunks, eff_block + 2), dtype=np.float32)
    locations = np.zeros((num_chunks, 3), dtype=np.int32)
    input_ids[:, 0] = cls_id

    cursor = 0
    for doc_idx, tokens in enumerate(token_ids):
        t_len = text_lengths[doc_idx]
        if t_len == 0:
            input_ids[cursor, 1] = sep_id
            attention_mask[cursor, :2] = 1.0
            locations[cursor] = (doc_idx, 0, 0)
            cursor += 1
            continue

        tok_arr = tokens if isinstance(tokens, np.ndarray) else np.asarray(tokens, dtype=np.int64)
        if t_len <= eff_block:
            input_ids[cursor, 1 : 1 + t_len] = tok_arr
            input_ids[cursor, 1 + t_len] = sep_id
            attention_mask[cursor, : t_len + 2] = 1.0
            locations[cursor] = (doc_idx, 0, t_len)
            cursor += 1
        else:
            for start_pos in range(0, t_len, stride):
                start, end = start_pos, start_pos + eff_block
                done = end >= t_len
                if done:
                    end = t_len
                    start = max(end - eff_block, 0)
                n = end - start
                input_ids[cursor, 1 : 1 + n] = tok_arr[start:end]
                input_ids[cursor, 1 + n] = sep_id
                attention_mask[cursor, : n + 2] = 1.0
                locations[cursor] = (doc_idx, start, end)
                cursor += 1
                if done:
                    break

    if cursor != num_chunks:
        raise ValueError("Window count differs from preallocation")
    return input_ids, attention_mask, locations, eff_block


def stitch_subword_logits(
    raw_batches: Sequence[NDArray[Any]],
    locations: Int32Array,
    text_lengths: Sequence[int],
    *,
    block_size: int,
    num_labels: int,
    weighting: Literal["uniform", "hat"] = "uniform",
    strip_cls_sep: bool = True,
    batch_size: int | None = None,
) -> list[Float16Array]:
    """Stitch window logits into per-document `float16` token logits using Identity 1 + `cumsum` counts."""
    num_docs = len(text_lengths)
    total_windows = len(locations)
    if batch_size is not None:
        expected = math.ceil(total_windows / batch_size)
        if len(raw_batches) != expected:
            raise ValueError("Wrong number of forward batches")

    if weighting == "uniform":
        stitched: list[Float16Array] = [
            np.empty((0, num_labels), dtype=np.float16) for _ in range(num_docs)
        ]

        # Fast 2D span-grouped multi-window stitch when all windows are in a single batch array
        if len(raw_batches) == 1 and total_windows > 0:
            batch_logits = raw_batches[0]
            if (
                batch_logits.ndim != 3
                or (batch_size is not None and len(batch_logits) != total_windows)
                or len(batch_logits) < total_windows
                or batch_logits.shape[2] != num_labels
            ):
                raise ValueError("Forward logits do not match prepared batch geometry")
            values = batch_logits[:total_windows, 1:-1, :] if strip_cls_sep else batch_logits[:total_windows]

            if max(text_lengths, default=0) <= block_size:
                for row in range(total_windows):
                    loc = locations[row]
                    doc_idx, start, end = int(loc[0]), int(loc[1]), int(loc[2])
                    n = end - start
                    if values.shape[1] < n:
                        raise ValueError("Forward logits are shorter than their window")
                    stitched[doc_idx] = np.asarray(values[row, :n], dtype=np.float16)
                return stitched

            doc_spans: dict[int, list[tuple[int, int, int]]] = {}
            for row in range(total_windows):
                loc = locations[row]
                doc_idx, start, end = int(loc[0]), int(loc[1]), int(loc[2])
                t_len = text_lengths[doc_idx]
                n = end - start
                if values.shape[1] < n:
                    raise ValueError("Forward logits are shorter than their window")
                if t_len <= block_size:
                    stitched[doc_idx] = np.asarray(values[row, :n], dtype=np.float16)
                else:
                    if doc_idx not in doc_spans:
                        doc_spans[doc_idx] = [(start, end, row)]
                    else:
                        doc_spans[doc_idx].append((start, end, row))

            by_geometry: dict[tuple[int, tuple[tuple[int, int], ...]], list[tuple[int, list[int]]]] = {}
            for doc_idx, span_rows in doc_spans.items():
                geom_key = (text_lengths[doc_idx], tuple((s, e) for s, e, _ in span_rows))
                rows_list = [r for _, _, r in span_rows]
                if geom_key not in by_geometry:
                    by_geometry[geom_key] = [(doc_idx, rows_list)]
                else:
                    by_geometry[geom_key].append((doc_idx, rows_list))

            for (t_len, span_seq), group in by_geometry.items():
                n_group = len(group)
                acc_2d = np.zeros((n_group, t_len, num_labels), dtype=np.float16)
                diff = np.zeros(t_len + 1, dtype=np.int32)
                row_matrix = np.asarray([ r_list for _, r_list in group ], dtype=np.intp)
                for w_idx, (s, e) in enumerate(span_seq):
                    n = e - s
                    w_rows = row_matrix[:, w_idx]
                    if w_idx == 0 and s == 0 and values.dtype == np.float16:
                        acc_2d[:, :n] = values[w_rows, :n]
                    else:
                        acc_2d[:, s:e] += values[w_rows, :n]
                    diff[s] += 1
                    diff[e] -= 1
                cnt_f16 = np.cumsum(diff[:-1]).astype(np.float16)[None, :, None]
                res_2d = (acc_2d / cnt_f16).astype(np.float16)
                for g_idx, (doc_idx, _) in enumerate(group):
                    stitched[doc_idx] = res_2d[g_idx]
            return stitched

        multi_acc: dict[int, Float16Array] = {}
        multi_diff: dict[int, Int32Array] = {}

        cursor = 0
        for batch_logits in raw_batches:
            real = min(batch_size if batch_size is not None else len(batch_logits), total_windows - cursor)
            if (
                batch_logits.ndim != 3
                or (batch_size is not None and len(batch_logits) != real)
                or len(batch_logits) < real
                or batch_logits.shape[2] != num_labels
            ):
                raise ValueError("Forward logits do not match prepared batch geometry")
            values = batch_logits[:real, 1:-1, :] if strip_cls_sep else batch_logits[:real]
            for row in range(real):
                loc = locations[cursor + row]
                doc_idx, start, end = int(loc[0]), int(loc[1]), int(loc[2])
                t_len = text_lengths[doc_idx]
                n = end - start
                if values.shape[1] < n:
                    raise ValueError("Forward logits are shorter than their window")
                if t_len <= block_size:
                    stitched[doc_idx] = np.asarray(values[row, :n], dtype=np.float16)
                else:
                    if doc_idx not in multi_acc:
                        acc = np.zeros((t_len, num_labels), dtype=np.float16)
                        if start == 0 and values.dtype == np.float16:
                            acc[:n] = values[row, :n]
                        else:
                            acc[start:end] += values[row, :n]
                        multi_acc[doc_idx] = acc
                        diff = np.zeros(t_len + 1, dtype=np.int32)
                        diff[start] += 1
                        diff[end] -= 1
                        multi_diff[doc_idx] = diff
                    else:
                        multi_acc[doc_idx][start:end] += values[row, :n]
                        diff = multi_diff[doc_idx]
                        diff[start] += 1
                        diff[end] -= 1
            cursor += real

        if cursor != total_windows:
            raise ValueError("Wrong number of forward batches")

        for doc_idx, acc in multi_acc.items():
            cnt_f16 = np.cumsum(multi_diff[doc_idx][:-1]).astype(np.float16)[:, None]
            stitched[doc_idx] = (acc / cnt_f16).astype(np.float16)
        return stitched

    if weighting != "hat":
        raise ValueError(f"Unknown weighting scheme: {weighting!r}")

    if block_size <= 1:
        weights = np.ones(block_size, dtype=np.float16)
    else:
        x = np.linspace(-(1 - 1 / block_size), 1 - 1 / block_size, block_size, dtype=np.float16)
        weights = 1 - np.abs(x)

    all_logits = [np.zeros((length, num_labels), dtype=np.float16) for length in text_lengths]
    all_counts = [np.zeros(length, dtype=np.float16) for length in text_lengths]

    cursor = 0
    for batch_logits in raw_batches:
        real = min(batch_size if batch_size is not None else len(batch_logits), total_windows - cursor)
        if (
            batch_logits.ndim != 3
            or (batch_size is not None and len(batch_logits) != real)
            or len(batch_logits) < real
            or batch_logits.shape[2] != num_labels
        ):
            raise ValueError("Forward logits do not match prepared batch geometry")
        values = batch_logits[:real, 1:-1, :] if strip_cls_sep else batch_logits[:real]
        for row in range(real):
            loc = locations[cursor + row]
            doc_idx, start, end = int(loc[0]), int(loc[1]), int(loc[2])
            n = end - start
            if values.shape[1] < n:
                raise ValueError("Forward logits are shorter than their window")
            w_slice = weights[:n]
            all_logits[doc_idx][start:end] += w_slice[:, None] * values[row, :n]
            all_counts[doc_idx][start:end] += w_slice
        cursor += real

    if cursor != total_windows:
        raise ValueError("Wrong number of forward batches")

    return [(logits / counts[:, None]).astype(np.float16) for logits, counts in zip(all_logits, all_counts)]


def project_subword_proba(
    text_len: int,
    tokens: Sequence[int] | NDArray[Any],
    stitched_logits: Float16Array,
    offsets: Any,
    special_tokens: Sequence[Any],
    *,
    newline_idx: int = Constants.NEWLINE_INDEX,
) -> Float32Array:
    """Map token logits to character probabilities in `<f4` using Identity 2 (`sigmoid` before scatter)."""
    char_probs = np.zeros(text_len, dtype="<f4")
    if text_len == 0 or len(tokens) == 0:
        return char_probs

    has_int_special = any(isinstance(tok, (int, np.integer)) for tok in special_tokens)
    if isinstance(offsets, GigaPackedDoc):
        deltas = offsets.deltas_view
        if len(deltas) == 0:
            return char_probs
        if not has_int_special and len(deltas) == len(tokens):
            with np.errstate(over="ignore"):
                tok_p = np.asarray(sigmoid(stitched_logits[:, newline_idx]), dtype="<f4")
            char_indices = np.cumsum(deltas) - 2
            char_probs[char_indices] = tok_p[: len(char_indices)]
            return char_probs
        off_seq: Sequence[OffsetSpan] = offsets.offsets
    elif hasattr(offsets, "offsets"):
        off_seq = offsets.offsets
    else:
        off_seq = offsets

    if len(off_seq) == 0:
        return char_probs

    if not has_int_special and len(off_seq) == len(tokens):
        with np.errstate(over="ignore"):
            tok_p = np.asarray(sigmoid(stitched_logits[:, newline_idx]), dtype="<f4")
        off_arr = np.asarray(off_seq, dtype=np.int64)
        char_probs[off_arr[:, 1] - 1] = tok_p[: len(off_arr)]
        return char_probs

    tokens_seq: Sequence[int] = tokens.tolist() if isinstance(tokens, np.ndarray) else tokens
    valid_indices, valid_offsets = get_token_spans(off_seq, tokens_seq, special_tokens)
    if len(valid_indices) == 0:
        return char_probs
    with np.errstate(over="ignore"):
        tok_p = np.asarray(sigmoid(stitched_logits[valid_indices, newline_idx]), dtype="<f4")
    char_probs[valid_offsets[:, 1] - 1] = tok_p
    return char_probs


def _slice_sentences_single_space(
    text: str,
    cuts: list[int],
    *,
    strip_whitespace: bool = False,
) -> list[str]:
    """Fast sentence slicer for `GigaPackedDoc` texts containing only single ASCII spaces."""
    text_len = len(text)
    sentences: list[str] = []
    append = sentences.append
    offset = 0
    idx = 0
    if strip_whitespace:
        for cut in cuts:
            idx = cut + 1
            if idx < text_len and text[idx] == " ":
                idx += 1
            s = text[offset:idx].strip()
            if s:
                append(s)
            offset = idx
        if idx != text_len:
            last = text[idx:].strip()
            if last:
                append(last)
    else:
        for cut in cuts:
            idx = cut + 1
            if idx < text_len and text[idx] == " ":
                idx += 1
            s = text[offset:idx]
            if s:
                append(s)
            offset = idx
        if idx != text_len:
            last = text[idx:]
            if last:
                append(last)
    return sentences


def decode_subword_split(
    text: str,
    tok_logits: Float16Array,
    offsets_or_encoding: Any,
    *,
    z_min: np.float16 | None,
    strip_whitespace: bool = False,
    split_on_input_newlines: bool = True,
    newline_idx: int = Constants.NEWLINE_INDEX,
) -> list[str]:
    """Decode sentence splits directly from `float16` token logits using Identity 3 (zero `O(len(text))` arrays)."""
    text_len = len(text)
    if text_len == 0:
        return []

    is_neg_inf = z_min is not None and float(z_min) == float("-inf")

    if z_min is None or len(tok_logits) == 0:
        if is_neg_inf:
            sentences = indices_to_sentences(text, range(text_len), strip_whitespace=strip_whitespace)
            if split_on_input_newlines and "\n" in text:
                newline_split: list[str] = []
                for idx, sentence in enumerate(sentences):
                    if idx < len(sentences) - 1 and sentence.endswith("\n"):
                        sentence = sentence[:-1]
                    newline_split.extend(sentence.split("\n"))
                return newline_split
            return sentences
        passing_arr = np.empty(0, dtype=np.intp)
        logits_1d: Float16Array = np.empty(0, dtype=np.float16)
    elif is_neg_inf:
        logits_1d = tok_logits[:, newline_idx] if tok_logits.ndim == 2 else tok_logits
        off_list_neg: Sequence[OffsetSpan] = (
            offsets_or_encoding.offsets if hasattr(offsets_or_encoding, "offsets") else offsets_or_encoding
        )
        n_tok_neg = min(len(off_list_neg), len(logits_1d))
        cuts_mask = np.ones(text_len, dtype=bool)
        if np.isnan(logits_1d[:n_tok_neg]).any():
            for j in range(n_tok_neg):
                cj = off_list_neg[j][1] - 1
                if cj < 0:
                    cj += text_len
                if 0 <= cj < text_len:
                    cuts_mask[cj] = not np.isnan(logits_1d[j])
        sentences = indices_to_sentences(text, np.flatnonzero(cuts_mask), strip_whitespace=strip_whitespace)
        if split_on_input_newlines and "\n" in text:
            newline_split_neg: list[str] = []
            for idx, sentence in enumerate(sentences):
                if idx < len(sentences) - 1 and sentence.endswith("\n"):
                    sentence = sentence[:-1]
                newline_split_neg.extend(sentence.split("\n"))
            return newline_split_neg
        return sentences
    else:
        logits_1d = tok_logits[:, newline_idx] if tok_logits.ndim == 2 else tok_logits
        passing_arr = np.flatnonzero(logits_1d >= z_min)

    if passing_arr.size == 0:
        sentences = indices_to_sentences(text, [], strip_whitespace=strip_whitespace)
    elif isinstance(offsets_or_encoding, GigaPackedDoc):
        deltas = offsets_or_encoding.deltas_view
        n_tok = min(len(deltas), len(logits_1d))
        if int(passing_arr[-1]) >= n_tok:
            passing_arr = passing_arr[passing_arr < n_tok]
        if passing_arr.size == 0:
            sentences = indices_to_sentences(text, [], strip_whitespace=strip_whitespace)
        else:
            if not offsets_or_encoding.has_zero_delta:
                max_k = int(passing_arr[-1]) + 1
                cuts = (np.cumsum(deltas[:max_k])[passing_arr] - 2).tolist()
            else:
                next_idx = np.minimum(passing_arr + 1, n_tok - 1)
                has_collision = bool(((passing_arr < n_tok - 1) & (deltas[next_idx] == 0)).any())
                if not has_collision:
                    max_k = int(passing_arr[-1]) + 1
                    cuts = (np.cumsum(deltas[:max_k])[passing_arr] - 2).tolist()
                else:
                    assert z_min is not None
                    c_all = np.cumsum(deltas[:n_tok]) - 2
                    last_in_run = np.searchsorted(c_all, c_all[passing_arr], side="right") - 1
                    keep = logits_1d[last_in_run] >= z_min
                    cuts = np.unique(c_all[passing_arr[keep]]).tolist()
            if offsets_or_encoding.is_single_space_ascii:
                return _slice_sentences_single_space(text, cuts, strip_whitespace=strip_whitespace)
            sentences = indices_to_sentences(text, cuts, strip_whitespace=strip_whitespace)
    else:
        off_list: Sequence[OffsetSpan] = (
            offsets_or_encoding.offsets if hasattr(offsets_or_encoding, "offsets") else offsets_or_encoding
        )
        n_tok = min(len(off_list), len(logits_1d))
        passing_set = set(passing_arr[passing_arr < n_tok].tolist())
        char_cuts: set[int] = set()

        is_monotonic = True
        prev_end = 0
        for j in range(n_tok):
            e_j = off_list[j][1]
            if e_j < 0 or e_j > text_len:
                is_monotonic = False
                break
            if e_j > 0:
                if e_j < prev_end:
                    is_monotonic = False
                    break
                prev_end = e_j

        for k in passing_set:
            c = off_list[k][1] - 1
            if c < 0:
                c += text_len
            if c < 0 or c >= text_len:
                continue
            overwritten = False
            if not is_monotonic or c == text_len - 1:
                for k2 in range(n_tok - 1, k, -1):
                    c2 = off_list[k2][1] - 1
                    if c2 < 0:
                        c2 += text_len
                    if c2 == c:
                        overwritten = k2 not in passing_set
                        break
            else:
                k2 = k + 1
                while k2 < n_tok:
                    end2 = off_list[k2][1]
                    if end2 == 0:
                        k2 += 1
                        continue
                    if end2 - 1 == c:
                        overwritten = k2 not in passing_set
                        k2 += 1
                        continue
                    if end2 - 1 > c:
                        break
                    k2 += 1
            if not overwritten:
                char_cuts.add(c)

        sentences = indices_to_sentences(text, sorted(char_cuts), strip_whitespace=strip_whitespace)

    if split_on_input_newlines and "\n" in text:
        newline_split_sentences: list[str] = []
        for idx, sentence in enumerate(sentences):
            if idx < len(sentences) - 1 and sentence.endswith("\n"):
                sentence = sentence[:-1]
            newline_split_sentences.extend(sentence.split("\n"))
        return newline_split_sentences

    return sentences


def extract_subword_logits_fast(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    *,
    stride: int,
    max_block_size: int,
    batch_size: int,
    pad_last_batch: bool = False,
    weighting: Literal["uniform", "hat"] = "uniform",
    max_content_block: int = 510,
) -> tuple[list[Float16Array], EncodedSubwordBatch]:
    """Tokenize, window-pack, forward, and stitch subword logits with `GigaPy` word caching & lazy offsets."""
    encoded = fast_tokenize_subword(tokenizer, texts)
    token_ids = encoded.token_arrays
    text_lengths = [len(t) for t in token_ids]
    downsampling_rate = int(getattr(model.config, "downsampling_rate", 1))
    eff_max_block = min(max_block_size, max_content_block)

    input_ids, attention_mask, locs, block_size = pack_subword_windows(
        token_ids,
        block_size=eff_max_block,
        stride=stride,
        downsampling_rate=downsampling_rate,
        cls_id=int(tokenizer.cls_token_id),
        sep_id=int(tokenizer.sep_token_id),
    )
    pad_token_id = int(tokenizer.pad_token_id)
    n_batches = math.ceil(len(attention_mask) / batch_size)
    raw_batches: list[NDArray[Any]] = []

    for batch_idx in range(n_batches):
        start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)
        batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]
        if len(batch_attention_mask) < batch_size and pad_last_batch:
            n_missing = batch_size - len(batch_attention_mask)
            batch_input_ids = np.pad(batch_input_ids, ((0, n_missing), (0, 0)), constant_values=pad_token_id)
            batch_attention_mask = np.pad(batch_attention_mask, ((0, n_missing), (0, 0)))
        logits = model(attention_mask=batch_attention_mask, input_ids=batch_input_ids)["logits"]
        raw_batches.append(logits[: end - start])

    stitched = stitch_subword_logits(
        raw_batches,
        locs,
        text_lengths,
        block_size=block_size,
        num_labels=int(model.config.num_labels),
        weighting=weighting,
        strip_cls_sep=True,
    )
    return stitched, encoded
