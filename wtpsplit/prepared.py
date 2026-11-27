"""CPU-only, portable token batches for producer/GPU-process separation.

Arrays own token IDs, character offsets and document boundaries. Transport via
pickle or place those arrays in shared memory; no tokenizer, model or tensor is
retained. The GPU process applies its ordinary windowing and decoding policy.
"""
from dataclasses import dataclass
import hashlib
import inspect
import os
from typing import Any, Literal, Sequence, TypedDict, cast
import warnings

import numpy as np
from numpy.typing import NDArray

from wtpsplit._host_kernels import EncodedSubwordBatch, can_use_rust_encode_batch
from wtpsplit._inference import (
    decode_probabilities, normalize_input_line_breaks, notified_texts, validate_segmentation_options,
)
from wtpsplit.chunking import LengthFunction


class PreparedSplitOptions(TypedDict, total=False):
    """The ordinary split options, applied without repeating tokenization."""

    threshold: float | None
    stride: int
    block_size: int
    batch_size: int
    pad_last_batch: bool
    weighting: Literal['uniform', 'hat']
    remove_whitespace_before_inference: bool
    outer_batch_size: int
    paragraph_threshold: float
    strip_whitespace: bool
    do_paragraph_segmentation: bool
    split_on_input_newlines: bool
    treat_newline_as_space: bool | None
    verbose: bool
    max_length: int | None
    length_fn: LengthFunction
    lazy: bool


Segments = list[list[str]] | list[list[list[str]]]


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Bind the complete deterministic fast-tokenizer configuration, not its ID."""
    backend = getattr(tokenizer, '_tokenizer', tokenizer)
    if backend is None or not hasattr(backend, 'to_str'):
        raise ValueError('Prepared batches require a serializable fast tokenizer')
    if getattr(getattr(backend, 'model', None), 'dropout', None):
        raise ValueError('Prepared batches require deterministic tokenization (BPE dropout disabled)')
    return hashlib.sha256(backend.to_str().encode()).hexdigest()


@dataclass(frozen=True)
class PreparedBatch:
    """Portable flat token arrays, original texts and stable document identities.

    Offsets use Unicode character indices, matching ``SaT.split``. Original
    texts are needed to reconstruct strings; inference text reflects only the
    declared line-break and whitespace policy. Supply corpus IDs in producers;
    defaults identify content plus its occurrence within this batch.
    """

    texts: tuple[str, ...]
    inference_texts: tuple[str, ...]
    document_ids: tuple[str, ...]
    input_ids: NDArray[np.int32]
    offsets: NDArray[np.int64]
    document_boundaries: NDArray[np.int64]
    tokenizer_sha256: str
    input_line_breaks: Literal['space', 'preserve']
    remove_whitespace_before_inference: bool = False

    def validate(self) -> None:
        """Refuse malformed IPC arrays and inconsistent duplicate documents."""
        size = len(self.texts)
        if any(not isinstance(ident, str) or not ident for ident in self.document_ids):
            raise ValueError('Prepared document identities must be nonempty strings')
        if len(self.inference_texts) != size or len(self.document_ids) != size or len(set(self.document_ids)) != size:
            raise ValueError('Prepared document texts and unique identities must have equal coverage')
        if not isinstance(self.input_ids, np.ndarray) or self.input_ids.dtype != np.int32 or self.input_ids.ndim != 1 or bool((self.input_ids < 0).any()):
            raise ValueError('Prepared input_ids must be a nonnegative flat int32 array')
        if not isinstance(self.offsets, np.ndarray) or self.offsets.dtype != np.int64 or self.offsets.shape != (len(self.input_ids), 2):
            raise ValueError('Prepared offsets must be an int64 [tokens, 2] array')
        bounds = self.document_boundaries
        if (not isinstance(bounds, np.ndarray) or bounds.dtype != np.int64 or bounds.shape != (size+1,) or bounds[0] != 0
                or bounds[-1] != len(self.input_ids) or bool((bounds[1:] < bounds[:-1]).any())):
            raise ValueError('Prepared document_boundaries must cover every token exactly once')
        seen: dict[str, int] = {}
        for index, (original, text) in enumerate(zip(self.texts, self.inference_texts)):
            normalized = normalize_input_line_breaks(original, self.input_line_breaks)
            if self.remove_whitespace_before_inference:
                normalized = normalized.replace(' ', '')
            if normalized != text:
                raise ValueError('Prepared text differs from its declared inference input policy')
            start, end = int(bounds[index]), int(bounds[index+1])
            spans = self.offsets[start:end]
            if bool((spans < 0).any()) or bool((spans[:, 1] < spans[:, 0]).any()) or bool((spans > len(text)).any()):
                raise ValueError('Prepared token offsets are outside their document')
            if text in seen:
                previous = seen[text]
                a, b = int(bounds[previous]), int(bounds[previous+1])
                if not np.array_equal(self.input_ids[a:b], self.input_ids[start:end]) or not np.array_equal(self.offsets[a:b], spans):
                    raise ValueError('Identical prepared text has inconsistent tokenization')
            seen[text] = index

    def encoded_for_texts(self, texts: Sequence[str]) -> EncodedSubwordBatch:
        """Create request-local views for the normal window builder, with no tokenizer call."""
        lookup = {text: i for i, text in enumerate(self.inference_texts)}
        ids, offsets = [], []
        for text in texts:
            if text not in lookup:
                raise ValueError('Inference requested text absent from the prepared batch')
            index = lookup[text]
            start, end = int(self.document_boundaries[index]), int(self.document_boundaries[index+1])
            ids.append(self.input_ids[start:end])
            offsets.append(self.offsets[start:end])
        return EncodedSubwordBatch(token_arrays=ids, _doc_items=offsets)

    @classmethod
    def from_tokenizer(cls, texts: Sequence[str], tokenizer: Any, *,
                       document_ids: Sequence[str] | None = None,
                       input_line_breaks: Literal['space', 'preserve'] = 'space',
                       remove_whitespace_before_inference: bool = False,
                       _tokenizer_fingerprint: str | None = None) -> 'PreparedBatch':
        """Prepare with a HF fast tokenizer or a raw Rust Tokenizer.

        A producer that exclusively owns an unchanged backend can bind its
        fingerprint once with the private argument. The receiving SaT always
        verifies it against its own tokenizer; normal callers recompute it.
        """
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        if isinstance(texts, str):
            raise TypeError('prepare expects a sequence of documents, not one string')
        originals = tuple(texts)
        normalized = tuple(normalize_input_line_breaks(t, input_line_breaks) for t in originals)
        if remove_whitespace_before_inference:
            normalized = tuple(t.replace(' ', '') for t in normalized)
        fingerprint = _tokenizer_fingerprint or tokenizer_fingerprint(tokenizer)
        if isinstance(tokenizer, Tokenizer):
            if tokenizer.truncation is not None or tokenizer.padding is not None:
                raise ValueError('Raw prepared tokenizers must disable truncation and padding')
            if getattr(tokenizer.model, 'dropout', None):
                raise ValueError('Prepared tokenizers must disable BPE dropout')
            encodings = tokenizer.encode_batch(list(normalized), add_special_tokens=False)
            ids = [r.ids for r in encodings]
            offsets = [r.offsets for r in encodings]
        elif can_use_rust_encode_batch(tokenizer):
            encodings = tokenizer._tokenizer.encode_batch(list(normalized), add_special_tokens=False)
            ids = [r.ids for r in encodings]
            offsets = [r.offsets for r in encodings]
        elif normalized:
            encoded = tokenizer(list(normalized), return_offsets_mapping=True, add_special_tokens=False, verbose=False)
            ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        else:
            ids, offsets = [], []
        if len(ids) != len(originals) or len(offsets) != len(originals):
            raise ValueError('Tokenizer did not return one token/offset row per document')
        if document_ids is None:
            counts: dict[str, int] = {}
            identities = []
            for text in originals:
                digest = hashlib.sha256(text.encode()).hexdigest()
                occurrence = counts.get(digest, 0)
                counts[digest] = occurrence+1
                identities.append(f'{digest}:{occurrence}')
        else:
            identities = list(document_ids)
        lengths = np.asarray([len(row) for row in ids], dtype=np.int64)
        bounds = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths)))
        flat_ids = np.concatenate([np.asarray(row, dtype=np.int32) for row in ids]) if ids else np.empty(0, np.int32)
        flat_offsets = (np.concatenate([np.asarray(row, dtype=np.int64).reshape(-1, 2) for row in offsets])
                        if offsets else np.empty((0, 2), np.int64))
        result = cls(originals, normalized, tuple(identities), flat_ids, flat_offsets, bounds,
                     fingerprint, input_line_breaks, remove_whitespace_before_inference)
        result.validate()
        return result


def split_prepared(sat: Any, batch: PreparedBatch, *, return_segments: bool,
                   options: PreparedSplitOptions) -> list[tuple[int, ...]] | Segments:
    """Reuse ordinary probabilities and decoding, retaining all per-position outputs."""
    batch.validate()
    if tokenizer_fingerprint(sat.tokenizer) != batch.tokenizer_sha256:
        raise ValueError('Prepared tokenizer fingerprint differs from this SaT tokenizer')
    if sat.__dict__.get('input_line_breaks', 'preserve') != batch.input_line_breaks:
        raise ValueError('Prepared input_line_breaks differs from this SaT input policy')
    if getattr(sat.model.config, 'language_adapter', 'off') == 'on':
        raise ValueError('Prepared inputs do not support language-adapter feature extraction')
    bound = inspect.signature(sat.split).bind(list(batch.texts), **options)
    bound.apply_defaults()
    resolved = bound.arguments
    if resolved['max_length'] is not None and resolved['strip_whitespace']:
        raise ValueError('strip_whitespace is incompatible with a reconstructing hard cap')
    if resolved['lazy']:
        raise ValueError('split_prepared returns a materialized batch; lazy=True is unsupported')
    if resolved['remove_whitespace_before_inference'] != batch.remove_whitespace_before_inference:
        raise ValueError('Prepare and split_prepared must use the same remove_whitespace_before_inference policy')
    force_newlines = resolved['split_on_input_newlines']
    if resolved['treat_newline_as_space'] is not None:
        warnings.warn('treat_newline_as_space is deprecated; use split_on_input_newlines', DeprecationWarning, stacklevel=3)
        force_newlines = not resolved['treat_newline_as_space']
    validate_segmentation_options(threshold=resolved['threshold'], max_length=resolved['max_length'],
                                  length_fn=resolved['length_fn'], split_on_input_newlines=force_newlines)
    texts = notified_texts(list(batch.texts), force_newlines and not resolved['do_paragraph_segmentation'])
    from wtpsplit.extract import CHAR_MODEL_TYPES

    is_character = bool(getattr(sat.model.config, 'use_character_head', False))
    fast_policy = os.environ.get('WTPSPLIT_PREPARED_FAST', '1')
    if fast_policy not in ('0', '1'):
        raise ValueError('WTPSPLIT_PREPARED_FAST must be 0 or 1')
    fast_options = (fast_policy == '1' and not resolved['do_paragraph_segmentation'] and resolved['max_length'] is None
                    and not resolved['remove_whitespace_before_inference'] and not resolved['verbose']
                    and ((is_character and getattr(sat, 'character_preprocessing', 'reference') == 'exact')
                         or (not is_character and sat.model.config.model_type not in CHAR_MODEL_TYPES)))
    if fast_options:
        split_keys = ('threshold', 'stride', 'block_size', 'batch_size', 'pad_last_batch', 'weighting',
                      'paragraph_threshold', 'remove_whitespace_before_inference', 'outer_batch_size',
                      'do_paragraph_segmentation', 'max_length', 'length_fn', 'verbose')
        splits = sat._split(texts, **{key: resolved[key] for key in split_keys},
                            split_on_input_newlines=force_newlines,
                            strip_whitespace=resolved['strip_whitespace'] if return_segments else False,
                            _prepared_batch=batch, _return_offsets=not return_segments)
        if return_segments:
            return cast(Segments, list(splits))
        return [tuple(ends) for ends in splits]
    probability_keys = ('stride', 'block_size', 'batch_size', 'pad_last_batch', 'weighting',
                        'remove_whitespace_before_inference', 'outer_batch_size', 'verbose')
    probabilities = sat._predict_proba(texts, **{key: resolved[key] for key in probability_keys},
                                      return_paragraph_probabilities=resolved['do_paragraph_segmentation'],
                                      _prepared_batch=batch)
    sentence_threshold = sat._resolve_sentence_threshold(resolved['threshold'])
    result: list[Any] = []
    for text, values in zip(texts, probabilities, strict=True):
        decoded = decode_probabilities(text, values, sentence_threshold=sentence_threshold,
            paragraph_threshold=resolved['paragraph_threshold'],
            strip_whitespace=resolved['strip_whitespace'] if return_segments else False,
            do_paragraph_segmentation=resolved['do_paragraph_segmentation'], split_on_input_newlines=force_newlines,
            max_length=resolved['max_length'], length_fn=resolved['length_fn'], language=sat.language)
        if return_segments:
            result.append(decoded.paragraphs if decoded.paragraphs is not None else decoded.sentences)
        else:
            if ''.join(decoded.sentences) != text:
                raise ValueError('Prepared segmentation did not reconstruct the original document')
            result.append(tuple(int(v) for v in np.cumsum([len(piece) for piece in decoded.sentences[:-1]])))
    return cast(list[tuple[int, ...]] | Segments, result)
