"""Shared, model-agnostic inference helpers for SaT and legacy WtP.

Model classes own checkpoint loading and conversion from logits to boundary
probabilities. This module owns input dispatch, batching, whitespace
restoration, option validation, and probability-to-segment decoding.
"""

from __future__ import annotations

import math
import os
import re
import threading
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from wtpsplit.chunking import LengthFunction, UnsplittableUnitError, forced_newline_ends, measure, source_spans, split_spans, resolve_budget
from wtpsplit.utils import indices_to_sentences

T = TypeVar("T")
FloatArray = NDArray[Any]
ProbabilityPair = tuple[FloatArray, FloatArray]

# How the model encoder sees input line breaks. "preserve" feeds the text unchanged (every
# released SaT/SaT-SM/WtP checkpoint). "space" replaces each LF/CR with one space before
# encoding, for checkpoints whose tokenizer/character head reacts to raw line breaks.
INPUT_LINE_BREAK_MODES = ("preserve", "space")
_LINE_BREAKS_TO_SPACES = str.maketrans({"\n": " ", "\r": " "})


class WtpsplitBehaviorChangeWarning(UserWarning):
    """Notice that 3.0 split/segment retain input line endings."""


_retention_notice_pid: int | None = None
_retention_notice_origin = os.getpid()
_retention_notice_lock = threading.Lock()


def notified_texts(texts: list[str], enabled: bool | None) -> list[str]:
    """Issue the retention notice once per process and return this input batch."""
    global _retention_notice_pid, _retention_notice_origin, _retention_notice_lock
    pid = os.getpid()
    if not enabled or _retention_notice_pid == pid or not any("\n" in t for t in texts):
        return texts
    if _retention_notice_origin != pid:
        # A fork must not inherit an acquired parent lock or its notice state.
        _retention_notice_origin = pid
        _retention_notice_lock = threading.Lock()
    with _retention_notice_lock:
        if _retention_notice_pid != pid:
            _retention_notice_pid = pid
            warnings.warn(
                "wtpsplit 3.0 keeps input line endings in split/segment pieces.\n"
                "Approximately old stripped strings: strip_whitespace=True (uncapped), or [s.strip() for s in pieces if s.strip()].\n"
                'Silence: warnings.filterwarnings("ignore", category=WtpsplitBehaviorChangeWarning).',
                WtpsplitBehaviorChangeWarning,
                stacklevel=6,
            )
    return texts


def owned_sentence_spans(
    text: str, ends: Iterable[int], *, force_newlines: bool, fast_newlines: bool = True,
) -> list[tuple[int, int]]:
    """Assign all trailing whitespace to the left and leading whitespace to the first piece.

    Forced LF cuts (including CRLF) retain blank-line runs. Boundary selection never
    creates an empty or whitespace-only piece in mixed-content text. Hard caps
    can still require subdividing a whitespace run into whitespace-only pieces.
    """
    if not text:
        return []
    first_content = next((i for i, c in enumerate(text) if not c.isspace()), len(text))
    if first_content == len(text):
        return [(0, len(text))]
    boundaries = {len(text), *(int(e) for e in ends if 0 < e <= len(text))}
    if force_newlines and "\n" in text:
        # Snap only the new forced cuts to grapheme ends; existing model cuts
        # keep their established ordinary-splitting convention.
        if fast_newlines:
            boundaries.update(forced_newline_ends(text))
        else:
            forced = [m.end() for m in re.finditer(r"\n", text)]
            if text.isascii():
                boundaries.update(forced)
            else:
                boundaries.update(b for _, b in source_spans(text, forced))
    result = []
    start = advanced = 0
    for candidate in sorted(boundaries):
        if candidate <= advanced:
            continue
        end = candidate
        while end < len(text) and text[end].isspace():
            end += 1
        advanced = end
        if end <= first_content:
            continue
        result.append((start, end))
        start = end
    return result


def retained_sentences(text: str, sentences: list[str], *, force_newlines: bool, strip_whitespace: bool) -> list[str]:
    """Apply shared newline ownership to an unstripped, source-preserving fast decode."""
    ends = []
    offset = 0
    for sentence in sentences:
        offset += len(sentence)
        ends.append(offset)
    spans = owned_sentence_spans(text, ends, force_newlines=force_newlines)
    pieces = [text[a:b] for a, b in spans]
    return [piece.strip() for piece in pieces if piece.strip()] if strip_whitespace else pieces


def validate_input_line_breaks(mode: str) -> str:
    if mode not in INPUT_LINE_BREAK_MODES:
        raise ValueError(f"input_line_breaks must be one of {INPUT_LINE_BREAK_MODES}, got {mode!r}")
    return mode


def normalize_input_line_breaks(text: str, mode: str = "space") -> str:
    """Return the text the encoder sees; always the same length as ``text``.

    ``"space"`` maps each LF (U+000A) and CR (U+000D) to one U+0020, so CRLF becomes two
    spaces. Other Unicode separators are unchanged. Offsets into the result are offsets
    into the original text.
    """
    if mode == "preserve":
        return text
    validate_input_line_breaks(mode)
    normalized = text.translate(_LINE_BREAKS_TO_SPACES)
    if len(normalized) != len(text):  # unreachable by construction; guards the offset contract
        raise AssertionError("input line-break normalization must preserve the character count")
    return normalized


def resolve_text_input(
    text_or_texts: str | Iterable[str],
    iterator_factory: Callable[[list[str]], Iterator[T]],
    *,
    lazy: bool,
) -> T | list[T] | Iterator[T]:
    """Apply consistent eager single/batch semantics around an iterator."""
    if isinstance(text_or_texts, str):
        return next(iterator_factory([text_or_texts]))

    results = iterator_factory(list(text_or_texts))
    return results if lazy else list(results)


def iter_probabilities(
    texts: list[str],
    *,
    outer_batch_size: int,
    remove_whitespace_before_inference: bool,
    extract_logits: Callable[..., list[FloatArray]],
    probability_fn: Callable[[FloatArray], ProbabilityPair],
    return_paragraph_probabilities: bool,
    outputs_are_probabilities: bool = False,
    prepare_batch: Callable[[list[str]], Any] | None = None,
) -> Iterator[FloatArray | ProbabilityPair]:
    """Run shared outer batching and restore spaces removed before inference."""
    from concurrent.futures import Future, ThreadPoolExecutor

    n_outer_batches = math.ceil(len(texts) / outer_batch_size)
    empty_sentinel = np.zeros((1, 1), dtype=np.float32) if outputs_are_probabilities else np.full((1, 1), -np.inf)

    def _prepare_outer(
        outer_batch_idx: int,
    ) -> tuple[list[list[int]], list[int], list[str], Any]:
        start = outer_batch_idx * outer_batch_size
        end = min((outer_batch_idx + 1) * outer_batch_size, len(texts))
        outer_batch_texts = texts[start:end]
        input_texts: list[str] = []
        space_positions: list[list[int]] = []

        for text in outer_batch_texts:
            if remove_whitespace_before_inference:
                positions: list[int] = []
                input_characters: list[str] = []
                for character in text:
                    if character == " ":
                        positions.append(len(input_characters) + len(positions))
                    else:
                        input_characters.append(character)
                input_text = "".join(input_characters)
                space_positions.append(positions)
            else:
                input_text = text
            input_texts.append(input_text)

        empty_string_indices = [i for i, text in enumerate(input_texts) if not text.strip()]
        nonempty_texts = [text for text in input_texts if text.strip()]
        prepared = prepare_batch(nonempty_texts) if (prepare_batch is not None and nonempty_texts) else None
        return space_positions, empty_string_indices, nonempty_texts, prepared

    use_prefetch = prepare_batch is not None and n_outer_batches > 1
    executor = ThreadPoolExecutor(max_workers=1) if use_prefetch else None
    try:
        next_future: Future[tuple[list[list[int]], list[int], list[str], Any]] | None = None
        if executor is not None and n_outer_batches > 0:
            next_future = executor.submit(_prepare_outer, 0)

        for outer_batch_idx in range(n_outer_batches):
            if next_future is not None:
                space_positions, empty_string_indices, nonempty_texts, prepared = next_future.result()
                if executor is not None and (outer_batch_idx + 1) < n_outer_batches:
                    next_future = executor.submit(_prepare_outer, outer_batch_idx + 1)
                else:
                    next_future = None
            else:
                space_positions, empty_string_indices, nonempty_texts, prepared = _prepare_outer(outer_batch_idx)

            if nonempty_texts:
                if prepare_batch is not None:
                    outer_batch_logits = extract_logits(nonempty_texts, prepared)
                else:
                    outer_batch_logits = extract_logits(nonempty_texts)
            else:
                outer_batch_logits = []

            for i in empty_string_indices:
                outer_batch_logits.insert(i, empty_sentinel)

            for i, logits in enumerate(outer_batch_logits):
                sentence_probs, newline_probs = probability_fn(logits)

                if remove_whitespace_before_inference:
                    full_newline_probs = list(newline_probs)
                    full_sentence_probs = list(sentence_probs)
                    for position in space_positions[i]:
                        full_newline_probs.insert(position, np.zeros_like(newline_probs[0]))
                        full_sentence_probs.insert(position, np.zeros_like(sentence_probs[0]))
                    newline_probs = np.asarray(full_newline_probs)
                    sentence_probs = np.asarray(full_sentence_probs)

                if return_paragraph_probabilities:
                    yield sentence_probs, newline_probs
                else:
                    yield sentence_probs
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def validate_segmentation_options(
    *,
    threshold: float | None,
    max_length: int | None,
    length_fn: LengthFunction = len,
    split_on_input_newlines: bool | None,
) -> None:
    """Validate the optional hard cap; thresholds retain ordinary meaning."""
    if max_length is not None:
        resolve_budget(max_length, length_fn)
    elif length_fn is not len:
        raise TypeError(
            "length_fn requires max_length; use split(max_length=N, length_fn=...) or segment(max_length=N, length_fn=...)"
        )


@dataclass
class DecodedText:
    sentences: list[str]
    paragraphs: list[list[str]] | None = None


def _decode_unit(
    text: str,
    probabilities: FloatArray,
    *,
    sentence_threshold: float,
    strip_whitespace: bool,
    max_length: int | None,
    length_fn: LengthFunction,
    language: str | None,
    split_on_input_newlines: bool | None = None,
) -> list[str]:
    if not text:
        return []
    if not text.strip():
        probabilities = np.zeros(len(text))
    if max_length is None:
        sentences = indices_to_sentences(text, np.where(probabilities > sentence_threshold)[0], strip_whitespace=False)
        ends = np.cumsum([len(sentence) for sentence in sentences]).tolist()
    else:
        if strip_whitespace:
            raise ValueError("strip_whitespace is incompatible with a reconstructing hard cap")
        ends = [b for _, b in source_spans(text, (np.where(probabilities > sentence_threshold)[0] + 1).tolist())]
    spans = owned_sentence_spans(text, ends, force_newlines=bool(split_on_input_newlines))
    sentences = [text[a:b] for a, b in spans]
    if strip_whitespace:
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if max_length is None:
        return sentences
    output = []
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        end = start + len(sentence)
        cursor = end
        if measure(sentence, length_fn) <= max_length:
            output.append(sentence)
        else:
            try:
                output.extend(
                    sentence[a:b]
                    for a, b in split_spans(
                        sentence,
                        probabilities[start:end],
                        max_length=max_length,
                        length_fn=length_fn,
                        threshold=sentence_threshold,
                    )
                )
            except UnsplittableUnitError as error:
                raise UnsplittableUnitError(start + error.position, error.measured_length, max_length) from error
    return output


def decode_probabilities(
    text: str,
    probabilities: FloatArray | ProbabilityPair,
    *,
    sentence_threshold: float,
    paragraph_threshold: float,
    strip_whitespace: bool,
    do_paragraph_segmentation: bool,
    split_on_input_newlines: bool | None,
    max_length: int | None,
    language: str | None,
    length_fn: LengthFunction = len,
) -> DecodedText:
    """Decode model probabilities without depending on a model implementation."""
    if not text.strip():
        zeros = np.zeros(len(text))
        probabilities = (zeros, zeros) if do_paragraph_segmentation else zeros
    if do_paragraph_segmentation:
        sentence_probs, newline_probs = cast(ProbabilityPair, probabilities)
        offset = 0
        paragraphs: list[list[str]] = []
        paragraph_ends = (np.where(newline_probs > paragraph_threshold)[0] + 1).tolist()
        if max_length is not None:
            paragraph_ends = [b for _, b in source_spans(text, paragraph_ends)]
        paragraph_texts = [text[a:b] for a, b in owned_sentence_spans(text, paragraph_ends, force_newlines=False)]
        for paragraph in paragraph_texts:
            paragraph_probs = sentence_probs[offset : offset + len(paragraph)]
            try:
                paragraphs.append(
                    _decode_unit(
                        paragraph,
                        paragraph_probs,
                        sentence_threshold=sentence_threshold,
                        strip_whitespace=strip_whitespace,
                        max_length=max_length,
                        length_fn=length_fn,
                        language=language,
                        split_on_input_newlines=None,
                    )
                )
            except UnsplittableUnitError as error:
                raise UnsplittableUnitError(offset + error.position, error.measured_length, max_length or 0) from error
            offset += len(paragraph)
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        return DecodedText(
            sentences=[sentence for paragraph in paragraphs for sentence in paragraph],
            paragraphs=paragraphs,
        )

    sentence_probs = cast(FloatArray, probabilities)
    sentences = _decode_unit(
        text,
        sentence_probs,
        sentence_threshold=sentence_threshold,
        strip_whitespace=strip_whitespace,
        max_length=max_length,
        length_fn=length_fn,
        language=language,
        split_on_input_newlines=split_on_input_newlines,
    )

    if split_on_input_newlines is False:
        warnings.warn(
            "split_on_input_newlines=False disables forced splitting at input newlines. "
            "This option does not replace newlines with spaces; the model may still "
            "predict boundaries around them. Returned segments may contain input newlines.",
            UserWarning,
            stacklevel=2,
        )
    return DecodedText(sentences=sentences)
