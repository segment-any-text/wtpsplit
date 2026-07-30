"""Shared, model-agnostic inference helpers for SaT and legacy WtP.

Model classes own checkpoint loading and conversion from logits to boundary
probabilities. This module owns input dispatch, batching, whitespace
restoration, option validation, and probability-to-segment decoding.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from wtpsplit.utils import indices_to_sentences
from wtpsplit.utils.constraints import _enforce_segment_constraints, constrained_segmentation
from wtpsplit.utils.priors import create_prior_function

T = TypeVar("T")
ProbabilityPair = tuple[np.ndarray, np.ndarray]


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
    extract_logits: Callable[[list[str]], list[np.ndarray]],
    probability_fn: Callable[[np.ndarray], ProbabilityPair],
    return_paragraph_probabilities: bool,
) -> Iterator[np.ndarray | ProbabilityPair]:
    """Run shared outer batching and restore spaces removed before inference."""
    n_outer_batches = math.ceil(len(texts) / outer_batch_size)

    for outer_batch_idx in range(n_outer_batches):
        start = outer_batch_idx * outer_batch_size
        end = min((outer_batch_idx + 1) * outer_batch_size, len(texts))
        outer_batch_texts = texts[start:end]
        input_texts: list[str] = []
        space_positions: list[list[int]] = []

        for text in outer_batch_texts:
            if remove_whitespace_before_inference:
                positions = []
                input_characters = []
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
        outer_batch_logits = extract_logits(nonempty_texts) if nonempty_texts else []

        for i in empty_string_indices:
            outer_batch_logits.insert(i, np.full((1, 1), -np.inf))

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


def validate_segmentation_options(
    *,
    threshold: float | None,
    min_length: int,
    max_length: int | None,
    prior_type: str,
    algorithm: str,
    split_on_input_newlines: bool | None,
) -> None:
    """Validate segmentation options shared by SaT and WtP."""
    if max_length is not None and min_length > max_length:
        raise ValueError(f"min_length ({min_length}) cannot be greater than max_length ({max_length})")
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")
    if max_length is not None and max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")

    valid_priors = ["uniform", "gaussian", "clipped_polynomial", "lognormal"]
    if prior_type not in valid_priors:
        raise ValueError(f"Unknown prior_type: '{prior_type}'. Must be one of {valid_priors}")
    valid_algorithms = ["viterbi", "greedy"]
    if algorithm not in valid_algorithms:
        raise ValueError(f"Unknown algorithm: '{algorithm}'. Must be one of {valid_algorithms}")

    if max_length is not None and threshold is not None:
        warnings.warn(
            "Both 'threshold' and 'max_length' are set. When using length-constrained "
            "segmentation (max_length), the threshold parameter is ignored.",
            UserWarning,
            stacklevel=2,
        )

    if (max_length is not None or min_length > 1) and split_on_input_newlines:
        warnings.warn(
            "When using length constraints (max_length/min_length), segments may contain newlines. "
            "split_on_input_newlines is ignored; use ''.join(segments) to reconstruct the original text. "
            "To split at newlines with constraints, pre-split your text at newlines and process each line.",
            UserWarning,
            stacklevel=2,
        )


@dataclass
class DecodedText:
    sentences: list[str]
    paragraphs: list[list[str]] | None = None


def _decode_unit(
    text: str,
    probabilities: np.ndarray,
    *,
    sentence_threshold: float,
    strip_whitespace: bool,
    min_length: int,
    max_length: int | None,
    prior_type: str,
    prior_kwargs: dict[str, Any] | None,
    algorithm: str,
    language: str | None,
    use_negative_evidence: bool = True,
) -> list[str]:
    if max_length is None and min_length <= 1:
        return indices_to_sentences(
            text,
            np.where(probabilities > sentence_threshold)[0],
            strip_whitespace=strip_whitespace,
        )

    local_prior_kwargs = {} if prior_kwargs is None else prior_kwargs.copy()
    if max_length is not None:
        local_prior_kwargs["max_length"] = max_length
    if language and "lang_code" not in local_prior_kwargs and "target_length" not in local_prior_kwargs:
        local_prior_kwargs["lang_code"] = language

    prior_fn = create_prior_function(prior_type, local_prior_kwargs)
    boundaries = constrained_segmentation(
        probabilities,
        prior_fn,
        min_length=min_length,
        max_length=max_length,
        algorithm=algorithm,
        use_negative_evidence=use_negative_evidence,
    )
    indices = [boundary - 1 for boundary in boundaries]
    return _enforce_segment_constraints(
        text,
        indices,
        min_length,
        max_length,
        strip_whitespace=strip_whitespace,
    )


def decode_probabilities(
    text: str,
    probabilities: np.ndarray | ProbabilityPair,
    *,
    sentence_threshold: float,
    paragraph_threshold: float,
    strip_whitespace: bool,
    do_paragraph_segmentation: bool,
    split_on_input_newlines: bool | None,
    min_length: int,
    max_length: int | None,
    prior_type: str,
    prior_kwargs: dict[str, Any] | None,
    algorithm: str,
    language: str | None,
    use_negative_evidence: bool = True,
) -> DecodedText:
    """Decode model probabilities without depending on a model implementation."""
    if do_paragraph_segmentation:
        sentence_probs, newline_probs = probabilities
        offset = 0
        paragraphs = []
        for paragraph in indices_to_sentences(text, np.where(newline_probs > paragraph_threshold)[0]):
            paragraph_probs = sentence_probs[offset : offset + len(paragraph)]
            paragraphs.append(
                _decode_unit(
                    paragraph,
                    paragraph_probs,
                    sentence_threshold=sentence_threshold,
                    strip_whitespace=strip_whitespace,
                    min_length=min_length,
                    max_length=max_length,
                    prior_type=prior_type,
                    prior_kwargs=prior_kwargs,
                    algorithm=algorithm,
                    language=language,
                    use_negative_evidence=use_negative_evidence,
                )
            )
            offset += len(paragraph)
        return DecodedText(
            sentences=[sentence for paragraph in paragraphs for sentence in paragraph],
            paragraphs=paragraphs,
        )

    sentence_probs = probabilities
    sentences = _decode_unit(
        text,
        sentence_probs,
        sentence_threshold=sentence_threshold,
        strip_whitespace=strip_whitespace,
        min_length=min_length,
        max_length=max_length,
        prior_type=prior_type,
        prior_kwargs=prior_kwargs,
        algorithm=algorithm,
        language=language,
        use_negative_evidence=use_negative_evidence,
    )

    if max_length is None and min_length <= 1:
        if split_on_input_newlines:
            newline_split_sentences = []
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 1 and sentence.endswith("\n"):
                    sentence = sentence[:-1]
                newline_split_sentences.extend(sentence.split("\n"))
            sentences = newline_split_sentences
        elif split_on_input_newlines is False:
            warnings.warn(
                "split_on_input_newlines=False will lead to newlines in the output "
                "if they were present in the input. Within the model, such newlines are "
                "treated as spaces. If you want to split on such newlines, "
                "set split_on_input_newlines=True.",
                UserWarning,
                stacklevel=2,
            )

    return DecodedText(sentences=sentences)
