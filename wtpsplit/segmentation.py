"""Structured sentence-segmentation results."""

from dataclasses import dataclass

import numpy as np


Span = tuple[int, int]


@dataclass(frozen=True)
class Segmentation:
    """Sentence boundaries and their source-text offsets.

    Spans are half-open ``(start, end)`` character offsets. Each confidence is
    the boundary probability at the final character of the corresponding span.
    ``paragraphs`` and ``paragraph_spans`` are populated only when paragraph
    segmentation is requested; ``sentences`` and ``spans`` are always flat.
    """

    text: str
    sentences: list[str]
    spans: list[Span]
    probabilities: np.ndarray
    confidences: list[float]
    paragraphs: list[list[str]] | None = None
    paragraph_spans: list[list[Span]] | None = None
    paragraph_probabilities: np.ndarray | None = None


def sentence_spans(text: str, sentences: list[str]) -> list[Span]:
    """Locate sentence strings sequentially in their source text."""
    spans = []
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            raise ValueError("Could not map a segmented sentence back to its source text.")
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def boundary_confidences(probabilities: np.ndarray, spans: list[Span]) -> list[float]:
    """Return the model boundary probability at each sentence's final character.

    This includes the final sentence: its score describes how strongly the
    model supports a boundary at the end of the supplied document.
    """
    if probabilities.size == 0:
        return [0.0] * len(spans)
    final_probability_index = len(probabilities) - 1
    return [
        float(probabilities[min(end - 1, final_probability_index)]) if end > start else 0.0
        for start, end in spans
    ]
