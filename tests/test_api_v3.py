import inspect

import numpy as np
import pytest

from wtpsplit import DEFAULT_STRIDE, SaT, Segmentation
from wtpsplit.utils import indices_to_sentences


@pytest.fixture(scope="module")
def sat():
    return SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])


def test_stride_default_is_consistent():
    assert inspect.signature(SaT.predict_proba).parameters["stride"].default == DEFAULT_STRIDE
    assert inspect.signature(SaT.segment).parameters["stride"].default == DEFAULT_STRIDE
    assert inspect.signature(SaT.split).parameters["stride"].default == DEFAULT_STRIDE


def test_segment_returns_text_spans_and_probabilities(sat):
    text = "This is a test sentence This is another test sentence."
    result = sat.segment(text, threshold=0.25)

    assert isinstance(result, Segmentation)
    assert result.sentences == ["This is a test sentence ", "This is another test sentence."]
    assert [text[start:end] for start, end in result.spans] == result.sentences
    assert result.probabilities.shape == (len(text),)
    assert result.confidences == [float(result.probabilities[end - 1]) for start, end in result.spans]
    assert sat.split(text, threshold=0.25) == result.sentences


def test_batch_apis_are_eager_by_default_with_lazy_opt_in(sat):
    texts = ["One sentence.", "Another sentence."]
    results = sat.segment(texts)

    assert isinstance(results, list)
    assert [result.text for result in results] == texts
    assert all(isinstance(result, Segmentation) for result in results)
    assert isinstance(sat.predict_proba(texts), list)
    assert isinstance(sat.split(texts), list)

    lazy_results = sat.segment(texts, lazy=True)
    assert not isinstance(lazy_results, list)
    assert [result.text for result in lazy_results] == texts


def test_default_split_matches_thresholded_probabilities(sat):
    text = "This is a test sentence This is another test sentence."
    probabilities = sat.predict_proba(text)
    manually_split = indices_to_sentences(text, np.where(probabilities > 0.25)[0])

    assert sat.split(text) == manually_split


@pytest.mark.parametrize(
    "text",
    [
        "שלום עולם. שלום עולם.",
        "Cafe\u0301 is written with a combining mark. Cafe\u0301 again.",
    ],
)
def test_segment_spans_round_trip_unicode_and_repeated_text(sat, text):
    result = sat.segment(text)

    assert [text[start:end] for start, end in result.spans] == result.sentences


def test_segment_spans_work_with_constraints_and_paragraphs(sat):
    constrained_text = "One long sentence that must be divided into several smaller segments."
    constrained = sat.segment(constrained_text, max_length=16)
    assert [constrained_text[start:end] for start, end in constrained.spans] == constrained.sentences

    paragraph_text = "First paragraph sentence.\nSecond paragraph sentence."
    paragraphs = sat.segment(paragraph_text, do_paragraph_segmentation=True)
    assert paragraphs.paragraphs is not None
    assert paragraphs.paragraph_spans is not None
    assert [paragraph_text[start:end] for start, end in paragraphs.spans] == paragraphs.sentences
