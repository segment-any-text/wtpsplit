"""Offline contract checks for the v3 length-aware API."""

import time

import numpy as np
import pytest

from wtpsplit import SaT, WtP
from length_reference import oracle
from wtpsplit.chunking import (
    TokenLength,
    UnsplittableUnitError,
    chunk_spans,
    grapheme_ends,
    resolve_budget,
    source_spans,
    split_spans,
)


def test_seeded_exhaustive_objective():
    rng = np.random.default_rng(112611)
    atoms = ["a", " ", "\n", "\r\n", "e\u0301", "字", "👩\u200d💻"]
    for _ in range(350):
        text = "".join(rng.choice(atoms, int(rng.integers(1, 7))))
        p = rng.choice([0, 0.1, 0.25, 0.5, 0.9, 1], len(text))
        cap = int(rng.integers(1, 8))
        threshold = float(rng.choice([0.25, 0.5, 0.8]))
        expected = oracle(text, p, cap, threshold=threshold)
        if expected is None:
            with pytest.raises(UnsplittableUnitError):
                split_spans(text, p, max_length=cap, threshold=threshold)
        else:
            assert split_spans(text, p, max_length=cap, threshold=threshold) == expected


def test_seeded_unicode_properties():
    rng = np.random.default_rng(18)
    atoms = ["a", " ", "  ", "\n", "e\u0301", "👩\u200d💻", "中", "ไทย", "\t"]
    for _ in range(150):
        text = "".join(rng.choice(atoms, int(rng.integers(0, 55))))
        cap = int(rng.integers(4, 18))
        p = rng.random(len(text))
        spans = split_spans(text, p, max_length=cap)
        assert "".join(text[a:b] for a, b in spans) == text
        legal = {0, *grapheme_ends(text)}
        assert all(a in legal and b in legal and 0 < b - a <= cap for a, b in spans)
        units = source_spans(text, [i for i in range(1, len(text)) if rng.random() < 0.2])
        packed = chunk_spans(text, p, units, max_length=cap)
        assert "".join(text[a:b] for a, b in packed) == text
        for overlap in [0, 1, 3]:
            chunks = chunk_spans(text, p, units, max_length=cap, overlap_sentences=overlap)
            previous = 0
            for a, b in chunks:
                assert b > previous and b - a <= cap
                if a < previous:
                    repeated = [(x, y) for x, y in units if a <= x < y <= previous]
                    assert len(repeated) <= overlap
                    assert repeated[0][0] == a and repeated[-1][1] == previous
                previous = b
            assert previous == len(text)


def test_next_fit_and_overlap():
    text = "aa bb cc dd "
    units = [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert chunk_spans(text, np.zeros(12), units, max_length=9) == [(0, 9), (9, 12)]
    assert chunk_spans(text, np.zeros(12), units, max_length=9, overlap_sentences=2) == [(0, 9), (3, 12)]
    assert split_spans("", [], max_length=1) == []
    with pytest.raises(UnsplittableUnitError, match="position 0: measured length 3"):
        split_spans("👩\u200d💻", [0.1] * 3, max_length=2)


class MergingTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text.replace("ab", "X"))


def test_nonadditive_tokenizer():
    cap, count = resolve_budget(2, TokenLength(MergingTokenizer()))
    assert count("ab") < count("a") + count("b")
    text = "abababc"
    spans = chunk_spans(text, np.zeros(len(text)), [(0, len(text))], max_length=cap, length_fn=count)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(count(text[a:b]) <= cap for a, b in spans)


def test_cached_tokenizer():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base", local_files_only=True)
    except OSError:
        pytest.skip("Tokenizer is unavailable in offline cache")
    cap, count = resolve_budget(8, TokenLength(tok))
    text = "Some text. More text. Unicode e\u0301."
    spans = split_spans(text, np.zeros(len(text)), max_length=cap, length_fn=count)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(count(text[a:b]) <= cap for a, b in spans)


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_invalid_cap(cap):
    with pytest.raises(ValueError, match="positive integer"):
        split_spans("a", [0.1], max_length=cap)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_counter(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        chunk_spans("a", [0.1], [(0, 1)], max_length=1, length_fn=lambda _: value)


@pytest.mark.parametrize("p", [[], [0.1, 0.2], [np.nan], [np.inf], [-0.1], [1.1]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError, match="probabilities"):
        split_spans("a", p, max_length=1)


def test_invalid_token_options():
    for kwargs in [{"reserve_tokens": -1}, {"reserve_tokens": True}, {"add_special_tokens": None}]:
        with pytest.raises((TypeError, ValueError)):
            TokenLength(MergingTokenizer(), **kwargs)


@pytest.mark.parametrize("cls,method", [(SaT, "split"), (SaT, "segment"), (WtP, "split")])
@pytest.mark.parametrize("keyword", ["min_length", "prior_type", "prior_kwargs", "algorithm", "use_negative_evidence"])
def test_retired_keywords(cls, method, keyword):
    with pytest.raises(TypeError, match="removed in 3.0;"):
        getattr(object.__new__(cls), method)("a", **{keyword: None})


def test_public_model_free_integration():
    sat = object.__new__(SaT)
    sat._resolve_sentence_threshold = lambda threshold: 0.5
    sat.predict_proba = lambda text, **kw: (np.zeros(len(text)), np.zeros(len(text)))
    text = "abc def ghi "
    assert "".join(sat.chunk(text, max_length=5)) == text
    spans = sat.chunk(text, max_length=5, return_spans=True)
    assert "".join(text[a:b] for a, b in spans) == text
    from wtpsplit._inference import decode_probabilities

    result = decode_probabilities(
        text,
        np.zeros(len(text)),
        sentence_threshold=0.5,
        paragraph_threshold=0.5,
        strip_whitespace=False,
        do_paragraph_segmentation=False,
        split_on_input_newlines=False,
        max_length=5,
        length_fn=len,
        language=None,
    )
    assert "".join(result.sentences) == text
    assert all(len(s) <= 5 for s in result.sentences)


def test_decode_performance():
    text = "字" * 50_000
    start = time.perf_counter()
    spans = split_spans(text, np.full(len(text), 0.01), max_length=2000)
    elapsed = time.perf_counter() - start
    print(f"50k/2000 decode: {elapsed:.6f}s")
    assert elapsed < 1
    assert "".join(text[a:b] for a, b in spans) == text


def test_additive_callback_oracle():
    rng = np.random.default_rng(32)

    def count(s):
        return len(s.encode("utf-8"))

    for _ in range(70):
        text = "".join(rng.choice(["a", "字", " ", "\r\n", "e\u0301"], 5))
        cap = int(rng.integers(4, 10))
        p = rng.uniform(0.01, 0.99, len(text))
        expected = oracle(text, p, cap, count=count)
        if expected is None:
            with pytest.raises(UnsplittableUnitError):
                split_spans(text, p, max_length=cap, length_fn=count)
        else:
            assert split_spans(text, p, max_length=cap, length_fn=count) == expected


def test_invalid_positions_and_spans():
    with pytest.raises(ValueError, match="grapheme"):
        split_spans("e\u0301", [0.1, 0.2], max_length=2, legal_positions=[1])
    for units in [[(1, 2)], [(0, 1)]]:
        with pytest.raises(ValueError, match="partition"):
            chunk_spans("ab", [0.1, 0.2], units, max_length=1)
    with pytest.raises(ValueError, match="overlap_sentences"):
        chunk_spans("", [], [], max_length=1, overlap_sentences=-1)


def test_explicit_lattice_and_tie_rule():
    assert split_spans("abcd", [0.5] * 4, max_length=3, legal_positions=[1, 2, 3]) == [(0, 3), (3, 4)]
    assert split_spans("abcd", [0.99] * 4, max_length=3, legal_positions=[2]) == [(0, 2), (2, 4)]


@pytest.mark.parametrize("method", ["split", "segment"])
def test_sat_public_dispatch_without_model(method):
    from types import SimpleNamespace

    sat = object.__new__(SaT)
    sat.model = SimpleNamespace(config=SimpleNamespace(use_character_head=False, model_type="xlm-roberta"))
    sat.tokenizer = None
    sat.language = None
    sat._resolve_sentence_threshold = lambda _: 0.5
    sat.predict_proba = lambda texts, **kw: (np.zeros(len(t)) for t in texts)
    for lazy in [False, True]:
        out = list(getattr(sat, method)(["ab cd ef", "hello"], max_length=3, lazy=lazy))
        for original, value in zip(["ab cd ef", "hello"], out):
            strings = value.sentences if method == "segment" else value
            assert "".join(strings) == original
            assert all(len(s) <= 3 for s in strings)


def test_wtp_public_dispatch_without_model():
    wtp = object.__new__(WtP)
    wtp.language = None
    wtp._resolve_mixture = lambda *args: None
    wtp.predict_proba = lambda texts, **kw: (np.zeros(len(t)) for t in texts)
    assert "".join(wtp.split("abc def", max_length=3)) == "abc def"


def test_paragraph_graphemes_and_error_offset():
    from wtpsplit._inference import decode_probabilities

    text = "a 👩\u200d💻"
    p = np.zeros(len(text))
    paragraphs = np.zeros(len(text))
    paragraphs[1] = 0.9
    with pytest.raises(UnsplittableUnitError) as caught:
        decode_probabilities(
            text,
            (p, paragraphs),
            sentence_threshold=0.5,
            paragraph_threshold=0.5,
            strip_whitespace=False,
            do_paragraph_segmentation=True,
            split_on_input_newlines=False,
            max_length=2,
            language=None,
        )
    assert caught.value.position == 2
