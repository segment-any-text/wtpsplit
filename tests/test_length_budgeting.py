"""Source budgets, overlap, Unicode and cached-model integration regressions."""

import inspect
import time
from pathlib import Path

from length_reference import oracle

import numpy as np
import pytest

from wtpsplit import SaT, TokenLength
from wtpsplit.chunking import UnsplittableUnitError, chunk_spans, resolve_budget, source_spans, split_spans


def test_evidence_and_final_run():
    text = "aaaa bbbb cccc"
    p = np.zeros(len(text))
    p[3] = 0.9
    p[8] = 0.1
    p[11] = 0.99  # Inside a fitting final word: not a legal cut.
    assert split_spans(text, p, max_length=12) == [(0, 5), (5, 14)]
    text = "aaaa bbbb cccc dddd "
    p = np.zeros(len(text))
    p[8] = 0.8
    assert split_spans(text, p, max_length=12) == [(0, 10), (10, 20)]


def test_threshold_and_tolerance():
    text = "a" * 40
    p = np.full(len(text), 0.6)
    shifted = split_spans(text, p, max_length=20, threshold=0.8)
    assert len(shifted) == 2
    assert len(split_spans(text, p, max_length=20, threshold=0.5)) > 2
    p = np.full(4, 0.5)
    p[0] = 0.5 + 1e-11
    assert split_spans("abcd", p, max_length=3, legal_positions=[1, 2, 3]) == [(0, 3), (3, 4)]


@pytest.mark.parametrize("text,cap", [("   ", 2), ("\r\n\r\n", 3), ("", 2)])
def test_whitespace_public(text, cap):
    from types import SimpleNamespace

    sat = object.__new__(SaT)
    sat.model = SimpleNamespace(config=SimpleNamespace(use_character_head=False, model_type="xlm-roberta"))
    sat.tokenizer = None
    sat.language = None
    sat._resolve_sentence_threshold = lambda _: 0.25

    def predict(texts, **kwargs):
        if isinstance(texts, str):
            return (np.zeros(1), np.zeros(1))
        return (np.zeros(1) for _ in texts)

    sat.predict_proba = predict
    outputs = [
        sat.chunk(text, max_length=cap),
        sat.split(text, max_length=cap, split_on_input_newlines=False),
        [text[a:b] for a, b in split_spans(text, [0], max_length=cap)],
    ]
    for output in outputs:
        assert "".join(output) == text
        assert all(len(piece) <= cap for piece in output)


def test_whitespace_impossible():
    with pytest.raises(UnsplittableUnitError) as caught:
        split_spans("\r\n", [0, 0], max_length=1)
    assert caught.value.position == 0 and caught.value.measured_length == 2


def test_generic_call_bound():
    calls = 0

    def count(text):
        nonlocal calls
        calls += 1
        return len(text) + len(text) // 3

    text = "x" * 20_000
    spans = split_spans(text, np.zeros(len(text)), max_length=100, length_fn=count)
    bound = 4 * len(text) + 2
    assert calls <= bound
    assert all(len(text[a:b]) + len(text[a:b]) // 3 <= 100 for a, b in spans)
    assert max(b - a for a, b in spans) == 75


def test_scale_performance():
    text = ("A sentence of about sixty characters in source order. " * 10_000)[:500_000]
    start = time.perf_counter()
    spans = source_spans(text, list(range(52, len(text), 53)))
    source_time = time.perf_counter() - start
    from wtpsplit._inference import decode_probabilities

    p = np.zeros(len(text))
    p[np.arange(52, len(text), 53)] = 0.9
    start = time.perf_counter()
    decoded = decode_probabilities(
        text,
        p,
        sentence_threshold=0.5,
        paragraph_threshold=0.5,
        strip_whitespace=False,
        do_paragraph_segmentation=False,
        split_on_input_newlines=False,
        max_length=2000,
        language=None,
    )
    split_time = time.perf_counter() - start
    start = time.perf_counter()
    dense = split_spans("字" * 500_000, np.full(500_000, 0.01), max_length=2000)
    dense_time = time.perf_counter() - start
    print(f"500k source={source_time:.6f}s capped_split={split_time:.6f}s dense={dense_time:.6f}s")
    assert source_time < 2 and split_time < 2 and dense_time < 2
    assert "".join(decoded.sentences) == text and spans and dense


def test_cached_tokenizer_fill():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base", local_files_only=True)
    cap, count = resolve_budget(64, TokenLength(tok))
    text = (
        "The committee met on Monday and after a long debate it approved the budget and discussed the next steps " * 45
    )
    spans = split_spans(text, np.zeros(len(text)), max_length=cap, length_fn=count)
    # This ASCII fixture has one space per whole-word legal boundary. Verify
    # independently that standalone word token IDs concatenate to the whole
    # encoding, then compute the exact positive-additive whole-word minimum.
    import re

    words = re.findall(r"\S+\s*", text)
    ids = [tok.encode(word, add_special_tokens=False, verbose=False) for word in words]
    assert [token for word in ids for token in word] == tok.encode(text, add_special_tokens=False, verbose=False)
    minimum = 1
    used = 0
    overhead = len(tok.encode("", add_special_tokens=True, verbose=False))
    for word in ids:
        if used + len(word) + overhead > 64:
            minimum += 1
            used = 0
        used += len(word)
    print(f"XLMR fill: pieces={len(spans)}, whole-word minimum={minimum}")
    assert len(spans) <= 1.25 * minimum
    assert all(count(text[a:b]) <= 64 for a, b in spans)
    assert "".join(text[a:b] for a, b in spans) == text


def test_specials_reserve_and_early_validation():
    class Specials:
        def encode(self, text, *, add_special_tokens=True):
            return list(text) + ([0, 0] if add_special_tokens else [])

    cap, count = resolve_budget(6, TokenLength(Specials(), reserve_tokens=1))
    spans = split_spans("abcdefgh", np.zeros(8), max_length=cap, length_fn=count)
    assert all(count("abcdefgh"[a:b]) <= 6 for a, b in spans)
    sat = object.__new__(SaT)
    sat.predict_proba = lambda *a, **kw: pytest.fail("inference must not run")
    for options in [
        {"overlap_sentences": -1},
        {"unit": "invalid"},
        {"min_length": 1},
        {"return_paragraph_probabilities": True},
    ]:
        with pytest.raises((ValueError, TypeError)):
            sat.chunk("abc", max_length=3, **options)
    with pytest.raises(ValueError, match="max_length.*0"):
        resolve_budget(0, TokenLength(Specials()))


@pytest.mark.slow
def test_cached_model_evidence():
    import torch

    torch.set_num_threads(1)
    from huggingface_hub import hf_hub_download

    model_path = str(Path(hf_hub_download("segment-any-text/sat-3l-sm", "model.safetensors", local_files_only=True)).parent)
    tokenizer_path = str(Path(hf_hub_download("facebookAI/xlm-roberta-base", "tokenizer.json", local_files_only=True)).parent)
    sat = SaT(
        model_path,
        tokenizer_name_or_path=tokenizer_path,
        device="cpu",
        from_pretrained_kwargs={"local_files_only": True},
    )
    text = "The committee met on Monday, and after a long debate it approved the budget. "
    p = sat.predict_proba(text)
    assert all(p[i] == 0 for i, c in enumerate(text) if c.isspace())
    expected = oracle(text, p, 55)
    spans = split_spans(text, p, max_length=55)
    assert spans == expected
    assert sat.split(text, max_length=55, threshold=0.25) == [
        text[a:b] for a, b in split_spans(text, p, max_length=55, threshold=0.25)
    ]


def test_packing_counts_linear():
    calls = 0

    def count(text):
        nonlocal calls
        calls += 1
        return len(text)

    text = "unit " * 1000
    units = [(i, i + 5) for i in range(0, len(text), 5)]
    spans = chunk_spans(text, np.zeros(len(text)), units, max_length=5, length_fn=count)
    assert len(spans) == 1000 and calls <= 2 * len(units)


def test_typed_surface_and_hints():
    import wtpsplit
    from typing import get_overloads

    assert {"split_spans", "chunk_spans", "UnsplittableUnitError"} <= set(wtpsplit.__all__)
    assert len(get_overloads(SaT.chunk)) >= 2
    assert "removed_options" not in inspect.signature(SaT.split).parameters
    for key, fragment in [
        ("min_length", "no minimum-length replacement"),
        ("prior_type", "no prior replacement"),
        ("prior_kwargs", "no prior-parameter replacement"),
        ("algorithm", "shared decoder"),
        ("use_negative_evidence", "fixed objective"),
    ]:
        with pytest.raises(TypeError, match=fragment):
            object.__new__(SaT).split("a", **{key: None})
    root = Path(__file__).resolve().parents[1]
    assert '"regex>=' in (root / "pyproject.toml").read_text()


def test_source_span_index_built_once(monkeypatch):
    original = np.searchsorted

    def search(array, *args, **kwargs):
        assert isinstance(array, np.ndarray), "do not reconvert the grapheme list per boundary"
        return original(array, *args, **kwargs)

    monkeypatch.setattr(np, "searchsorted", search)
    text = "one sentence. " * 100
    assert source_spans(text, list(range(13, len(text), 14)))


def test_longest_prefix_tie():
    assert split_spans("abcd", [0.5] * 4, max_length=3, legal_positions=[1, 2, 3]) == [(0, 3), (3, 4)]


def test_oracle_whitespace_lattice():
    rng = np.random.default_rng(202)
    for _ in range(80):
        text = "".join(rng.choice(["a", " ", "b", "\r\n", "e\u0301"], 5)) + " cc"
        p = rng.choice([0.0, 0.1, 0.5, 0.9], len(text))
        cap = int(rng.integers(3, 9))
        expected = oracle(text, p, cap)
        if expected is None:
            with pytest.raises(UnsplittableUnitError):
                split_spans(text, p, max_length=cap)
        else:
            assert split_spans(text, p, max_length=cap) == expected


def test_public_threshold_propagation():
    from types import SimpleNamespace

    sat = object.__new__(SaT)
    sat.model = SimpleNamespace(config=SimpleNamespace(use_character_head=False, model_type="xlm-roberta"))
    sat.tokenizer = None
    sat.language = None
    sat._resolve_sentence_threshold = lambda threshold: threshold

    def predict(texts, **kw):
        if isinstance(texts, str):
            return np.full(len(texts), 0.6), np.zeros(len(texts))
        return (np.full(len(t), 0.6) for t in texts)

    sat.predict_proba = predict
    assert len(sat.split("a" * 40, max_length=20, threshold=0.8)) == 2
    assert len(sat.chunk("a" * 40, max_length=20, threshold=0.8)) == 2


def test_whitespace_legacy_and_public_refusal():
    from wtpsplit import WtP

    wtp = object.__new__(WtP)
    wtp.language = None
    wtp._resolve_mixture = lambda *a: None
    wtp.predict_proba = lambda texts, **kw: (np.zeros(1) for _ in texts)
    for text, cap in [("   ", 2), ("\r\n\r\n", 3), ("", 2)]:
        assert "".join(wtp.split(text, max_length=cap)) == text
    with pytest.raises(UnsplittableUnitError) as caught:
        wtp.split("\r\n", max_length=1)
    assert caught.value.position == 0 and caught.value.measured_length == 2
    sat = object.__new__(SaT)
    with pytest.raises(UnsplittableUnitError):
        sat.chunk("\r\n", max_length=1)


def test_stub_declarations_match_public_api():
    import ast

    root = Path(__file__).resolve().parents[1]
    stub = ast.parse((root / "wtpsplit/__init__.pyi").read_text())
    sat = next(n for n in stub.body if isinstance(n, ast.ClassDef) and n.name == "SaT")
    assert len([n for n in sat.body if isinstance(n, ast.FunctionDef) and n.name == "chunk"]) == 3
    for cls in [n for n in stub.body if isinstance(n, ast.ClassDef) and n.name in {"SaT", "WtP"}]:
        for function in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"split", "segment"}]:
            args = {a.arg for a in function.args.args + function.args.kwonlyargs}
            assert "length_fn" in args
            assert not args & {"min_length", "prior_type", "prior_kwargs", "algorithm", "use_negative_evidence"}
            assert function.args.kwarg is None


def test_whitespace_probability_validation():
    for text in ["", "   ", "\r\n"]:
        with pytest.raises(ValueError, match="probabilities"):
            split_spans(text, [np.nan], max_length=2)
