"""Final API, all-counter evidence, reference parity and dense-speed contracts."""

from pathlib import Path
import inspect
import time

import numpy as np
import pytest

from length_reference import reference_decode
from wtpsplit import SaT, split_spans, UnsplittableUnitError
from wtpsplit.chunking import TokenLength
from wtpsplit.chunking import _decode


def test_single_budget_form_and_removed_hints():
    parameters = inspect.signature(SaT.chunk).parameters
    assert parameters["length_fn"].default is len
    for name in ["max_tokens", "tokenizer", "add_special_tokens", "reserve_tokens"]:
        assert name not in parameters
        with pytest.raises(TypeError, match="TokenLength"):
            object.__new__(SaT).chunk("text", **{name: 1})


def test_opaque_and_byte_counters_follow_evidence():
    text = "éa bb cc dd "
    p = np.zeros(len(text))
    p[1] = 0.9
    expected = [(0, 3), (3, len(text))]
    for count in [len, lambda s: len(s), lambda s: len(s.encode())]:
        assert split_spans(text, p, max_length=10, length_fn=count, legal_positions=[3, 6, 9]) == expected


def test_atomic_count_call_bound():
    text = "aa bb cc dd "
    calls = {}

    def count(s):
        calls[s] = calls.get(s, 0) + 1
        return len(s) + len(s) // 3

    p = np.zeros(len(text))
    p[1] = 0.9
    spans = split_spans(text, p, max_length=9, length_fn=count, legal_positions=[3, 6, 9])
    assert sum(calls.values()) <= 4 + 3 * 4
    assert calls["aa "] == 1 and calls["bb "] == 1 and calls["cc "] == 1 and calls["dd "] == 1
    assert all(len(text[a:b]) + len(text[a:b]) // 3 <= 9 for a, b in spans)
    assert "".join(text[a:b] for a, b in spans) == text


def test_kernel_matches_frozen_avl_reference():
    rng = np.random.default_rng(6082)
    for _ in range(600):
        n = int(rng.integers(1, 150))
        cuts = np.cumsum(rng.integers(1, 4, n)).tolist()
        cuts = [0, *cuts]
        lengths = np.asarray(cuts, dtype=np.int64)
        cap = int(rng.integers(3, 50))
        if rng.random() < 0.3:
            rewards = rng.integers(-5, 6, n + 1).astype(float) * 4e-10
        else:
            q = np.clip(rng.random(n + 1), 1e-12, 1 - 1e-12)
            rewards = np.log(q) - np.log1p(-q)
        rewards[0] = rewards[-1] = 0
        assert _decode(cuts, lengths, rewards, cap) == reference_decode(cuts, lengths, rewards, cap)


def test_dense_nonconstant_evidence_speed():
    rng = np.random.default_rng(7762)
    times = []
    for n, bound in [(50_000, 1), (500_000, 5)]:
        text = "字" * n
        p = rng.random(n) * 0.4
        start = time.perf_counter()
        spans = split_spans(text, p, max_length=2000)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        assert elapsed < bound
        assert "".join(text[a:b] for a, b in spans) == text
    print(f"dense random 50k={times[0]:.6f}s 500k={times[1]:.6f}s")


def test_deterministic_leftmost_error_position():
    text = "ab\r\ncd\r\nef"
    rng = np.random.default_rng(7)
    for p in [np.zeros(len(text)), rng.random(len(text)), np.ones(len(text))]:
        with pytest.raises(UnsplittableUnitError) as caught:
            split_spans(text, p, max_length=1)
        assert caught.value.position == 2
        assert caught.value.measured_length == 2 and caught.value.max_length == 1


@pytest.mark.parametrize("text,cap,count", [("éd ำ　ำ\nb", 2, lambda s: len(s.split())), (".ำd ำécbก", 6, len)])
def test_whitespace_spacing_mark_clusters_fit(text, cap, count):
    spans = split_spans(text, np.zeros(len(text)), max_length=cap, length_fn=count)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(count(text[a:b]) <= cap for a, b in spans)


def test_fixed_token_overhead_refusal():
    class Specials:
        def encode(self, text, *, add_special_tokens=True):
            return list(text) + ([0, 0] if add_special_tokens else [])

    count = TokenLength(Specials(), reserve_tokens=3)
    for cap in [4, 5]:
        with pytest.raises(ValueError, match=r"overhead \(5\)"):
            split_spans("hello", [0] * 5, max_length=cap, length_fn=count)
    assert split_spans("hello", [0] * 5, max_length=6, length_fn=count)


@pytest.mark.parametrize(
    "changes,error,name",
    [
        ({"text": None}, TypeError, "text"),
        ({"text": ["x"]}, TypeError, "text"),
        ({"length_fn": None}, TypeError, "length_fn"),
        ({"length_fn": 1}, TypeError, "length_fn"),
        ({"threshold": None}, TypeError, "threshold"),
        ({"threshold": "0.5"}, TypeError, "threshold"),
        ({"threshold": np.nan}, ValueError, "threshold"),
        ({"threshold": 2}, ValueError, "threshold"),
        ({"legal_positions": 3}, TypeError, "legal_positions"),
        ({"probabilities": ["invalid"]}, ValueError, "probabilities"),
    ],
)
def test_all_split_inputs_validated(changes, error, name):
    args = {"text": "x", "probabilities": [0.0], "max_length": 2}
    args.update(changes)
    with pytest.raises(error, match=name):
        split_spans(**args)


def test_whitespace_segment_keeps_full_probability_arrays():
    from types import SimpleNamespace

    sat = object.__new__(SaT)
    sat.model = SimpleNamespace(config=SimpleNamespace(use_character_head=False, model_type="xlm-roberta"))
    sat.tokenizer = None
    sat.language = None
    sat._resolve_sentence_threshold = lambda _: 0.25
    for paragraphs in [False, True]:

        def predict(texts, **kw):
            return ((np.zeros(1), np.zeros(1)) if paragraphs else np.zeros(1) for _ in texts)

        sat.predict_proba = predict
        result = sat.segment("   ", max_length=2, do_paragraph_segmentation=paragraphs, split_on_input_newlines=False)
        assert result.probabilities.shape == (3,)
        if paragraphs:
            assert result.paragraph_probabilities.shape == (3,)


@pytest.mark.slow
def test_capped_plain_newline_policy_cached_model():
    from huggingface_hub import hf_hub_download
    import torch

    torch.set_num_threads(1)
    model_path = str(Path(hf_hub_download("segment-any-text/sat-3l-sm", "model.safetensors", local_files_only=True)).parent)
    tokenizer_path = str(Path(hf_hub_download("facebookAI/xlm-roberta-base", "tokenizer.json", local_files_only=True)).parent)
    sat = SaT(
        model_path,
        tokenizer_name_or_path=tokenizer_path,
        device="cpu",
        from_pretrained_kwargs={"local_files_only": True},
    )
    texts = [
        "Title: A tale\nRating: G\nWords: 104",
        "- item one\n- item two\n- item three\n\nNew paragraph here.",
        "First sentence.\r\nSecond sentence.\nA final sentence.",
    ]
    for option in [False, True]:
        for text in texts:
            plain = sat.split(text, threshold=0.25, split_on_input_newlines=option)
            capped = sat.split(text, threshold=0.25, max_length=1000, split_on_input_newlines=option)
            assert capped == plain
            assert "".join(capped) == text


def test_token_budget_shared_across_split_chunk_and_segment():
    import wtpsplit

    assert wtpsplit.TokenLength is TokenLength

    class Specials:
        def encode(self, text, *, add_special_tokens=True):
            return list(text) + ([0, 0] if add_special_tokens else [])

    from types import SimpleNamespace

    sat = object.__new__(SaT)
    sat.model = SimpleNamespace(config=SimpleNamespace(use_character_head=False, model_type="xlm-roberta"))
    sat.tokenizer = None
    sat.language = None
    sat._resolve_sentence_threshold = lambda _: 0.5

    def predict(texts, **kw):
        if isinstance(texts, str):
            return np.zeros(len(texts)), np.zeros(len(texts))
        return (np.zeros(len(t)) for t in texts)

    sat.predict_proba = predict
    counter = TokenLength(Specials(), reserve_tokens=1)
    text = "abcdefgh"
    for pieces in [
        sat.chunk(text, max_length=6, length_fn=counter),
        sat.split(text, max_length=6, length_fn=counter),
        sat.segment(text, max_length=6, length_fn=counter).sentences,
    ]:
        assert "".join(pieces) == text
        assert all(counter(piece) <= 6 for piece in pieces)

    sat.predict_proba = lambda *a, **kw: pytest.fail("invalid overhead must fail before inference")
    for method in [sat.chunk, sat.split, sat.segment]:
        with pytest.raises(ValueError, match="overhead"):
            method(text, max_length=3, length_fn=counter)


def test_final_exports_and_documentation_contracts():
    import wtpsplit
    from wtpsplit import chunking

    assert "TokenLength" in wtpsplit.__all__ and "TokenLength" in chunking.__all__
    assert not hasattr(chunking, "AdditiveLength")
    root = Path(__file__).resolve().parents[1]
    section = (root / "OVERVIEW.md").read_text().split("## Length budgets")[1]
    assert "TokenLength(" in section and "max_length=" in section
