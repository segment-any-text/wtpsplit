"""Context fill, bounded counter work and the final budget surface."""

import inspect
import time

import numpy as np
import pytest

from wtpsplit import LengthFunction, TokenLength, chunk_spans, split_spans


def toy_count(text):
    return len(text.split()) + int(text.endswith(" "))


def test_context_discount_recovers_toy_fill_and_keeps_strong_evidence():
    text = " ".join(["w"] * 200)
    p = np.zeros(len(text))
    spans = split_spans(text, p, max_length=20, length_fn=toy_count)
    assert len(spans) <= 12
    assert all(toy_count(text[a:b]) <= 20 for a, b in spans)
    assert "".join(text[a:b] for a, b in spans) == text
    print(f"toy pieces={len(spans)} optimum=11")
    p[39 - 1] = 0.99  # Last character before a whole-word cut at source 40.
    spans = split_spans(text, p, max_length=20, length_fn=toy_count)
    assert 40 in {b for _, b in spans}


def test_bpe_like_discount_fill_and_superadditive_safety():
    text = " ".join(["word"] * 1000)

    class BPELike:
        def encode(self, text, *, add_special_tokens=True):
            # A standalone trailing space costs a token; the next word merges
            # it into its prefix, as in byte-level BPE tokenization.
            return [0] * (toy_count(text) + (2 if add_special_tokens else 0))

    count = TokenLength(BPELike(), add_special_tokens=False)
    spans = split_spans(text, np.zeros(len(text)), max_length=32, length_fn=count)
    fill = np.mean([count(text[a:b]) for a, b in spans]) / 32
    assert fill >= 0.85
    assert all(count(text[a:b]) <= 32 for a, b in spans)
    print(f"BPE-like pieces={len(spans)} mean fill={fill:.6f}")

    def superadditive(s):
        return len(s) ** 2

    spans = split_spans("a" * 200, np.zeros(200), max_length=64, length_fn=superadditive)
    assert "".join("a" * (b - a) for a, b in spans) == "a" * 200
    assert all(superadditive("a" * (b - a)) <= 64 for a, b in spans)


@pytest.mark.slow
def test_cached_xlmr_cjk_fill_and_spaced_exactness():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base", local_files_only=True)
    count = TokenLength(tok)
    rng = np.random.default_rng(9182)
    # Authored, seeded Unicode text; no external corpus or test-set content.
    text = "".join(chr(c) for c in rng.integers(0x4E00, 0x7000, 20_000))
    spans = split_spans(text, np.zeros(len(text)), max_length=64, length_fn=count)
    # Frozen count for the previous decoder on this identical text.
    old_pieces = 463
    assert len(spans) <= 1.1 * old_pieces
    assert all(count(text[a:b]) <= 64 for a, b in spans)
    assert "".join(text[a:b] for a, b in spans) == text
    print(
        f"XLMR CJK pieces={len(spans)} old23={old_pieces} mean count={np.mean([count(text[a:b]) for a, b in spans]):.6f}"
    )
    spaced = "The committee met on Monday and approved the budget " * 90
    import re

    atoms = re.findall(r"\S+\s*", spaced)
    ids = [tok.encode(s, add_special_tokens=False, verbose=False) for s in atoms]
    assert sum(ids, []) == tok.encode(spaced, add_special_tokens=False, verbose=False)
    expected, start, used, offset = [], 0, 0, 0
    for atom, tokens in zip(atoms, ids):
        if used + len(tokens) + count.overhead > 64:
            expected.append((start, offset))
            start, used = offset, 0
        offset += len(atom)
        used += len(tokens)
    expected.append((start, offset))
    actual = split_spans(spaced, np.zeros(len(spaced)), max_length=64, length_fn=count)
    assert actual == expected
    print(f"XLMR spaced pieces={len(actual)} exact minimum={len(expected)}")


@pytest.mark.parametrize("kind", ["one-outlier", "near-constant", "two-valued", "three-valued", "five-valued"])
def test_tie_heavy_half_million_speed(kind):
    rng = np.random.default_rng(4082)
    n = 500_000
    p = np.full(n, 0.01)
    if kind == "one-outlier":
        p[n // 2] = 0.02
    elif kind == "two-valued":
        p = rng.choice([0.01, 0.02], n)
    elif kind == "three-valued":
        p = rng.choice([0.1, 0.3, 0.45], n)
    elif kind == "five-valued":
        p = rng.choice([0.05, 0.15, 0.25, 0.35, 0.45], n)
    else:
        p += rng.uniform(-1e-14, 1e-14, n)
    start = time.perf_counter()
    spans = split_spans("字" * n, p, max_length=2000)
    elapsed = time.perf_counter() - start
    # Shared ceiling catches a regression in order of growth, not a speed
    # goal: discrete ties take several seconds and comps0 is shared.
    assert elapsed < 10
    assert all(0 < b - a <= 2000 for a, b in spans)
    assert sum(b - a for a, b in spans) == n
    print(f"tie-heavy {kind}: {elapsed:.6f}s")


@pytest.mark.parametrize("name", ["split", "segment", "wtp"])
@pytest.mark.parametrize("count", [lambda s: len(s)])
def test_length_function_without_budget_is_an_error(name, count):
    from test_newline_retention import fake_sat, fake_wtp

    instance = fake_wtp() if name == "wtp" else fake_sat()
    method = instance.split if name == "wtp" else getattr(instance, name)
    with pytest.raises(TypeError, match="length_fn requires max_length"):
        method("A sentence.", length_fn=count)


def test_tokenizer_interface_and_keyword_only_policy():
    class Minimal:
        model_max_length = 10
        is_fast = True

        def encode(self, text, *, add_special_tokens=True):
            return list(text) + ([0, 0] if add_special_tokens else [])

    count = TokenLength(Minimal(), add_special_tokens=False, reserve_tokens=3)
    assert count("abc") == 6
    with pytest.raises(TypeError):
        TokenLength(Minimal(), True, 3)
    parameters = inspect.signature(TokenLength).parameters
    assert parameters["add_special_tokens"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["reserve_tokens"].kind == inspect.Parameter.KEYWORD_ONLY


def test_public_exports_and_frozen_core_signatures():
    import wtpsplit

    assert LengthFunction is wtpsplit.LengthFunction
    assert "LengthFunction" in wtpsplit.__all__
    assert list(inspect.signature(chunk_spans).parameters) == [
        "text",
        "probabilities",
        "units",
        "max_length",
        "length_fn",
        "sentence_spans",
        "overlap_sentences",
        "threshold",
    ]


@pytest.mark.parametrize("n", [100, 1500])
@pytest.mark.parametrize("explicit", [False, True])
def test_large_counter_call_bound(n, explicit):
    rng = np.random.default_rng(531)
    text = " ".join("w" * int(k) for k in rng.integers(1, 5, n)) + " "
    positions = np.flatnonzero(np.fromiter((c == " " for c in text), bool)) + 1
    counters = [
        len,
        toy_count,
        lambda s: (len(s) + 1) // 2,
        lambda s: len(set(s)),
        lambda s: len(s) + len(s) // 3,
        lambda s: len(s) ** 2,
        lambda s: abs(len(s) - 4),
    ]
    ratios = []
    for counter in counters:
        calls = 0

        def counted(s):
            nonlocal calls
            calls += 1
            return counter(s)

        spans = split_spans(
            text,
            rng.random(len(text)) * 0.4,
            max_length=100,
            length_fn=counted,
            legal_positions=positions if explicit else None,
        )
        assert calls <= (10 if explicit else 12) * n
        ratios.append(calls / n)
        assert all(counter(text[a:b]) <= 100 for a, b in spans)
        assert "".join(text[a:b] for a, b in spans) == text
    print(f"counter stress atoms={n}, explicit={explicit}, seven counters, max calls/A={max(ratios):.4f}")
