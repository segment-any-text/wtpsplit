"""3.0 line-ending ownership, source reconstruction and process-wide notice."""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from wtpsplit import SaT, WtP, WtpsplitBehaviorChangeWarning
from wtpsplit import _inference
from wtpsplit._inference import decode_probabilities, retained_sentences


def fake_sat():
    sat = object.__new__(SaT)
    sat.model = SimpleNamespace(config=SimpleNamespace(use_character_head=False, model_type="xlm-roberta"))
    sat.tokenizer = None
    sat.language = None
    sat._resolve_sentence_threshold = lambda _: 0.5

    def predict(texts, **kw):
        for text in texts:
            p = np.zeros(len(text) if text.strip() else 1)
            yield (p, p.copy()) if kw.get("return_paragraph_probabilities") else p

    sat.predict_proba = predict
    return sat


def fake_wtp():
    model = object.__new__(WtP)
    model.language = None
    model._resolve_mixture = lambda *args: None

    def predict(texts, **kw):
        for text in texts:
            p = np.zeros(len(text) if text.strip() else 1)
            yield (p, p.copy()) if kw.get("return_paragraph_probabilities") else p

    model.predict_proba = predict
    return model


CASES = [
    ("A\nB", ["A\n", "B"]),
    ("A\r\nB", ["A\r\n", "B"]),
    ("A\rB", ["A\rB"]),
    ("A\n\nB", ["A\n\n", "B"]),
    ("\r\n\nA\rB\r\n", ["\r\n\nA\rB\r\n"]),
    ("  A\n  B  ", ["  A\n  ", "B  "]),
    ("A\r\n \n\tB\r", ["A\r\n \n\t", "B\r"]),
    ("\r\n\r", ["\r\n\r"]),
    ("", []),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_frozen_line_ending_lists(text, expected):
    sat = fake_sat()
    wtp = fake_wtp()
    assert sat.split(text) == expected
    assert sat.segment(text).sentences == expected
    assert wtp.split(text) == ([text] if text else [])
    assert retained_sentences(text, [text] if text else [], force_newlines=True, strip_whitespace=False) == expected
    assert "".join(expected) == text


@pytest.mark.parametrize("text,_expected", CASES)
def test_stripping_removes_line_endings_and_empty_pieces(text, _expected):
    expected = [piece.strip() for piece in _expected if piece.strip()]
    assert fake_sat().split(text, strip_whitespace=True) == expected
    assert fake_sat().segment(text, strip_whitespace=True).sentences == expected
    assert fake_wtp().split(text, strip_whitespace=True) == ([text.strip()] if text.strip() else [])
    assert all(expected)


def test_seeded_all_flag_cap_reconstruction_and_paragraphs():
    rng = np.random.default_rng(63793)
    atoms = ["Alpha", "Beta.", " ", "  ", "\n", "\r\n", "\r", "\n\n", "e\u0301", "字"]
    sat = fake_sat()
    wtp = fake_wtp()
    for _ in range(45):
        text = "".join(rng.choice(atoms, 20))
        for flag in [False, True]:
            for cap in [None, 12]:
                for paragraph in [False, True]:
                    options = dict(split_on_input_newlines=flag, do_paragraph_segmentation=paragraph)
                    if cap is not None:
                        options["max_length"] = cap
                    output = sat.split(text, **options)
                    flat = [p for group in output for p in group] if paragraph else output
                    assert "".join(flat) == text
                    assert all(piece for piece in flat)
                    if cap is not None:
                        assert all(len(piece) <= cap for piece in flat)
                    structured = sat.segment(text, **options)
                    assert "".join(structured.sentences) == text
                    assert [text[a:b] for a, b in structured.spans] == structured.sentences
                    legacy = wtp.split(text, **{k: v for k, v in options.items() if k != "split_on_input_newlines"})
                    flat_legacy = [p for group in legacy for p in group] if paragraph else legacy
                    assert "".join(flat_legacy) == text


def test_paragraph_boundaries_only_change_whitespace_ownership():
    text = "\nA\r\n\r\nB\rC\n"
    sentence = np.zeros(len(text))
    paragraph = np.zeros(len(text))
    paragraph[text.index("A")] = 0.9
    result = decode_probabilities(
        text,
        (sentence, paragraph),
        sentence_threshold=0.5,
        paragraph_threshold=0.5,
        strip_whitespace=False,
        do_paragraph_segmentation=True,
        split_on_input_newlines=True,
        max_length=None,
        language=None,
    )
    assert result.paragraphs == [["\nA\r\n\r\n"], ["B\rC\n"]]
    assert "".join("".join(group) for group in result.paragraphs) == text


def test_caps_can_require_whitespace_only_pieces():
    for text in ["A" + "\n" * 7 + "B", "\n" * 7 + "B"]:
        for model in [fake_sat(), fake_wtp()]:
            pieces = model.split(text, max_length=2)
            assert "".join(pieces) == text
            assert all(0 < len(piece) <= 2 for piece in pieces)
            assert any(piece.isspace() for piece in pieces)


def test_notice_once_across_apis_and_threads(monkeypatch):
    monkeypatch.setattr(_inference, "_retention_notice_pid", None)
    sat = fake_sat()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", WtpsplitBehaviorChangeWarning)
        sat.split("Single line")
        sat.split("Lone\rCR")
        assert not records
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: sat.split("A\nB"), range(8)))
        sat.segment("C\rD")
        fake_wtp().split("E\r\nF")
    changes = [w for w in records if issubclass(w.category, WtpsplitBehaviorChangeWarning)]
    assert len(changes) == 1
    message = str(changes[0].message)
    assert len(message.splitlines()) <= 3
    assert "3.0" in message and "strip_whitespace=True" in message and "filterwarnings" in message
    assert _inference._retention_notice_pid == os.getpid()


def test_notice_can_be_silenced_and_false_does_not_trigger_it(monkeypatch):
    monkeypatch.setattr(_inference, "_retention_notice_pid", None)
    sat = fake_sat()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", WtpsplitBehaviorChangeWarning)
        sat.split("A\nB", split_on_input_newlines=False)
        assert _inference._retention_notice_pid is None
        warnings.filterwarnings("ignore", category=WtpsplitBehaviorChangeWarning)
        sat.split("A\nB")
        sat.segment("C\rD")
        assert not records


def test_lazy_call_notifies_at_invocation(monkeypatch):
    monkeypatch.setattr(_inference, "_retention_notice_pid", None)
    with pytest.warns(WtpsplitBehaviorChangeWarning):
        output = fake_sat().split(["A\nB"], lazy=True)
    assert list(output) == [["A\n", "B"]]


@pytest.mark.slow
def test_cached_model_reconstruction_all_four_combinations():
    import torch
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(1)
    model_path = str(Path(hf_hub_download("segment-any-text/sat-3l-sm", "model.safetensors", local_files_only=True)).parent)
    tok_path = str(Path(hf_hub_download("facebookAI/xlm-roberta-base", "tokenizer.json", local_files_only=True)).parent)
    sat = SaT(
        model_path, tokenizer_name_or_path=tok_path, device="cpu", from_pretrained_kwargs={"local_files_only": True}
    )
    rng = np.random.default_rng(874)
    lines = ["A first sentence.", "The committee approved its budget.", "A final line."]
    texts = [
        "\r\n" + "".join(line + rng.choice(["\n", "\r\n", "\r", "\n\n", "\r\n\r\n"]) for line in rng.permutation(lines))
        for _ in range(4)
    ]
    for flag in [False, True]:
        for capped in [False, True]:
            for paragraphs in [False, True]:
                for text in texts:
                    options = dict(threshold=0.25, split_on_input_newlines=flag, do_paragraph_segmentation=paragraphs)
                    if capped:
                        options["max_length"] = 24
                    pieces = sat.split(text, **options)
                    flat = [s for group in pieces for s in group] if paragraphs else pieces
                    assert "".join(flat) == text
                    assert all(flat)
                    segmented = sat.segment(text, **options)
                    assert "".join(segmented.sentences) == text
                    assert [text[a:b] for a, b in segmented.spans] == segmented.sentences
        for text in texts:
            for paragraph in [False, True]:
                assert sat.split(
                    text, threshold=0.25, split_on_input_newlines=flag, do_paragraph_segmentation=paragraph
                ) == sat.split(
                    text,
                    threshold=0.25,
                    split_on_input_newlines=flag,
                    max_length=1000,
                    do_paragraph_segmentation=paragraph,
                )
    assert sat.split("First.\nSecond.", threshold=1.0) == ["First.\n", "Second."]
    assert sat.split("First.\n\nSecond.", threshold=1.0, strip_whitespace=True) == ["First.", "Second."]


def test_breaking_change_docs_and_exports_are_explicit():
    from pathlib import Path
    import wtpsplit

    root = Path(__file__).resolve().parents[1]
    assert "WtpsplitBehaviorChangeWarning" in wtpsplit.__all__
    assert (
        "WtpsplitBehaviorChangeWarning as WtpsplitBehaviorChangeWarning" in (root / "wtpsplit/__init__.pyi").read_text()
    )
    for method in [SaT.split, SaT.segment]:
        first = method.__doc__.splitlines()[0]
        assert "split_on_input_newlines=True" in first and "3.0" in first and "KEEPS" in first


# Frozen 2.2.2 indices_to_sentences (3ef928c8e, utils/__init__.py:369–403)
# and SaT's LF-only forcing (__init__.py:1322–1337). WtP's :637–640
# yields model sentences directly; paragraph branches :575/:1257 do not force.
def frozen_v222_sentences(text, p, *, sat, force=True, paragraph_p=None):
    def indices(source, probabilities):
        output, offset, end = [], 0, 0
        for at in np.flatnonzero(probabilities > 0.5):
            end = int(at) + 1
            while end < len(source) and source[end].isspace():
                end += 1
            piece = source[offset:end]
            if piece:
                output.append(piece)
            offset = end
        if end != len(source) and source[end:]:
            output.append(source[end:])
        return output

    if paragraph_p is not None:
        result, offset = [], 0
        for para in indices(text, paragraph_p):
            result.extend(indices(para, p[offset : offset + len(para)]))
            offset += len(para)
        return result
    sentences = indices(text, p)
    if not sat or not force:
        return sentences
    result = []
    for i, piece in enumerate(sentences):
        if i < len(sentences) - 1 and piece.endswith("\n"):
            piece = piece[:-1]
        result.extend(piece.split("\n"))
    return result


def content_cut_positions(text, pieces):
    # LF was deleted and blank placeholders had no source content in 2.2.2.
    # Compare the source anchors before owned trailing whitespace, ignoring
    # those empty placeholders; ownership of that gap is the approved change.
    cursor, ends = 0, set()
    for piece in pieces:
        if not piece.strip():
            continue
        start = text.index(piece, cursor)
        end = start + len(piece)
        ends.add(end - len(piece) + len(piece.rstrip()))
        cursor = end
    return ends


def test_cut_positions_match_frozen_222_for_both_classes():
    import inspect

    assert "split_on_input_newlines" not in inspect.signature(WtP.split).parameters
    rng = np.random.default_rng(201122)
    for _ in range(60):
        text = "".join(
            word + rng.choice(["\n", "\r\n", "\r", "\n\n", " "]) for word in ["Alpha.", "Beta.", "Gamma.", "Delta."]
        )
        p = rng.choice([0.0, 0.8], len(text))
        paragraph_p = rng.choice([0.0, 0.0, 0.8], len(text))
        for paragraphs in [False, True]:
            for sat_kind in [False, True]:
                model = fake_sat() if sat_kind else fake_wtp()

                def predict(texts, **kw):
                    return ((p, paragraph_p) if paragraphs else p for _ in texts)

                model.predict_proba = predict
                for force in [False, True] if sat_kind else [False]:
                    options = dict(threshold=0.5, do_paragraph_segmentation=paragraphs)
                    if sat_kind:
                        options["split_on_input_newlines"] = force
                    result = model.split(text, **options)
                    actual = [s for group in result for s in group] if paragraphs else result
                    old = frozen_v222_sentences(
                        text, p, sat=sat_kind, force=force, paragraph_p=paragraph_p if paragraphs else None
                    )
                    assert content_cut_positions(text, actual) == content_cut_positions(text, old)
                    assert "".join(actual) == text
                    assert model.split(text, max_length=1000, **options) == result
    assert fake_sat().split("A\rB") == ["A\rB"]
    assert fake_sat().split("A\r\nB") == ["A\r\n", "B"]
    assert fake_wtp().split("A\nB\rC") == ["A\nB\rC"]
