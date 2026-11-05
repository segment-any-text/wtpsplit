"""Exhaustive outcome-parity tests for the host-latency speedup identities."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import pytest
import torch

from scripts.benchmark_host_latency import (
    SLIDE_LANGUAGE_SAMPLES,
    equivalent_predict_proba_subword,
    equivalent_split_subword,
    equivalent_window_and_stitch,
    reference_window_and_stitch,
)
from wtpsplit._host_kernels import exact_float16_logit_threshold
from wtpsplit.utils import indices_to_sentences, sigmoid, token_to_char_probs


def test_exhaustive_float16_single_window_identity() -> None:
    """Across all finite IEEE-754 float16 bit patterns, `float16((0 + 1 * x) / 1)` equals `x` (and is bit-identical for nonzero finite `x`)."""
    all_u16 = np.arange(65536, dtype=np.uint16)
    all_f16 = all_u16.view(np.float16)
    finite = all_f16[np.isfinite(all_f16)]
    acc = np.zeros_like(finite, dtype=np.float16)
    w = np.ones_like(finite, dtype=np.float16)
    cnt = np.zeros_like(finite, dtype=np.float16)
    acc += w * finite
    cnt += w
    res = (acc / cnt).astype(np.float16)
    assert np.array_equal(res, finite)
    nonzero = finite != 0
    assert np.array_equal(res[nonzero].view(np.uint16), finite[nonzero].view(np.uint16))


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 0.0, 0.01, 0.025, 0.13, 0.25, 0.35, 0.40, 0.50, 0.85, 0.9999999, 1.0, 1.5, float("nan")],
)
def test_exhaustive_float16_logit_threshold_identity(threshold: float) -> None:
    """Across ALL 65,536 IEEE-754 float16 bit patterns (including +-0, +-inf, NaN), `sigmoid(x) > threshold` equals `x >= z_min`."""
    all_f16 = np.arange(65536, dtype=np.uint16).view(np.float16)
    z_min = exact_float16_logit_threshold(threshold)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = sigmoid(all_f16) > threshold
        actual = np.zeros_like(expected, dtype=bool) if z_min is None else (all_f16 >= z_min)
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize("weighting", ["uniform", "hat"])
@pytest.mark.parametrize("doc_lengths", [[12, 80, 510], [511, 640, 1200], [1, 150, 510, 820]])
def test_window_and_stitch_bit_parity(weighting: Literal["uniform", "hat"], doc_lengths: list[int]) -> None:
    """Verify bit-exact `input_ids`, `attention_mask`, and `stitched` logits across single & multi-window docs."""
    rng = np.random.default_rng(123)
    token_ids = [rng.integers(10, 25000, size=n).tolist() for n in doc_lengths]
    block_size, stride = 510, 64
    eff_block = min(block_size, max(doc_lengths))
    num_chunks = sum(int(np.ceil(max(n - eff_block, 0) / stride)) + 1 for n in doc_lengths)
    raw_logits = rng.standard_normal((num_chunks, eff_block, 1)).astype(np.float16)

    ref_in, ref_mask, ref_stitched = reference_window_and_stitch(
        token_ids, raw_logits, block_size=block_size, stride=stride, cls_id=0, sep_id=2, weighting=weighting
    )
    cand_in, cand_mask, cand_stitched = equivalent_window_and_stitch(
        token_ids, raw_logits, block_size=block_size, stride=stride, cls_id=0, sep_id=2, weighting=weighting
    )

    assert np.array_equal(ref_in, cand_in)
    assert np.array_equal(ref_mask, cand_mask)
    for r_s, c_s in zip(ref_stitched, cand_stitched):
        assert np.array_equal(r_s.view(np.uint16), c_s.view(np.uint16))


@pytest.mark.parametrize("strip_whitespace", [False, True])
@pytest.mark.parametrize("threshold", [0.025, 0.25, 0.35])
def test_decode_proba_and_split_pathological_offsets(strip_whitespace: bool, threshold: float) -> None:
    """Verify bit-exact `predict_proba` and `split` even with (0, 0) and colliding zero-width token spans."""
    rng = np.random.default_rng(99)
    n_docs, n_tokens = 32, 120
    texts = ["Word1. Word2! Word3? " * 20 for _ in range(n_docs)]
    tokens = [[10 + j for j in range(n_tokens)] for _ in range(n_docs)]
    offsets = [
        [
            (0, 0)
            if j in (0, 15, 70)
            else ((j * 3, j * 3) if j % 11 == 0 else (j * 3, min(j * 3 + 3, len(texts[0]))))
            for j in range(n_tokens)
        ]
        for _ in range(n_docs)
    ]
    stitched_logits = [(rng.standard_normal((n_tokens, 1)) - 1.5).astype(np.float16) for _ in range(n_docs)]
    special_tokens = ["<s>", "</s>", "<pad>"]

    ref_probs = [
        sigmoid(token_to_char_probs(texts[i], tokens[i], stitched_logits[i], special_tokens, offsets[i])[:, 0]).astype(
            np.float32, copy=False
        )
        for i in range(n_docs)
    ]
    cand_probs = equivalent_predict_proba_subword(texts, stitched_logits, offsets)
    for r_p, c_p in zip(ref_probs, cand_probs):
        assert np.array_equal(r_p.view(np.uint32), c_p.view(np.uint32))

    ref_splits = [
        indices_to_sentences(texts[i], np.where(ref_probs[i] > threshold)[0], strip_whitespace=strip_whitespace)
        for i in range(n_docs)
    ]
    cand_splits = equivalent_split_subword(
        texts, stitched_logits, offsets, threshold=threshold, strip_whitespace=strip_whitespace
    )
    assert ref_splits == cand_splits


def test_slide_languages_synthetic_parity() -> None:
    """Verify end-to-end windowing, stitching, predict_proba, and split parity across all 10 slide languages."""
    rng = np.random.default_rng(7)
    texts = [sample * 3 for sample in SLIDE_LANGUAGE_SAMPLES.values()]
    # Deterministic character-chunked pseudo-tokens covering every character of the 10 scripts
    token_ids = [[10 + (ord(ch) % 2000) for ch in t[::2]] for t in texts]
    offsets = [[(2 * idx, min(2 * idx + 2, len(t))) for idx in range(len(t[::2]))] for t in texts]

    block_size, stride = 128, 32
    eff_block = min(block_size, max(len(ids) for ids in token_ids))
    num_chunks = sum(int(np.ceil(max(len(ids) - eff_block, 0) / stride)) + 1 for ids in token_ids)
    raw_logits = (rng.standard_normal((num_chunks, eff_block, 1)) - 2.0).astype(np.float16)

    ref_in, ref_mask, ref_stitched = reference_window_and_stitch(
        token_ids, raw_logits, block_size=block_size, stride=stride, cls_id=0, sep_id=2
    )
    cand_in, cand_mask, cand_stitched = equivalent_window_and_stitch(
        token_ids, raw_logits, block_size=block_size, stride=stride, cls_id=0, sep_id=2
    )
    assert np.array_equal(ref_in, cand_in)
    assert np.array_equal(ref_mask, cand_mask)

    special_tokens = ["<s>", "</s>", "<pad>"]
    ref_probs = [
        sigmoid(token_to_char_probs(texts[i], token_ids[i], ref_stitched[i], special_tokens, offsets[i])[:, 0]).astype(
            np.float32, copy=False
        )
        for i in range(len(texts))
    ]
    cand_probs = equivalent_predict_proba_subword(texts, cand_stitched, offsets)
    for r_p, c_p in zip(ref_probs, cand_probs):
        assert np.array_equal(r_p.view(np.uint32), c_p.view(np.uint32))

    ref_splits = [indices_to_sentences(texts[i], np.where(ref_probs[i] > 0.25)[0]) for i in range(len(texts))]
    cand_splits = equivalent_split_subword(texts, cand_stitched, offsets, threshold=0.25)
    assert ref_splits == cand_splits


@pytest.mark.parametrize("weighting", ["uniform", "hat"])
@pytest.mark.parametrize("strip_whitespace", [False, True])
@pytest.mark.parametrize("split_on_input_newlines", [True, False])
def test_host_kernels_and_sat_end_to_end_parity(
    weighting: Literal["uniform", "hat"],
    strip_whitespace: bool,
    split_on_input_newlines: bool,
) -> None:
    """Verify `_host_kernels` and `SaT.split` / `SaT.segment` / `SaT.predict_proba` parity."""
    from types import SimpleNamespace
    import warnings
    from wtpsplit import SaT
    from wtpsplit._host_kernels import (
        decode_subword_split,
        exact_float16_logit_threshold,
        pack_subword_windows,
        project_subword_proba,
        stitch_subword_logits,
    )

    texts = [
        "Hello world.\nSecond line!  ",
        "",
        "   ",
        " \n ",
        SLIDE_LANGUAGE_SAMPLES["Yoruba"] * 3,
        SLIDE_LANGUAGE_SAMPLES["Burmese"] * 3,
    ]

    class PairTokenizer:
        cls_token_id = 0
        sep_token_id = 2
        pad_token_id = 1
        is_fast = False

        def __call__(
            self,
            batch_texts: list[str],
            *,
            return_offsets_mapping: bool = True,
            verbose: bool = False,
            add_special_tokens: bool = False,
        ) -> dict[str, list[Any]]:
            ids_out: list[list[int]] = []
            off_out: list[list[tuple[int, int]]] = []
            for t in batch_texts:
                ids_row: list[int] = []
                off_row: list[tuple[int, int]] = []
                for idx in range(0, len(t), 2):
                    ids_row.append(10 + (ord(t[idx]) % 200))
                    off_row.append((idx, min(idx + 2, len(t))))
                ids_out.append(ids_row)
                off_out.append(off_row)
            return {"input_ids": ids_out, "offset_mapping": off_out}

    class DeterministicSubwordModel:
        config = SimpleNamespace(
            model_type="xlm-token",
            num_labels=1,
            downsampling_rate=1,
            use_character_head=False,
            language_adapter="off",
            max_position_embeddings=514,
            sat_export_default_threshold=0.25,
        )

        def __call__(self, *, attention_mask: NDArray[Any], input_ids: NDArray[Any]) -> dict[str, NDArray[Any]]:
            pos = np.arange(input_ids.shape[1], dtype=np.float32)[None, :]
            val = (((input_ids * 13 + pos * 7) % 29) / 7.0 - 2.2).astype(np.float16)
            return {"logits": val[..., None]}

    sat: Any = SaT.__new__(SaT)
    sat.model = DeterministicSubwordModel()
    sat.tokenizer = PairTokenizer()
    sat.special_tokens = ["<s>", "</s>", "<pad>"]
    sat.use_lora = False
    sat.language = None
    sat.model_name_or_model = "sat-3l-sm"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        segs = list(
            sat.segment(
                texts,
                threshold=0.25,
                stride=8,
                block_size=24,
                batch_size=3,
                weighting=weighting,
                strip_whitespace=strip_whitespace,
                split_on_input_newlines=split_on_input_newlines,
            )
        )
        splits = list(
            sat.split(
                texts,
                threshold=0.25,
                stride=8,
                block_size=24,
                batch_size=3,
                weighting=weighting,
                strip_whitespace=strip_whitespace,
                split_on_input_newlines=split_on_input_newlines,
            )
        )

    assert splits == [s.sentences for s in segs]

    # Also check direct kernel functions against reference_window_and_stitch
    enc = sat.tokenizer(texts[4:])
    in_ids, mask, locs, eff_block = pack_subword_windows(
        enc["input_ids"], block_size=24, stride=8, cls_id=0, sep_id=2
    )
    raw_logits = sat.model(attention_mask=mask, input_ids=in_ids)["logits"]
    stitched = stitch_subword_logits(
        [raw_logits],
        locs,
        [len(x) for x in enc["input_ids"]],
        block_size=eff_block,
        num_labels=1,
        weighting=weighting,
        strip_cls_sep=True,
    )
    _, _, ref_stitched = reference_window_and_stitch(
        enc["input_ids"],
        raw_logits[:, 1:-1, :],
        block_size=24,
        stride=8,
        cls_id=0,
        sep_id=2,
        weighting=weighting,
    )
    for s_cand, s_ref in zip(stitched, ref_stitched):
        assert np.array_equal(s_cand.view(np.uint16), s_ref.view(np.uint16))

    z_min = exact_float16_logit_threshold(0.25)
    for i, t in enumerate(texts[4:]):
        p_cand = project_subword_proba(
            len(t), enc["input_ids"][i], stitched[i], enc["offset_mapping"][i], sat.special_tokens
        )
        p_ref = sigmoid(
            token_to_char_probs(t, enc["input_ids"][i], ref_stitched[i], sat.special_tokens, enc["offset_mapping"][i])[
                :, 0
            ]
        ).astype(np.float32)
        assert np.array_equal(p_cand.view(np.uint32), p_ref.view(np.uint32))
        s_split_cand = decode_subword_split(
            t,
            stitched[i],
            enc["offset_mapping"][i],
            z_min=z_min,
            strip_whitespace=strip_whitespace,
            split_on_input_newlines=split_on_input_newlines,
        )
        assert s_split_cand == splits[4 + i]


@pytest.mark.parametrize("num_hashes", [1, 8])
@pytest.mark.parametrize("num_labels", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("project_tokens_first", [False, True])
def test_char_head_on_device_hashes_and_sparse_gate_bit_parity(
    num_hashes: int,
    num_labels: int,
    kernel_size: int,
    project_tokens_first: bool,
) -> None:
    """Verify 100% bit-identity of 2D ordinals hashing and Halo-Compacted Sparse `CharacterResolutionHead`."""
    import torch
    from wtpsplit._char_preprocessing import build_char_inputs_vectorized
    from wtpsplit.char_head import CharacterResolutionHead, CharInputs

    texts = [s * 2 for s in list(SLIDE_LANGUAGE_SAMPLES.values())[:6]]
    offsets = [[(idx, min(idx + 2, len(t))) for idx in range(0, len(t), 2)] for t in texts]
    ci_3d: CharInputs = build_char_inputs_vectorized(  # type: ignore[no-untyped-call]
        texts, offsets, num_hashes=num_hashes
    )

    batch_size = len(texts)
    width = max(len(t) for t in texts)
    ordinals_2d = np.zeros((batch_size, width), dtype=np.int32)
    for i, t in enumerate(texts):
        ordinals_2d[i, : len(t)] = np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32).astype(np.int32)

    ci_shifted_3d = CharInputs(
        char_to_token=ci_3d.char_to_token + 1,
        is_token_final=ci_3d.is_token_final,
        position_in_token=ci_3d.position_in_token,
        hashes=ci_3d.hashes,
        mask=ci_3d.mask,
    )
    ci_shifted_2d = CharInputs(
        char_to_token=ci_3d.char_to_token + 1,
        is_token_final=ci_3d.is_token_final,
        position_in_token=ci_3d.position_in_token,
        hashes=torch.from_numpy(ordinals_2d),
        mask=ci_3d.mask,
    )

    torch.manual_seed(123)
    head = CharacterResolutionHead(
        hidden_size=32,
        num_labels=num_labels,
        num_hashes=num_hashes,
        kernel_size=kernel_size,
        project_tokens_first=project_tokens_first,
    ).eval()
    with torch.no_grad():
        head.delta.weight.normal_(std=0.05)

    max_tokens = max(len(o) for o in offsets) + 2
    token_hidden = torch.randn(batch_size, max_tokens, 32)
    token_logits = torch.randn(batch_size, max_tokens, num_labels) - 2.0

    with torch.no_grad():
        out_3d = head(token_hidden, token_logits, ci_shifted_3d)
        out_2d = head(token_hidden, token_logits, ci_shifted_2d)
        out_sparse_3d = head(token_hidden, token_logits, ci_shifted_3d, gate_logit_threshold=-1.0)
        out_sparse_2d = head(token_hidden, token_logits, ci_shifted_2d, gate_logit_threshold=-1.0)
        out_sparse_none = head(token_hidden, token_logits, ci_shifted_2d, gate_logit_threshold=100.0)
        tb_comp, tc_comp, tlog_comp = head.forward_sparse_compact(
            token_hidden, token_logits, ci_shifted_2d, gate_logit_threshold=-1.0
        )
        tb_none, tc_none, tlog_none = head.forward_sparse_compact(
            token_hidden, token_logits, ci_shifted_2d, gate_logit_threshold=100.0
        )
        assert tb_none.numel() == 0 and tc_none.numel() == 0 and tlog_none.shape == (0, num_labels)

        # Empty batch / char dimensions
        empty_ci = CharInputs(
            char_to_token=torch.empty((0, 0), dtype=torch.int64),
            is_token_final=torch.empty((0, 0), dtype=torch.int64),
            position_in_token=torch.empty((0, 0), dtype=torch.int64),
            hashes=torch.empty((0, 0), dtype=torch.int32),
            mask=torch.empty((0, 0), dtype=torch.int64),
        )
        out_empty = head(
            torch.empty((0, 2, 32)),
            torch.empty((0, 2, num_labels)),
            empty_ci,
            gate_logit_threshold=-1.0,
        )
        assert out_empty.shape == (0, 0, num_labels)
        eb, ec, elog = head.forward_sparse_compact(
            torch.empty((0, 2, 32)),
            torch.empty((0, 2, num_labels)),
            empty_ci,
            gate_logit_threshold=-1.0,
        )
        assert eb.numel() == 0 and ec.numel() == 0 and elog.shape == (0, num_labels)

        zero_chars_ci = CharInputs(
            char_to_token=torch.empty((2, 0), dtype=torch.int64),
            is_token_final=torch.empty((2, 0), dtype=torch.int64),
            position_in_token=torch.empty((2, 0), dtype=torch.int64),
            hashes=torch.empty((2, 0), dtype=torch.int32),
            mask=torch.empty((2, 0), dtype=torch.int64),
        )
        out_zero_chars = head(
            torch.zeros((2, 3, 32)),
            torch.zeros((2, 3, num_labels)),
            zero_chars_ci,
            gate_logit_threshold=-1.0,
        )
        assert out_zero_chars.shape == (2, 0, num_labels)

        with pytest.raises(ValueError, match="token_hidden must have at least one token"):
            head(
                torch.empty((batch_size, 0, 32)),
                torch.empty((batch_size, 0, num_labels)),
                ci_shifted_2d,
                gate_logit_threshold=-1.0,
            )
        with pytest.raises(ValueError, match="token_hidden must have at least one token"):
            head.forward_sparse_compact(
                torch.empty((batch_size, 0, 32)),
                torch.empty((batch_size, 0, num_labels)),
                ci_shifted_2d,
                gate_logit_threshold=-1.0,
            )

    # 1. 2D int32 ordinals produce 100% bit-identical outputs to 3D int64 pre-computed hashes
    assert torch.equal(out_3d, out_2d)
    assert torch.equal(out_sparse_3d, out_sparse_2d)

    # 2. On every gated token (`max_l token_logits > -1.0`), sparse output matches within 1 float32 SGEMM ULP (`<= 1e-6`)
    tok_pass = token_logits.max(dim=-1).values > -1.0
    target_chars = tok_pass.gather(1, ci_shifted_3d.char_to_token) & (ci_shifted_3d.mask != 0)
    assert int(target_chars.sum().item()) > 0
    torch.testing.assert_close(out_3d[target_chars], out_sparse_3d[target_chars], rtol=0, atol=1e-6)
    assert torch.equal(out_3d.half()[target_chars], out_sparse_3d.half()[target_chars])
    assert torch.equal(out_sparse_2d[tb_comp, tc_comp], tlog_comp)
    assert out_sparse_none.shape == out_3d.shape


def test_collision_chains_and_nonmonotonic_offsets_last_writer_wins() -> None:
    """Verify multi-token collision chains (pass, pass, fail) and non-monotonic offsets match dense projection."""
    from wtpsplit._host_kernels import decode_subword_split

    text = "abcdef"
    cases: list[tuple[list[tuple[int, int]], list[float]]] = [
        ([(0, 1), (0, 1), (0, 1)], [2.0, 2.0, -2.0]),
        ([(0, 1), (0, 1), (0, 1)], [-2.0, 2.0, 2.0]),
        ([(0, 1), (0, 1), (0, 1)], [2.0, -2.0, 2.0]),
        ([(0, 1), (0, 1), (0, 1), (0, 1)], [2.0, 2.0, 2.0, -2.0]),
        ([(3, 5), (0, 2), (3, 5)], [2.0, 2.0, -2.0]),
        ([(3, 5), (0, 2), (3, 5)], [-2.0, -2.0, 2.0]),
        ([(0, 6), (0, 0), (0, 6)], [2.0, -2.0, 2.0]),
    ]
    for offsets, raw_vals in cases:
        logits = np.asarray(raw_vals, dtype=np.float16)[:, None]
        tokens = list(range(10, 10 + len(offsets)))
        for threshold in (-0.1, 0.25, 0.5, 1.0, float("nan")):
            with np.errstate(over="ignore", invalid="ignore"):
                ref_p = sigmoid(token_to_char_probs(text, tokens, logits, [], offsets)[:, 0])
                ref_splits = indices_to_sentences(text, np.where(ref_p > threshold)[0])
            z_min = exact_float16_logit_threshold(threshold)
            cand_splits = decode_subword_split(
                text,
                logits,
                offsets,
                z_min=z_min,
                split_on_input_newlines=False,
            )
            assert cand_splits == ref_splits


def test_fast_rust_tokenizer_and_backend_truncation_padding_guard() -> None:
    """Verify fast Rust encode_batch dispatch and fallback when backend truncation/padding is active."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._host_kernels import can_use_rust_encode_batch, fast_tokenize_subword

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "c": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    texts = ["a b c a b", "b c"]
    assert can_use_rust_encode_batch(tok)
    enc_fast = fast_tokenize_subword(tok, texts)
    enc_ref = tok(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert enc_fast.input_ids == enc_ref["input_ids"]
    assert enc_fast.offset_mapping == enc_ref["offset_mapping"]

    # Now enable backend truncation on `_tokenizer`: `can_use_rust_encode_batch` must detect it and fall back
    tok._tokenizer.enable_truncation(max_length=2)
    assert not can_use_rust_encode_batch(tok)
    enc_after_trunc = fast_tokenize_subword(tok, texts)
    enc_ref_after_trunc = tok(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert enc_after_trunc.input_ids == enc_ref_after_trunc["input_ids"] == [[4, 5, 6, 4, 5], [5, 6]]
    assert enc_after_trunc.offset_mapping == enc_ref_after_trunc["offset_mapping"]

    # Disable truncation and enable backend padding only: `can_use_rust_encode_batch` must also detect it and fall back
    tok._tokenizer.no_truncation()
    assert can_use_rust_encode_batch(tok)
    tok._tokenizer.enable_padding(length=8, pad_id=1, pad_token="<pad>")
    assert not can_use_rust_encode_batch(tok)
    enc_after_pad = fast_tokenize_subword(tok, texts)
    enc_ref_after_pad = tok(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert enc_after_pad.input_ids == enc_ref_after_pad["input_ids"] == [[4, 5, 6, 4, 5], [5, 6]]
    assert enc_after_pad.offset_mapping == enc_ref_after_pad["offset_mapping"]


def test_giga_word_cache_and_fallback_edge_cases() -> None:
    """Verify GigaPy fused `(id, delta)` word cache, lazy offsets, and fallback on exotic/multi-space docs."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._host_kernels import (
        GigaPackedDoc,
        _get_or_init_word_cache,
        decode_subword_split,
        fast_tokenize_subword,
        project_subword_proba,
    )

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    vocab = [
        ("<s>", 0.0),
        ("<pad>", 0.0),
        ("</s>", 0.0),
        ("<unk>", 0.0),
        ("▁Hello", -1.0),
        ("▁world", -1.0),
        (".", -1.0),
        ("▁Test", -1.0),
        ("▁a", -1.5),
        ("b", -1.5),
        ("c", -1.5),
        ("!", -1.5),
    ]
    backend = tokenizers_mod.Tokenizer(tokenizers_mod.models.Unigram(vocab, unk_id=3, byte_fallback=False))
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Sequence(
        [
            tokenizers_mod.pre_tokenizers.WhitespaceSplit(),
            tokenizers_mod.pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True),
        ]
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    assert _get_or_init_word_cache(tok) is not None

    texts = [
        "Hello world. Test ab!",
        "Hello world. Test ab! ",
        "  Hello world.",
        "Hello  world.",
        "Hello\nworld.",
        "Hello\tworld.",
        "a\x01b Hello",
        "<s> Hello </s>",
        "",
        "   ",
    ]
    # Cold call followed by warm call
    batch_cold = fast_tokenize_subword(tok, texts, enable_word_cache=True)
    batch_warm = fast_tokenize_subword(tok, texts, enable_word_cache=True)
    ref = tok(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)

    assert isinstance(batch_warm.offsets_or_encoding(0), GigaPackedDoc)
    assert isinstance(batch_warm.offsets_or_encoding(1), GigaPackedDoc)
    assert batch_cold.as_dict()["input_ids"] == ref["input_ids"]
    assert batch_warm.as_dict()["input_ids"] == ref["input_ids"]
    assert batch_cold.offset_mapping == ref["offset_mapping"]
    assert batch_warm.offset_mapping == ref["offset_mapping"]

    rng = np.random.default_rng(77)
    z_min = exact_float16_logit_threshold(0.25)
    for i, text in enumerate(texts):
        ids_i = ref["input_ids"][i]
        off_i = ref["offset_mapping"][i]
        logits_i = (rng.standard_normal((len(ids_i), 1)) - 0.5).astype(np.float16)

        p_cand = project_subword_proba(
            len(text),
            batch_warm.input_ids[i],
            logits_i,
            batch_warm.offsets_or_encoding(i),
            ["<s>", "</s>", "<pad>"],
        )
        if len(ids_i) > 0 and len(text) > 0:
            p_ref = sigmoid(token_to_char_probs(text, ids_i, logits_i, ["<s>", "</s>", "<pad>"], off_i)[:, 0]).astype(
                np.float32
            )
        else:
            p_ref = np.zeros(len(text), dtype=np.float32)
        assert np.array_equal(p_cand.view(np.uint32), p_ref.view(np.uint32))

        s_cand = decode_subword_split(
            text,
            logits_i,
            batch_warm.offsets_or_encoding(i),
            z_min=z_min,
            split_on_input_newlines=False,
        )
        s_ref = indices_to_sentences(text, np.where(p_ref > 0.25)[0])
        assert s_cand == s_ref

    # Adding a custom non-special AddedToken invalidates the word cache and falls back cleanly
    tok.add_tokens([tokenizers_mod.AddedToken("\n", normalized=False)])
    assert _get_or_init_word_cache(tok) is None
    batch_after_add = fast_tokenize_subword(tok, texts[:5], enable_word_cache=True)
    ref_after_add = tok(texts[:5], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert batch_after_add.as_dict()["input_ids"] == ref_after_add["input_ids"]
    assert batch_after_add.offset_mapping == ref_after_add["offset_mapping"]


def test_giga_word_cache_adversarial_mutations_eviction_and_stitch_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the 4 adversarial cases: Replace normalizer, backend Lowercase mutation, cache eviction lifetime, and heterogeneous window spans."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    import wtpsplit._host_kernels as hk
    from wtpsplit._host_kernels import _get_or_init_word_cache, fast_tokenize_subword, stitch_subword_logits

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "Hello": 4, "world.": 5, "Test": 6, "a": 7, "b": 8},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.WhitespaceSplit()
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    # 1. Cache eviction lifetime: even with MAX_WORD_CACHE_SIZE = 2, previously returned batches
    # and mixed cached/uncached documents within the same batch retain exact offsets.
    monkeypatch.setattr(hk, "MAX_WORD_CACHE_SIZE", 2)
    b1 = fast_tokenize_subword(tok, ["Hello"])
    b2 = fast_tokenize_subword(tok, ["Hello", "a b"])
    ref_b1 = tok(["Hello"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    ref_b2 = tok(["Hello", "a b"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert b1.input_ids == ref_b1["input_ids"]
    assert b1.offset_mapping == ref_b1["offset_mapping"]
    assert b2.input_ids == ref_b2["input_ids"]
    assert b2.offset_mapping == ref_b2["offset_mapping"]
    monkeypatch.setattr(hk, "MAX_WORD_CACHE_SIZE", 65536)

    # 2. Same-type model replacement swapping "cat" and "dog" IDs while preserving vocab_size & probe
    tok._tokenizer.model = tokenizers_mod.models.WordLevel(
        {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
            "Hello": 4,
            "world.": 5,
            "Test": 6,
            "cat": 7,
            "dog": 8,
        },
        unk_token="<unk>",
    )
    assert fast_tokenize_subword(tok, ["cat"]).input_ids == [[7]]
    tok._tokenizer.model = tokenizers_mod.models.WordLevel(
        {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
            "Hello": 4,
            "world.": 5,
            "Test": 6,
            "cat": 8,
            "dog": 7,
        },
        unk_token="<unk>",
    )
    ref_swapped = tok(["cat"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert fast_tokenize_subword(tok, ["cat"]).input_ids == ref_swapped["input_ids"] == [[8]]

    # 3. Same-vocabulary BPE merge-table replacement (identical vocab, different merges)
    bpe_vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "Hello": 4,
        "world.": 5,
        "Test": 6,
        "a": 7,
        "b": 8,
        "ab": 9,
    }
    tok._tokenizer.model = tokenizers_mod.models.BPE(
        bpe_vocab,
        [("a", "b")],
        unk_token="<unk>",
    )
    b_merged = fast_tokenize_subword(tok, ["ab"])
    assert b_merged.input_ids == [[9]]
    assert b_merged.offset_mapping == [[(0, 2)]]

    tok._tokenizer.model = tokenizers_mod.models.BPE(
        bpe_vocab,
        [],
        unk_token="<unk>",
    )
    b_unmerged = fast_tokenize_subword(tok, ["ab"])
    ref_unmerged = tok(["ab"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert b_unmerged.input_ids == ref_unmerged["input_ids"] == [[7, 8]]
    assert b_unmerged.offset_mapping == ref_unmerged["offset_mapping"] == [[(0, 1), (1, 2)]]

    # Large-vocabulary (> 4096) BPE in-place property mutation (`model.dropout = 1.0` / `model.ignore_merges = True`)
    large_bpe_vocab = {**bpe_vocab, **{f"tok_{i}": 10 + i for i in range(5000)}}
    tok._tokenizer.model = tokenizers_mod.models.BPE(
        large_bpe_vocab,
        [("a", "b")],
        unk_token="<unk>",
    )
    assert fast_tokenize_subword(tok, ["ab"]).input_ids == [[9]]
    tok._tokenizer.model.dropout = 1.0
    assert _get_or_init_word_cache(tok) is None
    b_drop = fast_tokenize_subword(tok, ["ab"])
    ref_drop = tok(["ab"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert b_drop.input_ids == ref_drop["input_ids"] == [[7, 8]]

    # 4. Mutating backend normalizer to Lowercase invalidates the cache and matches reference
    tok._tokenizer.model = tokenizers_mod.models.WordLevel(
        {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "Hello": 4, "world.": 5, "Test": 6, "a": 7, "b": 8},
        unk_token="<unk>",
    )
    _ = fast_tokenize_subword(tok, ["Hello"])
    tok._tokenizer.normalizer = tokenizers_mod.normalizers.Lowercase()
    b_lower = fast_tokenize_subword(tok, ["Hello"])
    ref_lower = tok(["Hello"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert b_lower.input_ids == ref_lower["input_ids"] == [[3]]

    # 3. Cross-space Replace("a b", "Test") normalizer is rejected by _get_or_init_word_cache
    tok._tokenizer.normalizer = tokenizers_mod.normalizers.Replace("a b", "Test")
    assert _get_or_init_word_cache(tok) is None
    b_rep = fast_tokenize_subword(tok, ["a b"])
    ref_rep = tok(["a b"], return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    assert b_rep.input_ids == ref_rep["input_ids"] == [[6]]

    # 4. stitch_subword_logits with two equal-length documents having different window spans
    locs = np.array(
        [
            [0, 0, 3],
            [0, 2, 5],
            [1, 0, 3],
            [1, 1, 4],
            [1, 2, 5],
        ],
        dtype=np.int32,
    )
    rng_stitch = np.random.default_rng(19)
    raw = rng_stitch.standard_normal((5, 3, 1)).astype(np.float16)
    stitched = stitch_subword_logits(
        [raw],
        locs,
        [5, 5],
        block_size=3,
        num_labels=1,
        weighting="uniform",
        strip_cls_sep=False,
    )
    # Compare against multi-batch path (which uses per-document accumulation)
    stitched_multi_batch = stitch_subword_logits(
        [raw[:2], raw[2:]],
        locs,
        [5, 5],
        block_size=3,
        num_labels=1,
        weighting="uniform",
        strip_cls_sep=False,
    )
    assert np.array_equal(stitched[0].view(np.uint16), stitched_multi_batch[0].view(np.uint16))
    assert np.array_equal(stitched[1].view(np.uint16), stitched_multi_batch[1].view(np.uint16))

    # 5. Direct GigaPackedDoc(packed) constructor defaults to conservative has_zero_delta=True
    from wtpsplit._host_kernels import GigaPackedDoc, _slice_sentences_single_space, decode_subword_split

    direct_packed = np.array([[1, 2, 1], [2, 0, 1], [3, 2, 1]], dtype=np.int32)
    direct_doc = GigaPackedDoc(direct_packed)
    assert direct_doc.has_zero_delta is True
    direct_logits = np.array([[1.0], [-1.0], [-1.0]], dtype=np.float16)
    assert decode_subword_split(
        "a b",
        direct_logits,
        direct_doc,
        z_min=np.float16(0.0),
        split_on_input_newlines=False,
    ) == ["a b"]

    # 6. _slice_sentences_single_space matches indices_to_sentences on negative, duplicate, and trailing cuts
    for cut_seq in ([0, -3], [0, 0, 2], [1, 4], []):
        for strip_ws in (False, True):
            assert _slice_sentences_single_space(
                "ab cd",
                cut_seq,
                strip_whitespace=strip_ws,
            ) == indices_to_sentences("ab cd", cut_seq, strip_whitespace=strip_ws)


def _reference_get_extended_attention_mask(
    config: Any,
    attention_mask: torch.Tensor,
    input_shape: tuple[int, ...],
    lookahead: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if config.is_decoder and lookahead is None:
            from transformers.modeling_utils import ModuleUtilsMixin

            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(  # type: ignore[no-untyped-call]
                input_shape, attention_mask
            )
        elif lookahead is not None:
            _, seq_length = attention_mask.shape
            lookahead_mask = torch.tril(torch.ones(seq_length, seq_length), diagonal=lookahead).to(
                attention_mask.device
            )
            extended_attention_mask = attention_mask[:, None, None, :] * lookahead_mask
        else:
            extended_attention_mask = attention_mask[:, None, None, :].expand(-1, 1, attention_mask.size(1), -1)
    else:
        raise ValueError(f"Wrong shape {attention_mask.shape}")
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    return (1.0 - extended_attention_mask) * torch.finfo(dtype).min


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lookahead", [None, 0, 7, 48])
@pytest.mark.parametrize("is_padded", [False, True])
def test_step_a1_extended_attention_mask_byte_parity_and_storage(
    dtype: torch.dtype,
    lookahead: int | None,
    is_padded: bool,
) -> None:
    """Step A1 cached broadcast lookahead and padding masks match reference bytes and shrink storage."""
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.models import get_extended_attention_mask

    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=530,
        num_labels=1,
        lookahead=lookahead,
    )
    for seq_len in (16, 512, 520):
        batch_size = 8
        attn = torch.ones((batch_size, seq_len), dtype=dtype)
        if is_padded:
            attn[-2:, seq_len // 2 :] = 0
        ref = _reference_get_extended_attention_mask(cfg, attn, (batch_size, seq_len), lookahead, dtype)
        got = get_extended_attention_mask(cfg, attn, (batch_size, seq_len), lookahead, attn.device, dtype)
        assert got.shape == ref.shape == (batch_size, 1, seq_len, seq_len)
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))

        if not is_padded:
            assert got.stride(0) == 0
            if lookahead is None:
                assert got.stride(2) == 0
        elif lookahead is None:
            assert got.stride(2) == 0
            elem_size = torch.tensor([], dtype=dtype).element_size()
            assert got.untyped_storage().nbytes() == batch_size * seq_len * elem_size


def test_step_a1_encoder_and_split_layers_forward_byte_parity() -> None:
    """Step A1 produces byte-identical model logits across lookahead, split_layers, and edge cases."""
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.models import (
        _LOOKAHEAD_MASK_CACHE,
        _MAX_LOOKAHEAD_CACHE_ENTRIES,
        SubwordXLMForTokenClassification,
        get_extended_attention_mask,
    )

    torch.manual_seed(1234)
    for lookahead, split_layers in ((None, None), (12, None), (12, 1), (12, 2)):
        cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
            num_labels=1,
            lookahead=lookahead,
            lookahead_split_layers=split_layers,
            use_character_head=False,
        )
        model = SubwordXLMForTokenClassification(cfg).eval()  # type: ignore[no-untyped-call]
        input_ids = torch.randint(3, 120, (4, 24))
        for is_padded in (False, True):
            attn = torch.ones((4, 24), dtype=torch.float32)
            if is_padded:
                attn[-1, 15:] = 0.0
            with torch.no_grad():
                out_with_mask = model(input_ids=input_ids, attention_mask=attn).logits
                if not is_padded:
                    out_none_mask = model(input_ids=input_ids, attention_mask=None).logits
                    assert torch.equal(out_with_mask.view(torch.uint8), out_none_mask.view(torch.uint8))

    # 3D attention_mask and cache eviction bounds
    cfg_base = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2
    )
    mask_3d = torch.ones((2, 6, 6), dtype=torch.float32)
    mask_3d[1, :, 4:] = 0.0
    ref_3d = _reference_get_extended_attention_mask(cfg_base, mask_3d, (2, 6), None, torch.float32)
    got_3d = get_extended_attention_mask(cfg_base, mask_3d, (2, 6), None, mask_3d.device, torch.float32)
    assert torch.equal(got_3d.view(torch.uint8), ref_3d.view(torch.uint8))

    for la_val in range(_MAX_LOOKAHEAD_CACHE_ENTRIES + 8):
        _ = get_extended_attention_mask(
            cfg_base,
            torch.ones((1, 8), dtype=torch.float32),
            (1, 8),
            lookahead=la_val,
            dtype=torch.float32,
        )
    assert len(_LOOKAHEAD_MASK_CACHE) <= _MAX_LOOKAHEAD_CACHE_ENTRIES


class _ModuloEmbedSubwordModel(torch.nn.Module):
    """Lightweight deterministic PyTorch subword model for fast multi-batch Step A2 parity tests."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.config = cfg
        self.device = torch.device("cpu")
        self.emb = torch.nn.Embedding(128, 16)
        self.proj = torch.nn.Linear(16, int(cfg.num_labels))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        from wtpsplit.models import get_extended_attention_mask

        ext_mask = get_extended_attention_mask(
            self.config,
            attention_mask,
            input_ids.shape,
            self.config.lookahead,
            input_ids.device,
            torch.float32,
        )
        mod_ids = torch.remainder(input_ids, 128)
        hidden = self.emb(mod_ids) * attention_mask.unsqueeze(-1).to(torch.float32)
        # Sum attended positions per query row so masked <pad> columns contribute 0.0
        visible_count = (ext_mask[:, 0, :, :] == 0.0).to(torch.float32).sum(dim=-1, keepdim=True) / 16.0
        logits = self.proj(hidden) + visible_count
        return {"logits": logits}


@pytest.mark.parametrize("weighting", ["uniform", "hat"])
@pytest.mark.parametrize("pad_last_batch", [False, True])
@pytest.mark.parametrize("output_float32", [False, True])
def test_step_a2_deferred_d2h_and_device_cls_sep_slice_byte_parity(
    weighting: Literal["uniform", "hat"],
    pad_last_batch: bool,
    output_float32: bool,
) -> None:
    """Step A2 deferred coalesced D2H transfer and device-side CLS/SEP slice match per-batch reference bytes."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._host_kernels import extract_subword_logits_fast
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import PyTorchWrapper, extract

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "c": 6, "d": 7},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    torch.manual_seed(2026)
    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=514,
        num_labels=1,
        lookahead=4,
        use_character_head=False,
    )
    pt_model = _ModuloEmbedSubwordModel(cfg).eval()
    wrapper_fast = PyTorchWrapper(pt_model, output_float32=output_float32)

    # Force fallback per-batch model(...) path by attaching an instance __call__
    wrapper_ref = PyTorchWrapper(pt_model, output_float32=output_float32)
    wrapper_ref.__dict__["__call__"] = wrapper_ref.__call__

    texts = [
        " ".join(["a b c d"] * 12),
        " ".join(["b c d a"] * 15),
        " ".join(["c d a"] * 3),
        " ".join(["d a b c"] * 10),
        "",
    ]

    stitched_fast, _ = extract_subword_logits_fast(
        texts,
        wrapper_fast,
        tok,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
    )
    stitched_ref, _ = extract_subword_logits_fast(
        texts,
        wrapper_ref,
        tok,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
    )
    for a, b in zip(stitched_fast, stitched_ref):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    ext_fast, _, _, _ = extract(
        texts[:4],
        wrapper_fast,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        tokenizer=tok,
    )
    ext_ref, _, _, _ = extract(
        texts[:4],
        wrapper_ref,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        tokenizer=tok,
    )
    for a, b in zip(ext_fast, ext_ref):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize("weighting", ["uniform", "hat"])
@pytest.mark.parametrize("pad_last_batch", [False, True])
@pytest.mark.parametrize("downsampling_rate", [1, 2])
def test_step_a3_length_sorted_trimmed_subword_batching_parity(
    weighting: Literal["uniform", "hat"],
    pad_last_batch: bool,
    downsampling_rate: int,
) -> None:
    """Step A3 length-sorted dynamic width trimming shrinks batch widths and preserves exact outputs."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._host_kernels import extract_subword_logits_fast
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import PyTorchWrapper, extract

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "c": 6, "d": 7},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    torch.manual_seed(2026)
    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=514,
        num_labels=1,
        lookahead=4,
        downsampling_rate=downsampling_rate,
        use_character_head=False,
    )
    pt_model = _ModuloEmbedSubwordModel(cfg).eval()

    observed_widths_untrim: list[int] = []
    observed_widths_trim: list[int] = []

    class WidthRecorder(PyTorchWrapper):
        def __init__(self, inner: Any, bucket: list[int]) -> None:
            super().__init__(inner)
            self._bucket = bucket

        def forward_tensor(self, attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
            self._bucket.append(int(attention_mask.shape[1]))
            return super().forward_tensor(attention_mask, *args, **kwargs)

    w_untrim = WidthRecorder(pt_model, observed_widths_untrim)
    w_trim = WidthRecorder(pt_model, observed_widths_trim)
    # Attach to PyTorchWrapper type name so can_defer_d2h stays active
    WidthRecorder.__name__ = "PyTorchWrapper"

    # Interleave long multi-window and short single-window texts to exercise stable length sorting
    texts = [
        " ".join(["a b c d"] * 12),  # 48 tokens (multi-window)
        "a b",                        # 2 tokens (short)
        " ".join(["b c d a"] * 10),  # 40 tokens (multi-window)
        "c d a",                      # 3 tokens (short)
        "a",                          # 1 token (short)
        "",                           # empty
        "b c",                        # 2 tokens (short)
    ]

    s_untrim, _ = extract_subword_logits_fast(
        texts,
        w_untrim,
        tok,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        trim_subword_padding=False,
    )
    s_trim, _ = extract_subword_logits_fast(
        texts,
        w_trim,
        tok,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        trim_subword_padding=True,
    )
    assert min(observed_widths_trim) < min(observed_widths_untrim)
    for a, b in zip(s_untrim, s_trim):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    # Also verify fallback non-deferred path and extract()
    w_ref_trim = PyTorchWrapper(pt_model)
    w_ref_trim.__dict__["__call__"] = w_ref_trim.__call__
    s_ref_trim, _ = extract_subword_logits_fast(
        texts,
        w_ref_trim,
        tok,
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        trim_subword_padding=True,
    )
    for a, b in zip(s_untrim, s_ref_trim):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    e_untrim, _, _, _ = extract(
        texts[:5],
        PyTorchWrapper(pt_model),
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        tokenizer=tok,
        trim_subword_padding=False,
    )
    e_trim, _, _, _ = extract(
        texts[:5],
        PyTorchWrapper(pt_model),
        stride=8,
        max_block_size=16,
        batch_size=2,
        pad_last_batch=pad_last_batch,
        weighting=weighting,
        tokenizer=tok,
        trim_subword_padding=True,
    )
    for a, b in zip(e_untrim, e_trim):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize("pad_last_batch", [False, True])
@pytest.mark.parametrize("project_tokens_first", [False, True])
def test_step_a4_two_stage_character_head_ordinals_and_from_pretrained_parity(
    tmp_path: Any,
    pad_last_batch: bool,
    project_tokens_first: bool,
) -> None:
    """Step A4 two-stage character-head execution and 2D ordinals match 3D reference (including after from_pretrained)."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._char_preprocessing import extract_character_head_exact
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import PyTorchWrapper
    from wtpsplit.models import SubwordXLMForTokenClassification

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "hello": 4, "world": 5, ".": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    torch.manual_seed(42)
    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=20,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
        max_position_embeddings=514,
        lookahead=4,
        use_character_head=True,
        character_head_project_tokens_first=project_tokens_first,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model_init = SubwordXLMForTokenClassification(cfg).eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        assert model_init.character_head is not None
        model_init.character_head.delta.weight.normal_(std=0.05)
        model_init.classifier.bias.fill_(-4.0)
    model_init.save_pretrained(tmp_path)

    # Reload via from_pretrained to verify non-persistent _primes meta-device handling
    model = SubwordXLMForTokenClassification.from_pretrained(tmp_path).eval()  # type: ignore[no-untyped-call]
    assert model.character_head is not None

    wrapper_fast = PyTorchWrapper(model)
    wrapper_ref = PyTorchWrapper(model)
    wrapper_ref.__dict__["__call__"] = wrapper_ref.__call__

    texts = [
        "  hello world. hello world.  ",
        "中文。 hello. བོད་",
        "",
        "hello " * 10,
    ]

    # 1. Dense mode (gate_logit_threshold=None): 100% byte-identical
    model.character_head.gate_logit_threshold = None
    out_fast, _, _, enc_fast = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        wrapper_fast,
        stride=2,
        max_block_size=8,
        batch_size=3,
        pad_last_batch=pad_last_batch,
        tokenizer=tok,
        trim_padding=True,
    )
    out_ref, _, _, enc_ref = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        wrapper_ref,
        stride=2,
        max_block_size=8,
        batch_size=3,
        pad_last_batch=pad_last_batch,
        tokenizer=tok,
        trim_padding=True,
    )
    assert enc_fast["input_ids"] == enc_ref["input_ids"]
    assert enc_fast["offset_mapping"] == enc_ref["offset_mapping"]
    for a, b in zip(out_fast, out_ref):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    # 2. Gated mode (some tokens pass gate): matches single-stage gated reference
    model.character_head.gate_logit_threshold = -6.0
    out_gated_fast, _, _, _ = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        wrapper_fast,
        stride=2,
        max_block_size=8,
        batch_size=3,
        pad_last_batch=pad_last_batch,
        tokenizer=tok,
        trim_padding=True,
    )
    out_gated_ref, _, _, _ = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        wrapper_ref,
        stride=2,
        max_block_size=8,
        batch_size=3,
        pad_last_batch=pad_last_batch,
        tokenizer=tok,
        trim_padding=True,
    )
    for a, b in zip(out_gated_fast, out_gated_ref):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    # 3. High gate threshold (no tokens pass gate -> exercises fast bypass without char_hashes transfer)
    model.character_head.gate_logit_threshold = 100.0
    out_bypass_fast, _, _, _ = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        wrapper_fast,
        stride=2,
        max_block_size=8,
        batch_size=3,
        pad_last_batch=pad_last_batch,
        tokenizer=tok,
        trim_padding=True,
    )
    out_bypass_ref, _, _, _ = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        wrapper_ref,
        stride=2,
        max_block_size=8,
        batch_size=3,
        pad_last_batch=pad_last_batch,
        tokenizer=tok,
        trim_padding=True,
    )
    for a, b in zip(out_bypass_fast, out_bypass_ref):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


def test_step_b1_flex_attention_block_mask_parity_and_cache() -> None:
    """Step B1 FlexAttention BlockMask matches SDPA outputs across lookahead, split_layers, and padding."""
    from torch.nn.attention.flex_attention import BlockMask
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.model_registry import resolve_effective_lookahead
    from wtpsplit.models import (
        _FLEX_BLOCK_MASK_CACHE,
        _MAX_FLEX_CACHE_ENTRIES,
        SubwordXLMForTokenClassification,
        get_extended_attention_mask,
    )

    torch.manual_seed(99)
    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=1,
        lookahead=8,
        lookahead_split_layers=None,
        use_character_head=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = SubwordXLMForTokenClassification(cfg).eval()  # type: ignore[no-untyped-call]
    input_ids = torch.randint(3, 60, (2, 16))

    for lookahead, split_layers, is_padded in (
        (None, None, False),
        (0, None, False),
        (8, None, False),
        (8, 1, False),
        (8, 1, True),
    ):
        for c in [model.config, model.roberta.config, model.roberta.encoder.config] + [
            layer.attention.self.config for layer in model.roberta.encoder.layer
        ]:
            c.lookahead = lookahead
            c.lookahead_split_layers = split_layers
        model.roberta.lookahead_split_layers = split_layers
        model.roberta.effective_lookahead = resolve_effective_lookahead(model.config, supports_split_layers=True)

        attn = torch.ones((2, 16), dtype=torch.float32)
        if is_padded:
            attn[1, 10:] = 0.0

        for c in [model.config, model.roberta.config, model.roberta.encoder.config] + [
            layer.attention.self.config for layer in model.roberta.encoder.layer
        ]:
            c._attn_implementation = "sdpa"
        with torch.no_grad():
            out_sdpa = model(input_ids=input_ids, attention_mask=attn).logits

        for c in [model.config, model.roberta.config, model.roberta.encoder.config] + [
            layer.attention.self.config for layer in model.roberta.encoder.layer
        ]:
            c._attn_implementation = "flex_attention"
        eff_la = model.roberta.effective_lookahead
        bm = get_extended_attention_mask(model.config, attn, (2, 16), eff_la, attn.device, torch.float32)
        assert isinstance(bm, BlockMask)
        if not is_padded:
            bm2 = get_extended_attention_mask(model.config, attn, (2, 16), eff_la, attn.device, torch.float32)
            assert bm is bm2

        with torch.no_grad():
            out_flex = model(input_ids=input_ids, attention_mask=attn).logits

        valid_mask = attn.bool().unsqueeze(-1)
        max_diff = float(((out_sdpa - out_flex).abs() * valid_mask).max().item())
        assert max_diff <= 1e-5

    # Fallback when attention_mask[:, 0] == 0 or lookahead < 0 returns additive Tensor
    attn_all_zero = torch.zeros((2, 16), dtype=torch.float32)
    fallback_mask = get_extended_attention_mask(model.config, attn_all_zero, (2, 16), 8, attn_all_zero.device, torch.float32)
    assert isinstance(fallback_mask, torch.Tensor)

    # Cache bound check
    for la_val in range(_MAX_FLEX_CACHE_ENTRIES + 6):
        _ = get_extended_attention_mask(
            model.config,
            torch.ones((1, 8), dtype=torch.float32),
            (1, 8),
            lookahead=la_val,
            dtype=torch.float32,
        )
    assert len(_FLEX_BLOCK_MASK_CACHE) <= _MAX_FLEX_CACHE_ENTRIES


def test_step_b2_bucketed_compile_and_cudagraph_clone_parity() -> None:
    """Step B2 bucketed widths, Dynamo-disabled mask/primes helpers, and compiled clone protection preserve parity."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._host_kernels import (
        COMPILE_WIDTH_BUCKETS,
        extract_subword_logits_fast,
        round_to_compile_bucket,
    )
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import PyTorchWrapper

    wtpsplit_mod: Any = importlib.import_module("wtpsplit")

    assert round_to_compile_bucket(10, 512, 1) == 64
    assert round_to_compile_bucket(64, 512, 1) == 64
    assert round_to_compile_bucket(65, 512, 1) == 128
    assert round_to_compile_bucket(200, 512, 1) == 256
    assert round_to_compile_bucket(300, 512, 1) == 512
    assert round_to_compile_bucket(15, 20, 3) == 20

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "c": 6, "d": 7},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    torch.manual_seed(2026)
    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=514,
        num_labels=1,
        lookahead=4,
        use_character_head=False,
    )
    pt_model = _ModuloEmbedSubwordModel(cfg).eval()

    observed_widths: list[int] = []

    class BucketRecorder(PyTorchWrapper):
        def __init__(self, inner: Any) -> None:
            super().__init__(inner)

        def forward_tensor(self, attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
            observed_widths.append(int(attention_mask.shape[1]))
            return super().forward_tensor(attention_mask, *args, **kwargs)

    BucketRecorder.__name__ = "PyTorchWrapper"
    w_comp = BucketRecorder(pt_model)
    wtpsplit_mod._configure_pytorch_model(w_comp, compile={"backend": "eager"})
    assert getattr(w_comp, "_compiled", False) is True

    w_eager = PyTorchWrapper(pt_model)
    texts = [
        " ".join(["a b c d"] * 25),  # 100 tokens
        "a b",                        # 2 tokens
        " ".join(["b c d a"] * 20),  # 80 tokens
        "c d a",                      # 3 tokens
    ]
    s_eager, _ = extract_subword_logits_fast(
        texts,
        w_eager,
        tok,
        stride=64,
        max_block_size=128,
        batch_size=2,
        trim_subword_padding=False,
    )
    s_comp, _ = extract_subword_logits_fast(
        texts,
        w_comp,
        tok,
        stride=64,
        max_block_size=128,
        batch_size=2,
        trim_subword_padding=True,
    )
    assert all(w in COMPILE_WIDTH_BUCKETS or w == 102 for w in observed_widths)
    assert observed_widths[0] == 64
    for a, b in zip(s_eager, s_comp):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


def test_step_b3_varlen_unpadded_sequence_packing_parity() -> None:
    """Step B3 unpadded 1D varlen sequence packing matches padded SubwordXLMForTokenClassification across lookahead, split_layers, and FlexAttention."""
    import importlib
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._host_kernels import extract_subword_logits_fast
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import PyTorchWrapper
    from wtpsplit.model_registry import resolve_effective_lookahead
    from wtpsplit.models import SubwordXLMForTokenClassification

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4, "b": 5, "c": 6, "d": 7},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    torch.manual_seed(2027)
    cfg = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,
        lookahead=8,
        lookahead_split_layers=None,
        use_character_head=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = SubwordXLMForTokenClassification(cfg).eval()  # type: ignore[no-untyped-call]
    wrapper = PyTorchWrapper(model, output_float32=True)

    texts = [
        " ".join(["a b c d"] * 8),  # 32 tokens
        "a b",                       # 2 tokens
        "c d a b c",                 # 5 tokens
        " ".join(["b c d a"] * 6),  # 24 tokens
        "",
    ]

    for lookahead, split_layers, attn_impl in (
        (None, None, "sdpa"),
        (8, None, "sdpa"),
        (8, 1, "sdpa"),
        (8, 1, "flex_attention"),
    ):
        for c in [model.config, model.roberta.config, model.roberta.encoder.config] + [
            layer.attention.self.config for layer in model.roberta.encoder.layer
        ]:
            c.lookahead = lookahead
            c.lookahead_split_layers = split_layers
            c._attn_implementation = attn_impl
        model.roberta.lookahead_split_layers = split_layers
        model.roberta.effective_lookahead = resolve_effective_lookahead(model.config, supports_split_layers=True)

        s_pad, _ = extract_subword_logits_fast(
            texts,
            wrapper,
            tok,
            stride=16,
            max_block_size=32,
            batch_size=4,
            pack_sequences=False,
        )
        s_varlen, _ = extract_subword_logits_fast(
            texts,
            wrapper,
            tok,
            stride=16,
            max_block_size=32,
            batch_size=4,
            pack_sequences=True,
        )
        for a, b in zip(s_pad, s_varlen):
            if len(a) > 0:
                max_diff = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
                assert max_diff <= 1e-3
            else:
                assert len(b) == 0


def test_step_b4_ort_wrapper_giga_tokenization_and_jit_tracing_parity() -> None:
    """Step B4 SaTORTWrapper fast path (rank-2/rank-3 char_hashes, consolidated stitch) and TorchScript dynamic tracing match PyTorch."""
    import importlib
    from types import SimpleNamespace
    from transformers import PreTrainedTokenizerFast
    from wtpsplit._char_preprocessing import extract_character_head_exact
    from wtpsplit._host_kernels import extract_subword_logits_fast
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import PyTorchWrapper, SaTORTWrapper, extract
    from wtpsplit.models import SubwordXLMForTokenClassification

    tokenizers_mod: Any = importlib.import_module("tokenizers")
    backend = tokenizers_mod.Tokenizer(
        tokenizers_mod.models.WordLevel(
            {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "hello": 4, "world": 5, ".": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = tokenizers_mod.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers_mod.processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tok: Any = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
    )

    # 1. Verify TorchScript tracing of get_extended_attention_mask on unpadded dummy input
    # preserves dynamic sequence length and padding mask responsiveness
    for lookahead in (None, 4):
        torch.manual_seed(101)
        cfg_trace = SubwordXLMConfig(  # type: ignore[no-untyped-call]
            vocab_size=20,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=1,
            max_position_embeddings=64,
            lookahead=lookahead,
            use_character_head=False,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        m_trace = SubwordXLMForTokenClassification(cfg_trace).eval()  # type: ignore[no-untyped-call]

        class TraceExport(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                out: torch.Tensor = self.inner(input_ids=input_ids, attention_mask=attention_mask).logits
                return out

        # Trace on an UN-padded (2, 6) dummy input
        dummy_ids = torch.randint(3, 15, (2, 6))
        dummy_attn = torch.ones((2, 6), dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(TraceExport(m_trace), (dummy_ids, dummy_attn))  # type: ignore[no-untyped-call]

            # Evaluate traced module on a DIFFERENT shape (3, 12) with padding
            test_ids = torch.randint(3, 15, (3, 12))
            test_attn = torch.ones((3, 12), dtype=torch.float32)
            test_attn[2, 7:] = 0.0
            expected = m_trace(input_ids=test_ids, attention_mask=test_attn).logits
            actual = traced(test_ids, test_attn)
            assert torch.equal(expected, actual)

    # 2. Verify SaTORTWrapper parity on both subword (multi-batch consolidated stitch) and character_head (rank-3 and rank-2 char_hashes)
    torch.manual_seed(202)
    cfg_char = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=20,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
        max_position_embeddings=64,
        lookahead=4,
        use_character_head=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    m_char = SubwordXLMForTokenClassification(cfg_char).eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        assert m_char.character_head is not None
        m_char.character_head.delta.weight.normal_(std=0.05)

    class MockORTSession:
        def __init__(self, pt_mod: Any, hash_rank: int = 3, is_char: bool = True) -> None:
            self.pt_mod = pt_mod
            self.hash_rank = hash_rank
            self.is_char = is_char
            self.last_feeds: dict[str, Any] = {}

        def get_inputs(self) -> list[Any]:
            base = [
                SimpleNamespace(name="input_ids", type="tensor(int64)", shape=["B", "T"]),
                SimpleNamespace(name="attention_mask", type="tensor(float)", shape=["B", "T"]),
            ]
            if not self.is_char:
                return base
            hash_type = "tensor(int32)" if self.hash_rank == 2 else "tensor(int64)"
            hash_shape = ["B", "C"] if self.hash_rank == 2 else ["B", "C", 8]
            return base + [
                SimpleNamespace(name="char_to_token", type="tensor(int64)", shape=["B", "C"]),
                SimpleNamespace(name="char_is_token_final", type="tensor(int64)", shape=["B", "C"]),
                SimpleNamespace(name="char_position_in_token", type="tensor(int64)", shape=["B", "C"]),
                SimpleNamespace(name="char_hashes", type=hash_type, shape=hash_shape),
                SimpleNamespace(name="char_mask", type="tensor(int64)", shape=["B", "C"]),
            ]

        def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.last_feeds = feeds
            with torch.no_grad():
                t_feeds = {k: torch.from_numpy(v) for k, v in feeds.items()}
                out = self.pt_mod(**t_feeds).logits.numpy()
            return [out]

    texts = [
        "  hello world. hello world.  ",
        "中文。 hello. བོད་",
        "",
        "hello " * 10,
    ]
    pt_wrapper = PyTorchWrapper(m_char)
    out_pt, _, _, _ = extract_character_head_exact(  # type: ignore[no-untyped-call]
        texts,
        pt_wrapper,
        stride=2,
        max_block_size=8,
        batch_size=2,
        tokenizer=tok,
        trim_padding=True,
    )

    for rank in (3, 2):
        sess = MockORTSession(m_char, hash_rank=rank, is_char=True)
        ort_wrapper = SaTORTWrapper(cfg_char, sess)
        out_ort, _, _, _ = extract_character_head_exact(  # type: ignore[no-untyped-call]
            texts,
            ort_wrapper,
            stride=2,
            max_block_size=8,
            batch_size=2,
            tokenizer=tok,
            trim_padding=True,
        )
        assert sess.last_feeds["char_hashes"].ndim == rank
        for a, b in zip(out_pt, out_ort):
            assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    # Subword multi-batch consolidated stitch via SaTORTWrapper
    cfg_sub = SubwordXLMConfig(  # type: ignore[no-untyped-call]
        vocab_size=20,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
        max_position_embeddings=64,
        lookahead=4,
        use_character_head=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    m_sub = SubwordXLMForTokenClassification(cfg_sub).eval()  # type: ignore[no-untyped-call]
    ort_sub = SaTORTWrapper(cfg_sub, MockORTSession(m_sub, is_char=False))
    pt_sub = PyTorchWrapper(m_sub)
    s_ort, _ = extract_subword_logits_fast(
        texts,
        ort_sub,
        tok,
        stride=2,
        max_block_size=8,
        batch_size=2,
    )
    s_pt, _ = extract_subword_logits_fast(
        texts,
        pt_sub,
        tok,
        stride=2,
        max_block_size=8,
        batch_size=2,
    )
    for a, b in zip(s_ort, s_pt):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))

    e_ort, _, _, _ = extract(
        texts[:3],
        ort_sub,
        stride=2,
        max_block_size=8,
        batch_size=2,
        tokenizer=tok,
    )
    e_pt, _, _, _ = extract(
        texts[:3],
        pt_sub,
        stride=2,
        max_block_size=8,
        batch_size=2,
        tokenizer=tok,
    )
    for a, b in zip(e_ort, e_pt):
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16))













