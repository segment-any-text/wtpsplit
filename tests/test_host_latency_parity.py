"""Exhaustive outcome-parity tests for the host-latency speedup identities."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import pytest

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





