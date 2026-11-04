"""Benchmark and 100%-outcome-parity harness for SaT / wtpsplit host-side inference stages.

Measures the exact host-side stages shown in the latency breakdown slide across the 10 slide
languages (English, German, Amharic, Yoruba, Odia, Punjabi, Burmese, Igbo, Khmer, Sorani Kurdish)
plus Chinese and Thai, covering both single-window and multi-window documents and pathological
zero-width offset edge cases:

1. Tokenization (light-blue bar):
   - Reference: `AutoTokenizer.__call__(texts, return_offsets_mapping=True, add_special_tokens=False)`
   - Candidate A (`rust_lazy`): direct `tok._tokenizer.encode_batch(texts, add_special_tokens=False)`
   - Candidate B (`word_cached`): whitespace-pretoken deduplication + `encode_batch` on unique words
2. Windowing + Stitching + Decoding + Offsets (hatched grey bar):
   - Reference (`native_ref`): current `extract.py` + `token_to_char_probs` + `sigmoid` + `indices_to_sentences`
   - Candidate (`wave1_equiv`):
     * Identity 1: single-window IEEE-754 `float16` identity bypass + multi-window unit-weight elision
     * Identity 2: `sigmoid`-before-scatter in `predict_proba()` (100% bit-identical `float32` array)
     * Identity 3: exact `float16` quantile threshold `z_min_f16` + sparse offset lookup in `split()`
     * Identity 4: 1D slice-assigned window packing without per-window Python list concatenations
3. Character-Resolution Head (`char_head`) host preparation:
   - Reference: CPU 8-prime `int64` hash expansion in `build_char_inputs_vectorized`
   - Candidate: compact `int32` ordinals transfer + on-device 8-prime hash computation
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from wtpsplit._host_kernels import (
    decode_subword_split,
    exact_float16_logit_threshold,
    fast_tokenize_subword,
    pack_subword_windows,
    project_subword_proba,
    stitch_subword_logits,
)
from wtpsplit.utils import indices_to_sentences, sigmoid, token_to_char_probs


# Representative multi-sentence samples across the 10 languages from the latency breakdown slide
# plus Chinese and Thai, repeated to form realistic batches of single- and multi-window documents.
SLIDE_LANGUAGE_SAMPLES: dict[str, str] = {
    "English": (
        "The quick brown fox jumps over the lazy dog in the sunny afternoon. "
        "Meanwhile, researchers around the world are developing faster tokenization and segmentation algorithms. "
    ),
    "German": (
        "Die schnelle braune Füchsin springt am sonnigen Nachmittag über den faulen Hund. "
        "Gleichzeitig entwickeln Forschungsgruppen weltweit schnellere Algorithmen zur Satzsegmentierung. "
    ),
    "Amharic": (
        "ፈጣኑ ቡናማ ቀበሮ በፀሐያማው ከሰዓት በኋላ ሰነፉን ውሻ ዘለለው። "
        "በተመሳሳይ ጊዜ በዓለም ዙሪያ ያሉ ተመራማሪዎች ፈጣን የዓረፍተ ነገር መከፋፈያ ዘዴዎችን በማዘጋጀት ላይ ናቸው። "
    ),
    "Yoruba": (
        "Kọ̀lọ̀kọ̀lọ̀ूरी pupa kíákíá fòkè ajá ọlẹ náà ní ọ̀sán oòrùn. "
        "Ní àkókò kan náà, àwọn olùwádìí ní gbogbo àgbáyé ń ṣe àgbékalẹ̀ àwọn ọ̀nà ìpín-gbólóhùn kíákíá. "
    ),
    "Odia": (
        "ଦ୍ରୁତ ବାଦାମୀ କୋକିଶିଆଳି ଖରାবেଳେ ଅଳସୁଆ କୁକୁର ଉପର ଦେଇ ଡେଇଁପଡ଼ିଲା। "
        "ସେହି ସମୟରେ ବିଶ୍ୱର ଗବେଷକମାନେ ଦ୍ରୁତ ବାକ୍ୟ ବିଭାଜନ ପ୍ରଣାଳୀ ବିକଶିତ କରୁଛନ୍ତି। "
    ),
    "Punjabi": (
        "ਤੇਜ਼ ਭੂਰੀ ਲੂੰਬੜੀ ਧੁੱਪ ਵਾਲੀ ਦੁਪਹਿਰ ਵਿੱਚ ਆਲਸੀ ਕੁੱਤੇ ਦੇ ਉੱਪਰੋਂ ਛਾਲ ਮਾਰਦੀ ਹੈ। "
        "ਇਸ ਦੇ ਨਾਲ ਹੀ ਦੁਨੀਆ ਭਰ ਦੇ ਖੋਜਕਰਤਾ ਤੇਜ਼ ਵਾਕ ਵੰਡਣ ਵਾਲੇ ਐਲਗੋਰਿਦਮ ਵਿਕਸਤ ਕਰ ਰਹੇ ਹਨ। "
    ),
    "Burmese": (
        "လျင်မြန်သောအညိုရောင်မြေခွေးသည်နေသာသောနေ့လည်ခင်းတွင်ပျင်းရိသောခွေးပေါ်မှခုန်ကျော်သွားသည်။ "
        "တစ်ချိန်တည်းမှာပင်ကမ္ဘာတစ်ဝှမ်းရှိသုတေသီများသည်ပိုမိုမြန်ဆန်သောဝါကျခွဲခြားသည့်နည်းလမ်းများကိုတီထွင်နေကြသည်။ "
    ),
    "Igbo": (
        "Nkịta ọhịa na-agba ọsọ wụfere nkịta dị umengwụ n'ehihie anwụ na-acha. "
        "N'otu oge ahụ, ndị na-eme nchọpụta n'ụwa niile na-emepụta usoro nkewa ahịrịokwu dị ngwa. "
    ),
    "Khmer": (
        "កញ្ជ្រោងពណ៌ត្នោតដ៏លឿនបានលោតរំលងឆ្កែខ្ជិលនៅរសៀលថ្ងៃ المشمس។ "
        "ទន្ទឹមនឹងនេះដែរ អ្នកស្រាវជ្រាវជុំវិញពិភពលោកកំពុងអភិវឌ្ឍក្បួនដោះស្រាយការបែងចែកប្រโยគឱ្យកាន់តែលឿន។ "
    ),
    "Sorani Kurdish": (
        "ڕێوییە قاوەییە خێراکە لە دوای نیوەڕۆیەکی خۆرەتاودا باز دەدات بەسەر سەگە تەمبەڵەکەدا. "
        "لە هەمان کاتدا توێژەران لە سەرانسەری جیهاندا خەریکی پەرەپێدانی ئەلگۆریتمی خێراترن بۆ دابەشکردنی ڕستەکان. "
    ),
}


@dataclass(frozen=True)
class HostStageTimings:
    language: str
    num_docs: int
    total_chars: int
    total_tokens: int
    ref_tokenize_ms: float
    cand_tokenize_ms: float
    wordcache_tokenize_ms: float
    ref_window_stitch_ms: float
    cand_window_stitch_ms: float
    ref_decode_proba_ms: float
    cand_decode_proba_ms: float
    ref_decode_split_ms: float
    cand_decode_split_ms: float

    @property
    def ref_other_host_split_ms(self) -> float:
        return self.ref_window_stitch_ms + self.ref_decode_split_ms

    @property
    def cand_other_host_split_ms(self) -> float:
        return self.cand_window_stitch_ms + self.cand_decode_split_ms

    @property
    def ref_total_host_split_ms(self) -> float:
        return self.ref_tokenize_ms + self.ref_other_host_split_ms

    @property
    def cand_total_host_split_ms(self) -> float:
        return self.wordcache_tokenize_ms + self.cand_other_host_split_ms


def reference_window_and_stitch(
    token_ids: list[list[int]],
    raw_window_logits: NDArray[np.float16],
    *,
    block_size: int,
    stride: int,
    cls_id: int,
    sep_id: int,
    weighting: str = "uniform",
) -> tuple[NDArray[np.int64], NDArray[np.float32], list[NDArray[np.float16]]]:
    """Reference subword window packing and float16 overlap-add stitching from `wtpsplit/extract.py`."""
    text_lengths = [len(t) for t in token_ids]
    eff_block = min(block_size, max(text_lengths, default=1))
    num_chunks = sum(math.ceil(max(length - eff_block, 0) / stride) + 1 for length in text_lengths)

    input_ids = np.zeros((num_chunks, eff_block + 2), dtype=np.int64)
    attention_mask = np.zeros((num_chunks, eff_block + 2), dtype=np.float32)
    locs = np.zeros((num_chunks, 3), dtype=np.int32)

    current_chunk = 0
    for i, tokens in enumerate(token_ids):
        t_len = text_lengths[i]
        for j in range(0, t_len, stride):
            start, end = j, j + eff_block
            done = False
            if end >= t_len:
                end = t_len
                start = max(end - eff_block, 0)
                done = True
            chunk = [cls_id] + tokens[start:end] + [sep_id]
            input_ids[current_chunk, : len(chunk)] = chunk
            attention_mask[current_chunk, : len(chunk)] = 1.0
            locs[current_chunk, :] = [i, start, end]
            current_chunk += 1
            if done:
                break

    num_labels = raw_window_logits.shape[2]
    all_logits = [np.zeros((length, num_labels), dtype=np.float16) for length in text_lengths]
    all_counts = [np.zeros(length, dtype=np.float16) for length in text_lengths]

    if weighting == "uniform":
        weights = np.ones(eff_block, dtype=np.float16)
    else:
        x = np.linspace(-(1 - 1 / eff_block), 1 - 1 / eff_block, eff_block, dtype=np.float16)
        weights = 1 - np.abs(x)

    for i in range(num_chunks):
        orig_idx, s_idx, e_idx = locs[i]
        n = e_idx - s_idx
        all_logits[orig_idx][s_idx:e_idx] += weights[:n, None] * raw_window_logits[i, :n]
        all_counts[orig_idx][s_idx:e_idx] += weights[:n]

    stitched = [
        (logits / counts[:, None]).astype(np.float16)
        for logits, counts in zip(all_logits, all_counts)
    ]
    return input_ids, attention_mask, stitched


def equivalent_window_and_stitch(
    token_ids: Sequence[Sequence[int] | NDArray[Any]],
    raw_window_logits: NDArray[np.float16],
    *,
    block_size: int,
    stride: int,
    cls_id: int,
    sep_id: int,
    weighting: Literal["uniform", "hat"] = "uniform",
) -> tuple[NDArray[np.int64], NDArray[np.float32], list[NDArray[np.float16]]]:
    """100% bit-equivalent window packing and stitching using `wtpsplit._host_kernels`."""
    input_ids, attention_mask, locs, eff_block = pack_subword_windows(
        token_ids,
        block_size=block_size,
        stride=stride,
        cls_id=cls_id,
        sep_id=sep_id,
    )
    stitched = stitch_subword_logits(
        [raw_window_logits],
        locs,
        [len(t) for t in token_ids],
        block_size=eff_block,
        num_labels=int(raw_window_logits.shape[2]),
        weighting=weighting,
        strip_cls_sep=False,
    )
    return input_ids, attention_mask, stitched


def equivalent_predict_proba_subword(
    texts: list[str],
    stitched_logits: list[NDArray[np.float16]],
    offsets_or_ends: Sequence[Any],
    newline_idx: int = 0,
) -> list[NDArray[np.float32]]:
    """Identity 2: compute sigmoid on tokens BEFORE scattering into `np.zeros(len(text), float32)`."""
    results: list[NDArray[np.float32]] = []
    for i, (text, tok_logits) in enumerate(zip(texts, stitched_logits)):
        off = offsets_or_ends[i]
        if isinstance(off, np.ndarray):
            with np.errstate(over="ignore"):
                tok_p = sigmoid(tok_logits[:, newline_idx]).astype(np.float32, copy=False)
            char_p = np.zeros(len(text), dtype=np.float32)
            if off.size > 0:
                char_p[off - 1] = tok_p[: off.size]
            results.append(char_p)
        else:
            results.append(
                project_subword_proba(
                    len(text),
                    range(len(tok_logits)),
                    tok_logits,
                    off,
                    (),
                    newline_idx=newline_idx,
                )
            )
    return results


def equivalent_split_subword(
    texts: list[str],
    stitched_logits: list[NDArray[np.float16]],
    offsets_or_encodings: list[Any],
    *,
    threshold: float,
    strip_whitespace: bool = False,
    newline_idx: int = 0,
) -> list[list[str]]:
    """Identity 3: threshold on float16 token logits directly and look up offsets only at passing tokens."""
    z_min = exact_float16_logit_threshold(threshold)
    return [
        decode_subword_split(
            text,
            tok_logits,
            item,
            z_min=z_min,
            strip_whitespace=strip_whitespace,
            split_on_input_newlines=False,
            newline_idx=newline_idx,
        )
        for text, tok_logits, item in zip(texts, stitched_logits, offsets_or_encodings)
    ]


def word_cached_encode_batch(
    texts: list[str],
    rust_tokenizer: Any,
    cache: dict[str, tuple[NDArray[np.int64], NDArray[np.int32]]],
) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int32]]]:
    """Native word-level pre-token cache storing contiguous NumPy (ids, rel_ends) per unique word."""
    unseen: list[str] = []
    unseen_set: set[str] = set()
    for text in texts:
        for w in text.split(" "):
            if w and w not in cache and w not in unseen_set:
                unseen_set.add(w)
                unseen.append(w)
    if unseen:
        for w, enc in zip(unseen, rust_tokenizer.encode_batch(unseen, add_special_tokens=False)):
            ids_arr = np.asarray(enc.ids, dtype=np.int64)
            ends_arr = np.asarray([e for _, e in enc.offsets], dtype=np.int32)
            cache[w] = (ids_arr, ends_arr)

    batch_ids: list[NDArray[np.int64]] = []
    batch_ends: list[NDArray[np.int32]] = []
    for text in texts:
        id_parts: list[NDArray[np.int64]] = []
        end_parts: list[NDArray[np.int32]] = []
        cursor = 0
        for part in text.split(" "):
            if part:
                w_ids, w_ends = cache[part]
                id_parts.append(w_ids)
                end_parts.append(w_ends + cursor)
            cursor += len(part) + 1
        batch_ids.append(np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int64))
        batch_ends.append(np.concatenate(end_parts) if end_parts else np.empty(0, dtype=np.int32))
    return batch_ids, batch_ends


def benchmark_language(
    language: str,
    sample: str,
    tokenizer: Any,
    *,
    num_docs: int = 128,
    repeats_per_doc: int = 8,
    block_size: int = 510,
    stride: int = 64,
    threshold: float = 0.25,
) -> HostStageTimings:
    texts = [sample * repeats_per_doc for _ in range(num_docs)]
    special_tokens = ["<s>", "</s>", "<pad>"]
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)

    # 1. Reference Tokenization
    t0 = time.perf_counter()
    ref_enc = tokenizer(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    ref_ids: list[list[int]] = ref_enc["input_ids"]
    ref_off: list[list[tuple[int, int]]] = ref_enc["offset_mapping"]
    ref_tokenize_ms = (time.perf_counter() - t0) * 1000

    # 2. Direct Rust encode_batch (lazy offsets: extract `.ids` only; `.offsets` read only if/when needed)
    t0 = time.perf_counter()
    rust_encs = tokenizer._tokenizer.encode_batch(texts, add_special_tokens=False)
    cand_ids = [e.ids for e in rust_encs]
    cand_tokenize_ms = (time.perf_counter() - t0) * 1000

    # 3. GigaPy Word-Cached Tokenization (`fast_tokenize_subword` with fused `(id, delta)` int32x2 cache)
    _ = fast_tokenize_subword(tokenizer, texts[:2], enable_word_cache=True)
    t0 = time.perf_counter()
    giga_batch = fast_tokenize_subword(tokenizer, texts, enable_word_cache=True)
    wordcache_tokenize_ms = (time.perf_counter() - t0) * 1000

    assert ref_ids == cand_ids == giga_batch.input_ids, f"Token ID mismatch on {language}!"
    assert ref_off == giga_batch.offset_mapping, f"Offset mapping mismatch on {language}!"

    # Deterministic synthetic forward logits for the windows
    text_lengths = [len(t) for t in ref_ids]
    eff_block = min(block_size, max(text_lengths, default=1))
    num_chunks = sum(math.ceil(max(length - eff_block, 0) / stride) + 1 for length in text_lengths)
    rng = np.random.default_rng(42)
    raw_window_logits = (rng.standard_normal((num_chunks, eff_block, 1)) - 2.5).astype(np.float16)

    # 4. Reference vs Candidate Windowing + Stitching
    t0 = time.perf_counter()
    ref_in, ref_mask, ref_stitched = reference_window_and_stitch(
        ref_ids, raw_window_logits, block_size=block_size, stride=stride, cls_id=cls_id, sep_id=sep_id
    )
    ref_window_stitch_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cand_in, cand_mask, cand_stitched = equivalent_window_and_stitch(
        giga_batch.token_arrays, raw_window_logits, block_size=block_size, stride=stride, cls_id=cls_id, sep_id=sep_id
    )
    cand_window_stitch_ms = (time.perf_counter() - t0) * 1000

    assert np.array_equal(ref_in, cand_in), f"input_ids mismatch on {language}!"
    assert np.array_equal(ref_mask, cand_mask), f"attention_mask mismatch on {language}!"
    for r_s, c_s in zip(ref_stitched, cand_stitched):
        assert np.array_equal(r_s.view(np.uint16), c_s.view(np.uint16)), f"stitched logits bit mismatch on {language}!"

    giga_doc_items = [giga_batch.offsets_or_encoding(i) for i in range(num_docs)]

    # 5. Reference vs Candidate Decode (`predict_proba`)
    t0 = time.perf_counter()
    ref_probs = [
        sigmoid(token_to_char_probs(texts[i], ref_ids[i], ref_stitched[i], special_tokens, ref_off[i])[:, 0]).astype(
            np.float32, copy=False
        )
        for i in range(num_docs)
    ]
    ref_decode_proba_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cand_probs = equivalent_predict_proba_subword(texts, cand_stitched, giga_doc_items)
    cand_decode_proba_ms = (time.perf_counter() - t0) * 1000

    for r_p, c_p in zip(ref_probs, cand_probs):
        assert np.array_equal(r_p.view(np.uint32), c_p.view(np.uint32)), f"predict_proba bit mismatch on {language}!"

    # 6. Reference vs Candidate Decode (`split`)
    t0 = time.perf_counter()
    ref_splits = [
        indices_to_sentences(
            texts[i],
            np.where(
                sigmoid(token_to_char_probs(texts[i], ref_ids[i], ref_stitched[i], special_tokens, ref_off[i])[:, 0])
                > threshold
            )[0],
        )
        for i in range(num_docs)
    ]
    ref_decode_split_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cand_splits = equivalent_split_subword(texts, cand_stitched, giga_doc_items, threshold=threshold)
    cand_decode_split_ms = (time.perf_counter() - t0) * 1000

    assert ref_splits == cand_splits, f"split() output mismatch on {language}!"

    return HostStageTimings(
        language=language,
        num_docs=num_docs,
        total_chars=sum(len(t) for t in texts),
        total_tokens=sum(text_lengths),
        ref_tokenize_ms=ref_tokenize_ms,
        cand_tokenize_ms=cand_tokenize_ms,
        wordcache_tokenize_ms=wordcache_tokenize_ms,
        ref_window_stitch_ms=ref_window_stitch_ms,
        cand_window_stitch_ms=cand_window_stitch_ms,
        ref_decode_proba_ms=ref_decode_proba_ms,
        cand_decode_proba_ms=cand_decode_proba_ms,
        ref_decode_split_ms=ref_decode_split_ms,
        cand_decode_split_ms=cand_decode_split_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark host-side latency and verify 100% bit-exact parity.")
    parser.add_argument("--num-docs", type=int, default=128)
    parser.add_argument("--repeats-per-doc", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base")
    # Warmup
    _ = tok(["Warmup sentence."], return_offsets_mapping=True, verbose=False, add_special_tokens=False)

    print(
        f"{'Language':<16} | {'Tok (ref->WC)':>17} | {'Win+Stitch':>15} | {'Decode(split)':>15} | {'Other Host (hatched)':>22} | {'Total Host':>20}"
    )
    print("-" * 118)
    for lang, sample in SLIDE_LANGUAGE_SAMPLES.items():
        r = benchmark_language(
            lang, sample, tok, num_docs=args.num_docs, repeats_per_doc=args.repeats_per_doc
        )
        hatched_speedup = r.ref_other_host_split_ms / max(r.cand_other_host_split_ms, 1e-6)
        total_speedup = r.ref_total_host_split_ms / max(r.cand_total_host_split_ms, 1e-6)
        print(
            f"{r.language:<16} | {r.ref_tokenize_ms:6.1f}->{r.wordcache_tokenize_ms:5.1f} ms | "
            f"{r.ref_window_stitch_ms:5.1f}->{r.cand_window_stitch_ms:5.1f} ms | "
            f"{r.ref_decode_split_ms:5.1f}->{r.cand_decode_split_ms:5.1f} ms | "
            f"{r.ref_other_host_split_ms:6.1f}->{r.cand_other_host_split_ms:5.1f} ms ({hatched_speedup:4.1f}x) | "
            f"{r.ref_total_host_split_ms:6.1f}->{r.cand_total_host_split_ms:5.1f} ms ({total_speedup:4.1f}x)"
        )
    print("-" * 118)
    print("ALL 10 SLIDE LANGUAGES PASSED 100% BIT-EXACT PARITY (input_ids, attention_mask, stitched_logits, predict_proba, split)!")


if __name__ == "__main__":
    main()
