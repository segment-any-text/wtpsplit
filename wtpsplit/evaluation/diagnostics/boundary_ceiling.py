"""Measure the tokenization-induced upper bound on sentence-boundary recall.

`token_to_char_probs` in `wtpsplit/utils/__init__.py` initialises every character
position to `-inf` and only assigns logits to the *last character of each token*.
A boundary that falls anywhere else is therefore unreachable at any threshold: the
model cannot emit it. That caps recall independently of how well the model is
trained, and the cap depends entirely on the tokenizer.

This script computes, per language, the fraction of gold boundaries that a given
tokenizer leaves reachable. Comparing tokenizers (XLM-R's SentencePiece against
mmBERT's Gemma 2 vocabulary) shows how much of a recall gap a backbone swap fixes
for free, and how much needs an architectural fix.

Usage:
    python -m wtpsplit.evaluation.diagnostics.boundary_ceiling \
        --source ud --ud-path data/external/ud-treebanks-v2.18 \
        --tokenizers facebookAI/xlm-roberta-base jhu-clsp/mmBERT-base \
        --output data/diagnostics/ceiling_ud.csv
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.utils import Constants

BOUQUET_DATASET = "facebook/bouquet"
BOUQUET_REVISION = "9a6070a9652e350dda1d353c4fd198533199a911"

# Scripts written without spaces between words. `Constants.SEPARATORS` only covers the
# 85 languages in language_info.csv and silently defaults to " " for everything else,
# which would be wrong for these.
NO_WHITESPACE_SCRIPTS = {"CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN"}


@dataclass
class LanguageCeiling:
    """Per-language result. `ceiling` is the headline number."""

    corpus: str
    lang: str
    treebank: str
    script: str
    tokenizer: str
    separator: str
    n_docs: int
    n_boundaries: int
    n_reachable: int
    ceiling: float
    chars_per_token: float
    unk_rate: float
    n_chars: int


def dominant_script(text: str) -> str:
    """Coarse script label from the most common non-ASCII letter."""
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts[name.split()[0]] += 1
    if not counts:
        return "UNKNOWN"
    return counts.most_common(1)[0][0]


def to_iso639_1(code: str) -> str | None:
    """`deu_Latn` -> `de`, `th` -> `th`.

    Falls back through `prefer_macrolanguage()` so individual languages map onto the
    macrolanguage SaT knows: `cmn_Hans` and `yue_Hant` both become `zh`.
    """
    base = code.split("_")[0]
    # Keep training/evaluation correct without making the optional `langcodes`
    # package a runtime requirement. These include every legacy no-whitespace
    # language plus Thai, whose separator must remain a space.
    manual = {
        "zho": "zh",
        "cmn": "zh",
        "yue": "zh",
        "wuu": "zh",
        "lzh": "zh",
        "jpn": "ja",
        "khm": "km",
        "mya": "my",
        "tha": "th",
        "kor": "ko",
        "arb": "ar",
        "pes": "fa",
        "khk": "mn",
    }
    if base in manual:
        return manual[base]
    if len(base) == 2:
        return base
    try:
        import langcodes
    except ImportError:
        return None
    try:
        language = langcodes.Language.get(base)
    except Exception:
        return None

    candidates = [language]
    try:
        candidates.append(language.prefer_macrolanguage())
    except Exception:
        pass
    for candidate in candidates:
        tag = getattr(candidate, "language", None)
        if tag and len(tag) == 2:
            return tag
    return None


def separator_for(lang: str, script: str) -> str:
    """Separator used to join gold sentences into a document.

    `Constants.SEPARATORS` is authoritative wherever it has an entry, so BOUQuET's
    NLLB-style codes are mapped to ISO 639-1 first. This matters: Thai is marked
    `no_whitespace=False` in `language_info.csv` because Thai *does* use spaces at
    sentence boundaries even though it has none between words. Falling back to a
    script-based guess would join Thai sentences with `""` and destroy the boundary cue,
    which understates Thai badly.

    The script heuristic applies only to languages `language_info.csv` has never heard of.
    """
    iso = to_iso639_1(lang)
    if iso and iso in Constants.LANG_CODE_TO_INDEX:
        return Constants.SEPARATORS[iso]
    return "" if script in NO_WHITESPACE_SCRIPTS else " "


def boundary_candidates(sentences: Sequence[str], separator: str) -> list[set[int]]:
    """Character indices at which a probability spike yields the correct split.

    Mirrors `indices_to_sentences`: an index `i` fires a boundary at `i + 1`, after
    which trailing whitespace is consumed. So for a boundary between sentence `k` and
    `k + 1`, any index from the last content character through the last separator
    character reconstructs the same split. With an empty separator there is exactly
    one such index, which is why no-whitespace scripts are the fragile case.

    Only internal boundaries are returned; end-of-text is appended unconditionally by
    `indices_to_sentences` and is free for every model.
    """
    candidates = []
    pos = 0
    for sentence in sentences[:-1]:
        end_content = pos + len(sentence)  # exclusive
        candidates.append(set(range(end_content - 1, end_content + len(separator))))
        pos = end_content + len(separator)
    return candidates


def reachable_positions(text: str, tokenizer) -> tuple[set[int], int, int]:
    """Character indices that can carry a logit, plus token and unk counts.

    `token_to_char_probs` writes each token's logits to `offset[1] - 1`.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    reachable = set()
    for start, end in encoding["offset_mapping"]:
        if end > start:
            reachable.add(end - 1)
    n_unk = sum(1 for i in encoding["input_ids"] if i == tokenizer.unk_token_id)
    return reachable, len(encoding["input_ids"]), n_unk


def measure(
    documents: Sequence[Sequence[str]],
    separator: str,
    tokenizer,
) -> tuple[int, int, int, int, int]:
    """Returns (n_boundaries, n_reachable, n_chars, n_tokens, n_unk)."""
    n_boundaries = n_reachable = n_chars = n_tokens = n_unk = 0
    for sentences in documents:
        if len(sentences) < 2:
            continue
        text = separator.join(sentences)
        reachable, tokens, unks = reachable_positions(text, tokenizer)
        for candidates in boundary_candidates(sentences, separator):
            n_boundaries += 1
            if candidates & reachable:
                n_reachable += 1
        n_chars += len(text)
        n_tokens += tokens
        n_unk += unks
    return n_boundaries, n_reachable, n_chars, n_tokens, n_unk


# --------------------------------------------------------------------------------------
# Corpus loaders. Each yields (lang, treebank_or_subset, [document, ...]) where a
# document is a list of gold sentences.
# --------------------------------------------------------------------------------------


def _chunk(sentences: list[str], per_doc: int) -> list[list[str]]:
    return [sentences[i : i + per_doc] for i in range(0, len(sentences), per_doc)]


def load_ud(
    ud_path: Path, sentences_per_doc: int, max_sentences: int
) -> Iterator[tuple[str, str, list[list[str]]]]:
    """Read `# text = ` lines from every treebank's test split (dev as fallback)."""
    treebanks = sorted(d for d in ud_path.iterdir() if d.is_dir() and d.name.startswith("UD_"))
    if not treebanks:
        raise FileNotFoundError(f"No UD_* directories under {ud_path}")

    for treebank in tqdm(treebanks, desc="UD treebanks"):
        conllu = sorted(treebank.glob("*-ud-test.conllu")) or sorted(treebank.glob("*-ud-dev.conllu"))
        if not conllu:
            continue
        lang = conllu[0].name.split("_")[0]

        sentences: list[str] = []
        with open(conllu[0], encoding="utf-8") as f:
            for line in f:
                if not line.startswith("# text = "):
                    continue
                sentence = preprocess_sentence(line[len("# text = ") :])
                if sentence:
                    sentences.append(sentence)
                if len(sentences) >= max_sentences:
                    break

        if len(sentences) < 2:
            continue
        yield lang, treebank.name, _chunk(sentences, sentences_per_doc)


def load_bouquet(
    split: str,
    max_langs: int | None,
    revision: str | None = BOUQUET_REVISION,
) -> Iterator[tuple[str, str, list[list[str]]]]:
    """BOUQuET is sentence-parallel across 275 varieties, grouped by `par_id`.

    Gated on the Hub, so this needs `huggingface-cli login` plus accepting the terms.
    Segments are joined per paragraph, which is exactly the document a segmenter sees.
    """
    import datasets

    ds = datasets.load_dataset(
        BOUQUET_DATASET,
        "sentence_level",
        split=split,
        revision=revision,
    )
    frame = ds.to_pandas()

    langs = sorted(frame["src_lang"].unique())
    if max_langs:
        langs = langs[:max_langs]

    for lang in tqdm(langs, desc="BOUQuET languages"):
        part = frame[frame["src_lang"] == lang]
        documents = []
        for _, paragraph in part.groupby("par_id"):
            sentences = [preprocess_sentence(t) for t in paragraph["src_text"]]
            sentences = [s for s in sentences if s]
            if len(sentences) >= 2:
                documents.append(sentences)
        if documents:
            yield lang, "bouquet", documents


def load_flores(split: str, max_langs: int | None) -> Iterator[tuple[str, str, list[list[str]]]]:
    """FLORES+ passages are contiguous, so consecutive sentences form real documents."""
    import datasets

    ds = datasets.load_dataset("openlanguagedata/flores_plus", split=split)
    frame = ds.to_pandas()

    langs = sorted(frame["iso_639_3"].astype(str) + "_" + frame["iso_15924"].astype(str))
    langs = sorted(set(langs))
    if max_langs:
        langs = langs[:max_langs]

    frame["lang"] = frame["iso_639_3"].astype(str) + "_" + frame["iso_15924"].astype(str)
    for lang in tqdm(langs, desc="FLORES languages"):
        part = frame[frame["lang"] == lang]
        documents = []
        for _, group in part.groupby("URL"):
            sentences = [preprocess_sentence(t) for t in group["text"]]
            sentences = [s for s in sentences if s]
            if len(sentences) >= 2:
                documents.append(sentences)
        if documents:
            yield lang, "flores_plus", documents


def load_pth(path: Path, sentences_per_doc: int) -> Iterator[tuple[str, str, list[list[str]]]]:
    """The repo's own eval format: {lang: {"sentence": {dataset: {"data": [...]}}}}."""
    import torch

    data = torch.load(path, weights_only=True)
    for lang, lang_data in tqdm(data.items(), desc="pth languages"):
        for dataset_name, dataset in lang_data.get("sentence", {}).items():
            rows = dataset.get("data") or []
            if not rows:
                continue
            if isinstance(rows[0], list):
                documents = [[preprocess_sentence(s) for s in doc] for doc in rows]
            else:
                documents = _chunk([preprocess_sentence(s) for s in rows], sentences_per_doc)
            documents = [d for d in documents if len(d) >= 2]
            if documents:
                yield lang, dataset_name, documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["ud", "bouquet", "flores", "pth"], required=True)
    parser.add_argument("--ud-path", type=Path, default=Path("data/external/ud-treebanks-v2.18"))
    parser.add_argument("--pth-path", type=Path, default=Path("data/all_data.pth"))
    parser.add_argument("--split", default="dev", help="split for bouquet/flores")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["facebookAI/xlm-roberta-base", "jhu-clsp/mmBERT-base"],
    )
    parser.add_argument("--sentences-per-doc", type=int, default=10)
    parser.add_argument("--max-sentences", type=int, default=2000, help="per treebank, UD only")
    parser.add_argument("--max-langs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/boundary_ceiling.csv"))
    args = parser.parse_args(argv)

    # Surveying many tokenizers means some will be gated or need an unavailable
    # dependency. Skip those rather than losing the whole sweep to one bad name.
    tokenizers = {}
    for name in args.tokenizers:
        print(f"loading tokenizer {name}", file=sys.stderr)
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as error:  # noqa: BLE001 - any load failure is a skip
            print(f"  skipping {name}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        # The ceiling is defined over character offsets, which only fast tokenizers
        # report. Slow-only ones (ByT5 and friends) cannot be measured this way.
        if not tokenizer.is_fast:
            print(f"  skipping {name}: no fast implementation, so no offset mapping", file=sys.stderr)
            continue
        tokenizers[name] = tokenizer
    if not tokenizers:
        print("no tokenizers could be loaded", file=sys.stderr)
        return 1

    if args.source == "ud":
        corpus = load_ud(args.ud_path, args.sentences_per_doc, args.max_sentences)
    elif args.source == "bouquet":
        corpus = load_bouquet(args.split, args.max_langs)
    elif args.source == "flores":
        corpus = load_flores(args.split, args.max_langs)
    else:
        corpus = load_pth(args.pth_path, args.sentences_per_doc)

    results: list[LanguageCeiling] = []
    for lang, subset, documents in corpus:
        joined = " ".join(documents[0])[:2000]
        script = dominant_script(joined)
        separator = separator_for(lang, script)

        for name, tokenizer in tokenizers.items():
            n_bound, n_reach, n_chars, n_tokens, n_unk = measure(documents, separator, tokenizer)
            if n_bound == 0:
                continue
            results.append(
                LanguageCeiling(
                    corpus=args.source,
                    lang=lang,
                    treebank=subset,
                    script=script,
                    tokenizer=name,
                    separator=repr(separator),
                    n_docs=len(documents),
                    n_boundaries=n_bound,
                    n_reachable=n_reach,
                    ceiling=n_reach / n_bound,
                    chars_per_token=n_chars / max(n_tokens, 1),
                    unk_rate=n_unk / max(n_tokens, 1),
                    n_chars=n_chars,
                )
            )

    if not results:
        print("no results produced", file=sys.stderr)
        return 1

    import pandas as pd

    frame = pd.DataFrame([asdict(r) for r in results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\nwrote {len(frame)} rows to {args.output}")

    print("\n=== ceiling by tokenizer ===")
    for name, group in frame.groupby("tokenizer"):
        weighted = group["n_reachable"].sum() / group["n_boundaries"].sum()
        print(
            f"{name:35s} macro={group['ceiling'].mean():.4f} micro={weighted:.4f} "
            f"below_99%={int((group['ceiling'] < 0.99).sum()):4d}/{len(group)}"
        )

    print("\n=== worst 25 (by lowest ceiling across tokenizers) ===")
    pivot = frame.pivot_table(index=["lang", "treebank", "script"], columns="tokenizer", values="ceiling")
    pivot["min"] = pivot.min(axis=1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pivot.sort_values("min").head(25).round(4))

    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "source": args.source,
        "n_languages": int(frame["lang"].nunique()),
        "by_tokenizer": {
            name: {
                "macro_ceiling": float(group["ceiling"].mean()),
                "micro_ceiling": float(group["n_reachable"].sum() / group["n_boundaries"].sum()),
                "n_below_99": int((group["ceiling"] < 0.99).sum()),
                "n_below_95": int((group["ceiling"] < 0.95).sum()),
            }
            for name, group in frame.groupby("tokenizer")
        },
        "by_script": {
            script: {
                name: float(sub["n_reachable"].sum() / sub["n_boundaries"].sum())
                for name, sub in group.groupby("tokenizer")
            }
            for script, group in frame.groupby("script")
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
