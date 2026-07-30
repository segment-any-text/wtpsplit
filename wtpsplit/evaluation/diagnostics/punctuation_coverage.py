"""Audit how much sentence-terminal punctuation the auxiliary objective can see.

SaT's auxiliary task predicts which punctuation character occurs at a position, using
the set in `wtpsplit/data/punctuation.txt` filtered to characters the tokenizer can
represent as a single token (`punctuation_xlmr_unk.txt`, produced by
`wtpsplit/utils/remove_unks.py`). Characters that fail either step never become a
training signal.

Two distinct gaps matter for a massively multilingual successor:

1. Characters in `punctuation.txt` that the tokenizer maps to `<unk>`.
2. Sentence terminators that are missing from `punctuation.txt` altogether, which is
   the larger gap once coverage extends past the original 85 languages.

The reference set is Unicode's `Sentence_Terminal` property, so the audit is complete
by construction rather than by enumeration of languages we happen to know about.

Usage:
    python -m wtpsplit.evaluation.diagnostics.punctuation_coverage \
        --tokenizers facebookAI/xlm-roberta-base jhu-clsp/mmBERT-base \
        --output data/diagnostics/punctuation_coverage.csv
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import regex
from transformers import AutoTokenizer

from wtpsplit.utils import Constants


def unicode_sentence_terminators() -> list[str]:
    """Every character with Unicode property Sentence_Terminal=Yes."""
    return [chr(c) for c in range(0x110000) if regex.match(r"\p{Sentence_Terminal}", chr(c))]


def script_of(char: str) -> str:
    for script in ["Latin", "Greek", "Cyrillic", "Armenian", "Hebrew", "Arabic", "Syriac", "Thaana", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam", "Sinhala", "Thai", "Lao", "Tibetan", "Myanmar", "Georgian", "Hangul", "Ethiopic", "Cherokee", "Khmer", "Mongolian", "Han", "Hiragana", "Katakana", "Yi", "Tagalog", "Buginese", "Balinese", "Javanese", "Sundanese", "Batak", "Lepcha", "Ol_Chiki", "Vai", "Bamum", "Tifinagh", "Adlam", "Nko", "Samaritan", "Mandaic", "Coptic", "Runic", "Ogham", "Limbu", "Tai_Le", "New_Tai_Lue", "Buhid", "Hanunoo", "Tagbanwa", "Cham", "Kayah_Li", "Rejang", "Saurashtra", "Sylotinagri", "Phags_Pa"]:
        if regex.match(rf"\p{{Script={script}}}", char):
            return script
    if regex.match(r"\p{Common}", char):
        return "Common"
    if regex.match(r"\p{Inherited}", char):
        return "Inherited"
    return "Other"


def char_name(char: str) -> str:
    try:
        return unicodedata.name(char)
    except ValueError:
        return f"U+{ord(char):04X}"


def is_single_token(char: str, tokenizer) -> bool:
    """Match `remove_unks.py`: a character survives only as a whole vocabulary entry."""
    return tokenizer.convert_tokens_to_ids(char) != tokenizer.unk_token_id


def load_punctuation_file(name: str) -> list[str]:
    path = Path(Constants.ROOT_DIR) / "data" / name
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_extended_set() -> list[str]:
    """`punctuation.txt` plus every Unicode sentence terminator it is missing.

    Existing entries keep their order and position so the set stays readable as a diff of
    the original; new terminators are appended in codepoint order for determinism.

    This is written to *new* files rather than replacing `punctuation.txt`, because the
    auxiliary head is sized as `AUX_OFFSET + 1 + len(PUNCTUATION_CHARS)` — editing the
    original in place would silently invalidate every released `sat-*` checkpoint.
    """
    base = load_punctuation_file("punctuation.txt")
    known = set(base)
    additions = sorted((c for c in unicode_sentence_terminators() if c not in known), key=ord)
    return base + additions


def write_filtered(chars: list[str], tokenizer, stem: str, *, with_unk: bool) -> Path:
    """Write the subset the tokenizer can represent as single tokens.

    Mirrors `scripts/build_punctuation_vocab.py`: characters the tokenizer cannot encode
    as one token are dropped, optionally collapsed into a single `<unk>` bucket.
    """
    path = Path(Constants.ROOT_DIR) / "data" / f"{stem}.txt"
    added_unk = False
    with open(path, "w", encoding="utf-8") as handle:
        for char in chars:
            if is_single_token(char, tokenizer):
                handle.write(char + "\n")
            elif with_unk and not added_unk:
                handle.write("<unk>\n")
                added_unk = True
    return path


def report_token_collisions(chars: list[str], tokenizer, name: str) -> int:
    """Count characters that share a token id with an earlier character.

    `get_subword_label_dict` keys the auxiliary label map by token id, so two characters
    mapping to the same id collide and the later one silently wins. XLM-R normalises some
    forms (fullwidth `！` folds onto `!`), so this is not hypothetical.
    """
    seen: dict[int, str] = {}
    collisions = 0
    for char in chars:
        if not is_single_token(char, tokenizer):
            continue
        token_id = tokenizer.convert_tokens_to_ids(char)
        if token_id in seen and seen[token_id] != char:
            collisions += 1
        else:
            seen[token_id] = char
    if collisions:
        print(f"  WARNING {name}: {collisions} characters collide on a shared token id")
    return collisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["facebookAI/xlm-roberta-base", "jhu-clsp/mmBERT-base"],
    )
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/punctuation_coverage.csv"))
    parser.add_argument(
        "--write-extended",
        action="store_true",
        help="write punctuation_extended.txt (punctuation.txt plus every missing Unicode "
        "sentence terminator) and its per-tokenizer filtered variants",
    )
    parser.add_argument(
        "--write-punctuation-for",
        default=None,
        help="tokenizer name; regenerates punctuation_<tag>.txt / _unk.txt under wtpsplit/data",
    )
    parser.add_argument("--tag", default="mmbert", help="filename tag used with --write-punctuation-for")
    args = parser.parse_args(argv)

    tokenizers = {}
    for name in args.tokenizers:
        print(f"loading tokenizer {name}", file=sys.stderr)
        tokenizers[name] = AutoTokenizer.from_pretrained(name)

    terminators = unicode_sentence_terminators()
    base_set = set(load_punctuation_file("punctuation.txt"))
    xlmr_set = set(load_punctuation_file("punctuation_xlmr_unk.txt"))

    import pandas as pd

    rows = []
    for char in terminators:
        row: dict[str, object] = {
            "char": char,
            "codepoint": f"U+{ord(char):04X}",
            "name": char_name(char),
            "script": script_of(char),
            "in_punctuation_txt": char in base_set,
            "in_punctuation_xlmr_unk_txt": char in xlmr_set,
        }
        for name, tokenizer in tokenizers.items():
            row[f"single_token::{name}"] = is_single_token(char, tokenizer)
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows to {args.output}\n")

    total = len(frame)
    print("=== Unicode sentence terminators: how many are usable as an auxiliary signal? ===")
    print(f"{'total Sentence_Terminal characters':52s} {total:4d}")
    print(f"{'  present in punctuation.txt':52s} {int(frame['in_punctuation_txt'].sum()):4d}")
    print(f"{'  present in punctuation_xlmr_unk.txt':52s} {int(frame['in_punctuation_xlmr_unk_txt'].sum()):4d}")
    for name in tokenizers:
        col = f"single_token::{name}"
        both = int((frame["in_punctuation_txt"] & frame[col]).sum())
        print(f"{'  single token in ' + name:52s} {int(frame[col].sum()):4d}   (and in punctuation.txt: {both})")

    print("\n=== gap 1: in punctuation.txt but lost to the <unk> filter ===")
    for name in tokenizers:
        lost = frame[frame["in_punctuation_txt"] & ~frame[f"single_token::{name}"]]
        chars = " ".join(lost["char"].tolist())
        print(f"{name}: {len(lost)} lost -> {chars}")

    print("\n=== gap 2: sentence terminators absent from punctuation.txt entirely ===")
    missing = frame[~frame["in_punctuation_txt"]]
    by_script = missing.groupby("script").size().sort_values(ascending=False)
    print(f"{len(missing)} of {total} terminators are not in punctuation.txt")
    print("\nby script:")
    for script, count in by_script.items():
        sample = " ".join(missing[missing["script"] == script]["char"].tolist()[:12])
        recoverable = {
            name: int(missing[missing["script"] == script][f"single_token::{name}"].sum()) for name in tokenizers
        }
        rec = " ".join(f"{n.split('/')[-1]}={v}" for n, v in recoverable.items())
        print(f"  {script:14s} {count:3d}  [{rec}]  {sample}")

    if args.write_extended:
        extended = build_extended_set()
        base = load_punctuation_file("punctuation.txt")
        path = Path(Constants.ROOT_DIR) / "data" / "punctuation_extended.txt"
        path.write_text("".join(c + "\n" for c in extended), encoding="utf-8")

        print("\n=== extended punctuation set ===")
        print(f"  punctuation.txt          : {len(base)} characters")
        print(f"  punctuation_extended.txt : {len(extended)} characters (+{len(extended) - len(base)})")

        print(f"\n{'tokenizer':38s} {'usable now':>11s} {'usable extended':>16s} {'num_labels':>11s}")
        for name, tokenizer in tokenizers.items():
            tag = "mmbert" if "mmBERT" in name else "xlmr"
            written = write_filtered(extended, tokenizer, f"punctuation_extended_{tag}_unk", with_unk=True)
            write_filtered(extended, tokenizer, f"punctuation_extended_{tag}", with_unk=False)

            before = sum(1 for c in base if is_single_token(c, tokenizer))
            after = sum(1 for c in extended if is_single_token(c, tokenizer))
            entries = len(load_punctuation_file(written.name))
            print(f"{name:38s} {before:11d} {after:16d} {Constants.AUX_OFFSET + 1 + entries:11d}")
            report_token_collisions(extended, tokenizer, name)

        terminators = set(unicode_sentence_terminators())
        print("\n  sentence terminators specifically:")
        for name, tokenizer in tokenizers.items():
            before = sum(1 for c in base if c in terminators and is_single_token(c, tokenizer))
            after = sum(1 for c in extended if c in terminators and is_single_token(c, tokenizer))
            print(f"    {name:36s} {before} -> {after}")

    if args.write_punctuation_for:
        tokenizer = tokenizers.get(args.write_punctuation_for) or AutoTokenizer.from_pretrained(
            args.write_punctuation_for
        )
        data_dir = Path(Constants.ROOT_DIR) / "data"
        base_chars = load_punctuation_file("punctuation.txt")

        plain = data_dir / f"punctuation_{args.tag}.txt"
        with open(plain, "w", encoding="utf-8") as f:
            for char in base_chars:
                if is_single_token(char, tokenizer):
                    f.write(char + "\n")

        with_unk = data_dir / f"punctuation_{args.tag}_unk.txt"
        added_unk = False
        with open(with_unk, "w", encoding="utf-8") as f:
            for char in base_chars:
                if is_single_token(char, tokenizer):
                    f.write(char + "\n")
                elif not added_unk:
                    f.write("<unk>\n")
                    added_unk = True

        print(f"\nwrote {plain} and {with_unk}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
