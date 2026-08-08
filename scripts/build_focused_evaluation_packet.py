#!/usr/bin/env python
"""Build a small, non-gold candidate packet from BOUQuET development data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.evaluation.diagnostics.boundary_ceiling import (
    BOUQUET_REVISION,
    dominant_script,
    load_bouquet,
    separator_for,
)

DEFAULT_LANGUAGES = ("roh_Latn", "ibo_Latn", "ssw_Latn", "tha_Thai", "khm_Khmr")


def make_row(
    language: str,
    document_number: int,
    sentences: list[str],
    split: str = "dev",
    source_revision: str = BOUQUET_REVISION,
) -> dict:
    script = dominant_script("".join(sentences))
    separator = separator_for(language, script)
    text = separator.join(sentences)
    offsets = []
    cursor = 0
    for sentence in sentences[:-1]:
        cursor += len(sentence) + len(separator)
        offsets.append(cursor)
    return {
        "id": f"bouquet-{split}-{language}-{document_number:03d}",
        "language_script": language,
        "script": script,
        "slice": "romansh" if language == "roh_Latn" else "reported_under_splitting",
        "source": "facebook/bouquet",
        "source_version": source_revision,
        "license": "CC-BY-4.0",
        "split": split,
        "reference_type": "partial",
        "annotation_status": "candidate_not_gold",
        "text": text,
        "gold_offsets": offsets,
        "notes": "Known inter-segment boundaries only; within-segment sentence boundaries may be missing.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--documents-per-language", type=int, default=5)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--revision", default=BOUQUET_REVISION)
    parser.add_argument(
        "--arrow-file",
        type=Path,
        help="Optional cached bouquet-dev.arrow; avoids Hub/cache writes in a restricted environment.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    args.output = args.output or Path(
        f"data/evaluation/focused_bouquet_{args.split}_candidates_v1.jsonl"
    )
    args.manifest = args.manifest or Path(
        "data/manifests/focused_eval_candidates_v1.json"
        if args.split == "dev"
        else "data/manifests/focused_eval_test_candidates_v1.json"
    )
    wanted = set(args.languages)
    rows = []
    if args.arrow_file:
        import datasets

        frame = datasets.Dataset.from_file(str(args.arrow_file)).to_pandas()
        loaded = []
        for language in sorted(frame["src_lang"].unique()):
            part = frame[frame["src_lang"] == language]
            documents = []
            for _, paragraph in part.groupby("par_id"):
                sentences = [preprocess_sentence(text) for text in paragraph["src_text"]]
                sentences = [sentence for sentence in sentences if sentence]
                if len(sentences) >= 2:
                    documents.append(sentences)
            if documents:
                loaded.append((language, "bouquet", documents))
    else:
        loaded = load_bouquet(args.split, None, revision=args.revision)
    for language, _, documents in loaded:
        if language in wanted:
            rows.extend(
                make_row(
                    language,
                    index,
                    document,
                    args.split,
                    args.revision,
                )
                for index, document in enumerate(documents[: args.documents_per_language])
            )
    found = {row["language_script"] for row in rows}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Languages absent from local BOUQuET dev data: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(content, encoding="utf-8")
    manifest = {
        "version": (
            "focused_eval_candidates_v1"
            if args.split == "dev"
            else "focused_eval_test_candidates_v1"
        ),
        "status": "candidate_not_gold",
        "source": f"facebook/bouquet sentence_level {args.split}",
        "source_revision": args.revision,
        "output": str(args.output),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "documents": len(rows),
        "by_language_script": {language: sum(row["language_script"] == language for row in rows) for language in sorted(wanted)},
        "next_action": "Double-annotate Romansh and real OCR; adjudicate exact character offsets before paper scoring.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
