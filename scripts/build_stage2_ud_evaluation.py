#!/usr/bin/env python
"""Build an exact-reference Stage 2 evaluation packet from pinned local UD files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from wtpsplit.evaluation.diagnostics.boundary_ceiling import dominant_script, separator_for


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def conllu_sentences(path: Path) -> list[str]:
    sentences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# text = "):
            text = line.removeprefix("# text = ").strip()
            if text:
                sentences.append(text)
    if not sentences:
        raise ValueError(f"no '# text = ' sentences found in {path}")
    return sentences


def packet_rows(
    sentences: list[str],
    *,
    language_script: str,
    repository: str,
    release: str,
    license_name: str,
    split: str,
    sentences_per_document: int,
) -> list[dict]:
    rows = []
    for start in range(0, len(sentences), sentences_per_document):
        group = sentences[start : start + sentences_per_document]
        if len(group) < 2:
            continue
        script = dominant_script("".join(group))
        separator = separator_for(language_script, script)
        text = separator.join(group)
        offsets = []
        cursor = 0
        for sentence in group[:-1]:
            cursor += len(sentence) + len(separator)
            offsets.append(cursor)
        rows.append(
            {
                "id": f"ud218-{repository}-{split}-{start // sentences_per_document:05d}",
                "language_script": language_script,
                "script": script,
                "slice": "ud_exact",
                "source": repository,
                "source_version": release,
                "license": license_name,
                "split": split,
                "reference_type": "exact",
                "annotation_status": "upstream_exact_reference",
                "text": text,
                "gold_offsets": offsets,
            }
        )
    return rows


def locate_split(root: Path, row: dict, split: str) -> Path | None:
    path = root / row["repository"] / f"{row['treebank_stem']}-ud-{split}.conllu"
    return path if path.is_file() else None


def build_packet(
    selection: dict,
    root: Path,
    sentences_per_document: int,
    languages: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    rows = []
    sources = []
    for source in selection["rows"]:
        if source.get("selection_status") != "frozen":
            continue
        language = source["language_script"]
        if languages is not None and language not in languages:
            continue
        source_record = {
            "language_script": language,
            "repository": source["repository"],
            "release": source["release_ref"],
            "license": source["source_license"],
            "splits": {},
        }
        for split in ("dev", "test"):
            path = locate_split(root, source, split)
            if path is None:
                source_record["splits"][split] = "missing"
                continue
            sentences = conllu_sentences(path)
            built = packet_rows(
                sentences,
                language_script=language,
                repository=source["repository"],
                release=source["release_ref"],
                license_name=source["source_license"],
                split=split,
                sentences_per_document=sentences_per_document,
            )
            rows.extend(built)
            source_record["splits"][split] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "sentences": len(sentences),
                "documents": len(built),
            }
        sources.append(source_record)
    return rows, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/manifests/ud_2_18_frozen_selection_v1.json"),
    )
    parser.add_argument(
        "--ud-root",
        type=Path,
        default=Path("data/external/ud-treebanks-v2.18-frozen"),
    )
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--sentences-per-document", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/stage2_ud_2_18_exact_v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/stage2_ud_2_18_exact_v1.json"),
    )
    args = parser.parse_args()
    if args.sentences_per_document < 2:
        raise ValueError("--sentences-per-document must be at least two")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows, sources = build_packet(
        selection,
        args.ud_root,
        args.sentences_per_document,
        set(args.language) or None,
    )
    if not any(row["split"] == "dev" for row in rows):
        raise RuntimeError("no exact development documents were found")
    if not any(row["split"] == "test" for row in rows):
        raise RuntimeError("no exact test documents were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(content, encoding="utf-8")
    counts = Counter((row["split"] for row in rows))
    manifest = {
        "version": "stage2_ud_2_18_exact_v1",
        "status": "exact_reference_dev_test",
        "selection": str(args.selection),
        "ud_root": str(args.ud_root),
        "output": str(args.output),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "documents": dict(sorted(counts.items())),
        "languages": len({row["language_script"] for row in rows}),
        "sentences_per_document": args.sentences_per_document,
        "sources": sources,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "documents", "languages", "sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
