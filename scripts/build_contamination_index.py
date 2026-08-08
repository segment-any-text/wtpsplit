#!/usr/bin/env python
"""Build a text-free evaluation-contamination fingerprint index."""

from __future__ import annotations

import argparse
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path

from wtpsplit.data_acquisition.contamination import ReferenceIndex
from wtpsplit.evaluation.diagnostics.boundary_ceiling import (
    load_bouquet,
    separator_for,
)


def add_ud(index: ReferenceIndex, root: Path, selection: dict) -> int:
    added = 0
    seen_repositories = set()
    for row in selection["rows"]:
        if row["selection_status"] != "frozen":
            continue
        repository = row["repository"]
        if repository in seen_repositories:
            continue
        seen_repositories.add(repository)
        filename = row["evaluation_url_verified"].rsplit("/", 1)[-1]
        path = root / repository / filename
        if not path.exists():
            raise FileNotFoundError(f"Frozen UD file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as binary:
            while chunk := binary.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != row["evaluation_sha256"]:
            raise ValueError(f"Frozen UD hash mismatch: {path}")
        with path.open(encoding="utf-8") as handle:
            sentence_number = 0
            for line in handle:
                if not line.startswith("# text = "):
                    continue
                sentence_number += 1
                index.add_text(
                    line[len("# text = ") :].strip(),
                    f"ud:{repository}:{sentence_number}",
                    "ud_2.18_test",
                    row["language_script"],
                )
                added += 1
    return added


def add_bouquet(
    index: ReferenceIndex,
    splits: list[str],
    revision: str | None = None,
) -> int:
    added = 0
    for split in splits:
        for language, _, documents in load_bouquet(
            split,
            None,
            revision=revision,
        ):
            for document_number, segments in enumerate(documents):
                separator = separator_for(language, "")
                index.add_text(
                    separator.join(segments),
                    f"bouquet:{split}:{language}:{document_number}",
                    f"bouquet_{split}_partial",
                    language,
                )
                added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ud-root",
        type=Path,
        default=Path("data/external/ud-treebanks-v2.18-frozen"),
    )
    parser.add_argument(
        "--ud-selection",
        type=Path,
        default=Path("data/manifests/ud_2_18_frozen_selection_v1.json"),
    )
    parser.add_argument("--skip-bouquet", action="store_true")
    parser.add_argument("--bouquet-revision")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/mmsat_contamination_index_v1.json.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/mmsat_contamination_index_v1.json"),
    )
    args = parser.parse_args()
    if not args.skip_bouquet and not args.bouquet_revision:
        parser.error("--bouquet-revision is required unless --skip-bouquet is used")
    selection = json.loads(args.ud_selection.read_text(encoding="utf-8"))
    index = ReferenceIndex()
    ud_sentences = add_ud(index, args.ud_root, selection)
    bouquet_documents = (
        0
        if args.skip_bouquet
        else add_bouquet(
            index,
            ["dev", "test"],
            revision=args.bouquet_revision,
        )
    )
    payload = index.to_payload()
    payload.update(
        {
            "created": date.today().isoformat(),
            "inputs": {
                "ud_selection": str(args.ud_selection),
                "ud_root": str(args.ud_root),
                "bouquet_splits": [] if args.skip_bouquet else ["dev", "test"],
                "bouquet_revision": args.bouquet_revision,
            },
            "input_units": {
                "ud_sentences": ud_sentences,
                "bouquet_documents": bouquet_documents,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, mode="wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    digest = hashlib.sha256()
    with args.output.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    manifest = {
        "version": "mmsat_contamination_index_manifest_v2",
        "created": date.today().isoformat(),
        "artifact": str(args.output),
        "artifact_sha256": digest.hexdigest(),
        "artifact_bytes": args.output.stat().st_size,
        "inputs": payload["inputs"],
        "input_sha256": {
            "ud_selection": hashlib.sha256(args.ud_selection.read_bytes()).hexdigest(),
        },
        "counts": payload["counts"],
        "input_units": payload["input_units"],
        "algorithm": payload["algorithm"],
        "reproducibility": {
            "status": "rebuildable_from_pinned_inputs",
            "bouquet_revision": args.bouquet_revision,
            "bouquet_included": not args.skip_bouquet,
        },
        "rebuild_command": "python scripts/build_contamination_index.py",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **payload["counts"],
                **payload["input_units"],
                "output_bytes": args.output.stat().st_size,
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
