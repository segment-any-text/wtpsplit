#!/usr/bin/env python
"""Filter FineWeb2 sample shards and write train/valid .filtered.jsonl trees.

Reads raw ``*.jsonl`` (not already under train/ or valid/) from the sample
directory, drops contamination-index hits, then deterministically splits each
language into train and valid shards that ``configs/mmsat_3l.json`` expects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wtpsplit.data_acquisition.contamination import load_index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external/fineweb2-stage1-samples"),
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/external/fineweb2-stage1-samples/train"),
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=Path("data/external/fineweb2-stage1-samples/valid"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/external/mmsat_contamination_index_v1.json.gz"),
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--language-column", default="lang")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/manifests/stage1_filtered_splits_v1.json"),
    )
    return parser.parse_args()


def raw_shards(input_dir: Path) -> list[Path]:
    shards = []
    for path in sorted(input_dir.glob("*.jsonl")):
        if ".filtered." in path.name or ".receipt." in path.name:
            continue
        shards.append(path)
    return shards


def assign_split(row_id: str, valid_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "valid" if bucket < valid_fraction else "train"


def main() -> int:
    args = parse_args()
    if not 0.0 < args.valid_fraction < 1.0:
        raise SystemExit("--valid-fraction must be in (0, 1)")
    shards = raw_shards(args.input_dir)
    if not shards:
        raise SystemExit(
            f"No raw *.jsonl shards under {args.input_dir}. "
            "Run sample_fineweb2.py --execute first."
        )
    if not args.index.is_file():
        raise SystemExit(
            f"Missing contamination index: {args.index}. "
            "Run build_contamination_index.py first."
        )
    for directory in (args.train_dir, args.valid_dir):
        directory.mkdir(parents=True, exist_ok=True)
    index = load_index(args.index)
    receipt_rows = []
    totals = {
        "input": 0,
        "kept_train": 0,
        "kept_valid": 0,
        "exact_removed": 0,
        "near_quarantined": 0,
        "invalid": 0,
    }
    for shard in shards:
        language = shard.stem
        train_out = args.train_dir / f"{language}.filtered.jsonl"
        valid_out = args.valid_dir / f"{language}.filtered.jsonl"
        for target in (train_out, valid_out):
            if target.exists() and not args.overwrite:
                raise SystemExit(
                    f"Output exists; refuse to overwrite without --overwrite: {target}"
                )
        train_tmp = train_out.with_suffix(train_out.suffix + ".part")
        valid_tmp = valid_out.with_suffix(valid_out.suffix + ".part")
        counts = {
            "input": 0,
            "kept_train": 0,
            "kept_valid": 0,
            "exact_removed": 0,
            "near_quarantined": 0,
            "invalid": 0,
        }
        try:
            with (
                shard.open(encoding="utf-8") as source,
                train_tmp.open("w", encoding="utf-8") as train_handle,
                valid_tmp.open("w", encoding="utf-8") as valid_handle,
            ):
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    counts["input"] += 1
                    try:
                        row = json.loads(line)
                        text = row[args.text_column]
                        if not isinstance(text, str) or not text:
                            raise TypeError("empty or non-string text")
                    except (json.JSONDecodeError, KeyError, TypeError):
                        counts["invalid"] += 1
                        continue
                    if args.language_column not in row and "language_script" in row:
                        row[args.language_column] = row["language_script"]
                    match = index.match(text, row.get(args.language_column))
                    if match:
                        key = (
                            "exact_removed"
                            if match["match"] == "exact"
                            else "near_quarantined"
                        )
                        counts[key] += 1
                        continue
                    row_id = str(
                        row.get("id")
                        or row.get("doc_id")
                        or f"{language}:{line_number}"
                    )
                    split = assign_split(row_id, args.valid_fraction, args.seed)
                    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    if split == "valid":
                        valid_handle.write(payload + "\n")
                        counts["kept_valid"] += 1
                    else:
                        train_handle.write(payload + "\n")
                        counts["kept_train"] += 1
            train_tmp.replace(train_out)
            valid_tmp.replace(valid_out)
        except BaseException:
            train_tmp.unlink(missing_ok=True)
            valid_tmp.unlink(missing_ok=True)
            raise
        for key, value in counts.items():
            totals[key] += value
        receipt_rows.append(
            {
                "language_script": language,
                "input": str(shard),
                "input_sha256": sha256_file(shard),
                "train": str(train_out),
                "train_sha256": sha256_file(train_out),
                "valid": str(valid_out),
                "valid_sha256": sha256_file(valid_out),
                **counts,
            }
        )
    receipt = {
        "version": "mmsat_stage1_filtered_splits_v1",
        "seed": args.seed,
        "valid_fraction": args.valid_fraction,
        "index": str(args.index),
        "index_sha256": sha256_file(args.index),
        "totals": totals,
        "shards": receipt_rows,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
