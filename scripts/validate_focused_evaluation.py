#!/usr/bin/env python
"""Validate provenance and annotation state of a focused evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "id",
    "language_script",
    "script",
    "slice",
    "source",
    "source_version",
    "license",
    "reference_type",
    "annotation_status",
    "text",
    "gold_offsets",
}


def validate(rows: list[dict], require_gold: bool = False) -> dict:
    ids = set()
    by_status: dict[str, int] = {}
    for line_number, row in enumerate(rows, 1):
        missing = REQUIRED - set(row)
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        if row["id"] in ids:
            raise ValueError(f"line {line_number}: duplicate id {row['id']}")
        ids.add(row["id"])
        offsets = row["gold_offsets"]
        if offsets != sorted(set(offsets)) or any(not 0 < value < len(row["text"]) for value in offsets):
            raise ValueError(f"line {line_number}: invalid gold_offsets")
        status = row["annotation_status"]
        by_status[status] = by_status.get(status, 0) + 1
        if require_gold and status != "adjudicated_gold":
            raise ValueError(f"line {line_number}: {status!r} is not adjudicated_gold")
        if require_gold and row["reference_type"] != "exact":
            raise ValueError(f"line {line_number}: final gold must use exact references")
        if not row["license"] or not row["source_version"]:
            raise ValueError(f"line {line_number}: source license/version must be recorded")
    return {"documents": len(rows), "by_status": by_status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--require-gold", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(validate(rows, args.require_gold), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
