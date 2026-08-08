#!/usr/bin/env python
"""Validate Stage-1 corpus receipts without loading the corpus into memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wtpsplit.data_acquisition.stage1_validation import validate_stage1_metadata
from wtpsplit.data_acquisition.stage1_web import load_language_map, load_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, nargs="+")
    expected = parser.add_mutually_exclusive_group()
    expected.add_argument("--mapping", type=Path)
    expected.add_argument("--scaleout-plan", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-contamination-filter", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--minimum-cap-fill", type=float, default=0.95)
    parser.add_argument("--max-reported-issues", type=int, default=50)
    parser.add_argument("--write-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_languages = None
    if args.mapping:
        expected_languages = load_language_map(args.mapping)
    elif args.scaleout_plan:
        expected_languages = load_plan(args.scaleout_plan)
    reports = [
        validate_stage1_metadata(
            path,
            expected_languages=expected_languages,
            require_complete=not args.allow_incomplete,
            require_contamination_filter=args.require_contamination_filter,
            verify_hashes=args.verify_hashes,
            minimum_cap_fill=args.minimum_cap_fill,
            max_reported_issues=args.max_reported_issues,
        )
        for path in args.metadata
    ]
    report = {
        "schema": "stage1-corpus-validation-batch-v1",
        "valid": all(item["valid"] for item in reports),
        "corpora": reports,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
