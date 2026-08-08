"""Audit whether one BOUQuET row can contain multiple English sentences.

BOUQuET repeats the same English target unit across its source languages. This
script deduplicates those units by ``uniq_id`` and compares two independent
English sentence detectors: PySBD rules and a cached SaT checkpoint. Agreement
is a conservative automatic lower bound; all disagreements remain inspectable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--sat-model", default="sat-12l-sm")
    parser.add_argument("--sat-threshold", type=float, default=0.075)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bouquet-revision", default=BOUQUET_REVISION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/bouquet_segment_audit_dev.json"),
    )
    return parser.parse_args()


def deduplicate_english_units(dataset) -> tuple[list[dict[str, Any]], int]:
    """Return one English target per uniq_id and count inconsistent repeats."""
    by_id: dict[str, dict[str, Any]] = {}
    inconsistent_ids = set()
    for row in dataset:
        unit_id = row["uniq_id"]
        text = row["tgt_text"].strip()
        if not text:
            continue
        if unit_id in by_id and by_id[unit_id]["text"] != text:
            inconsistent_ids.add(unit_id)
            continue
        by_id.setdefault(
            unit_id,
            {
                "uniq_id": unit_id,
                "paragraph_id": row["par_id"],
                "domain": row["domain"],
                "tags": row["tags"],
                "text": text,
            },
        )
    return list(by_id.values()), len(inconsistent_ids)


def normalized_segments(segments: list[str]) -> list[str]:
    return [segment.strip() for segment in segments if segment.strip()]


def summarize(
    units: list[dict[str, Any]],
    pysbd_outputs: list[list[str]],
    sat_outputs: list[list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not (len(units) == len(pysbd_outputs) == len(sat_outputs)):
        raise ValueError("Units and detector outputs must have the same length.")

    classifications = Counter()
    pysbd_counts = Counter()
    sat_counts = Counter()
    consensus_missing_boundaries = 0
    flagged = []
    for unit, pysbd_segments, sat_segments in zip(
        units,
        pysbd_outputs,
        sat_outputs,
    ):
        pysbd_segments = normalized_segments(pysbd_segments)
        sat_segments = normalized_segments(sat_segments)
        pysbd_count = len(pysbd_segments)
        sat_count = len(sat_segments)
        pysbd_multi = pysbd_count > 1
        sat_multi = sat_count > 1
        if pysbd_multi and sat_multi:
            classification = "consensus_multi"
            consensus_missing_boundaries += min(pysbd_count, sat_count) - 1
        elif pysbd_multi or sat_multi:
            classification = "disagreement"
        else:
            classification = "consensus_single"
        classifications[classification] += 1
        pysbd_counts[pysbd_count] += 1
        sat_counts[sat_count] += 1
        if classification != "consensus_single":
            flagged.append(
                {
                    **unit,
                    "classification": classification,
                    "pysbd_segments": pysbd_segments,
                    "sat_segments": sat_segments,
                }
            )

    total = len(units)
    paragraphs = len({unit["paragraph_id"] for unit in units})
    known_row_boundaries = total - paragraphs
    summary = {
        "unique_units": total,
        "paragraphs": paragraphs,
        "known_row_boundaries": known_row_boundaries,
        "consensus_single": classifications["consensus_single"],
        "consensus_multi": classifications["consensus_multi"],
        "disagreement": classifications["disagreement"],
        "consensus_missing_boundaries_lower_bound": consensus_missing_boundaries,
        "missing_boundary_share_lower_bound": (
            consensus_missing_boundaries
            / (known_row_boundaries + consensus_missing_boundaries)
            if known_row_boundaries + consensus_missing_boundaries
            else 0.0
        ),
        "consensus_multi_rate": (
            classifications["consensus_multi"] / total if total else 0.0
        ),
        "union_multi_rate": (
            (
                classifications["consensus_multi"]
                + classifications["disagreement"]
            )
            / total
            if total
            else 0.0
        ),
        "pysbd_sentence_count_distribution": dict(sorted(pysbd_counts.items())),
        "sat_sentence_count_distribution": dict(sorted(sat_counts.items())),
    }
    return summary, flagged


def main() -> int:
    args = parse_args()

    import datasets
    import pysbd

    from wtpsplit import SaT
    from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_DATASET

    dataset = datasets.load_dataset(
        BOUQUET_DATASET,
        "sentence_level",
        split=args.split,
        revision=args.bouquet_revision,
    )
    units, inconsistent_ids = deduplicate_english_units(dataset)
    texts = [unit["text"] for unit in units]

    segmenter = pysbd.Segmenter(language="en", clean=False)
    pysbd_outputs = [segmenter.segment(text) for text in texts]

    sat = SaT(args.sat_model, device="cuda")
    sat_outputs = sat.split(
        texts,
        threshold=args.sat_threshold,
        batch_size=args.batch_size,
    )
    summary, flagged = summarize(units, pysbd_outputs, sat_outputs)
    result = {
        "protocol": {
            "dataset": "facebook/bouquet sentence_level",
            "dataset_revision": args.bouquet_revision,
            "split": args.split,
            "unit": "unique English tgt_text deduplicated by uniq_id",
            "rule_detector": "pysbd en",
            "model_detector": args.sat_model,
            "model_threshold": args.sat_threshold,
            "interpretation": (
                "Detector agreement is a conservative automatic lower bound, "
                "not a replacement for manual adjudication."
            ),
        },
        "inventory": {
            "dataset_rows": len(dataset),
            "source_languages": len(set(dataset["src_lang"])),
            "inconsistent_repeated_english_units": inconsistent_ids,
        },
        "summary": summary,
        "flagged_units": flagged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
