#!/usr/bin/env python
"""Build a per-language FineWeb2 Stage-1 inventory and sampling plan."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request

FINEWEB2_DISTRIBUTION_URL = (
    "https://raw.githubusercontent.com/huggingface/fineweb-2/"
    "main/fineweb2-language-distribution.csv"
)
FINEWEB2_DATASET = "HuggingFaceFW/fineweb-2"
FINEWEB2_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"


def integer(value: str) -> int | None:
    return None if value in {"", "-"} else int(value)


def ratio(value: str) -> float | None:
    return None if value in {"", "-"} else float(value)


def resource_bucket(documents: int) -> str:
    if documents < 1_000:
        return "micro_lt_1k_documents"
    if documents < 10_000:
        return "low_1k_10k_documents"
    if documents < 100_000:
        return "medium_10k_100k_documents"
    if documents < 1_000_000:
        return "high_100k_1m_documents"
    return "very_high_ge_1m_documents"


def stage1_priority(
    evaluation: dict[str, Any] | None,
    baseline_category: str | None,
) -> str:
    if baseline_category == "severe_bouquet_failure":
        return "P0_measured_failure"
    if evaluation and evaluation["evaluation_tier"] == "exact_and_partial":
        return "P0_exact_partial_anchor"
    if evaluation:
        return "P1_evaluation_coverage"
    return "P2_backbone_coverage_extension"


def recommended_cap(priority: str, documents: int) -> int:
    caps = {
        "P0_measured_failure": 50_000,
        "P0_exact_partial_anchor": 50_000,
        "P1_evaluation_coverage": 20_000,
        "P2_backbone_coverage_extension": 5_000,
    }
    return min(documents, caps[priority])


def quality_flags(row: dict[str, str], documents: int) -> list[str]:
    flags = []
    bible = ratio(row["bible_ratio"])
    wiki = ratio(row["wiki_ratio"])
    if documents < 100:
        flags.append("extremely_small")
    if bible is not None and bible >= 0.8:
        flags.append("bible_dominated")
    if wiki is not None and wiki >= 0.8:
        flags.append("wikipedia_dominated")
    if row["code"] == "und":
        flags.append("undetermined_language")
    return flags or ["no_distribution_level_flag"]


def build_rows(
    distribution: list[dict[str, str]],
    evaluation_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluation = {
        row["language_script"]: row
        for row in evaluation_coverage["rows"]
        if row.get("reportable_evaluation", True)
    }
    test_rows = {
        row["subset"]: row
        for row in distribution
        if row["split"] == "test" and not row["subset"].endswith("_removed")
    }
    output = []
    for source in distribution:
        if source["split"] != "train" or source["subset"].endswith("_removed"):
            continue
        documents = integer(source["documents"])
        if documents is None:
            continue
        identity = source["subset"]
        eval_row = evaluation.get(identity)
        test_row = test_rows.get(identity)
        priority = stage1_priority(
            eval_row,
            eval_row["bouquet_baseline_category"] if eval_row else None,
        )
        flags = quality_flags(source, documents)
        output.append(
            {
                "language_script": identity,
                "iso639_3": source["code"],
                "script": source["script"],
                "language_name": source["name"],
                "language_family": source["family"],
                "train_documents": documents,
                "train_words": integer(source["words"]),
                "train_utf8_bytes": integer(source["utf8_bytes"]),
                "train_parquet_bytes": integer(source["parquet_bytes"]),
                "official_test_documents": (
                    integer(test_row["documents"]) if test_row else None
                ),
                "official_test_parquet_bytes": (
                    integer(test_row["parquet_bytes"]) if test_row else None
                ),
                "stage1_validation_source": (
                    "fineweb2_official_test"
                    if test_row
                    else "deterministic_train_holdout_required"
                ),
                "bible_ratio": ratio(source["bible_ratio"]),
                "wiki_ratio": ratio(source["wiki_ratio"]),
                "resource_bucket": resource_bucket(documents),
                "quality_flags": flags,
                "in_evaluation_union": eval_row is not None,
                "evaluation_tier": (
                    eval_row["evaluation_tier"] if eval_row else "none"
                ),
                "bouquet_baseline_category": (
                    eval_row["bouquet_baseline_category"]
                    if eval_row
                    else "not_locally_measured"
                ),
                "stage1_sampling_priority": priority,
                "recommended_pilot_cap_documents": recommended_cap(
                    priority,
                    documents,
                ),
                "hf_dataset": FINEWEB2_DATASET,
                "hf_config": identity,
                "hf_split": "train",
                "hf_streaming_supported": source["code"] != "und",
                "hf_parquet_path": (
                    f"hf://datasets/{FINEWEB2_DATASET}/data/{identity}/train"
                ),
                "license": "ODC-By 1.0; also subject to Common Crawl Terms of Use",
                "training_readiness": (
                    "candidate_after_quality_filter_and_exact_eval_dedup"
                    if source["code"] != "und"
                    else "exclude_until_language_identification"
                ),
                "contamination_policy": (
                    "Remove exact and near-duplicate matches to every frozen "
                    "evaluation document before training."
                ),
            }
        )
    output.sort(
        key=lambda row: (
            row["stage1_sampling_priority"],
            row["language_script"],
        )
    )
    return output


def csv_value(value: Any) -> Any:
    return ";".join(str(item) for item in value) if isinstance(value, list) else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-csv", type=Path)
    parser.add_argument(
        "--evaluation-coverage",
        type=Path,
        default=Path("data/manifests/mmsat_dataset_coverage_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_coverage_v1.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_coverage_v1.csv"),
    )
    args = parser.parse_args()
    if args.distribution_csv:
        distribution_text = args.distribution_csv.read_text(encoding="utf-8")
        distribution_source = str(args.distribution_csv)
    else:
        with urllib.request.urlopen(FINEWEB2_DISTRIBUTION_URL, timeout=60) as response:
            distribution_text = response.read().decode("utf-8")
        distribution_source = FINEWEB2_DISTRIBUTION_URL
    distribution = list(csv.DictReader(distribution_text.splitlines()))
    evaluation_text = args.evaluation_coverage.read_text(encoding="utf-8")
    evaluation = json.loads(evaluation_text)
    rows = build_rows(distribution, evaluation)
    evaluation_set = {
        row["language_script"]
        for row in evaluation["rows"]
        if row.get("reportable_evaluation", True)
    }
    fineweb_set = {row["language_script"] for row in rows}
    fineweb_by_language: dict[str, list[str]] = {}
    for identity in fineweb_set:
        fineweb_by_language.setdefault(identity.split("_", 1)[0], []).append(identity)
    missing_details = [
        {
            "language_script": identity,
            "gap_category": (
                "same_language_other_script_available"
                if identity.split("_", 1)[0] in fineweb_by_language
                else "language_not_available"
            ),
            "same_language_configs": sorted(
                fineweb_by_language.get(identity.split("_", 1)[0], [])
            ),
        }
        for identity in sorted(evaluation_set - fineweb_set)
    ]
    payload = {
        "version": "fineweb2_stage1_coverage_v1",
        "retrieved": date.today().isoformat(),
        "dataset": FINEWEB2_DATASET,
        "dataset_revisions": {FINEWEB2_DATASET: FINEWEB2_REVISION},
        "distribution_source": distribution_source,
        "distribution_sha256": hashlib.sha256(distribution_text.encode("utf-8")).hexdigest(),
        "evaluation_coverage": str(args.evaluation_coverage),
        "evaluation_coverage_sha256": hashlib.sha256(evaluation_text.encode("utf-8")).hexdigest(),
        "license": "ODC-By 1.0; also subject to Common Crawl Terms of Use",
        "policy": {
            "filtered_only": "Removed subsets are excluded.",
            "sampling": "Caps are planning defaults, not final mixture weights.",
            "quality": "Bible/wiki ratios are flags, not automatic rejection rules.",
            "contamination": "Frozen evaluation documents must be deduplicated before training.",
            "unknown": "und_* subsets are excluded pending language identification.",
        },
        "counts": {
            "filtered_train_language_script_pairs": len(rows),
            "streamable_identified_pairs": sum(
                row["hf_streaming_supported"] for row in rows
            ),
            "evaluation_union_pairs_available": len(evaluation_set & fineweb_set),
            "evaluation_union_pairs_missing": len(evaluation_set - fineweb_set),
            "missing_but_same_language_other_script": sum(
                row["gap_category"] == "same_language_other_script_available"
                for row in missing_details
            ),
            "bible_dominated_pairs": sum(
                "bible_dominated" in row["quality_flags"] for row in rows
            ),
            "micro_pairs": sum(
                row["resource_bucket"] == "micro_lt_1k_documents" for row in rows
            ),
            "planned_pilot_documents_at_caps": sum(
                row["recommended_pilot_cap_documents"] for row in rows
            ),
        },
        "evaluation_union_missing": sorted(evaluation_set - fineweb_set),
        "evaluation_union_missing_details": missing_details,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: csv_value(value) for key, value in row.items()}
            for row in rows
        )
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
