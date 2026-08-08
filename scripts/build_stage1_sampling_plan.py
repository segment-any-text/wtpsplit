#!/usr/bin/env python
"""Freeze FineWeb2 sampling quotas and review evaluation-coverage gaps."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

GAP_DECISIONS = {
    "kor_Kore": ("equivalent_script_alias", "kor_Hang", "Hang and Kore are equivalent labels here."),
    "cmn_Hans": ("content_filter_required", "cmn_Hani", "Hani does not distinguish Simplified Chinese."),
    "cmn_Hant": ("content_filter_required", "cmn_Hani", "Hani does not distinguish Traditional Chinese."),
    "wuu_Hans": ("content_filter_required", "wuu_Hani", "Hani does not distinguish Simplified Chinese."),
    "yue_Hant": ("content_filter_required", "yue_Hani", "Hani does not distinguish Traditional Chinese."),
    "por_Latn_braz1246": (
        "variety_parent_only",
        "por_Latn",
        "General Portuguese cannot establish Brazilian-variety-specific coverage.",
    ),
    "arz_Latn": ("wrong_script_no_substitution", "arz_Arab", "Arabic-script data cannot substitute for Latin-script text."),
    "brh_Latn": ("wrong_script_no_substitution", "brh_Arab", "Arabic-script data cannot substitute for Latin-script text."),
    "ckb_Latn": ("wrong_script_no_substitution", "ckb_Arab", "Arabic-script data cannot substitute for Latin-script text."),
    "got_Latn": ("wrong_script_no_substitution", "got_Goth", "Gothic-script data cannot substitute for Latin transliteration."),
    "knc_Arab": ("wrong_script_no_substitution", "knc_Latn", "Latin-script data cannot substitute for Arabic-script text."),
    "min_Arab": ("wrong_script_no_substitution", "min_Latn", "Latin-script data cannot substitute for Arabic-script text."),
    "ota_Latn": ("wrong_script_no_substitution", "ota_Arab", "Arabic-script data cannot substitute for Latin transliteration."),
    "pan_Arab": ("wrong_script_no_substitution", "pan_Guru", "Gurmukhi/Latin data cannot substitute for Shahmukhi."),
    "sdh_Latn": ("wrong_script_no_substitution", "sdh_Arab", "Arabic-script data cannot substitute for Latin-script text."),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_documents(row: dict[str, Any]) -> tuple[int, list[str]]:
    documents = row["train_documents"]
    priority = row["stage1_sampling_priority"]
    cap = {
        "P0_measured_failure": 50_000,
        "P0_exact_partial_anchor": 50_000,
        "P1_evaluation_coverage": 20_000,
        "P2_backbone_coverage_extension": 5_000,
    }[priority]
    reasons = [f"priority_cap_{cap}"]
    target = min(documents, cap)
    if row["bible_ratio"] is not None and row["bible_ratio"] >= 0.8:
        reasons.append("bible_dominated_requires_source_complement")
        if documents >= 10_000:
            target = min(target, 5_000)
            reasons.append("bible_dominated_large_source_cap_5000")
    if documents < 1_000:
        reasons.append("retain_all_micro_source")
    return target, reasons


def validation_documents(row: dict[str, Any], target: int) -> tuple[int, str]:
    official = row["official_test_documents"]
    if official:
        return official, "fineweb2_official_test"
    if target < 20:
        return max(1, target // 5), "deterministic_train_holdout"
    return min(1_000, max(20, round(target * 0.01))), "deterministic_train_holdout"


def build_sampling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if not row["hf_streaming_supported"]:
            continue
        target, reasons = target_documents(row)
        valid_docs, valid_source = validation_documents(row, target)
        output.append(
            {
                **row,
                "target_train_documents": max(0, target - (valid_docs if valid_source == "deterministic_train_holdout" else 0)),
                "target_validation_documents": valid_docs,
                "validation_source_frozen": valid_source,
                "sampling_reasons": reasons,
                "sampling_status": (
                    "eligible_with_source_composition_warning"
                    if "bible_dominated" in row["quality_flags"]
                    else "eligible"
                ),
            }
        )
    total = sum(row["target_train_documents"] for row in output)
    for row in output:
        row["target_mixture_weight"] = (
            row["target_train_documents"] / total if total else 0.0
        )
    return output


def build_gap_rows(
    fineweb: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluation_index = {
        row["language_script"]: row
        for row in evaluation["rows"]
    }
    output = []
    for gap in fineweb["evaluation_union_missing_details"]:
        identity = gap["language_script"]
        decision = GAP_DECISIONS.get(identity)
        if decision:
            status, candidate, rationale = decision
        else:
            status, candidate, rationale = (
                "language_absent",
                None,
                "No FineWeb2 filtered training configuration exists for this language.",
            )
        eval_row = evaluation_index[identity]
        output.append(
            {
                "language_script": identity,
                "evaluation_tier": eval_row["evaluation_tier"],
                "segmentation_risk_category": eval_row["segmentation_risk_category"],
                "gap_category": gap["gap_category"],
                "review_decision": status,
                "candidate_fineweb2_config": candidate,
                "candidate_is_direct_substitute": status == "equivalent_script_alias",
                "rationale": rationale,
                "stage1_route": (
                    f"use_{candidate}"
                    if status == "equivalent_script_alias"
                    else (
                        "candidate_requires_document_script_filter"
                        if status == "content_filter_required"
                        else "no_fineweb2_stage1_supervision"
                    )
                ),
                "stage2_route": (
                    "native_or_projected_supervision_only_after_source_audit"
                ),
            }
        )
    return output


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_payload(payload: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0]))
        writer.writeheader()
        writer.writerows(
            {key: csv_value(value) for key, value in row.items()}
            for row in payload["rows"]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fineweb-coverage",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_coverage_v1.json"),
    )
    parser.add_argument(
        "--evaluation-coverage",
        type=Path,
        default=Path("data/manifests/mmsat_dataset_coverage_v1.json"),
    )
    parser.add_argument(
        "--sampling-json",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_sampling_plan_v1.json"),
    )
    parser.add_argument(
        "--sampling-csv",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_sampling_plan_v1.csv"),
    )
    parser.add_argument(
        "--gaps-json",
        type=Path,
        default=Path("data/manifests/stage1_coverage_gap_review_v1.json"),
    )
    parser.add_argument(
        "--gaps-csv",
        type=Path,
        default=Path("data/manifests/stage1_coverage_gap_review_v1.csv"),
    )
    args = parser.parse_args()
    fineweb = json.loads(args.fineweb_coverage.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_coverage.read_text(encoding="utf-8"))
    sampling_rows = build_sampling_rows(fineweb["rows"])
    sampling_payload = {
        "version": "fineweb2_stage1_sampling_plan_v1",
        "retrieved": date.today().isoformat(),
        "source_manifest": str(args.fineweb_coverage),
        "source_manifest_sha256": file_sha256(args.fineweb_coverage),
        "evaluation_coverage": str(args.evaluation_coverage),
        "evaluation_coverage_sha256": file_sha256(args.evaluation_coverage),
        "dataset_revisions": fineweb.get("dataset_revisions", {}),
        "policy": {
            "unit": "language-script configuration",
            "quotas": "Priority-specific hard caps; retain all micro sources.",
            "source_composition": (
                "Bible-dominated sources remain visible; large ones are capped at 5k "
                "and all require complementary-source analysis."
            ),
            "validation": "Official FineWeb2 test when present; otherwise deterministic train holdout.",
            "weight": "Normalized target document quota, not raw corpus size.",
        },
        "counts": {
            "eligible_language_script_pairs": len(sampling_rows),
            "target_train_documents": sum(
                row["target_train_documents"] for row in sampling_rows
            ),
            "official_test_pairs": sum(
                row["validation_source_frozen"] == "fineweb2_official_test"
                for row in sampling_rows
            ),
            "deterministic_holdout_pairs": sum(
                row["validation_source_frozen"] == "deterministic_train_holdout"
                for row in sampling_rows
            ),
            "source_composition_warning_pairs": sum(
                row["sampling_status"] == "eligible_with_source_composition_warning"
                for row in sampling_rows
            ),
        },
        "rows": sampling_rows,
    }
    write_payload(
        sampling_payload,
        args.sampling_json,
        args.sampling_csv,
    )

    gap_rows = build_gap_rows(fineweb, evaluation)
    gap_payload = {
        "version": "stage1_coverage_gap_review_v1",
        "retrieved": date.today().isoformat(),
        "source_manifest": str(args.fineweb_coverage),
        "source_manifest_sha256": file_sha256(args.fineweb_coverage),
        "evaluation_coverage": str(args.evaluation_coverage),
        "evaluation_coverage_sha256": file_sha256(args.evaluation_coverage),
        "policy": {
            "identity": "Never silently merge language varieties or script labels.",
            "direct_alias": "Only Korean Hang/Kore is accepted automatically.",
            "hani": "Hani candidates require document-level script filtering.",
            "wrong_script": "Same language in another script is not a substitute.",
        },
        "counts": {
            "gaps": len(gap_rows),
            "direct_aliases": sum(
                row["candidate_is_direct_substitute"] for row in gap_rows
            ),
            "content_filter_candidates": sum(
                row["review_decision"] == "content_filter_required" for row in gap_rows
            ),
            "wrong_script_or_variety": sum(
                row["review_decision"]
                in {"wrong_script_no_substitution", "variety_parent_only"}
                for row in gap_rows
            ),
            "languages_absent": sum(
                row["review_decision"] == "language_absent" for row in gap_rows
            ),
        },
        "rows": gap_rows,
    }
    write_payload(gap_payload, args.gaps_json, args.gaps_csv)
    print(
        json.dumps(
            {
                "sampling": sampling_payload["counts"],
                "gaps": gap_payload["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
