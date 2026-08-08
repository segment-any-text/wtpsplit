#!/usr/bin/env python
"""Simulate the complete Stage-1 build/validate/train/evaluate pipeline offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_stage1 import bounded_caps
from scripts.prepare_stage1_runs import (
    MATCHED_CONFIGS,
    build_run_plan,
    unsupported_config_keys,
)
from wtpsplit.data_acquisition.stage1_web import (
    load_mc4_caps,
    load_plan,
    resolve_web_sources,
)


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data/manifests/sat_lang_to_fineweb2_v1.json"
CAPS = ROOT / "data/manifests/mc4_test_per_lang_char_mass.json"
PLAN = ROOT / "data/manifests/fineweb2_stage1_sampling_plan_v1.json"
REMAPS = ROOT / "data/manifests/stage1_fineweb_script_remaps_v1.json"
REVISIONS = {
    "HuggingFaceFW/fineweb-2": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "allenai/c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
}


def build_report(*, nproc: int, smoke_chars: int) -> dict:
    caps = load_mc4_caps(CAPS)
    matched: dict[str, dict] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for corpus in ("mc4", "fineweb2"):
        sources = resolve_web_sources(
            corpus=corpus,
            language_map=MAPPING,
            languages=caps,
            plan=PLAN if corpus == "fineweb2" else None,
            script_remaps=REMAPS if corpus == "fineweb2" else None,
            revisions=REVISIONS,
        )
        smoke_caps = bounded_caps(caps, max_chars_per_language=smoke_chars)
        matched[corpus] = {
            "languages": len(sources),
            "full_character_budget": sum(caps.values()),
            "smoke_character_budget": sum(smoke_caps.values()),
            "source_datasets": sorted({source.dataset for source in sources.values()}),
            "units": {
                "primary": "paragraph",
                "alternative": "document",
            },
            "build_commands": {
                unit: (
                    f"python scripts/build_stage1.py --corpus {corpus} --unit {unit} "
                    f"--max-chars-per-language {smoke_chars}"
                )
                for unit in ("paragraph", "document")
            },
        }

    plan_payload = json.loads(PLAN.read_text(encoding="utf-8"))
    scaleout = load_plan(plan_payload)
    required = {
        "language_script",
        "hf_dataset",
        "hf_config",
        "hf_split",
        "target_train_documents",
        "target_mixture_weight",
    }
    for language_script, row in scaleout.items():
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"scale-out {language_script}: missing {missing}")
    if len(scaleout) < 1000:
        warnings.append(f"scale-out plan has only {len(scaleout)} language-script pairs")

    run_plan = build_run_plan(world_size=nproc, root=ROOT)
    for arm in run_plan["arms"]:
        if arm["distributed_batch"]["global_batch_size"] != 512:
            errors.append(f"{arm['arm']}: global batch is not 512")

    matched_config_errors = {}
    for relative in MATCHED_CONFIGS.values():
        config = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        unsupported = unsupported_config_keys(config)
        if unsupported:
            matched_config_errors[relative] = unsupported
            errors.append(f"{relative}: unsupported training keys: {unsupported}")

    pilot_config_path = ROOT / "configs/mmsat_3l.json"
    pilot_config = json.loads(pilot_config_path.read_text(encoding="utf-8"))
    pilot_unsupported = unsupported_config_keys(pilot_config)
    if pilot_unsupported:
        errors.append(
            f"configs/mmsat_3l.json: unsupported training keys: {pilot_unsupported}"
        )

    contamination = ROOT / "data/external/mmsat_contamination_index_v1.json.gz"
    if not contamination.is_file():
        warnings.append("contamination index is not present locally; FineWeb materialization remains blocked")
    return {
        "schema": "stage1-full-pipeline-dry-run-v1",
        "dry_run": True,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "matched_85_language_track": matched,
        "matched_configs_valid": not matched_config_errors,
        "primary_training_matrix": run_plan,
        "scaleout_track": {
            "status": "planned_not_materialized",
            "pilot_config": "configs/mmsat_3l.json",
            "pilot_config_valid": not pilot_unsupported,
            "language_script_pairs": len(scaleout),
            "target_train_documents": sum(int(row["target_train_documents"]) for row in scaleout.values()),
            "source_composition_warning_pairs": int(
                plan_payload.get("counts", {}).get("source_composition_warning_pairs", 0)
            ),
            "sampling_command": "python scripts/sample_fineweb2.py",
            "validation_command": (
                "python scripts/validate_stage1.py CORPUS/metadata.json "
                "--scaleout-plan data/manifests/fineweb2_stage1_sampling_plan_v1.json"
            ),
        },
        "pipeline": [
            "resolve pinned sources and frozen quotas",
            "build paragraph-primary and document-alternative shards",
            "validate receipts and optionally all artifact hashes",
            "train crossed corpus × backbone paragraph arms",
            "run the frozen Stage-1 intrinsic evaluation contract",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--smoke-chars-per-language", type=int, default=10_000)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    report = build_report(nproc=args.nproc_per_node, smoke_chars=args.smoke_chars_per_language)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
