#!/usr/bin/env python
"""Validate a Stage 2 config, corpus manifests, and local training artifacts."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch
from transformers import TrainingArguments

from wtpsplit.train.sm_arguments import SentenceTrainingArguments
from wtpsplit.train.sm_data import prepare_sentence_datasets


DEFAULT_MANIFEST_BY_ARTIFACT = {
    "all_data_11_05-all.pth": Path("data/manifests/sat_historical_corpus_v1.json"),
    "mmsat_stage2_abc_pilot_v1.pth": Path(
        "data/manifests/mmsat_stage2_abc_pilot_v1.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    allowed = {
        *(field.name for field in fields(SentenceTrainingArguments)),
        *(field.name for field in fields(TrainingArguments)),
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        errors.append(f"unsupported config keys: {', '.join(unknown)}")

    for key in ("data_path", "model_name_or_path", "tokenizer_name_or_path", "output_dir"):
        if not config.get(key):
            errors.append(f"missing required config value: {key}")
    if config.get("no_sm_corruption") is not True:
        warnings.append("Stage 2 normally uses clean supervision (`no_sm_corruption=true`)")
    fraction = float(config.get("replay_fraction", 0.0))
    replay_path = config.get("replay_data_path")
    if not 0 <= fraction < 1:
        errors.append("replay_fraction must be in [0, 1)")
    if bool(replay_path) != (fraction > 0):
        errors.append("replay_data_path and a positive replay_fraction must be set together")
    if config.get("eval_strategy", "no") != "no" and config.get("report_to") in {None, "none"}:
        warnings.append("step evaluation is disabled by train_SM when report_to is none")
    if config.get("run_final_evaluation"):
        warnings.append("Stage 2 training corpora contain no evaluation text; score an external packet instead")
    return errors, warnings


def validate_manifest(manifest_path: Path, data_path: Path) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"], warnings, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    if data_path.is_file() and expected:
        actual = sha256_file(data_path)
        if actual != expected:
            errors.append(f"SHA-256 mismatch for {data_path}: {actual} != {expected}")
    elif not expected:
        warnings.append(f"manifest has no artifact SHA-256: {manifest_path}")
    return errors, warnings, manifest


def default_manifest_for(data_path: Path) -> Path | None:
    """Resolve only known artifact names; arbitrary corpora require an explicit receipt."""

    return DEFAULT_MANIFEST_BY_ARTIFACT.get(data_path.name)


def validate_corpus(path: Path, *, requested_dataset: str | None = None) -> dict:
    corpus = torch.load(path, map_location="cpu", weights_only=True)
    train, evaluation = prepare_sentence_datasets(
        corpus,
        selected_languages=None,
        requested_training_dataset=requested_dataset,
        no_sm_corruption=True,
        max_train_sentences_per_dataset=1,
        max_eval_instances_per_dataset=1,
    )
    return {
        "languages": len(train),
        "datasets": {
            language: sorted(values)
            for language, values in sorted(train.items())
        },
        "embedded_evaluation_languages": sum(
            any(instances for instances in values.values())
            for values in evaluation.values()
        ),
    }


def check_local_path(
    path: Path,
    *,
    label: str,
    require_data: bool,
    errors: list[str],
    warnings: list[str],
) -> bool:
    if path.is_file():
        return True
    message = f"missing {label}: {path}"
    (errors if require_data else warnings).append(message)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum/stage2.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Artifact receipt; defaults by the configured data filename when known.",
    )
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=Path("data/manifests/mmsat_stage2_historical_control_v1.json"),
    )
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_errors, config_warnings = validate_config(config)
    errors.extend(config_errors)
    warnings.extend(config_warnings)

    data_path = Path(config.get("data_path", ""))
    manifest_path = args.manifest or default_manifest_for(data_path)
    data_ready = check_local_path(
        data_path,
        label="Stage 2 corpus",
        require_data=args.require_data,
        errors=errors,
        warnings=warnings,
    )
    manifest: dict = {}
    if manifest_path is None:
        warnings.append(
            f"no default manifest for {data_path}; pass --manifest to verify artifact identity"
        )
    else:
        manifest_errors, manifest_warnings, manifest = validate_manifest(
            manifest_path,
            data_path,
        )
        errors.extend(manifest_errors)
        warnings.extend(manifest_warnings)

    replay_path_value = config.get("replay_data_path")
    replay_ready = False
    replay_manifest: dict = {}
    if replay_path_value:
        replay_path = Path(replay_path_value)
        replay_ready = check_local_path(
            replay_path,
            label="replay corpus",
            require_data=args.require_data,
            errors=errors,
            warnings=warnings,
        )
        replay_errors, replay_warnings, replay_manifest = validate_manifest(
            args.replay_manifest,
            replay_path,
        )
        errors.extend(replay_errors)
        warnings.extend(replay_warnings)

    corpus = validate_corpus(data_path, requested_dataset=config.get("training_dataset")) if data_ready else None
    replay = validate_corpus(Path(replay_path_value)) if replay_ready else None
    report = {
        "schema_version": "mmsat-stage2-validation-v1",
        "valid": not errors,
        "config": str(args.config),
        "errors": errors,
        "warnings": warnings,
        "data": {
            "path": str(data_path),
            "manifest": str(manifest_path) if manifest_path else None,
            "status": manifest.get("status"),
            "corpus": corpus,
        },
        "replay": {
            "path": replay_path_value,
            "fraction": config.get("replay_fraction", 0.0),
            "manifest": str(args.replay_manifest) if replay_path_value else None,
            "status": replay_manifest.get("status"),
            "corpus": replay,
        },
        "launch": f"uv run python wtpsplit/train/train_SM.py {args.config}",
        "evaluation": "Use scripts/evaluate_mmsat.py with an external exact/partial JSONL packet.",
    }
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
