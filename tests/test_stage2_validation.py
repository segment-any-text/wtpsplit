import json
from pathlib import Path

import torch

from scripts.validate_stage2 import (
    default_manifest_for,
    validate_config,
    validate_corpus,
    validate_manifest,
)


def tiny_corpus() -> dict:
    return {
        "bod_Tibt": {
            "sentence": {
                "nllb": {
                    "meta": {"train_data": ["one", "two"]},
                    "data": [],
                }
            }
        }
    }


def test_config_replay_fields_are_paired():
    valid = {
        "data_path": "a.pth",
        "model_name_or_path": "xlm-roberta-base",
        "tokenizer_name_or_path": "xlm-roberta-base",
        "output_dir": "runs/test",
        "no_sm_corruption": True,
        "replay_data_path": "b.pth",
        "replay_fraction": 0.5,
        "eval_strategy": "no",
    }
    assert validate_config(valid) == ([], [])
    invalid = {**valid, "replay_fraction": 0.0}
    errors, _ = validate_config(invalid)
    assert "set together" in errors[0]


def test_corpus_validation_reads_stage2_schema(tmp_path: Path):
    path = tmp_path / "data.pth"
    torch.save(tiny_corpus(), path)
    result = validate_corpus(path)
    assert result["languages"] == 1
    assert result["datasets"] == {"bod_Tibt": ["uncorrupted"]}
    assert result["embedded_evaluation_languages"] == 0


def test_manifest_hash_is_checked(tmp_path: Path):
    data = tmp_path / "data.pth"
    data.write_bytes(b"stage two")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha256": "bad"}), encoding="utf-8")
    errors, _, _ = validate_manifest(manifest, data)
    assert errors and "SHA-256 mismatch" in errors[0]


def test_default_manifest_is_selected_by_artifact_name():
    assert default_manifest_for(Path("data/mmsat_stage2_abc_pilot_v1.pth")) == Path(
        "data/manifests/mmsat_stage2_abc_pilot_v1.json"
    )
    assert default_manifest_for(Path("data/all_data_11_05-all.pth")) == Path(
        "data/manifests/sat_historical_corpus_v1.json"
    )
    assert default_manifest_for(Path("data/private_experiment.pth")) is None
