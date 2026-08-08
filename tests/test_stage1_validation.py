from pathlib import Path

from scripts.build_stage1 import bounded_caps
from scripts.dry_run_stage1 import build_report
from wtpsplit.data_acquisition.stage1_validation import validate_stage1_metadata
from wtpsplit.data_acquisition.stage1_web import (
    BuildOptions,
    JsonlShardWriter,
    SourceSpec,
    build_paragraphs,
)


def test_bounded_caps_keeps_all_languages_and_never_exceeds_source_cap():
    caps = {"a": 100, "b": 10_000}
    assert bounded_caps(caps, max_chars_per_language=500) == {"a": 100, "b": 500}
    assert bounded_caps(caps, cap_scale=0.01) == {"a": 1, "b": 100}


def test_validator_checks_receipt_and_can_verify_hashes(tmp_path: Path):
    source = SourceSpec(
        lang="xx",
        corpus="fineweb2",
        dataset="fixture/web",
        config="xx_Latn",
        revision="rev-1",
    )
    build_paragraphs(
        caps={"xx": 10},
        sources={"xx": source},
        options=BuildOptions(output_dir=tmp_path, unit="paragraph", valid_ratio=0.2),
        document_provider=lambda _: iter(["One.\nTwo!", "Three?\nFour."]),
        writer=JsonlShardWriter(),
    )
    report = validate_stage1_metadata(
        tmp_path / "metadata.json",
        expected_languages=["xx"],
        verify_hashes=True,
        minimum_cap_fill=0.5,
    )
    assert report["valid"]
    assert report["counts"]["languages_complete"] == 1

    train_path = tmp_path / "train" / "xx.parquet"
    train_path.write_text(train_path.read_text() + "{}\n", encoding="utf-8")
    bad = validate_stage1_metadata(tmp_path / "metadata.json", expected_languages=["xx"], verify_hashes=True)
    assert not bad["valid"]
    assert any("sha256 mismatch" in error for error in bad["errors"])


def test_full_stage1_dry_run_covers_matched_scaleout_and_backbones():
    report = build_report(nproc=4, smoke_chars=1_000)
    assert report["valid"]
    assert report["matched_85_language_track"]["fineweb2"]["languages"] == 85
    assert report["scaleout_track"]["language_script_pairs"] == 1870
    assert report["scaleout_track"]["source_composition_warning_pairs"] == 1123
    assert len(report["primary_training_matrix"]["arms"]) == 4
