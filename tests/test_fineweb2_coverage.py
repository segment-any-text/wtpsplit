from pathlib import Path

from scripts.build_fineweb2_coverage import build_rows, resource_bucket
from scripts.sample_fineweb2 import output_path, sample_one, select_rows


def test_resource_buckets_have_fixed_boundaries():
    assert resource_bucket(999) == "micro_lt_1k_documents"
    assert resource_bucket(1_000) == "low_1k_10k_documents"
    assert resource_bucket(1_000_000) == "very_high_ge_1m_documents"


def test_build_rows_joins_evaluation_and_flags_bible_dominance():
    distribution = [
        {
            "subset": "abc_Latn",
            "split": "train",
            "code": "abc",
            "script": "Latn",
            "name": "Example",
            "family": "Test",
            "words": "1000",
            "documents": "100",
            "utf8_bytes": "5000",
            "parquet_bytes": "2000",
            "bible_ratio": "0.900",
            "wiki_ratio": "0.000",
        }
    ]
    evaluation = {
        "rows": [
            {
                "language_script": "abc_Latn",
                "evaluation_tier": "partial_only",
                "bouquet_baseline_category": "severe_bouquet_failure",
            }
        ]
    }
    rows = build_rows(distribution, evaluation)
    assert rows[0]["stage1_sampling_priority"] == "P0_measured_failure"
    assert "bible_dominated" in rows[0]["quality_flags"]


def test_sample_selection_and_output_path(tmp_path: Path):
    rows = [
        {
            "language_script": "abc_Latn",
            "hf_streaming_supported": True,
            "stage1_sampling_priority": "P0_measured_failure",
        },
        {
            "language_script": "und_Latn",
            "hf_streaming_supported": False,
            "stage1_sampling_priority": "P2_backbone_coverage_extension",
        },
    ]
    assert select_rows(rows, {"abc_Latn"}, set()) == [rows[0]]
    assert output_path(tmp_path, "abc_Latn") == tmp_path / "abc_Latn.jsonl"


def test_existing_sample_is_reused_without_network(tmp_path: Path):
    target = tmp_path / "abc_Latn.jsonl"
    target.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
    result = sample_one(
        {"language_script": "abc_Latn"},
        target,
        max_documents=10,
        remaining_bytes=1000,
        seed=42,
        shuffle_buffer=10,
        revision="revision-1",
    )
    assert result["status"] == "cached"
    assert result["documents"] == 2
