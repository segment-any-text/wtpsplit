from scripts.build_stage1_sampling_plan import (
    build_gap_rows,
    build_sampling_rows,
)


def sample_row(**updates):
    row = {
        "language_script": "abc_Latn",
        "train_documents": 20_000,
        "official_test_documents": 100,
        "hf_streaming_supported": True,
        "stage1_sampling_priority": "P1_evaluation_coverage",
        "bible_ratio": 0.9,
        "quality_flags": ["bible_dominated"],
    }
    row.update(updates)
    return row


def test_sampling_caps_large_bible_dominated_sources():
    rows = build_sampling_rows([sample_row()])
    assert rows[0]["target_train_documents"] == 5_000
    assert rows[0]["target_validation_documents"] == 100
    assert rows[0]["target_mixture_weight"] == 1.0


def test_micro_source_is_retained_with_holdout():
    rows = build_sampling_rows(
        [
            sample_row(
                train_documents=50,
                official_test_documents=None,
                bible_ratio=0.0,
                quality_flags=["no_distribution_level_flag"],
            )
        ]
    )
    assert rows[0]["target_train_documents"] == 30
    assert rows[0]["target_validation_documents"] == 20


def test_gap_review_only_accepts_equivalent_korean_alias():
    fineweb = {
        "evaluation_union_missing_details": [
            {"language_script": "kor_Kore", "gap_category": "same_language_other_script_available"},
            {"language_script": "arz_Latn", "gap_category": "same_language_other_script_available"},
        ]
    }
    evaluation = {
        "rows": [
            {
                "language_script": identity,
                "evaluation_tier": "partial_only",
                "segmentation_risk_category": "x",
            }
            for identity in ("kor_Kore", "arz_Latn")
        ]
    }
    rows = build_gap_rows(fineweb, evaluation)
    assert rows[0]["candidate_is_direct_substitute"]
    assert not rows[1]["candidate_is_direct_substitute"]
