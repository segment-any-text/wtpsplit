import pytest

from wtpsplit.evaluation.frozen import evaluate, fit_thresholds, validate_rows


def row(identifier, split, language, scores, gold, reference="exact", script="LATIN", slice_name="core"):
    return {
        "id": identifier,
        "split": split,
        "language_script": language,
        "script": script,
        "slice": slice_name,
        "reference_type": reference,
        "text": "abcd",
        "scores": scores,
        "gold_offsets": gold,
    }


def test_thresholds_are_fit_only_on_exact_dev_and_fall_back_globally():
    rows = [
        row("dev-a", "dev", "aaa_Latn", [0.01, 0.9, 0.01], [2]),
        row("partial-dev-cannot-leak", "dev", "bbb_Latn", [0.9, 0.9, 0.9], [1], "partial"),
        row("test-cannot-leak", "test", "aaa_Latn", [0.8, 0.8, 0.8], [1]),
        row("unseen-dev-language", "test", "bbb_Latn", [0.01, 0.9, 0.01], [2]),
    ]
    thresholds = fit_thresholds(validate_rows(rows), "per-language", grid=(0.1, 0.5, 0.95))
    assert thresholds == {"aaa_Latn": 0.5, "bbb_Latn": 0.5}


def test_partial_reference_does_not_count_unknown_predictions_as_false_positives():
    rows = [
        row("dev", "dev", "aaa_Latn", [0.01, 0.9, 0.01], [2]),
        row("test-exact", "test", "aaa_Latn", [0.9, 0.9, 0.01], [2]),
        row("test-partial", "test", "bbb_Latn", [0.9, 0.9, 0.01], [2], "partial"),
    ]
    result = evaluate(rows, bootstrap=0)
    assert result["micro_exact"]["fp"] == 1
    assert result["known_boundary_partial"]["recall"] == 1.0
    assert "precision" not in result["known_boundary_partial"]
    assert result["exact_errors"]["over_splitting_false_positives"] == 1
    assert result["macro_language_f1_ci"] is None


def test_rejects_invalid_offsets():
    rows = [row("bad", "dev", "aaa_Latn", [0.1, 0.2, 0.3], [4])]
    with pytest.raises(ValueError, match="gold_offsets"):
        validate_rows(rows)
