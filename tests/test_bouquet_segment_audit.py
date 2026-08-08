import pytest

from scripts.audit_bouquet_segments import (
    deduplicate_english_units,
    summarize,
)


def test_deduplicate_english_units_and_detect_inconsistency():
    rows = [
        {
            "uniq_id": "S1",
            "tgt_text": "One.",
            "par_id": "P1",
            "domain": "test",
            "tags": "",
        },
        {
            "uniq_id": "S1",
            "tgt_text": "One.",
            "par_id": "P1",
            "domain": "test",
            "tags": "",
        },
        {
            "uniq_id": "S2",
            "tgt_text": "Two.",
            "par_id": "P1",
            "domain": "test",
            "tags": "",
        },
        {
            "uniq_id": "S2",
            "tgt_text": "Different.",
            "par_id": "P1",
            "domain": "test",
            "tags": "",
        },
    ]
    units, inconsistencies = deduplicate_english_units(rows)
    assert [unit["uniq_id"] for unit in units] == ["S1", "S2"]
    assert inconsistencies == 1


def test_summarize_classifications():
    units = [
        {"uniq_id": "single", "paragraph_id": "P1"},
        {"uniq_id": "multi", "paragraph_id": "P1"},
        {"uniq_id": "disagreement", "paragraph_id": "P1"},
    ]
    summary, flagged = summarize(
        units,
        [["One."], ["One.", "Two."], ["One.", "Two."]],
        [["One."], ["One.", "Two."], ["One. Two."]],
    )
    assert summary["consensus_single"] == 1
    assert summary["consensus_multi"] == 1
    assert summary["disagreement"] == 1
    assert summary["paragraphs"] == 1
    assert summary["known_row_boundaries"] == 2
    assert summary["consensus_missing_boundaries_lower_bound"] == 1
    assert summary["missing_boundary_share_lower_bound"] == pytest.approx(1 / 3)
    assert summary["consensus_multi_rate"] == pytest.approx(1 / 3)
    assert summary["union_multi_rate"] == pytest.approx(2 / 3)
    assert [item["classification"] for item in flagged] == [
        "consensus_multi",
        "disagreement",
    ]


def test_summarize_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        summarize([{"uniq_id": "S1", "paragraph_id": "P1"}], [], [])
