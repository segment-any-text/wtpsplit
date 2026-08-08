import json

import pytest

from wtpsplit.evaluation.intrinsic_summary import summarize_intrinsic_result


def test_intrinsic_summary_uses_language_macro(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "en": {
                    "ud": {"u": 0.9, "t": 0.8},
                    "opus100": {"u": 0.7, "t": 0.6},
                },
                "de": {"ud": {"u": 0.5, "t": 0.4}},
            }
        )
    )
    result = summarize_intrinsic_result(raw, tmp_path / "summary.json")
    assert result["language_macro_f1_u"] == pytest.approx(0.65)
    assert result["by_dataset"]["ud"]["macro_f1_u"] == pytest.approx(0.7)
    assert result["languages"] == 2
