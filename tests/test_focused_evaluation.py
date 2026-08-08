import pytest

from scripts.build_focused_evaluation_packet import make_row
from scripts.validate_focused_evaluation import validate
from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_REVISION


def test_candidate_packet_is_partial_and_provenanced():
    row = make_row("roh_Latn", 0, ["In Satz.", "In zweiter Satz."])
    assert row["reference_type"] == "partial"
    assert row["annotation_status"] == "candidate_not_gold"
    assert row["source_version"] == BOUQUET_REVISION
    assert validate([row])["documents"] == 1


def test_candidate_cannot_be_reported_as_gold():
    row = make_row("roh_Latn", 0, ["In Satz.", "In zweiter Satz."])
    with pytest.raises(ValueError, match="not adjudicated_gold"):
        validate([row], require_gold=True)
