import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))


def test_scaleout_manifest_hash_chain_and_dataset_revision() -> None:
    evaluation_path = MANIFESTS / "mmsat_dataset_coverage_v1.json"
    coverage_path = MANIFESTS / "fineweb2_stage1_coverage_v1.json"
    evaluation = load(evaluation_path.name)
    coverage = load(coverage_path.name)
    sampling = load("fineweb2_stage1_sampling_plan_v1.json")
    gaps = load("stage1_coverage_gap_review_v1.json")

    assert evaluation["provenance"]["status"] == "frozen_upstream_research_input"
    assert coverage["evaluation_coverage_sha256"] == sha256(evaluation_path)
    assert sampling["evaluation_coverage_sha256"] == sha256(evaluation_path)
    assert gaps["evaluation_coverage_sha256"] == sha256(evaluation_path)
    assert sampling["source_manifest_sha256"] == sha256(coverage_path)
    assert gaps["source_manifest_sha256"] == sha256(coverage_path)
    assert sampling["dataset_revisions"] == coverage["dataset_revisions"]
    assert all(coverage["dataset_revisions"].values())


def test_matched_manifests_are_pinned_and_describe_the_current_builder() -> None:
    caps = load("mc4_test_per_lang_char_mass.json")
    mapping = load("sat_lang_to_fineweb2_v1.json")
    remaps = load("stage1_fineweb_script_remaps_v1.json")

    assert caps["version"] == "mc4_test_per_lang_char_mass_v1"
    assert caps["source_revision"]
    assert "topup_stage1" not in json.dumps(mapping)
    assert "topup_stage1" not in json.dumps(remaps)


def test_contamination_manifest_pins_the_frozen_ud_selection() -> None:
    contamination = load("mmsat_contamination_index_v1.json")
    ud_selection = MANIFESTS / "ud_2_18_frozen_selection_v1.json"

    assert contamination["input_sha256"]["ud_selection"] == sha256(ud_selection)
    assert contamination["reproducibility"]["original_bouquet_revision"] is None
