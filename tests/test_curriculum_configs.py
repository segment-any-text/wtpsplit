import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs/curriculum"


def load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_curriculum_configs_are_small_direct_training_inputs():
    stage1_mc4 = load("stage1_mc4.json")
    stage1_fineweb = load("stage1_fineweb.json")
    stage1_mc4_documents = load("stage1_mc4_documents.json")
    stage1_fineweb_documents = load("stage1_fineweb_documents.json")
    stage1_mc4_mmbert = load("stage1_mc4_mmbert.json")
    stage1_fineweb_mmbert = load("stage1_fineweb_mmbert.json")
    stage2 = load("stage2.json")
    stage2_character = load("stage2_character.json")
    stage3 = load("stage3.json")

    assert stage1_mc4["stage1_dataset_revision"]
    assert stage1_fineweb["train_text_path"].endswith("/train")
    stage1_arms = (
        stage1_mc4,
        stage1_fineweb,
        stage1_mc4_documents,
        stage1_fineweb_documents,
        stage1_mc4_mmbert,
        stage1_fineweb_mmbert,
    )
    for arm in stage1_arms:
        assert arm["max_steps"] == 200000
        assert arm["per_device_train_batch_size"] == 64
        assert arm["gradient_accumulation_steps"] == 8
        assert arm["seed"] == 42
    assert stage1_mc4["non_punctuation_sample_ratio"] == 0.1
    assert stage1_fineweb["non_punctuation_sample_ratio"] == 0.1
    assert stage1_mc4_documents["non_punctuation_sample_ratio"] is None
    assert stage1_fineweb_documents["non_punctuation_sample_ratio"] is None
    assert "documents-matched/train" in stage1_mc4_documents["train_text_path"]
    assert (
        "documents-matched/train"
        in stage1_fineweb_documents["train_text_path"]
    )
    for arm in (stage1_mc4_mmbert, stage1_fineweb_mmbert):
        assert arm["model_name_or_path"] == "jhu-clsp/mmBERT-base"
        assert "tokenizer_name_or_path" not in arm
        assert arm["custom_punctuation_file"] == (
            "punctuation_extended_mmbert_unk.txt"
        )
    assert stage2["no_sm_corruption"] is True
    assert stage2["model_name_or_path"] == "xlm-roberta-base"
    assert stage2["replay_fraction"] == 0.5
    assert stage2["replay_data_path"].endswith("historical_control_v1.pth")
    assert stage2["eval_strategy"] == "no"
    differing_head_fields = {
        key
        for key in set(stage2) | set(stage2_character)
        if stage2.get(key) != stage2_character.get(key)
    }
    assert differing_head_fields == {
        "character_head_init",
        "output_dir",
        "use_character_head",
    }
    assert stage2_character["use_character_head"] is True
    assert stage2_character["character_head_init"] == "random"
    assert stage3["no_sm_corruption"] is False
    assert "training_dataset" not in stage2
    assert "training_dataset" not in stage3


def test_selection_file_points_to_concrete_outputs():
    selections = load("selections.json")
    assert selections["stage1"]["status"] == "pending_matched_runs"
    assert selections["stage1"]["checkpoint"] is None
    assert selections["stage1"]["unit"] in {"paragraph", "document"}
    assert set(selections["stage1"]["alternatives"]) == {
        "fineweb_document",
        "fineweb_paragraph",
        "fineweb_paragraph_mmbert",
        "mc4_document",
        "mc4_paragraph",
        "mc4_paragraph_mmbert",
        "raw_backbone",
    }
    assert set(selections["stage1"]["evidence"]) == {
        "fineweb_document",
        "fineweb_paragraph",
        "fineweb_paragraph_mmbert",
        "mc4_document",
        "mc4_paragraph",
        "mc4_paragraph_mmbert",
    }
    assert selections["stage2_data"]["corpus"].endswith(".pth")
    assert selections["stage2_model"]["checkpoint"].startswith("runs/")
    assert selections["stage2_model"]["status"].startswith("provisional")
    experiments = selections["stage2_experiments"]
    assert experiments["character_head"]["status"].startswith("pending")
    assert experiments["character_head"]["config"].endswith(
        "stage2_character.json"
    )
    tier_d = experiments["tier_d_reverse_document_projection"]
    assert tier_d["status"].startswith("open_research")
    assert tier_d["role"] == (
        "alternative_to_tier_c_forward_sentence_translation"
    )
    assert selections["stage3"]["status"].startswith("optional")
