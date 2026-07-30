import pytest
import torch

from wtpsplit.evaluation.diagnostics.boundary_ceiling import separator_for
from wtpsplit.train.sm_data import (
    is_monolingual_language_code,
    prepare_sentence_datasets,
    select_training_dataset,
)


def dataset(train_data, eval_data=None):
    return {
        "meta": {"train_data": train_data},
        "data": [] if eval_data is None else eval_data,
    }


@pytest.mark.parametrize(
    ("language_code", "expected"),
    [
        ("en", True),
        ("bod_Tibt", True),
        ("arz_Arab", True),
        ("arz_Latn", True),
        ("sat_Olck", True),
        ("en-de", False),
        ("en_de", False),
        ("deu_latn", False),
        ("deu_Latin", False),
        ("foo_Latn_extra", False),
    ],
)
def test_monolingual_language_code_accepts_iso_script_suffixes(language_code, expected):
    assert is_monolingual_language_code(language_code) is expected


def test_default_selection_preserves_stage3_priority_and_projected_fallback():
    mixed = {
        "ud": dataset(["UD sentence."]),
        "projected": dataset(["Projected sentence."]),
    }
    projected_only = {"projected": dataset(["Projected sentence."])}

    assert select_training_dataset(mixed) == "ud"
    assert select_training_dataset(mixed, "projected") == "projected"
    assert select_training_dataset(projected_only) == "projected"


def test_tatoeba_is_selected_after_ud_and_before_legacy_opus100():
    mixed = {
        "tatoeba": dataset(["Tatoeba sentence."]),
        "opus100": dataset(["Legacy OPUS-100 sentence."]),
    }
    with_ud = {
        "ud": dataset(["Gold sentence."]),
        **mixed,
    }

    assert select_training_dataset(mixed) == "tatoeba"
    assert select_training_dataset(with_ud) == "ud"
    assert select_training_dataset(mixed, "tatoeba") == "tatoeba"


def test_script_language_codes_resolve_sentence_separators():
    assert separator_for("zho_Hans", "CJK") == ""
    assert separator_for("bod_Tibt", "TIBETAN") == ""
    assert separator_for("khm_Khmr", "KHMER") == ""
    # Thai uses spaces at sentence boundaries despite lacking inter-word spaces.
    assert separator_for("tha_Thai", "THAI") == " "
    assert separator_for("deu_Latn", "LATIN") == " "


def test_projected_pth_schema_accepts_script_variants_and_rejects_pairs(tmp_path):
    corpus = {
        "bod_Tibt": {
            "sentence": {
                "projected": dataset(
                    ["འདི་ནི་ཚོད་ལྟའི་ཚིག་གྲུབ་ཅིག་ཡིན།", "འདི་ནི་གཞན་ཞིག་ཡིན།"],
                    [["ཚོད་ལྟ།", "གཞན་ཞིག།"]],
                )
            }
        },
        "arz_Latn": {
            "sentence": {
                "projected": dataset(
                    ["Di gomla lel tadrib.", "Di gomla tanya."],
                    [["Gomla.", "Gomla tanya."]],
                )
            }
        },
        "en-de": {"sentence": {"projected": dataset(["Code switched."])}},
        "en_de": {"sentence": {"projected": dataset(["Also code switched."])}},
    }
    data_path = tmp_path / "stage2-projected.pth"
    torch.save(corpus, data_path)

    loaded = torch.load(data_path, weights_only=True)
    train, evaluation = prepare_sentence_datasets(
        loaded,
        selected_languages=None,
        requested_training_dataset="projected",
        no_sm_corruption=True,
        max_train_sentences_per_dataset=1,
        max_eval_instances_per_dataset=1,
    )

    assert set(train) == {"bod_Tibt", "arz_Latn"}
    assert set(evaluation) == {"bod_Tibt", "arz_Latn", "en-de", "en_de"}
    assert train["bod_Tibt"]["uncorrupted"] == ["འདི་ནི་ཚོད་ལྟའི་ཚིག་གྲུབ་ཅིག་ཡིན།"]
    assert train["arz_Latn"]["uncorrupted"] == ["Di gomla lel tadrib."]
    assert evaluation["bod_Tibt"]["projected"] == [["ཚོད་ལྟ།", "གཞན་ཞིག།"]]


def test_explicit_projected_selection_overrides_ud_in_mixed_corpus():
    corpus = {
        "deu_Latn": {
            "sentence": {
                "ud": dataset(["Gold sentence."]),
                "projected": dataset(["Weakly projected sentence."]),
            }
        }
    }

    train, _ = prepare_sentence_datasets(
        corpus,
        selected_languages={"deu_Latn"},
        requested_training_dataset="projected",
        no_sm_corruption=True,
        max_train_sentences_per_dataset=10,
        max_eval_instances_per_dataset=1,
    )

    assert train["deu_Latn"]["uncorrupted"] == ["Weakly projected sentence."]


def test_projected_data_requires_corruptions_unless_disabled():
    corpus = {
        "bod_Tibt": {
            "sentence": {
                "projected": dataset(["ཚོད་ལྟ།"]),
            }
        }
    }

    with pytest.raises(ValueError, match="no_sm_corruption=true"):
        prepare_sentence_datasets(
            corpus,
            selected_languages=None,
            requested_training_dataset="projected",
            no_sm_corruption=False,
            max_train_sentences_per_dataset=10,
            max_eval_instances_per_dataset=1,
        )


def test_empty_historical_corruption_entries_are_preserved_for_matched_stage3_runs():
    corpus = {
        "en": {
            "sentence": {
                "ud": dataset(["Clean."]),
                "ud-corrupted-asr": dataset(["", "First usable.", "Second usable."]),
                "ud-corrupted-social-media": dataset(["Social."]),
            }
        }
    }

    train, _ = prepare_sentence_datasets(
        corpus,
        selected_languages={"en"},
        requested_training_dataset=None,
        no_sm_corruption=False,
        max_train_sentences_per_dataset=1,
        max_eval_instances_per_dataset=1,
    )

    assert train["en"]["corrupted-asr"] == [""]


def test_empty_projected_sentences_are_rejected():
    corpus = {
        "bod_Tibt": {
            "sentence": {
                "projected": dataset(["ཚོད་ལྟ།", ""]),
            }
        }
    }

    with pytest.raises(ValueError, match="contains empty sentences"):
        prepare_sentence_datasets(
            corpus,
            selected_languages={"bod_Tibt"},
            requested_training_dataset="projected",
            no_sm_corruption=True,
            max_train_sentences_per_dataset=10,
            max_eval_instances_per_dataset=1,
        )


def test_requested_projected_language_fails_early_when_data_is_missing():
    corpus = {"bod_Tibt": {"sentence": {"ud": dataset(["Gold sentence."])}}}

    with pytest.raises(ValueError, match="Requested languages produced no training data"):
        prepare_sentence_datasets(
            corpus,
            selected_languages={"bod_Tibt"},
            requested_training_dataset="projected",
            no_sm_corruption=True,
            max_train_sentences_per_dataset=10,
            max_eval_instances_per_dataset=1,
        )
