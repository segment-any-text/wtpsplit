import pytest

from scripts.build_stage2_abc_pilot import merge_corpora


def language(dataset, sentences):
    return {
        "sentence": {
            dataset: {
                "meta": {"train_data": sentences},
                "data": [],
            }
        }
    }


def test_merge_corpora_keeps_one_source_per_language():
    merged = merge_corpora(
        {"eng_Latn": language("ud", ["English."])},
        {"bod_Tibt": language("nllb", ["བོད་ཡིག།"])},
    )
    assert set(merged) == {"eng_Latn", "bod_Tibt"}


def test_merge_corpora_rejects_language_overlap():
    with pytest.raises(ValueError, match="exactly one"):
        merge_corpora(
            {"eng_Latn": language("ud", ["English."])},
            {"eng_Latn": language("nllb", ["Synthetic."])},
        )


def test_merge_corpora_rejects_embedded_evaluation_text():
    invalid = language("ud", ["English."])
    invalid["sentence"]["ud"]["data"] = [["Evaluation."]]
    with pytest.raises(ValueError, match="no evaluation text"):
        merge_corpora({"eng_Latn": invalid}, {})
