from scripts.build_historical_stage2_control import build_control


def dataset(sentences):
    return {
        "meta": {"train_data": sentences},
        "data": [],
    }


def test_build_control_maps_historical_codes_and_records_missing():
    historical = {
        "en": {"sentence": {"ud": dataset(["One.", "Two."])}},
        "ig": {"sentence": {"opus100": dataset(["Otu.", "Abụọ."])}},
    }
    corpus, rows, missing = build_control(
        historical,
        {"eng_Latn", "ibo_Latn", "bod_Tibt"},
        10,
        lambda _: set(),
    )
    assert set(corpus) == {"eng_Latn", "ibo_Latn"}
    assert {row["dataset"] for row in rows} == {"ud", "opus100"}
    assert missing == ["bod_Tibt"]
