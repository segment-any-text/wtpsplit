from scripts.audit_tier_b_parallel import normalized_hash
from scripts.build_tier_c_forward import build_corpus, clean_translations


class FakeTranslator:
    def __call__(self, sentences, language):
        return [f"{language}: {index}" for index, _ in enumerate(sentences)]


def test_clean_translations_filters_invalid_outputs():
    source = ["One.", "Two.", "Three.", "Four.", "Five."]
    translated = ["Uno.", "", "Three.", "Uno.", "Reserved."]
    clean, stats = clean_translations(
        source,
        translated,
        {normalized_hash("Reserved.")},
    )
    assert clean == ["Uno."]
    assert stats == {
        "empty_removed": 1,
        "source_identical_removed": 1,
        "duplicate_removed": 1,
        "evaluation_overlap_removed": 1,
    }


def test_build_corpus_emits_supported_training_schema():
    corpus, statistics = build_corpus(
        ["First.", "Second."],
        ["bod_Tibt", "dzo_Tibt"],
        FakeTranslator(),
        lambda _: set(),
    )
    assert set(corpus) == {"bod_Tibt", "dzo_Tibt"}
    assert corpus["bod_Tibt"]["sentence"]["nllb"]["data"] == []
    assert statistics["dzo_Tibt"]["clean_train_sentences"] == 2
