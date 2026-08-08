import json

from wtpsplit.data_acquisition.contamination import ReferenceIndex, normalize_text


def test_normalization_catches_exact_whitespace_and_case_variation():
    index = ReferenceIndex()
    index.add_text("This is a frozen evaluation sentence.", "ref-1", "test", "eng_Latn")
    match = index.match("  THIS is a frozen   evaluation sentence. ", "eng_Latn")
    assert match["match"] == "exact"


def test_near_duplicate_is_quarantined_conservatively():
    index = ReferenceIndex()
    index.add_text(
        "A sufficiently long evaluation document contains several words and a stable structure.",
        "ref-1",
        "test",
        "eng_Latn",
    )
    match = index.match(
        "A sufficiently long evaluation document contains several words and one stable structure.",
        "eng_Latn",
    )
    assert match is not None
    assert match["match"] in {"exact", "near"}


def test_payload_roundtrip_preserves_matches():
    index = ReferenceIndex()
    index.add_text(
        "Another long frozen evaluation sentence for a round trip.",
        "ref-2",
        "test",
        "eng_Latn",
    )
    restored = ReferenceIndex.from_payload(json.loads(json.dumps(index.to_payload())))
    assert restored.match("Another long frozen evaluation sentence for a round trip.", "eng_Latn")["match"] == "exact"
    assert normalize_text(" A  B ") == "a b"


def test_unrelated_long_text_is_not_flagged():
    index = ReferenceIndex()
    index.add_text(
        "This reference discusses multilingual sentence segmentation and evaluation.",
        "ref-3",
        "test",
        "eng_Latn",
    )
    assert (
        index.match(
            "A recipe combines flour, butter, sugar, and fruit before baking in an oven.",
            "eng_Latn",
        )
        is None
    )


def test_same_text_in_another_language_is_not_contamination():
    index = ReferenceIndex()
    index.add_text("Shared punctuation-heavy boilerplate.", "ref-4", "test", "eng_Latn")
    assert index.match("Shared punctuation-heavy boilerplate.", "deu_Latn") is None
