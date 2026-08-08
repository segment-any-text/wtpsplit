from scripts.audit_ud_tier_a import (
    SOURCES,
    license_policy,
    normalized_hash,
    parse_conllu,
)


def test_tier_a_sources_are_unique():
    languages = [row["language_script"] for row in SOURCES]
    assert len(languages) == 13
    assert len(languages) == len(set(languages))


def test_parse_conllu_counts_text_and_documents():
    parsed = parse_conllu(
        "# newdoc id = d1\n"
        "# text = First sentence.\n"
        "1\tFirst\t_\t_\t_\t_\t0\troot\t_\t_\n\n"
        "# text = Second sentence.\n"
        "1\tSecond\t_\t_\t_\t_\t0\troot\t_\t_\n"
    )
    assert parsed["sentences"] == 2
    assert parsed["documents"] == 1
    assert len(parsed["hashes"]) == 2


def test_normalized_hash_ignores_preprocessing_whitespace():
    assert normalized_hash("One   sentence.") == normalized_hash(
        "One sentence."
    )


def test_noncommercial_license_is_not_release_ready():
    assert "legal_review" in license_policy("CC BY-NC-SA 4.0")
