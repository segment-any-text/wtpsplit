import io
import zipfile

from scripts.audit_tier_b_parallel import (
    TATOEBA_LICENSE,
    clean_pairs,
    normalized_hash,
    parse_tatoeba_archive,
)


def archive_bytes(
    source_lines,
    target_lines,
    *,
    readme=f"License: {TATOEBA_LICENSE}",
):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Tatoeba.en-xx.en", "\n".join(source_lines))
        archive.writestr("Tatoeba.en-xx.xx", "\n".join(target_lines))
        archive.writestr("README", readme)
    return buffer.getvalue()


def test_parse_tatoeba_archive_requires_alignment_and_pinned_license():
    pairs, readme = parse_tatoeba_archive(
        archive_bytes(["One.", "Two."], ["Uno.", "Dos."]),
        "en-xx",
        "en",
        "xx",
    )
    assert pairs == [("One.", "Uno."), ("Two.", "Dos.")]
    assert TATOEBA_LICENSE in readme


def test_clean_pairs_deduplicates_and_excludes_evaluation():
    forbidden = {normalized_hash("Reserved evaluation sentence.")}
    clean, stats = clean_pairs(
        [
            ("English.", "Target."),
            ("English duplicate.", " Target. "),
            ("Same.", "Same."),
            ("Reserved.", "Reserved evaluation sentence."),
            ("Empty.", " "),
            ("Second.", "Another target."),
        ],
        forbidden,
    )
    assert clean == ["Target.", "Another target."]
    assert stats == {
        "empty_target_removed": 1,
        "source_target_identical_removed": 1,
        "duplicate_target_removed": 1,
        "evaluation_overlap_removed": 1,
    }


def test_normalized_hash_ignores_whitespace_variants():
    assert normalized_hash("A   sentence.") == normalized_hash(" A sentence. ")
