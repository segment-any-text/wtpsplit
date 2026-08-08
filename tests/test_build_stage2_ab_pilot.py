from scripts.build_stage2_ab_pilot import deterministic_cap


def test_deterministic_cap_normalizes_deduplicates_and_limits():
    sentences = [" A sentence. ", "A   sentence.", "B.", "C."]
    first = deterministic_cap(sentences, limit=2)
    second = deterministic_cap(list(reversed(sentences)), limit=2)
    assert first == second
    assert len(first) == 2
    assert len(set(first)) == 2


def test_deterministic_cap_drops_empty_sentences():
    assert deterministic_cap(["", " ", "Kept."], limit=10) == ["Kept."]
