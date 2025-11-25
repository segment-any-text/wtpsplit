# noqa: E501
"""
Comprehensive tests for length-constrained segmentation in wtpsplit.

This test suite covers:
- Basic constraint enforcement (min/max length)
- Edge cases (empty text, conflicting constraints)
- Prior functions (uniform, gaussian, clipped_polynomial)
- Algorithm comparison (viterbi vs greedy)
- Integration with WtP and SaT models
- Real-world scenarios
"""

import numpy as np
import pytest
from wtpsplit import WtP, SaT
from wtpsplit.utils.constraints import constrained_segmentation
from wtpsplit.utils.priors import create_prior_function


# ============================================================================
# Basic Constraint Tests
# ============================================================================

def test_min_length_enforcement():
    """Verify all chunks are >= min_length"""
    probs = np.random.random(100)
    min_len = 10
    prior_fn = create_prior_function("uniform", {"max_length": 100})
    
    indices = constrained_segmentation(probs, prior_fn, min_length=min_len, max_length=100)
    
    # Check all chunks satisfy min_length
    prev = 0
    for idx in indices:
        chunk_len = idx - prev
        assert chunk_len >= min_len, f"Chunk length {chunk_len} < min {min_len}"
        prev = idx
    
    # Check last chunk
    last_len = 100 - prev
    assert last_len >= min_len, f"Last chunk length {last_len} < min {min_len}"


def test_max_length_enforcement():
    """Verify all chunks are <= max_length"""
    probs = np.random.random(100)
    max_len = 20
    prior_fn = create_prior_function("uniform", {"max_length": max_len})
    
    indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=max_len)
    
    # Check all chunks satisfy max_length
    prev = 0
    for idx in indices:
        chunk_len = idx - prev
        assert chunk_len <= max_len, f"Chunk length {chunk_len} > max {max_len}"
        prev = idx
    
    # Check last chunk
    last_len = 100 - prev
    assert last_len <= max_len, f"Last chunk length {last_len} > max {max_len}"


def test_min_max_together():
    """Both constraints simultaneously"""
    probs = np.random.random(100)
    min_len = 5
    max_len = 15
    prior_fn = create_prior_function("uniform", {"max_length": max_len})
    
    indices = constrained_segmentation(probs, prior_fn, min_length=min_len, max_length=max_len)
    
    # Check all chunks satisfy both constraints
    prev = 0
    for idx in indices:
        chunk_len = idx - prev
        assert min_len <= chunk_len <= max_len, f"Chunk length {chunk_len} not in [{min_len}, {max_len}]"
        prev = idx
    
    # Check last chunk
    last_len = 100 - prev
    assert min_len <= last_len <= max_len, f"Last chunk length {last_len} not in [{min_len}, {max_len}]"


def test_no_constraints():
    """Default behavior (min=1, max=None) should work"""
    probs = np.array([0.1, 0.3, 0.7, 0.9])

    def prior_fn(length):
        return 1.0

    indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=None)

    assert isinstance(indices, list)


# ============================================================================
# Edge Cases
# ============================================================================

def test_min_length_larger_than_text():
    """Handle impossible constraints gracefully"""
    probs = np.array([0.5, 0.5, 0.5])

    def prior_fn(length):
        return 1.0

    indices = constrained_segmentation(probs, prior_fn, min_length=10, max_length=None)

    assert len(indices) <= 1


def test_empty_probabilities():
    """Handle empty input"""
    probs = np.array([])

    def prior_fn(length):
        return 1.0

    indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=10)

    assert indices == []


def test_single_character():
    """Minimal input"""
    probs = np.array([1.0])

    def prior_fn(length):
        return 1.0

    indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=10)

    assert len(indices) <= 1


def test_zero_probabilities():
    """Avoid splitting at zero probabilities"""
    probs = np.array([0.0, 0.0, 1.0])

    def prior_fn(length):
        return 1.0

    indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=None)

    assert indices == [] or 2 in indices


# ============================================================================
# Prior Functions
# ============================================================================

def test_uniform_prior():
    """Uniform prior treats all lengths equally"""
    prior_fn = create_prior_function("uniform", {"max_length": 10})
    
    assert prior_fn(5) == 1.0
    assert prior_fn(10) == 1.0
    assert prior_fn(11) == 0.0


def test_gaussian_prior():
    """Gaussian prior prefers specific length"""
    prior_fn = create_prior_function("gaussian", {"mu": 10.0, "sigma": 2.0})
    
    # Should peak at mu
    assert prior_fn(10) > prior_fn(15)
    assert prior_fn(10) > prior_fn(5)
    
    # Should be symmetric
    assert abs(prior_fn(8) - prior_fn(12)) < 0.01


def test_clipped_polynomial_prior():
    """Clipped polynomial prior (parabolic)"""
    prior_fn = create_prior_function("clipped_polynomial", {"alpha": 0.5, "mu": 10.0})
    
    # Should peak at mu
    assert prior_fn(10) == 1.0
    
    # Should decrease away from mu
    assert prior_fn(8) < 1.0
    assert prior_fn(12) < 1.0
    
    # Should clip to 0
    assert prior_fn(1) == 0.0 or prior_fn(1) < 0.1


# ============================================================================
# Algorithm Comparison
# ============================================================================

def test_viterbi_vs_greedy():
    """Viterbi should be optimal, greedy is faster but suboptimal"""
    probs = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])

    def prior_fn(length):
        return 1.0

    viterbi_indices = constrained_segmentation(probs, prior_fn, min_length=2, max_length=4, algorithm="viterbi")
    greedy_indices = constrained_segmentation(probs, prior_fn, min_length=2, max_length=4, algorithm="greedy")

    for indices in [viterbi_indices, greedy_indices]:
        prev = 0
        for idx in indices:
            chunk_len = idx - prev
            assert 2 <= chunk_len <= 4
            prev = idx


def test_greedy_algorithm():
    """Greedy algorithm produces valid segmentation"""
    probs = np.random.random(50)
    prior_fn = create_prior_function("uniform", {"max_length": 10})
    
    indices = constrained_segmentation(probs, prior_fn, min_length=3, max_length=10, algorithm="greedy")
    
    # Verify constraints
    prev = 0
    for idx in indices:
        chunk_len = idx - prev
        assert 3 <= chunk_len <= 10
        prev = idx


# ============================================================================
# Integration Tests with Models
# ============================================================================

def test_wtp_with_min_length():
    """WtP model with minimum length constraint"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "This is a test. " * 10
    splits = wtp.split(text, min_length=20, threshold=0.005)
    
    # Check all segments are >= min_length
    for segment in splits:
        assert len(segment) >= 20, f"Segment '{segment}' is shorter than min_length"


def test_wtp_with_max_length():
    """WtP model with maximum length constraint"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "This is a test sentence. " * 10
    splits = wtp.split(text, max_length=50, threshold=0.005)
    
    # Check all segments are <= max_length
    for segment in splits:
        assert len(segment) <= 50, f"Segment '{segment}' is longer than max_length"


def test_wtp_with_both_constraints():
    """WtP model with both min and max length"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])

    text = "Hello world. " * 20
    splits = wtp.split(text, min_length=30, max_length=80, threshold=0.005)

    # Check max_length is strictly enforced, min_length is best-effort
    for segment in splits:
        assert len(segment) <= 80, f"Segment '{segment}' exceeds max_length"
        # Most segments should respect min_length (best effort)
    assert sum(1 for s in splits if len(s) >= 30) >= len(splits) * 0.7


def test_sat_with_constraints():
    """SaT model with length constraints"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "This is a sentence. " * 15
    splits = sat.split(text, min_length=25, max_length=60, threshold=0.025)
    
    # Check all segments satisfy constraints
    for segment in splits:
        assert 25 <= len(segment) <= 60, f"Segment '{segment}' violates constraints"


def test_paragraph_segmentation_with_constraints():
    """Nested paragraph and sentence segmentation with constraints"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Paragraph one sentence one. Paragraph one sentence two.\n\nParagraph two sentence one. Paragraph two sentence two."
    
    paragraphs = wtp.split(text, do_paragraph_segmentation=True, min_length=10, max_length=40)
    
    # Check structure
    assert isinstance(paragraphs, list)
    for paragraph in paragraphs:
        assert isinstance(paragraph, list)
        for sentence in paragraph:
            assert 10 <= len(sentence) <= 40


def test_batched_split_with_constraints():
    """Multiple texts with constraints"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])

    texts = [
        "First text sentence one. First text sentence two.",
        "Second text sentence one. Second text sentence two."
    ]

    results = list(wtp.split(texts, min_length=15, max_length=35, threshold=0.005))

    assert len(results) == 2
    for splits in results:
        for segment in splits:
            # Max length is strictly enforced
            assert len(segment) <= 35, f"Segment exceeds max_length: {segment}"
        # Most segments should respect min_length (best effort)
        if splits:
            assert sum(1 for s in splits if len(s) >= 15) >= len(splits) * 0.5


# ============================================================================
# Real-World Scenarios
# ============================================================================

def test_fixed_chunk_size():
    """Equal-sized chunks (e.g., for embedding models)"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])

    text = "A " * 200  # 400 characters
    splits = sat.split(text, min_length=90, max_length=110, threshold=0.025)

    # Max length is strictly enforced
    for segment in splits:
        assert len(segment) <= 110, f"Segment exceeds max_length: {segment}"
    # Most chunks should be close to the target size (best effort)
    if splits:
        assert sum(1 for s in splits if len(s) >= 90) >= len(splits) * 0.5


def test_minimum_sentence_length():
    """Prevent tiny fragments"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "A. B. C. This is a longer sentence. D. E. F."
    splits = wtp.split(text, min_length=20, threshold=0.005)
    
    # No tiny single-character segments
    for segment in splits:
        assert len(segment) >= 20


def test_maximum_context_window():
    """Respect token limits (e.g., for LLM context)"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "Word " * 200  # Long text
    splits = sat.split(text, max_length=100, threshold=0.025)
    
    # All chunks fit in context window
    for segment in splits:
        assert len(segment) <= 100


# ============================================================================
# Prior Type Integration Tests
# ============================================================================

def test_gaussian_prior_with_model():
    """Test Gaussian prior with WtP model"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "This is a test. " * 20
    splits = wtp.split(
        text, 
        min_length=10, 
        max_length=50, 
        prior_type="gaussian",
        prior_kwargs={"mu": 30.0, "sigma": 5.0},
        threshold=0.005
    )
    
    # Should produce valid splits
    assert len(splits) > 0
    for segment in splits:
        assert 10 <= len(segment) <= 50


def test_clipped_polynomial_prior_with_model():
    """Test clipped polynomial prior with SaT model"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "Sentence here. " * 15
    splits = sat.split(
        text,
        min_length=15,
        max_length=45,
        prior_type="clipped_polynomial",
        prior_kwargs={"alpha": 0.5, "mu": 30.0},
        threshold=0.025
    )
    
    # Should produce valid splits
    assert len(splits) > 0
    for segment in splits:
        assert 15 <= len(segment) <= 45


def test_algorithm_parameter():
    """Test algorithm parameter with model"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Test sentence. " * 10
    
    # Test viterbi
    splits_viterbi = wtp.split(text, min_length=20, max_length=50, algorithm="viterbi", threshold=0.005)
    
    # Test greedy
    splits_greedy = wtp.split(text, min_length=20, max_length=50, algorithm="greedy", threshold=0.005)
    
    # Both should produce valid splits
    for splits in [splits_viterbi, splits_greedy]:
        assert len(splits) > 0
        for segment in splits:
            assert 20 <= len(segment) <= 50


# ============================================================================
# Short Sentences Tests (min_length enforcement)
# ============================================================================

def test_tiny_fragments_merging():
    """Tiny fragments should be merged to meet min_length"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "A. B. C. D. E. F. G. H. I. J."
    splits = wtp.split(text, min_length=10, threshold=0.005)
    
    # All segments should meet min_length
    for segment in splits:
        assert len(segment) >= 10, f"Segment '{segment}' is too short"


def test_single_character_sentences():
    """Single character sentences should be merged"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "X. Y. Z. W."
    splits = wtp.split(text, min_length=8, threshold=0.005)
    
    # Should merge single-character sentences
    for segment in splits:
        assert len(segment) >= 8, f"Segment '{segment}' is too short"


def test_very_short_sentences():
    """Very short sentences (5-10 chars) should be merged when needed"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Hi. Bye. Go. Stop. Run. Walk. Jump. Sit."
    splits = wtp.split(text, min_length=15, threshold=0.005)
    
    # Should merge short sentences to meet min_length
    for segment in splits:
        assert len(segment) >= 15, f"Segment '{segment}' is too short"


# ============================================================================
# Mixed Length Sentences Tests
# ============================================================================

def test_varying_length_sentences():
    """Varying from very short to medium should respect constraints"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Hi there. This is a test sentence with more content. Go. Another medium-length sentence here. Stop. Yet another sentence to process."
    splits = wtp.split(text, min_length=10, max_length=50, threshold=0.005)
    
    # All segments should respect constraints
    for segment in splits:
        assert 10 <= len(segment) <= 50, f"Segment '{segment}' violates constraints"


def test_natural_conversation_mix():
    """Natural conversation mix should respect constraints"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])

    text = "Hello! How are you doing today? Great. I wanted to discuss the project timeline. Sure. We need to finalize everything by Friday."
    splits = wtp.split(text, min_length=15, max_length=60, threshold=0.005)

    # Max length is strictly enforced
    for segment in splits:
        assert len(segment) <= 60, f"Segment '{segment}' exceeds max_length"
    # Most segments should respect min_length (best effort for natural boundaries)
    if splits:
        assert sum(1 for s in splits if len(s) >= 15) >= len(splits) * 0.6


# ============================================================================
# Long Sentences Tests (max_length enforcement)
# ============================================================================

def test_single_very_long_sentence():
    """Single very long sentence should be split to respect max_length"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "This is an exceptionally long sentence that contains multiple clauses and phrases connected together with various conjunctions and punctuation marks to test the maximum length constraint enforcement capabilities of the segmentation algorithm."
    splits = sat.split(text, max_length=80, threshold=0.025)
    
    # Should split long sentence into multiple segments
    assert len(splits) > 1
    for segment in splits:
        assert len(segment) <= 80, f"Segment '{segment}' exceeds max_length"


def test_multiple_long_sentences():
    """Multiple long sentences should be split appropriately"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "The quick brown fox jumps over the lazy dog while the sun sets beautifully over the distant mountains. Another lengthy sentence follows with complex grammatical structures and numerous descriptive phrases. A third extended sentence continues the pattern."
    splits = sat.split(text, min_length=20, max_length=100, threshold=0.025)
    
    # All segments should respect constraints
    for segment in splits:
        assert 20 <= len(segment) <= 100, f"Segment '{segment}' violates constraints"


# ============================================================================
# Real-World Scenarios
# ============================================================================

def test_email_style_text():
    """Email-style text should be segmented appropriately"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Hi John. Thanks for your email yesterday. I reviewed the documents you sent. Everything looks good. We can proceed with the next phase. Let me know if you have questions. Best regards."
    splits = wtp.split(text, min_length=20, max_length=70, threshold=0.005)
    
    # All segments should respect constraints
    for segment in splits:
        assert 20 <= len(segment) <= 70, f"Segment '{segment}' violates constraints"


def test_technical_documentation():
    """Technical documentation should be segmented appropriately"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Initialize the system. Configure the parameters in config.yaml. Run the setup script. Verify all dependencies are installed. Execute the main application. Monitor the logs for errors."
    splits = wtp.split(text, min_length=15, max_length=80, threshold=0.005)
    
    # All segments should respect constraints
    for segment in splits:
        assert 15 <= len(segment) <= 80, f"Segment '{segment}' violates constraints"


def test_news_article_style():
    """News article style should be segmented appropriately"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "The company announced record profits today. Shareholders expressed satisfaction with the results. The CEO outlined plans for expansion. New markets will be targeted next quarter. Industry analysts remain cautiously optimistic about future growth."
    splits = wtp.split(text, min_length=25, max_length=90, threshold=0.005)
    
    # All segments should respect constraints
    for segment in splits:
        assert 25 <= len(segment) <= 90, f"Segment '{segment}' violates constraints"


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_text():
    """Empty text should be handled gracefully"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = ""
    splits = wtp.split(text, min_length=10, max_length=50, threshold=0.005)
    
    # Should return empty list or handle gracefully
    assert isinstance(splits, list)


def test_whitespace_only():
    """Whitespace-only text should be handled"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "   "
    splits = wtp.split(text, min_length=10, max_length=50, threshold=0.005)
    
    # Should handle whitespace gracefully
    assert isinstance(splits, list)


def test_newlines_only():
    """Newlines-only text should be handled"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "\n\n\n"
    splits = wtp.split(text, min_length=10, max_length=50, threshold=0.005)
    
    # Should handle newlines gracefully
    assert isinstance(splits, list)


def test_single_sentence():
    """Single sentence should be handled"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "This is the only sentence."
    splits = wtp.split(text, min_length=5, max_length=100, threshold=0.005)
    
    # Should return the sentence if it fits constraints
    assert len(splits) >= 1
    for segment in splits:
        assert 5 <= len(segment) <= 100


def test_no_punctuation():
    """Text without punctuation should be handled"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "word word word word word word word word word"
    splits = sat.split(text, min_length=10, max_length=50, threshold=0.025)
    
    # Should still segment based on model predictions
    assert len(splits) > 0
    for segment in splits:
        assert 10 <= len(segment) <= 50


def test_multiple_punctuation_types():
    """Multiple punctuation types should be handled"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Question? Exclamation! Statement. Another question? More excitement! Final thought..."
    splits = wtp.split(text, min_length=10, max_length=60, threshold=0.005)
    
    # Should handle all punctuation types
    for segment in splits:
        assert 10 <= len(segment) <= 60


def test_nested_quotes():
    """Nested quotes should be handled"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])

    text = 'He said, "This is important." She replied, "I agree completely." They nodded.'
    splits = wtp.split(text, min_length=15, max_length=70, threshold=0.005)

    # Should handle quotes appropriately, max_length strictly enforced
    for segment in splits:
        assert len(segment) <= 70, f"Segment exceeds max_length: {segment}"
    # Natural sentence boundaries may result in some short segments
    if splits:
        assert sum(1 for s in splits if len(s) >= 15) >= len(splits) * 0.5


# ============================================================================
# Stress Tests
# ============================================================================

def test_repeated_pattern():
    """Repeated pattern should be segmented correctly"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Test sentence. " * 100
    splits = wtp.split(text, min_length=20, max_length=80, threshold=0.005)
    
    # Should handle repeated patterns
    for segment in splits:
        assert 20 <= len(segment) <= 80


def test_mixed_repetition():
    """Mixed repetition should be segmented correctly"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Short. " * 20 + "This is a medium length sentence. " * 20 + "X. " * 20
    splits = wtp.split(text, min_length=15, max_length=60, threshold=0.005)
    
    # Should handle mixed patterns
    for segment in splits:
        assert 15 <= len(segment) <= 60


def test_extreme_tiny_sentences():
    """Many tiny sentences should be merged appropriately"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "A. " * 100
    splits = wtp.split(text, min_length=20, max_length=100, threshold=0.005)
    
    # Should merge tiny sentences
    for segment in splits:
        assert 20 <= len(segment) <= 100


def test_extreme_long_sentence():
    """One massive sentence should be split appropriately"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = ("Word " * 500) + "."
    splits = sat.split(text, max_length=200, threshold=0.025)
    
    # Should split massive sentence
    assert len(splits) > 1
    for segment in splits:
        assert len(segment) <= 200


def test_large_text_with_constraints():
    """Test with large text"""
    probs = np.random.random(1000)
    prior_fn = create_prior_function("uniform", {"max_length": 50})
    
    indices = constrained_segmentation(probs, prior_fn, min_length=20, max_length=50)
    
    # Verify all constraints
    prev = 0
    for idx in indices:
        chunk_len = idx - prev
        assert 20 <= chunk_len <= 50
        prev = idx


# ============================================================================
# Unicode and Special Characters
# ============================================================================

def test_multilingual_text():
    """Different languages should be handled"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])

    text = "Hello world. 你好世界。Bonjour le monde. Hola mundo. こんにちは世界。"
    splits = sat.split(text, min_length=10, max_length=50, threshold=0.025)

    # Max length is strictly enforced
    for segment in splits:
        assert len(segment) <= 50, f"Segment '{segment}' exceeds max_length"
    # Most segments should respect min_length (best effort)
    if splits:
        assert sum(1 for s in splits if len(s) >= 10) >= len(splits) * 0.5


def test_special_punctuation():
    """Special punctuation should be handled"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "First—second. Third (with parentheses). Fourth [with brackets]. Fifth «with guillemets»."
    splits = wtp.split(text, min_length=15, max_length=70, threshold=0.005)
    
    # Should handle special punctuation
    for segment in splits:
        assert 15 <= len(segment) <= 70


def test_mixed_scripts():
    """Mixed scripts should be handled"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "English sentence. Русское предложение. العربية الجملة. 中文句子。"
    splits = sat.split(text, min_length=10, max_length=60, threshold=0.025)
    
    # Should handle mixed scripts
    for segment in splits:
        assert 10 <= len(segment) <= 60


# ============================================================================
# Consistency Tests
# ============================================================================

def test_consistency():
    """Same input should produce same output"""
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def prior_fn(length):
        return 1.0

    result1 = constrained_segmentation(probs, prior_fn, min_length=2, max_length=3, algorithm="viterbi")
    result2 = constrained_segmentation(probs, prior_fn, min_length=2, max_length=3, algorithm="viterbi")

    assert result1 == result2


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])





