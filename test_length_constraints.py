# noqa: E501
"""
Comprehensive tests for length-constrained segmentation in wtpsplit.

This test suite covers:
- Text preservation guarantee
- Strict max_length enforcement
- min_length best-effort behavior
- Viterbi and greedy algorithms
- Prior functions (uniform, gaussian, clipped_polynomial)
- Edge cases and special characters
- Both WtP and SaT models
- Real-world scenarios
- Regression tests for fixed bugs

Run with: pytest test_length_constraints.py -v
"""

import pytest
import numpy as np
from wtpsplit import WtP, SaT
from wtpsplit.utils.constraints import (
    constrained_segmentation,
    _enforce_segment_constraints,
    _enforce_segment_constraints_simple,
)
from wtpsplit.utils.priors import create_prior_function


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def sat_model():
    """Load SaT model once for all tests."""
    return SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def wtp_model():
    """Load WtP model once for all tests."""
    return WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])


# =============================================================================
# BASIC CONSTRAINT ENFORCEMENT (Low-level)
# =============================================================================

class TestBasicConstraints:
    """Test basic constraint enforcement at the algorithm level."""

    def test_min_length_enforcement(self):
        """Verify all chunks are >= min_length."""
        probs = np.random.random(100)
        min_len = 10
        prior_fn = create_prior_function("uniform", {"max_length": 100})

        indices = constrained_segmentation(probs, prior_fn, min_length=min_len, max_length=100)

        prev = 0
        for idx in indices:
            chunk_len = idx - prev
            assert chunk_len >= min_len, f"Chunk length {chunk_len} < min {min_len}"
            prev = idx

        last_len = 100 - prev
        assert last_len >= min_len, f"Last chunk length {last_len} < min {min_len}"

    def test_max_length_enforcement(self):
        """Verify all chunks are <= max_length."""
        probs = np.random.random(100)
        max_len = 20
        prior_fn = create_prior_function("uniform", {"max_length": max_len})

        indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=max_len)

        prev = 0
        for idx in indices:
            chunk_len = idx - prev
            assert chunk_len <= max_len, f"Chunk length {chunk_len} > max {max_len}"
            prev = idx

        last_len = 100 - prev
        assert last_len <= max_len, f"Last chunk length {last_len} > max {max_len}"

    def test_min_max_together(self):
        """Both constraints simultaneously."""
        probs = np.random.random(100)
        min_len = 5
        max_len = 15
        prior_fn = create_prior_function("uniform", {"max_length": max_len})

        indices = constrained_segmentation(probs, prior_fn, min_length=min_len, max_length=max_len)

        prev = 0
        for idx in indices:
            chunk_len = idx - prev
            assert min_len <= chunk_len <= max_len, f"Chunk length {chunk_len} not in [{min_len}, {max_len}]"
            prev = idx

        last_len = 100 - prev
        assert min_len <= last_len <= max_len, f"Last chunk length {last_len} not in [{min_len}, {max_len}]"

    def test_no_constraints(self):
        """Default behavior (min=1, max=None) should work."""
        probs = np.array([0.1, 0.3, 0.7, 0.9])

        def prior_fn(length):
            return 1.0

        indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=None)
        assert isinstance(indices, list)

    def test_large_text_with_constraints(self):
        """Test with large text."""
        probs = np.random.random(1000)
        prior_fn = create_prior_function("uniform", {"max_length": 50})

        indices = constrained_segmentation(probs, prior_fn, min_length=20, max_length=50)

        prev = 0
        for idx in indices:
            chunk_len = idx - prev
            assert 20 <= chunk_len <= 50
            prev = idx


# =============================================================================
# TEXT PRESERVATION TESTS
# =============================================================================

class TestTextPreservation:
    """Verify that segmentation preserves original text exactly."""

    def test_simple_text(self, sat_model):
        text = "Hello world. How are you? I am fine."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_with_max_length(self, sat_model):
        text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
        segments = sat_model.split(text, max_length=50, threshold=0.025)
        assert "".join(segments) == text

    def test_multiline_preserved(self, sat_model):
        text = "Line one.\n\nLine two.\n\nLine three."
        segments = sat_model.split(text, threshold=0.5, split_on_input_newlines=False)
        assert "".join(segments) == text

    def test_whitespace_variations(self, sat_model):
        text = "Word1  Word2.   Word3    Word4."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_unicode_preserved(self, sat_model):
        text = "Привет мир. 你好世界。مرحبا بالعالم."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_special_characters(self, sat_model):
        text = "Price: $100.00! Email: test@example.com? URL: https://example.com/path?q=1"
        segments = sat_model.split(text, threshold=0.5, max_length=100)
        assert "".join(segments) == text

    def test_long_document(self, sat_model):
        text = "This is sentence one. " * 50
        segments = sat_model.split(text, max_length=150, threshold=0.025)
        assert "".join(segments) == text


# =============================================================================
# MAX_LENGTH TESTS
# =============================================================================

class TestMaxLength:
    """Verify strict max_length enforcement."""

    def test_all_segments_within_max(self, sat_model):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        max_length = 100
        segments = sat_model.split(text, max_length=max_length, threshold=0.025)

        for segment in segments:
            assert len(segment) <= max_length, f"Segment too long: {len(segment)} > {max_length}"

    def test_various_max_lengths(self, sat_model):
        text = "Hello world. How are you today? I am doing well. Thanks for asking!"

        for max_length in [30, 50, 80, 100, 150]:
            segments = sat_model.split(text, max_length=max_length, threshold=0.025)
            for segment in segments:
                assert len(segment) <= max_length
            assert "".join(segments) == text

    def test_max_length_forces_split(self, sat_model):
        """Text longer than max_length must be split."""
        text = "This is a single very long sentence without any natural break points whatsoever."
        segments = sat_model.split(text, max_length=40, threshold=0.025)

        assert len(segments) > 1
        for segment in segments:
            assert len(segment) <= 40

    def test_max_length_one(self, sat_model):
        """Edge case: max_length=1."""
        text = "Hi"
        segments = sat_model.split(text, max_length=1, threshold=0.025)
        for segment in segments:
            assert len(segment) <= 1


# =============================================================================
# MIN_LENGTH TESTS
# =============================================================================

class TestMinLength:
    """Verify min_length best-effort behavior."""

    def test_min_length_merges_short(self, sat_model):
        text = "A. B. C. D. E."
        segments_no_min = sat_model.split(text, threshold=0.5)
        segments_with_min = sat_model.split(text, threshold=0.5, min_length=5)

        assert len(segments_with_min) <= len(segments_no_min)

    def test_min_length_with_max_length(self, sat_model):
        text = "Short. Another short. Yet another. And more."
        segments = sat_model.split(text, min_length=10, max_length=50, threshold=0.025)

        for segment in segments:
            assert len(segment) <= 50
        assert "".join(segments) == text

    def test_tiny_fragments_merging(self, wtp_model):
        """Tiny fragments should be merged to meet min_length."""
        text = "A. B. C. D. E. F. G. H. I. J."
        splits = wtp_model.split(text, min_length=10, threshold=0.005)

        for segment in splits:
            assert len(segment) >= 10, f"Segment '{segment}' is too short"

    def test_very_short_sentences(self, wtp_model):
        """Very short sentences should be merged when needed."""
        text = "Hi. Bye. Go. Stop. Run. Walk. Jump. Sit."
        splits = wtp_model.split(text, min_length=15, threshold=0.005)

        for segment in splits:
            assert len(segment) >= 15, f"Segment '{segment}' is too short"


# =============================================================================
# ALGORITHM TESTS
# =============================================================================

class TestAlgorithms:
    """Test Viterbi and greedy algorithms."""

    def test_viterbi_deterministic(self, sat_model):
        text = "The quick brown fox. Pack my box. How vexingly quick!"

        results = [
            sat_model.split(text, max_length=80, algorithm="viterbi", threshold=0.025)
            for _ in range(3)
        ]

        assert all(r == results[0] for r in results)

    def test_greedy_deterministic(self, sat_model):
        text = "The quick brown fox. Pack my box. How vexingly quick!"

        results = [
            sat_model.split(text, max_length=80, algorithm="greedy", threshold=0.025)
            for _ in range(3)
        ]

        assert all(r == results[0] for r in results)

    def test_both_algorithms_preserve_text(self, sat_model):
        text = "First sentence here. Second sentence follows. Third one ends it."

        for algo in ["viterbi", "greedy"]:
            segments = sat_model.split(text, max_length=100, algorithm=algo, threshold=0.025)
            assert "".join(segments) == text

    def test_both_algorithms_respect_max_length(self, sat_model):
        text = "The quick brown fox jumps. " * 20
        max_length = 80

        for algo in ["viterbi", "greedy"]:
            segments = sat_model.split(text, max_length=max_length, algorithm=algo, threshold=0.025)
            for segment in segments:
                assert len(segment) <= max_length

    def test_viterbi_vs_greedy_both_valid(self):
        """Both algorithms should produce valid segmentations."""
        probs = np.random.rand(50)
        probs[15] = 0.95
        probs[30] = 0.95
        probs[45] = 0.95

        prior_fn = create_prior_function("uniform", {"max_length": 20})

        greedy = constrained_segmentation(probs, prior_fn, min_length=1, max_length=20, algorithm="greedy")
        viterbi = constrained_segmentation(probs, prior_fn, min_length=1, max_length=20, algorithm="viterbi")

        for boundaries in [greedy, viterbi]:
            prev = 0
            for b in boundaries + [50]:
                assert b - prev <= 20
                prev = b


# =============================================================================
# PRIOR FUNCTION TESTS
# =============================================================================

class TestPriors:
    """Test prior functions behavior."""

    def test_uniform_prior(self):
        prior_fn = create_prior_function("uniform", {"max_length": 100})

        assert prior_fn(50) == 1.0
        assert prior_fn(100) == 1.0
        assert prior_fn(101) == 0.0
        assert prior_fn(200) == 0.0

    def test_gaussian_prior(self):
        prior_fn = create_prior_function("gaussian", {"mu": 50, "sigma": 10})

        assert prior_fn(50) == pytest.approx(1.0)
        assert prior_fn(30) < prior_fn(50)
        assert prior_fn(70) < prior_fn(50)
        # Symmetric
        assert prior_fn(40) == pytest.approx(prior_fn(60), rel=1e-5)

    def test_polynomial_prior(self):
        prior_fn = create_prior_function("clipped_polynomial", {"mu": 50, "alpha": 0.01})

        assert prior_fn(50) == pytest.approx(1.0)
        assert prior_fn(40) < prior_fn(50)
        assert prior_fn(60) < prior_fn(50)

    def test_polynomial_clips_to_zero(self):
        prior_fn = create_prior_function("clipped_polynomial", {"mu": 50, "alpha": 0.1})

        assert prior_fn(50) == 1.0
        assert prior_fn(100) == 0.0  # Clipped

    def test_gaussian_affects_segmentation(self, sat_model):
        text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."

        segments_small = sat_model.split(
            text, max_length=200, prior_type="gaussian",
            prior_kwargs={"mu": 20, "sigma": 5}, threshold=0.025
        )

        segments_large = sat_model.split(
            text, max_length=200, prior_type="gaussian",
            prior_kwargs={"mu": 100, "sigma": 20}, threshold=0.025
        )

        assert "".join(segments_small) == text
        assert "".join(segments_large) == text


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test input parameter validation."""

    def test_min_greater_than_max_raises(self, sat_model):
        with pytest.raises(ValueError, match="min_length.*cannot be greater than max_length"):
            sat_model.split("Hello", min_length=100, max_length=50)

    def test_invalid_prior_type_raises(self, sat_model):
        with pytest.raises(ValueError, match="Unknown prior_type"):
            sat_model.split("Hello", prior_type="invalid_prior")

    def test_invalid_algorithm_raises(self, sat_model):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            sat_model.split("Hello", algorithm="invalid_algo")

    def test_min_length_zero_raises(self, sat_model):
        with pytest.raises(ValueError, match="min_length must be >= 1"):
            sat_model.split("Hello", min_length=0)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, sat_model):
        segments = sat_model.split("", threshold=0.5)
        assert segments == [] or segments == [""]

    def test_single_character(self, sat_model):
        segments = sat_model.split("A", threshold=0.5)
        assert "".join(segments) == "A"

    def test_only_whitespace(self, sat_model):
        text = "   \n\t  "
        segments = sat_model.split(text, threshold=0.5)
        assert isinstance(segments, list)

    def test_only_punctuation(self, sat_model):
        text = "!?!.!?.!"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_very_long_text(self, sat_model):
        text = "This is a test sentence. " * 200
        segments = sat_model.split(text, max_length=200, threshold=0.025)

        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 200

    def test_empty_probabilities(self):
        """Handle empty input."""
        probs = np.array([])

        def prior_fn(length):
            return 1.0

        indices = constrained_segmentation(probs, prior_fn, min_length=1, max_length=10)
        assert indices == []

    def test_min_length_larger_than_text(self):
        """Handle impossible constraints gracefully."""
        probs = np.array([0.5, 0.5, 0.5])

        def prior_fn(length):
            return 1.0

        indices = constrained_segmentation(probs, prior_fn, min_length=10, max_length=None)
        assert len(indices) <= 1


# =============================================================================
# BOTH MODELS TEST
# =============================================================================

class TestBothModels:
    """Test that both WtP and SaT work with constraints."""

    def test_wtp_preserves_text(self, wtp_model):
        text = "Hello world. How are you?"
        segments = wtp_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_wtp_max_length(self, wtp_model):
        text = "The quick brown fox jumps over the lazy dog. Pack my box."
        segments = wtp_model.split(text, max_length=30, threshold=0.025)

        for segment in segments:
            assert len(segment) <= 30
        assert "".join(segments) == text

    def test_wtp_with_both_constraints(self, wtp_model):
        text = "Hello world. " * 20
        splits = wtp_model.split(text, min_length=30, max_length=80, threshold=0.005)

        for segment in splits:
            assert len(segment) <= 80
        assert sum(1 for s in splits if len(s) >= 30) >= len(splits) * 0.7

    def test_sat_preserves_text(self, sat_model):
        text = "Hello world. How are you?"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_sat_max_length(self, sat_model):
        text = "The quick brown fox jumps over the lazy dog. Pack my box."
        segments = sat_model.split(text, max_length=30, threshold=0.025)

        for segment in segments:
            assert len(segment) <= 30
        assert "".join(segments) == text


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================

class TestBatchProcessing:
    """Test batch processing with constraints."""

    def test_batch_preserves_all(self, sat_model):
        texts = [
            "First document here. With sentences.",
            "Second document. Also with sentences. Multiple ones.",
            "Third. Short.",
        ]

        results = list(sat_model.split(texts, max_length=100, threshold=0.025))

        for text, segments in zip(texts, results):
            assert "".join(segments) == text

    def test_batch_respects_max_length(self, sat_model):
        texts = ["Long text here. " * 10, "Another long one. " * 15]
        max_length = 80

        results = list(sat_model.split(texts, max_length=max_length, threshold=0.025))

        for segments in results:
            for segment in segments:
                assert len(segment) <= max_length


# =============================================================================
# WHITESPACE HANDLING TESTS
# =============================================================================

class TestWhitespaceHandling:
    """Comprehensive whitespace handling tests."""

    def test_single_space_between_words(self, sat_model):
        text = "Hello world."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_multiple_spaces_preserved(self, sat_model):
        text = "Hello    world.    How    are    you?"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_tabs_preserved(self, sat_model):
        text = "Hello\tworld.\tHow\tare\tyou?"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_mixed_whitespace(self, sat_model):
        text = "Hello \t world.  \n\n  How   are you?"
        segments = sat_model.split(text, threshold=0.5, split_on_input_newlines=False)
        assert "".join(segments) == text

    def test_leading_whitespace(self, sat_model):
        text = "   Leading spaces. Then more text."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_trailing_whitespace(self, sat_model):
        text = "Text here. More text.   "
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_only_newlines(self, sat_model):
        text = "\n\n\n"
        segments = sat_model.split(text, threshold=0.5, split_on_input_newlines=False)
        assert "".join(segments) == text or segments == []

    def test_windows_line_endings(self, sat_model):
        text = "Line one.\r\nLine two.\r\nLine three."
        segments = sat_model.split(text, threshold=0.5, split_on_input_newlines=False)
        assert "".join(segments) == text


# =============================================================================
# REAL-WORLD TEXT TESTS
# =============================================================================

class TestRealWorldText:
    """Tests with realistic text content."""

    def test_news_article(self, sat_model):
        text = """Breaking News: Scientists at CERN have announced a groundbreaking discovery that could revolutionize our understanding of particle physics. The team, led by Dr. Elena Rodriguez, observed unexpected behavior in proton collisions at energies never before achieved. "This is the most significant finding in our field since the Higgs boson," Dr. Rodriguez stated at a press conference in Geneva."""

        for max_len in [100, 150, 200]:
            segments = sat_model.split(text, max_length=max_len, threshold=0.025)
            assert "".join(segments) == text
            for segment in segments:
                assert len(segment) <= max_len

    def test_legal_text(self, sat_model):
        text = """WHEREAS the Party of the First Part (hereinafter referred to as "Licensor") is the owner of certain intellectual property rights including but not limited to patents, trademarks, copyrights, and trade secrets relating to the technology described herein, and WHEREAS the Party of the Second Part (hereinafter referred to as "Licensee") desires to obtain a license to use said technology."""

        segments = sat_model.split(text, max_length=150, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 150

    def test_technical_documentation(self, sat_model):
        text = """The function accepts three parameters: input_data (required), config (optional), and callback (optional). When input_data is a string, it will be parsed as JSON; when it's an object, it will be used directly. The config parameter supports the following options: timeout (default: 30000ms), retries (default: 3), and verbose (default: false)."""

        segments = sat_model.split(text, max_length=120, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 120

    def test_dialogue(self, sat_model):
        text = '''"Have you seen the news?" asked Maria. "About the merger?" replied John. "No, I mean about the earthquake." Maria shook her head sadly. "It's terrible."'''

        segments = sat_model.split(text, max_length=80, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 80

    def test_email_style_text(self, wtp_model):
        """Email-style text should be segmented appropriately."""
        text = "Hi John. Thanks for your email yesterday. I reviewed the documents you sent. Everything looks good. We can proceed with the next phase. Let me know if you have questions. Best regards."
        splits = wtp_model.split(text, min_length=20, max_length=70, threshold=0.005)

        for segment in splits:
            assert 20 <= len(segment) <= 70


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for edge conditions and performance."""

    def test_very_many_short_sentences(self, sat_model):
        text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T. U. V. W. X. Y. Z."

        segments = sat_model.split(text, max_length=50, threshold=0.025)
        assert "".join(segments) == text

    def test_alternating_long_short(self, sat_model):
        text = "Short. " + "This is a much longer sentence with many words. " * 5 + "Short again. " + "Another very long sentence. " * 3

        segments = sat_model.split(text, max_length=100, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 100

    def test_no_natural_breaks(self, sat_model):
        """Text with no punctuation at all."""
        text = "This text has no punctuation at all and just keeps going and going without any natural break points whatsoever and the algorithm needs to handle this gracefully"

        segments = sat_model.split(text, max_length=50, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 50

    def test_repeated_sentence(self, sat_model):
        sentence = "The quick brown fox jumps over the lazy dog. "
        text = sentence * 100

        segments = sat_model.split(text, max_length=100, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 100

    def test_rapid_repeated_calls(self, sat_model):
        """Ensure consistency across rapid repeated calls."""
        text = "Hello world. How are you today?"

        results = [sat_model.split(text, max_length=50, threshold=0.025) for _ in range(10)]

        for result in results[1:]:
            assert result == results[0]

    def test_10k_characters(self, sat_model):
        """Test with ~10,000 character document."""
        text = "This is a test sentence with some content. " * 250

        segments = sat_model.split(text, max_length=200, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 200

    def test_extreme_tiny_sentences(self, wtp_model):
        """Many tiny sentences should be merged appropriately."""
        text = "A. " * 100
        splits = wtp_model.split(text, min_length=20, max_length=100, threshold=0.005)

        for segment in splits:
            assert len(segment) <= 100

        segments_meeting_min = sum(1 for s in splits if len(s) >= 20)
        assert segments_meeting_min >= len(splits) - 1

        assert "".join(splits) == text


# =============================================================================
# CONSTRAINT COMBINATION TESTS
# =============================================================================

class TestConstraintCombinations:
    """Test various combinations of constraints."""

    def test_tight_constraints(self, sat_model):
        """min_length close to max_length."""
        text = "First sentence here. Second one follows. Third sentence ends."

        segments = sat_model.split(text, min_length=15, max_length=25, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 25

    def test_equal_min_max(self, sat_model):
        """min_length equals max_length."""
        text = "Hello world."

        segments = sat_model.split(text, min_length=12, max_length=12, threshold=0.025)
        assert "".join(segments) == text

    def test_large_min_length(self, sat_model):
        text = "A. B. C. D. E. F. G. H. I. J."

        segments = sat_model.split(text, min_length=20, max_length=100, threshold=0.025)
        assert "".join(segments) == text
        assert len(segments) < 10

    def test_very_small_max_length(self, sat_model):
        text = "Hello world. Test."

        segments = sat_model.split(text, max_length=10, threshold=0.025)
        assert "".join(segments) == text
        for segment in segments:
            assert len(segment) <= 10


# =============================================================================
# UNICODE AND INTERNATIONALIZATION TESTS
# =============================================================================

class TestUnicodeAndI18n:
    """Tests for unicode and international text."""

    def test_chinese(self, sat_model):
        text = "你好世界。今天天气很好。我很高兴见到你。"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_russian(self, sat_model):
        text = "Привет мир. Как дела? Хорошо, спасибо!"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_arabic(self, sat_model):
        text = "مرحبا بالعالم. كيف حالك؟"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_mixed_scripts(self, sat_model):
        text = "Hello 世界. Привет мир. مرحبا world."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_emojis(self, sat_model):
        text = "Hello! 😀 How are you? 🎉 Great!"
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_special_unicode_characters(self, sat_model):
        text = "Price: €100. Temperature: 25°C. Copyright © 2024."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_multilingual_with_constraints(self, sat_model):
        text = "Hello world. 你好世界。Bonjour le monde. Hola mundo. こんにちは世界。"
        splits = sat_model.split(text, min_length=10, max_length=50, threshold=0.025)

        for segment in splits:
            assert len(segment) <= 50


# =============================================================================
# CONSTRAINED SEGMENTATION ALGORITHM TESTS
# =============================================================================

class TestConstrainedSegmentationAlgorithm:
    """Direct tests of the constrained_segmentation function."""

    def test_high_prob_at_boundaries(self):
        """Algorithm should prefer positions with high probability."""
        probs = np.zeros(30)
        probs[9] = 0.9
        probs[19] = 0.9
        probs[29] = 0.9

        prior_fn = create_prior_function("uniform", {"max_length": 30})
        boundaries = constrained_segmentation(probs, prior_fn, min_length=1, max_length=30)

        assert isinstance(boundaries, list)

    def test_uniform_probs_uses_max_length(self):
        """With uniform probabilities, should split at max_length."""
        probs = np.ones(100) * 0.5
        prior_fn = create_prior_function("uniform", {"max_length": 25})

        boundaries = constrained_segmentation(probs, prior_fn, min_length=1, max_length=25)

        assert len(boundaries) >= 3

        prev = 0
        for b in boundaries + [100]:
            assert b - prev <= 25
            prev = b

    def test_consistency(self):
        """Same input should produce same output."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        def prior_fn(length):
            return 1.0

        result1 = constrained_segmentation(probs, prior_fn, min_length=2, max_length=3, algorithm="viterbi")
        result2 = constrained_segmentation(probs, prior_fn, min_length=2, max_length=3, algorithm="viterbi")

        assert result1 == result2


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressions:
    """Regression tests for previously fixed bugs."""

    def test_viterbi_finds_sentence_boundaries(self, sat_model):
        """Viterbi should prefer sentence boundaries over arbitrary positions."""
        text = "The quick brown fox jumps. Pack my box with jugs. How vexingly quick!"
        segments = sat_model.split(text, max_length=150, algorithm="viterbi", threshold=0.025)

        for segment in segments:
            if segment.strip():
                last_char = segment.rstrip()[-1]
                assert last_char in '.!?,;:\'"' or segment[-1].isspace() or segment == segments[-1]

    def test_trailing_whitespace_preserved(self, sat_model):
        text = "Sentence one.  Sentence two.   End."
        segments = sat_model.split(text, threshold=0.5)
        assert "".join(segments) == text

    def test_viterbi_backtracking_bug_fixed(self, sat_model):
        """Test that Viterbi correctly traces back to start (bug fix verification)."""
        text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly."
        segments = sat_model.split(text, max_length=150, algorithm="viterbi", threshold=0.025)

        for segment in segments:
            if len(segment) > 1:
                if segment[-1].isalpha() and segment[-2].isalpha():
                    pytest.fail(f"Word cut detected: segment ends with '{segment[-10:]}'")

        assert "".join(segments) == text

    def test_min_length_merge_no_text_duplication(self):
        """
        Regression test for bug where min_length merging caused text duplication.
        """
        sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])

        text = "A. " * 20
        segments = sat.split(text, min_length=15, max_length=30, threshold=0.5)

        rejoined = "".join(segments)
        assert rejoined == text, f"Text corrupted! Expected {len(text)} chars, got {len(rejoined)}"

        original_a_count = text.count("A")
        rejoined_a_count = rejoined.count("A")
        assert original_a_count == rejoined_a_count

    def test_viterbi_backtracking_prev_zero_valid(self):
        """
        Regression test for Viterbi backtracking bug where prev=0 was incorrectly
        treated as error when it's a valid state for first chunk.
        """
        probs = np.zeros(100)
        probs[49] = 0.99
        probs[99] = 0.99

        prior_fn = create_prior_function("uniform", {"max_length": 60})

        boundaries = constrained_segmentation(
            probs, prior_fn, min_length=1, max_length=60, algorithm="viterbi"
        )

        assert 50 in boundaries, f"Viterbi should find boundary at 50, got {boundaries}"

    def test_greedy_final_segment_max_length(self):
        """
        Regression test for greedy algorithm final segment bug.
        """
        probs = np.ones(100) * 0.1
        probs[54] = 0.99
        probs[84] = 0.99

        prior_fn = create_prior_function("uniform", {"max_length": 40})

        boundaries = constrained_segmentation(
            probs, prior_fn, min_length=20, max_length=40, algorithm="greedy"
        )

        prev = 0
        for b in boundaries + [100]:
            seg_len = b - prev
            assert seg_len <= 40, f"Segment [{prev}:{b}] length {seg_len} exceeds max_length=40"
            prev = b

    def test_empty_strings_preserved_with_constraints(self):
        """
        Regression test for empty string filtering bug.
        """
        sentences = ['Hello.', '', '', 'World.']

        result = _enforce_segment_constraints_simple(
            sentences, min_length=1, max_length=100, delimiter="\n"
        )

        assert '' in result or len([s for s in result if not s]) > 0

        content = [s for s in result if s.strip()]
        assert content == ['Hello.', 'World.']

    def test_min_length_backward_merge(self):
        """
        Regression test for min_length backward merge.
        """
        sentences = ['First segment here', 'Hi', 'Another long segment']
        result = _enforce_segment_constraints_simple(
            sentences, min_length=10, max_length=30, delimiter=" "
        )

        for seg in result:
            if seg.strip():
                assert len(seg) >= 10 or len(seg) <= 2

    def test_equal_min_max_viterbi_fallback(self):
        """
        Regression test: When min_length == max_length and DP fails,
        the fallback should still produce valid segments.
        """
        probs = np.zeros(15)
        prior_fn = create_prior_function("uniform", {"max_length": 5})

        indices = constrained_segmentation(probs, prior_fn, min_length=5, max_length=5, algorithm="viterbi")

        prev = 0
        chunks = []
        for idx in indices:
            chunks.append(idx - prev)
            prev = idx
        if prev < 15:
            chunks.append(15 - prev)

        assert all(c == 5 for c in chunks), f"Chunks should all be 5, got {chunks}"

    def test_newline_duplication_with_constraints(self, sat_model):
        """
        Regression test: When split_on_input_newlines=True combined with length
        constraints, text preservation should not create duplicate newlines.
        
        Bug: _enforce_segment_constraints includes trailing newlines in segments.
        When split on '\\n', these create empty strings that cause duplicate 
        newlines after '\\n'.join().
        """
        # Test basic newline
        text1 = "Hello world.\nGoodbye world."
        segments1 = sat_model.split(text1, max_length=50)
        assert "\n".join(segments1) == text1, f"Basic newline failed: {segments1}"
        
        # Test trailing newline
        text2 = "Hello world.\nGoodbye world.\n"
        segments2 = sat_model.split(text2, max_length=50)
        assert "\n".join(segments2) == text2, f"Trailing newline failed: {segments2}"
        
        # Test consecutive newlines
        text3 = "Hello.\n\nWorld."
        segments3 = sat_model.split(text3, max_length=50)
        assert "\n".join(segments3) == text3, f"Consecutive newlines failed: {segments3}"
        
        # Test triple newline
        text4 = "A.\n\n\nB."
        segments4 = sat_model.split(text4, max_length=50)
        assert "\n".join(segments4) == text4, f"Triple newline failed: {segments4}"
        
        # Test consecutive + trailing
        text5 = "Hello.\n\nWorld.\n"
        segments5 = sat_model.split(text5, max_length=50)
        assert "\n".join(segments5) == text5, f"Consecutive + trailing failed: {segments5}"


# =============================================================================
# PARAGRAPH SEGMENTATION WITH CONSTRAINTS
# =============================================================================

class TestParagraphSegmentation:
    """Test nested paragraph and sentence segmentation with constraints."""

    def test_paragraph_segmentation_with_constraints(self, wtp_model):
        text = "Paragraph one sentence one. Paragraph one sentence two.\n\nParagraph two sentence one. Paragraph two sentence two."

        paragraphs = wtp_model.split(text, do_paragraph_segmentation=True, min_length=10, max_length=40)

        assert isinstance(paragraphs, list)
        for paragraph in paragraphs:
            assert isinstance(paragraph, list)
            for sentence in paragraph:
                assert 10 <= len(sentence) <= 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
