# noqa: E501
from wtpsplit import WtP, SaT
import numpy as np


def test_weighting():
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])

    text = "This is a test sentence This is another test sentence."
    splits_default = sat.split(text, threshold=0.25)
    splits_uniform = sat.split(text, threshold=0.25, weighting="uniform")
    splits_hat = sat.split(text, threshold=0.25, weighting="hat")
    expected_splits = ["This is a test sentence ", "This is another test sentence."] 
    assert splits_default == splits_uniform == splits_hat == expected_splits
    assert "".join(splits_default) == text


def test_split_ort():
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])

    splits = sat.split("This is a test sentence This is another test sentence.", threshold=0.25)
    assert splits == ["This is a test sentence ", "This is another test sentence."]


def test_split_torch():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

    splits = sat.split("This is a test sentence This is another test sentence.", threshold=0.025)
    assert splits == ["This is a test sentence ", "This is another test sentence."]


def test_split_torch_sm():
    sat = SaT("segment-any-text/sat-12l-sm", hub_prefix=None)

    splits = sat.split("This is a test sentence. This is another test sentence.", threshold=0.25)
    assert splits == ["This is a test sentence. ", "This is another test sentence."]


def test_move_device():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)
    sat.half().to("cpu")


def test_strip_whitespace():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

    splits = sat.split(
        "This is a test sentence This is another test sentence.   ", strip_whitespace=True, threshold=0.025
    )
    assert splits == ["This is a test sentence", "This is another test sentence."]


def test_strip_newline_behaviour():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

    splits = sat.split(
        "Yes\nthis is a test sentence. This is another test sentence.",
    )
    assert splits == ["Yes", "this is a test sentence. ", "This is another test sentence."]
    
def test_strip_newline_behaviour_as_spaces():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

    splits = sat.split(
        "Yes\nthis is a test sentence. This is another test sentence.", treat_newline_as_space=True
    )
    assert splits == ["Yes\nthis is a test sentence. ", "This is another test sentence."]


def test_split_noisy():
    sat = SaT("segment-any-text/sat-12l-sm", hub_prefix=None)

    splits = sat.split("this is a sentence :) this is another sentence lol")
    assert splits == ["this is a sentence :) ", "this is another sentence lol"]


def test_split_batched():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

    splits = list(sat.split(["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]))

    assert splits == [
        ["Paragraph-A ", "Paragraph-B"],
        ["Paragraph-C100 ", "Paragraph-D"],
    ]


def test_split_lora():
    ud = SaT("segment-any-text/sat-3l", hub_prefix=None, style_or_domain="ud", language="en")
    opus = SaT("segment-any-text/sat-3l", hub_prefix=None, style_or_domain="opus100", language="en")
    ersatz = SaT("segment-any-text/sat-3l", hub_prefix=None, style_or_domain="ersatz", language="en")

    text = "’I couldn’t help it,’ said Five, in a sulky tone; ’Seven jogged my elbow.’ | On which Seven looked up and said, ’That’s right, Five! Always lay the blame (...)!’"

    splits_ud = ud.split(text)
    splits_opus100 = opus.split(text)
    splits_ersatz = ersatz.split(text)

    assert splits_ud != splits_opus100 != splits_ersatz


def test_split_paragraphs():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

    text = " ".join(
        """
Text segmentation is the process of dividing written text into meaningful units, such as words, sentences, or topics. The term applies both to mental processes used by humans when reading text, and to artificial processes implemented in computers, which are the subject of natural language processing. The problem is non-trivial, because while some written languages have explicit word boundary markers, such as the word spaces of written English and the distinctive initial, medial and final letter shapes of Arabic, such signals are sometimes ambiguous and not present in all written languages.
Daniel Wroughton Craig CMG (born 2 March 1968) is an English actor who gained international fame by playing the fictional secret agent James Bond for five installments in the film series, from Casino Royale (2006) up to No Time to Die (2021).
""".strip().split()
    )

    splits = sat.split(text, do_paragraph_segmentation=True)

    paragraph1 = "".join(splits[0])
    paragraph2 = "".join(splits[1])

    assert paragraph1.startswith("Text segmentation is")
    assert paragraph2.startswith("Daniel Wroughton Craig CMG (born 2 March 1968) is")
    
def test_split_empty_strings():
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)
    
    text = "   "
    splits = sat.split(text)
    assert splits == ["   "]
    text = "   \n"
    splits = sat.split(text)
    assert splits == ["   ", ""]
    text = ""
    splits = sat.split(text)
    assert splits == []


def test_split_ort_wtp():
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])

    splits = wtp.split("This is a test sentence This is another test sentence.", threshold=0.005)
    assert splits == ["This is a test sentence ", "This is another test sentence."]


def test_split_torch_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = wtp.split("This is a test sentence This is another test sentence.", threshold=0.005)
    assert splits == ["This is a test sentence ", "This is another test sentence."]


def test_split_torch_canine_wtp():
    wtp = WtP("benjamin/wtp-canine-s-1l", hub_prefix=None)

    splits = wtp.split("This is a test sentence. This is another test sentence.", lang_code="en")
    assert splits == ["This is a test sentence. ", "This is another test sentence."]


def test_move_device_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)
    wtp.half().to("cpu")


def test_strip_whitespace_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = wtp.split(
        "This is a test sentence This is another test sentence.   ", strip_whitespace=True, threshold=0.005
    )
    assert splits == ["This is a test sentence", "This is another test sentence."]


def test_split_long_wtp():
    prefix = "x" * 2000

    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = wtp.split(prefix + " This is a test sentence. This is another test sentence.")
    assert splits == [prefix + " " + "This is a test sentence. ", "This is another test sentence."]


def test_split_batched_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = list(wtp.split(["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]))

    assert splits == [
        ["Paragraph-A ", "Paragraph-B"],
        ["Paragraph-C100 ", "Paragraph-D"],
    ]


def test_split_style_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    text = "’I couldn’t help it,’ said Five, in a sulky tone; ’Seven jogged my elbow.’ | On which Seven looked up and said, ’That’s right, Five! Always lay the blame (...)!’"

    splits_ud = wtp.split(text, lang_code="en", style="ud")
    splits_opus100 = wtp.split(text, lang_code="en", style="opus100")
    splits_ersatz = wtp.split(text, lang_code="en", style="ersatz")

    assert splits_ud != splits_opus100 != splits_ersatz


def test_split_paragraphs_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    text = " ".join(
        """
Text segmentation is the process of dividing written text into meaningful units, such as words, sentences, or topics. The term applies both to mental processes used by humans when reading text, and to artificial processes implemented in computers, which are the subject of natural language processing. The problem is non-trivial, because while some written languages have explicit word boundary markers, such as the word spaces of written English and the distinctive initial, medial and final letter shapes of Arabic, such signals are sometimes ambiguous and not present in all written languages.
Daniel Wroughton Craig CMG (born 2 March 1968) is an English actor who gained international fame by playing the fictional secret agent James Bond for five installments in the film series, from Casino Royale (2006) up to No Time to Die (2021).
""".strip().split()
    )

    splits = wtp.split(text, do_paragraph_segmentation=True)

    paragraph1 = "".join(splits[0])
    paragraph2 = "".join(splits[1])

    assert paragraph1.startswith("Text segmentation is")
    assert paragraph2.startswith("Daniel Wroughton Craig CMG (born 2 March 1968) is")


def test_split_paragraphs_with_language_adapters_wtp():
    wtp = WtP("benjamin/wtp-canine-s-3l", hub_prefix=None)

    text = " ".join(
        """
Text segmentation is the process of dividing written text into meaningful units, such as words, sentences, or topics. The term applies both to mental processes used by humans when reading text, and to artificial processes implemented in computers, which are the subject of natural language processing. The problem is non-trivial, because while some written languages have explicit word boundary markers, such as the word spaces of written English and the distinctive initial, medial and final letter shapes of Arabic, such signals are sometimes ambiguous and not present in all written languages.
Daniel Wroughton Craig CMG (born 2 March 1968) is an English actor who gained international fame by playing the fictional secret agent James Bond for five installments in the film series, from Casino Royale (2006) up to No Time to Die (2021).
""".strip().split()
    )

    splits = wtp.split(text, do_paragraph_segmentation=True, lang_code="en")

    paragraph1 = "".join(splits[0])
    paragraph2 = "".join(splits[1])

    assert paragraph1.startswith("Text segmentation is")
    assert paragraph2.startswith("Daniel Wroughton Craig CMG (born 2 March 1968) is")


def test_split_threshold_wtp():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    punct_threshold = wtp.get_threshold("en", "ud", return_punctuation_threshold=True)
    threshold = wtp.get_threshold("en", "ud")

    assert punct_threshold != threshold

    # test threshold is being used
    splits = wtp.split("This is a test sentence. This is another test sentence.", threshold=1.0)
    assert splits == ["This is a test sentence. This is another test sentence."]

    splits = wtp.split(
        "This is a test sentence. This is another test sentence.", style="ud", lang_code="en", threshold=1.0
    )
    assert splits == ["This is a test sentence. This is another test sentence."]

    splits = wtp.split("This is a test sentence. This is another test sentence.", threshold=-1e-3)
    # space might still be included in a character split
    assert splits[:3] == list("Thi")

# ============================================================================
# Length-Constrained Segmentation Tests
# ============================================================================

def test_min_length_constraint_wtp():
    """Test minimum length constraint with WtP"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Short. Test. Hello. World. This is longer."
    splits = wtp.split(text, min_length=15, threshold=0.005)
    
    # All segments should be >= 15 characters
    for segment in splits:
        assert len(segment) >= 15, f"Segment '{segment}' is shorter than min_length"
    
    # Text should be preserved
    assert "".join(splits) == text


def test_max_length_constraint_sat():
    """Test maximum length constraint with SaT"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "This is a test sentence. " * 10
    splits = sat.split(text, max_length=60, threshold=0.025)
    
    # All segments should be <= 60 characters
    for segment in splits:
        assert len(segment) <= 60, f"Segment '{segment}' is longer than max_length"
    
    # Text should be preserved  
    assert "".join(splits) == text


def test_min_max_constraints_together():
    """Test both constraints simultaneously"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Hello world. " * 15
    splits = wtp.split(text, min_length=25, max_length=65, threshold=0.005)
    
    # All segments should satisfy both constraints
    for segment in splits:
        assert 25 <= len(segment) <= 65, f"Segment '{segment}' violates constraints"
    
    # Text should be preserved
    assert "".join(splits) == text


def test_gaussian_prior():
    """Test Gaussian prior preference"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    text = "Sentence. " * 30
    splits = sat.split(
        text,
        min_length=20,
        max_length=60,
        prior_type="gaussian",
        prior_kwargs={"mu": 40.0, "sigma": 5.0},
        threshold=0.025
    )
    
    # Should produce valid splits
    for segment in splits:
        assert 20 <= len(segment) <= 60
    
    # Text should be preserved
    assert "".join(splits) == text


def test_greedy_algorithm():
    """Test greedy algorithm"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = "Test sentence. " * 10
    splits = wtp.split(text, min_length=20, max_length=50, algorithm="greedy", threshold=0.005)
    
    # Should produce valid splits
    for segment in splits:
        assert 20 <= len(segment) <= 50
    
    # Text should be preserved
    assert "".join(splits) == text


def test_constraints_with_paragraph_segmentation():
    """Test constraints with nested paragraph segmentation"""
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])
    
    text = " ".join([
        "First paragraph first sentence. First paragraph second sentence.",
        "Second paragraph first sentence. Second paragraph second sentence."
    ])
    
    paragraphs = wtp.split(text, do_paragraph_segmentation=True, min_length=20, max_length=70)
    
    # Check structure
    assert isinstance(paragraphs, list)
    for paragraph in paragraphs:
        assert isinstance(paragraph, list)
        for sentence in paragraph:
            assert 20 <= len(sentence) <= 70


def test_constraints_preserved_in_batched():
    """Test constraints work with batched processing"""
    sat = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    
    texts = [
        "First batch text. " * 5,
        "Second batch text. " * 5,
    ]
    
    results = list(sat.split(texts, min_length=25, max_length=55, threshold=0.025))
    
    assert len(results) == 2
    for splits in results:
        for segment in splits:
            assert 25 <= len(segment) <= 55


def test_constraint_low_level():
    """Test constrained_segmentation directly"""
    from wtpsplit.utils.constraints import constrained_segmentation
    from wtpsplit.utils.priors import create_prior_function
    
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior_fn = create_prior_function("uniform", {"max_length": 5})
    
    indices = constrained_segmentation(probs, prior_fn, min_length=3, max_length=5, algorithm="viterbi")
    
    # Verify constraints on chunk lengths
    prev = 0
    for idx in indices:
        chunk_len = idx - prev
        assert 3 <= chunk_len <= 5, f"Chunk length {chunk_len} violates constraints"
        prev = idx
    
    # Check last chunk
    if prev < len(probs):
        last_len = len(probs) - prev
        assert 3 <= last_len <= 5, f"Last chunk length {last_len} violates constraints"
