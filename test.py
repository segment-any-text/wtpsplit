# noqa: E501
from wtpsplit import WtP, SaT


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