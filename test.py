# noqa: E501
from wtpsplit import WtP


def test_split_ort():
    wtp = WtP("wtp-bert-mini", ort_providers=["CPUExecutionProvider"])

    splits = wtp.split("This is a test sentence This is another test sentence.")
    assert splits == ["This is a test sentence ", "This is another test sentence."]

def test_split_torch():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = wtp.split("This is a test sentence This is another test sentence.")
    assert splits == ["This is a test sentence ", "This is another test sentence."]

def test_move_device():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)
    wtp.half().to("cpu")

def test_strip_whitespace():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = wtp.split("This is a test sentence This is another test sentence.   ", strip_whitespace=True)
    assert splits == ["This is a test sentence", "This is another test sentence."]

def test_split_long():
    prefix = "x" * 2000

    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = wtp.split(prefix + " This is a test sentence. This is another test sentence.")
    assert splits == [prefix + " ", "This is a test sentence. ", "This is another test sentence."]


def test_split_batched():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    splits = list(wtp.split(["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]))

    assert splits == [
        ["Paragraph-A ", "Paragraph-B"],
        ["Paragraph-C100 ", "Paragraph-D"],
    ]


def test_split_style():
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None)

    text = "’I couldn’t help it,’ said Five, in a sulky tone; ’Seven jogged my elbow.’ | On which Seven looked up and said, ’That’s right, Five! Always lay the blame (...)!’"

    splits_ud = wtp.split(text, lang_code="en", style="ud")
    splits_opus100 = wtp.split(text, lang_code="en", style="opus100")
    splits_ersatz = wtp.split(text, lang_code="en", style="ersatz")

    assert splits_ud != splits_opus100 != splits_ersatz


def test_split_paragraphs():
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


def test_split_paragraphs_with_language_adapters():
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


def test_split_threshold():
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

    splits = wtp.split("This is a test sentence. This is another test sentence.", threshold=0.0)
    # space might still be included in a character split
    assert splits[:3] == list("Thi")
