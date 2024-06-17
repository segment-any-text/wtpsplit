# noqa: E501
from wtpsplit import SaT


# def test_split_ort():
#     sat = SaT("segment-any-text/sat-3l", ort_providers=["CPUExecutionProvider"])

#     splits = sat.split("This is a test sentence This is another test sentence.", threshold=0.005)
#     assert splits == ["This is a test sentence ", "This is another test sentence."]


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
