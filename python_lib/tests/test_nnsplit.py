import nnsplit
from pathlib import Path
import torch
from nnsplit import train, tokenizer, utils


def test_external_tokenizer():
    tok = tokenizer.SoMaJoTokenizer("de")

    raw_text = "Das ist ein Test. Das ist noch ein Test."
    tokenized_sentences = [
        "Das ist ein Test .",
        "Das ist noch ein Test .",
    ]  # note that the dot is a separate token

    splitted = tok.split([raw_text])[0]
    for i, sentence in enumerate(splitted):
        assert " ".join(x.text for x in sentence) == tokenized_sentences[i]


def test_prepare_data():
    sample_path = (
        Path(__file__) / ".." / ".." / ".." / "data" / "sample-monolingual.xml"
    ).resolve()

    max_n_sentences = 10
    x, y = train.prepare_data(sample_path, "en", max_n_sentences=max_n_sentences)
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert len(x) == len(y) and len(x) <= max_n_sentences


def test_train_model():
    x = torch.zeros([10, nnsplit.defaults.CUT_LENGTH])
    y = torch.zeros([10, nnsplit.CUT_LENGTH, 2])
    _ = train.train_from_tensors(x, y, n_epochs=2)


def test_load_model():
    x = torch.zeros([10, nnsplit.defaults.CUT_LENGTH])
    y = torch.zeros([10, nnsplit.CUT_LENGTH, 2])
    learner = train.train_from_tensors(x, y, n_epochs=2)

    utils.store_model(learner, "model.pt")
    model = utils.load_model("model.pt")

    model(torch.zeros([10, nnsplit.defaults.CUT_LENGTH]))


# def test_split_german():
#     samples = [
#         [
#             "Das ist ein Test. Das ist noch ein Test.",
#             ["Das ist ein Test .", "Das ist noch ein Test ."],
#         ],
#         [
#             "Das ist ein Test Das ist auch ein Test.",
#             ["Das ist ein Test", "Das ist auch ein Test ."],
#         ],
#     ]
#     splitter = nnsplit.NNSplit("de")

#     for inp, out in samples:
#         result = splitter.split([inp])

#         for sentence in result:
#             print(sentence)
