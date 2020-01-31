import nnsplit
from pathlib import Path
import numpy as np
import torch
from nnsplit import train, tokenizer, utils, models


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

    utils.store_model(learner, "model")
    model = utils.load_model("model", torch.device("cpu"))

    model(torch.zeros([10, nnsplit.defaults.CUT_LENGTH]))


def test_keras_and_pytorch_same():
    model = models.Network()
    keras_model = model.get_keras_equivalent()
    inp = np.random.randint(0, 127 + 2, [1, 100])

    torch_output = model(torch.from_numpy(inp)).detach().cpu().detach().numpy()
    keras_output = keras_model.predict(inp)

    assert np.allclose(torch_output, keras_output, rtol=0.0, atol=1e-5)


def test_split_german():
    samples = [
        [
            "Das ist ein Test. Das ist noch ein Test.",
            ["Das ist ein Test .", "Das ist noch ein Test ."],
        ],
        [
            "Das ist ein Test Das ist auch ein Test.",
            ["Das ist ein Test", "Das ist auch ein Test ."],
        ],
    ]  # whitespaces in the expected string denote token splits
    splitter = nnsplit.NNSplit("de")

    for inp, out in samples:
        result = splitter.split([inp])[0]

        for i, sentence in enumerate(result):
            assert " ".join(x.text for x in sentence) == out[i]
