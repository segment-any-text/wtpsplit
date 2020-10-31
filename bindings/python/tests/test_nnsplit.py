import nnsplit
from pathlib import Path

MODEL_PATH = (Path(__file__) / ".." / ".." / ".." / ".." / "models").resolve()


def test_splitter_model_works():
    model = nnsplit.NNSplit(MODEL_PATH / "de" / "model.onnx")
    splits = model.split(["Das ist ein Test Das ist noch ein Test."])[0]

    assert [str(x) for x in splits] == ["Das ist ein Test ", "Das ist noch ein Test."]


def test_splitter_model_works_with_args():
    model = nnsplit.NNSplit(MODEL_PATH / "de" / "model.onnx", threshold=1.0)
    splits = model.split(["Das ist ein Test Das ist noch ein Test."])[0]

    assert [str(x) for x in splits] == ["Das ist ein Test Das ist noch ein Test."]


def test_getting_levels_works():
    model = nnsplit.NNSplit(MODEL_PATH / "de" / "model.onnx")

    assert model.get_levels() == [
        "Sentence",
        "Token",
        "_Whitespace",
        "Compound constituent",
    ]
