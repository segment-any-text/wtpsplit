import nnsplit

def test_splitter_model_works():
    model = nnsplit.NNSplit.load("de")
    splits = model.split(["Das ist ein Test Das ist noch ein Test."])[0]

    assert [str(x) for x in splits] == ["Das ist ein Test ", "Das ist noch ein Test."]

def test_splitter_model_works_with_args():
    model = nnsplit.NNSplit.load("de", threshold=1.0)
    splits = model.split(["Das ist ein Test Das ist noch ein Test."])[0]

    assert [str(x) for x in splits] == ["Das ist ein Test Das ist noch ein Test."]
