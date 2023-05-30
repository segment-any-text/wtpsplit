from dataclasses import dataclass

import numpy as np
from sklearn import linear_model
from sklearn.metrics import f1_score
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit.models  # noqa
from wtpsplit.extract import extract
from wtpsplit.utils import ROOT_DIR, encode


@dataclass
class Args:
    model_path: str
    lang: str
    block_size: int = 512
    stride: int = 64
    batch_size: int = 32
    n_subsample: int = None
    device: str = "cuda"


def load_iwslt(path, fix_space=True):
    char_labels = []

    label_dict = {
        "O": 0,
        "QUESTION": 1,
        "PERIOD": 2,
        "COMMA": 3,
    }

    text = ""

    for i, row in enumerate(open(path)):
        token, label = row.rstrip().split("\t")

        if not fix_space or (i > 0 and token not in {"'m", "n't", "'ll", "'s"}):
            text += " "
            char_labels.append(0)

        text += token
        char_labels.extend([0] * len(token))
        char_labels[-1] = label_dict[label]

    return text, np.array(char_labels)


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    if args.lang == "en":
        train_text, train_char_labels = load_iwslt(
            ROOT_DIR / "data" / "external" / "punctuation_annotation" / "en" / "train2012",
        )
        test_text, test_char_labels = load_iwslt(
            ROOT_DIR / "data" / "external" / "punctuation_annotation" / "en" / "test2011",
        )
    else:
        train_text, train_char_labels = load_iwslt(
            ROOT_DIR / "data" / "external" / "punctuation_annotation" / "bn" / "train",
        )
        test_text, test_char_labels = load_iwslt(
            ROOT_DIR / "data" / "external" / "punctuation_annotation" / "bn" / "test_ref",
        )

    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(args.device)

    if args.n_subsample is None:
        args.n_subsample = len(train_text)

    train_logits = extract(
        [encode(train_text[: args.n_subsample])],
        model,
        lang_code=args.lang,
        stride=args.stride,
        block_size=args.block_size,
        batch_size=args.batch_size,
        use_hidden_states=False,
        pad_last_batch=False,
    )[0].numpy()

    clf = linear_model.LogisticRegression(penalty="none", multi_class="multinomial", max_iter=10_000)
    clf.fit(train_logits, train_char_labels[: args.n_subsample])

    test_logits = extract(
        [encode(test_text)],
        model,
        lang_code=args.lang,
        stride=args.stride,
        block_size=args.block_size,
        batch_size=args.batch_size,
        use_hidden_states=False,
        pad_last_batch=False,
    )[0].numpy()

    test_preds = clf.predict(test_logits)

    question, period, comma = (
        f1_score(test_char_labels == 1, test_preds == 1),
        f1_score(test_char_labels == 2, test_preds == 2),
        f1_score(test_char_labels == 3, test_preds == 3),
    )
    avg = np.mean([question, period, comma])
    print(question, period, comma, avg)
