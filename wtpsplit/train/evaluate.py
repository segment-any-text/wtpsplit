import numpy as np
import pysbd
import sklearn.metrics

from wtpsplit.extract import extract, PyTorchWrapper
from wtpsplit.utils import Constants


def compute_iou(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def compute_f1(pred, true):
    pred = set(pred)
    true = set(true)

    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)

    return (
        f1,
        precision,
    )


def get_metrics(labels, preds):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, preds)

    # we can use raw logits (no sigmoid) since we only care about the ordering here
    pr_auc = sklearn.metrics.auc(recall, precision)

    metrics = {
        "pr_auc": pr_auc,
    }
    info = {
        "probs": preds,
        "labels": labels,
        "recalls": recall,
        "precisions": precision,
        "thresholds": thresholds,
    }

    return metrics, info


def get_token_spans(tokenizer: object, offsets_mapping: list, tokens: list):
    token_spans = []
    for idx, token in enumerate(tokens):
        # Skip special tokens like [CLS], [SEP]
        if idx >= len(offsets_mapping):
            continue
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        char_start, char_end = offsets_mapping[idx]
        token_spans.append((char_start, char_end))

    return token_spans


def token_to_char_probs(text: str, tokens: list, token_probs: np.ndarray, tokenizer, offsets_mapping):
    # some very low number since at non-ending position, predicting a newline is impossible
    char_probs = np.zeros(len(text)) - 10000
    token_spans = get_token_spans(tokenizer, offsets_mapping, tokens)

    for i, ((start, end), prob, token) in enumerate(zip(token_spans, token_probs, tokens)):
        # assign the token's prob to the last char of the token
        char_probs[end - 1] = prob

    return char_probs


def evaluate_sentence(
    lang_code,
    sentences,
    model,
    stride,
    block_size,
    batch_size,
    use_pysbd=False,
    positive_index=None,
):
    if positive_index is None:
        positive_index = Constants.NEWLINE_INDEX

    # preprocessing, many OPUS100 (and some UD) sentences start with "- "
    sentences = [sentence.lstrip("-").strip() for sentence in sentences]

    separator = Constants.SEPARATORS[lang_code]
    text = separator.join(sentences)

    logits, offsets_mapping, tokenizer = extract(
        [text],
        PyTorchWrapper(model.backbone),
        lang_code=lang_code,
        stride=stride,
        block_size=block_size,
        batch_size=batch_size,
        verbose=True,
    )
    logits = logits[0]
    if offsets_mapping is not None:
        offsets_mapping = offsets_mapping[0]

    true_end_indices = np.cumsum(np.array([len(s) for s in sentences])) + np.arange(len(sentences)) * len(separator)
    newline_labels = np.zeros(len(text))
    newline_labels[true_end_indices - 1] = 1

    if "xlm" in model.config.model_type:
        tokens = tokenizer.tokenize(text, verbose=False)
        char_probs = token_to_char_probs(text, tokens, logits[:, positive_index], tokenizer, offsets_mapping)
    else:
        char_probs = logits[:, positive_index]
    metrics, info = get_metrics(newline_labels, char_probs)

    info["newline_labels"] = newline_labels

    if use_pysbd:
        segmenter = pysbd.Segmenter(language=lang_code, clean=False, char_span=True)
        predicted_end_indices_pysbd = np.array([x.start + len(x.sent.rstrip()) for x in segmenter.segment(text)])
        newline_probs_pysbd = np.zeros(len(text))
        newline_probs_pysbd[predicted_end_indices_pysbd - 1] = 1.0

        info["newline_probs_pysbd"] = newline_probs_pysbd

    return metrics["pr_auc"], info
