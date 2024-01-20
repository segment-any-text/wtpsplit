import numpy as np
import pysbd
import sklearn.metrics
import logging

from wtpsplit.extract import extract, PyTorchWrapper
from wtpsplit.utils import Constants
from wtpsplit.evaluation import token_to_char_probs
import random

logger = logging.getLogger(__name__)


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


def evaluate_sentence(
    lang_code,
    sentences,
    model,
    stride,
    block_size,
    batch_size,
    use_pysbd=False,
    positive_index=None,
    do_lowercase=False,
    do_remove_punct=False,
):
    if positive_index is None:
        positive_index = Constants.NEWLINE_INDEX

    # preprocessing, many OPUS100 (and some UD) sentences start with "- "
    sentences = [sentence.lstrip("-").strip() for sentence in sentences]

    separator = Constants.SEPARATORS[lang_code]
    if do_lowercase:
        sentences = [sentence.lower() for sentence in sentences]
    if do_remove_punct:
        for punct in Constants.PUNCTUATION_CHARS:
            sentences = [sentence.replace(punct, "") for sentence in sentences]
    text = separator.join(sentences)

    logits, offsets_mapping, tokenizer = extract(
        [text],
        PyTorchWrapper(model.backbone),
        lang_code=lang_code,
        stride=stride,
        block_size=block_size,
        batch_size=batch_size,
    )
    logits = logits[0]
    if offsets_mapping is not None:
        offsets_mapping = offsets_mapping[0]

    true_end_indices = np.cumsum(np.array([len(s) for s in sentences])) + np.arange(len(sentences)) * len(separator)
    newline_labels = np.zeros(len(text))
    newline_labels[true_end_indices - 1] = 1

    if "xlm" in model.config.model_type:
        tokens = tokenizer.tokenize(text, verbose=False)
        char_probs = token_to_char_probs(text, tokens, logits, tokenizer, offsets_mapping)
    else:
        char_probs = logits
    newline_probs = char_probs[:, positive_index]
    metrics, info = get_metrics(newline_labels, newline_probs)

    info["newline_labels"] = newline_labels

    if use_pysbd:
        segmenter = pysbd.Segmenter(language=lang_code, clean=False, char_span=True)
        predicted_end_indices_pysbd = np.array([x.start + len(x.sent.rstrip()) for x in segmenter.segment(text)])
        newline_probs_pysbd = np.zeros(len(text))
        newline_probs_pysbd[predicted_end_indices_pysbd - 1] = 1.0

        info["newline_probs_pysbd"] = newline_probs_pysbd

    return metrics["pr_auc"], info


def evaluate_sentence_pairwise(
    lang_code,
    sentences,
    model,
    stride,
    block_size,
    batch_size,
    pair_sample_pct: float = 0.01,
    max_pairs: int = 10,
    use_pysbd=False,
    positive_index=None,
    do_lowercase=False,
    do_remove_punct=False,
):
    if positive_index is None:
        positive_index = Constants.NEWLINE_INDEX

    # Preprocess sentences
    sentences = [sentence.lstrip("-").strip() for sentence in sentences]

    separator = Constants.SEPARATORS[lang_code]
    metrics_list = []

    model = PyTorchWrapper(model.backbone)
    model.model = model.model.to("cpu")

    # Generate all possible sentence pairs
    all_pairs = list(zip(sentences[:-1], sentences[1:]))

    # Randomly sample N% of the sentence pairs
    sample_size = (
        min(int(len(all_pairs) * pair_sample_pct) + 1, max_pairs)
        if max_pairs is not None
        else int(len(all_pairs) * pair_sample_pct) + 1
    )
    # set seed so we get the same pairs every time
    random.seed(42)
    sampled_pairs = random.sample(all_pairs, sample_size)

    separator = Constants.SEPARATORS[lang_code]
    metrics_list = []

    for sentence1, sentence2 in sampled_pairs:
        if do_lowercase:
            sentence1 = sentence1.lower()
            sentence2 = sentence2.lower()
        if do_remove_punct:
            for punct in Constants.PUNCTUATION_CHARS:
                sentence1 = sentence1.replace(punct, "")
                sentence2 = sentence2.replace(punct, "")

        pair_text = sentence1 + separator + sentence2

        logits, offsets_mapping, tokenizer = extract(
            [pair_text],
            model,
            lang_code=lang_code,
            stride=stride,
            block_size=block_size,
            batch_size=batch_size,
            pairwise=True,
        )
        logits = logits[0]
        if offsets_mapping is not None:
            offsets_mapping = offsets_mapping[0]

        # Calculate newline labels and probabilities
        newline_labels = np.zeros(len(pair_text))
        true_end_index = len(sentence1)
        newline_labels[true_end_index] = 1

        if "xlm" in model.config.model_type:
            tokens = tokenizer.tokenize(pair_text, verbose=False)
            char_probs = token_to_char_probs(pair_text, tokens, logits, tokenizer, offsets_mapping)
        else:
            char_probs = logits
        newline_probs = char_probs[:, positive_index]

        # Get metrics for the pair
        pair_metrics, _ = get_metrics(newline_labels, newline_probs)
        metrics_list.append(pair_metrics["pr_auc"])

    # Compute and return the average metric
    average_metric = np.mean(metrics_list)
    return average_metric, _
