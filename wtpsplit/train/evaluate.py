import copy
import logging
import sys

import numpy as np
import pysbd
import sklearn.metrics

from wtpsplit.evaluation import token_to_char_probs
from wtpsplit.evaluation.intrinsic_pairwise import generate_pairs, process_logits_pairwise
from wtpsplit.extract import PyTorchWrapper, extract
from wtpsplit.utils import Constants
from wtpsplit.evaluation.intrinsic import corrupt

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
    sentences = [corrupt(sentence, do_lowercase, do_remove_punct) for sentence in sentences]
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
    pair_sample_pct: float = 0.1,
    max_pairs: int = sys.maxsize,
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
    accuracy_list = []


    # get pairs of sentences (non-overlapping)
    sampled_pairs = generate_pairs(
        sentences=sentences,
        pair_sample_pct=pair_sample_pct,
        max_n_pairs=max_pairs,
        min_pair_length=0,
        do_lowercase=do_lowercase,
        do_remove_punct=do_remove_punct,
    )

    # get logits for each pair
    logits = process_logits_pairwise(
        pairs=sampled_pairs,
        model=PyTorchWrapper(model.backbone),
        lang_code=lang_code,
        block_size=block_size,
        batch_size=batch_size,
        verbose=False,
    )

    # simulate performance for WtP-U
    DEFAULT_THRESHOLD = 0.01
    

    for i, (sentence1, sentence2) in enumerate(sampled_pairs):
        newline_probs = logits[i][:, positive_index]

        pair_text = sentence1 + separator + sentence2

        # Calculate newline labels and probabilities
        true_end_indices = np.cumsum(np.array([len(s) for s in [sentence1, sentence2]])) + np.arange(2) * len(separator)
        newline_labels = np.zeros(len(pair_text))
        newline_labels[true_end_indices - 1] = 1

        # Get metrics for the pair
        pair_metrics, _ = get_metrics(newline_labels, newline_probs)
        metrics_list.append(pair_metrics["pr_auc"])
        predicted_labels = newline_probs > np.log(DEFAULT_THRESHOLD / (1 - DEFAULT_THRESHOLD))  # inverse sigmoid
        # for accuracy, check if the single label in between is correctly predicted (ignore the one at the end)
        if sum(predicted_labels[:-1]) > 0:
            correct = (np.where(newline_labels[:-1])[0] == np.where(predicted_labels[:-1])[0]).all()
            accuracy_list.append(correct)
        else:
            accuracy_list.append(False)

    # Compute and return the average metric
    average_metric = np.mean(metrics_list)
    avg_accuracy = np.mean(accuracy_list)
    return average_metric, avg_accuracy
