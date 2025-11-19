import logging
import sys
from typing import Literal

import numpy as np
import pysbd
import sklearn.metrics

from wtpsplit.evaluation.intrinsic_pairwise import generate_k_mers, process_logits_k_mers
from wtpsplit.extract import PyTorchWrapper, extract
from wtpsplit.utils import Constants, sigmoid, corrupt, token_to_char_probs

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


def get_metrics(labels, preds, threshold: float = 0.01):
    # Compute precision-recall curve and AUC
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, sigmoid(preds))
    pr_auc = sklearn.metrics.auc(recall, precision)

    # Compute F1 scores for all thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find best F1 score and its corresponding threshold
    best_f1_index = np.argmax(f1_scores[:-1])  # Exclude last value because it corresponds to recall of 0.
    best_f1 = f1_scores[best_f1_index]
    best_threshold = thresholds[best_f1_index]

    # Compute F1 score for a specific threshold (e.g., 0.01 after applying sigmoid)
    f1_at_specific_threshold = sklearn.metrics.f1_score(labels, sigmoid(preds) > threshold)

    metrics = {"pr_auc": pr_auc}
    info = {
        "probs": preds,
        "labels": labels,
        "recalls": recall,
        "precisions": precision,
        "f1_scores": f1_scores,
        "f1_best": best_f1,
        "threshold_best": sigmoid(best_threshold).item(),  # TODO: is the 2nd sigmoid a bug? TBD.
        "f1": f1_at_specific_threshold,
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
    weighting: Literal["uniform", "hat"] = "uniform",
    use_pysbd=False,
    positive_index=None,
    do_lowercase=False,
    do_remove_punct=False,
    threshold: float = 0.01,
):
    if positive_index is None:
        positive_index = Constants.NEWLINE_INDEX

    # preprocessing, many OPUS100 (and some UD) sentences start with "- "
    sentences = [sentence.lstrip("-").strip() for sentence in sentences]

    separator = Constants.SEPARATORS[lang_code]
    sentences = [corrupt(sentence, do_lowercase, do_remove_punct) for sentence in sentences]
    text = separator.join(sentences)

    logits, offsets_mapping, tokenizer, _ = extract(
        [text],
        PyTorchWrapper(model.backbone),
        lang_code=lang_code,
        stride=stride,
        max_block_size=block_size,
        batch_size=batch_size,
        weighting=weighting,
    )
    logits = logits[0]
    if offsets_mapping is not None:
        offsets_mapping = offsets_mapping[0]

    true_end_indices = np.cumsum(np.array([len(s) for s in sentences])) + np.arange(len(sentences)) * len(separator)
    newline_labels = np.zeros(len(text))
    newline_labels[true_end_indices - 1] = 1

    if "xlm" in model.config.model_type:
        tokens = tokenizer.tokenize(text, verbose=False)
        char_probs = token_to_char_probs(
            text, tokens, logits, [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token], offsets_mapping
        )
    else:
        char_probs = logits
    newline_probs = char_probs[:, positive_index]
    metrics, info = get_metrics(newline_labels, newline_probs, threshold=threshold)

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
    threshold: float = 0.01,
):
    if positive_index is None:
        positive_index = Constants.NEWLINE_INDEX

    # Preprocess sentences
    sentences = [sentence.lstrip("-").strip() for sentence in sentences]

    separator = Constants.SEPARATORS[lang_code]
    metrics_list = []
    accuracy_list = []

    # get pairs of sentences (non-overlapping)
    sampled_pairs = generate_k_mers(
        sentences=sentences,
        k=2,
        sample_pct=pair_sample_pct,
        max_n_samples=max_pairs,
        min_k_mer_length=0,
    )

    # get logits for each pair
    logits, n_tokens_list = process_logits_k_mers(
        pairs=sampled_pairs,
        model=PyTorchWrapper(model.backbone),
        lang_code=lang_code,
        block_size=block_size,
        batch_size=batch_size,
        verbose=False,
    )

    # simulate performance for WtP-U
    for i, (sentence1, sentence2) in enumerate(sampled_pairs):
        newline_probs = logits[i][:, positive_index]

        pair_text = sentence1 + separator + sentence2

        # Calculate newline labels and probabilities
        true_end_indices = np.cumsum(np.array([len(s) for s in [sentence1, sentence2]])) + np.arange(2) * len(separator)
        newline_labels = np.zeros(len(pair_text))
        newline_labels[true_end_indices - 1] = 1

        # Get metrics for the pair
        pair_metrics, _ = get_metrics(newline_labels, newline_probs, threshold=threshold)
        metrics_list.append(pair_metrics["pr_auc"])
        predicted_labels = newline_probs > np.log(threshold / (1 - threshold))  # inverse sigmoid
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


def evaluate_sentence_kmers(
    lang_code,
    sentences,
    model,
    stride,
    block_size,
    batch_size,
    k: int = 3,
    sample_pct: float = 0.1,
    max_n_samples: int = sys.maxsize,
    use_pysbd=False,
    positive_index=None,
    do_lowercase=False,
    do_remove_punct=False,
    threshold: float = 0.01,
):
    if positive_index is None:
        positive_index = Constants.NEWLINE_INDEX

    # Preprocess sentences
    sentences = [sentence.lstrip("-").strip() for sentence in sentences]

    separator = Constants.SEPARATORS[lang_code]
    metrics_list = []
    accuracy_list = []
    accuracy_list_optimal = []
    info_list = []

    # get pairs of sentences (non-overlapping)
    sampled_k_mers = generate_k_mers(
        sentences=sentences,
        k=k,
        do_lowercase=do_lowercase,
        do_remove_punct=do_remove_punct,
        sample_pct=sample_pct,
        max_n_samples=max_n_samples,
        min_k_mer_length=0,
    )

    # get logits for each pair
    logits, n_tokens_list = process_logits_k_mers(
        pairs=sampled_k_mers,
        model=PyTorchWrapper(model.backbone),
        lang_code=lang_code,
        block_size=block_size,
        batch_size=batch_size,
        verbose=False,
    )

    for i, k_mer in enumerate(sampled_k_mers):
        newline_probs = logits[i][:, positive_index]

        k_mer_text = separator.join(k_mer)
        true_end_indices = np.cumsum(np.array([len(s) for s in k_mer])) + np.arange(len(k_mer)) * len(separator)
        newline_labels = np.zeros(len(k_mer_text))
        newline_labels[true_end_indices - 1] = 1

        # Get metrics for the k-mer
        k_mer_metrics, info = get_metrics(newline_labels, newline_probs, threshold=threshold)
        metrics_list.append(k_mer_metrics["pr_auc"])
        info_list.append(info)

        predicted_labels = newline_probs > np.log(threshold / (1 - threshold))  # inverse sigmoid
        predicted_labels_optimal = newline_probs > np.log(
            info["threshold_best"] / (1 - info["threshold_best"])
        )  # inverse sigmoid
        # For accuracy, check if all the labels in between are correctly predicted (ignore the one at the end)
        intermediate_newline_labels = newline_labels[:-1]  # Exclude the end
        intermediate_predicted_labels = predicted_labels[:-1]
        intermediate_predicted_labels_opt = predicted_labels_optimal[:-1]
        correct = np.array_equal(intermediate_newline_labels, intermediate_predicted_labels)
        correct_optimal = np.array_equal(intermediate_newline_labels, intermediate_predicted_labels_opt)
        accuracy_list.append(correct)
        accuracy_list_optimal.append(correct_optimal)

    # Compute and return the average metric and accuracy
    average_metric = np.mean(metrics_list)
    avg_accuracy = np.mean(accuracy_list)
    # get averages for info_list
    # print(len(info_list), len(sampled_k_mers))
    if len(sampled_k_mers) > 0:
        avg_info = {
            key: np.mean([info[key] for info in info_list])
            for key in info_list[0].keys()
            if isinstance(info_list[0][key], (int, float))
        }
        avg_info["accuracy_optimal"] = np.mean(accuracy_list_optimal)
    else:
        avg_info = {}
        avg_info["f1"] = 0
        avg_info["f1_best"] = 0
        avg_info["threshold_best"] = 0
    return average_metric, avg_accuracy, avg_info
