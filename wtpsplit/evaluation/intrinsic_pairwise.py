import copy
import json
from dataclasses import dataclass
from typing import List, Tuple
import os
import time
import random
import sys

import h5py
import skops.io as sio
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser
import numpy as np

import wtpsplit.models  # noqa: F401
from wtpsplit.evaluation import evaluate_mixture, get_labels, train_mixture, token_to_char_probs
from wtpsplit.extract import PyTorchWrapper
from wtpsplit.extract_batched import extract_batched
from wtpsplit.utils import Constants
from wtpsplit.evaluation.intrinsic import compute_statistics, corrupt


@dataclass
class Args:
    model_path: str
    # eval data in the format:
    # {
    #    "<lang_code>": {
    #        "sentence": {
    #            "<dataset_name>": {
    #                 "meta": {
    #                     "train_data": ["train sentence 1", "train sentence 2"]
    #                 },
    #                 "data": ["test sentence 1", "test sentence 2"]
    #            }
    #        }
    #    }
    # }
    eval_data_path: str = "data/eval.pth"
    valid_text_path: str = None  # "data/sentence/valid.parquet"
    device: str = "cpu"
    block_size: int = 512
    batch_size: int = 128
    include_langs: List[str] = None
    threshold: float = 0.01
    max_n_train_sentences: int = 10_000
    save_suffix: str = ""
    do_lowercase: bool = False
    do_remove_punct: bool = False
    # pairwise-specific args
    max_n_pairs: int = sys.maxsize
    pair_sample_pct: float = 1
    min_pair_length: int = 0


def process_logits_pairwise(pairs, model, lang_code, args):
    logits_list = []
    # create batches of sentence pairs
    batched_pairs = [pairs[i : i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
    for batch in tqdm(batched_pairs):
        pair_texts = [pair[0] + Constants.SEPARATORS[lang_code] + pair[1] for pair in batch]
        logits, offsets_mapping, tokenizer = extract_batched(
            pair_texts,
            model,
            lang_code=lang_code,
            block_size=args.block_size,
            batch_size=args.batch_size,
            pad_last_batch=True,
        )

        for pair, logit, offset_mapping in zip(pair_texts, logits, offsets_mapping):
            if "xlm" in model.config.model_type:
                tokens = tokenizer.tokenize(pair, verbose=False)

                # padding is also removed here (via offset_mapping)
                logits = token_to_char_probs(pair, tokens, logit, tokenizer, offset_mapping)
                logits_list.append(logits)
            else:
                if len(logit) < offset_mapping:
                    # truncated input --> pad back
                    logit = np.pad(
                        logit, ((0, offset_mapping - len(logit)), (0, 0)), "constant", constant_values=np.min(logit)
                    )
                # since we pad to equal length, we need to remove the padding
                logits_list.append(logit[:offset_mapping])

    return logits_list


def generate_pairs(sentences: List[str], args) -> List[Tuple[str, str]]:
    """Generate sentence pairs from a list of sentences.

    Args:
        sentences (List[str]): Input list of sentences.
        args: Args object containing the following relevant attributes:
            - pair_sample_pct: Percentage of all possible pairs to sample.
            - min_pair_length: Minimum length of a sentence pair.
            - max_n_sentences: Maximum number of sentences to sample.
            - do_lowercase: Whether to lowercase the sentences.
            - do_remove_punct: Whether to remove punctuation from the sentences.

    Returns:
        List[Tuple[str, str]]: List of sentence pairs.
    """
    random.seed(42)
    n_pairs = len(sentences) // 2
    sample_size = min(round(n_pairs * args.pair_sample_pct), args.max_n_pairs)

    # If we need to sample a subset of all possible pairs, do so efficiently
    if sample_size < n_pairs:
        sampled_indices = set(random.sample(range(n_pairs), sample_size))
        all_pairs = [
            (sentences[2 * i], sentences[2 * i + 1])
            for i in sampled_indices
            if len(sentences[2 * i]) + len(sentences[2 * i + 1]) > args.min_pair_length
        ]
    else:
        # Generate all pairs that meet the min_pair_length criterion
        all_pairs = [
            (sentences[i], sentences[i + 1])
            for i in range(0, len(sentences) - 1, 2)
            if len(sentences[i]) + len(sentences[i + 1]) > args.min_pair_length
        ]

    # corrupt pairs
    all_pairs = [(corrupt(pair[0], args), corrupt(pair[1], args)) for pair in all_pairs]
    return all_pairs


def load_or_compute_logits(args, model, eval_data, valid_data=None, save_str: str = None):
    logits_path = Constants.CACHE_DIR / "intrinsic_pairwise" / f"{save_str}.h5"

    if not os.path.exists(Constants.CACHE_DIR / "intrinsic_pairwise"):
        os.makedirs(Constants.CACHE_DIR / "intrinsic_pairwise")

    total_test_time = 0  # Initialize total test processing time

    # FIXME: revert to "a"
    start_time = time.time()
    with h5py.File(logits_path, "w") as f, torch.no_grad():
        for lang_code in Constants.LANGINFO.index:
            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            print(f"Processing {lang_code}...")
            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]

            # eval data
            for dataset_name, dataset in eval_data[lang_code]["sentence"].items():
                print(dataset_name)
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"]
                    all_pairs_test = generate_pairs(test_sentences, args)

                    start_time = time.time()  # Start timing for test logits processing
                    test_logits = process_logits_pairwise(
                        all_pairs_test,
                        model,
                        lang_code,
                        args,
                    )
                    end_time = time.time()

                    test_logit_lengths = []
                    # store start and end indices for each pair, used later to slice the logits
                    all_logit_lengths = np.append(0, np.cumsum([len(logits) for logits in test_logits]))
                    # append tuple of start and end indices for each pair
                    for i in range(len(test_logits)):
                        test_logit_lengths.append((all_logit_lengths[i], all_logit_lengths[i + 1] - 1))

                    test_logits = np.concatenate(test_logits)
                    total_test_time += end_time - start_time  # Accumulate test processing time

                    # get_labels returns 2nd label at end of seq, which we do not want.
                    # label is at position -2 --> remove and add back 0 to end of sequence
                    test_labels = [
                        np.append(get_labels(lang_code, [pair[0], pair[1]], after_space=False)[:-2], 0)
                        for pair in all_pairs_test
                    ]

                    # flatten; append 0 eos to account for later indexing/slicing
                    test_labels = np.append(np.concatenate(test_labels), 0)
                    assert len(test_labels) == len(test_logits) + 1

                    dset_group.create_dataset("test_logits", data=test_logits)
                    dset_group.create_dataset("test_labels", data=test_labels)
                    dset_group.create_dataset("test_logit_lengths", data=test_logit_lengths)

                train_sentences = dataset["meta"].get("train_data")
                if train_sentences is not None and "train_logits" not in dset_group:
                    train_sentences = train_sentences[: args.max_n_train_sentences]
                    all_pairs_train = generate_pairs(train_sentences, args)

                    train_logits = process_logits_pairwise(all_pairs_train, model, lang_code, args)
                    train_logits = np.concatenate(train_logits)

                    train_labels = [
                        np.append(get_labels(lang_code, [pair[0], pair[1]], after_space=False)[:-2], 0)
                        for pair in all_pairs_train
                    ]
                    train_labels = np.append(np.concatenate(train_labels), 0)
                    assert len(train_labels) == len(train_logits) + 1

                    dset_group.create_dataset("train_logits", data=train_logits)
                    dset_group.create_dataset("train_labels", data=train_labels)

    end_time = time.time()
    return h5py.File(logits_path, "r"), total_test_time / 60  # to minutes


def main(args):
    save_str = f"{args.model_path.replace('/','_')}_b{args.block_size}_u{args.threshold}{args.save_suffix}"
    if args.do_lowercase:
        save_str += "_lc"
    if args.do_remove_punct:
        save_str += "_rmp"

    eval_data = torch.load(args.eval_data_path)
    if args.valid_text_path is not None:
        valid_data = load_dataset("parquet", data_files=args.valid_text_path, split="train")
    else:
        valid_data = None

    print("Loading model...")
    model = PyTorchWrapper(AutoModelForTokenClassification.from_pretrained(args.model_path).to(args.device))

    # first, logits for everything.
    f, total_test_time = load_or_compute_logits(args, model, eval_data, valid_data, save_str)

    # now, compute the intrinsic scores.
    results = {}
    clfs = {}
    # Initialize lists to store scores for each metric across all languages
    u_scores, t_scores, punct_scores = [], [], []
    u_accs, t_accs, punct_accs = [], [], []

    for lang_code, dsets in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        print(f"Predicting {lang_code}...")
        results[lang_code] = {}
        clfs[lang_code] = {}

        for dataset_name, dataset in dsets["sentence"].items():
            sentences = dataset["data"]
            sent_pairs = generate_pairs(sentences, args)

            if "train_logits" in f[lang_code][dataset_name]:
                feature_indices = None
                # it is sufficient to feed in 1 long sequence of tokens here since we only use logits for LR
                clf = train_mixture(
                    [lang_code],
                    f[lang_code][dataset_name]["train_logits"][:],
                    f[lang_code][dataset_name]["train_labels"][:],
                    features=feature_indices,
                )
                # XXX: clf thresholds are still fitted on max. F1 score, not accuracy!
                # (but still without a positive label at the end)
                if clf[0] is not None:
                    print(clf)

                score_t = []
                score_punct = []
                # acc: average of correct 100% pairwise segmentation
                acc_t = []
                acc_punct = []

                # evaluate each pair
                for i, pair in enumerate(sent_pairs):
                    start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                    single_score_t, single_score_punct, info = evaluate_mixture(
                        lang_code,
                        f[lang_code][dataset_name]["test_logits"][:][start:end],
                        list(pair),
                        *clf,
                    )
                    score_t.append(single_score_t)
                    score_punct.append(single_score_punct)
                    acc_t.append(info["info_newline"]["correct_pairwise"])
                    acc_punct.append(info["info_transformed"]["correct_pairwise"])

                clfs[lang_code][dataset_name] = clf

                clf = list(copy.deepcopy(clf))
                clf[-1] = args.threshold
            else:
                score_t = score_punct = None
                acc_t = acc_punct = None

            score_u = []
            acc_u = []
            for i, pair in enumerate(sent_pairs):
                start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                single_score_u, _, info = evaluate_mixture(
                    lang_code,
                    f[lang_code][dataset_name]["test_logits"][:][start:end],
                    list(pair),
                    *clf,
                )
                score_u.append(single_score_u)
                acc_u.append(info["info_newline"]["correct_pairwise"])

            score_u = np.mean(score_u)
            score_t = np.mean(score_t) if score_t else None
            score_punct = np.mean(score_punct) if score_punct else None
            acc_u = np.mean(acc_u)
            acc_t = np.mean(acc_t) if score_t else None
            acc_punct = np.mean(acc_punct) if score_punct else None

            results[lang_code][dataset_name] = {
                "u": score_u,
                "t": score_t,
                "punct": score_punct,
                "u_acc": acc_u,
                "t_acc": acc_t,
                "punct_acc": acc_punct,
            }

            # just for printing
            score_t = score_t or 0.0
            score_punct = score_punct or 0.0
            acc_t = acc_t or 0.0
            acc_punct = acc_punct or 0.0

            u_scores.append((score_u, lang_code))
            u_accs.append((acc_u, lang_code))
            t_scores.append((score_t, lang_code))
            t_accs.append((acc_t, lang_code))
            punct_scores.append((score_punct, lang_code))
            punct_accs.append((acc_punct, lang_code))

            print(f"{lang_code} {dataset_name} {score_u:.3f} {score_t:.3f} {score_punct:.3f}")
            print(f"ACC: {acc_u:.3f} {acc_t:.3f} {acc_punct:.3f}")

    # Compute statistics for each metric across all languages
    results_avg = {
        "u": compute_statistics(u_scores),
        "t": compute_statistics(t_scores),
        "punct": compute_statistics(punct_scores),
        "u_acc": compute_statistics(u_accs),
        "t_acc": compute_statistics(t_accs),
        "punct_acc": compute_statistics(punct_accs),
        "include_langs": args.include_langs,
    }

    json.dump(
        results,
        open(
            Constants.CACHE_DIR / "intrinsic_pairwise" / f"{save_str}.json",
            "w",
        ),
        indent=4,
    )

    # Write results_avg to JSON
    json.dump(
        results_avg,
        open(
            Constants.CACHE_DIR / "intrinsic_pairwise" / f"{save_str}_AVG.json",
            "w",
        ),
        indent=4,
    )
    os.remove(f.filename)
    return results, results_avg, total_test_time


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    results, results_avg, total_test_time = main(args)
    print(total_test_time)
