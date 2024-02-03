import copy
import json
from dataclasses import dataclass
from typing import List
import os
import time

import h5py
import skops.io as sio
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser
import numpy as np

import wtpsplit.models  # noqa: F401
from wtpsplit.evaluation import evaluate_mixture, get_labels, train_mixture, token_to_char_probs
from wtpsplit.extract import PyTorchWrapper, extract
from wtpsplit.utils import Constants


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
    # TODO: for songs/etc., maybe feed in each sample separately?
    eval_data_path: str = "data/eval.pth"
    valid_text_path: str = None  # "data/sentence/valid.parquet"
    device: str = "cpu"
    block_size: int = 512
    stride: int = 64
    batch_size: int = 32
    include_langs: List[str] = None
    custom_language_list: str = None
    threshold: float = 0.01
    max_n_train_sentences: int = 10_000
    save_suffix: str = ""
    do_lowercase: bool = False
    do_remove_punct: bool = False


def process_logits(text, model, lang_code, args):
    # Extract necessary data
    logits, offsets_mapping, tokenizer, _ = extract(
        [text],
        model,
        lang_code=lang_code,
        stride=args.stride,
        block_size=args.block_size,
        batch_size=args.batch_size,
        pad_last_batch=True,
        verbose=True,
    )
    logits = logits[0]
    if offsets_mapping is not None:
        offsets_mapping = offsets_mapping[0]

    if "xlm" in model.config.model_type:
        tokens = tokenizer.tokenize(text, verbose=False)

        # Use the vectorized function to convert token probabilities to character probabilities for the entire array
        char_probs = token_to_char_probs(text, tokens, logits, tokenizer, offsets_mapping)

        logits = char_probs

    return logits


def corrupt(text: str, args: Args):
    if args.do_lowercase:
        text = text.lower()
    if args.do_remove_punct:
        for punct in Constants.PUNCTUATION_CHARS:
            text = text.replace(punct, "")
    return text


def load_or_compute_logits(args, model, eval_data, valid_data=None, save_str: str = None):
    logits_path = Constants.CACHE_DIR / "intrinsic" / f"{save_str}.h5"

    if not os.path.exists(Constants.CACHE_DIR / "intrinsic"):
        os.makedirs(Constants.CACHE_DIR / "intrinsic")

    if args.custom_language_list is not None:
        with open(args.custom_language_list, "r") as f:
            # file is a csv: l1,l2,...
            use_langs = f.read().strip().split(",")
    else:
        use_langs = Constants.LANGINFO.index

    total_test_time = 0  # Initialize total test processing time

    # TODO: revert to "a"
    with h5py.File(logits_path, "w") as f, torch.no_grad():
        for lang_code in use_langs:
            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            print(f"Processing {lang_code}...")
            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]

            # valid data
            if valid_data is not None and "valid" not in lang_group:
                valid_sentences = [sample["text"].strip() for sample in valid_data if sample["lang"] == lang_code]
                assert len(valid_sentences) > 0

                valid_sentences = [corrupt(sentence, args) for sentence in valid_sentences]
                separator = Constants.SEPARATORS[lang_code]
                valid_text = separator.join(valid_sentences)

                valid_logits = process_logits(valid_text, model, lang_code, args)

                lang_group.create_dataset("valid", data=valid_logits)

            # eval data
            for dataset_name, dataset in eval_data[lang_code]["sentence"].items():
                print(dataset_name)
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"]
                    test_sentences = [corrupt(sentence, args) for sentence in test_sentences]
                    test_text = Constants.SEPARATORS[lang_code].join(test_sentences)

                    start_time = time.time()  # Start timing for test logits processing
                    test_logits = process_logits(test_text, model, lang_code, args)
                    end_time = time.time()  # End timing for test logits processing
                    total_test_time += end_time - start_time  # Accumulate test processing time

                    test_labels = get_labels(lang_code, test_sentences, after_space=False)

                    dset_group.create_dataset("test_logits", data=test_logits)
                    dset_group.create_dataset("test_labels", data=test_labels)

                train_sentences = dataset["meta"].get("train_data")
                if train_sentences is not None and "train_logits" not in dset_group:
                    train_sentences = [corrupt(sentence, args) for sentence in train_sentences]
                    train_sentences = train_sentences[: args.max_n_train_sentences]
                    train_text = Constants.SEPARATORS[lang_code].join(train_sentences)

                    train_logits = process_logits(train_text, model, lang_code, args)
                    train_labels = get_labels(lang_code, train_sentences, after_space=False)

                    dset_group.create_dataset("train_logits", data=train_logits)
                    dset_group.create_dataset("train_labels", data=train_labels)

    end_time = time.time()
    return h5py.File(logits_path, "r"), total_test_time / 60  # to minutes


def compute_statistics(values):
    if not values:  # Check for empty values list
        return {"mean": None, "median": None, "std": None, "min": None, "min_lang": None, "max": None, "max_lang": None}

    scores, langs = zip(*values)  # Unpack scores and languages
    min_index = np.argmin(scores)
    max_index = np.argmax(scores)
    return {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": scores[min_index],
        "min_lang": langs[min_index],
        "max": scores[max_index],
        "max_lang": langs[max_index],
    }


def main(args):
    save_str = (
        f"{args.model_path.replace('/','_')}_b{args.block_size}_s{args.stride}_u{args.threshold}{args.save_suffix}"
    )
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

    for lang_code, dsets in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        print(f"Predicting {lang_code}...")
        results[lang_code] = {}
        clfs[lang_code] = {}

        for dataset_name, dataset in dsets["sentence"].items():
            sentences = dataset["data"]

            if "train_logits" in f[lang_code][dataset_name]:
                feature_indices = None
                clf = train_mixture(
                    [lang_code],
                    f[lang_code][dataset_name]["train_logits"][:],
                    f[lang_code][dataset_name]["train_labels"][:],
                    features=feature_indices,
                )
                if clf[0] is not None:
                    print(clf)

                score_t, score_punct, _ = evaluate_mixture(
                    lang_code,
                    f[lang_code][dataset_name]["test_logits"][:],
                    sentences,
                    *clf,
                )

                clfs[lang_code][dataset_name] = clf

                clf = list(copy.deepcopy(clf))
                clf[-1] = args.threshold
            else:
                score_t = score_punct = None

            score_u, _, _ = evaluate_mixture(lang_code, f[lang_code][dataset_name]["test_logits"][:], sentences, *clf)

            results[lang_code][dataset_name] = {
                "u": score_u,
                "t": score_t,
                "punct": score_punct,
            }

            # just for printing
            score_t = score_t or 0.0
            score_punct = score_punct or 0.0

            u_scores.append((score_u, lang_code))
            t_scores.append((score_t, lang_code))
            punct_scores.append((score_punct, lang_code))
            print(f"{lang_code} {dataset_name} {score_u:.3f} {score_t:.3f} {score_punct:.3f}")

    # Compute statistics for each metric across all languages
    results_avg = {
        "u": compute_statistics(u_scores),
        "t": compute_statistics(t_scores),
        "punct": compute_statistics(punct_scores),
        "include_langs": args.include_langs,
    }

    sio.dump(
        clfs,
        open(
            Constants.CACHE_DIR / "intrinsic" / f"{save_str}.skops",
            "wb",
        ),
    )
    json.dump(
        results,
        open(
            Constants.CACHE_DIR / "intrinsic" / f"{save_str}.json",
            "w",
        ),
        indent=4,
    )

    # Write results_avg to JSON
    json.dump(
        results_avg,
        open(
            Constants.CACHE_DIR / "intrinsic" / f"{save_str}_AVG.json",
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
