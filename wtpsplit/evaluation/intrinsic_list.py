import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser

import adapters
import wtpsplit.models  # noqa: F401
from wtpsplit.evaluation import evaluate_mixture, get_labels, token_to_char_probs, train_mixture
from wtpsplit.extract import PyTorchWrapper, extract
from wtpsplit.utils import Constants

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class Args:
    model_path: str
    adapter_path: str = None
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
    block_size: int = 510
    stride: int = 64
    batch_size: int = 1
    include_langs: List[str] = None
    custom_language_list: str = None
    threshold: float = 0.01
    max_n_train_sentences: int = 10_000
    save_suffix: str = ""
    do_lowercase: bool = False
    do_remove_punct: bool = False
    do_strip: bool = False
    tqdm: bool = False


def process_logits_list(text, model, lang_code, block_size, stride, batch_size, verbose=True) -> List[np.ndarray]:
    logits_list = []

    for chunk in tqdm(text, disable=not verbose):
        merged_chunk = Constants.SEPARATORS[lang_code].join(chunk)
        # Extract necessary data
        logits, offsets_mapping, tokenizer = extract(
            [merged_chunk],
            model,
            lang_code=lang_code,
            stride=args.stride,
            block_size=block_size,
            batch_size=1,
            pad_last_batch=True,
        )
        logits = logits[0]
        if offsets_mapping is not None:
            offsets_mapping = offsets_mapping[0]

        if "xlm" in model.config.model_type:
            tokens = tokenizer.tokenize(merged_chunk, verbose=False)

            # padding is also removed here (via offset_mapping)
            logits = token_to_char_probs(merged_chunk, tokens, logits, tokenizer, offsets_mapping)
            logits_list.append(logits)
        else:
            raise NotImplementedError("Only XLM models are supported for now")

    return logits_list


def corrupt(text: str, do_lowercase: bool, do_remove_punct: bool):
    if do_lowercase:
        text = text.lower()
    if do_remove_punct:
        for punct in Constants.PUNCTUATION_CHARS:
            text = text.replace(punct, "")
    return text


def load_or_compute_logits(args, model, eval_data, valid_data=None, save_str: str = None):
    logits_path = Constants.CACHE_DIR / "intrinsic_list" / f"{save_str}.h5"

    if not os.path.exists(Constants.CACHE_DIR / "intrinsic_list"):
        os.makedirs(Constants.CACHE_DIR / "intrinsic_list")

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

            # eval data
            for dataset_name, dataset in eval_data[lang_code]["sentence"].items():
                # train on all mldb, eval on mldbW 
                if "mldbW" in args.eval_data_path and (
                    "mldbW" not in args.model_path and "mldbW" not in args.adapter_path
                ):
                    dataset_load_name = "unk"
                else:
                    dataset_load_name = dataset_name
                try:
                    if args.adapter_path:
                        model.model.load_adapter(
                            args.adapter_path + "/" + dataset_load_name + "/" + lang_code,
                            set_active=True,
                            with_head=True,
                            load_as="text",
                        )
                    if hasattr(model.model.config, "unfreeze_ln"):
                        if model.model.config.unfreeze_ln:
                            ln_dict = torch.load(
                                args.adapter_path + "/" + dataset_load_name + "/" + lang_code + "/ln_dict.pth"
                            )
                            for n, p in model.backbone.named_parameters():
                                if "LayerNorm" in n:
                                    p.data = ln_dict[n].data
                    if not os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")):
                        model_path = os.path.join(args.model_path, dataset_load_name, "en")
                        print(model_path)
                        model = PyTorchWrapper(
                            AutoModelForTokenClassification.from_pretrained(model_path).to(args.device)
                        )
                except Exception as e:
                    print(f"Error loading adapter for {dataset_load_name} in {lang_code}: {e}")
                    continue
                print(dataset_name, dataset_load_name)
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"]
                    if args.do_strip:
                        test_sentences = [
                            [sentence.lstrip("-").strip() for sentence in chunk] for chunk in test_sentences
                        ]
                    test_sentences = [
                        [
                            corrupt(sentence, do_lowercase=args.do_lowercase, do_remove_punct=args.do_remove_punct)
                            for sentence in chunk
                        ]
                        for chunk in test_sentences
                    ]

                    start_time = time.time()  # Start timing for test logits processing
                    test_logits = process_logits_list(
                        test_sentences,
                        model,
                        lang_code,
                        args.block_size,
                        args.stride,
                        args.batch_size,
                    )
                    end_time = time.time()  # End timing for test logits processing
                    total_test_time += end_time - start_time  # Accumulate test processing time
                    test_logit_lengths = []
                    # store start and end indices for each pair, used later to slice the logits
                    all_logit_lengths = np.append(0, np.cumsum([len(logits) for logits in test_logits]))
                    # append tuple of start and end indices for each pair
                    for i in range(len(test_logits)):
                        test_logit_lengths.append((all_logit_lengths[i], all_logit_lengths[i + 1] - 1))

                    test_logits = np.concatenate(test_logits)
                    test_labels = [
                        get_labels(lang_code, test_chunk, after_space=False)[:-1] for test_chunk in test_sentences
                    ]
                    test_labels = np.append(np.concatenate(test_labels), 0)
                    assert len(test_labels) == len(test_logits) + 1

                    dset_group.create_dataset("test_logits", data=test_logits)
                    dset_group.create_dataset("test_labels", data=test_labels)
                    dset_group.create_dataset("test_logit_lengths", data=test_logit_lengths)

                train_sentences = dataset["meta"].get("train_data")
                if train_sentences is not None and "train_logits" not in dset_group:
                    train_sentences = [
                        [
                            corrupt(sentence, do_lowercase=args.do_lowercase, do_remove_punct=args.do_remove_punct)
                            for sentence in chunk
                        ]
                        for chunk in train_sentences
                    ]
                    if args.do_strip:
                        train_sentences = [
                            [sentence.lstrip("-").strip() for sentence in chunk] for chunk in train_sentences
                        ]
                    train_sentences = train_sentences[: args.max_n_train_sentences]

                    train_logits = process_logits_list(
                        train_sentences,
                        model,
                        lang_code,
                        args.block_size,
                        args.stride,
                        args.batch_size,
                    )
                    train_logit_lengths = []
                    # store start and end indices for each pair, used later to slice the logits
                    all_logit_lengths = np.append(0, np.cumsum([len(logits) for logits in train_logits]))
                    # append tuple of start and end indices for each pair
                    for i in range(len(train_logits)):
                        train_logit_lengths.append((all_logit_lengths[i], all_logit_lengths[i + 1] - 1))

                    train_logits = np.concatenate(train_logits)
                    train_labels = [
                        get_labels(lang_code, train_chunk, after_space=False)[:-1] for train_chunk in train_sentences
                    ]
                    train_labels = np.append(np.concatenate(train_labels), 0)
                    assert len(train_labels) == len(train_logits) + 1

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
    save_model_path = args.model_path
    if args.adapter_path:
        save_model_path = args.adapter_path
    save_str = (
        f"{save_model_path.replace('/','_')}_b{args.block_size}_s{args.stride}_u{args.threshold}{args.save_suffix}"
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
    # if model_path does not contain a model, take first subfolder
    if not os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")):
        model_path = os.path.join(args.model_path, os.listdir(args.model_path)[0], "en")
        print("joined")
        print(model_path)
    else:
        model_path = args.model_path
    model = PyTorchWrapper(AutoModelForTokenClassification.from_pretrained(model_path).to(args.device))
    if args.adapter_path:
        model_type = model.model.config.model_type
        # adapters need xlm-roberta as model type.
        model.model.config.model_type = "xlm-roberta"
        adapters.init(model.model)
        # reset model type (used later)
        model.model.config.model_type = model_type
        if "meta-clf" in args.adapter_path:
            clf = model.model.classifier
            model.model.classifier = torch.nn.Sequential(clf, torch.nn.Linear(clf.out_features, 1))

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
            if args.do_strip:
                sentences = [[sentence.lstrip("-").strip() for sentence in chunk] for chunk in sentences]
            sentences = [
                [
                    corrupt(sentence, do_lowercase=args.do_lowercase, do_remove_punct=args.do_remove_punct)
                    for sentence in chunk
                ]
                for chunk in sentences
            ]
            # check if f[lang_code][dataset_name] exists
            if lang_code not in f or dataset_name not in f[lang_code]:
                continue

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

                score_t = []
                score_punct = []
                for i, chunk in tqdm(enumerate(sentences), total=len(sentences), disable=args.tqdm):
                    start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                    single_score_t, single_score_punct, info = evaluate_mixture(
                        lang_code,
                        f[lang_code][dataset_name]["test_logits"][:][start:end],
                        list(chunk),
                        *clf,
                    )
                    score_t.append(single_score_t)
                    score_punct.append(single_score_punct)

                clfs[lang_code][dataset_name] = clf

                clf = list(copy.deepcopy(clf))
                clf[-1] = args.threshold
            else:
                score_t = score_punct = None

            score_u = []
            for i, chunk in tqdm(enumerate(sentences), total=len(sentences), disable=args.tqdm):
                start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                single_score_u, _, info = evaluate_mixture(
                    lang_code,
                    f[lang_code][dataset_name]["test_logits"][:][start:end],
                    list(chunk),
                    *clf,
                )
                score_u.append(single_score_u)

            score_u = np.mean(score_u)
            score_t = np.mean(score_t) if score_t else None
            score_punct = np.mean(score_punct) if score_punct else None

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

    # sio.dump(
    #     clfs,
    #     open(
    #         Constants.CACHE_DIR / "intrinsic_list" / f"{save_str}.skops",
    #         "wb",
    #     ),
    # )
    json.dump(
        results,
        open(
            Constants.CACHE_DIR / "intrinsic_list" / f"{save_str}.json",
            "w",
        ),
        indent=4,
    )

    # Write results_avg to JSON
    json.dump(
        results_avg,
        open(
            Constants.CACHE_DIR / "intrinsic_list" / f"{save_str}_AVG.json",
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
