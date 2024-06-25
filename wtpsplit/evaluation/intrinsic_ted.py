import copy
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
import spacy_alignments as tokenizations
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser

import adapters
import wtpsplit.models  # noqa: F401
from wtpsplit.evaluation import get_labels, train_mixture
from wtpsplit.evaluation.evaluate_sepp_nlg_subtask1 import evaluate_subtask1
from wtpsplit.evaluation.intrinsic import process_logits
from wtpsplit.extract import PyTorchWrapper, extract
from wtpsplit.utils import Constants, sigmoid

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_token_labels(a, b, a_labels):
    a2b, b2a = tokenizations.get_alignments(a, b)

    b_labels = []

    for i, label_indices in enumerate(b2a):
        aligned_subwords = []

        if label_indices:
            for j in label_indices:
                if j < len(a_labels):
                    aligned_subwords.append(a_labels[j])

        if True in aligned_subwords:
            b_labels.append(1)
        else:
            b_labels.append(0)
    if not np.sum(a_labels) == sum(b_labels):
        print(np.sum(a_labels), sum(b_labels))
    b_labels[-1] = 1  # last is always 1
    return b_labels


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
    eval_data_path: str = "data/ted2020.pth"
    valid_text_path: str = None  # "data/sentence/valid.parquet"
    device: str = "cpu"
    block_size: int = 512
    stride: int = 64
    batch_size: int = 32
    include_langs: List[str] = None
    include_splits: List[str] = None
    threshold: float = 0.025
    max_n_train_sentences: int = 100
    max_n_test_sentences: int = sys.maxsize
    save_suffix: str = ""
    skip_adaptation: bool = False
    clf_from_scratch: bool = False
    skip_punct: bool = True


def process_logits_and_tokens(text, model, lang_code, args):
    # variation of process_logits used in intrinsic.py for word-based evals, returning tokens as well.
    if isinstance(text, list):
        logits = []
        tokens = []
        for short_seq in tqdm(text, desc="Evaluating...", disable=False):
            current_logits, current_offsets_mapping, tokenizer, _ = extract(
                [short_seq],
                model,
                lang_code=lang_code,
                stride=args.stride,
                max_block_size=args.block_size,
                batch_size=args.batch_size,
                pad_last_batch=True,
                verbose=False,
            )
            current_logits = current_logits[0]
            if current_offsets_mapping is not None:
                current_offsets_mapping = current_offsets_mapping[0]

            current_tokens = tokenizer.encode(short_seq, verbose=False, add_special_tokens=False)

            logits.append(current_logits)
            tokens.append(current_tokens)
    else:
        raise NotImplementedError
    return logits, tokens


def load_or_compute_logits(args, model, eval_data, valid_data=None, save_str: str = None):
    logits_path = Constants.CACHE_DIR / "ted2020" / f"{save_str}.h5"

    if not os.path.exists(Constants.CACHE_DIR / "ted2020"):
        os.makedirs(Constants.CACHE_DIR / "ted2020")

    use_langs = eval_data.keys()

    total_test_time = 0  # Initialize total test processing time

    with h5py.File(logits_path, "a") as f, torch.no_grad():
        if not args.include_splits:
            splits = ["surprise_test", "test"]
        else:
            splits = args.include_splits
        for lang_code in tqdm(use_langs, desc="Languages"):
            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]

            # eval data
            for dataset_name, dataset in tqdm(eval_data[lang_code]["sentence"].items(), desc=lang_code):
                if dataset_name not in splits:
                    continue
                try:
                    if args.adapter_path:
                        if args.clf_from_scratch:
                            model.model.classifier = torch.nn.Linear(model.model.classifier.in_features, 1)
                        # we trained adapters on "train" split but uniformly save as "surprise_test"
                        model.model.load_adapter(
                            args.adapter_path + "/" + "surprise_test" + "/" + lang_code,
                            set_active=True,
                            with_head=True,
                            load_as="text",
                        )
                    if not os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")) and not os.path.exists(
                        os.path.join(args.model_path, "model.safetensors")
                    ):
                        model_path = os.path.join(args.model_path, dataset_name, "en")
                        if not os.path.exists(model_path):
                            model_path = args.model_path
                        model = PyTorchWrapper(
                            AutoModelForTokenClassification.from_pretrained(model_path).to(args.device)
                        )
                except Exception as e:
                    print(f"Error loading adapter for {dataset_name} in {lang_code}: {e}")
                    continue
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"][: args.max_n_test_sentences]
                    if isinstance(test_sentences[0], list):
                        # short-seq eval: list of lists
                        test_text = [
                            Constants.SEPARATORS.get(lang_code, " ").join(sentence) for sentence in test_sentences
                        ]
                    else:
                        raise NotImplementedError

                    start_time = time.time()
                    test_logits, test_tokens = process_logits_and_tokens(test_text, model, lang_code, args)
                    end_time = time.time()
                    total_test_time += end_time - start_time
                    if isinstance(test_sentences[0], list):
                        test_logit_lengths = []
                        # store start and end indices for each pair, used later to slice the logits
                        # (h5py does not like different length np arrays as list elements)
                        all_logit_lengths = np.append(0, np.cumsum([len(logits) for logits in test_logits]))
                        # append tuple of start and end indices for each pair
                        for i in range(len(test_logits)):
                            test_logit_lengths.append((all_logit_lengths[i], all_logit_lengths[i + 1] - 1))
                        test_logits = np.concatenate(test_logits)
                        test_tokens = np.concatenate(test_tokens)

                        dset_group.create_dataset("test_logit_lengths", data=test_logit_lengths)

                    dset_group.create_dataset("test_logits", data=test_logits)
                    dset_group.create_dataset("test_tokens", data=test_tokens)

                train_sentences = dataset["meta"].get("train_data")
                if train_sentences is not None and "train_logits" not in dset_group and not args.skip_adaptation:
                    train_sentences = train_sentences[: args.max_n_train_sentences]
                    if isinstance(train_sentences[0], list):
                        # short-seq eval: list of lists
                        train_text = [
                            Constants.SEPARATORS.get(lang_code, " ").join(sentence) for sentence in train_sentences
                        ]
                    train_logits = process_logits(train_text, model, lang_code, args)
                    if isinstance(train_sentences[0], list):
                        train_logits = np.concatenate(train_logits)
                        train_labels = [
                            get_labels(lang_code, short_seq, after_space=False)[:-1] for short_seq in train_sentences
                        ]

                        # flatten; append 0 eos to account for later indexing/slicing
                        train_labels = np.append(np.concatenate(train_labels), 1)
                        assert len(train_labels) == len(train_logits) + 1

                    dset_group.create_dataset("train_logits", data=train_logits)
                    dset_group.create_dataset("train_labels", data=train_labels)

    end_time = time.time()
    return h5py.File(logits_path, "r"), total_test_time / 60  # to minutes


def compute_statistics(values):
    if not values:  # Check for empty values list
        return {"mean": None, "median": None, "std": None, "min": None, "min_lang": None, "max": None, "max_lang": None}

    scores, langs = zip(*values)
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
    save_str = f"{save_model_path.replace('/','_')}_b{args.block_size}_s{args.stride}"

    eval_data = torch.load(args.eval_data_path)
    if args.valid_text_path is not None:
        valid_data = load_dataset("parquet", data_files=args.valid_text_path, split="train")
    else:
        valid_data = None

    print("Loading model...")
    # if model_path does not contain a model, take first subfolder
    if not os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")) and not os.path.exists(
        os.path.join(args.model_path, "model.safetensors")
    ):
        try:
            model_path = os.path.join(args.model_path, os.listdir(args.model_path)[0], "en")
        except:  # noqa
            model_path = args.model_path
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

    save_str += f"{args.save_suffix}"
    if args.max_n_test_sentences < sys.maxsize:
        save_str += f"_n{args.max_n_test_sentences}"

    # first, logits for everything.
    f, total_test_time = load_or_compute_logits(args, model, eval_data, valid_data, save_str)

    save_str += f"_u{args.threshold}"

    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model)

    clfs = {}
    # now, compute the intrinsic scores.
    for lang_code, dsets in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        print(f"Predicting {lang_code}...")

        for dataset_name, dataset in dsets["sentence"].items():
            if not os.path.exists(Constants.CACHE_DIR / "ted2020" / save_str / lang_code / dataset_name):
                os.makedirs(Constants.CACHE_DIR / "ted2020" / save_str / lang_code / dataset_name)
            for supervision in ["u", "t", "punct"]:
                if not os.path.exists(
                    Constants.CACHE_DIR / "ted2020" / save_str / lang_code / dataset_name / supervision
                ):
                    os.makedirs(Constants.CACHE_DIR / "ted2020" / save_str / lang_code / dataset_name / supervision)
            if "surprise" not in dataset_name:
                continue
            sentences = dataset["data"][: args.max_n_test_sentences]
            # check if f[lang_code][dataset_name] exists
            if lang_code not in f or dataset_name not in f[lang_code]:
                continue

            if "train_logits" in f[lang_code][dataset_name] and not args.skip_adaptation and "surprise" in dataset_name:
                feature_indices = None
                clf = train_mixture(
                    [lang_code],
                    f[lang_code][dataset_name]["train_logits"][:],
                    f[lang_code][dataset_name]["train_labels"][:],
                    features=feature_indices,
                    skip_punct=args.skip_punct,
                )
                if clf[0] is not None:
                    print(clf)

                clf = list(copy.deepcopy(clf))
                # 1 clf for each lang: train data is same for both.
                clfs[lang_code] = clf

            gt_dir = Constants.ROOT_DIR.parent / "data" / "sepp_nlg_2021_data" / lang_code / dataset_name
            test_files = sorted([f for f in gt_dir.glob("*.tsv") if f.is_file()])

            if isinstance(sentences[0], list):
                for i, short_seq in enumerate(sentences):
                    start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                    current_logits = f[lang_code][dataset_name]["test_logits"][:][start : end + 1]
                    current_tokens = tokenizer.convert_ids_to_tokens(
                        f[lang_code][dataset_name]["test_tokens"][:][start : end + 1]
                    )

                    with open(test_files[i], "r", encoding="utf-8") as file:
                        idx = test_files[i].name.split(".")[0]
                        lines = file.read().strip().split("\n")
                        rows = [line.split("\t") for line in lines]
                        gt_words = [row[0] for row in rows]
                        gt_labels = [row[1] for row in rows]
                        if gt_labels[-1] != "1":
                            print("0 label at end!")
                    u_preds = sigmoid(current_logits[:, 0]) > args.threshold
                    if not args.skip_adaptation:
                        t_preds = sigmoid(current_logits[:, 0]) > clfs[lang_code][-1]
                    else:
                        t_preds = None
                    if not args.skip_adaptation and not args.skip_punct:
                        punct_preds = clfs[lang_code][0].predict_proba(current_logits)[:, 1] > clf[2]
                    else:
                        punct_preds = None

                    # write to tsv as per the challenge reqs
                    # can then be evaluated via evaluate_sepp_nlg_subtask1.py
                    for supervision, preds in zip(["u", "t", "punct"], [u_preds, t_preds, punct_preds]):
                        if preds is None:
                            continue

                        word_labels = get_token_labels(current_tokens, gt_words, preds)

                        with open(
                            Constants.CACHE_DIR
                            / "ted2020"
                            / save_str
                            / lang_code
                            / dataset_name
                            / supervision
                            / f"{idx}.tsv",
                            "w",
                            encoding="utf-8",
                        ) as file:
                            for word, label in zip(gt_words, word_labels):
                                file.write(f"{word}\t{label}\n")
            else:
                raise NotImplementedError
    return total_test_time, save_str


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    total_test_time, save_str = main(args)
    print(total_test_time)
    supervision = ["u", "t", "punct"]
    if args.skip_adaptation:
        supervision.remove("t")
    if args.skip_punct:
        supervision.remove("punct")
    if not args.include_langs:
        include_langs = ["en", "de", "fr", "it"]
    else:
        include_langs = args.include_langs
    if not args.include_splits:
        include_splits = ["surprise_test", "test"]
    else:
        include_splits = args.include_splits
    results = evaluate_subtask1(include_splits, include_langs, save_str, supervision, args.max_n_test_sentences)
