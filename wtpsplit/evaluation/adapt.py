import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Union

import h5py
import numpy as np
import skops.io as sio
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser

import adapters
import wtpsplit.models  # noqa: F401
from wtpsplit.evaluation import evaluate_mixture, get_labels, train_mixture
from wtpsplit.evaluation.intrinsic_baselines import split_language_data
from wtpsplit.extract import PyTorchWrapper, extract
from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
from wtpsplit.utils import Constants, token_to_char_probs

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


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
    eval_data_path: str = "data/all_data.pth"
    valid_text_path: str = None  # "data/sentence/valid.parquet"
    device: str = "cpu"
    block_size: int = 512
    stride: int = 64
    batch_size: int = 32
    include_langs: List[str] = None
    custom_language_list: str = None
    threshold: float = 0.01
    max_n_train_sentences: int = 10000
    max_n_test_sentences: int = -1  # -1 is all
    keep_logits: bool = False
    skip_adaptation: bool = False
    skip_punct: bool = True
    skip_corrupted: bool = False
    clf_from_scratch: bool = False  # for FT + LoRA
    return_indices: bool = True
    exclude_every_k: int = 10
    save_suffix: str = ""
    num_hidden_layers: Union[int, None] = None  # for original XLM-R


def process_logits(text, model, lang_code, args):
    # Extract necessary data
    if isinstance(text, list):
        logits = []
        for short_seq in tqdm(text, desc="Listwise", disable=False):
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

            if "xlm" in model.config.model_type:
                tokens = tokenizer.tokenize(short_seq, verbose=False)

                char_probs = token_to_char_probs(
                    short_seq,
                    tokens,
                    current_logits,
                    [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token],
                    current_offsets_mapping,
                )

                current_logits = char_probs

            logits.append(current_logits)
    else:
        logits, offsets_mapping, tokenizer, _ = extract(
            [text],
            model,
            lang_code=lang_code,
            stride=args.stride,
            max_block_size=args.block_size,
            batch_size=args.batch_size,
            pad_last_batch=True,
            verbose=False,
        )
        logits = logits[0]
        if offsets_mapping is not None:
            offsets_mapping = offsets_mapping[0]

        if "xlm" in model.config.model_type:
            tokens = tokenizer.tokenize(text, verbose=False)

            special_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]

            # convert token probabilities to character probabilities for the entire array
            char_probs = token_to_char_probs(text, tokens, logits, special_tokens, offsets_mapping)

            logits = char_probs

    return logits


def load_or_compute_logits(args, model, eval_data, valid_data=None, save_str: str = None):
    logits_path = Constants.CACHE_DIR / "intrinsic" / f"{save_str}.h5"

    if not os.path.exists(Constants.CACHE_DIR / "intrinsic"):
        os.makedirs(Constants.CACHE_DIR / "intrinsic")

    if args.custom_language_list is not None:
        with open(args.custom_language_list, "r") as f:
            # file is a csv: l1,l2,...
            use_langs = f.read().strip().split(",")
    else:
        use_langs = eval_data.keys()

    total_test_time = 0

    with h5py.File(logits_path, "a") as f, torch.no_grad():
        for lang_code in tqdm(use_langs, desc="Languages"):
            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]

            # valid data
            if valid_data is not None and "valid" not in lang_group:
                if args.adapter_path:
                    raise NotImplementedError("Adapters not supported for valid data")
                valid_sentences = [sample["text"].strip() for sample in valid_data if sample["lang"] == lang_code]
                assert len(valid_sentences) > 0

                separator = Constants.SEPARATORS.get(lang_code, " ")
                valid_text = separator.join(valid_sentences)

                valid_logits = process_logits(valid_text, model, lang_code, args)

                lang_group.create_dataset("valid", data=valid_logits)

            # eval data
            for dataset_name, dataset in tqdm(eval_data[lang_code]["sentence"].items(), desc=lang_code):
                if args.skip_corrupted and "corrupted" in dataset_name:
                    continue
                if "asr" in dataset_name and not any(
                    x in dataset_name for x in ["lyrics", "short", "code", "ted2020", "legal"]
                ):
                    logger.warning(f"SKIP: {lang_code} {dataset_name}")
                    continue
                if "legal" in dataset_name and not ("laws" in dataset_name or "judgements" in dataset_name):
                    logger.warning(f"SKIP: {lang_code} {dataset_name}")
                    continue
                if "social-media" in dataset_name:
                    logger.warning(f"SKIP: {lang_code} {dataset_name}")
                    continue
                if "nllb" in dataset_name:
                    continue
                if "-" in lang_code and "canine" in args.model_path and "no-adapters" not in args.model_path:
                    # code-switched data: eval 2x
                    lang_code = lang_code.split("_")[1].lower()
                try:
                    if args.adapter_path:
                        if args.clf_from_scratch:
                            model.model.classifier = torch.nn.Linear(model.model.classifier.in_features, 1)

                        dataset_load_name = dataset_name
                        model.model.load_adapter(
                            args.adapter_path + "/" + dataset_load_name + "/" + lang_code,
                            set_active=True,
                            with_head=True,
                            load_as="text",
                        )
                except Exception as e:
                    logger.error(f"Error loading adapter for {dataset_name} in {lang_code}: {e}")
                    continue
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"]
                    if not test_sentences:
                        continue
                    max_n_sentences = args.max_n_test_sentences
                    test_sentences = test_sentences[:max_n_sentences]
                    if isinstance(test_sentences[0], list):
                        # short-seq eval: list of lists
                        test_text = [
                            Constants.SEPARATORS.get(lang_code, " ").join(sentence) for sentence in test_sentences
                        ]
                    else:
                        test_text = Constants.SEPARATORS.get(lang_code, " ").join(test_sentences)

                    start_time = time.time()
                    test_logits = process_logits(test_text, model, lang_code, args)
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
                        # NOTE: handled differently than in intrinsic_pairwise.py
                        # here, we keep the label at the end
                        # in intrinsic_pairwise.py, we only consider non-ending labels.
                        test_labels = [
                            get_labels(lang_code, short_seq, after_space=False)[:-1] for short_seq in test_sentences
                        ]

                        # flatten; append 0 eos to account for later indexing/slicing
                        test_labels = np.append(np.concatenate(test_labels), 1)
                        assert len(test_labels) == len(test_logits) + 1
                        dset_group.create_dataset("test_logit_lengths", data=test_logit_lengths)
                    else:
                        test_labels = get_labels(lang_code, test_sentences, after_space=False)
                    if args.skip_punct:
                        # remove punct logits
                        test_logits = test_logits[:, 0]
                        # back to [N, 1]
                        test_logits = np.expand_dims(test_logits, axis=1)
                    dset_group.create_dataset("test_logits", data=test_logits)
                    dset_group.create_dataset("test_labels", data=test_labels)

                train_sentences = dataset["meta"].get("train_data")
                if train_sentences is not None and "train_logits" not in dset_group and not args.skip_adaptation:
                    train_sentences = train_sentences[: args.max_n_train_sentences]
                    if isinstance(train_sentences[0], list):
                        # short-seq eval: list of lists
                        train_text = [
                            Constants.SEPARATORS.get(lang_code, " ").join(sentence) for sentence in train_sentences
                        ]
                    else:
                        train_text = Constants.SEPARATORS.get(lang_code, " ").join(train_sentences)

                    train_logits = process_logits(train_text, model, lang_code, args)
                    if isinstance(train_sentences[0], list):
                        train_logits = np.concatenate(train_logits)
                        train_labels = [
                            get_labels(lang_code, short_seq, after_space=False)[:-1] for short_seq in train_sentences
                        ]

                        # flatten; append 0 eos to account for later indexing/slicing
                        train_labels = np.append(np.concatenate(train_labels), 1)
                        assert len(train_labels) == len(train_logits) + 1
                    else:
                        train_labels = get_labels(lang_code, train_sentences, after_space=False)

                    if args.skip_punct:
                        # remove punct logits
                        train_logits = train_logits[:, 0]
                        # back to [N, 1]
                        train_logits = np.expand_dims(train_logits, axis=1)
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
    save_str = f"{save_model_path.replace('/', '_')}_b{args.block_size}_s{args.stride}"

    eval_data = torch.load(args.eval_data_path)
    if "canine" in args.model_path and "no-adapters" not in args.model_path:
        eval_data = split_language_data(eval_data)
    if args.valid_text_path is not None:
        valid_data = load_dataset("parquet", data_files=args.valid_text_path, split="train")
    else:
        valid_data = None

    logger.warning("Loading model...")
    model_path = args.model_path
    if args.model_path == "xlm-roberta-base" or args.model_path == "xlm-roberta-large":
        # init models here
        config = SubwordXLMConfig.from_pretrained(
            args.model_path,
            num_hidden_layers=args.num_hidden_layers,
            num_labels=1,
        )
        model = PyTorchWrapper(
            SubwordXLMForTokenClassification.from_pretrained(model_path, config=config).to(args.device)
        )
    else:
        model = PyTorchWrapper(AutoModelForTokenClassification.from_pretrained(model_path).to(args.device))
    if args.adapter_path:
        model_type = model.model.config.model_type
        # adapters need xlm-roberta as model type.
        model.model.config.model_type = "xlm-roberta"
        adapters.init(model.model)
        # reset model type (used later)
        model.model.config.model_type = model_type

    save_str += f"{args.save_suffix}"
    if args.max_n_test_sentences < sys.maxsize and args.max_n_test_sentences != -1:
        save_str += f"_n{args.max_n_test_sentences}"
    if args.max_n_test_sentences == -1:
        args.max_n_test_sentences = sys.maxsize

    # first, logits for everything.
    f, total_test_time = load_or_compute_logits(args, model, eval_data, valid_data, save_str)

    save_str += f"_u{args.threshold}"
    if args.exclude_every_k > 0 or "lyrics" in args.eval_data_path:
        save_str += f"_k{args.exclude_every_k}"

    # now, compute the intrinsic scores.
    results = {}
    clfs = {}
    if args.return_indices:
        indices = {}

    u_scores, t_scores, punct_scores = [], [], []

    for lang_code, dsets in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        logger.warning(f"Predicting {lang_code}...")
        results[lang_code] = {}
        clfs[lang_code] = {}
        if args.return_indices:
            indices[lang_code] = {}

        for dataset_name, dataset in dsets["sentence"].items():
            sentences = dataset["data"]
            if not sentences:
                continue
            max_n_sentences = args.max_n_test_sentences
            sentences = sentences[:max_n_sentences]
            if len(sentences) == 0:
                continue
            if lang_code not in f or dataset_name not in f[lang_code]:
                continue

            # to be in line w/ LLM eval; for fair comparison
            if "lyrics" in dataset_name or "short" in dataset_name:
                exclude_every_k = 0
            else:
                exclude_every_k = args.exclude_every_k

            if "train_logits" in f[lang_code][dataset_name] and not args.skip_adaptation:
                feature_indices = None
                clf = train_mixture(
                    [lang_code],
                    f[lang_code][dataset_name]["train_logits"][:],
                    f[lang_code][dataset_name]["train_labels"][:],
                    features=feature_indices,
                    skip_punct=args.skip_punct,
                )
                if clf[0] is not None:
                    logger.warning(clf)

                if isinstance(sentences[0], list):
                    acc_t, acc_punct = [], []
                    score_t, score_punct = [], []
                    t_indices, punct_indices = [], []
                    for i, short_seq in enumerate(sentences):
                        start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                        single_score_t, single_score_punct, info, cur_t_indices, cur_punct_indices = evaluate_mixture(
                            lang_code,
                            f[lang_code][dataset_name]["test_logits"][:][start:end],
                            list(short_seq),
                            args.return_indices,
                            exclude_every_k,
                            *clf,
                        )
                        score_t.append(single_score_t)
                        score_punct.append(single_score_punct)
                        acc_t.append(info["info_newline"]["correct_pairwise"] if info["info_newline"] else None)
                        acc_punct.append(
                            info["info_transformed"]["correct_pairwise"] if info["info_transformed"] else None
                        )
                        # indices: accumulate from start
                        t_indices.append(
                            cur_t_indices["pred_indices"] if cur_t_indices and cur_t_indices["pred_indices"] else []
                        )
                        punct_indices.append(
                            cur_punct_indices["pred_indices"]
                            if cur_punct_indices and cur_punct_indices["pred_indices"]
                            else []
                        )

                else:
                    score_t, score_punct, _, t_indices, punct_indices = evaluate_mixture(
                        lang_code,
                        f[lang_code][dataset_name]["test_logits"][:],
                        sentences,
                        args.return_indices,
                        exclude_every_k,
                        *clf,
                    )

                clfs[lang_code][dataset_name] = clf

                clf = list(copy.deepcopy(clf))
                clf[-1] = args.threshold
            else:
                score_t = score_punct = None
                clf = [None, None, None, args.threshold]
                t_indices, punct_indices = None, None

            if isinstance(sentences[0], list):
                acc_u = []
                score_u = []
                u_indices, true_indices = [], []
                length = []
                for i, short_seq in enumerate(sentences):
                    start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                    single_score_u, _, info, cur_u_indices, _ = evaluate_mixture(
                        lang_code,
                        f[lang_code][dataset_name]["test_logits"][:][start:end],
                        list(short_seq),
                        args.return_indices,
                        exclude_every_k,
                        *clf,
                    )
                    score_u.append(single_score_u)
                    acc_u.append(info["info_newline"]["correct_pairwise"])
                    # indices: accumulate from start
                    u_indices.append(cur_u_indices["pred_indices"] if cur_u_indices["pred_indices"] else [])
                    true_indices.append(cur_u_indices["true_indices"] if cur_u_indices["true_indices"] else [])
                    length.append(cur_u_indices["length"])

            else:
                score_u, _, _, u_indices, _ = evaluate_mixture(
                    lang_code,
                    f[lang_code][dataset_name]["test_logits"][:],
                    sentences,
                    args.return_indices,
                    exclude_every_k,
                    *clf,
                )

            if isinstance(sentences[0], list):
                score_u = np.mean(score_u)
                score_t = np.mean(score_t) if score_t and not args.skip_adaptation else None
                score_punct = (
                    np.mean(score_punct) if score_punct and not (args.skip_punct or args.skip_adaptation) else None
                )
                acc_u = np.mean(acc_u)
                acc_t = np.mean(acc_t) if score_t else None
                acc_punct = np.mean(acc_punct) if score_punct else None

                results[lang_code][dataset_name] = {
                    "u": score_u,
                    "t": score_t,
                    "punct": score_punct,
                    "acc_u": acc_u,
                    "acc_t": acc_t,
                    "acc_punct": acc_punct,
                }
            else:
                results[lang_code][dataset_name] = {
                    "u": score_u,
                    "t": score_t,
                    "punct": score_punct,
                }

            if args.return_indices:
                if isinstance(sentences[0], list):
                    indices[lang_code][dataset_name] = {
                        "u": {"predicted_indices": u_indices, "true_indices": true_indices, "length": length},
                        "t": {"predicted_indices": t_indices, "true_indices": t_indices, "length": length}
                        if t_indices
                        else None,
                        "punct": {"predicted_indices": punct_indices, "true_indices": t_indices, "length": length}
                        if punct_indices
                        else None,
                    }
                else:
                    indices[lang_code][dataset_name] = {
                        "u": {
                            "predicted_indices": [u_indices["pred_indices"]],
                            "true_indices": [u_indices["true_indices"]],
                            "length": [u_indices["length"]],
                        },
                        "t": {
                            "predicted_indices": [t_indices["pred_indices"]],
                            "true_indices": [t_indices["true_indices"]],
                            "length": [t_indices["length"]],
                        }
                        if t_indices is not None
                        else None,
                        "punct": {
                            "predicted_indices": [punct_indices["pred_indices"]],
                            "true_indices": [punct_indices["true_indices"]],
                            "length": [punct_indices["length"]],
                        }
                        if punct_indices is not None
                        else None,
                    }

            if score_u is not None:
                u_scores.append((score_u, lang_code))
            if score_t is not None:
                t_scores.append((score_t, lang_code))
            if score_punct is not None:
                punct_scores.append((score_punct, lang_code))

            # just for logging
            score_t = score_t or 0.0
            score_punct = score_punct or 0.0
            logger.warning(f"{lang_code} {dataset_name} {score_u:.3f} {score_t:.3f} {score_punct:.3f}")

    # Compute statistics for each metric across all languages
    results_avg = {
        "u": compute_statistics(u_scores),
        "t": compute_statistics(t_scores),
        "punct": compute_statistics(punct_scores),
        "include_langs": args.include_langs,
    }

    if not args.skip_adaptation:
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
    print(Constants.CACHE_DIR / "intrinsic" / f"{save_str}.json")

    json.dump(
        results_avg,
        open(
            Constants.CACHE_DIR / "intrinsic" / f"{save_str}_AVG.json",
            "w",
        ),
        indent=4,
    )
    if args.return_indices:
        json.dump(
            indices,
            open(
                Constants.CACHE_DIR / "intrinsic" / f"{save_str}_IDX.json",
                "w",
            ),
            default=int,
            indent=4,
        )
        print(Constants.CACHE_DIR / "intrinsic" / f"{save_str}_IDX.json")
        print("Indices saved to file.")

    if not args.keep_logits:
        os.remove(f.filename)
    return results, results_avg, total_test_time


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    results, results_avg, total_test_time = main(args)
    print(total_test_time)
