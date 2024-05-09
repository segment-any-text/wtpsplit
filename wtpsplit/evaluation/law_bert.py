import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List
import warnings

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser, pipeline

from wtpsplit.evaluation import evaluate_mixture
from wtpsplit.utils import Constants, corrupt_asr

logger = logging.getLogger()
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning)


def corrupt_document(document, lang):
    """Corrupt sentences in a document for ASR simulation."""
    return [corrupt_asr(sentence, lang=lang)[0] for sentence in document]


def process_documents(documents, lang):
    """Process each document and return a list of corrupted documents."""
    return [corrupt_document(document, lang) for document in documents]


def handle_legal_data(eval_data, lang_code):
    """Process legal data for a specific language, corrupting the sentences."""
    sections = ["test", "train"]
    corrupted_sentences = {}

    for section in sections:
        key = "data" if section == "test" else "meta"
        if key in eval_data[lang_code]["sentence"]["legal-data"].keys():
            original_data = eval_data[lang_code]["sentence"]["legal-data"][key]
            documents = original_data if section == "test" else original_data["train_data"]
            if not documents:
                corrupted_sentences[section] = None
                continue
            corrupted_docs = process_documents(documents, lang=lang_code.split("_")[0])
            corrupted_sentences[section] = corrupted_docs

    if corrupted_sentences:
        eval_data[lang_code]["sentence"][f"legal-data-corrupted"] = {
            "data": corrupted_sentences.get("test", []),
            "meta": {
                "train_data": corrupted_sentences.get("train", []),
            },
        }
        print(f"Created corrupted legal data for {lang_code}")


@dataclass
class Args:
    eval_data_path: str = "data/all_data_04_05.pth"
    device: str = "cpu"
    include_langs: List[str] = None
    max_n_test_sentences: int = sys.maxsize
    stride: int = 32
    save_suffix: str = ""
    return_indices: bool = False
    type: str = "both"  # laws, judgements, both, specific
    lang_support: str = "multi"  # mono, multi
    corrupt_legal: bool = False  # FIXME


def get_law_preds(texts, model, model_name, args) -> List[List[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        aggregation_strategy="simple",
        stride=args.stride,
    )
    sentences = pipe(texts)
    sent_end_preds_all = []
    for text, sent_preds in zip(texts, sentences):
        sent_end_indices = [short_seq["end"] - 1 for short_seq in sent_preds]
        # indices to binary list
        sent_end_preds = [1 if i in sent_end_indices else 0 for i in range(len(text))]
        sent_end_preds_all.append(sent_end_preds)
    return sent_end_preds_all


def load_or_compute_logits(args, eval_data, save_str: str = None):
    logits_path = Constants.CACHE_DIR / "law_bert" / f"{save_str}.h5"
    base_name = "rcds/distilbert-SBD"

    if not os.path.exists(Constants.CACHE_DIR / "law_bert"):
        os.makedirs(Constants.CACHE_DIR / "law_bert")

    use_langs = eval_data.keys()
    # law eval data is only one with _
    use_langs = [lang_code for lang_code in use_langs if "laws" in lang_code or "judgements" in lang_code]

    # create corrupted versions of legal data (unlike others, not contained in eval_data)
    if args.corrupt_legal:
        for lang_code in use_langs:
            if "laws" in lang_code or "judgements" in lang_code:
                handle_legal_data(eval_data, lang_code)
        torch.save(eval_data, args.eval_data_path)

    total_test_time = 0  # Initialize total test processing time

    with h5py.File(logits_path, "w") as f, torch.no_grad():
        for lang_code in tqdm(use_langs, desc="Languages"):
            current_name = base_name
            if args.lang_support == "multi":
                current_name += "-fr-es-it-en-de"
            elif args.lang_support == "mono":
                if lang_code.split("_")[0] == "pt":
                    current_name += "-fr-es-it-en-de"
                else:
                    current_name += f"-{lang_code.split('_')[0]}"
                if lang_code.split("_")[0] == "en":
                    current_name += "-judgements-laws"
            else:
                raise NotImplementedError
            if lang_code.split("_")[0] == "en" and args.lang_support == "mono":
                pass
            elif args.type == "laws":
                current_name += "-laws"
            elif args.type == "judgements":
                current_name += "-judgements"
            elif args.type == "both":
                current_name += "-judgements-laws"
            elif args.type == "specific":
                current_name += f"-{lang_code.split('_')[1]}"
            else:
                raise NotImplementedError

            model = AutoModelForTokenClassification.from_pretrained(current_name).to(args.device)

            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            print(f"{lang_code}, model: {current_name}")
            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]

            # eval data
            for dataset_name, dataset in tqdm(eval_data[lang_code]["sentence"].items(), desc=lang_code):
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"][: args.max_n_test_sentences]
                    if args.corrupt_legal:
                        test_sentences = [
                            corrupt_asr(sentence, lang=lang_code.split("_")[0]) for sentence in test_sentences
                        ]
                    if isinstance(test_sentences[0], list):
                        # short-seq eval: list of lists
                        test_text = [
                            Constants.SEPARATORS.get(lang_code, " ").join(sentence) for sentence in test_sentences
                        ]
                    else:
                        raise NotImplementedError

                    start_time = time.time()  # Start timing for test logits processing
                    test_logits = get_law_preds(test_text, model, current_name, args)
                    end_time = time.time()  # End timing for test logits processing
                    total_test_time += end_time - start_time  # Accumulate test processing time
                    if isinstance(test_sentences[0], list):
                        test_logit_lengths = []
                        # store start and end indices for each pair, used later to slice the logits
                        # (h5py does not like different length np arrays as list elements)
                        all_logit_lengths = np.append(0, np.cumsum([len(logits) for logits in test_logits]))
                        # append tuple of start and end indices for each pair
                        for i in range(len(test_logits)):
                            test_logit_lengths.append((all_logit_lengths[i], all_logit_lengths[i + 1] - 1))
                        test_logits = np.concatenate(test_logits)
                        test_logits = np.expand_dims(test_logits, axis=1)
                        dset_group.create_dataset("test_logit_lengths", data=test_logit_lengths)
                    else:
                        raise NotImplementedError

                    dset_group.create_dataset("test_logits", data=test_logits)

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
    save_model_path = f"rcds/distilbert-SBD-{args.lang_support}-{args.type}_s{args.stride}"
    save_str = f"{save_model_path.replace('/','_')}"

    eval_data = torch.load(args.eval_data_path)

    if args.corrupt_legal:
        save_str += "_corrupt"

    save_str += f"{args.save_suffix}"

    # first, logits for everything.
    f, total_test_time = load_or_compute_logits(args, eval_data, save_str)

    # now, compute the law_bert scores.
    results = {}
    clfs = {}
    if args.return_indices:
        indices = {}
    # Initialize lists to store scores for each metric across all languages
    u_scores = []

    for lang_code, dsets in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        print(f"Predicting {lang_code}...")
        results[lang_code] = {}
        clfs[lang_code] = {}
        if args.return_indices:
            indices[lang_code] = {}

        for dataset_name, dataset in dsets["sentence"].items():
            sentences = dataset["data"][: args.max_n_test_sentences]
            # check if f[lang_code][dataset_name] exists
            if lang_code not in f or dataset_name not in f[lang_code]:
                continue

            clf = [None, None, None, 0.5]

            if isinstance(sentences[0], list):
                acc_u = []
                score_u = []
                u_indices, true_indices = [], []
                length = 0
                for i, short_seq in enumerate(sentences):
                    start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                    single_score_u, _, info, cur_u_indices, _ = evaluate_mixture(
                        lang_code,
                        f[lang_code][dataset_name]["test_logits"][:][start:end],
                        list(short_seq),
                        args.return_indices,
                        *clf,
                    )
                    score_u.append(single_score_u)
                    acc_u.append(info["info_newline"]["correct_pairwise"])
                    # indices: accumulate from start
                    u_indices.extend(
                        [idx + start for idx in cur_u_indices["pred_indices"]] if cur_u_indices["pred_indices"] else []
                    )
                    true_indices.extend(
                        [idx + start for idx in cur_u_indices["true_indices"]] if cur_u_indices["true_indices"] else []
                    )
                    length += cur_u_indices["length"] - 1

            else:
                raise NotImplementedError

            if isinstance(sentences[0], list):
                score_u = np.mean(score_u)
                acc_u = np.mean(acc_u)

                results[lang_code][dataset_name] = {
                    "u": score_u,
                    "acc_u": acc_u,
                }
            else:
                raise NotImplementedError

            if args.return_indices:
                if isinstance(sentences[0], list):
                    indices[lang_code][dataset_name] = {
                        "u": u_indices,
                        "true_indices": true_indices,
                        "length": length,
                    }
                else:
                    raise NotImplementedError

            if score_u is not None:
                u_scores.append((score_u, lang_code))

            # just for printing
            print(f"{lang_code} {dataset_name} {score_u:.3f}")

    # Compute statistics for each metric across all languages
    results_avg = {
        "u": compute_statistics(u_scores),
        "include_langs": args.include_langs,
    }

    json.dump(
        results,
        open(
            Constants.CACHE_DIR / "law_bert" / f"{save_str}.json",
            "w",
        ),
        indent=4,
    )
    print(Constants.CACHE_DIR / "law_bert" / f"{save_str}.json")

    # Write results_avg to JSON
    json.dump(
        results_avg,
        open(
            Constants.CACHE_DIR / "law_bert" / f"{save_str}_AVG.json",
            "w",
        ),
        indent=4,
    )
    if args.return_indices:
        json.dump(
            indices,
            open(
                Constants.CACHE_DIR / "law_bert" / f"{save_str}_IDX.json",
                "w",
            ),
            default=int,
            # indent=4,
        )
        print(Constants.CACHE_DIR / "law_bert" / f"{save_str}_IDX.json")
        print("Indices saved to file.")

    return results, results_avg, total_test_time


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    results, results_avg, total_test_time = main(args)
    print(total_test_time)
