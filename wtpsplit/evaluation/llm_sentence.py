import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List
import re
import sys

import numpy as np
import h5py
import pandas as pd
import torch
from genalog.text import alignment
from pandarallel import pandarallel
from tqdm import tqdm
from transformers import HfArgumentParser
import cohere
import replicate

from wtpsplit.evaluation import get_labels, evaluate_sentences_llm
from wtpsplit.evaluation.intrinsic_pairwise import generate_k_mers
from wtpsplit.utils import Constants
import time

pandarallel.initialize(progress_bar=True, nb_workers=32)

logging.getLogger().setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "Separate the following text into sentences by adding a newline between each sentence. "
    "Do not modify the text in any way and keep the exact ordering of words! "
    "If you modify it, remove or add anything, you get fined $1000 per word. "
    "Provide a concise answer without any introduction. "
    "Indicate sentence boundaries only via a single newline, no more than this! "
)

LYRICS_PROMPT = (
    "Separate the following song's lyrics into semantic units "
    "(e.g., verse, chorus, bridge, intro/outro, etc - "
    "similar to how they are presented in a lyrics booklet) "
    "via double newlines, but do not annotate them. "
    "Only include the song in the output, no annotations. "
    "Do not modify the song in any way and keep the exact ordering of words! "
    "If you modify it, remove or add anything, you get fined $1000 per word. "
    "Indicate semantic units by double newlines. "
)


@dataclass
class Args:
    eval_data_path: str = "data/all_data_11_05"
    type: str = "lyrics"  # all, lyrics, pairs, short_proc
    llm_provider: str = "cohere"  # cohere, replicate
    label_delimiter: str = "|"  # NOT \n or \n\n
    gap_char = "@"
    # model: str = "mistralai/mixtral-8x7b-instruct-v0.1"
    # model: str = "meta/meta-llama-3-8b-instruct"
    model: str = "command-r"
    save_suffix: str = "Pv2"
    include_langs: List[str] = None
    custom_language_list: str = None
    max_n_test_sentences: int = -1
    k: int = 10
    n_shots: int = 0


def replicate_provider(text, train_data, lang_code, args):
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    llm_prompt = prompt_factory(text, train_data, lang_code, args)
    # print(llm_prompt)
    n_tries = 0
    while n_tries < 1:
        try:
            llm_input = {
                "system_prompt": "",
                "prompt": llm_prompt,
                # "max_new_tokens": 50_000,
                "max_tokens": 4000,
            }
            llm_output = api.run(args.model, llm_input)
            llm_output = "".join(llm_output)
            # print(llm_output)
            return llm_output
        except Exception as e:
            n_tries += 1
            print(e)
            time.sleep(10)
            continue
    return ""


def cohere_provider(text, train_data, lang_code, args):
    api = cohere.Client(os.environ["COHERE_API_TOKEN"])
    llm_prompt = prompt_factory(text, train_data, lang_code, args)
    n_tries = 0
    while True:
        try:
            llm_output = api.chat(
                model=args.model, preamble="", message=llm_prompt, max_tokens=4000, seed=42
            ).text.replace("\\n", "\n")
            return llm_output
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code and str(e.status_code)[0] == "4":
                print("API refusal.")
                llm_output = ""
                return llm_output
            elif status_code and str(e.status_code)[0] == "5":
                # Cohere API issue: retry
                n_tries += 1
                time.sleep(10)
                continue
            else:
                raise e


def cohere_tokenize(text, args):
    api = cohere.Client(os.environ["COHERE_API_TOKEN"])
    tokens = api.tokenize(text=text, model=args.model, offline=False).token_strings
    return tokens


def load_h5_to_dataframe(filepath, args):
    with h5py.File(filepath, "r") as h5file:
        all_data = []

        for lang_code, lang_group in h5file.items():
            for dataset_name, dset_group in lang_group.items():
                if (
                    "test_preds" in dset_group
                    and "test_chunks" in dset_group
                    and not any("0" in key for key in dset_group.keys())
                ):
                    test_preds_data = dset_group["test_preds"].asstr()[:]
                    test_chunks_data = dset_group["test_chunks"].asstr()[:]

                    if len(test_preds_data) != len(test_chunks_data):
                        raise ValueError("Mismatched lengths between test_preds and test_chunks.")

                    # append  item with metadata
                    for test_pred, test_chunk in zip(test_preds_data, test_chunks_data):
                        all_data.append(
                            {
                                "lang": lang_code,
                                "dataset_name": dataset_name,
                                "test_preds": test_pred,
                                "test_chunks": test_chunk.tolist(),
                                "doc_id": -1,
                            }
                        )
                elif any("0" in key for key in dset_group.keys()):
                    # each list is saved with x_i, x_i+1, ...
                    doc_ids = [key.split("_")[-1] for key in dset_group.keys() if "test_preds" in key]
                    for doc_id in doc_ids:
                        test_preds_data = dset_group[f"test_preds_{doc_id}"].asstr()[:]
                        test_chunks_data = dset_group[f"test_chunks_{doc_id}"].asstr()[:]

                        if len(test_preds_data) != len(test_chunks_data):
                            raise ValueError("Mismatched lengths between test_preds and test_chunks.")

                        # append item with metadata
                        for test_pred, test_chunk in zip(test_preds_data, test_chunks_data):
                            all_data.append(
                                {
                                    "lang": lang_code,
                                    "dataset_name": dataset_name,
                                    "test_preds": test_pred,
                                    "test_chunks": test_chunk.tolist(),
                                    "doc_id": int(doc_id),
                                }
                            )
                else:
                    pass

        df = pd.DataFrame(all_data)
        return df


def load_or_compute_logits(args, eval_data, save_str: str = None):
    logits_dir = Constants.CACHE_DIR / "llm_sent" / "preds"
    logits_dir.mkdir(parents=True, exist_ok=True)
    logits_path = logits_dir / f"{save_str}.h5"

    if args.custom_language_list is not None:
        with open(args.custom_language_list, "r") as f:
            # file is a csv: l1,l2,...
            use_langs = f.read().strip().split(",")
    else:
        use_langs = eval_data.keys()

    with h5py.File(logits_path, "a") as f:
        for lang_code in tqdm(use_langs, desc="Languages"):
            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            print(f"Processing {lang_code}...")
            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]
            for dataset_name, dataset in tqdm(eval_data[lang_code]["sentence"].items(), desc=lang_code):
                if "corrupted-asr" in dataset_name and (
                    "lyrics" not in dataset_name
                    and "short" not in dataset_name
                    and "code" not in dataset_name
                    and "ted" not in dataset_name
                    and "legal" not in dataset_name
                ):
                    print("SKIP: ", lang_code, dataset_name)
                    continue
                if "legal" in dataset_name and not ("laws" in dataset_name or "judgements" in dataset_name):
                    print("SKIP: ", lang_code, dataset_name)
                    continue
                if "social-media" in dataset_name:
                    continue
                if "nllb" in dataset_name:
                    continue
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                if args.type == "pairs" and dataset_name != "ersatz" and dataset_name != "ted2020-corrupted-asr":
                    continue
                if (args.k != 10 or args.n_shots != 0) and dataset_name != "ersatz":
                    print("SKIP: ", lang_code, dataset_name)
                    continue
                else:
                    dset_group = lang_group[dataset_name]
                if "test_preds" not in dset_group and "test_preds_0" not in dset_group:
                    test_sentences = dataset["data"]
                    if not test_sentences:
                        continue
                    if (
                        isinstance(test_sentences[0], list)
                        and "lyrics" not in dataset_name
                        and "short" not in dataset_name
                        and args.type != "pairs"
                    ):
                        # documents: only 10% of documents. 1000 sentences --> 100 docs
                        max_n_sentences = args.max_n_test_sentences // 10
                        # shuffle docs
                        np.random.seed(42)
                        test_sentences = np.random.permutation(test_sentences).tolist()
                    else:
                        max_n_sentences = args.max_n_test_sentences
                    test_sentences = test_sentences[:max_n_sentences]
                    if isinstance(test_sentences[0], list) or args.type == "pairs":
                        if args.type == "pairs":
                            all_pairs = generate_k_mers(
                                test_sentences,
                                k=2,
                                do_lowercase=False,
                                do_remove_punct=False,
                                sample_pct=0.5
                            )
                            test_sentences = all_pairs
                        # list of lists: chunk each sublist
                        if "short" in dataset_name or "lyrics" in dataset_name or args.type == "pairs":
                            # only here: no chunking
                            test_chunks = test_sentences
                            test_texts = [
                                [Constants.SEPARATORS[lang_code].join(test_chunk).strip()] for test_chunk in test_chunks
                            ]
                        else:
                            test_chunks = [
                                [test_sentences[i][j : j + args.k] for j in range(0, len(test_sentences[i]), args.k)]
                                for i in range(len(test_sentences))
                            ]
                            # last batch for each sublist: pad with None to enable saving w/ h5py
                            for i in range(len(test_chunks)):
                                test_chunks[i][-1] += [""] * (args.k - len(test_chunks[i][-1]))
                            # join sentences in each chunk
                            test_texts = [
                                [
                                    Constants.SEPARATORS[lang_code].join(test_chunk).strip()
                                    for test_chunk in test_sublist
                                ]
                                for test_sublist in test_chunks
                            ]
                        if args.n_shots:
                            train_sentences = eval_data[lang_code]["sentence"][dataset_name]["meta"]["train_data"][:100]
                            if train_sentences:
                                if "short" in dataset_name or args.type == "pairs":
                                    # here: entire samples (tweets e.g.)
                                    train_chunks = train_sentences
                                    train_texts = ["\n".join(train_chunk).strip() for train_chunk in train_chunks]
                                elif "lyrics" in dataset_name:
                                    # here: entire samples (songs)
                                    train_chunks = train_sentences
                                    train_texts = ["\n\n".join(train_chunk).strip() for train_chunk in train_chunks]
                                else:
                                    # flatten to have diversity among lengthy few-shot examples
                                    train_sentences = [item for sublist in train_sentences[:100] for item in sublist]
                                    train_chunks = [
                                        train_sentences[i : i + args.k] for i in range(0, len(train_sentences), args.k)
                                    ]
                                    train_texts = ["\n".join(train_chunk).strip() for train_chunk in train_chunks]
                            else:
                                train_texts = None
                        else:
                            train_texts = None
                        for i in tqdm(range(len(test_texts))):
                            test_preds = get_llm_preds(test_texts[i], train_texts, lang_code, args, verbose=False)
                            dset_group.create_dataset(f"test_preds_{i}", data=test_preds)
                            dset_group.create_dataset(
                                f"test_chunks_{i}",
                                data=[test_chunks[i]]
                                if "short" in dataset_name or "lyrics" in dataset_name or args.type == "pairs"
                                else test_chunks[i],
                            )

                    else:
                        test_chunks = [test_sentences[i : i + args.k] for i in range(0, len(test_sentences), args.k)]
                        # last batch: pad with None to enable saving w/ h5py
                        test_chunks[-1] += [""] * (args.k - len(test_chunks[-1]))
                        test_texts = [
                            Constants.SEPARATORS[lang_code].join(test_chunk).strip() for test_chunk in test_chunks
                        ]
                        if args.n_shots:
                            train_sentences = eval_data[lang_code]["sentence"][dataset_name]["meta"]["train_data"]
                            if train_sentences:
                                train_sentences = train_sentences[:100]
                                train_chunks = [
                                    train_sentences[i : i + args.k] for i in range(0, len(train_sentences), args.k)
                                ]
                                train_texts = ["\n".join(train_chunk).strip() for train_chunk in train_chunks]
                            else:
                                train_texts = None
                        else:
                            train_texts = None
                        test_preds = get_llm_preds(test_texts, train_texts, lang_code, args)
                        dset_group.create_dataset("test_preds", data=test_preds)
                        dset_group.create_dataset("test_chunks", data=test_chunks)
    return h5py.File(logits_path, "r")


def get_llm_preds(
    test_texts,
    train_data,
    lang_code,
    args,
    verbose=True,
):
    if args.llm_provider == "cohere":
        llm_provider = cohere_provider
    elif args.llm_provider == "replicate":
        llm_provider = replicate_provider
    else:
        raise ValueError(f"Unknown LLM provider: {args.llm_provider}")
    output = []
    for test_chunk in tqdm(test_texts, disable=not verbose):
        output.append(llm_provider(test_chunk, train_data, lang_code, args))
    return output


def prompt_factory(test_chunk, train_data, lang_code, args):
    n_shots = args.n_shots if train_data is not None else 0
    main_prompt = LYRICS_PROMPT if args.type == "lyrics" else SYSTEM_PROMPT

    prompt_start = (
        main_prompt
        + f"When provided with multiple examples, you are to respond only to the last one: Output {n_shots + 1}."
        if n_shots
        else main_prompt
    )

    llm_prompt = (
        prompt_start
        + "\n\n"
        + create_few_shot_prompt(train_data, lang_code, args)
        + (f"# Input {n_shots + 1}:\n\n" if n_shots else "# Input:\n\n")
        + test_chunk
        + (f"\n\n# Output {n_shots + 1}:\n\n" if n_shots else "\n\n# Output:\n\n")
    )
    return llm_prompt


def create_few_shot_prompt(train_data, lang_code, args):
    if train_data is None:
        return ""
    num_samples = min(args.n_shots, len(train_data))
    samples_prompt = ""
    counter = 1
    for _, sample in enumerate(train_data[:num_samples]):
        current_input = sample.replace("\n", Constants.SEPARATORS.get(lang_code, " "))
        samples_prompt += f"# Input {counter}:\n\n{current_input}\n\n# Output {counter}:\n\n{sample}\n\n"
        counter += 1
    return samples_prompt


def postprocess_llm_output(llm_output, lang):
    """Clean LLM output by removing specified characters and cleaning up lines."""
    if llm_output == "":
        # API refusal
        return ""
    llm_output = llm_output.strip(" -\n")

    # neither of them must be present in the output.
    llm_output = llm_output.replace(args.gap_char, " ")
    llm_output = llm_output.replace(args.label_delimiter, " ")
    llm_output = llm_output.replace("\n\n", args.label_delimiter)
    llm_output = llm_output.replace("\n", args.label_delimiter)
    # replace multiple newlines with 1
    llm_output = re.sub(r"\n+", "\n", llm_output)

    # remove leading #, # Input, :
    llm_output = llm_output.strip("#").strip().strip("Input").strip(":").strip()
    # remove trailing #, Output, .
    llm_output = llm_output.strip(":").strip("Output").strip().strip("#")
    # replace multiple occurences of label_delimiter with only 1
    llm_output = re.sub(r"{0}+".format(re.escape(args.label_delimiter)), args.label_delimiter, llm_output)

    # Split into lines, strip each line, remove empty lines, and join back into a single string
    llm_output = " ".join([line.strip() for line in llm_output.split("\n") if line.strip()])

    return llm_output


def align_llm_output(row):
    """Align input and output, including formatting."""
    try:
        aligned_in, aligned_llm = alignment.align(
            row["test_chunks"],
            row["test_preds"],
            gap_char=args.gap_char,
        )
        # same as aligned_in, aligned_llm, but with additional formatting. Latter used to debug only.
        formatted_alignment = alignment._format_alignment(aligned_in, aligned_llm).split("\n")
    except:    # ruff: ignore=E722
        print("Alignment failed: ", row.name)
        formatted_alignment = [row["test_chunks"], "", " " * len(row["test_chunks"])]
    return pd.Series(
        {
            "alignment": formatted_alignment[1],
            "alignment_in": formatted_alignment[0],
            "alignment_out": formatted_alignment[2],
        }
    )


def process_alignment(row, args):
    gt = row["alignment_in"]
    preds = row["alignment_out"]
    old_label_count = preds.count("|")

    # we need the following to ensure that no label is overwritten.
    processed_gt = []
    processed_preds = []

    pred_index = 0  # separate index for preds to handle skipped characters in gt

    for gt_char in gt:
        if gt_char == args.gap_char:
            # check if the corresponding preds char is a label delimiter and needs to be shifted
            if pred_index < len(preds) and preds[pred_index] == args.label_delimiter:
                # only shift if there's room and it's safe to shift back
                if processed_preds:
                    processed_preds[-1] = args.label_delimiter
            # do not add the gap character from gt to processed_gt
        else:
            processed_gt.append(gt_char)
            # only add characters from preds if not out of bounds
            if pred_index < len(preds):
                processed_preds.append(preds[pred_index])

        # Increment regardless of whether it's a gap to keep alignment
        pred_index += 1

    gt = "".join(processed_gt)
    preds = "".join(processed_preds)
    # re-check
    new_label_count = preds.count(args.label_delimiter)
    if old_label_count != new_label_count:
        # happens in some garbage LLM predictions, which can be easily picked up, so it is fine (favors LLM; less FPs)
        print("Gap char not removed correctly or labels shifted improperly: ", row.name, row.lang, row.dataset_name)
    assert len(gt) == len(preds), f"Length mismatch: {len(gt)} vs {len(preds)}"

    if Constants.SEPARATORS.get(row["lang"], " ") == "":
        # ensure that, after label calculation, both seqs are of same lengths & labels at proper positions.
        missing_indices = [
            i for i, char in enumerate(gt) if char == args.label_delimiter and preds[i] != args.label_delimiter
        ]
        extra_indices = [
            i for i, char in enumerate(preds) if char == args.label_delimiter and gt[i] != args.label_delimiter
        ]
        preds = "".join([char for i, char in enumerate(preds) if i not in missing_indices])
        for i in extra_indices:
            preds = preds[:i] + " " + preds[i:]

    # GT
    sentences = gt.split(args.label_delimiter)
    labels = get_labels(row.lang, sentences)
    # PREDS
    pred_sentences = preds.split(args.label_delimiter)
    predictions = get_labels(row.lang, pred_sentences)
    return labels, predictions


def calc_hallucination_deletion_rate(row):
    gt = row["alignment_in"]
    preds = row["alignment_out"]
    if all([char == args.gap_char for char in preds]):
        # all @: alignment failure, just garbage output
        return 0.0, 0.0

    hallucination_count = 0
    deletion_count = 0

    for gt_char, pred_char in zip(gt, preds):
        if gt_char == args.gap_char and pred_char != args.label_delimiter:
            hallucination_count += 1
        if pred_char == args.gap_char and gt_char != args.label_delimiter:
            deletion_count += 1

    deletion_rate = deletion_count / len(gt)
    hallucination_rate = hallucination_count / len(gt)
    return hallucination_rate, deletion_rate


def calculate_global_metric_averages(results):
    """dict of results[lang_code][dataset_name] -> dict of metrics -> float"""
    metric_totals = {}
    metric_counts = {}

    for lang_datasets in results.values():
        for metrics in lang_datasets.values():
            # aggregate
            metrics = metrics[args.model]
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_key not in metric_totals:
                        metric_totals[metric_key] = 0
                        metric_counts[metric_key] = 0

                    metric_totals[metric_key] += metric_value
                    metric_counts[metric_key] += 1

    # global average
    global_metric_averages = {
        metric: metric_totals[metric] / metric_counts[metric] for metric in metric_totals if metric_counts[metric] > 0
    }

    return global_metric_averages


def main(args):
    print(args)

    # Define paths for different categories
    avg_dir = Constants.CACHE_DIR / "llm_sent" / "AVG"
    default_dir = Constants.CACHE_DIR / "llm_sent" / "results"
    alignment_dir = Constants.CACHE_DIR / "llm_sent" / "alignments"

    # Ensure directories exist
    avg_dir.mkdir(parents=True, exist_ok=True)
    default_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "all" or args.type == "pairs":
        eval_data_path = args.eval_data_path + "-all.pth"
    elif args.type == "lyrics":
        eval_data_path = args.eval_data_path + "-lyrics.pth"
    elif args.type == "short_proc":
        eval_data_path = args.eval_data_path + "-short_proc.pth"
    else:
        raise ValueError(f"Unknown type: {args.type}")

    assert len(args.gap_char) == len(args.label_delimiter) == 1

    eval_data = torch.load(eval_data_path)

    save_str = (
        f"{args.model.split('/')[-1]}_k{args.k}_s{args.n_shots}"
    ).replace("/", "_")
    
    if args.max_n_test_sentences < sys.maxsize and args.max_n_test_sentences != -1:
        save_str += f"_n{args.max_n_test_sentences}"
    if args.max_n_test_sentences == -1:
        args.max_n_test_sentences = sys.maxsize
    save_str += f"{args.save_suffix}"
    save_str += f"-{args.type}"

    print(save_str)

    outputs = load_or_compute_logits(args, eval_data, save_str)

    # create df based on test_chunks and test_logits
    df = load_h5_to_dataframe(outputs.filename, args)
    print("Loaded df.")
    # df = df[df["lang"].isin(["ja", "en"])]  # DEBUG

    # postprocess
    df["test_preds"] = df.apply(lambda row: postprocess_llm_output(row["test_preds"], row["lang"]), axis=1)

    # remove empty strings (just needed for h5py storage)
    df["test_chunks"] = df["test_chunks"].apply(lambda x: [item for item in x if item != ""])
    # replace \n
    # NOTE: length and labels remain the same, crucially!
    df["test_chunks"] = df.apply(
        lambda row: args.label_delimiter.join(
            chunk.replace(args.label_delimiter, " ").replace(args.gap_char, " ") for chunk in row["test_chunks"]
        ),
        axis=1,
    )
    print("Processed df.")

    # align with Needleman Wunsch algorithm
    alignment = df.parallel_apply(align_llm_output, axis=1)
    df = df.join(alignment)
    print("Aligned df.")

    def concatenate_texts(group):
        # refusal: @@@@ --> taken into account with success_METRIC
        # alignment failure: "     " --> garbage output, no pos. labels
        return pd.Series(
            {
                "test_preds": args.label_delimiter.join(group["test_preds"]),
                "test_chunks": args.label_delimiter.join(group["test_chunks"]),
                "alignment": args.label_delimiter.join(group["alignment"]),
                "alignment_in": args.label_delimiter.join(group["alignment_in"]),
                "alignment_out": args.label_delimiter.join(group["alignment_out"]),
            }
        )

    # concat chunks
    # old_df = df.copy()
    df = df.groupby(["lang", "dataset_name", "doc_id"]).apply(concatenate_texts).reset_index()

    df["hallucination_rate"], df["deletion_rate"] = zip(
        *df.apply(
            calc_hallucination_deletion_rate,
            axis=1,
        )
    )

    results = {}
    indices = {}
    for lang_code in df["lang"].unique():
        results[lang_code] = {}
        indices[lang_code] = {}
        for dataset_name in df["dataset_name"].unique():
            results[lang_code][dataset_name] = {args.model: {}}  # Initialize nested dict with model
            indices[lang_code][dataset_name] = {args.model: {}}
            if "lyrics" in dataset_name or "short" in dataset_name or args.type == "pairs":
                exclude_every_k = 0
            else:
                exclude_every_k = args.k
            n_docs = len(df[(df["lang"] == lang_code) & (df["dataset_name"] == dataset_name)])
            if n_docs == 0:
                # combination non-existing
                continue
            indices[lang_code][dataset_name][args.model] = {}
            if n_docs > 1:
                # list of lists, TODO
                rows = df[(df["lang"] == lang_code) & (df["dataset_name"] == dataset_name)]
                metrics = []
                for i, row in rows.iterrows():
                    # apply processing before label calculation
                    labels, preds = process_alignment(row, args)
                    doc_metrics = {}
                    doc_metrics["refused"] = [
                        int(len(set(row["alignment_out"])) == 1 and args.gap_char in set(row["alignment_out"]))
                    ]
                    if doc_metrics["refused"][0]:
                        preds[-1] = 0
                    doc_metrics.update(
                        evaluate_sentences_llm(labels, preds, return_indices=True, exclude_every_k=exclude_every_k)
                    )
                    doc_metrics["length"] = [doc_metrics["length"]]
                    doc_metrics["hallucination_rate"] = row["hallucination_rate"]
                    doc_metrics["deletion_rate"] = row["deletion_rate"]

                    metrics.append(doc_metrics)
                # Initialization and collection of data
                avg_results = {}
                concat_indices = {}
                for doc in metrics:
                    for key, value in doc.items():
                        if isinstance(value, (float, int)):
                            # Store all numeric results
                            if key not in avg_results:
                                avg_results[key] = []
                                avg_results[key + "_success"] = []  # Initialize success list

                            # Append to the general results
                            avg_results[key].append(value)

                            # Append to success results if not refused
                            if not doc["refused"][0]:
                                avg_results[key + "_success"].append(value)
                        elif isinstance(value, list):
                            # Concatenate list values, handle 'refused' by only adding the first item of the list
                            if key not in concat_indices:
                                concat_indices[key] = []
                            if key == "refused" or key == "length":
                                concat_indices[key].append(value[0])  # Only the first item for 'refused'
                            else:
                                concat_indices[key].append(value)  # Extend with the full list otherwise

                # Calculate the average for numeric values and success metrics
                for key in list(avg_results):  # Use list to include newly added success keys safely during iteration
                    if avg_results[key]:  # Ensure there's data to calculate average
                        avg_results[key] = sum(avg_results[key]) / len(avg_results[key])

                # Store the results and indices
                results[lang_code][dataset_name][args.model] = avg_results
                indices[lang_code][dataset_name][args.model] = concat_indices
            else:
                # one long string
                row = df[(df["lang"] == lang_code) & (df["dataset_name"] == dataset_name)].iloc[0]

                # apply processing before label calculation
                labels, preds = process_alignment(row, args)
                # metrics!
                metrics = evaluate_sentences_llm(labels, preds, return_indices=True, exclude_every_k=exclude_every_k)
                metrics["hallucination_rate"] = row["hallucination_rate"]
                metrics["deletion_rate"] = row["deletion_rate"]
                indices[lang_code][dataset_name][args.model]["true_indices"] = [metrics.pop("true_indices")]
                indices[lang_code][dataset_name][args.model]["predicted_indices"] = [metrics.pop("predicted_indices")]
                indices[lang_code][dataset_name][args.model]["length"] = [metrics.pop("length")]
                results[lang_code][dataset_name][args.model] = metrics

    out_dict = {
        "metrics": calculate_global_metric_averages(results),
        "include_langs": args.include_langs,
        "max_n_test_sentences": args.max_n_test_sentences,
        "k": args.k,
        "n_success": len(df[df["test_preds"] != ""]),
        "success_rate": len(df[df["test_preds"] != ""]) / len(df),
        "model": args.model,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": LYRICS_PROMPT if args.type == "lyrics" else SYSTEM_PROMPT,
    }

    json.dump(
        out_dict,
        open(
            avg_dir / f"{save_str}.json",
            "w",
            encoding="utf-8",
        ),
        indent=4,
    )
    json.dump(
        results,
        open(
            default_dir / f"{save_str}.json",
            "w",
            encoding="utf-8",
        ),
        indent=4,
    )
    json.dump(
        indices,
        open(
            alignment_dir / f"{save_str}_IDX.json",
            "w",
        ),
        default=int,
        indent=4,
    )
    print(alignment_dir / f"{save_str}_IDX.json")
    print("Indices saved to file.")

    df.to_csv(default_dir / "csv" / f"{save_str}.csv", index=False)
    print(out_dict)
    print(save_str)

    print(avg_dir / f"{save_str}.json")
    print(default_dir / f"{save_str}.json")


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    main(args)
