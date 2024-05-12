import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List
import re

import numpy as np
import h5py
import pandas as pd
import torch
from genalog.text import alignment
from pandarallel import pandarallel
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import HfArgumentParser
import cohere
import replicate

from wtpsplit.evaluation import get_labels
from wtpsplit.evaluation.intrinsic import corrupt
from wtpsplit.utils import Constants

pandarallel.initialize(progress_bar=False, nb_workers=32)

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
    type: str = "lyrics"  # all, lyrics
    llm_provider: str = "cohere"  # cohere, replicate
    label_delimiter: str = "|"  # NOT \n or \n\n
    gap_char = "@"
    # model: str = "mistralai/mixtral-8x7b-instruct-v0.1"
    # model: str = "meta/meta-llama-3-70b-instruct"
    model: str = "command-r"
    save_suffix: str = "Pv2-TEST2"
    include_langs: List[str] = None
    custom_language_list: str = None
    max_n_test_sentences: int = 100
    k: int = 10
    n_shots: int = 0


def replicate_provider(text, train_data, lang_code, args):
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    llm_prompt = prompt_factory(text, train_data, lang_code, args)
    # print(llm_prompt)
    llm_input = {
        "system_prompt": "",
        "prompt": llm_prompt,
        # "max_new_tokens": 50_000,
        "max_tokens": 50_000,
    }
    llm_output = api.run(args.model, llm_input)
    llm_output = "".join(llm_output)
    # print(llm_output)
    return llm_output


def cohere_provider(text, train_data, lang_code, args):
    api = cohere.Client(os.environ["COHERE_API_TOKEN"])
    llm_prompt = prompt_factory(text, train_data, lang_code, args)
    # print(llm_prompt)
    llm_output = api.chat(model=args.model, preamble="", message=llm_prompt, max_tokens=4000, seed=42).text.replace(
        "\\n", "\n"
    )
    # print(llm_output)
    return llm_output


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
                if "corrupted" in dataset_name and (
                    dataset_name != "ted2020-corrupted-asr" and not ("lyrics" in dataset_name and "asr" in dataset_name)
                ):
                    print("SKIP: ", lang_code, dataset_name)
                    continue
                if "legal" in dataset_name and not ("laws" in dataset_name or "judgements" in dataset_name):
                    print("SKIP: ", lang_code, dataset_name)
                    continue
                if "social-media" in dataset_name:
                    continue
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]
                if "test_preds" not in dset_group and "test_preds_0" not in dset_group:
                    test_sentences = dataset["data"]
                    if not test_sentences:
                        continue
                    if isinstance(test_sentences[0], list):
                        max_n_test_sentences = args.max_n_test_sentences // 10
                    else:
                        max_n_test_sentences = args.max_n_test_sentences
                    test_sentences = test_sentences[:max_n_test_sentences]
                    if isinstance(test_sentences[0], list):
                        # list of lists: chunk each sublist
                        if "short" in dataset_name or "lyrics" in dataset_name:
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
                                if "short" in dataset_name:
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
                                if "short" in dataset_name or "lyrics" in dataset_name
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
        try:
            output.append(llm_provider(test_chunk, train_data, lang_code, args))
        except Exception as e:
            print(f"API Error: {e}")
            output.append("")
    return output


def prompt_factory(test_chunk, train_data, lang_code, args):
    n_shots = args.n_shots if train_data is not None else 0
    main_prompt = LYRICS_PROMPT if args.type == "lyrics" else SYSTEM_PROMPT

    prompt_start = (
        main_prompt
        + f"When provided with multiple examples, you are to respond only to the last one: # Output {n_shots + 1}."
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


def postprocess_llm_output(llm_output):
    """
    Cleans the LLM output by removing specified characters and cleaning up lines.
    """
    if llm_output == "":
        # API refusal
        return ""
    llm_output = llm_output.strip(" -\n")

    # neither of them must be present in the output.
    llm_output = llm_output.replace(args.gap_char, " ")
    llm_output = llm_output.replace(args.label_delimiter, " ")
    llm_output = llm_output.replace("\n\n", args.label_delimiter)
    llm_output = llm_output.replace("\n", args.label_delimiter)

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
    """
    Attempts to align input and output, including formatting.
    """
    try:
        aligned_in, aligned_llm = alignment.align(
            row["test_chunks"],
            row["test_preds"],
            gap_char=args.gap_char,
        )
        # same as aligned_in, aligned_llm, but with additional formatting. Latter used to debug only.
        formatted_alignment = alignment._format_alignment(aligned_in, aligned_llm).split("\n")
    except:
        print("Alignment failed: ", row.name)
        formatted_alignment = [row["test_chunks"], "", " " * len(row["test_chunks"])]
    return pd.Series(
        {
            "alignment": formatted_alignment[1],
            "alignment_in": formatted_alignment[0],
            "alignment_out": formatted_alignment[2],
        }
    )


def get_llm_metrics(row):
    gt = row["alignment_in"]
    preds = row["alignment_out"]

    # find gap_char idcs
    gap_indices = [m.start() for m in re.finditer(args.gap_char, gt)]
    # remove gap_char indices from gt and preds --> same len as original!
    gt = "".join([char for i, char in enumerate(gt) if i not in gap_indices])
    preds = "".join([char for i, char in enumerate(preds) if i not in gap_indices])

    assert (
        args.label_delimiter.join([chunk.replace(args.label_delimiter, "") for chunk in row.test_chunks]).count(
            args.label_delimiter
        )
        if isinstance(row.test_chunks, list)
        else row.test_chunks.count(args.label_delimiter) == len(gt.split(args.label_delimiter)) - 1
    )
    # GT
    sentences = gt.split(args.label_delimiter)
    labels = get_labels(row.lang, sentences)
    if not ("lyrics" in row.dataset_name or "short" in row.dataset_name):
        # XXX: must be logically aligned with evaluate_sentences in wtpsplit/evaluation/__init__.py
        # label at end is excluded! As used for main tables --> comparable.
        labels = labels[:-1]

    chunk_len = len(labels)
    true_indices = np.where(labels)[0].tolist()

    if len(gt) == len(preds):
        # alignment is good.
        predicted_sentences = preds.split(args.label_delimiter)
    else:
        # alignment not found, bad LLM prediction.
        return (0.0, 0.0, 0.0, true_indices, [], chunk_len)
    if row["test_preds"] == "":
        # API refusal
        return (0.0, 0.0, 0.0, true_indices, [], chunk_len)
    predictions = get_labels(row.lang, predicted_sentences)
    if not ("lyrics" in row.dataset_name or "short" in row.dataset_name):
        predictions = predictions[:-1]
    assert len(labels) == len(predictions)

    f1, precision, recall = (
        f1_score(labels, predictions, zero_division=0),
        precision_score(labels, predictions, zero_division=0),
        recall_score(labels, predictions, zero_division=0),
    )

    if len(row["test_chunks"].split(args.label_delimiter)) != args.k and (
        row["doc_id"] == -1 or row["is_last_and_truncated"]
    ):
        # re-scale to original
        # XXX: hacky, but we later take mean over lang-dataset combination w/o this, final chunk is overrepresented
        f1 = f1 * len(row["test_chunks"].split(args.label_delimiter)) / args.k
        precision = precision * len(row["test_chunks"].split(args.label_delimiter)) / args.k
        recall = recall * len(row["test_chunks"].split(args.label_delimiter)) / args.k

    # Compute F1 score
    pred_indices = np.where(predictions)[0].tolist()
    return f1, precision, recall, true_indices, pred_indices, chunk_len


def calc_hallucination_deletion_rate(row):
    gt = row["alignment_in"]
    preds = row["alignment_out"]
    if all([char == args.gap_char for char in preds]):
        # all @
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

    if args.type == "all":
        eval_data_path = args.eval_data_path + "-all.pth"
    elif args.type == "lyrics":
        eval_data_path = args.eval_data_path + "-lyrics.pth"
    else:
        raise ValueError(f"Unknown type: {args.type}")

    assert len(args.gap_char) == len(args.label_delimiter) == 1

    eval_data = torch.load(eval_data_path)

    save_str = (
        f"{args.model.split('/')[-1]}_k{args.k}_n{args.max_n_test_sentences}_s{args.n_shots}{args.save_suffix}"
    ).replace("/", "_")
    save_str += f"-{args.type}"

    print(save_str)

    outputs = load_or_compute_logits(args, eval_data, save_str)

    # create df based on test_chunks and test_logits
    df = load_h5_to_dataframe(outputs.filename, args)
    print("Loaded df.")

    # postprocess
    df["test_preds"] = df.apply(lambda row: postprocess_llm_output(row["test_preds"]), axis=1)

    # remove empty strings (just needed for h5py storage)
    df["test_chunks"] = df["test_chunks"].apply(lambda x: [item for item in x if item != ""])
    # replace \n
    # NOTE: length and labels remain the same, crucially!
    df["test_chunks"] = df["test_chunks"].apply(
        lambda x: args.label_delimiter.join(
            chunk.replace(args.label_delimiter, " ").replace(args.gap_char, " ") for chunk in x
        )
    )
    print("Processed df.")

    # needed to later down-scale F1 score contributions for last chunk
    df["is_last"] = df["doc_id"] != df["doc_id"].shift(-1)
    dataset_condition = ~df["dataset_name"].str.contains("lyrics|short")
    df["is_last_and_truncated"] = df["is_last"] & dataset_condition

    # align with Needleman Wunsch algorithm
    alignment = df.parallel_apply(align_llm_output, axis=1)
    df = df.join(alignment)
    print("Aligned df.")

    (
        df["f1"],
        df["precision"],
        df["recall"],
        df["true_indices"],
        df["pred_indices"],
        df["chunk_len"],
    ) = zip(*df.apply(get_llm_metrics, axis=1))

    df["hallucination_rate"], df["deletion_rate"] = zip(
        *df.apply(
            calc_hallucination_deletion_rate,
            axis=1,
        )
    )
    # get stats
    doc_level_metrics = df.groupby(["lang", "dataset_name", "doc_id"]).agg(
        {
            "f1": ["mean"],
            "precision": ["mean"],
            "recall": ["mean"],
        }
    )
    # mean of mean --> macro F1 (for list of lists)
    metrics = doc_level_metrics.groupby(["lang", "dataset_name"]).mean().reset_index()

    # metrics without API complaints
    doc_level_metrics_success = (
        df[df.test_preds != ""]
        .groupby(["lang", "dataset_name", "doc_id"])
        .agg(
            {
                "f1": ["mean"],
                "precision": ["mean"],
                "recall": ["mean"],
            }
        )
    )
    metrics_success = doc_level_metrics_success.groupby(["lang", "dataset_name"]).mean().reset_index()
    df["cumulative_chunk_len"] = df.groupby(["lang", "dataset_name", "doc_id"])["chunk_len"].cumsum() - df["chunk_len"]
    # group by chunk len and get max. --> same for all rows of a doc belonging to a lang-dataset combination

    # adjust indices by adding cumulative chunk length
    def adjust_indices(indices, cumulative_len):
        return [index + cumulative_len for index in indices]

    # adjust indices in each row
    df["true_indices_adj"] = df.apply(
        lambda row: adjust_indices(row["true_indices"], row["cumulative_chunk_len"]), axis=1
    )
    df["pred_indices_adj"] = df.apply(
        lambda row: adjust_indices(row["pred_indices"], row["cumulative_chunk_len"]), axis=1
    )

    out_dict = {
        "metrics": {
            "f1": metrics["f1"]["mean"].mean(),
            "success_f1": metrics_success["f1"]["mean"].mean(),
            "precision": metrics["precision"]["mean"].mean(),
            "success_precision": metrics_success["precision"]["mean"].mean(),
            "recall": metrics["recall"]["mean"].mean(),
            "success_recall": metrics_success["recall"]["mean"].mean(),
            "median": df["f1"].median(),
            "std": df["f1"].std(),
            "min": df["f1"].min(),
            "min_lang": df.loc[df["f1"].idxmin()]["lang"],
            "max": df["f1"].max(),
            "max_lang": df.loc[df["f1"].idxmax()]["lang"],
            "hallucination_rate": df["hallucination_rate"].mean(),
            "deletion_rate": df["deletion_rate"].mean(),
        },
        "include_langs": args.include_langs,
        "max_n_test_sentences": args.max_n_test_sentences,
        "k": args.k,
        "n_success": len(df[df["test_preds"] != ""]),
        "success_rate": len(df[df["test_preds"] != ""]) / len(df),
        "model": args.model,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": SYSTEM_PROMPT if args.type == "all" else LYRICS_PROMPT,
    }

    fine_grained_dict = {}
    fine_grained_indices = {}
    for lang in df["lang"].unique():
        fine_grained_dict[lang] = {}
        fine_grained_indices[lang] = {}
        for dataset in df["dataset_name"].unique():
            local_df = df[(df["lang"] == lang) & (df["dataset_name"] == dataset)]
            if len(local_df) == 0:
                continue
            results = {
                "f1": local_df["f1"].mean(),
                "precision": local_df["precision"].mean(),
                "recall": local_df["recall"].mean(),
                "success_f1": local_df[local_df["test_preds"] != ""]["f1"].mean(),
                "success_precision": local_df[local_df["test_preds"] != ""]["precision"].mean(),
                "success_recall": local_df[local_df["test_preds"] != ""]["recall"].mean(),
                "success_rate": len(local_df[local_df["test_preds"] != ""]) / len(local_df) if len(local_df) else 0,
                "hallucination_rate": local_df["hallucination_rate"].mean(),
                "deletion_rate": local_df["deletion_rate"].mean(),
            }
            # indices: concat all lists
            if any(local_df["doc_id"] > -1):
                # group by doc id first
                fine_grained_indices[lang][dataset] = {}
                fine_grained_indices[lang][dataset]["true_indices"] = []
                fine_grained_indices[lang][dataset]["pred_indices"] = []
                fine_grained_indices[lang][dataset]["length"] = []
                for doc_id in local_df["doc_id"].unique():
                    fine_grained_indices[lang][dataset]["true_indices"].append(
                        [
                            item
                            for sublist in local_df[local_df["doc_id"] == doc_id]["true_indices_adj"].tolist()
                            for item in sublist
                        ]
                    )
                    fine_grained_indices[lang][dataset]["pred_indices"].append(
                        [
                            item
                            for sublist in local_df[local_df["doc_id"] == doc_id]["pred_indices_adj"].tolist()
                            for item in sublist
                        ]
                    )
                    fine_grained_indices[lang][dataset]["length"].append(
                        local_df[local_df["doc_id"] == doc_id]["chunk_len"].sum()
                    )
            else:
                fine_grained_indices[lang][dataset] = {
                    "true_indices": [item for sublist in local_df["true_indices_adj"].tolist() for item in sublist],
                    "pred_indices": [item for sublist in local_df["pred_indices_adj"].tolist() for item in sublist],
                    "length": local_df["chunk_len"].sum(),
                }

            fine_grained_dict[lang][dataset] = results

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
        fine_grained_dict,
        open(
            default_dir / f"{save_str}.json",
            "w",
            encoding="utf-8",
        ),
        indent=4,
    )
    json.dump(
        fine_grained_indices,
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
    # TODO: count how many left outs/hallucinations are from LLM.


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    main(args)
