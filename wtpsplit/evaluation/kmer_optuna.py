import copy
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process
from typing import List

import numpy as np
import optuna
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit.models  # noqa: F401
from wtpsplit.evaluation import evaluate_mixture
from wtpsplit.evaluation.intrinsic import compute_statistics
from wtpsplit.evaluation.intrinsic_pairwise import calculate_threshold, generate_k_mers, load_or_compute_logits
from wtpsplit.extract import PyTorchWrapper

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
    skip_adaptation: bool = True
    keep_logits: bool = True

    # k_mer-specific args
    min_k: int = 2
    max_k: int = 4
    max_n_samples: int = sys.maxsize
    sample_pct: float = 0.5
    min_k_mer_length: int = 0
    adjust_threshold: bool = True
    # threshold
    # threshold_increase_type: str = "linear"
    threshold_min_length: int = 0
    threshold_max_length: int = 256
    threshold_max: float = 0.1
    # optuna args
    n_trials: int = 16
    n_jobs: int = 32


def objective(trial: optuna.Trial, args: Args, eval_data: dict, f_list) -> float:
    # Suggest values for the hyperparameters we want to optimize
    args.threshold_min_length = trial.suggest_int("threshold_min_length", 0, 256)
    args.threshold_max_length = trial.suggest_int("threshold_max_length", 0, 256)
    args.threshold_max = trial.suggest_float("threshold_max", 0.00, 0.5)

    # Execute the main function and retrieve results
    all_results = []
    all_results_avg = []
    all_mean_u_acc = []
    for i, k in enumerate(range(args.min_k, args.max_k + 1)):
        args.k = k
        f = f_list[i]
        results, results_avg = main(args, eval_data, f)
        all_results.append(results)
        all_results_avg.append(results_avg)
        all_mean_u_acc.append(results_avg["u_acc"]["mean"])

        # Store results in the trial's user attributes
        trial.set_user_attr(f"{k}_detailed_results", results)
        trial.set_user_attr(f"{k}_average_results", results_avg)

    # Objective is to maximize the average U accuracy
    # return list as tuple
    return tuple(all_mean_u_acc)


def load_data_and_model(args):
    logger.info("Loading model...")
    model = PyTorchWrapper(AutoModelForTokenClassification.from_pretrained(args.model_path).to(args.device))

    logger.info("Loading evaluation data...")
    eval_data = torch.load(args.eval_data_path)

    # Possibly other initialization code here
    return model, eval_data


def main(args, eval_data, f):
    # now, compute the intrinsic scores.
    results = {}
    clfs = {}
    # Initialize lists to store scores for each metric across all languages
    u_scores = []
    u_accs = []
    thresholds_adj = []

    for lang_code, dsets in tqdm(eval_data.items(), desc="Languages", total=len(eval_data), disable=True):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        results[lang_code] = {}
        clfs[lang_code] = {}

        for dataset_name, dataset in dsets["sentence"].items():
            sentences = dataset["data"]
            sent_k_mers = generate_k_mers(
                sentences,
                k=args.k,
                do_lowercase=args.do_lowercase,
                do_remove_punct=args.do_remove_punct,
                sample_pct=args.sample_pct,
                max_n_samples=args.max_n_samples,
                min_k_mer_length=args.min_k_mer_length,
            )

            clf = [None, None, None, args.threshold]

            score_u = []
            acc_u = []
            thresholds = []
            for i, k_mer in enumerate(sent_k_mers):
                start, end = f[lang_code][dataset_name]["test_logit_lengths"][i]
                if args.adjust_threshold:
                    seq_len = f[lang_code][dataset_name]["test_n_logits"][i]
                    threshold_adjusted = calculate_threshold(
                        sequence_length=seq_len,
                        max_length=args.threshold_max_length,
                        min_length=args.threshold_min_length,
                        max_threshold=args.threshold_max,
                        default_threshold=args.threshold,
                    )
                    clf[-1] = threshold_adjusted
                    thresholds.append(threshold_adjusted)
                else:
                    raise NotImplementedError("Optuna runs are to select the optimal threshold config!")
                single_score_u, _, info = evaluate_mixture(
                    lang_code,
                    f[lang_code][dataset_name]["test_logits"][:][start:end],
                    list(k_mer),
                    *clf,
                )
                score_u.append(single_score_u)
                acc_u.append(info["info_newline"]["correct_pairwise"])

            score_u = np.mean(score_u)
            acc_u = np.mean(acc_u)
            threshold = np.mean(thresholds)

            results[lang_code][dataset_name] = {
                "u": score_u,
                "u_acc": acc_u,
                "threshold_adj": threshold,
            }

            u_scores.append((score_u, lang_code))
            u_accs.append((acc_u, lang_code))
            thresholds_adj.append((threshold, lang_code))

    # Compute statistics for each metric across all languages
    results_avg = {
        "u": compute_statistics(u_scores),
        "u_acc": compute_statistics(u_accs),
        "threshold_adj": compute_statistics(thresholds_adj),
        "include_langs": args.include_langs,
    }

    return results, results_avg


def run_optimization(storage_url, study_name, args, eval_data, f_list):
    """
    Function to run Optuna optimization in a separate process.
    """
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(
        lambda trial: objective(trial, copy.deepcopy(args), eval_data, f_list),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"Completed optimization for study: {study_name}")


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    model, eval_data = load_data_and_model(args)

    # first, logits for everything.
    f_list = []
    for k in range(args.min_k, args.max_k + 1):
        args.k = k
        save_str = f"{args.model_path.replace('/','_')}_b{args.block_size}_u{args.threshold}_k_{k}{args.save_suffix}"
        print(save_str)
        out, _ = load_or_compute_logits(args, model, eval_data, None, save_str)
        f_list.append(out)

    # replace k_[max_k] with k_mink-max_k in save_str
    save_str = save_str.replace(f"k_{args.max_k}", f"k_{args.min_k}-{args.max_k}")

    # storage using SQLite URL
    storage_url = "mysql://root@localhost/example"
    study_name = f"{save_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        directions=["maximize"] * (args.max_k - args.min_k + 1),
        load_if_exists=True,
    )

    # Create multiple studies and launch them in separate processes
    processes = []
    for i in range(args.n_jobs):
        proc = Process(target=run_optimization, args=(storage_url, study_name, args, eval_data, f_list))
        processes.append(proc)
        proc.start()

    # Wait for all processes to complete
    for proc in processes:
        proc.join()

