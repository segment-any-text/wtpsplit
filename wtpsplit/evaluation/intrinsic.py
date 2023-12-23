import copy
import json
from dataclasses import dataclass
from typing import List

import h5py
import skops.io as sio
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit.models # noqa: F401
from wtpsplit.evaluation import evaluate_mixture, get_labels, train_mixture
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
    eval_data_path: str = "data/eval.pth"
    valid_text_path: str = None#"data/sentence/valid.parquet"
    device: str = "cpu"
    block_size: int = 512
    stride: int = 64
    batch_size: int = 32
    include_langs: List[str] = None


def load_or_compute_logits(args, model, eval_data, valid_data=None, max_n_train_sentences=10_000):
    logits_path = Constants.CACHE_DIR / (model.config.mixture_name + "_logits.h5")

    # TODO: revert to "a"
    with h5py.File(logits_path, "w") as f, torch.no_grad():
        for lang_code in Constants.LANGINFO.index:
            if args.include_langs is not None and lang_code not in args.include_langs:
                continue

            if lang_code not in f:
                lang_group = f.create_group(lang_code)
            else:
                lang_group = f[lang_code]

            # valid data
            if valid_data is not None and "valid" not in lang_group:
                sentences = [sample["text"].strip() for sample in valid_data if sample["lang"] == lang_code]
                assert len(sentences) > 0

                separator = Constants.SEPARATORS[lang_code]
                text = separator.join(sentences)

                valid_logits = extract(
                    [text],
                    model,
                    lang_code=lang_code,
                    stride=args.stride,
                    block_size=args.block_size,
                    batch_size=args.batch_size,
                    pad_last_batch=True,
                    verbose=True,
                )[0]
                lang_group.create_dataset("valid", data=valid_logits)

            # eval data
            for dataset_name, dataset in eval_data[lang_code]["sentence"].items():
                if dataset_name not in lang_group:
                    dset_group = lang_group.create_group(dataset_name)
                else:
                    dset_group = lang_group[dataset_name]

                if "test_logits" not in dset_group:
                    test_sentences = dataset["data"]
                    test_text = Constants.SEPARATORS[lang_code].join(test_sentences)

                    test_logits = extract(
                        [test_text],
                        model,
                        lang_code=lang_code,
                        stride=args.stride,
                        block_size=args.block_size,
                        batch_size=args.batch_size,
                        pad_last_batch=True,
                        verbose=True,
                    )[0]
                    test_labels = get_labels(lang_code, test_sentences, after_space=False)

                    dset_group.create_dataset("test_logits", data=test_logits)
                    dset_group.create_dataset("test_labels", data=test_labels)

                train_sentences = dataset["meta"].get("train_data")
                if train_sentences is not None and "train_logits" not in dset_group:
                    train_sentences = train_sentences[:max_n_train_sentences]
                    train_text = Constants.SEPARATORS[lang_code].join(train_sentences)

                    train_logits = extract(
                        [train_text],
                        model,
                        lang_code=lang_code,
                        stride=args.stride,
                        block_size=args.block_size,
                        batch_size=args.batch_size,
                        pad_last_batch=False,
                    )[0]
                    train_labels = get_labels(lang_code, train_sentences, after_space=False)

                    dset_group.create_dataset("train_logits", data=train_logits)
                    dset_group.create_dataset("train_labels", data=train_labels)

    return h5py.File(logits_path, "r")


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    eval_data = torch.load(args.eval_data_path)
    if args.valid_text_path is not None:
        valid_data = load_dataset("parquet", data_files=args.valid_text_path, split="train")
    else:
        valid_data = None
    
    model = PyTorchWrapper(AutoModelForTokenClassification.from_pretrained(args.model_path).to(args.device))

    # first, logits for everything.
    f = load_or_compute_logits(args, model, eval_data, valid_data)

    # now, compute the intrinsic scores.
    results = {}
    clfs = {}

    for lang_code, dsets in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        results[lang_code] = {}
        clfs[lang_code] = {}

        for dataset_name, dataset in dsets["sentence"].items():
            sentences = dataset["data"]

            if "train_logits" in f[lang_code][dataset_name]:
                feature_indices = None
                # TODO: tokenize here
                clf = train_mixture(
                    [lang_code],
                    f[lang_code][dataset_name]["train_logits"][:],
                    f[lang_code][dataset_name]["train_labels"][:],
                    features=feature_indices,
                )

                # TODO: tokenize here, too
                score_t, score_punct, _ = evaluate_mixture(
                    lang_code,
                    f[lang_code][dataset_name]["test_logits"][:],
                    sentences,
                    *clf,
                )

                clfs[lang_code][dataset_name] = clf

                clf = list(copy.deepcopy(clf))
                clf[-1] = 0.01
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
            print(f"{lang_code} {dataset_name} {score_u:.3f} {score_t:.3f} {score_punct:.3f}")

    sio.dump(
        clfs,
        open(
            Constants.CACHE_DIR / (model.config.mixture_name + ".skops"),
            "wb",
        ),
    )
    json.dump(
        results,
        open(
            Constants.CACHE_DIR / (model.config.mixture_name + "_intrinsic_results.json"),
            "w",
        ),
        indent=4,
    )
