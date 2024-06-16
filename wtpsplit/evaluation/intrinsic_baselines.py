import json
from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm
from transformers import HfArgumentParser

from wtpsplit.evaluation import (
    LanguageError,
    ersatz_sentencize,
    evaluate_sentences,
    punkt_sentencize,
    pysbd_sentencize,
    spacy_dp_sentencize,
    spacy_sent_sentencize,
)
from wtpsplit.utils import Constants


def split_language_data(eval_data):
    # used if 2 language codes given (i.e., code-switching)
    new_eval_data = {}

    for lang_code, lang_data in eval_data.items():
        if "-" in lang_code:
            lang1, lang2 = lang_code.split("-")
            new_lang1 = f"{lang_code}_{lang1.upper()}"
            new_lang2 = f"{lang_code}_{lang2.upper()}"

            new_eval_data[new_lang1] = lang_data
            new_eval_data[new_lang2] = lang_data
        else:
            new_eval_data[lang_code] = lang_data

    return new_eval_data


@dataclass
class Args:
    eval_data_path: str = "data/all_data.pth"
    include_langs: List[str] = None
    exclude_every_k: int = 10


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    eval_data = torch.load(args.eval_data_path)
    eval_data = split_language_data(eval_data)
    results = {}
    indices = {}

    for lang, lang_data in tqdm(eval_data.items()):
        if args.include_langs is not None and lang not in args.include_langs:
            continue

        results[lang] = {}
        indices[lang] = {}

        for dataset_name, dataset in lang_data["sentence"].items():
            if "nllb" in dataset_name:
                continue
            if not dataset["data"]:
                continue
            results[lang][dataset_name] = {}
            indices[lang][dataset_name] = {}

            if "-" in lang:
                # code-switched data: eval 2x
                lang_code = lang.split("_")[1].lower()
            else:
                lang_code = lang

            for f, name in [
                (punkt_sentencize, "punkt"),
                (spacy_dp_sentencize, "spacy_dp"),
                (spacy_sent_sentencize, "spacy_sent"),
                (pysbd_sentencize, "pysbd"),
                (ersatz_sentencize, "ersatz"),
            ]:
                print(f"Running {name} on {dataset_name} in {lang_code}...")
                indices[lang][dataset_name][name] = {}
                if "lyrics" in dataset_name or "short" in dataset_name:
                    exclude_every_k = 0
                else:
                    exclude_every_k = args.exclude_every_k
                try:
                    if isinstance(dataset["data"][0], list):
                        all_sentences = dataset["data"]
                        metrics = []
                        for i, sentences in enumerate(all_sentences):
                            text = Constants.SEPARATORS[lang_code].join(sentences)
                            doc_metrics = {}
                            doc_metrics = evaluate_sentences(
                                lang_code,
                                sentences,
                                f(lang_code, text),
                                return_indices=True,
                                exclude_every_k=exclude_every_k,
                            )
                            f1 = doc_metrics[0]
                            doc_metrics = doc_metrics[1]
                            doc_metrics["f1"] = f1
                            doc_metrics["length"] = [doc_metrics["length"]]
                            metrics.append(doc_metrics)
                        avg_results = {}
                        concat_indices = {}
                        for doc in metrics:
                            for key, value in doc.items():
                                if not isinstance(value, list):
                                    # numeric
                                    if key not in avg_results:
                                        avg_results[key] = []

                                    avg_results[key].append(value)
                                elif isinstance(value, list):
                                    # concat
                                    if key not in concat_indices:
                                        concat_indices[key] = []
                                    if key == "length":
                                        concat_indices[key].append(value[0])
                                    else:
                                        concat_indices[key].append(value)

                        # avg
                        for key in list(avg_results):
                            if avg_results[key]:
                                avg_results[key] = sum(avg_results[key]) / len(avg_results[key])

                        # Store the results and indices
                        results[lang][dataset_name][name] = avg_results
                        indices[lang][dataset_name][name] = concat_indices
                    else:
                        # sentences = [preprocess_sentence(s) for s in dataset["data"]]
                        sentences = dataset["data"]
                        text = Constants.SEPARATORS[lang_code].join(sentences)

                        metrics = evaluate_sentences(
                            lang_code,
                            sentences,
                            f(lang_code, text),
                            return_indices=True,
                            exclude_every_k=exclude_every_k,
                        )
                        f1 = metrics[0]
                        metrics = metrics[1]
                        metrics["f1"] = f1
                        print(f1)
                        indices[lang][dataset_name][name]["true_indices"] = [metrics.pop("true_indices")]
                        indices[lang][dataset_name][name]["predicted_indices"] = [metrics.pop("predicted_indices")]
                        indices[lang][dataset_name][name]["length"] = [metrics.pop("length")]
                        results[lang][dataset_name][name] = metrics
                except LanguageError as e:
                    print("Language not supported for", name)
                    results[lang][dataset_name][name] = None

    json.dump(results, open(Constants.CACHE_DIR / "intrinsic_baselines.json", "w"), indent=4, default=int)
    json.dump(indices, open(Constants.CACHE_DIR / "intrinsic_baselines_IDX.json", "w"), indent=4, default=int)
    print(Constants.CACHE_DIR / "intrinsic_baselines.json")
