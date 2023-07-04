import json
from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm
from transformers import HfArgumentParser

from wtpsplit.evaluation import (LanguageError, ersatz_sentencize,
                                 evaluate_sentences, preprocess_sentence,
                                 punkt_sentencize, pysbd_sentencize,
                                 spacy_dp_sentencize, spacy_sent_sentencize)
from wtpsplit.utils import Constants


@dataclass
class Args:
    eval_data_path: str = "data/eval.pth"
    include_langs: List[str] = None


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    eval_data = torch.load(args.eval_data_path)
    results = {}

    for lang_code, lang_data in tqdm(eval_data.items()):
        if args.include_langs is not None and lang_code not in args.include_langs:
            continue

        results[lang_code] = {}

        for dataset_name, dataset in lang_data["sentence"].items():
            results[lang_code][dataset_name] = {}

            sentences = [preprocess_sentence(s) for s in dataset["data"]]

            text = Constants.SEPARATORS[lang_code].join(sentences)

            for f, name in [
                (punkt_sentencize, "punkt"),
                (spacy_dp_sentencize, "spacy_dp"),
                (spacy_sent_sentencize, "spacy_sent"),
                (pysbd_sentencize, "pysbd"),
                (ersatz_sentencize, "ersatz"),
            ]:
                print(f"Running {name} on {dataset_name} in {lang_code}...")
                try:
                    results[lang_code][dataset_name][name] = evaluate_sentences(
                        lang_code, sentences, f(lang_code, text)
                    )
                except LanguageError:
                    results[lang_code][dataset_name][name] = None

    json.dump(results, open(Constants.CACHE_DIR / "intrinsic_baselines.json", "w"), indent=4)
