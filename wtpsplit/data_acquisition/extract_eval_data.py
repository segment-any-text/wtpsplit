import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import HfArgumentParser

import conllu
from datasets import load_dataset
from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.utils import Constants

UD_TREEBANK_PATH = "data/external/ud-treebanks-v2.10"  # source: https://universaldependencies.org/#download
WIKTEXTRACT_DATA_PATH = "data/external/raw-wiktextract-data.json"  # source: https://kaikki.org/dictionary/rawdata.html
# source:
# https://uni-tuebingen.de/fakultaeten/philosophische-fakultaet/fachbereiche/neuphilologie/seminar-fuer-sprachwissenschaft/arbeitsbereiche/allg-sprachwissenschaft-computerlinguistik/ressourcen/lexica/germanet-1/beschreibung/compounds/
GERMANET_DATA_PATH = "data/external/split_compounds_from_GermaNet17.0_modified-2022-06-28.txt"
AUCOPRO_DATA_PATH = "data/external/Data.AUCOPRO.Splitting"  # source: https://sourceforge.net/projects/aucopro/
ERSATZ_DATA_PATH = "data/external/ersatz-test-suite/segmented"  # source: https://github.com/rewicks/ersatz-test-suite

# copied from Table 8 in https://aclanthology.org/2021.acl-long.309.pdf
ERSATZ_TEST_DATASETS = {
    "ar": "iwsltt2017.ar",
    "cs": "wmt20.cs-en.cs",
    "de": "wmt20.de-en.de",
    "en": "wsj.03-06.en",
    "es": "wmt13.es-en.es",
    "et": "wmt18.et-en.et",
    "fi": "wmt19.fi-en.fi",
    "fr": "wmt20.fr-de.fr",
    "gu": "wmt19.gu-en.gu",
    "hi": "wmt14.hi-en.hi",
    "iu": "wmt20.iu-en.iu",
    "ja": "wmt20.ja-en.ja",
    "kk": "wmt19.kk-en.kk",
    "km": "wmt20.km-en.km",
    "lt": "wmt19.lt-en.lt",
    "lv": "wmt17.lv-en.lv",
    "pl": "wmt20.pl-en.pl",
    "ps": "wmt20.ps-en.ps",
    "ro": "wmt16.ro-en.ro",
    "ru": "wmt20.ru-en.ru",
    "ta": "wmt20.ta-en.ta",
    "tr": "wmt18.tr-en.tr",
    "zh": "wmt20.zh-en.zh",
}
ERSATZ_TRAIN_DATASETS = {
    "ar": "news-commentary-v15.dev.ar",
    "cs": "wmt18.cs-en.cs",
    "de": "wmt19.de-en.de",
    "en": "merged.nc-wsj.en",
    "es": None,
    "et": "newscrawl.2019.dev.et",
    "fi": "wmt18.fi-en.fi",
    "fr": "wmt15.fr-en.fr",
    "gu": "newscrawl.2019.dev.gu",
    "hi": "newscrawl.2013.dev.hi",
    "iu": "nhi-3.0.iu",
    "ja": "newscrawl.2019.dev.ja",
    "kk": "newscrawl.2019.dev.kk",
    "km": "wikimatrix.dev.km",
    "lt": "newscrawl.2019.dev.lt",
    "lv": "newscrawl.2019.dev.lv",
    "pl": "newscrawl.2019.dev.pl",
    "ps": "wikimatrix.dev.ps",
    "ro": "newscrawl.2019.dev.ro",
    "ru": "wmt18.ru-en.ru",
    "ta": "newscrawl.2019.dev.ta",
    "tr": "wmt16.tr-en.tr",
    "zh": "wmt18.zh-en.zh",
}


@dataclass
class Args:
    output_file: str = "data/eval.pth"
    include_train_data: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    eval_data = {lang_code: {"sentence": {}, "compound": {}} for lang_code in Constants.LANGINFO.index}

    # Ersatz data
    for lang_code in tqdm(Constants.LANGINFO.index):
        if lang_code in ERSATZ_TEST_DATASETS:
            eval_data[lang_code]["sentence"]["ersatz"] = {
                "meta": {
                    "train_data": [
                        preprocess_sentence(line)
                        for line in open(
                            os.path.join(
                                ERSATZ_DATA_PATH,
                                lang_code,
                                ERSATZ_TRAIN_DATASETS[lang_code],
                            )
                        )
                    ]
                    if args.include_train_data and ERSATZ_TRAIN_DATASETS[lang_code] is not None
                    else None,
                },
                "data": [
                    preprocess_sentence(line)
                    for line in open(os.path.join(ERSATZ_DATA_PATH, lang_code, ERSATZ_TEST_DATASETS[lang_code]))
                ],
            }

    # UD + OPUS100 sentences
    for lang_code in tqdm(Constants.LANGINFO.index):
        opus_dset_name = Constants.LANGINFO.loc[lang_code, "opus100"]

        if opus_dset_name not in (np.nan, None):
            other_lang_code = set(opus_dset_name.split("-")) - {lang_code}
            assert len(other_lang_code) == 1
            other_lang_code = other_lang_code.pop()

            dset_args = ["opus100", opus_dset_name]

            try:
                opus100_sentences = [
                    preprocess_sentence(sample[lang_code])
                    for sample in load_dataset(*dset_args, split="test")["translation"]
                    if sample[lang_code].strip() != sample[other_lang_code].strip()
                ]
                try:
                    opus100_train_sentences = [
                        preprocess_sentence(sample[lang_code])
                        for sample in load_dataset(*dset_args, split="train")["translation"]
                        if sample[lang_code].strip() != sample[other_lang_code].strip()
                    ]
                except ValueError:
                    opus100_train_sentences = None
            except ValueError:
                opus100_sentences = [
                    preprocess_sentence(sample[lang_code])
                    for sample in load_dataset(*dset_args, split="train")["translation"]
                    if sample[lang_code].strip() != sample[other_lang_code].strip()
                ]
                opus100_train_sentences = None

            eval_data[lang_code]["sentence"]["opus100"] = {
                "meta": {"train_data": opus100_train_sentences if args.include_train_data else None},
                "data": opus100_sentences,
            }

        if Constants.LANGINFO.loc[lang_code, "ud"] not in (np.nan, None):
            ud_data = conllu.parse(
                open(
                    glob.glob(
                        os.path.join(
                            UD_TREEBANK_PATH,
                            Constants.LANGINFO.loc[lang_code, "ud"],
                            "*-ud-test.conllu",
                        )
                    )[0]
                ).read()
            )

            try:
                ud_train_data = conllu.parse(
                    open(
                        glob.glob(
                            os.path.join(
                                UD_TREEBANK_PATH,
                                Constants.LANGINFO.loc[lang_code, "ud"],
                                "*-ud-train.conllu",
                            )
                        )[0]
                    ).read()
                )
                ud_train_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_train_data]
            except IndexError:
                ud_train_sentences = None

            ud_sentences = [preprocess_sentence(sentence.metadata["text"]) for sentence in ud_data]
            eval_data[lang_code]["sentence"]["ud"] = {
                "meta": {"train_data": ud_train_sentences if args.include_train_data else None},
                "data": ud_sentences,
            }

    torch.save(eval_data, args.output_file)
