from pathlib import Path

import torch
from mosestokenizer import MosesTokenizer
from tqdm.auto import tqdm

from wtpsplit.utils import Constants


def process_tsv(file_path, detokenizer):
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            word, boundary = parts[0], parts[1]
            current_sentence.append(word)
            if boundary == "1":
                detokenized_sentence = detokenizer.detokenize(current_sentence)
                # detokenized_sentence = ' '.join(current_sentence)
                sentences.append(detokenized_sentence)
                current_sentence = []

    return sentences


def build_data_dictionary(root_dir):
    data_dict = {}

    for lang in tqdm(["fr", "de", "en", "it"]):
        detokenizer = MosesTokenizer(lang)
        data_dict[lang] = {"sentence": {}}
        for dataset in ["test", "surprise_test"]:
            data_dict[lang]["sentence"][dataset] = {"meta": {"train_data": []}, "data": []}

        train_data_path = Path(root_dir) / lang / "train"
        train_files = sorted([f for f in train_data_path.glob("*.tsv") if f.is_file()])
        all_train_sentences = []
        for file_path in tqdm(train_files, desc=f"{lang} train"):
            train_sentences = process_tsv(file_path, detokenizer)
            all_train_sentences.append(train_sentences)

        # use train data for both test sets (same training data)
        for dataset in ["test", "surprise_test"]:
            data_dict[lang]["sentence"][dataset]["meta"]["train_data"] = all_train_sentences

        # test + surprise_test data
        for dataset in ["test", "surprise_test"]:
            test_data_path = Path(root_dir) / lang / dataset
            test_files = sorted([f for f in test_data_path.glob("*.tsv") if f.is_file()])
            all_test_sentences = []
            for file_path in tqdm(test_files, desc=f"{lang} {dataset}"):
                test_sentences = process_tsv(file_path, detokenizer)
                all_test_sentences.append(test_sentences)
            data_dict[lang]["sentence"][dataset]["data"] = all_test_sentences

    return data_dict


# this must be downloaded first from the linked drive here
# https://sites.google.com/view/sentence-segmentation
root_dir = Constants.ROOT_DIR.parent / "data/sepp_nlg_2021_data"
data = build_data_dictionary(root_dir)

torch.save(data, Constants.ROOT_DIR.parent / "data" / "ted2020_join.pth")
