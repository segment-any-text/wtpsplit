"""Clean tweet evaluation data without import-time file mutations."""

import argparse
import re
from pathlib import Path

import torch


EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)
EMOTICON_PATTERN = re.compile(r"[:;=Xx][\-oO']*[)(\[\]DdPp3><|\\/]")


def remove_emojis_and_special_chars(text: str) -> str:
    return EMOTICON_PATTERN.sub("", EMOJI_PATTERN.sub("", text))


def transform_data(data: dict) -> dict:
    def pair_sentences(sequences):
        paired_sequences = []
        for sequence in sequences:
            processed_sequence = []
            for sentence in sequence:
                words = sentence.strip().split()
                filtered_words = [
                    remove_emojis_and_special_chars(word)
                    for word in words
                    if not (word.startswith(("http", "#", "@")))
                ]
                cleaned_sentence = " ".join(filtered_words)
                if cleaned_sentence:
                    processed_sequence.append(cleaned_sentence.strip())
            if processed_sequence and len(processed_sequence) < 6:
                paired_sequences.append(processed_sequence)
        return paired_sequences

    transformed_data = {}
    for lang_code, lang_data in data.items():
        short_datasets = {
            dataset_name: {
                "meta": {"train_data": pair_sentences(content["meta"]["train_data"])},
                "data": pair_sentences(content["data"]),
            }
            for dataset_name, content in lang_data.get("sentence", {}).items()
            if "short" in dataset_name
        }
        transformed_data[lang_code] = {"sentence": short_datasets}
    return transformed_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = torch.load(args.input, weights_only=True)
    torch.save(transform_data(data), args.output)


if __name__ == "__main__":
    main()
