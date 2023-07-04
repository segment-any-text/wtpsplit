import json
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from wtpsplit.utils import Constants

all_chars = [chr(c) for c in range(0x110000)]
punctuation_chars = "".join(c for c in all_chars if "S" in unicodedata.category(c) or "P" in unicodedata.category(c))


@dataclass
class Args:
    output_dir: str = "data/sentence"
    target_chars: int = 400 * 256 * 512
    valid_ratio = 0.001


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_text_path = output_dir / "train.parquet"
    valid_text_path = output_dir / "valid.parquet"
    metadata_path = output_dir / "metadata.json"

    if not (train_text_path.exists() and valid_text_path.exists()):
        languages = [lang for lang, row in Constants.LANGINFO.iterrows()]

        chars_per_language = math.ceil(args.target_chars / len(languages))

        train_data = {
            "text": [],
            "ends_with_punctuation": [],
            "lang": [],
        }
        valid_data = {
            "text": [],
            "ends_with_punctuation": [],
            "lang": [],
        }

        char_counts = {}
        metadata = {}

        for lang_code in tqdm(languages):
            dset = load_dataset(
                "mc4",
                "iw" if lang_code == "he" else lang_code,  # patch old lang code
                streaming=True,
                split="train",
            )

            char_counts[lang_code] = 0

            bar = tqdm(total=chars_per_language, desc=f"Downloading {lang_code}...")

            ends_with_punctuation_ratio = 0.0
            whitespace_ratio = 0.0
            n_paragraphs = 0

            for sample in iter(dset):
                for paragraph in sample["text"].split("\n"):
                    stripped_paragraph = paragraph.strip()
                    paragraph = paragraph + "\n"

                    if len(paragraph.strip()) == 0:
                        continue

                    paragraph_ends_with_punctuation = len(stripped_paragraph.rstrip(punctuation_chars)) != len(
                        stripped_paragraph
                    )
                    paragraph_whitespace_ratio = sum(c.isspace() for c in paragraph) / len(paragraph)

                    ends_with_punctuation_ratio += paragraph_ends_with_punctuation
                    whitespace_ratio += paragraph_whitespace_ratio
                    n_paragraphs += 1

                    if random.random() < args.valid_ratio:
                        valid_data["text"].append(paragraph)
                        valid_data["ends_with_punctuation"].append(paragraph_ends_with_punctuation)
                        valid_data["lang"].append(lang_code)
                    else:
                        train_data["text"].append(paragraph)
                        train_data["ends_with_punctuation"].append(paragraph_ends_with_punctuation)
                        train_data["lang"].append(lang_code)

                    char_length = len(paragraph)

                    char_counts[lang_code] += char_length
                    bar.update(char_length)

                    if char_counts[lang_code] >= chars_per_language:
                        break

                if char_counts[lang_code] >= chars_per_language:
                    break

            metadata[lang_code] = {
                "ends_with_punctuation_ratio": ends_with_punctuation_ratio / n_paragraphs,
                "whitespace_ratio": whitespace_ratio / n_paragraphs,
            }

            bar.close()

        json.dump(metadata, open(metadata_path, "w"), indent=4)

        train_df = pd.DataFrame.from_dict(train_data)
        valid_df = pd.DataFrame.from_dict(valid_data)

        train_df.to_parquet(train_text_path)
        valid_df.to_parquet(valid_text_path)
