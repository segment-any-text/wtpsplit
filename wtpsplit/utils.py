import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from iso639 import languages

ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

NEWLINE_INDEX = 0
AUX_OFFSET = 1

LANGINFO = pd.read_csv(os.path.join(ROOT_DIR, "data", "language_info.csv"), index_col=0)
PUNCTUATION_CHARS = [x.strip() for x in open(os.path.join(ROOT_DIR, "data", "punctuation.txt")).readlines()]
PUNCTUATION_MAP = json.load(open(os.path.join(ROOT_DIR, "data", "punctuation.json")))
LANG_CODE_TO_INDEX = {lang: i for i, lang in enumerate(LANGINFO.index)}
SEPARATORS = {lang: ("" if row["no_whitespace"] else " ") for lang, row in LANGINFO.iterrows()}


@dataclass
class LabelArgs:
    newline_remove_prob: float = 1.0
    auxiliary_remove_prob: float = 0.5
    hyphen_remove_prob: float = 0.9
    newline_whitespace_prob: float = 0.99
    hyphen_smooth_prob: float = 0.9
    newline_chars: List[str] = field(default_factory=lambda: ["\n"])
    auxiliary_chars: List[str] = field(default_factory=lambda: PUNCTUATION_CHARS.copy())
    hyphen_chars: List[str] = field(default_factory=lambda: ["-", "‐"])
    use_auxiliary: bool = False


def get_label_dict(label_args):
    label_dict = {}

    for i, c in enumerate(label_args.auxiliary_chars):
        label_dict[ord(c)] = 1 + AUX_OFFSET + i

    for c in label_args.newline_chars:
        label_dict[ord(c)] = 1 + NEWLINE_INDEX

    return label_dict


def encode(text):
    return [ord(c) for c in text]


def label(input_ids, label_dict):
    return [label_dict.get(input_id, 0) for input_id in input_ids[1:]] + [0]


def lang_code_to_lang(lang_code):
    if lang_code == "el":
        return "Greek"
    elif lang_code == "tl":
        return "Filipino"
    elif lang_code == "ug":
        return "Uyghur"

    try:
        return languages.get(alpha2=lang_code).name
    except KeyError:
        return languages.get(part3=lang_code).name


def corrupt(
    input_ids,
    block_ids,
    lang,
    label_args,
    label_dict,
    pack_samples=False,
    min_length=None,
):
    input_ids = input_ids.copy()
    block_ids = block_ids.copy()
    labels = label(input_ids, label_dict)

    separator = SEPARATORS[lang]

    try:
        i = next(index for index, label in enumerate(labels) if label != 0)
    except StopIteration:
        return input_ids, block_ids, labels

    while True:
        if min_length is not None and len(input_ids) <= min_length:
            break

        if labels[i] == NEWLINE_INDEX + 1:
            if random.random() < label_args.newline_remove_prob:
                if separator == " " and random.random() < label_args.newline_whitespace_prob:
                    input_ids[i + 1] = ord(" ")
                else:
                    del input_ids[i + 1]
                    del labels[i + 1]

                    if pack_samples:
                        last_index_in_block = i
                        while last_index_in_block + 1 == len(block_ids) or (
                            last_index_in_block < len(block_ids) and block_ids[last_index_in_block + 1] == block_ids[i]
                        ):
                            last_index_in_block += 1
                        input_ids.insert(last_index_in_block, 0)
                        labels.insert(last_index_in_block, 0)
                    else:
                        del block_ids[i + 1]
        elif label_args.use_auxiliary and labels[i] > AUX_OFFSET:  # auxiliary
            if pack_samples:
                raise NotImplementedError()

            if random.random() < label_args.auxiliary_remove_prob:
                del input_ids[i + 1]
                del labels[i + 1]
                del block_ids[i + 1]

        try:
            i = i + 1 + next(index for index, label in enumerate(labels[i + 1 :]) if label != 0)
        except StopIteration:
            break

    return input_ids, block_ids, labels


def indices_to_sentences(text, indices):
    sentences = []

    offset = 0
    idx = 0
    for idx in indices:
        idx = idx + 1
        while idx < len(text) and text[idx].isspace():
            idx += 1

        sentences.append(text[offset:idx])
        offset = idx

    if idx != len(text):
        sentences.append(text[idx:])

    return sentences


def reconstruct_sentences(text, partial_sentences):
    # for consistency
    partial_sentences = [x.strip() for x in partial_sentences]

    fixed_sentences = []
    i = 0
    for x in partial_sentences:
        try:
            idx = i + text[i:].index(x[:-1])
        except ValueError:
            reduced = x[:-2]

            while len(reduced) > 0 and not reduced.isspace():
                idx = i + text[i:].find(reduced)

                if idx == -1:
                    reduced = reduced[:-1]
                else:
                    break

            if idx == -1:
                raise ValueError("irrecoverable error in reconstruction")

        if idx > i:
            fixed_sentences.append(text[i:idx])
            i = idx

    if i != len(text):
        fixed_sentences.append(text[i:])

    return fixed_sentences
