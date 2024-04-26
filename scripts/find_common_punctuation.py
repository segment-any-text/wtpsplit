from collections import Counter
from transformers import HfArgumentParser
from dataclasses import dataclass
from datasets import load_dataset
import unicodedata
import json
import pickle
import pandas as pd
import os
from pathlib import Path


ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


LANGINFO = pd.read_csv(os.path.join(ROOT_DIR, "data", "language_info.csv"), index_col=0)


@dataclass
class Args:
    file: str = "data/sentence/train.parquet"
    txt_output: str = "data/punctuation.txt"
    json_output: str = "data/punctuation.json"
    counter_output: str = "data/punctuation_counter.pkl"
    top_n: int = 30


all_chars = [chr(c) for c in range(0x110000)]
punctuation_chars = set(c for c in all_chars if "S" in unicodedata.category(c) or "P" in unicodedata.category(c))

if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    dset = load_dataset("parquet", data_files=args.file, split="train")

    counters = {lang_code: Counter() for lang_code in LANGINFO.index}

    for i in range(len(dset)):
        sample = dset[i]

        counters[sample["lang"]].update(c for c in sample["text"] if c in punctuation_chars)

    punctuation_to_include = sorted({x[0] for c in counters.values() for x in c.most_common(args.top_n)})

    json.dump(
        {key: [x[0] for x in c.most_common(args.top_n)] for key, c in counters.items()},
        open(args.json_output, "w"),
        indent=4,
    )
    pickle.dump(counters, open(args.counter_output, "wb"))
    open(args.txt_output, "w").writelines([p + "\n" for p in punctuation_to_include])
