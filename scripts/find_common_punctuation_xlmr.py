import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import HfArgumentParser, XLMRobertaTokenizer

ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
LANGINFO = pd.read_csv(os.path.join(ROOT_DIR, "data", "language_info.csv"), index_col=0)


@dataclass
class Args:
    file: str = "data/sentence/valid.parquet"
    txt_output: str = "data/punctuation_xlmr"
    json_output: str = "data/punctuation_xlmr"
    top_n: int = 25
    include_whitespace: bool = True


tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

punctuation_pattern = re.compile(r"^▁+[^\w\s]+?$")


def is_punctuation(token, include_whitespace=False):
    # check if token matches the regular expression
    if punctuation_pattern.match(token):
        return include_whitespace
    # fallback
    return all("P" in unicodedata.category(ch) for ch in token) or all("S" in unicodedata.category(ch) for ch in token)


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    dset = load_dataset("parquet", data_files=args.file, split="train")
    counters = {lang_code: Counter() for lang_code in LANGINFO.index}

    for i in range(len(dset)):
        sample = dset[i]
        tokens = tokenizer.tokenize(sample["text"])
        counters[sample["lang"]].update(token for token in tokens if is_punctuation(token, args.include_whitespace))

    punctuation_to_include = sorted({x[0] for c in counters.values() for x in c.most_common(args.top_n)})

    json_output = f"{args.json_output}_top{args.top_n}_{'with' if args.include_whitespace else 'without'}.json"
    txt_output = f"{args.txt_output}_top{args.top_n}_{'with' if args.include_whitespace else 'without'}.txt"
    json.dump(
        {key: [x[0] for x in c.most_common(args.top_n)] for key, c in counters.items()},
        open(json_output, "w"),
        indent=4,
    )
    open(txt_output, "w").writelines([p + "\n" for p in punctuation_to_include])
