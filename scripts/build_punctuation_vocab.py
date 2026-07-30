"""Build tokenizer-specific punctuation files for training."""

import argparse
import re
from pathlib import Path

from tokenizers import AddedToken
from transformers import AutoTokenizer

from wtpsplit.utils import Constants


_TOKENIZER_FILE_TAGS = {
    "facebookai/xlm-roberta-base": "xlmr",
    "xlm-roberta-base": "xlmr",
    "jhu-clsp/mmbert-base": "mmbert",
    "mmbert-base": "mmbert",
}


def punctuation_file_tag(tokenizer_name: str) -> str:
    """Return the filename tag associated with a tokenizer identifier."""
    normalized_name = tokenizer_name.strip().rstrip("/").casefold()
    if tag := _TOKENIZER_FILE_TAGS.get(normalized_name):
        return tag

    tag = re.sub(r"[^a-z0-9]+", "_", Path(normalized_name).name).strip("_")
    if not tag:
        raise ValueError(f"Cannot derive a punctuation filename from {tokenizer_name!r}.")
    return tag


def write_punctuation_files(tokenizer_name: str, output_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = punctuation_file_tag(tokenizer_name)

    known = []
    unknown_added = False
    with (output_dir / f"punctuation_{tag}_unk.txt").open("w", encoding="utf-8") as with_unknown:
        for character in Constants.PUNCTUATION_CHARS:
            token_id = tokenizer.convert_tokens_to_ids(character)
            if token_id != tokenizer.unk_token_id:
                known.append(character)
                with_unknown.write(f"{character}\n")
            elif not unknown_added:
                with_unknown.write("<unk>\n")
                unknown_added = True

    (output_dir / f"punctuation_{tag}.txt").write_text(
        "".join(f"{character}\n" for character in known),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="facebookAI/xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, default=Constants.ROOT_DIR / "data")
    args = parser.parse_args()
    write_punctuation_files(args.tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
