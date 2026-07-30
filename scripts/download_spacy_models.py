"""Download the spaCy models used by research baselines."""

import subprocess
import sys

SPACY_MODELS = ["xx_sent_ud_sm"]


def main() -> None:
    for model in SPACY_MODELS:
        subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)


if __name__ == "__main__":
    main()
