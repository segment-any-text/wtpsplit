"""Create the minimal nested dataset used by research examples."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path("dummy-dataset.pth"))
    args = parser.parse_args()

    torch.save(
        {
            "language_code": {
                "sentence": {
                    "dummy-dataset": {
                        "meta": {"train_data": ["train sentence 1", "train sentence 2"]},
                        "data": ["train sentence 1", "train sentence 2"],
                    }
                }
            }
        },
        args.output,
    )


if __name__ == "__main__":
    main()
