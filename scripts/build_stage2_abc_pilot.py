"""Merge the audited Tier-A/B and generated Tier-C pilot corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wtpsplit.train.sm_data import SUPPORTED_TRAINING_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier-ab-pth",
        type=Path,
        default=Path("data/mmsat_stage2_ab_pilot_v1.pth"),
    )
    parser.add_argument(
        "--tier-c-pth",
        type=Path,
        default=Path("data/mmsat_stage2_tier_c_forward_v1.pth"),
    )
    parser.add_argument(
        "--output-pth",
        type=Path,
        default=Path("data/mmsat_stage2_abc_pilot_v1.pth"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/manifests/mmsat_stage2_abc_pilot_v1.json"),
    )
    return parser.parse_args()


def merge_corpora(tier_ab: dict, tier_c: dict) -> dict:
    overlap = set(tier_ab) & set(tier_c)
    if overlap:
        raise ValueError(
            "A language must use exactly one highest-quality source; overlap: "
            + ", ".join(sorted(overlap))
        )
    merged = {**tier_ab, **tier_c}
    for language, language_data in merged.items():
        sentence_datasets = language_data.get("sentence", {})
        if len(sentence_datasets) != 1:
            raise ValueError(
                f"{language}: expected exactly one training dataset."
            )
        dataset_name, dataset = next(iter(sentence_datasets.items()))
        if dataset_name not in SUPPORTED_TRAINING_DATASETS:
            raise ValueError(
                f"{language}: unsupported dataset {dataset_name!r}."
            )
        train_data = dataset.get("meta", {}).get("train_data")
        if not train_data or any(
            not isinstance(sentence, str) or not sentence
            for sentence in train_data
        ):
            raise ValueError(f"{language}: invalid training sentences.")
        if dataset.get("data") != []:
            raise ValueError(
                f"{language}: pilot corpus must embed no evaluation text."
            )
    return merged


def main() -> int:
    args = parse_args()
    tier_ab = torch.load(
        args.tier_ab_pth,
        map_location="cpu",
        weights_only=True,
    )
    tier_c = torch.load(
        args.tier_c_pth,
        map_location="cpu",
        weights_only=True,
    )
    merged = merge_corpora(tier_ab, tier_c)
    args.output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_pth)
    rows = []
    for language in sorted(merged):
        dataset_name, dataset = next(
            iter(merged[language]["sentence"].items())
        )
        tier = {"ud": "A", "tatoeba": "B", "nllb": "C"}[dataset_name]
        rows.append(
            {
                "language_script": language,
                "tier": tier,
                "dataset": dataset_name,
                "sentences": len(dataset["meta"]["train_data"]),
            }
        )
    digest = hashlib.sha256(args.output_pth.read_bytes()).hexdigest()
    artifact = {
        "version": "mmsat_stage2_abc_pilot_v1",
        "status": "materialized_training_only_internal_pilot",
        "sha256": digest,
        "counts": {
            "languages": len(rows),
            "tier_a_languages": sum(row["tier"] == "A" for row in rows),
            "tier_b_languages": sum(row["tier"] == "B" for row in rows),
            "tier_c_languages": sum(row["tier"] == "C" for row in rows),
            "sentences": sum(row["sentences"] for row in rows),
        },
        "open_panel_pairs": [
            "sat_Olck",
            "chr_Cher",
            "div_Thaa",
            "roh_Latn",
        ],
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["counts"], indent=2))
    print(digest)
    print(args.output_pth)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
