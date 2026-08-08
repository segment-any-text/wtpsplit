"""Build the equal-step historical-supervision control for the Stage-2 pilot.

Maps the historical SaT `.pth` onto the pilot language-script ids. Use only as a
compatibility baseline: provenance and eval separation are not good enough for
new paper training data. Exact BOUQuET matches are still removed so the control
does not train on the pilot evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset

from scripts.audit_tier_b_parallel import normalized_hash
from scripts.build_stage2_ab_pilot import deterministic_cap
from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_DATASET, BOUQUET_REVISION
from wtpsplit.train.sm_data import select_training_dataset


# Historical SaT keys do not use the language-script identifiers used by the
# Stage 2 builders. Keep this compatibility mapping beside the only code that
# needs it; it is not the language inventory for future scaling.
HISTORICAL_CODES = {
    "eng_Latn": "en",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "rus_Cyrl": "ru",
    "ukr_Cyrl": "uk",
    "cmn_Hans": "zh",
    "jpn_Jpan": "ja",
    "kor_Kore": "ko",
    "tha_Thai": "th",
    "khm_Khmr": "km",
    "ibo_Latn": "ig",
    "amh_Ethi": "am",
    "hin_Deva": "hi",
    "heb_Hebr": "he",
    "hye_Armn": "hy",
    "kat_Geor": "ka",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-data",
        type=Path,
        default=Path("data/all_data_11_05-all.pth"),
    )
    parser.add_argument(
        "--expanded-manifest",
        type=Path,
        default=Path("data/manifests/mmsat_stage2_abc_pilot_v1.json"),
    )
    parser.add_argument("--max-sentences", type=int, default=10_000)
    parser.add_argument("--bouquet-revision", default=BOUQUET_REVISION)
    parser.add_argument(
        "--output-pth",
        type=Path,
        default=Path("data/mmsat_stage2_historical_control_v1.pth"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/manifests/mmsat_stage2_historical_control_v1.json"
        ),
    )
    return parser.parse_args()


def bouquet_hashes(
    language_script: str,
    revision: str | None = BOUQUET_REVISION,
) -> set[str]:
    hashes = set()
    for split in ("dev", "test"):
        dataset = load_dataset(
            BOUQUET_DATASET,
            language_script,
            split=split,
            revision=revision,
        )
        hashes.update(
            normalized_hash(text)
            for text in dataset["src_text"]
            if preprocess_sentence(text)
        )
    return hashes


def build_control(
    historical_data: dict,
    expanded_languages: set[str],
    max_sentences: int,
    evaluation_hashes,
) -> tuple[dict, list[dict], list[str]]:
    corpus = {}
    rows = []
    missing = []
    for language_script in sorted(expanded_languages):
        historical_code = HISTORICAL_CODES.get(language_script)
        if not historical_code or historical_code not in historical_data:
            missing.append(language_script)
            continue
        sentence_datasets = historical_data[historical_code].get(
            "sentence",
            {},
        )
        dataset_name = select_training_dataset(sentence_datasets)
        if dataset_name is None:
            missing.append(language_script)
            continue
        raw = sentence_datasets[dataset_name]["meta"]["train_data"]
        forbidden = evaluation_hashes(language_script)
        clean = [
            preprocess_sentence(text)
            for text in raw
            if isinstance(text, str)
            and preprocess_sentence(text)
            and normalized_hash(text) not in forbidden
        ]
        materialized = deterministic_cap(clean, max_sentences)
        corpus[language_script] = {
            "sentence": {
                dataset_name: {
                    "meta": {
                        "train_data": materialized,
                        "source": "historical SaT compatibility corpus",
                    },
                    "data": [],
                }
            }
        }
        rows.append(
            {
                "language_script": language_script,
                "historical_code": historical_code,
                "dataset": dataset_name,
                "raw_sentences": len(raw),
                "materialized_sentences": len(materialized),
            }
        )
    return corpus, rows, missing


def main() -> int:
    args = parse_args()
    historical = torch.load(
        args.historical_data,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    expanded = json.loads(
        args.expanded_manifest.read_text(encoding="utf-8")
    )
    expanded_languages = {
        row["language_script"] for row in expanded["rows"]
    }
    corpus, rows, missing = build_control(
        historical,
        expanded_languages,
        args.max_sentences,
        lambda language: bouquet_hashes(language, args.bouquet_revision),
    )
    args.output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, args.output_pth)
    artifact = {
        "version": "mmsat_stage2_historical_control_v1",
        "status": "compatibility_control_not_release_training_data",
        "sha256": hashlib.sha256(args.output_pth.read_bytes()).hexdigest(),
        "bouquet_revision": args.bouquet_revision,
        "counts": {
            "languages": len(rows),
            "sentences": sum(
                row["materialized_sentences"] for row in rows
            ),
        },
        "missing_expanded_languages": missing,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["counts"], indent=2))
    print("missing", ", ".join(missing))
    print(args.output_pth)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
