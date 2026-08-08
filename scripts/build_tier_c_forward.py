"""Build Tier-C forward-translation supervision from English UD EWT Tier-A train.

Uses the evaluation-filtered English Tier-A training sentences. Each sentence
is translated on its own; the target unit keeps that single boundary (no
bilingual alignment). BOUQuET and FLORES are not training sources here.
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
from wtpsplit.data_acquisition.translation import NllbSentenceTranslator
from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_DATASET, BOUQUET_REVISION

DEFAULT_MAX_SENTENCES = 500
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
DEFAULT_LANGUAGES = [
    "bod_Tibt",
    "dzo_Tibt",
    "ibo_Latn",
    "ssw_Latn",
    "amh_Ethi",
    "arz_Arab",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pth",
        type=Path,
        default=Path("data/mmsat_stage2_ab_pilot_v1.pth"),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
    )
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES)
    parser.add_argument(
        "--mt-model",
        default="facebook/nllb-200-distilled-600M",
    )
    parser.add_argument("--mt-revision", default=NLLB_REVISION)
    parser.add_argument("--bouquet-revision", default=BOUQUET_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-pth",
        type=Path,
        default=Path("data/mmsat_stage2_tier_c_forward_v1.pth"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/manifests/mmsat_stage2_tier_c_forward_v1.json"
        ),
    )
    return parser.parse_args()


def bouquet_hashes(
    language_script: str,
    revision: str | None = None,
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


def clean_translations(
    source: list[str],
    translated: list[str],
    forbidden_hashes: set[str],
) -> tuple[list[str], dict]:
    if len(source) != len(translated):
        raise ValueError("Translation output count does not match input.")
    clean = []
    seen = set()
    empty = identical = duplicate = evaluation_overlap = 0
    for source_text, target_text in zip(source, translated, strict=True):
        source_normalized = preprocess_sentence(source_text)
        target_normalized = preprocess_sentence(target_text)
        if not target_normalized:
            empty += 1
            continue
        if source_normalized == target_normalized:
            identical += 1
            continue
        target_hash = normalized_hash(target_normalized)
        if target_hash in seen:
            duplicate += 1
            continue
        if target_hash in forbidden_hashes:
            evaluation_overlap += 1
            continue
        seen.add(target_hash)
        clean.append(target_normalized)
    return clean, {
        "empty_removed": empty,
        "source_identical_removed": identical,
        "duplicate_removed": duplicate,
        "evaluation_overlap_removed": evaluation_overlap,
    }


def build_corpus(
    source_sentences: list[str],
    languages: list[str],
    translator,
    evaluation_hashes,
) -> tuple[dict, dict]:
    corpus = {}
    statistics = {}
    for language in languages:
        translated = translator(source_sentences, language)
        clean, cleaning = clean_translations(
            source_sentences,
            translated,
            evaluation_hashes(language),
        )
        if not clean:
            raise ValueError(f"{language}: no clean translations remain.")
        corpus[language] = {
            "sentence": {
                "nllb": {
                    "meta": {
                        "train_data": clean,
                        "source": "UD_English-EWT r2.18",
                        "construction": "sentence-wise forward translation",
                    },
                    "data": [],
                }
            }
        }
        statistics[language] = {
            "source_sentences": len(source_sentences),
            **cleaning,
            "clean_train_sentences": len(clean),
        }
    return corpus, statistics


def main() -> int:
    args = parse_args()
    if args.max_sentences < 1:
        raise ValueError("--max-sentences must be positive.")
    source_corpus = torch.load(
        args.source_pth,
        map_location="cpu",
        weights_only=True,
    )
    english = source_corpus["eng_Latn"]["sentence"]["ud"]["meta"][
        "train_data"
    ]
    source_sentences = deterministic_cap(english, args.max_sentences)
    translator = NllbSentenceTranslator(
        args.mt_model,
        args.device,
        args.batch_size,
        args.max_new_tokens,
        args.local_files_only,
        args.mt_revision,
    )
    unsupported = [
        language
        for language in args.languages
        if not translator.supports(language)
    ]
    if unsupported:
        raise ValueError(
            "Requested Tier-C languages unsupported by the translator: "
            + ", ".join(unsupported)
        )
    corpus, statistics = build_corpus(
        source_sentences,
        args.languages,
        translator,
        lambda language: bouquet_hashes(language, args.bouquet_revision),
    )
    args.output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, args.output_pth)
    artifact = {
        "version": "mmsat_stage2_tier_c_forward_v1",
        "status": "internal_pilot_training_only",
        "sha256": hashlib.sha256(args.output_pth.read_bytes()).hexdigest(),
        "source": {
            "dataset": "UD_English-EWT",
            "version": "r2.18",
            "license": "CC BY-SA 4.0",
            "sentences": len(source_sentences),
            "selection": "deterministic SHA-256 cap from clean Tier A",
        },
        "hub_revisions": {
            "bouquet": args.bouquet_revision,
            "mt_model": args.mt_revision,
        },
        "construction": (
            "Translate every trusted English sentence independently and use "
            "the target unit end as an inherited boundary."
        ),
        "mt_model": args.mt_model,
        "requested_languages": args.languages,
        "statistics": statistics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(args.output_pth)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
