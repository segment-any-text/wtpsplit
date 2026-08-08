"""Build the capped Tier-A/B Stage-2 pilot corpus.

Tier A: pinned UD 2.18 train splits. Tier B: audited Tatoeba artifact. Drops
exact matches to designated UD eval splits and BOUQuET dev/test. Caps each
language with deterministic hash order for a small laptop pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset

from scripts.audit_ud_tier_a import SOURCES, fetch_treebank, normalized_hash
from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_DATASET, BOUQUET_REVISION

MAX_SENTENCES_PER_LANGUAGE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier-a-audit",
        type=Path,
        default=Path("data/manifests/ud_tier_a_audit_v1.json"),
    )
    parser.add_argument(
        "--tier-b-audit",
        type=Path,
        default=Path("data/manifests/tier_b_parallel_audit_v1.json"),
    )
    parser.add_argument(
        "--tier-b-pth",
        type=Path,
        default=Path("data/tier_b_tatoeba_clean_v1.pth"),
    )
    parser.add_argument(
        "--output-pth",
        type=Path,
        default=Path("data/mmsat_stage2_ab_pilot_v1.pth"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/manifests/mmsat_stage2_ab_pilot_v1.json"),
    )
    parser.add_argument("--bouquet-revision", default=BOUQUET_REVISION)
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


def deterministic_cap(
    sentences: list[str],
    limit: int = MAX_SENTENCES_PER_LANGUAGE,
) -> list[str]:
    unique = {
        preprocess_sentence(sentence)
        for sentence in sentences
        if preprocess_sentence(sentence)
    }
    return sorted(
        unique,
        key=lambda text: (
            hashlib.sha256(text.encode("utf-8")).digest(),
            text,
        ),
    )[:limit]


def clean_tier_a(
    source: dict,
    bouquet_revision: str | None = None,
) -> tuple[list[str], dict]:
    training = fetch_treebank(
        source["training_repo"],
        source["training_stem"],
    )
    if not training["train"]:
        raise ValueError(
            f"No official train split for {source['language_script']}."
        )
    if source["evaluation_repo"] == source["training_repo"]:
        evaluation = training
    else:
        evaluation = fetch_treebank(
            source["evaluation_repo"],
            source["evaluation_stem"],
        )
    forbidden = bouquet_hashes(
        source["language_script"],
        revision=bouquet_revision,
    )
    for split in ("dev", "test"):
        if evaluation[split]:
            forbidden.update(evaluation[split]["hashes"])
    raw = training["train"]["texts"]
    clean = [
        preprocess_sentence(text)
        for text in raw
        if normalized_hash(text) not in forbidden
    ]
    unique_count = len(set(clean))
    capped = deterministic_cap(clean)
    return capped, {
        "raw_sentences": len(raw),
        "after_evaluation_exclusion": len(clean),
        "unique_after_evaluation_exclusion": unique_count,
        "materialized_sentences": len(capped),
    }


def main() -> int:
    args = parse_args()
    tier_a_artifact = json.loads(
        args.tier_a_audit.read_text(encoding="utf-8")
    )
    tier_b_artifact = json.loads(
        args.tier_b_audit.read_text(encoding="utf-8")
    )
    ready_a = {
        row["language_script"]: row
        for row in tier_a_artifact["rows"]
        if row["tier_a_readiness"].startswith("ready")
    }
    ready_b = {
        row["language_script"]: row
        for row in tier_b_artifact["rows"]
        if row["selected_training_tier_v2"] == "B"
    }
    sources = {
        source["language_script"]: source
        for source in SOURCES
    }
    tier_b_corpus = torch.load(
        args.tier_b_pth,
        map_location="cpu",
        weights_only=True,
    )

    corpus = {}
    rows = []
    for language_script in sorted(ready_a):
        source = sources[language_script]
        print(f"materializing Tier A {language_script}", flush=True)
        sentences, stats = clean_tier_a(
            source,
            bouquet_revision=args.bouquet_revision,
        )
        corpus[language_script] = {
            "sentence": {
                "ud": {
                    "meta": {
                        "train_data": sentences,
                        "source": source["training_repo"],
                        "version": "UD r2.18",
                        "license": ready_a[language_script]["license"],
                    },
                    "data": [],
                }
            }
        }
        rows.append(
            {
                "language_script": language_script,
                "tier": "A",
                "dataset": "ud",
                "source": source["training_repo"],
                **stats,
            }
        )

    for language_script in sorted(ready_b):
        print(f"materializing Tier B {language_script}", flush=True)
        source_dataset = tier_b_corpus[language_script]["sentence"]["tatoeba"]
        raw = source_dataset["meta"]["train_data"]
        sentences = deterministic_cap(raw)
        corpus[language_script] = {
            "sentence": {
                "tatoeba": {
                    "meta": {
                        **source_dataset["meta"],
                        "train_data": sentences,
                    },
                    "data": [],
                }
            }
        }
        rows.append(
            {
                "language_script": language_script,
                "tier": "B",
                "dataset": "tatoeba",
                "source": "OPUS Tatoeba",
                "raw_sentences": ready_b[language_script][
                    "raw_alignment_pairs"
                ],
                "after_evaluation_exclusion": len(raw),
                "unique_after_evaluation_exclusion": len(set(raw)),
                "materialized_sentences": len(sentences),
            }
        )

    args.output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, args.output_pth)
    artifact = {
        "version": "mmsat_stage2_ab_pilot_v1",
        "status": "materialized_training_only_no_evaluation_text",
        "sha256": hashlib.sha256(args.output_pth.read_bytes()).hexdigest(),
        "bouquet_revision": args.bouquet_revision,
        "policy": {
            "max_sentences_per_language": MAX_SENTENCES_PER_LANGUAGE,
            "selection": "deterministic SHA-256 ordering after deduplication",
            "evaluation_exclusion": (
                "Exact normalized matches to designated UD dev/test and "
                "BOUQuET dev/test are removed."
            ),
            "tier_b_release": (
                "Internal pilot only until attribution and redistribution "
                "review is complete."
            ),
        },
        "counts": {
            "languages": len(rows),
            "tier_a_languages": sum(row["tier"] == "A" for row in rows),
            "tier_b_languages": sum(row["tier"] == "B" for row in rows),
            "materialized_sentences": sum(
                row["materialized_sentences"] for row in rows
            ),
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["counts"], indent=2))
    print(args.output_pth)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
