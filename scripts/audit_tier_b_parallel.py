"""Audit and write Tier-B Tatoeba supervision for the pilot.

Historical SaT used OPUS-100, which mixes upstream OPUS corpora without
per-example license provenance, so it is a compatibility source only. The
paper-training stand-in is a pinned Tatoeba release (CC BY 2.0 FR).

Downloads small pair archives only. Deduplicates targets, drops exact matches
to local historical eval and BOUQuET dev/test, writes CSV/JSON audits plus a
compact training ``.pth``, and does not keep the zips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

import torch
from datasets import load_dataset

from wtpsplit.evaluation import preprocess_sentence
from wtpsplit.evaluation.diagnostics.boundary_ceiling import BOUQUET_DATASET, BOUQUET_REVISION

TATOEBA_VERSION = "v2026-07-08"
TATOEBA_LICENSE = "CC BY 2.0 FR"
TATOEBA_LICENSE_URL = "https://tatoeba.org/en/terms_of_use"
MIN_UNIQUE_SENTENCES = 500
ARCHIVE_URL = (
    "https://object.pouta.csc.fi/OPUS-Tatoeba/"
    f"{TATOEBA_VERSION}/moses/{{pair}}.txt.zip"
)

SOURCES = [
    {
        "language_script": "ukr_Cyrl",
        "historical_code": "uk",
        "pair": "en-uk",
        "source_code": "en",
        "target_code": "uk",
    },
    {
        "language_script": "khm_Khmr",
        "historical_code": "km",
        "pair": "en-km",
        "source_code": "en",
        "target_code": "km",
    },
    {
        "language_script": "ibo_Latn",
        "historical_code": "ig",
        "pair": "en-ig",
        "source_code": "en",
        "target_code": "ig",
    },
    {
        "language_script": "amh_Ethi",
        "historical_code": "am",
        "pair": "am-en",
        "source_code": "en",
        "target_code": "am",
    },
    {
        "language_script": "hin_Deva",
        "historical_code": "hi",
        "pair": "en-hi",
        "source_code": "en",
        "target_code": "hi",
    },
    {
        "language_script": "kat_Geor",
        "historical_code": "ka",
        "pair": "en-ka",
        "source_code": "en",
        "target_code": "ka",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-data",
        type=Path,
        default=Path("data/all_data_11_05-all.pth"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/manifests/tier_b_parallel_audit_v1.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/manifests/tier_b_parallel_audit_v1.csv"),
    )
    parser.add_argument(
        "--output-pth",
        type=Path,
        default=Path("data/tier_b_tatoeba_clean_v1.pth"),
    )
    parser.add_argument("--bouquet-revision", default=BOUQUET_REVISION)
    return parser.parse_args()


def normalize(text: str) -> str:
    return preprocess_sentence(text)


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def iter_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_strings(item)


def historical_evaluation_hashes(language_data) -> set[str]:
    hashes = set()
    for dataset in language_data.get("sentence", {}).values():
        if not isinstance(dataset, dict):
            continue
        for text in iter_strings(dataset.get("data", [])):
            if normalize(text):
                hashes.add(normalized_hash(text))
    return hashes


def historical_opus_stats(language_data) -> dict:
    dataset = language_data.get("sentence", {}).get("opus100", {})
    train = dataset.get("meta", {}).get("train_data") or []
    evaluation = dataset.get("data") or []
    train_hashes = {
        normalized_hash(text)
        for text in train
        if isinstance(text, str) and normalize(text)
    }
    evaluation_hashes = {
        normalized_hash(text)
        for text in evaluation
        if isinstance(text, str) and normalize(text)
    }
    return {
        "historical_opus100_train_sentences": len(train),
        "historical_opus100_unique_train_sentences": len(train_hashes),
        "historical_opus100_eval_sentences": len(evaluation),
        "historical_opus100_train_eval_overlap": len(
            train_hashes & evaluation_hashes
        ),
    }


def bouquet_evaluation_hashes(
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
            if normalize(text)
        )
    return hashes


def parse_tatoeba_archive(
    payload: bytes,
    pair: str,
    source_code: str,
    target_code: str,
) -> tuple[list[tuple[str, str]], str]:
    prefix = f"Tatoeba.{pair}"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        source_lines = (
            archive.read(f"{prefix}.{source_code}")
            .decode("utf-8")
            .splitlines()
        )
        target_lines = (
            archive.read(f"{prefix}.{target_code}")
            .decode("utf-8")
            .splitlines()
        )
        readme = archive.read("README").decode("utf-8")
    if len(source_lines) != len(target_lines):
        raise ValueError(f"Unaligned Tatoeba archive for {pair}.")
    if TATOEBA_LICENSE not in readme:
        raise ValueError(
            f"Pinned license string missing from Tatoeba archive for {pair}."
        )
    return list(zip(source_lines, target_lines, strict=True)), readme


def fetch_tatoeba(source: dict) -> tuple[list[tuple[str, str]], str, str]:
    url = ARCHIVE_URL.format(pair=source["pair"])
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    pairs, readme = parse_tatoeba_archive(
        payload,
        source["pair"],
        source["source_code"],
        source["target_code"],
    )
    return pairs, readme, url


def clean_pairs(
    pairs: list[tuple[str, str]],
    forbidden_hashes: set[str],
) -> tuple[list[str], dict]:
    accepted = []
    seen = set()
    empty = identical = duplicate = evaluation_overlap = 0
    for source_text, target_text in pairs:
        source_normalized = normalize(source_text)
        target_normalized = normalize(target_text)
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
        accepted.append(target_normalized)
    stats = {
        "empty_target_removed": empty,
        "source_target_identical_removed": identical,
        "duplicate_target_removed": duplicate,
        "evaluation_overlap_removed": evaluation_overlap,
    }
    return accepted, stats


def audit_source(
    source: dict,
    historical_data,
    bouquet_revision: str | None = BOUQUET_REVISION,
) -> tuple[dict, list[str]]:
    language_script = source["language_script"]
    print(f"auditing {language_script}", flush=True)
    language_data = historical_data.get(source["historical_code"], {})
    local_eval_hashes = historical_evaluation_hashes(language_data)
    bouquet_hashes = bouquet_evaluation_hashes(language_script, bouquet_revision)
    pairs, _, archive_url = fetch_tatoeba(source)
    clean, cleaning = clean_pairs(
        pairs,
        local_eval_hashes | bouquet_hashes,
    )
    ready = len(clean) >= MIN_UNIQUE_SENTENCES
    lengths = [len(text) for text in clean]
    row = {
        **source,
        "historical_source": "Helsinki-NLP/opus-100",
        "historical_source_decision": (
            "compatibility_only_missing_per_example_upstream_provenance"
        ),
        **historical_opus_stats(language_data),
        "flores_decision": "blocked_evaluation_only_terms",
        "replacement_source": "OPUS Tatoeba",
        "replacement_version": TATOEBA_VERSION,
        "replacement_archive_url": archive_url,
        "replacement_license": TATOEBA_LICENSE,
        "replacement_license_url": TATOEBA_LICENSE_URL,
        "raw_alignment_pairs": len(pairs),
        **cleaning,
        "clean_unique_train_sentences": len(clean),
        "median_characters": statistics.median(lengths) if lengths else 0,
        "tier_b_readiness": (
            "ready_internal_pilot_attribution_review_before_redistribution"
            if ready
            else "insufficient_volume_route_to_tier_c"
        ),
        "selected_training_tier_v2": "B" if ready else "C",
    }
    return row, clean


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    historical_data = torch.load(
        args.historical_data,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    rows = []
    corpus = {}
    for source in SOURCES:
        row, clean = audit_source(source, historical_data, args.bouquet_revision)
        rows.append(row)
        if row["selected_training_tier_v2"] == "B":
            corpus[source["language_script"]] = {
                "sentence": {
                    "tatoeba": {
                        "meta": {
                            "train_data": clean,
                            "source": "OPUS Tatoeba",
                            "version": TATOEBA_VERSION,
                            "license": TATOEBA_LICENSE,
                        },
                        "data": [],
                    }
                }
            }

    artifact = {
        "version": "tier_b_parallel_audit_v1",
        "bouquet_revision": args.bouquet_revision,
        "policy": {
            "pilot_floor_unique_sentences": MIN_UNIQUE_SENTENCES,
            "historical_opus100": (
                "Compatibility only: heterogeneous upstream provenance is not "
                "retained per example."
            ),
            "flores": "Evaluation-only access terms prohibit model training.",
            "tatoeba": (
                "Internal pilot after exact evaluation exclusion; complete "
                "attribution/release review before redistributing derived text."
            ),
        },
        "counts": {
            "sources_audited": len(rows),
            "tier_b_ready": sum(
                row["selected_training_tier_v2"] == "B" for row in rows
            ),
            "routed_to_tier_c": sum(
                row["selected_training_tier_v2"] == "C" for row in rows
            ),
            "clean_tier_b_sentences": sum(
                row["clean_unique_train_sentences"]
                for row in rows
                if row["selected_training_tier_v2"] == "B"
            ),
        },
        "rows": rows,
    }
    write_csv(args.output_csv, rows)
    args.output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, args.output_pth)
    artifact["sha256"] = hashlib.sha256(args.output_pth.read_bytes()).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["counts"], indent=2))
    for row in rows:
        print(
            row["language_script"],
            row["tier_b_readiness"],
            f"clean={row['clean_unique_train_sentences']}",
        )
    print(args.output_json)
    print(args.output_csv)
    print(args.output_pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
