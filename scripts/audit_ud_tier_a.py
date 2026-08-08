"""Audit Tier-A UD sources vs historical SaT and current UD.

Downloads the needed CoNLL-U splits from UniversalDependencies repos into
memory, writes CSV/JSON audit artifacts, and does not leave a full UD checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import torch

from wtpsplit.evaluation import preprocess_sentence

SOURCES = [
    {
        "language_script": "eng_Latn",
        "historical_code": "en",
        "historical_declared_repo": "UD_English-GUM",
        "training_repo": "UD_English-EWT",
        "training_stem": "en_ewt",
        "evaluation_repo": "UD_English-EWT",
        "evaluation_stem": "en_ewt",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "deu_Latn",
        "historical_code": "de",
        "training_repo": "UD_German-GSD",
        "training_stem": "de_gsd",
        "evaluation_repo": "UD_German-GSD",
        "evaluation_stem": "de_gsd",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "spa_Latn",
        "historical_code": "es",
        "training_repo": "UD_Spanish-AnCora",
        "training_stem": "es_ancora",
        "evaluation_repo": "UD_Spanish-AnCora",
        "evaluation_stem": "es_ancora",
        "license": "CC BY 4.0",
    },
    {
        "language_script": "fra_Latn",
        "historical_code": "fr",
        "training_repo": "UD_French-GSD",
        "training_stem": "fr_gsd",
        "evaluation_repo": "UD_French-GSD",
        "evaluation_stem": "fr_gsd",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "rus_Cyrl",
        "historical_code": "ru",
        "historical_declared_repo": "UD_Russian-Taiga",
        "training_repo": "UD_Russian-GSD",
        "training_stem": "ru_gsd",
        "evaluation_repo": "UD_Russian-SynTagRus",
        "evaluation_stem": "ru_syntagrus",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "ukr_Cyrl",
        "historical_code": "uk",
        "training_repo": "UD_Ukrainian-IU",
        "training_stem": "uk_iu",
        "evaluation_repo": None,
        "evaluation_stem": None,
        "license": "CC BY-NC-SA 4.0",
    },
    {
        "language_script": "cmn_Hans",
        "historical_code": "zh",
        "training_repo": "UD_Chinese-GSDSimp",
        "training_stem": "zh_gsdsimp",
        "evaluation_repo": "UD_Chinese-GSD",
        "evaluation_stem": "zh_gsd",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "jpn_Jpan",
        "historical_code": "ja",
        "training_repo": "UD_Japanese-GSD",
        "training_stem": "ja_gsd",
        "evaluation_repo": "UD_Japanese-GSD",
        "evaluation_stem": "ja_gsd",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "kor_Kore",
        "historical_code": "ko",
        "training_repo": "UD_Korean-Kaist",
        "training_stem": "ko_kaist",
        "evaluation_repo": "UD_Korean-GSD",
        "evaluation_stem": "ko_gsd",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "tha_Thai",
        "historical_code": "th",
        "historical_declared_repo": "UD_Thai-PUD",
        "training_repo": "UD_Thai-TUD",
        "training_stem": "th_tud",
        "evaluation_repo": "UD_Thai-PUD",
        "evaluation_stem": "th_pud",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "hin_Deva",
        "historical_code": "hi",
        "training_repo": "UD_Hindi-HDTB",
        "training_stem": "hi_hdtb",
        "evaluation_repo": "UD_Hindi-HDTB",
        "evaluation_stem": "hi_hdtb",
        "license": "CC BY-NC-SA 4.0",
    },
    {
        "language_script": "heb_Hebr",
        "historical_code": "he",
        "training_repo": "UD_Hebrew-IAHLTwiki",
        "training_stem": "he_iahltwiki",
        "evaluation_repo": "UD_Hebrew-HTB",
        "evaluation_stem": "he_htb",
        "license": "CC BY-SA 4.0",
    },
    {
        "language_script": "hye_Armn",
        "historical_code": "hy",
        "training_repo": "UD_Armenian-BSUT",
        "training_stem": "hy_bsut",
        "evaluation_repo": "UD_Armenian-ArmTDP",
        "evaluation_stem": "hy_armtdp",
        "license": "CC BY-SA 4.0",
    },
]
SPLITS = ("train", "dev", "test")
UD_REF = "r2.18"
RAW_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    f"{{repo}}/{UD_REF}/{{stem}}-ud-{{split}}.conllu"
)


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
        default=Path("data/manifests/ud_tier_a_audit_v1.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/manifests/ud_tier_a_audit_v1.csv"),
    )
    return parser.parse_args()


def normalized_hash(text: str) -> str:
    normalized = preprocess_sentence(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_conllu(payload: str) -> dict:
    texts = []
    document_ids = set()
    for line in payload.splitlines():
        if line.startswith("# text = "):
            texts.append(line.removeprefix("# text = "))
        elif line.startswith("# newdoc id = "):
            document_ids.add(line.removeprefix("# newdoc id = "))
    hash_list = [normalized_hash(text) for text in texts]
    return {
        "sentences": len(texts),
        "documents": len(document_ids) if document_ids else None,
        "characters": sum(map(len, texts)),
        "texts": texts,
        "hashes": set(hash_list),
        "hash_list": hash_list,
    }


def fetch_split(repo: str, stem: str, split: str) -> dict | None:
    url = RAW_URL.format(repo=repo, stem=stem, split=split)
    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    result = parse_conllu(payload)
    result["url"] = url
    return result


def fetch_treebank(repo: str | None, stem: str | None):
    if repo is None or stem is None:
        return {split: None for split in SPLITS}
    return {
        split: fetch_split(repo, stem, split)
        for split in SPLITS
    }


def overlap(left, right):
    if not left or not right:
        return 0
    return len(left & right)


def license_policy(license_name: str) -> str:
    if "-NC-" in license_name:
        return "research_only_pending_release_legal_review"
    return "usable_with_attribution_and_sharealike_compliance"


def audit_source(source, historical_data):
    print(f"auditing {source['language_script']}", flush=True)
    training = fetch_treebank(
        source["training_repo"],
        source["training_stem"],
    )
    if source["evaluation_repo"] == source["training_repo"]:
        evaluation = training
    else:
        evaluation = fetch_treebank(
            source["evaluation_repo"],
            source["evaluation_stem"],
        )
    local = historical_data[source["historical_code"]]["sentence"]["ud"]
    local_train = local["meta"].get("train_data") or []
    local_eval = local.get("data") or []
    local_train_hashes = {normalized_hash(text) for text in local_train}
    local_eval_hashes = {normalized_hash(text) for text in local_eval}

    row = {
        **source,
        "historical_declared_repo": source.get(
            "historical_declared_repo",
            source["training_repo"],
        ),
        "license_url": (
            "https://github.com/UniversalDependencies/"
            f"{source['training_repo']}/blob/{UD_REF}/LICENSE.txt"
        ),
        "license_policy": license_policy(source["license"]),
        "historical_pth_train_sentences": len(local_train),
        "historical_pth_eval_sentences": len(local_eval),
    }
    for prefix, treebank in (
        ("training_source", training),
        ("evaluation_source", evaluation),
    ):
        for split in SPLITS:
            value = treebank[split]
            row[f"{prefix}_{split}_sentences"] = (
                value["sentences"] if value else 0
            )
            row[f"{prefix}_{split}_documents"] = (
                value["documents"] if value else None
            )
    for split in SPLITS:
        training_hashes = (
            training[split]["hashes"] if training[split] else set()
        )
        evaluation_hashes = (
            evaluation[split]["hashes"] if evaluation[split] else set()
        )
        row[f"pth_train_overlap_training_{split}"] = overlap(
            local_train_hashes,
            training_hashes,
        )
        row[f"pth_train_overlap_evaluation_{split}"] = overlap(
            local_train_hashes,
            evaluation_hashes,
        )
        row[f"pth_eval_overlap_evaluation_{split}"] = overlap(
            local_eval_hashes,
            evaluation_hashes,
        )

    current_train_hashes = (
        training["train"]["hashes"] if training["train"] else set()
    )
    evaluation_dev_hashes = (
        evaluation["dev"]["hashes"] if evaluation["dev"] else set()
    )
    evaluation_test_hashes = (
        evaluation["test"]["hashes"] if evaluation["test"] else set()
    )
    row["current_train_overlap_evaluation_dev"] = overlap(
        current_train_hashes,
        evaluation_dev_hashes,
    )
    row["current_train_overlap_evaluation_test"] = overlap(
        current_train_hashes,
        evaluation_test_hashes,
    )
    forbidden_hashes = evaluation_dev_hashes | evaluation_test_hashes
    row["current_train_sentences_after_eval_dedup"] = sum(
        item not in forbidden_hashes
        for item in training["train"]["hash_list"]
    )
    has_training_split = row["training_source_train_sentences"] > 0
    historical_test_overlap = row["pth_train_overlap_evaluation_test"]
    if not local_train:
        historical_reuse = "unavailable"
    elif historical_test_overlap:
        historical_reuse = "blocked_for_exact_evaluation"
    else:
        historical_reuse = "usable_with_recorded_source_caveats"
    row["historical_pth_reuse_status"] = historical_reuse

    if not has_training_split:
        readiness = "blocked_no_ud_train_split"
    elif "-NC-" in source["license"]:
        readiness = "research_only_pending_release_legal_review"
    elif (
        row["current_train_overlap_evaluation_dev"]
        or row["current_train_overlap_evaluation_test"]
    ):
        readiness = "ready_after_exact_eval_dedup"
    else:
        readiness = "ready_with_train_only_and_attribution"
    row["tier_a_readiness"] = readiness
    row["contamination_decision"] = (
        "rebuild from the current official train split; remove normalized "
        "matches to designated evaluation dev/test; never reuse evaluation data"
        if has_training_split
        else "do not use this treebank for training"
    )
    return row


def serializable(row):
    return {
        key: value
        for key, value in row.items()
        if key != "hashes"
    }


def write_csv(path: Path, rows):
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
    rows = [
        serializable(audit_source(source, historical_data))
        for source in SOURCES
    ]
    summary = {
        "version": "ud_tier_a_audit_v1",
        "ud_release_target": "v2.18/current official repositories",
        "ud_git_ref": UD_REF,
        "historical_corpus": str(args.historical_data),
        "policy": (
            "Use only official train splits. Never train on dev/test. "
            "Non-commercial sources require release-specific legal review."
        ),
        "counts": {
            "sources": len(rows),
            "ready": sum(
                row["tier_a_readiness"]
                == "ready_with_train_only_and_attribution"
                for row in rows
            ),
            "ready_after_dedup": sum(
                row["tier_a_readiness"] == "ready_after_exact_eval_dedup"
                for row in rows
            ),
            "research_only": sum(
                row["tier_a_readiness"]
                == "research_only_pending_release_legal_review"
                for row in rows
            ),
            "blocked": sum(
                row["tier_a_readiness"].startswith("blocked")
                for row in rows
            ),
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    print(json.dumps(summary["counts"], indent=2))
    for row in rows:
        print(
            row["language_script"],
            row["tier_a_readiness"],
            f"train={row['training_source_train_sentences']}",
            f"test_overlap={row['pth_train_overlap_evaluation_test']}",
        )
    print(args.output_json)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
