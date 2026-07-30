"""Frozen, model-agnostic evaluation contract for mmSaT experiments.

Rows use character offsets *between* characters.  A score at position ``i`` in
``scores`` is the probability of a boundary at offset ``i + 1``.  The last
character is never scored because the document-final boundary is implicit.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

DEFAULT_GRID = (0.005, 0.01, 0.02, 0.025, 0.03, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row.setdefault("_line", line_number)
                rows.append(row)
    return rows


def validate_rows(rows: Iterable[dict]) -> list[dict]:
    rows = list(rows)
    seen = set()
    for row in rows:
        missing = {"id", "language_script", "split", "reference_type", "text", "scores", "gold_offsets"} - set(row)
        if missing:
            raise ValueError(f"Row {row.get('id', row.get('_line'))!r} misses {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate document id: {row['id']}")
        seen.add(row["id"])
        if row["split"] not in {"dev", "test"}:
            raise ValueError(f"{row['id']}: split must be dev or test")
        if row["reference_type"] not in {"exact", "partial"}:
            raise ValueError(f"{row['id']}: reference_type must be exact or partial")
        if len(row["scores"]) not in {max(0, len(row["text"]) - 1), len(row["text"])}:
            raise ValueError(f"{row['id']}: scores must have len(text)-1 entries (len(text) is tolerated)")
        offsets = row["gold_offsets"]
        if offsets != sorted(set(offsets)) or any(not 0 < x < len(row["text"]) for x in offsets):
            raise ValueError(f"{row['id']}: gold_offsets must be unique, sorted, and internal")
    return rows


def predicted_offsets(row: dict, threshold: float) -> set[int]:
    return {index + 1 for index, score in enumerate(row["scores"][: max(0, len(row["text"]) - 1)]) if score >= threshold}


def counts(rows: Iterable[dict], thresholds: dict[str, float]) -> dict[str, int]:
    tp = fp = fn = 0
    for row in rows:
        predicted = predicted_offsets(row, thresholds[row["language_script"]])
        gold = set(row["gold_offsets"])
        tp += len(predicted & gold)
        fn += len(gold - predicted)
        if row["reference_type"] == "exact":
            fp += len(predicted - gold)
    return {"tp": tp, "fp": fp, "fn": fn}


def metrics_from_counts(value: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = value["tp"], value["fp"], value["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**value, "precision": precision, "recall": recall, "f1": f1}


def _f1(rows: list[dict], threshold: float) -> float:
    languages = {row["language_script"] for row in rows}
    return float(metrics_from_counts(counts(rows, {language: threshold for language in languages}))["f1"])


def fit_thresholds(rows: list[dict], mode: str = "global", grid: Iterable[float] = DEFAULT_GRID) -> dict[str, float]:
    """Fit on exact-reference development rows only; test and partial rows cannot leak."""
    train = [row for row in rows if row["split"] == "dev" and row["reference_type"] == "exact"]
    if not train:
        raise ValueError("Threshold fitting requires at least one exact-reference dev row")
    languages = sorted({row["language_script"] for row in rows})
    grid = tuple(float(value) for value in grid)

    def best(subset: list[dict]) -> float:
        # Stable tie break prefers the larger threshold (fewer false positives).
        return max(grid, key=lambda threshold: (_f1(subset, threshold), threshold))

    global_threshold = best(train)
    if mode == "global":
        return {language: global_threshold for language in languages}
    if mode != "per-language":
        raise ValueError("threshold mode must be global or per-language")
    output = {}
    for language in languages:
        subset = [row for row in train if row["language_script"] == language]
        output[language] = best(subset) if subset else global_threshold
    return output


def _partial_recall(rows: list[dict], thresholds: dict[str, float]) -> dict[str, float | int] | None:
    partial = [row for row in rows if row["reference_type"] == "partial"]
    if not partial:
        return None
    value = counts(partial, thresholds)
    return {
        "true_positive": value["tp"],
        "false_negative": value["fn"],
        "known_boundaries": value["tp"] + value["fn"],
        "recall": value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0,
    }


def _group_metrics(rows: list[dict], thresholds: dict[str, float], field: str) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "unknown"))].append(row)
    output = {}
    for key, value in sorted(grouped.items()):
        exact = [row for row in value if row["reference_type"] == "exact"]
        exact_metrics = metrics_from_counts(counts(exact, thresholds)) if exact else None
        output[key] = {
            "documents": len(value),
            "exact": exact_metrics,
            "partial_known_boundary": _partial_recall(value, thresholds),
        }
    return output


def _bootstrap_macro(rows: list[dict], thresholds: dict[str, float], samples: int, seed: int) -> dict[str, float] | None:
    rows = [row for row in rows if row["reference_type"] == "exact"]
    if not samples:
        return None
    if not rows:
        return None
    by_language = defaultdict(list)
    for row in rows:
        by_language[row["language_script"]].append(row)
    values = []
    rng = random.Random(seed)
    languages = sorted(by_language)
    for _ in range(samples):
        sampled = [rng.choice(by_language[language]) for language in languages for _ in by_language[language]]
        per_language = _group_metrics(sampled, thresholds, "language_script")
        values.append(float(np.mean([metric["exact"]["f1"] for metric in per_language.values()])))
    return {
        "samples": samples,
        "seed": seed,
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def evaluate(
    rows: list[dict],
    threshold_mode: str = "global",
    bootstrap: int = 1000,
    seed: int = 13,
    grid: Iterable[float] = DEFAULT_GRID,
) -> dict:
    rows = validate_rows(rows)
    thresholds = fit_thresholds(rows, threshold_mode, grid)
    test = [row for row in rows if row["split"] == "test"]
    exact = [row for row in test if row["reference_type"] == "exact"]
    partial = [row for row in test if row["reference_type"] == "partial"]
    if not test:
        raise ValueError("Evaluation requires at least one test row")

    per_language = _group_metrics(test, thresholds, "language_script")
    exact_language_f1 = [
        metric["exact"]["f1"] for metric in per_language.values() if metric["exact"] is not None
    ]
    macro_f1 = float(np.mean(exact_language_f1)) if exact_language_f1 else None
    exact_metrics = metrics_from_counts(counts(exact, thresholds)) if exact else None
    result = {
        "schema_version": "mmsat-eval-v1",
        "contract": {
            "threshold_fit": "exact-reference dev only",
            "exact_reference": "precision/recall/F1",
            "partial_reference": "known-boundary recall only; unlabelled predictions are not false positives",
            "offsets": "UTF-8 decoded Python character offsets between characters",
        },
        "input_sha256": hashlib.sha256(
            "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows).encode()
        ).hexdigest(),
        "threshold_mode": threshold_mode,
        "thresholds": thresholds,
        "test_documents": len(test),
        "micro_exact": exact_metrics,
        "exact_errors": (
            {"under_splitting_false_negatives": exact_metrics["fn"], "over_splitting_false_positives": exact_metrics["fp"]}
            if exact_metrics
            else None
        ),
        "known_boundary_partial": _partial_recall(partial, thresholds),
        "macro_language_exact_f1": macro_f1,
        "macro_language_f1_ci": _bootstrap_macro(test, thresholds, bootstrap, seed),
        "by_language_script": per_language,
        "by_script": _group_metrics(test, thresholds, "script"),
        "by_slice": _group_metrics(test, thresholds, "slice"),
    }
    return result
