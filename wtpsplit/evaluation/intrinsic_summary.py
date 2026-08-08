"""Normalize Stage-1 intrinsic adaptation output for arm comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any


def summarize_intrinsic_result(
    raw_result: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    raw = json.loads(Path(raw_result).read_text(encoding="utf-8"))
    by_language: dict[str, dict[str, float]] = {}
    by_dataset: dict[str, dict[str, list[float]]] = {}
    for language, datasets in raw.items():
        language_values: dict[str, list[float]] = {"u": [], "t": []}
        for dataset, metrics in datasets.items():
            target = by_dataset.setdefault(dataset, {"u": [], "t": []})
            for metric in ("u", "t"):
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    numeric = float(value)
                    language_values[metric].append(numeric)
                    target[metric].append(numeric)
        by_language[language] = {
            f"macro_f1_{metric}": statistics.mean(values)
            for metric, values in language_values.items()
            if values
        }
    summary = {
        "schema_version": "mmsat-stage1-intrinsic-v1",
        "language_macro_f1_u": statistics.mean(
            values["macro_f1_u"]
            for values in by_language.values()
            if "macro_f1_u" in values
        ),
        "language_macro_f1_t": statistics.mean(
            values["macro_f1_t"]
            for values in by_language.values()
            if "macro_f1_t" in values
        ),
        "languages": len(by_language),
        "by_language": by_language,
        "by_dataset": {
            dataset: {
                f"macro_f1_{metric}": statistics.mean(values)
                for metric, values in metrics.items()
                if values
            }
            for dataset, metrics in sorted(by_dataset.items())
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
