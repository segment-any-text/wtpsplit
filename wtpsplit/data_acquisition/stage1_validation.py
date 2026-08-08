"""Bounded-memory validation for Stage-1 corpus build receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from wtpsplit.data_acquisition.stage1_web import REQUIRED_COLUMNS, file_sha256


def _bounded_append(target: list[str], message: str, *, limit: int, counters: dict[str, int], kind: str) -> None:
    counters[kind] += 1
    if len(target) < limit:
        target.append(message)


def validate_stage1_metadata(
    metadata_path: str | Path,
    *,
    expected_languages: Iterable[str] | None = None,
    require_complete: bool = True,
    require_contamination_filter: bool = False,
    verify_hashes: bool = False,
    minimum_cap_fill: float = 0.95,
    max_reported_issues: int = 50,
) -> dict[str, Any]:
    """Validate one corpus in O(languages) memory with bounded issue output.

    Hash verification is optional because reading every shard is expensive at
    thousand-language scale. Structural checks and artifact existence remain on.
    """

    path = Path(metadata_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    languages = payload.get("languages")
    if not isinstance(languages, Mapping):
        raise ValueError(f"{path}: metadata has no language mapping")
    expected = set(expected_languages or languages)
    actual = set(map(str, languages))
    errors: list[str] = []
    warnings: list[str] = []
    counters = {"errors": 0, "warnings": 0}

    def error(message: str) -> None:
        _bounded_append(errors, message, limit=max_reported_issues, counters=counters, kind="errors")

    def warning(message: str) -> None:
        _bounded_append(
            warnings,
            message,
            limit=max_reported_issues,
            counters=counters,
            kind="warnings",
        )

    if payload.get("unit") not in {"paragraph", "document"}:
        error(f"metadata unit is invalid: {payload.get('unit')!r}")
    if not payload.get("build_fingerprint"):
        error("metadata build fingerprint is missing")

    for lang in sorted(expected - actual):
        error(f"{lang}: missing metadata entry")
    for lang in sorted(actual - expected):
        warning(f"{lang}: unexpected metadata entry")

    complete = 0
    total_units = 0
    total_chars = 0
    root = path.parent
    for lang in sorted(actual & expected):
        row = languages[lang]
        if not isinstance(row, Mapping):
            error(f"{lang}: malformed metadata entry")
            continue
        if row.get("status") != "complete":
            if require_complete:
                error(f"{lang}: status is {row.get('status')!r}, expected 'complete'")
            continue
        complete += 1
        counts = row.get("counts") or {}
        total_units += int(counts.get("units") or 0)
        total_chars += int(row.get("chars") or 0)
        validation = row.get("validation") or {}
        if validation.get("required_columns") != list(REQUIRED_COLUMNS):
            error(f"{lang}: required-column contract is missing or changed")
        if not validation.get("no_empty_datasets"):
            error(f"{lang}: train or validation split is empty")
        if int(validation.get("split_overlap_count") or 0):
            error(f"{lang}: train/validation overlap is non-zero")
        fill = float(row.get("chars_fill_ratio") or 0.0)
        if fill < minimum_cap_fill:
            warning(f"{lang}: character-cap fill {fill:.3f} < {minimum_cap_fill:.3f}")
        contamination = validation.get("contamination") or {}
        if require_contamination_filter and not contamination.get("filter_enabled"):
            error(f"{lang}: contamination filter was required but not enabled")
        source = row.get("source") or {}
        if not source.get("revision"):
            error(f"{lang}: source revision is not pinned")

        artifacts = row.get("artifacts") or {}
        for split in ("train", "valid"):
            artifact = artifacts.get(split) or {}
            artifact_path = Path(str(artifact.get("path") or root / split / f"{lang}.parquet"))
            if not artifact_path.is_absolute() and not artifact_path.is_file():
                artifact_path = root / split / f"{lang}.parquet"
            if not artifact_path.is_file():
                error(f"{lang}/{split}: artifact does not exist: {artifact_path}")
                continue
            if int(artifact.get("bytes") or 0) <= 0:
                error(f"{lang}/{split}: artifact byte count is empty")
            artifact_rows = int(artifact.get("rows") or 0)
            if artifact_rows <= 0:
                error(f"{lang}/{split}: artifact row count is empty")
            if artifact_rows != int(counts.get(split) or 0):
                error(f"{lang}/{split}: artifact and split row counts disagree")
            if verify_hashes and artifact.get("sha256") != file_sha256(artifact_path):
                error(f"{lang}/{split}: sha256 mismatch")

    return {
        "schema": "stage1-corpus-validation-v1",
        "metadata": str(path),
        "valid": counters["errors"] == 0,
        "counts": {
            "languages_expected": len(expected),
            "languages_present": len(actual),
            "languages_complete": complete,
            "units": total_units,
            "characters": total_chars,
            "errors": counters["errors"],
            "warnings": counters["warnings"],
        },
        "errors": errors,
        "warnings": warnings,
        "issues_truncated": {
            "errors": max(0, counters["errors"] - len(errors)),
            "warnings": max(0, counters["warnings"] - len(warnings)),
        },
        "checks": {
            "artifact_hashes": verify_hashes,
            "contamination_filter_required": require_contamination_filter,
            "minimum_cap_fill": minimum_cap_fill,
        },
    }
