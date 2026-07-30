"""Stage-1 source loading with explicit local-shard and legacy-Hub paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def local_data_files(path: Path, require_filtered: bool) -> tuple[str, list[str]]:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        filtered = sorted(path.glob("*.filtered.jsonl"))
        if filtered:
            candidates = filtered
        else:
            candidates = sorted(
                item
                for pattern in ("*.jsonl", "*.json", "*.parquet", "*.csv")
                for item in path.glob(pattern)
                if ".receipt." not in item.name
            )
    else:
        raise FileNotFoundError(path)
    if not candidates:
        raise FileNotFoundError(f"No supported data shards under {path}")
    if require_filtered and any(".filtered." not in item.name for item in candidates):
        raise ValueError(
            "Stage-1 contamination filtering is required, but an input shard "
            f"is not marked .filtered: {candidates[0]}"
        )
    extensions = {item.suffix.lower() for item in candidates}
    if len(extensions) != 1:
        raise ValueError(f"Mixed local Stage-1 shard formats: {sorted(extensions)}")
    extension = next(iter(extensions))
    builder = {
        ".jsonl": "json",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
    }.get(extension)
    if builder is None:
        raise ValueError(f"Unsupported Stage-1 shard format: {extension}")
    return builder, [str(item) for item in candidates]


class _LocalTable:
    """Minimal column-oriented table for local JSON/JSONL without HuggingFace datasets."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def rename_column(self, old: str, new: str) -> _LocalTable:
        for row in self._rows:
            row[new] = row.pop(old)
        self.column_names = [new if name == old else name for name in self.column_names]
        return self

    def select(self, indices: Any) -> _LocalTable:
        return _LocalTable([self._rows[index] for index in indices])

    def __getitem__(self, key: Any) -> Any:
        if key == slice(None):
            return {column: [row[column] for row in self._rows] for column in self.column_names}
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        raise TypeError(f"Unsupported key for local Stage-1 table: {key!r}")


def _read_json_rows(files: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in files:
        path = Path(file)
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
        elif isinstance(payload, dict):
            rows.append(payload)
        else:
            raise ValueError(f"Unsupported JSON Stage-1 payload in {path}")
    return rows


def normalize_columns(dataset: Any, text_column: str = "text") -> Any:
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Stage-1 dataset is missing text column {text_column!r}: "
            f"{dataset.column_names}"
        )
    if "lang" not in dataset.column_names:
        if "language_script" in dataset.column_names:
            dataset = dataset.rename_column("language_script", "lang")
        else:
            raise ValueError(
                "Stage-1 dataset must provide `lang` or `language_script`"
            )
    return dataset


def load_stage1_dataset(
    path: str | Path,
    split: str,
    fallback_dataset: str | None,
    fallback_config: str | None = None,
    require_filtered: bool = False,
    text_column: str = "text",
    cache_dir: str | Path | None = None,
) -> Any:
    local_path = Path(path)
    if local_path.exists():
        builder, files = local_data_files(local_path, require_filtered)
        # Local JSON/JSONL is loadable without the research `datasets` dependency.
        if builder == "json":
            dataset: Any = _LocalTable(_read_json_rows(files))
        else:
            from datasets import load_dataset

            dataset = load_dataset(
                builder,
                data_files={split: files},
                split=split,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
    else:
        if fallback_dataset is None:
            raise FileNotFoundError(
                f"Local Stage-1 path does not exist and no fallback is configured: {path}"
            )
        from datasets import load_dataset

        dataset = load_dataset(
            fallback_dataset,
            fallback_config,
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    return normalize_columns(dataset, text_column)


def validate_text_batch(
    dataset: Any,
    batch_size: int = 2,
    text_column: str = "text",
) -> dict[str, Any]:
    if len(dataset) < batch_size:
        raise ValueError(
            f"Need at least {batch_size} Stage-1 examples, found {len(dataset)}"
        )
    batch = dataset.select(range(batch_size))[:]
    texts = batch[text_column]
    languages = batch["lang"]
    if not all(isinstance(text, str) and text for text in texts):
        raise ValueError("Stage-1 batch contains empty or non-string text")
    if not all(isinstance(language, str) and language for language in languages):
        raise ValueError("Stage-1 batch contains empty or non-string language ids")
    return {
        "batch_size": batch_size,
        "languages": languages,
        "characters": sum(len(text) for text in texts),
    }
