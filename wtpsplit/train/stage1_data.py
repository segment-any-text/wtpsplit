"""Stage-1 source loading with explicit local-shard and legacy-Hub paths."""

from __future__ import annotations

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
    from datasets import load_dataset

    local_path = Path(path)
    if local_path.exists():
        builder, files = local_data_files(local_path, require_filtered)
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
