import json
from pathlib import Path

import pytest

from wtpsplit.train.stage1_data import (
    load_stage1_dataset,
    local_data_files,
    validate_text_batch,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_local_filtered_jsonl_reaches_a_valid_batch(tmp_path: Path):
    path = tmp_path / "abc_Latn.filtered.jsonl"
    write_jsonl(
        path,
        [
            {"text": "First paragraph.\nSecond line.", "lang": "abc_Latn"},
            {"text": "Another document.", "lang": "abc_Latn"},
        ],
    )
    dataset = load_stage1_dataset(
        path,
        split="train",
        fallback_dataset=None,
        require_filtered=True,
        cache_dir=tmp_path / "cache",
    )
    batch = validate_text_batch(dataset, batch_size=2)
    assert batch["languages"] == ["abc_Latn", "abc_Latn"]
    assert batch["characters"] > 20


def test_directory_prefers_filtered_shards(tmp_path: Path):
    write_jsonl(tmp_path / "raw.jsonl", [{"text": "raw", "lang": "x_Latn"}])
    write_jsonl(
        tmp_path / "x_Latn.filtered.jsonl",
        [{"text": "filtered", "lang": "x_Latn"}],
    )
    builder, files = local_data_files(tmp_path, require_filtered=True)
    assert builder == "json"
    assert files == [str(tmp_path / "x_Latn.filtered.jsonl")]


def test_rejects_unfiltered_file_when_gate_enabled(tmp_path: Path):
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, [{"text": "raw", "lang": "x_Latn"}])
    with pytest.raises(ValueError, match="contamination filtering"):
        local_data_files(path, require_filtered=True)


def test_validation_rejects_empty_stage1_source():
    class EmptyDataset:
        def __len__(self):
            return 0

    with pytest.raises(ValueError, match="Need at least 1"):
        validate_text_batch(EmptyDataset(), batch_size=1)


def test_loader_rejects_only_empty_local_files(tmp_path: Path):
    path = tmp_path / "empty.filtered.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="only empty files"):
        local_data_files(path, require_filtered=True)
