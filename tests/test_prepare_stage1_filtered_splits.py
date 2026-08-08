import gzip
import json
from pathlib import Path

from scripts.prepare_stage1_filtered_splits import assign_split, main, raw_shards
from wtpsplit.data_acquisition.contamination import ReferenceIndex


def test_assign_split_is_deterministic():
    assert assign_split("doc-1", 0.1, 42) == assign_split("doc-1", 0.1, 42)


def test_prepare_stage1_filtered_splits_writes_train_valid(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "samples"
    input_dir.mkdir()
    shard = input_dir / "eng_Latn.jsonl"
    rows = [
        {"id": "keep-train", "lang": "eng_Latn", "text": "Hello world one."},
        {"id": "keep-valid", "lang": "eng_Latn", "text": "Hello world two."},
        {"id": "drop-exact", "lang": "eng_Latn", "text": "DROP ME EXACT"},
    ]
    shard.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    index = ReferenceIndex()
    index.add_text("DROP ME EXACT", "ref-1", "eval", "eng_Latn")
    index_path = tmp_path / "index.json.gz"
    with gzip.open(index_path, "wt", encoding="utf-8") as handle:
        json.dump(index.to_payload(), handle)

    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_stage1_filtered_splits.py",
            "--input-dir",
            str(input_dir),
            "--train-dir",
            str(train_dir),
            "--valid-dir",
            str(valid_dir),
            "--index",
            str(index_path),
            "--receipt",
            str(receipt),
            "--valid-fraction",
            "0.5",
            "--seed",
            "0",
        ],
    )
    assert main() == 0
    assert list(raw_shards(input_dir)) == [shard]
    train_lines = (train_dir / "eng_Latn.filtered.jsonl").read_text(encoding="utf-8").splitlines()
    valid_lines = (valid_dir / "eng_Latn.filtered.jsonl").read_text(encoding="utf-8").splitlines()
    kept_ids = {
        json.loads(line)["id"] for line in train_lines + valid_lines
    }
    assert kept_ids == {"keep-train", "keep-valid"}
    assert "drop-exact" not in kept_ids
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["totals"]["exact_removed"] == 1
    assert receipt_payload["index_sha256"]
    assert receipt_payload["shards"][0]["train_sha256"]
    assert receipt_payload["shards"][0]["valid_sha256"]
