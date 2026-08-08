#!/usr/bin/env python
"""Run the established Stage-1 intrinsic protocol.

Writes a compact, comparable summary beside the raw result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from wtpsplit.evaluation.intrinsic_summary import (
    summarize_intrinsic_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--eval-data",
        default="data/all_data_11_05-all.pth",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ud", "opus100", "ersatz"],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_artifact_hashes(model: str) -> dict[str, str]:
    root = Path(model)
    if not root.is_dir():
        return {}
    names = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "training_args.bin",
    )
    return {
        name: sha256(root / name)
        for name in names
        if (root / name).is_file()
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir or args.output.parent / f"{args.output.stem}_work"
    cache_dir = work_dir / "cache"
    intrinsic_dir = cache_dir / "intrinsic"
    intrinsic_dir.mkdir(parents=True, exist_ok=True)
    before = {
        path: path.stat().st_mtime_ns for path in intrinsic_dir.glob("*.json")
    }
    command = [
        sys.executable,
        "-m",
        "wtpsplit.evaluation.adapt",
        "--model_path",
        args.model,
        "--eval_data_path",
        args.eval_data,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--block_size",
        str(args.block_size),
        "--stride",
        str(args.stride),
        "--include_datasets",
        *args.datasets,
        "--threshold",
        "0.01",
        "--max_n_train_sentences",
        "10000",
        "--max_n_test_sentences",
        "-1",
        "--exclude_every_k",
        "10",
        "--keep_logits",
        "--save_suffix",
        f"_{args.output.stem}",
    ]
    environment = os.environ.copy()
    environment["WTPSPLIT_CACHE"] = str(cache_dir.resolve())
    log_path = work_dir / "evaluation.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        raise SystemExit(
            f"Intrinsic evaluation failed ({completed.returncode}); "
            f"see {log_path}"
        )
    candidates = [
        path
        for path in intrinsic_dir.glob("*.json")
        if not path.name.endswith(("_AVG.json", "_IDX.json"))
        and (
            path not in before
            or path.stat().st_mtime_ns != before[path]
        )
    ]
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one new intrinsic result, "
            f"found {len(candidates)} under {intrinsic_dir}"
        )
    raw_output = args.output.with_suffix(".raw.json")
    shutil.copy2(candidates[0], raw_output)
    summarize_intrinsic_result(raw_output, args.output)
    receipt = {
        "schema_version": "mmsat-stage1-evaluation-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "model": args.model,
        "model_artifact_sha256": model_artifact_hashes(args.model),
        "eval_data": args.eval_data,
        "eval_data_sha256": sha256(Path(args.eval_data)),
        "raw_result": str(raw_output),
        "raw_sha256": sha256(raw_output),
        "summary": str(args.output),
        "summary_sha256": sha256(args.output),
    }
    args.output.with_suffix(".run.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
