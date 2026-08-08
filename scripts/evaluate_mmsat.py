#!/usr/bin/env python
"""Run the frozen mmSaT evaluation contract from scored JSONL documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from wtpsplit.evaluation.frozen import DEFAULT_GRID, evaluate, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSONL packets following docs/STAGE2.md",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-output", type=Path, help="Persist model probabilities for exact reruns")
    parser.add_argument("--model", help="Run SaT for rows without scores (checkpoint name or local path)")
    parser.add_argument("--tokenizer", default="facebookAI/xlm-roberta-base")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--efficiency-json", type=Path, help="Optional benchmark_timing/other structured report to embed")
    parser.add_argument("--threshold-mode", choices=("global", "per-language"), default="global")
    parser.add_argument("--threshold-grid", type=float, nargs="+", default=DEFAULT_GRID)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [row for path in args.input for row in read_jsonl(path)]
    missing_scores = [row for row in rows if "scores" not in row]
    inference = None
    if missing_scores:
        if not args.model:
            raise ValueError(f"{len(missing_scores)} rows lack scores; pass --model or provide scored rows")
        from wtpsplit import SaT

        sat = SaT(args.model, tokenizer_name_or_path=args.tokenizer, device=args.device)
        if args.device and args.device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        probabilities = list(
            sat.predict_proba(
                [row["text"] for row in missing_scores],
                batch_size=args.batch_size,
                stride=args.stride,
            )
        )
        if args.device and args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        for row, scores in zip(missing_scores, probabilities):
            row["scores"] = scores.tolist()
        character_count = sum(len(row["text"]) for row in missing_scores)
        inference = {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "device": args.device,
            "documents": len(missing_scores),
            "characters": character_count,
            "seconds": elapsed,
            "documents_per_second": len(missing_scores) / elapsed,
            "characters_per_second": character_count / elapsed,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(args.device)
                if args.device and args.device.startswith("cuda")
                else None
            ),
            "note": "End-to-end evaluation-corpus throughput; use benchmark_timing.py for warm microbenchmarks.",
        }
    if args.scored_output:
        args.scored_output.parent.mkdir(parents=True, exist_ok=True)
        args.scored_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    result = evaluate(rows, args.threshold_mode, args.bootstrap, args.seed, args.threshold_grid)
    result["requested_threshold_grid"] = args.threshold_grid
    result["inference"] = inference
    if args.efficiency_json:
        result["external_efficiency_report"] = json.loads(args.efficiency_json.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("schema_version", "test_documents", "micro_exact")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
