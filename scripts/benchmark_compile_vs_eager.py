#!/usr/bin/env python3
"""Compare SaT eager PyTorch inference against ``torch.compile``.

Local benchmarking helper; separates load,
``optimize()``, first warmup, and steady-state timings, and synchronizes CUDA
around timed regions when running on GPU.

Examples:
  python scripts/benchmark_compile_vs_eager.py
  python scripts/benchmark_compile_vs_eager.py --warmup 10 --timed 40 --long-chars 80000 --batch
  CUDA_VISIBLE_DEVICES= python scripts/benchmark_compile_vs_eager.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sync_cuda(enabled: bool) -> None:
    if not enabled:
        return
    import torch

    torch.cuda.synchronize()


Payload = Union[str, list[str]]
Workload = tuple[str, Payload, bool, Optional[int]]


def _split_once(sat, payload: Payload, *, is_batch: bool, split_kwargs: dict[str, Any]) -> None:
    if is_batch:
        list(sat.split(payload, **split_kwargs))
    else:
        sat.split(payload, **split_kwargs)


def _time_runs(
    sat,
    payload: Payload,
    *,
    warmup: int,
    timed: int,
    is_batch: bool,
    split_kwargs: dict[str, Any],
    sync_cuda: bool,
) -> tuple[list[float], float]:
    """Return timed samples and the first warmup duration in seconds."""
    warmup_times = []
    for _ in range(warmup):
        _sync_cuda(sync_cuda)
        t0 = time.perf_counter()
        _split_once(sat, payload, is_batch=is_batch, split_kwargs=split_kwargs)
        _sync_cuda(sync_cuda)
        warmup_times.append(time.perf_counter() - t0)

    times = []
    for _ in range(timed):
        _sync_cuda(sync_cuda)
        t0 = time.perf_counter()
        _split_once(sat, payload, is_batch=is_batch, split_kwargs=split_kwargs)
        _sync_cuda(sync_cuda)
        times.append(time.perf_counter() - t0)

    first_warmup_s = warmup_times[0] if warmup_times else 0.0
    return times, first_warmup_s


def _stats(times_s: list[float]) -> dict[str, float]:
    if not times_s:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(times_s),
        "std": statistics.pstdev(times_s) if len(times_s) > 1 else 0.0,
        "min": min(times_s),
        "p50": statistics.median(times_s),
        "max": max(times_s),
    }


def _print_stats(label: str, times_s: list[float]) -> None:
    s = _stats(times_s)
    print(
        f"  {label}: "
        f"mean={s['mean'] * 1000:.2f}ms  "
        f"std={s['std'] * 1000:.2f}ms  "
        f"min={s['min'] * 1000:.2f}ms  "
        f"p50={s['p50'] * 1000:.2f}ms  "
        f"max={s['max'] * 1000:.2f}ms  "
        f"(n={len(times_s)})"
    )


def _load_model(model_name: str, *, device: str, dtype: str):
    import torch
    from wtpsplit import SaT

    sat = SaT(model_name, hub_prefix=None)
    if device == "cuda":
        sat.to("cuda")
    if dtype == "float16":
        sat.half()
    elif dtype == "bfloat16":
        sat.to(dtype=torch.bfloat16)
    return sat


def _make_workloads(long_chars: int, include_batch: bool) -> list[Workload]:
    short = "This is a test sentence. This is another test sentence."
    workloads: list[Workload] = [("short", short, False, None)]

    if long_chars > 0:
        unit = "This is one sentence in a long stream. "
        long_text = (unit * (long_chars // len(unit) + 1))[:long_chars]
        workloads.append((f"long-{long_chars}", long_text, False, long_chars))

    if include_batch:
        workloads.append(
            (
                "batch-2",
                [
                    "Paragraph-A Paragraph-B",
                    "Paragraph-C100 Paragraph-D",
                ],
                True,
                None,
            )
        )

    return workloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="segment-any-text/sat-3l", help="Hugging Face model id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend passed to optimize()")
    parser.add_argument("--mode", default="reduce-overhead", help="torch.compile mode passed to optimize()")
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--split-batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timed", type=int, default=30)
    parser.add_argument("--long-chars", type=int, default=0)
    parser.add_argument("--batch", action="store_true", help="Also benchmark a two-item batch")
    args = parser.parse_args()

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0 or newer")

    split_kwargs = {
        "threshold": args.threshold,
        "stride": args.stride,
        "block_size": args.block_size,
        "batch_size": args.split_batch_size,
    }
    sync_cuda = device == "cuda"
    workloads = _make_workloads(args.long_chars, args.batch)

    print(
        f"torch={torch.__version__}  device={device}  dtype={args.dtype}  "
        f"warmup={args.warmup}  timed={args.timed}"
    )
    print(
        f"model={args.model}  split_batch_size={args.split_batch_size}  "
        f"block_size={args.block_size}  stride={args.stride}"
    )
    print()

    print("Loading eager model")
    t0 = time.perf_counter()
    sat_eager = _load_model(args.model, device=device, dtype=args.dtype)
    eager_load_s = time.perf_counter() - t0
    print(f"  load+device+dtype: {eager_load_s:.3f}s")
    print()

    eager_means: dict[str, float] = {}
    print("EAGER")
    for label, payload, is_batch, _ in workloads:
        times, _ = _time_runs(
            sat_eager,
            payload,
            warmup=args.warmup,
            timed=args.timed,
            is_batch=is_batch,
            split_kwargs=split_kwargs,
            sync_cuda=sync_cuda,
        )
        _print_stats(label, times)
        eager_means[label] = _stats(times)["mean"]
    print()

    print("Loading compiled model")
    t0 = time.perf_counter()
    sat_compiled = _load_model(args.model, device=device, dtype=args.dtype)
    compiled_load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sat_compiled.optimize(backend=args.backend, mode=args.mode)
    optimize_s = time.perf_counter() - t0
    print(f"  load+device+dtype: {compiled_load_s:.3f}s")
    print(f"  optimize(): {optimize_s:.3f}s")
    print("  first warmup per workload usually includes graph compilation")
    print()

    compiled_means: dict[str, float] = {}
    print(f"TORCH.COMPILE ({args.backend})")
    for label, payload, is_batch, _ in workloads:
        times, first_warmup_s = _time_runs(
            sat_compiled,
            payload,
            warmup=args.warmup,
            timed=args.timed,
            is_batch=is_batch,
            split_kwargs=split_kwargs,
            sync_cuda=sync_cuda,
        )
        print(f"  {label}: first warmup={first_warmup_s * 1000:.1f}ms")
        _print_stats(label, times)
        compiled_means[label] = _stats(times)["mean"]
    print()

    print("SUMMARY (steady-state mean; speedup > 1 means compiled is faster)")
    for label, eager_mean_s in eager_means.items():
        compiled_mean_s = compiled_means[label]
        speedup = eager_mean_s / compiled_mean_s if compiled_mean_s else 0.0
        reduction = (1.0 - compiled_mean_s / eager_mean_s) * 100.0 if eager_mean_s else 0.0
        print(f"  {label}: speedup={speedup:.2f}x  wall_time_reduction={reduction:.1f}%")

    for label, _, _, chars in workloads:
        if chars is None:
            continue
        eager_mean_s = eager_means[label]
        compiled_mean_s = compiled_means[label]
        print(
            f"  {label}: throughput chars/s "
            f"eager={chars / eager_mean_s:,.0f}  compiled={chars / compiled_mean_s:,.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
