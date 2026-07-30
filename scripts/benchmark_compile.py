"""Compare eager and ``torch.compile`` SaT inference on a fixed workload."""

import argparse
import statistics
import time

import torch

from wtpsplit import SaT

TEXT = (
    "Sentence segmentation is useful for retrieval and language processing. "
    "This fixed benchmark repeats a realistic paragraph to amortize Python overhead. "
) * 8


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def measure(model: SaT, texts: list[str], device: str, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        model.split(texts)
    synchronize(device)

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.split(texts)
        synchronize(device)
        timings.append(time.perf_counter() - started)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sat-3l-sm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    texts = [TEXT] * args.batch_size

    eager = SaT(args.model, device=args.device)
    eager_times = measure(eager, texts, args.device, args.warmups, args.repeats)
    del eager

    compiled = SaT(args.model, device=args.device, compile=True)
    compiled_times = measure(compiled, texts, args.device, args.warmups, args.repeats)

    eager_median = statistics.median(eager_times)
    compiled_median = statistics.median(compiled_times)
    speedup = eager_median / compiled_median
    print(f"device={args.device} model={args.model} batch={args.batch_size}")
    print(f"eager median:    {eager_median * 1000:.2f} ms")
    print(f"compiled median: {compiled_median * 1000:.2f} ms")
    print(f"speedup:         {speedup:.2f}x")


if __name__ == "__main__":
    main()
