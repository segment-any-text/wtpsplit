"""Measure in-process SaT LoRA adaptation on a deterministic workload."""

import argparse
import time

from wtpsplit import SaT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sat-3l-sm")
    parser.add_argument("--sentences", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sentences = [
        f"Gold sentence number {index} demonstrates the desired segmentation style." for index in range(args.sentences)
    ]
    sat = SaT(args.model, device=args.device)

    started = time.perf_counter()
    sat.adapt(sentences, language="en", epochs=args.epochs, show_progress=False)
    elapsed = time.perf_counter() - started

    print(
        f"device={args.device} model={args.model} sentences={args.sentences} "
        f"epochs={args.epochs} elapsed={elapsed:.2f}s"
    )
    print(f"first_loss={sat.adaptation_history[0]:.6f} final_loss={sat.adaptation_history[-1]:.6f}")


if __name__ == "__main__":
    main()
