"""Time SaT and legacy WtP loading and inference on an explicit device.

CUDA timings synchronize before and after every measured call. Without this,
``perf_counter`` measures only asynchronous kernel submission rather than the
inference itself.

Run from the repository root, for example:

    python scripts/benchmark_timing.py --device cuda --warmups 2 --repeats 10
    python scripts/benchmark_timing.py --device cpu --warmups 2 --repeats 10
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mean_std(times_s):
    n = len(times_s)
    mean = sum(times_s) / n
    var = sum((t - mean) ** 2 for t in times_s) / max(n - 1, 1)
    return mean, (var**0.5) if var else 0.0


def _synchronize(torch, device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _measure(torch, device, warmups, repeats, operation):
    for _ in range(warmups):
        operation()
    _synchronize(torch, device)

    times = []
    for _ in range(repeats):
        _synchronize(torch, device)
        started = time.perf_counter()
        operation()
        _synchronize(torch, device)
        times.append(time.perf_counter() - started)
    return _mean_std(times)


def _load(torch, device, loader):
    _synchronize(torch, device)
    started = time.perf_counter()
    model = loader()
    _synchronize(torch, device)
    return model, time.perf_counter() - started


def main():
    import torch
    from wtpsplit import SaT, WtP

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"--device {args.device!r} requested, but CUDA is unavailable")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    tf_version = "unknown"
    try:
        import transformers

        tf_version = transformers.__version__
    except Exception:
        pass

    device = args.device
    print(
        f"transformers={tf_version}  warmups={args.warmups}  "
        f"repeats={args.repeats}  device={device}"
    )
    if device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(torch.device(device))}")
    print()

    # Fixed inputs
    short = "This is a test sentence. This is another test sentence."
    batch = ["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]
    timings = []

    # --- SaT (no adapter) ---
    sat, load_sat = _load(
        torch,
        device,
        lambda: SaT("segment-any-text/sat-3l", hub_prefix=None, device=device),
    )
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: sat.split(short, threshold=0.025),
    )
    timings.append(("SaT (no adapter) load", load_sat, None))
    timings.append(("SaT (no adapter) split(short)", mean, std))
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: list(sat.split(batch, threshold=0.025)),
    )
    timings.append(("SaT (no adapter) split(batch)", mean, std))

    # --- SaT with LoRA ---
    sat_ud, load_sat_lora = _load(
        torch,
        device,
        lambda: SaT(
            "segment-any-text/sat-3l",
            hub_prefix=None,
            domain="ud",
            language="en",
            device=device,
        ),
    )
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: sat_ud.split(short),
    )
    timings.append(("SaT (LoRA ud) load", load_sat_lora, None))
    timings.append(("SaT (LoRA ud) split(short)", mean, std))

    # --- SaT-SM (no adapter) ---
    sat_sm, load_sm = _load(
        torch,
        device,
        lambda: SaT("segment-any-text/sat-12l-sm", hub_prefix=None, device=device),
    )
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: sat_sm.split(short, threshold=0.25),
    )
    timings.append(("SaT-SM (no adapter) load", load_sm, None))
    timings.append(("SaT-SM (no adapter) split(short)", mean, std))

    # --- WtP Bert (no adapter) ---
    def load_wtp():
        model = WtP("benjamin/wtp-bert-mini", hub_prefix=None, ignore_legacy_warning=True)
        model.to(device)
        return model

    wtp, load_wtp_time = _load(torch, device, load_wtp)
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: wtp.split(short, threshold=0.005),
    )
    timings.append(("WtP Bert (no adapter) load", load_wtp_time, None))
    timings.append(("WtP Bert (no adapter) split(short)", mean, std))

    # --- WtP Bert with style ---
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: wtp.split(short, language="en", domain="ud", threshold=0.005),
    )
    timings.append(("WtP Bert (style ud) split(short)", mean, std))

    # --- WtP Canine ---
    def load_wtp_c():
        model = WtP("benjamin/wtp-canine-s-3l", hub_prefix=None, ignore_legacy_warning=True)
        model.to(device)
        return model

    wtp_c, load_c = _load(torch, device, load_wtp_c)
    mean, std = _measure(
        torch,
        device,
        args.warmups,
        args.repeats,
        lambda: wtp_c.split(short, language="en"),
    )
    timings.append(("WtP Canine load", load_c, None))
    timings.append(("WtP Canine split(short)", mean, std))

    # --- Report ---
    print("Load time (s); inference latency (ms)")
    print("-" * 72)
    for name, mean_t, std_t in timings:
        if std_t is None:
            print(f"  {name:<45}  {mean_t:>8.3f} s")
        else:
            print(f"  {name:<45}  {mean_t * 1000:>8.2f} ± {std_t * 1000:.2f} ms")
    print()
    total_load = load_sat + load_sat_lora + load_sm + load_wtp_time + load_c
    print(f"  Total load time (all models): {total_load:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
