"""
Quick timing benchmark for wtpsplit with tf4/tf5.
Measures load time and inference time for SaT, SaT-SM, and WtP (no adapter and with).
Usage: python scripts/benchmark_timing.py [--repeats N]
Run from repo root. For comparable results use same device, e.g.:
  CUDA_VISIBLE_DEVICES="" conda run -n wtp-hari python scripts/benchmark_timing.py
  CUDA_VISIBLE_DEVICES="" conda run -n wtpsplit-tf5 python scripts/benchmark_timing.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPEATS = 5
if "--repeats" in sys.argv:
    i = sys.argv.index("--repeats")
    REPEATS = int(sys.argv[i + 1])


def _mean_std(times_s):
    n = len(times_s)
    mean = sum(times_s) / n
    var = sum((t - mean) ** 2 for t in times_s) / max(n - 1, 1)
    return mean, (var**0.5) if var else 0.0


def main():
    import torch
    from wtpsplit import SaT, WtP

    tf_version = "unknown"
    try:
        import transformers

        tf_version = transformers.__version__
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"transformers={tf_version}  repeats={REPEATS}  device={device}")
    print()

    # Fixed inputs
    short = "This is a test sentence. This is another test sentence."
    batch = ["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]
    timings = []

    # --- SaT (no adapter) ---
    t0 = time.perf_counter()
    sat = SaT("segment-any-text/sat-3l", hub_prefix=None)
    load_sat = time.perf_counter() - t0
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sat.split(short, threshold=0.025)
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("SaT (no adapter) load", load_sat, None))
    timings.append(("SaT (no adapter) split(short)", mean, std))
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        list(sat.split(batch, threshold=0.025))
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("SaT (no adapter) split(batch)", mean, std))

    # --- SaT with LoRA ---
    t0 = time.perf_counter()
    sat_ud = SaT("segment-any-text/sat-3l", hub_prefix=None, style_or_domain="ud", language="en")
    load_sat_lora = time.perf_counter() - t0
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sat_ud.split(short)
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("SaT (LoRA ud) load", load_sat_lora, None))
    timings.append(("SaT (LoRA ud) split(short)", mean, std))

    # --- SaT-SM (no adapter) ---
    t0 = time.perf_counter()
    sat_sm = SaT("segment-any-text/sat-12l-sm", hub_prefix=None)
    load_sm = time.perf_counter() - t0
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sat_sm.split(short, threshold=0.25)
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("SaT-SM (no adapter) load", load_sm, None))
    timings.append(("SaT-SM (no adapter) split(short)", mean, std))

    # --- WtP Bert (no adapter) ---
    t0 = time.perf_counter()
    wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None, ignore_legacy_warning=True)
    load_wtp = time.perf_counter() - t0
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        wtp.split(short, threshold=0.005)
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("WtP Bert (no adapter) load", load_wtp, None))
    timings.append(("WtP Bert (no adapter) split(short)", mean, std))

    # --- WtP Bert with style ---
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        wtp.split(short, lang_code="en", style="ud", threshold=0.005)
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("WtP Bert (style ud) split(short)", mean, std))

    # --- WtP Canine ---
    t0 = time.perf_counter()
    wtp_c = WtP("benjamin/wtp-canine-s-3l", hub_prefix=None, ignore_legacy_warning=True)
    load_c = time.perf_counter() - t0
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        wtp_c.split(short, lang_code="en")
        times.append(time.perf_counter() - t0)
    mean, std = _mean_std(times)
    timings.append(("WtP Canine load", load_c, None))
    timings.append(("WtP Canine split(short)", mean, std))

    # --- Report ---
    print("Timing (s)")
    print("-" * 60)
    for name, mean_t, std_t in timings:
        if std_t is None:
            print(f"  {name:<45}  {mean_t:>8.3f}")
        else:
            print(f"  {name:<45}  {mean_t:>8.3f}  ± {std_t:.3f}")
    print()
    total_load = load_sat + load_sat_lora + load_sm + load_wtp + load_c
    print(f"  Total load time (all models): {total_load:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
