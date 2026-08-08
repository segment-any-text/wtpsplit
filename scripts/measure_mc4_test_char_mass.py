#!/usr/bin/env python
"""Measure per-language character mass in mC4-TEST (Hub parquet cache)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


MC4_TEST_REVISION = "6c109b67925b989746262bb67f0214f59bb1f8a2"


def _hub_data_dir(hf_home: Path, revision: str) -> Path:
    root = hf_home / "hub" / "datasets--markus583--mC4-TEST" / "snapshots"
    snapshot = root / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"No mC4-TEST snapshot for revision {revision} under {root}")
    data = snapshot / "data"
    if not data.is_dir():
        raise FileNotFoundError(f"No data/ under {snapshot}")
    return data


def _agg_file(path: Path) -> dict[str, dict[str, int]]:
    """Return lang -> {rows, chars, punct_rows, nonpunct_rows} for one parquet."""
    table = pq.read_table(
        path,
        columns=["lang", "text", "ends_with_punctuation"],
    )
    langs = table["lang"].to_numpy(zero_copy_only=False)
    lengths = pc.utf8_length(table["text"]).to_numpy()
    punct = table["ends_with_punctuation"].to_numpy()

    # factorize for vectorized group sums
    codes, uniques = pd_factorize(langs)
    n = len(uniques)
    rows = np.bincount(codes, minlength=n)
    chars = np.bincount(codes, weights=lengths.astype(np.float64), minlength=n).astype(np.int64)
    punct_rows = np.bincount(codes, weights=punct.astype(np.float64), minlength=n).astype(np.int64)

    out: dict[str, dict[str, int]] = {}
    for i, lang in enumerate(uniques):
        r = int(rows[i])
        p = int(punct_rows[i])
        out[str(lang)] = {
            "rows": r,
            "chars": int(chars[i]),
            "punct_rows": p,
            "nonpunct_rows": r - p,
        }
    return out


def pd_factorize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-encode strings without requiring pandas."""
    uniques, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64), uniques


def merge_into(
    total: dict[str, dict[str, int]],
    part: dict[str, dict[str, int]],
) -> None:
    for lang, stats in part.items():
        slot = total[lang]
        for k, v in stats.items():
            slot[k] += v


def summarize_split(per_lang: dict[str, dict[str, int]]) -> dict:
    langs = sorted(per_lang)
    char_list = [per_lang[l]["chars"] for l in langs]
    row_list = [per_lang[l]["rows"] for l in langs]
    arr = np.asarray(char_list, dtype=np.float64)
    return {
        "n_languages": len(langs),
        "languages": langs,
        "total_rows": int(sum(row_list)),
        "total_chars": int(sum(char_list)),
        "chars_min": int(arr.min()) if len(arr) else 0,
        "chars_max": int(arr.max()) if len(arr) else 0,
        "chars_mean": float(arr.mean()) if len(arr) else 0.0,
        "chars_median": float(np.median(arr)) if len(arr) else 0.0,
        "chars_std": float(arr.std()) if len(arr) else 0.0,
        "chars_cv": float(arr.std() / arr.mean()) if len(arr) and arr.mean() else 0.0,
        # SaT-style equal_chars total budget ≈ n_langs * per-lang (use mean of dump)
        "recommended_equal_chars_target_chars": int(round(arr.mean() * len(arr))) if len(arr) else 0,
        "recommended_chars_per_language": int(round(arr.mean())) if len(arr) else 0,
        # Conservative match to scarcest language (no upsample)
        "match_min_chars_per_language": int(arr.min()) if len(arr) else 0,
        "match_min_total_chars": int(arr.min() * len(arr)) if len(arr) else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/mc4_test_per_lang_char_mass.json"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=("train", "valid"),
    )
    parser.add_argument("--source-revision", default=MC4_TEST_REVISION)
    args = parser.parse_args()

    data_dir = _hub_data_dir(args.hf_home, args.source_revision)
    print(time.strftime("%F %T"), "data_dir", data_dir, flush=True)

    payload: dict = {
        "version": "mc4_test_per_lang_char_mass_v1",
        "dataset": "markus583/mC4-TEST",
        "source_revision": args.source_revision,
        "finished_at": None,
        "splits": {},
    }

    for split in args.splits:
        files = sorted(data_dir.glob(f"{split}-*-of-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split}-*-of-*.parquet under {data_dir}")
        print(
            time.strftime("%F %T"),
            f"scan {split}: {len(files)} files",
            flush=True,
        )
        total: dict[str, dict[str, int]] = defaultdict(
            lambda: {"rows": 0, "chars": 0, "punct_rows": 0, "nonpunct_rows": 0}
        )
        for i, path in enumerate(files, 1):
            t0 = time.time()
            part = _agg_file(path)
            merge_into(total, part)
            dt = time.time() - t0
            if i == 1 or i % 10 == 0 or i == len(files):
                print(
                    f"  [{i}/{len(files)}] {path.name} "
                    f"langs_so_far={len(total)} ({dt:.1f}s)",
                    flush=True,
                )

        per_lang = {k: dict(v) for k, v in sorted(total.items())}
        for lang, stats in per_lang.items():
            r = stats["rows"]
            stats["punct_row_ratio"] = (stats["punct_rows"] / r) if r else 0.0
            stats["mean_chars_per_row"] = (stats["chars"] / r) if r else 0.0

        summary = summarize_split(per_lang)
        payload["splits"][split] = {
            "summary": summary,
            "per_lang": per_lang,
        }
        s = summary
        print(
            time.strftime("%F %T"),
            f"{split}: langs={s['n_languages']} rows={s['total_rows']} "
            f"chars={s['total_chars']} mean/lang={s['chars_mean']:.0f} "
            f"min={s['chars_min']} max={s['chars_max']} cv={s['chars_cv']:.4f}",
            flush=True,
        )

    payload["finished_at"] = time.strftime("%F %T")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Flat CSV for train (primary matching surface)
    if "train" in payload["splits"]:
        csv_path = args.output.with_suffix(".csv")
        train_pl = payload["splits"]["train"]["per_lang"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "lang",
                    "rows",
                    "chars",
                    "punct_rows",
                    "nonpunct_rows",
                    "punct_row_ratio",
                    "mean_chars_per_row",
                ],
            )
            w.writeheader()
            for lang, stats in train_pl.items():
                w.writerow({"lang": lang, **stats})
        print(time.strftime("%F %T"), "wrote", csv_path, flush=True)

    print(time.strftime("%F %T"), "DONE", args.output, flush=True)
    print(json.dumps({k: v["summary"] for k, v in payload["splits"].items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
