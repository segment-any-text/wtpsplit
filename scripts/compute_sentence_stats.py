#!/usr/bin/env python3
"""
Compute sentence length statistics from Universal Dependencies treebanks.

Usage:
    python scripts/compute_sentence_stats.py --output_dir wtpsplit/data/ -v
"""

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.request
from collections import defaultdict
from datetime import datetime

import conllu
import numpy as np


def compute_stats(lengths: list[int]) -> dict:
    """Compute target_length and spread from sentence lengths."""
    if len(lengths) < 10:
        return None

    arr = np.array(lengths)
    target_length = int(np.median(arr))

    # IQR-based spread (robust to outliers)
    q75, q25 = np.percentile(arr, [75, 25])
    spread = max(int((q75 - q25) / 1.35), 10)

    return {
        "target_length": target_length,
        "spread": spread,
        "n_sentences": len(lengths),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def load_ud_from_hf() -> dict[str, list[int]]:
    """Download and process UD treebanks from official release."""
    lang_lengths = defaultdict(list)

    ud_version = "2.14"
    url = f"https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v{ud_version}.tgz"

    print(f"Downloading UD v{ud_version} (~500MB)...", file=sys.stderr)

    with urllib.request.urlopen(url) as response:
        print("Reading archive into memory...", file=sys.stderr)
        fileobj = io.BytesIO(response.read())

        print("Processing CONLL-U files...", file=sys.stderr)
        file_count = 0

        with tarfile.open(fileobj=fileobj, mode="r:gz") as tar:
            for member in tar:
                if member.name.endswith(".conllu"):
                    filename = os.path.basename(member.name)
                    lang_code = filename.split("_")[0]

                    try:
                        f = tar.extractfile(member)
                        if f is not None:
                            content = f.read().decode("utf-8")
                            data = conllu.parse(content)

                            for sentence in data:
                                if "text" in sentence.metadata:
                                    lang_lengths[lang_code].append(len(sentence.metadata["text"]))

                            file_count += 1
                            if file_count % 50 == 0:
                                print(f"  Processed {file_count} files...", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Could not parse {member.name}: {e}", file=sys.stderr)

        print(f"  Processed {file_count} CONLL-U files total", file=sys.stderr)

    return dict(lang_lengths)


def json_to_python(json_path: str) -> str:
    """Convert JSON stats to Python dict format for priors.py."""
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = ["LANG_SENTENCE_STATS = {"]
    for lang_code, s in sorted(data["stats"].items()):
        lines.append(f'    "{lang_code}": {{"target_length": {s["target_length"]}, "spread": {s["spread"]}}},')
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute sentence length statistics from UD")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory (for downloading)")
    parser.add_argument("--to-python", type=str, metavar="JSON_FILE", help="Convert JSON to Python dict format")
    parser.add_argument("--min_sentences", type=int, default=100, help="Minimum sentences per language")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Convert existing JSON to Python format
    if args.to_python:
        print(json_to_python(args.to_python))
        return

    if not args.output_dir:
        parser.error("--output_dir required (or use --to-python to convert existing JSON)")

    # Load data
    lang_lengths = load_ud_from_hf()
    print(f"Loaded data for {len(lang_lengths)} languages", file=sys.stderr)

    # Compute statistics
    stats = {}
    for lang_code, lengths in sorted(lang_lengths.items()):
        if len(lengths) >= args.min_sentences:
            stats[lang_code] = compute_stats(lengths)
            if args.verbose:
                s = stats[lang_code]
                print(
                    f"  {lang_code}: {len(lengths):>6} sentences, median={s['target_length']:>3}, spread={s['spread']}",
                    file=sys.stderr,
                )
        elif args.verbose:
            print(f"  {lang_code}: skipped ({len(lengths)} < {args.min_sentences} sentences)", file=sys.stderr)

    print(f"\nComputed statistics for {len(stats)} languages", file=sys.stderr)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "metadata": {
            "source": "Universal Dependencies v2.14",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "stats": {k: {"target_length": v["target_length"], "spread": v["spread"]} for k, v in stats.items()},
    }

    json_path = os.path.join(args.output_dir, "sentence_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
