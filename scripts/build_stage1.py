#!/usr/bin/env python
"""Build a resumable, revision-pinned Stage-1 FineWeb or document corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wtpsplit.data_acquisition.stage1_web import (
    BuildOptions,
    build_stage1_web,
    file_sha256,
    load_mc4_caps,
    resolve_web_sources,
)


FINEWEB2_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("fineweb2", "mc4"), required=True)
    parser.add_argument(
        "--unit",
        choices=("paragraph", "document"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--char-caps",
        type=Path,
        default=Path("data/manifests/mc4_test_per_lang_char_mass.json"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/manifests/sat_lang_to_fineweb2_v1.json"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_sampling_plan_v1.json"),
    )
    parser.add_argument(
        "--script-remaps",
        type=Path,
        default=Path("data/manifests/stage1_fineweb_script_remaps_v1.json"),
    )
    parser.add_argument(
        "--contamination-index",
        type=Path,
        default=Path("data/external/mmsat_contamination_index_v1.json.gz"),
    )
    parser.add_argument("--skip-contamination-filter", action="store_true")
    parser.add_argument("--language", action="append", dest="languages")
    bounds = parser.add_mutually_exclusive_group()
    bounds.add_argument(
        "--max-chars-per-language",
        type=int,
        help="Bound every language cap for a smoke build without changing the source manifest.",
    )
    bounds.add_argument(
        "--cap-scale",
        type=float,
        help="Multiply every language cap (for example 0.001 for a bounded smoke build).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the build plan without opening datasets or writing artifacts.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--non-punctuation-sample-ratio", type=float)
    parser.add_argument(
        "--language-info",
        type=Path,
        default=Path("wtpsplit/data/language_info.csv"),
    )
    parser.add_argument("--fineweb2-revision", default=FINEWEB2_REVISION)
    parser.add_argument("--fineweb-revision", default=FINEWEB_REVISION)
    parser.add_argument("--c4-revision", default=C4_REVISION)
    return parser.parse_args()


def no_punctuation_languages(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return tuple(str(row[rows.fieldnames[0]]) for row in rows if row.get("no_punctuation", "").lower() == "true")


def default_output(corpus: str, unit: str) -> Path:
    return Path(f"data/external/{corpus}-stage1-{unit}s")


def bounded_caps(
    caps: dict[str, int],
    *,
    max_chars_per_language: int | None = None,
    cap_scale: float | None = None,
) -> dict[str, int]:
    """Return deterministic smoke caps while preserving every selected language."""

    if max_chars_per_language is not None:
        if max_chars_per_language <= 0:
            raise ValueError("max_chars_per_language must be positive")
        return {lang: min(int(cap), max_chars_per_language) for lang, cap in caps.items()}
    if cap_scale is not None:
        if not 0.0 < cap_scale <= 1.0:
            raise ValueError("cap_scale must be in (0, 1]")
        return {lang: max(1, int(int(cap) * cap_scale)) for lang, cap in caps.items()}
    return {lang: int(cap) for lang, cap in caps.items()}


def main() -> int:
    args = parse_args()
    revisions = {
        "HuggingFaceFW/fineweb-2": args.fineweb2_revision,
        "HuggingFaceFW/fineweb": args.fineweb_revision,
        "allenai/c4": args.c4_revision,
    }
    sources = resolve_web_sources(
        corpus=args.corpus,
        language_map=args.mapping,
        languages=args.languages,
        plan=args.plan if args.corpus == "fineweb2" else None,
        script_remaps=(args.script_remaps if args.corpus == "fineweb2" else None),
        revisions=revisions,
    )
    all_caps = load_mc4_caps(args.char_caps)
    caps = bounded_caps(
        {language: all_caps[language] for language in sources},
        max_chars_per_language=args.max_chars_per_language,
        cap_scale=args.cap_scale,
    )

    output_dir = args.output_dir or default_output(args.corpus, args.unit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "corpus": args.corpus,
                    "unit": args.unit,
                    "output_dir": str(output_dir),
                    "languages": len(sources),
                    "total_character_budget": sum(caps.values()),
                    "minimum_language_budget": min(caps.values()),
                    "maximum_language_budget": max(caps.values()),
                    "source_datasets": sorted({source.dataset for source in sources.values()}),
                    "revisions": revisions,
                    "contamination_filter": (args.corpus == "fineweb2" and not args.skip_contamination_filter),
                    "next_step": "rerun without --dry-run to stream and write the corpus",
                },
                indent=2,
            )
        )
        return 0

    input_paths = [args.char_caps, args.mapping]
    contamination_matcher = None
    if args.corpus == "fineweb2":
        input_paths.extend((args.plan, args.script_remaps))
        if not args.skip_contamination_filter:
            from wtpsplit.data_acquisition.contamination import load_index

            index = load_index(args.contamination_index)
            contamination_matcher = index.match
            input_paths.append(args.contamination_index)

    ratio = args.non_punctuation_sample_ratio
    if ratio is None and args.unit == "document":
        ratio = 0.1
    metadata = build_stage1_web(
        caps=caps,
        sources=sources,
        options=BuildOptions(
            output_dir=output_dir,
            unit=args.unit,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            non_punctuation_sample_ratio=ratio,
            no_punctuation_languages=no_punctuation_languages(args.language_info),
            compression=args.compression,
        ),
        contamination_matcher=contamination_matcher,
        input_hashes={path.as_posix(): file_sha256(path) for path in input_paths},
    )
    complete = metadata["validation"]["languages_complete"]
    expected = metadata["validation"]["languages_expected"]
    print(f"Stage-1 build complete: {complete}/{expected} languages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
