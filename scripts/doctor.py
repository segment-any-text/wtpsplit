#!/usr/bin/env python
"""Check whether the local machine can run the documented curriculum."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


# Files needed to build and train Stage 1 (paragraph + document variants)
STAGE1_FILES = (
    "configs/curriculum/stage1_mc4.json",
    "configs/curriculum/stage1_fineweb.json",
    "configs/curriculum/stage1_mc4_documents.json",
    "configs/curriculum/stage1_fineweb_documents.json",
    "configs/curriculum/stage1_mc4_mmbert.json",
    "configs/curriculum/stage1_fineweb_mmbert.json",
    "configs/mmsat_3l.json",
    "configs/curriculum/selections.json",
    "data/manifests/mc4_test_per_lang_char_mass.json",
    "data/manifests/sat_lang_to_fineweb2_v1.json",
    "data/manifests/fineweb2_stage1_sampling_plan_v1.json",
    "data/manifests/fineweb2_stage1_coverage_v1.json",
    "data/manifests/mmsat_dataset_coverage_v1.json",
    "data/manifests/stage1_coverage_gap_review_v1.json",
    "data/manifests/stage1_fineweb_script_remaps_v1.json",
    "data/manifests/ud_2_18_frozen_selection_v1.json",
    "data/manifests/mmsat_contamination_index_v1.json",
)

# Needed for Stage-1 intrinsic evaluation and for Stage 3 training.
HISTORICAL_PACKET = "data/all_data_11_05-all.pth"

# Present once Stage 2/3 configs land; not required to start Stage 1.
LATER_STAGE_FILES = (
    "configs/curriculum/stage2.json",
    "configs/curriculum/stage3.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--minimum-free-gb", type=float, default=300)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--require-later-stages",
        action="store_true",
        help="Also require Stage-2/3 curriculum configs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a required check fails.",
    )
    return parser.parse_args()


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": commit.returncode == 0,
        "commit": commit.stdout.strip() or None,
        "dirty": (
            bool(status.stdout.strip()) if not status.returncode else None
        ),
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    probe = output_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1024**3

    required_paths = list(STAGE1_FILES) + [HISTORICAL_PACKET]
    if args.require_later_stages:
        required_paths.extend(LATER_STAGE_FILES)
    files = {path: Path(path).is_file() for path in required_paths}
    later = {path: Path(path).is_file() for path in LATER_STAGE_FILES}

    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("datasets", "pyarrow", "torch", "transformers")
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_devices = torch.cuda.device_count()
    except ImportError:
        cuda_available = False
        cuda_devices = 0
    try:
        from huggingface_hub import get_token

        hub_token = bool(get_token())
    except ImportError:
        hub_token = False
    checks = {
        "python": {
            "ok": sys.version_info >= (3, 10),
            "version": sys.version.split()[0],
        },
        "modules": modules,
        "git": git_state(),
        "huggingface_token": hub_token,
        "cuda": {
            "available": cuda_available,
            "devices": cuda_devices,
            "required": args.require_gpu,
        },
        "disk": {
            "path": str(probe),
            "free_gb": free_gb,
            "minimum_free_gb": args.minimum_free_gb,
            "ok": free_gb >= args.minimum_free_gb,
        },
        "files": files,
        "later_stage_files": later,
    }
    failures = []
    if not checks["python"]["ok"]:
        failures.append("python")
    if not all(modules.values()):
        failures.append("modules")
    if not all(files.values()):
        failures.append("files")
    if args.require_gpu and not cuda_available:
        failures.append("cuda")
    if not checks["disk"]["ok"]:
        failures.append("disk")
    checks["ok"] = not failures
    checks["failures"] = failures
    print(json.dumps(checks, indent=2, sort_keys=True))
    return int(args.strict and bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
