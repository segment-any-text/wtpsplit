#!/usr/bin/env python
"""Plan or render the four primary Stage-1 corpus × backbone runs."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CONFIGS = {
    "mc4_xlmr": "configs/curriculum/stage1_mc4.json",
    "fineweb2_xlmr": "configs/curriculum/stage1_fineweb.json",
    "mc4_mmbert": "configs/curriculum/stage1_mc4_mmbert.json",
    "fineweb2_mmbert": "configs/curriculum/stage1_fineweb_mmbert.json",
}
FOLLOWUP_CONFIGS = {
    "mc4_documents_xlmr": "configs/curriculum/stage1_mc4_documents.json",
    "fineweb2_documents_xlmr": "configs/curriculum/stage1_fineweb_documents.json",
}
MATCHED_CONFIGS = {**PRIMARY_CONFIGS, **FOLLOWUP_CONFIGS}


def unsupported_config_keys(config: dict) -> list[str]:
    """Return keys that the Stage-1 training parser cannot consume."""

    from transformers import TrainingArguments

    from wtpsplit.train.train import Args
    from wtpsplit.utils import LabelArgs

    allowed = {
        field.name
        for dataclass_type in (Args, TrainingArguments, LabelArgs)
        for field in fields(dataclass_type)
    }
    return sorted(set(config) - allowed)


def distributed_settings(
    config: dict, *, world_size: int, target_global_batch: int = 512
) -> dict:
    """Derive accumulation without changing the declared global batch."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    micro_batch = int(config.get("per_device_train_batch_size") or 0)
    divisor = micro_batch * world_size
    if divisor <= 0 or target_global_batch % divisor:
        raise ValueError(
            f"global batch {target_global_batch} is not divisible by "
            f"per-device batch {micro_batch} × world size {world_size}"
        )
    accumulation = target_global_batch // divisor
    return {
        "per_device_train_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "world_size": world_size,
        "global_batch_size": micro_batch * accumulation * world_size,
    }


def build_run_plan(
    *, world_size: int, render_dir: Path | None = None, root: Path = ROOT
) -> dict:
    """Return a dry plan; write configs only when ``render_dir`` is supplied."""

    arms = []
    for arm, relative in PRIMARY_CONFIGS.items():
        source = root / relative
        config = json.loads(source.read_text(encoding="utf-8"))
        unsupported = unsupported_config_keys(config)
        if unsupported:
            raise ValueError(f"{relative}: unsupported training keys: {unsupported}")
        distributed = distributed_settings(config, world_size=world_size)
        rendered = None
        command = None
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            rendered = render_dir / f"{arm}.json"
            rendered.write_text(
                json.dumps(
                    {
                        **config,
                        "gradient_accumulation_steps": distributed[
                            "gradient_accumulation_steps"
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            command = " ".join(
                shlex.quote(part)
                for part in (
                    "torchrun",
                    f"--nproc_per_node={world_size}",
                    "wtpsplit/train/train.py",
                    str(rendered),
                )
            )
        arms.append(
            {
                "arm": arm,
                "source_config": relative,
                "rendered_config": str(rendered) if rendered else None,
                "distributed_batch": distributed,
                "launch_command": command,
            }
        )
    return {
        "schema": "stage1-run-plan-v1",
        "dry_run": render_dir is None,
        "target_global_batch": 512,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    report = build_run_plan(
        world_size=args.nproc_per_node,
        render_dir=args.render_dir,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
