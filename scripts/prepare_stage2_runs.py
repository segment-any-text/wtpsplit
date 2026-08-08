#!/usr/bin/env python
"""Render a matched Stage 2 no-replay/replay experiment from one base config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from scripts.validate_stage2 import validate_config


def render_pair(
    base: dict,
    *,
    model: str,
    tokenizer: str,
    data_path: str,
    replay_data_path: str,
    replay_fraction: float,
    output_root: str,
    seed: int,
    max_steps: int,
    use_character_head: bool = False,
    character_head_init: str = "identity",
) -> dict[str, dict]:
    common = {
        **base,
        "model_name_or_path": model,
        "tokenizer_name_or_path": tokenizer,
        "data_path": data_path,
        "seed": seed,
        "data_seed": seed,
        "max_steps": max_steps,
        "use_character_head": use_character_head,
        "character_head_init": character_head_init,
        "eval_strategy": "no",
        "report_to": "none",
        "run_final_evaluation": False,
    }
    for key in (
        "replay_data_path",
        "replay_fraction",
        "replay_training_dataset",
        "replay_languages",
        "max_replay_train_sentences_per_dataset",
    ):
        common.pop(key, None)
    no_replay = {**common, "output_dir": f"{output_root}/no_replay"}
    replay = {
        **common,
        "output_dir": f"{output_root}/replay",
        "replay_data_path": replay_data_path,
        "replay_fraction": replay_fraction,
    }
    for name, config in (("no_replay", no_replay), ("replay", replay)):
        errors, _ = validate_config(config)
        if errors:
            raise ValueError(f"{name}: {'; '.join(errors)}")
    return {"no_replay": no_replay, "replay": replay}


def launch_command(path: Path) -> str:
    return " ".join(
        shlex.quote(part)
        for part in ("uv", "run", "python", "wtpsplit/train/train_SM.py", str(path))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/curriculum/stage2_no_replay.json"),
    )
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--tokenizer")
    parser.add_argument("--data-path", default="data/mmsat_stage2_abc_pilot_v1.pth")
    parser.add_argument(
        "--replay-data-path",
        default="data/mmsat_stage2_historical_control_v1.pth",
    )
    parser.add_argument("--replay-fraction", type=float, default=0.5)
    parser.add_argument("--output-root", default="runs/stage2_retention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument(
        "--head",
        choices=("token", "character"),
        default="token",
        help="Render the ordinary token head or the optional character head.",
    )
    parser.add_argument(
        "--character-head-init",
        choices=("identity", "random"),
        default="random",
        help="Character-head initialization; ignored for --head token.",
    )
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.replay_fraction < 1:
        raise ValueError("--replay-fraction must be between zero and one")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    configs = render_pair(
        base,
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        data_path=args.data_path,
        replay_data_path=args.replay_data_path,
        replay_fraction=args.replay_fraction,
        output_root=args.output_root,
        seed=args.seed,
        max_steps=args.max_steps,
        use_character_head=args.head == "character",
        character_head_init=(
            args.character_head_init if args.head == "character" else "identity"
        ),
    )
    report = {"dry_run": args.render_dir is None, "head": args.head, "arms": {}}
    for name, config in configs.items():
        path = (
            args.render_dir / f"{name}.json"
            if args.render_dir
            else Path("<render-dir>") / f"{name}.json"
        )
        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=True)
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        report["arms"][name] = {
            "config": str(path),
            "output_dir": config["output_dir"],
            "replay_fraction": config.get("replay_fraction", 0.0),
            "launch": launch_command(path),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
