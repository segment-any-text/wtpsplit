import json
from pathlib import Path

import pytest

from transformers import HfArgumentParser, TrainingArguments

from scripts.prepare_stage1_runs import (
    MATCHED_CONFIGS,
    PRIMARY_CONFIGS,
    unsupported_config_keys,
)
from wtpsplit.train.train import Args
from wtpsplit.utils import LabelArgs


ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = "configs/mmsat_3l.json"


@pytest.mark.parametrize("relative", [*MATCHED_CONFIGS.values(), PILOT_CONFIG])
def test_stage1_config_has_only_supported_training_keys(relative: str):
    config = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert unsupported_config_keys(config) == []


@pytest.mark.parametrize("relative", [*MATCHED_CONFIGS.values(), PILOT_CONFIG])
def test_stage1_config_parses_with_the_training_entrypoint(relative: str):
    config = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    config.update(use_cpu=True, bf16=False, fp16=False, tf32=False)
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs])
    parser.parse_dict(config)


@pytest.mark.parametrize("relative", PRIMARY_CONFIGS.values())
def test_primary_stage1_config_preserves_one_gpu_global_batch(relative: str):
    config = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    ) == 512
