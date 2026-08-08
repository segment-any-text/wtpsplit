from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file
from transformers import BertConfig, BertForTokenClassification

from wtpsplit.train.trainer import Trainer
from wtpsplit.train.utils import Model


def test_stage1_trainer_saves_loadable_unwrapped_backbone(tmp_path: Path):
    backbone = BertForTokenClassification(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_labels=3,
        )
    )
    wrapped = Model(backbone, do_auxiliary_training=True)
    # Exercise the custom save implementation without requiring the optional
    # accelerate runtime merely to construct a full HF Trainer.
    trainer = object.__new__(Trainer)
    trainer.model = wrapped
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.data_collator = None

    trainer._save(str(tmp_path), state_dict=wrapped.state_dict())

    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "model.safetensors").is_file()
    state_dict = load_file(str(tmp_path / "model.safetensors"))
    assert state_dict
    assert not any(key.startswith("backbone.") for key in state_dict)

    restored = BertForTokenClassification.from_pretrained(tmp_path)
    assert restored.config.num_labels == 3
