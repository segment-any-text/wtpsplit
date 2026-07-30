"""Tests for dependency-free in-process LoRA adaptation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoTokenizer

from wtpsplit import SaT, _manual_lora_merge
from wtpsplit.adaptation import (
    _inject_lora,
    adapt_model,
    save_adapter,
)
from wtpsplit.configs import SubwordModernBertConfig
from wtpsplit.models_modernbert import SubwordModernBertForTokenClassification
from scripts.evaluate_adaptation import evaluate_model

GOLD_SENTENCES = [
    "One sentence.",
    "Another gold sentence.",
    "A third example.",
    "The final segment.",
]


class PerfectSentenceModel:
    def split(self, text):
        segments = []
        start = 0
        for index in range(len(text) - 1):
            if text[index : index + 2] == ". ":
                segments.append(text[start : index + 2])
                start = index + 2
        segments.append(text[start:])
        return segments


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [1, *[3 + ord(character) % 61 for character in text], 2],
            "offset_mapping": [(0, 0), *[(index, index + 1) for index in range(len(text))], (0, 0)],
        }


class XLMBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(hidden_size, hidden_size)
        self.attention.self.value = nn.Linear(hidden_size, hidden_size)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden):
        return torch.tanh(
            self.attention.self.query(hidden) + self.attention.self.value(hidden) + self.intermediate.dense(hidden)
        )


class TinyXLM(nn.Module):
    def __init__(self, num_labels=1, classifier_labels=None):
        super().__init__()
        hidden_size = 8
        self.config = SimpleNamespace(
            model_type="xlm-token",
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_labels=num_labels,
            id2label={index: f"LABEL_{index}" for index in range(num_labels)},
            label2id={f"LABEL_{index}": index for index in range(num_labels)},
        )
        self.embeddings = nn.Embedding(64, hidden_size)
        self.roberta = nn.Module()
        self.roberta.encoder = nn.Module()
        self.roberta.encoder.layer = nn.ModuleList([XLMBlock(hidden_size)])
        self.classifier = nn.Linear(hidden_size, classifier_labels or num_labels)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embeddings(input_ids)
        for layer in self.roberta.encoder.layer:
            hidden = layer(hidden)
        return {"logits": self.classifier(hidden)}


class ModernBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Module()
        self.attn.Wqkv = nn.Linear(hidden_size, hidden_size * 3)
        self.mlp = nn.Module()
        self.mlp.Wi = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, hidden):
        query, key, value = self.attn.Wqkv(hidden).chunk(3, dim=-1)
        inputs, gate = self.mlp.Wi(hidden).chunk(2, dim=-1)
        return torch.tanh((query + key + value) / 3 + inputs * torch.sigmoid(gate))


class TinyModernBert(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 8
        self.config = SimpleNamespace(
            model_type="modernbert-token",
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_labels=1,
            id2label={0: "LABEL_0"},
            label2id={"LABEL_0": 0},
        )
        self.embeddings = nn.Embedding(64, hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ModernBlock(hidden_size)])
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embeddings(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return {"logits": self.classifier(hidden)}


def make_sat(model=None):
    torch.manual_seed(0)
    model = model or TinyXLM()
    return SimpleNamespace(
        model=SimpleNamespace(model=model),
        tokenizer=TinyTokenizer(),
        model_name_or_model="tiny-sat",
        ort_providers=None,
        _compiled=False,
        use_lora=False,
    )


def adapt_kwargs(**overrides):
    kwargs = {
        "language": "en",
        "epochs": 5,
        "learning_rate": 0.03,
        "batch_size": 2,
        "block_size": 64,
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.0,
        "seed": 13,
        "show_progress": False,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("model", "expected_suffixes"),
    [
        (TinyXLM(), {"attention.self.query", "attention.self.value", "intermediate.dense"}),
        (TinyModernBert(), {"attn.Wqkv", "mlp.Wi"}),
    ],
)
def test_lora_targets_cover_both_sat_backbones(model, expected_suffixes):
    targets = _inject_lora(model, rank=2, alpha=4, dropout=0)
    try:
        names = {name for name, _, _, _ in targets}
        assert all(any(name.endswith(suffix) for name in names) for suffix in expected_suffixes)
    finally:
        for _, parent, child_name, wrapped in targets:
            setattr(parent, child_name, wrapped.base)


def test_held_out_quality_evaluator_scores_documents():
    metrics = evaluate_model(PerfectSentenceModel(), GOLD_SENTENCES, "en", document_size=2)
    assert metrics == {
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "documents": 2,
    }


def test_adaptation_reduces_loss_and_retains_adapter_without_writes(tmp_path, monkeypatch):
    sat = make_sat()
    monkeypatch.chdir(tmp_path)
    history = adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=12))

    assert history[-1] < history[0]
    assert sat._adapter_state.history == history
    assert sat._adapter_state.weights
    assert list(tmp_path.iterdir()) == []


def test_adaptation_runs_end_to_end_on_modernbert():
    sat = make_sat(TinyModernBert())
    history = adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=2))
    assert len(history) == 2
    assert all(".attn.Wqkv." in key or ".mlp.Wi." in key for key in sat._adapter_state.weights)


def test_adaptation_rolls_back_model_when_merge_fails(monkeypatch):
    sat = make_sat()
    original = deepcopy(sat.model.model.state_dict())

    def fail_merge(self):
        raise RuntimeError("synthetic merge failure")

    monkeypatch.setattr("wtpsplit.adaptation._LoRALinear.merge", fail_merge)
    with pytest.raises(RuntimeError, match="synthetic"):
        adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=1))

    for name, value in sat.model.model.state_dict().items():
        assert torch.equal(value, original[name])
    assert not hasattr(sat, "_adapter_state")


@pytest.mark.parametrize(
    ("sentences", "error", "match"),
    [
        ("one string", TypeError, "iterable"),
        (["only one"], ValueError, "at least two"),
        (["valid", "bad\nline"], ValueError, "newline"),
        (["valid", " "], ValueError, "whitespace"),
    ],
)
def test_adaptation_validates_gold_sentences(sentences, error, match):
    with pytest.raises(error, match=match):
        adapt_model(make_sat(), sentences, **adapt_kwargs(epochs=1))


def test_adaptation_rejects_incompatible_head():
    sat = make_sat(TinyXLM(num_labels=1, classifier_labels=2))
    with pytest.raises(ValueError, match="head sizes disagree"):
        adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=1))


def test_adaptation_rejects_onnx_compiled_and_already_adapted_models():
    sat = make_sat()
    sat.ort_providers = ["CPUExecutionProvider"]
    with pytest.raises(ValueError, match="PyTorch"):
        adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=1))

    sat = make_sat()
    sat._compiled = True
    with pytest.raises(ValueError, match="eager"):
        adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=1))

    sat = make_sat()
    sat.use_lora = True
    with pytest.raises(ValueError, match="already adapted"):
        adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=1))


def test_malformed_head_is_rejected_before_partial_merge(tmp_path):
    sat = make_sat()
    adapt_model(sat, GOLD_SENTENCES, **adapt_kwargs(epochs=1))
    save_adapter(sat, tmp_path)

    head_path = tmp_path / "pytorch_model_head.bin"
    head = torch.load(head_path, weights_only=True)
    head["classifier.weight"] = torch.zeros(2, 8)
    torch.save(head, head_path)

    fresh = make_sat()
    original = deepcopy(fresh.model.model.state_dict())
    with pytest.raises(ValueError, match="shape mismatch"):
        _manual_lora_merge(fresh.model.model, tmp_path)
    for name, value in fresh.model.model.state_dict().items():
        assert torch.equal(value, original[name])


def test_public_adapt_save_and_reload_round_trip(tmp_path):
    sat = SaT("segment-any-text/sat-3l-sm", hub_prefix=None, device="cpu")
    returned = sat.adapt(
        GOLD_SENTENCES,
        language="en",
        epochs=1,
        batch_size=2,
        block_size=64,
        show_progress=False,
    )
    probabilities = sat.predict_proba("One sentence. Another gold sentence.")

    assert returned is sat
    assert sat.use_lora
    assert len(sat.adaptation_history) == 1
    assert sat.save_adapter(tmp_path) == tmp_path
    assert {path.name for path in tmp_path.iterdir()} == {
        "adapter_config.json",
        "pytorch_adapter.bin",
        "head_config.json",
        "pytorch_model_head.bin",
    }

    reloaded = SaT("segment-any-text/sat-3l-sm", hub_prefix=None, lora_path=tmp_path)
    reloaded_probabilities = reloaded.predict_proba("One sentence. Another gold sentence.")
    np.testing.assert_allclose(reloaded_probabilities, probabilities, rtol=0, atol=0)


def test_real_modernbert_architecture_adapts_and_reloads(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    config = SubwordModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        local_attention=128,
        num_labels=1,
        lookahead=4,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        attn_implementation="sdpa",
    )
    model = SubwordModernBertForTokenClassification(config)
    model.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)

    sat = SaT(base_path, tokenizer_name_or_path=base_path, hub_prefix=None, device="cpu")
    sat.adapt(
        GOLD_SENTENCES,
        language="en",
        epochs=1,
        batch_size=2,
        block_size=64,
        rank=2,
        alpha=4,
        show_progress=False,
    )
    probabilities = sat.predict_proba("One sentence. Another gold sentence.")
    sat.save_adapter(adapter_path)

    reloaded = SaT(
        base_path,
        tokenizer_name_or_path=base_path,
        hub_prefix=None,
        lora_path=adapter_path,
        device="cpu",
    )
    reloaded_probabilities = reloaded.predict_proba("One sentence. Another gold sentence.")
    np.testing.assert_allclose(reloaded_probabilities, probabilities, rtol=0, atol=0)
