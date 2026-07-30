"""LoRA training for in-process SaT adaptation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from wtpsplit.utils import Constants

if TYPE_CHECKING:
    from collections.abc import Iterable


_TARGET_SUFFIXES = {
    "xlm-token": (
        "attention.self.query",
        "attention.self.value",
        "intermediate.dense",
    ),
    "modernbert-token": (
        "attn.Wqkv",
        "mlp.Wi",
    ),
}


@dataclass
class AdapterState:
    weights: dict[str, torch.Tensor]
    head_weights: dict[str, torch.Tensor]
    history: list[float]
    rank: int
    alpha: float
    dropout: float
    target_modules: list[str]
    model_type: str
    model_class: str
    model_name: str
    hidden_size: int
    num_hidden_layers: int
    num_labels: int
    id2label: dict
    label2id: dict


@dataclass
class _TrainingExample:
    input_ids: list[int]
    labels: list[float]


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + delta * self.scaling

    def merge(self) -> None:
        delta = self.lora_B @ self.lora_A
        with torch.no_grad():
            self.base.weight.add_(delta.to(self.base.weight.dtype) * self.scaling)

    def unmerge(self) -> None:
        delta = self.lora_B @ self.lora_A
        with torch.no_grad():
            self.base.weight.sub_(delta.to(self.base.weight.dtype) * self.scaling)


def _validate_sentences(sentences: Iterable[str]) -> list[str]:
    if isinstance(sentences, str):
        raise TypeError("`sentences` must be an iterable of gold sentences, not one string.")

    values = list(sentences)
    if len(values) < 2:
        raise ValueError("`sentences` must contain at least two gold sentences.")

    normalized = []
    for index, sentence in enumerate(values):
        if not isinstance(sentence, str):
            raise TypeError(f"`sentences[{index}]` must be a string, got {type(sentence).__name__}.")
        if "\n" in sentence or "\r" in sentence:
            raise ValueError(f"`sentences[{index}]` contains a newline. Pass one gold sentence per list item.")
        sentence = sentence.strip()
        if not sentence:
            raise ValueError(f"`sentences[{index}]` is empty or whitespace-only.")
        normalized.append(sentence)
    return normalized


def _separator_for(language: str | None) -> str:
    if language is None:
        return " "
    if language not in Constants.SEPARATORS:
        raise ValueError(
            f"Unknown language code {language!r}. Pass a code present in `wtpsplit.utils.Constants.SEPARATORS`, "
            "or omit `language` to use a space separator."
        )
    return Constants.SEPARATORS[language]


def _encode_group(tokenizer, sentences: list[str], separator: str) -> _TrainingExample:
    text = separator.join(sentences)
    boundaries = []
    offset = 0
    for sentence in sentences:
        offset += len(sentence)
        boundaries.append(offset)
        offset += len(separator)

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = encoded["offset_mapping"]
    labels = [0.0] * len(offsets)

    for boundary in boundaries:
        candidates = [
            index for index, (start, end) in enumerate(offsets) if end > start and start < boundary and end <= boundary
        ]
        if not candidates:
            candidates = [
                index for index, (start, end) in enumerate(offsets) if end > start and start < boundary <= end
            ]
        if not candidates:
            raise ValueError(
                "Could not align a gold sentence boundary to a tokenizer position. "
                "Check the sentence text and tokenizer."
            )
        labels[candidates[-1]] = 1.0

    return _TrainingExample(input_ids=encoded["input_ids"], labels=labels)


def _prepare_examples(tokenizer, sentences: list[str], separator: str, block_size: int) -> list[_TrainingExample]:
    if block_size < 8:
        raise ValueError("`block_size` must be at least 8.")

    examples = []
    current: list[str] = []
    for index, sentence in enumerate(sentences):
        candidate = [*current, sentence]
        encoded = _encode_group(tokenizer, candidate, separator)
        if len(encoded.input_ids) <= block_size:
            current = candidate
            continue

        if not current:
            raise ValueError(
                f"`sentences[{index}]` needs {len(encoded.input_ids)} tokens, exceeding block_size={block_size}."
            )
        examples.append(_encode_group(tokenizer, current, separator))
        current = [sentence]
        single = _encode_group(tokenizer, current, separator)
        if len(single.input_ids) > block_size:
            raise ValueError(
                f"`sentences[{index}]` needs {len(single.input_ids)} tokens, exceeding block_size={block_size}."
            )

    if current:
        examples.append(_encode_group(tokenizer, current, separator))
    return examples


def _batch(examples: list[_TrainingExample], indices: torch.Tensor, pad_token_id: int, device):
    selected = [examples[index] for index in indices.tolist()]
    length = max(len(example.input_ids) for example in selected)
    input_ids = torch.full((len(selected), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(selected), length), dtype=torch.long)
    labels = torch.full((len(selected), length), -1.0, dtype=torch.float32)
    for row, example in enumerate(selected):
        size = len(example.input_ids)
        input_ids[row, :size] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row, :size] = 1
        labels[row, :size] = torch.tensor(example.labels, dtype=torch.float32)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = module_name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child_name


def _inject_lora(
    model: nn.Module, rank: int, alpha: float, dropout: float
) -> list[tuple[str, nn.Module, str, _LoRALinear]]:
    model_type = model.config.model_type
    if model_type not in _TARGET_SUFFIXES:
        raise ValueError(
            f"`SaT.adapt()` does not know LoRA targets for model_type={model_type!r}. "
            f"Supported model types: {sorted(_TARGET_SUFFIXES)}."
        )

    targets = []
    try:
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name.endswith(_TARGET_SUFFIXES[model_type]):
                parent, child_name = _resolve_parent(model, name)
                wrapped = _LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, wrapped)
                targets.append((name, parent, child_name, wrapped))
    except Exception:
        for _, parent, child_name, wrapped in targets:
            setattr(parent, child_name, wrapped.base)
        raise

    if not targets:
        raise ValueError(
            f"No LoRA target layers were found for model_type={model_type!r}; "
            "the model architecture may be incompatible with this wtpsplit version."
        )
    return targets


def _validate_head(model: nn.Module) -> dict[str, nn.Parameter]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise ValueError("`SaT.adapt()` requires a linear token-classification head named `classifier`.")

    expected = int(model.config.num_labels)
    actual = classifier.out_features
    if actual != expected:
        raise ValueError(
            "The base model and classification head sizes disagree: "
            f"config.num_labels={expected}, classifier.out_features={actual}. "
            "Reload the matching base checkpoint before adapting."
        )
    return {name: parameter for name, parameter in model.named_parameters() if name.startswith("classifier.")}


def _linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def adapt_model(
    sat,
    sentences: Iterable[str],
    *,
    language: str | None,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    block_size: int,
    rank: int,
    alpha: float,
    dropout: float,
    seed: int,
    show_progress: bool,
) -> list[float]:
    """Train, merge, and retain a reloadable LoRA adapter on ``sat``."""
    if sat.ort_providers is not None:
        raise ValueError("`SaT.adapt()` requires a PyTorch model; ONNX Runtime models cannot be trained.")
    if getattr(sat, "_compiled", False):
        raise ValueError("`SaT.adapt()` requires an eager model. Construct `SaT(..., compile=False)` before adapting.")
    if sat.use_lora or hasattr(sat, "_adapter_state"):
        raise ValueError("This SaT instance is already adapted. Construct a fresh base model to adapt again.")
    if epochs < 1:
        raise ValueError("`epochs` must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("`learning_rate` must be positive.")
    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")
    if rank < 1:
        raise ValueError("`rank` must be at least 1.")
    if alpha <= 0:
        raise ValueError("`alpha` must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("`dropout` must be in the range [0, 1).")

    values = _validate_sentences(sentences)
    separator = _separator_for(language)
    examples = _prepare_examples(sat.tokenizer, values, separator, block_size)

    model = sat.model.model
    head_parameters = _validate_head(model)
    original_training = model.training
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    original_head = {name: parameter.detach().cpu().clone() for name, parameter in head_parameters.items()}
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    targets = []
    merged_targets = []
    success = False
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for parameter in model.parameters():
            parameter.requires_grad = False
        targets = _inject_lora(model, rank=rank, alpha=alpha, dropout=dropout)
        for parameter in head_parameters.values():
            parameter.requires_grad = True

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
        steps_per_epoch = math.ceil(len(examples) / batch_size)
        total_steps = epochs * steps_per_epoch
        scheduler = _linear_schedule(optimizer, max(1, int(total_steps * 0.1)), total_steps)
        generator = torch.Generator().manual_seed(seed)
        device = next(model.parameters()).device
        pad_token_id = sat.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define `pad_token_id` for adaptation.")

        history = []
        progress = tqdm(range(epochs), disable=not show_progress, desc="Adapting SaT", unit="epoch")
        model.train()
        for _ in progress:
            permutation = torch.randperm(len(examples), generator=generator)
            epoch_loss = 0.0
            for start in range(0, len(examples), batch_size):
                indices = permutation[start : start + batch_size]
                input_ids, attention_mask, labels = _batch(examples, indices, pad_token_id, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output["logits"][:, :, 0]
                valid = labels >= 0
                loss = F.binary_cross_entropy_with_logits(logits[valid].float(), labels[valid])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.detach().item()

            epoch_loss /= steps_per_epoch
            history.append(epoch_loss)
            progress.set_postfix(loss=f"{epoch_loss:.4f}")

        adapter_weights = {}
        target_names = []
        for name, _, _, wrapped in targets:
            target_names.append(name)
            adapter_weights[f"{name}.loras.text.lora_A"] = wrapped.lora_A.detach().cpu().clone()
            adapter_weights[f"{name}.loras.text.lora_B"] = wrapped.lora_B.detach().cpu().clone()

        head_weights = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("classifier.")
        }
        config = model.config
        state = AdapterState(
            weights=adapter_weights,
            head_weights=head_weights,
            history=history,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_names,
            model_type=config.model_type,
            model_class=type(model).__name__,
            model_name=str(sat.model_name_or_model),
            hidden_size=int(config.hidden_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_labels=int(config.num_labels),
            id2label=dict(getattr(config, "id2label", {})),
            label2id=dict(getattr(config, "label2id", {})),
        )
        for _, _, _, wrapped in targets:
            wrapped.merge()
            merged_targets.append(wrapped)
        sat._adapter_state = state
        success = True
        return history
    finally:
        if not success:
            for wrapped in reversed(merged_targets):
                wrapped.unmerge()
        for _, parent, child_name, wrapped in targets:
            setattr(parent, child_name, wrapped.base)

        if not success:
            current_parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in original_head.items():
                    current_parameters[name].copy_(
                        value.to(device=current_parameters[name].device, dtype=current_parameters[name].dtype)
                    )

        current_parameters = dict(model.named_parameters())
        for name, requires_grad in original_requires_grad.items():
            if name in current_parameters:
                current_parameters[name].requires_grad = requires_grad
        model.train(original_training if not success else False)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def save_adapter(sat, output_dir: str | Path) -> Path:
    state: AdapterState | None = getattr(sat, "_adapter_state", None)
    if state is None:
        raise RuntimeError("No in-process adapter is available. Call `sat.adapt(...)` before `save_adapter()`.")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"`output_dir` must not contain files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "config": {
            "alpha": state.alpha,
            "architecture": "lora",
            "dropout": state.dropout,
            "r": state.rank,
            "target_modules": state.target_modules,
        },
        "hidden_size": state.hidden_size,
        "model_class": state.model_class,
        "model_name": state.model_name,
        "model_type": state.model_type,
        "num_hidden_layers": state.num_hidden_layers,
        "name": "text",
        "version": "wtpsplit-3",
    }
    head_config = {
        "config": None,
        "hidden_size": state.hidden_size,
        "id2label": state.id2label,
        "label2id": state.label2id,
        "model_class": state.model_class,
        "model_name": state.model_name,
        "model_type": state.model_type,
        "name": None,
        "num_labels": state.num_labels,
        "version": "wtpsplit-3",
    }

    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "head_config.json").write_text(
        json.dumps(head_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(state.weights, output_dir / "pytorch_adapter.bin")
    torch.save(state.head_weights, output_dir / "pytorch_model_head.bin")
    return output_dir
