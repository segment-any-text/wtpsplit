"""Character-resolution boundary head.

SaT predicts one logit per subword token and `token_to_char_probs` then assigns it to the
*last character* of that token, leaving every other character at `-inf`. A boundary that
falls elsewhere is therefore unreachable at any threshold, which caps recall independently
of training.

This head removes the cap on both sides, turning the backbone swap from a trade into a
strict improvement. It predicts a logit at *every* character position from the covering
token's hidden state plus character-local features, so the reachable set is the whole
string and the ceiling is 1.0 by construction.

Behaviour-preserving initialisation. At initialisation the head reproduces the
existing last-character mapping exactly: the residual path is zero-initialised and a
learned bias suppresses non-final characters. An existing SaT checkpoint can therefore be
warm-started and fine-tuned rather than retrained.

The head is enabled 3 with `use_character_head=true` (in stage 3?). Both supported backbone
wrappers attach it from their serialized config, and `train_SM.py` dynamically constructs
the per-character inputs and padded targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import default_data_collator

from wtpsplit.utils import PRIMES

__all__ = [
    "CharacterResolutionHead",
    "CharacterDataCollator",
    "CharInputs",
    "build_char_inputs",
    "char_boundary_labels",
    "char_boundary_label_batch",
]

# Large enough to zero out a character after a sigmoid, small enough to stay finite in
# fp16 so the head can be trained in mixed precision (`-inf` would produce NaNs).
SUPPRESSION_BIAS = 12.0


@dataclass
class CharInputs:
    """Per-character indexing into the token sequence.

    All tensors are `[batch, num_chars]` except `hashes`, which is
    `[batch, num_chars, num_hashes]`.
    """

    char_to_token: torch.Tensor  # index of the covering token; 0 where padded
    is_token_final: torch.Tensor  # 1 where the character ends its token
    position_in_token: torch.Tensor  # offset of the character within its token, clamped
    hashes: torch.Tensor  # CANINE-style hashed codepoints
    mask: torch.Tensor  # 1 for real characters


def build_char_inputs(
    texts: list[str],
    offset_mappings: list[list[tuple[int, int]]],
    *,
    num_hashes: int = 8,
    num_buckets: int = 8192,
    max_position_in_token: int = 15,
    device: torch.device | None = None,
) -> CharInputs:
    """Map characters onto the tokens covering them.

    `offset_mappings` is what a fast tokenizer returns with `return_offsets_mapping=True`,
    already stripped of special tokens. Characters covered by no token (which a tokenizer
    can produce for stripped whitespace) point at token 0 and are suppressed by `mask`.
    """
    batch = len(texts)
    width = max((len(t) for t in texts), default=0)

    char_to_token = np.zeros((batch, width), dtype=np.int64)
    is_token_final = np.zeros((batch, width), dtype=np.int64)
    position_in_token = np.zeros((batch, width), dtype=np.int64)
    ordinals = np.zeros((batch, width), dtype=np.int64)
    mask = np.zeros((batch, width), dtype=np.int64)

    for i, (text, offsets) in enumerate(zip(texts, offset_mappings)):
        length = len(text)
        mask[i, :length] = 1
        ordinals[i, :length] = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)

        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            end = min(end, length)
            char_to_token[i, start:end] = token_index
            position_in_token[i, start:end] = np.minimum(np.arange(end - start), max_position_in_token)
            is_token_final[i, end - 1] = 1

    # Same hashing scheme as the CANINE-style char models in `wtpsplit.utils.hash_encode`,
    # vectorised over the batch.
    primes = np.array(PRIMES[:num_hashes], dtype=np.int64)
    hashes = ((ordinals[..., None] + 1) * primes) % num_buckets

    def tensor(array):
        return torch.as_tensor(array, device=device)

    return CharInputs(
        char_to_token=tensor(char_to_token),
        is_token_final=tensor(is_token_final),
        position_in_token=tensor(position_in_token),
        hashes=tensor(hashes),
        mask=tensor(mask),
    )


def char_boundary_labels(sentences: list[str], separator: str) -> np.ndarray:
    """Per-character boundary targets for a document built from gold sentences.

    A label of 1 marks the **last content character** of each sentence, which is the
    position `indices_to_sentences` needs in order to reconstruct that split. This is the
    same convention `train_SM.py` already computes internally as
    `start_position + len(sentence) - 1` before downsampling it onto tokens — the
    character positions are available, they are simply discarded today.

    The final sentence gets no label: `indices_to_sentences` always appends the tail, so
    a boundary there is free for every model and training on it teaches nothing.
    """
    text_length = sum(len(s) for s in sentences) + len(separator) * max(len(sentences) - 1, 0)
    labels = np.zeros(text_length, dtype=np.int64)

    position = 0
    for index, sentence in enumerate(sentences):
        end = position + len(sentence)
        if index < len(sentences) - 1:
            labels[end - 1] = 1
        position = end + len(separator)
    return labels


def char_boundary_label_batch(
    documents: list[list[str]],
    separators: list[str],
    *,
    width: int | None = None,
    ignore_index: int = -100,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Padded `[batch, num_chars]` targets, with padding set to `ignore_index`."""
    per_document = [char_boundary_labels(s, sep) for s, sep in zip(documents, separators)]
    if width is None:
        width = max((len(labels) for labels in per_document), default=0)

    batch = np.full((len(per_document), width), ignore_index, dtype=np.int64)
    for i, labels in enumerate(per_document):
        length = min(len(labels), width)
        batch[i, :length] = labels[:length]
    return torch.as_tensor(batch, device=device)


class CharacterDataCollator:
    """Dynamically pad character features while token blocks remain fixed-width."""

    def __init__(self, special_tokens_before: int = 1):
        self.special_tokens_before = special_tokens_before

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        features = [dict(feature) for feature in features]
        texts = [feature.pop("text") for feature in features]
        offsets = [feature.pop("offset_mapping") for feature in features]
        labels = [feature.pop("labels") for feature in features]
        batch = default_data_collator(features)

        char_inputs = build_char_inputs(texts, offsets)
        batch["char_to_token"] = char_inputs.char_to_token + self.special_tokens_before
        batch["char_is_token_final"] = char_inputs.is_token_final
        batch["char_position_in_token"] = char_inputs.position_in_token
        batch["char_hashes"] = char_inputs.hashes
        batch["char_mask"] = char_inputs.mask

        width = char_inputs.mask.shape[1]
        padded_labels = torch.full((len(labels), width), -100, dtype=torch.long)
        for index, values in enumerate(labels):
            padded_labels[index, : len(values)] = torch.as_tensor(values)
        batch["labels"] = padded_labels
        return batch


class CharacterResolutionHead(nn.Module):
    """Per-character boundary logits from per-token hidden states.

    The output is

        logit(c) = token_logit(t(c)) + bias(is_final(c)) + delta(c)

    where `delta` is a small convolutional network over character features. At
    initialisation `delta` is exactly zero and `bias` is `[-SUPPRESSION_BIAS, 0]`, so the
    head reproduces the current last-character behaviour and training only has to learn
    where to *relax* it.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 1,
        *,
        num_hashes: int = 8,
        num_buckets: int = 8192,
        char_embedding_size: int = 64,
        bottleneck_size: int = 128,
        kernel_size: int = 5,
        max_position_in_token: int = 15,
    ):
        super().__init__()
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.max_position_in_token = max_position_in_token

        self.char_embeddings = nn.Embedding(num_buckets, char_embedding_size)
        self.position_embeddings = nn.Embedding(max_position_in_token + 1, char_embedding_size)
        self.final_embeddings = nn.Embedding(2, char_embedding_size)

        self.project = nn.Linear(hidden_size + char_embedding_size, bottleneck_size)
        self.activation = nn.GELU()
        # Depthwise-separable conv so neighbouring characters can disambiguate which
        # position inside a token carries the boundary, at negligible cost.
        self.context = nn.Conv1d(
            bottleneck_size,
            bottleneck_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=bottleneck_size,
        )
        self.delta = nn.Linear(bottleneck_size, num_labels)

        # `bias[1]` (token-final) stays at 0; `bias[0]` suppresses every other character.
        self.suppression = nn.Embedding(2, num_labels)

        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        """Initialise so the head reproduces `token_to_char_probs` exactly."""
        self.reset_missing_to_identity(
            delta_weight=True,
            delta_bias=True,
            suppression=True,
        )

    def reset_missing_to_identity(
        self,
        *,
        delta_weight: bool,
        delta_bias: bool,
        suppression: bool,
    ) -> None:
        """Restore identity values for components absent from a loaded checkpoint."""
        if delta_weight:
            nn.init.zeros_(self.delta.weight)
        if delta_bias:
            nn.init.zeros_(self.delta.bias)
        with torch.no_grad():
            if suppression:
                self.suppression.weight[0].fill_(-SUPPRESSION_BIAS)
                self.suppression.weight[1].fill_(0.0)

    def _depthwise(self, hidden: torch.Tensor) -> torch.Tensor:
        """`self.context` applied as shifted multiply-adds instead of `nn.Conv1d`.

        Mathematically identical and uses the same parameters, so checkpoints stay
        interchangeable with a standard depthwise `Conv1d`. PyTorch's depthwise kernel is
        pathologically slow on CPU for long sequences — it measured 1277 ms of a 1298 ms
        head, against 21 ms for everything else — whereas `kernel_size` shifted
        elementwise products are trivial.
        """
        x = hidden.transpose(1, 2)  # [B, C, T]
        kernel_size = self.context.kernel_size[0]
        padding = kernel_size // 2
        padded = nn.functional.pad(x, (padding, padding))
        weight = self.context.weight.squeeze(1)  # [C, k]

        length = x.size(-1)
        out = self.context.bias.view(1, -1, 1).expand_as(x).clone()
        for offset in range(kernel_size):
            out = out + weight[:, offset].view(1, -1, 1) * padded[:, :, offset : offset + length]
        return out.transpose(1, 2)

    def forward(
        self,
        token_hidden: torch.Tensor,  # [B, T, D]
        token_logits: torch.Tensor,  # [B, T, num_labels]
        char_inputs: CharInputs,
    ) -> torch.Tensor:
        """Returns per-character logits `[B, C, num_labels]`."""
        index = char_inputs.char_to_token.unsqueeze(-1)

        gathered_hidden = token_hidden.gather(1, index.expand(-1, -1, token_hidden.size(-1)))
        gathered_logits = token_logits.gather(1, index.expand(-1, -1, token_logits.size(-1)))

        char_features = (
            self.char_embeddings(char_inputs.hashes).sum(dim=-2)
            + self.position_embeddings(char_inputs.position_in_token)
            + self.final_embeddings(char_inputs.is_token_final)
        )

        hidden = self.activation(self.project(torch.cat([gathered_hidden, char_features], dim=-1)))
        hidden = self._depthwise(hidden)
        delta = self.delta(hidden)

        logits = gathered_logits + self.suppression(char_inputs.is_token_final) + delta
        return logits.masked_fill(char_inputs.mask.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)
