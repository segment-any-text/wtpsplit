"""Pure preparation and reduction stages for bounded SaT inference.

Window order, accumulator precision and rounding mirror the original extractors.
These functions do not load models, create workers or change runtime settings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from wtpsplit._host_kernels import pack_subword_windows, project_subword_proba, stitch_subword_logits
from wtpsplit._inference_features import BatchFeatures, CharacterWindow
from wtpsplit.utils import Constants, sigmoid

Array = NDArray[np.number[Any]]
ModelBatch = dict[str, Array]
Offsets = list[list[tuple[int, int]]]


class TokenizerEmptyError(ValueError):
    """A nonblank subword document has no tokens to score."""


class Tokenized(TypedDict):
    input_ids: list[list[int]]
    offset_mapping: Offsets


class Tokenizer(Protocol):
    """The offset-preserving subset of the fast tokenizer interface."""

    def __call__(
        self, texts: list[str], *, return_offsets_mapping: bool, verbose: bool, add_special_tokens: bool
    ) -> Tokenized: ...


@dataclass(frozen=True)
class InferenceSpec:
    """Resolved encoder geometry. Content excludes CLS/SEP slots."""

    kind: Literal["char", "subword"]
    num_labels: int
    content: int
    downsampling_rate: int
    stride: int
    block_size: int
    batch_size: int
    cls_id: int
    sep_id: int
    pad_id: int
    special_tokens: tuple[str, ...]
    weighting: Literal["uniform", "hat"] = "uniform"

    def __post_init__(self) -> None:
        for name in ("num_labels", "content", "downsampling_rate", "stride", "block_size", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stride > min(self.content, self.block_size):
            raise ValueError("stride must not exceed the content window")
        if self.kind not in ("char", "subword") or self.weighting not in ("uniform", "hat"):
            raise ValueError("Unknown inference kind or weighting")


@dataclass(frozen=True)
class PreparedRequest:
    batches: list[ModelBatch]
    state: RequestState


@dataclass(frozen=True)
class RequestState:
    original_lengths: tuple[int, ...]
    nonempty_indices: tuple[int, ...]
    lengths: tuple[int, ...]
    locations: NDArray[np.int64] | NDArray[np.int32]
    block_size: int
    input_ids: list[list[int]]
    offsets: Offsets


def _encoded(tokenizer: Tokenizer, texts: list[str]) -> tuple[list[list[int]], Offsets]:
    """Use tokenizer-native lists without copying each token/span in Python.

    Tokenizers satisfy the typed ids/offset contract. Validate request geometry;
    the feature builder handles unusual span geometry with its exact fallback.
    """
    encoded = tokenizer(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) != len(texts) or len(offsets) != len(texts):
        raise ValueError("Tokenizer must return one ids/offsets list per text")
    if any(len(row_ids) != len(row_offsets) for row_ids, row_offsets in zip(ids, offsets, strict=True)):
        raise ValueError("Tokenizer ids and offsets must align")
    return ids, offsets


def prepare(texts: Sequence[str], spec: InferenceSpec, tokenizer: Tokenizer) -> PreparedRequest | None:
    """Assemble inputs; reject a tokenizer-empty nonblank subword document.

    Empty and whitespace-only documents receive one zero probability per source
    character, including zero entries for a zero-length document. Public SaT's
    historical empty-input sentinel remains owned by its existing API.
    """
    indices = tuple(i for i, text in enumerate(texts) if text.strip())
    nonempty = [texts[i] for i in indices]
    ids, offsets = _encoded(tokenizer, nonempty) if nonempty else ([], [])
    if spec.kind == "subword":
        for index, row in enumerate(ids):
            if not row:
                raise TokenizerEmptyError(
                    f"Tokenizer produced no tokens for nonblank document at index {indices[index]}; "
                    "subword inference cannot score this input"
                )
    if spec.kind == "char":
        locations: NDArray[np.int64] | NDArray[np.int32]
        batches, locations = _prepare_char(nonempty, ids, offsets, spec)
        block_size = min(spec.block_size, spec.content)
        lengths = tuple(map(len, nonempty))
    else:
        batches, locations, block_size = _prepare_subword(ids, spec)
        lengths = tuple(map(len, ids))
    return PreparedRequest(
        batches,
        RequestState(tuple(map(len, texts)), indices, lengths, locations, block_size, ids, offsets),
    )


def _prepare_char(
    texts: list[str], ids: list[list[int]], offsets: Offsets, spec: InferenceSpec
) -> tuple[list[ModelBatch], NDArray[np.int64]]:
    block = min(spec.block_size, spec.content)
    cached = BatchFeatures(texts, offsets)
    windows: list[CharacterWindow] = []
    for index, (text, tokens, mapping) in enumerate(zip(texts, ids, offsets, strict=True)):
        for start in range(0, len(tokens), spec.stride):
            end = start + block
            done = end >= len(tokens)
            if done:
                end = len(tokens)
                start = max(end - block, 0)
            char_start = 0 if start == 0 else mapping[start][0]
            char_end = len(text) if end == len(tokens) else mapping[end - 1][1]
            reuse = cached.can_slice(index, start, end, char_start, char_end)
            relative = (
                [(max(s - char_start, 0), min(e - char_start, char_end - char_start)) for s, e in mapping[start:end]]
                if not reuse
                else []
            )
            windows.append(
                CharacterWindow(
                    index, start, char_start, char_end, text[char_start:char_end], relative, reuse, tokens[start:end]
                )
            )
            if done:
                break
    batches: list[ModelBatch] = []
    for start in range(0, len(windows), spec.batch_size):
        batch = windows[start : start + spec.batch_size]
        width = max(len(w.input_ids) + 2 for w in batch)
        input_ids = np.full((len(batch), width), spec.pad_id, dtype=np.int64)
        attention = np.zeros_like(input_ids, dtype=np.float32)
        for row, window in enumerate(batch):
            sequence = [spec.cls_id, *window.input_ids, spec.sep_id]
            input_ids[row, : len(sequence)] = sequence
            attention[row, : len(sequence)] = 1
        batches.append(dict(input_ids=input_ids, attention_mask=attention, **cached.batch(batch)))
    locations = np.array([(w.text_index, w.char_start, w.char_end) for w in windows], dtype=np.int64).reshape(-1, 3)
    return batches, locations


def _prepare_subword(ids: list[list[int]], spec: InferenceSpec) -> tuple[list[ModelBatch], NDArray[np.int32], int]:
    if not ids:
        return [], np.empty((0, 3), dtype=np.int32), 0
    input_ids, attention, locations, block = pack_subword_windows(
        ids,
        block_size=min(spec.block_size, spec.content),
        stride=spec.stride,
        downsampling_rate=spec.downsampling_rate,
        cls_id=spec.cls_id,
        sep_id=spec.sep_id,
    )
    count = len(locations)
    return (
        [
            dict(input_ids=input_ids[s : s + spec.batch_size], attention_mask=attention[s : s + spec.batch_size])
            for s in range(0, count, spec.batch_size)
        ],
        locations,
        block,
    )


def _weights(length: int, weighting: str, dtype: type[np.float16] | type[np.float32]) -> Array:
    if weighting == "uniform" or length <= 1:
        return np.ones(length, dtype=dtype)
    x = np.linspace(-(1 - 1 / length), 1 - 1 / length, length, dtype=dtype)
    return 1 - np.abs(x)


def finish(state: RequestState, logits: Sequence[Array], spec: InferenceSpec) -> list[NDArray[np.float32]]:
    """Reduce in window order, round as the extractor does, and retain all positions."""
    probabilities: list[NDArray[np.float32]] = [np.zeros(n, dtype="<f4") for n in state.original_lengths]
    if spec.kind == "subword":
        stitched = stitch_subword_logits(
            logits,
            cast(NDArray[np.int32], state.locations),
            state.lengths,
            block_size=state.block_size,
            num_labels=spec.num_labels,
            weighting=spec.weighting,
            strip_cls_sep=True,
            batch_size=spec.batch_size,
        )
        for local, original in enumerate(state.nonempty_indices):
            probability = project_subword_proba(
                state.original_lengths[original],
                state.input_ids[local],
                stitched[local],
                state.offsets[local],
                spec.special_tokens,
            )
            if (
                len(probability) != state.original_lengths[original]
                or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))
            ):
                raise ValueError("Invalid character probability vector")
            probabilities[original] = probability
        return probabilities

    expected = math.ceil(len(state.locations) / spec.batch_size)
    if len(logits) != expected:
        raise ValueError("Wrong number of forward batches")
    accumulated = [np.zeros((n, spec.num_labels), dtype=np.float32) for n in state.lengths]
    counts: list[NDArray[np.float32]] = [np.zeros(n, dtype=np.float32) for n in state.lengths]
    cursor = 0
    for values in logits:
        real = min(spec.batch_size, len(state.locations) - cursor)
        if values.ndim != 3 or len(values) != real or values.shape[2] != spec.num_labels:
            raise ValueError("Forward logits do not match prepared batch geometry")
        for row in range(real):
            index, start, end = state.locations[cursor + row]
            n = int(end - start)
            if values.shape[1] < n:
                raise ValueError("Forward logits are shorter than their window")
            local_weights = _weights(n, spec.weighting, np.float32)
            accumulated[index][start:end] += values[row, :n] * local_weights[:, None]
            counts[index][start:end] += local_weights
        cursor += real
    for local, original in enumerate(state.nonempty_indices):
        covered = counts[local] > 0
        accumulated[local][covered] /= counts[local][covered, None]
        accumulated[local][~covered] = -12.0
        rounded = accumulated[local].astype(np.float16)
        with np.errstate(over="ignore"):
            probability = np.asarray(sigmoid(rounded[:, Constants.NEWLINE_INDEX]), dtype="<f4")
        if (
            len(probability) != state.original_lengths[original]
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
        ):
            raise ValueError("Invalid character probability vector")
        probabilities[original] = probability
    return probabilities
