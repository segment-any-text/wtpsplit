"""Explicit inference sessions over SaT's existing model and extractors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from wtpsplit._inference import normalize_input_line_breaks
from wtpsplit._inference_pipeline import InferenceTiming, StagedPool, results
from wtpsplit._inference_stages import Array, InferenceSpec, ModelBatch, Tokenizer, finish, prepare

if TYPE_CHECKING:
    from wtpsplit import SaT
    from wtpsplit._inference_transfer import PinnedInputs


@dataclass(frozen=True)
class InferenceSettings:
    """Numerical options plus independent, bounded CPU staging controls."""

    stride: int = 64
    block_size: int = 512
    batch_size: int = 32
    weighting: Literal["uniform", "hat"] = "uniform"
    input_line_breaks: Literal["preserve", "space"] = "preserve"
    workers: int = 0
    ahead: int = 6
    max_unfinished: int = 8
    transport_root: str | None = None
    pinned_inputs: bool = True

    def __post_init__(self) -> None:
        for name in ("stride", "block_size", "batch_size", "ahead", "max_unfinished"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if self.stride > self.block_size:
            raise ValueError("stride exceeds block_size")
        if self.weighting not in ("uniform", "hat") or self.input_line_breaks not in ("preserve", "space"):
            raise ValueError("Invalid weighting or input_line_breaks")
        if type(self.pinned_inputs) is not bool:
            raise ValueError("pinned_inputs must be boolean")


class Forward(Protocol):
    def __call__(self, **kwargs: Array) -> dict[str, Array]: ...


def inference_spec(model: SaT, settings: InferenceSettings) -> InferenceSpec | None:
    """Resolve staged geometry; unsupported models keep the library extractor."""
    from wtpsplit.extract import CHAR_MODEL_TYPES, max_content_block_size

    config = model.model.config
    if config.model_type in CHAR_MODEL_TYPES:
        return None
    if getattr(config, "language_adapter", "off") == "on":
        return None
    if getattr(config, "use_character_head", False):
        if getattr(model, "character_preprocessing", "reference") != "exact" or not model.trim_padding:
            return None
        kind: Literal["char", "subword"] = "char"
    else:
        kind = "subword"
    tokenizer = model.tokenizer
    return InferenceSpec(
        kind,
        int(config.num_labels),
        max_content_block_size(config),
        int(getattr(config, "downsampling_rate", 1)),
        settings.stride,
        settings.block_size,
        settings.batch_size,
        int(tokenizer.cls_token_id),
        int(tokenizer.sep_token_id),
        int(tokenizer.pad_token_id),
        tuple(model.special_tokens),
        settings.weighting,
    )


class InferenceSession:
    """Reuse a model and optional CPU workers across requests.

    Forward runs on the caller. Concurrent use of one model/session is unsupported.
    Closing a session releases CPU workers and temporary input buffers. It never
    changes model precision, attention lookahead, model defaults or global state.
    """

    def __init__(self, model: SaT, settings: InferenceSettings | None = None) -> None:
        from wtpsplit.extract import PyTorchWrapper

        self.model = model
        self.settings = settings or InferenceSettings(input_line_breaks=model.input_line_breaks)
        if model.input_line_breaks == "space" and self.settings.input_line_breaks == "preserve":
            raise ValueError("Cannot preserve line breaks with a model configured to replace them")
        self.spec = inference_spec(model, self.settings)
        self._pinned_inputs: PinnedInputs | None = None
        wrapper = model.model
        if type(wrapper) is PyTorchWrapper and self.settings.pinned_inputs:
            from wtpsplit._inference_transfer import PinnedInputs

            self._pinned_inputs = PinnedInputs()
            wrapper = cast(Callable[..., PyTorchWrapper], PyTorchWrapper)(
                wrapper.model,
                autocast_device=wrapper.autocast_device,
                autocast_dtype=wrapper.autocast_dtype,
                autocast_enabled=wrapper.autocast_enabled,
                output_float32=wrapper.output_float32,
                pinned_inputs=self._pinned_inputs,
            )
        self._forward = cast(Forward, wrapper)
        self._pool = (
            StagedPool(
                cast(Tokenizer, model.tokenizer),
                self.spec,
                self.settings.workers,
                transport_root=self.settings.transport_root,
            )
            if self.spec is not None and self.settings.workers
            else None
        )
        self._closed = False
        self._last_tokens: int | None = 0

    def _forward_batches(self, batches: list[ModelBatch]) -> list[Array]:
        self._last_tokens = sum(batch["input_ids"].size for batch in batches)
        return [self._forward(**batch)["logits"] for batch in batches]

    def _fallback(self, texts: list[str]) -> list[NDArray[np.float32]]:
        settings = self.settings
        if self.model.input_line_breaks == "space" and settings.input_line_breaks == "preserve":
            raise ValueError("Cannot preserve line breaks with a model configured to replace them")
        self._last_tokens = None  # The fallback owns its forwarding; do not invent token counts.
        active = [text for text in texts if text.strip()]
        values = iter(
            self.model.predict_proba(
                active,
                stride=settings.stride,
                block_size=settings.block_size,
                batch_size=settings.batch_size,
                weighting=settings.weighting,
                outer_batch_size=max(len(active), 1),
            )
        )
        return [
            np.asarray(next(values), dtype="<f4") if text.strip() else np.zeros(len(text), dtype="<f4")
            for text in texts
        ]

    def _normalize(self, texts: list[str]) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("Expected a list of texts; wrap a single string in a list")
        return [normalize_input_line_breaks(text, self.settings.input_line_breaks) for text in texts]

    def probabilities(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Return float32 values for every character, including empty inputs."""
        if self._closed:
            raise RuntimeError("Inference session is closed")
        self._last_tokens = 0
        effective = self._normalize(texts)
        if self.spec is None:
            return self._fallback(effective)
        prepared = prepare(effective, self.spec, cast(Tokenizer, self.model.tokenizer))
        if prepared is None:
            return self._fallback(effective)
        return finish(prepared.state, self._forward_batches(prepared.batches), self.spec)

    def batches(self, requests: Iterable[list[str]]) -> Iterator[tuple[list[NDArray[np.float32]], InferenceTiming]]:
        """Stream ordered batches; workers overlap preparation, forward and finish."""
        import time

        if self._closed:
            raise RuntimeError("Inference session is closed")
        if self._pool is None:
            for texts in requests:
                start = time.monotonic()
                values = self.probabilities(texts)
                seconds = time.monotonic() - start
                if self._closed:
                    raise RuntimeError("Inference session is closed")
                yield values, InferenceTiming(seconds, 0.0, 0.0, seconds, 0.0, self._last_tokens, self.spec is None)
            return
        if isinstance(requests, (str, bytes)):
            raise TypeError("Expected requests containing lists of texts, not a string")
        normalized = ((None, self._normalize(texts)) for texts in requests)
        for _, values, timing in results(
            normalized,
            self._pool,
            self._forward_batches,
            self._fallback,
            ahead=self.settings.ahead,
            max_unfinished=self.settings.max_unfinished,
        ):
            if self._closed:
                raise RuntimeError("Inference session is closed")
            yield values, timing

    def close(self) -> None:
        try:
            if self._pool is not None:
                self._pool.close()
        finally:
            self._closed = True
            if self._pinned_inputs is not None:
                self._pinned_inputs.clear()

    def __enter__(self) -> InferenceSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
