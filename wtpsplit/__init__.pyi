from collections.abc import Iterable, Iterator, Sequence
from os import PathLike
from typing import Any, Literal, overload
from typing_extensions import Unpack

import numpy as np
from numpy.typing import NDArray
from torch import device as TorchDevice

from wtpsplit._inference import WtpsplitBehaviorChangeWarning as WtpsplitBehaviorChangeWarning
from wtpsplit.chunking import LengthFunction as LengthFunction, TokenLength as TokenLength
from wtpsplit.chunking import UnsplittableUnitError as UnsplittableUnitError
from wtpsplit.chunking import split_spans as split_spans, chunk_spans as chunk_spans
from wtpsplit.constants import DEFAULT_STRIDE as DEFAULT_STRIDE
from wtpsplit.prepared import PreparedBatch as PreparedBatch, PreparedSplitOptions, Segments
from wtpsplit.segmentation import Segmentation as Segmentation

__version__: str
__all__: list[str]

ProbabilityArray = NDArray[np.floating[Any]]
ProbabilityResult = ProbabilityArray | tuple[ProbabilityArray, ProbabilityArray]
SentenceResult = list[str] | list[list[str]]

class WtP:
    def __init__(self, model_name_or_model: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def predict_proba(self, text_or_texts: Any, **kwargs: Any) -> Any: ...
    def split(
        self,
        text_or_texts: str | Iterable[str],
        language: str | None = ...,
        domain: str | None = ...,
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        *,
        lang_code: str | None = ...,
        style: str | None = ...,
        lazy: bool = ...,
    ) -> SentenceResult | list[SentenceResult] | Iterator[SentenceResult]: ...

class SaT:
    def prepare(
        self, texts: Sequence[str], *, document_ids: Sequence[str] | None = ...,
        remove_whitespace_before_inference: bool = ...,
    ) -> PreparedBatch: ...
    @overload
    def split_prepared(
        self, batch: PreparedBatch, *, return_segments: Literal[False] = ...,
        **split_options: Unpack[PreparedSplitOptions],
    ) -> list[tuple[int, ...]]: ...
    @overload
    def split_prepared(
        self, batch: PreparedBatch, *, return_segments: Literal[True],
        **split_options: Unpack[PreparedSplitOptions],
    ) -> Segments: ...
    @overload
    def split_prepared(
        self, batch: PreparedBatch, *, return_segments: bool,
        **split_options: Unpack[PreparedSplitOptions],
    ) -> list[tuple[int, ...]] | Segments: ...
    @overload
    def chunk(
        self,
        text: str,
        *,
        max_length: int,
        length_fn: LengthFunction = len,
        unit: Literal["sentence", "paragraph"] = "sentence",
        overlap_sentences: int = 0,
        return_spans: Literal[False] = False,
        threshold: float | None = None,
        paragraph_threshold: float = 0.5,
        **inference_options: Any,
    ) -> list[str]: ...
    @overload
    def chunk(
        self,
        text: str,
        *,
        max_length: int,
        length_fn: LengthFunction = len,
        unit: Literal["sentence", "paragraph"] = "sentence",
        overlap_sentences: int = 0,
        return_spans: Literal[True],
        threshold: float | None = None,
        paragraph_threshold: float = 0.5,
        **inference_options: Any,
    ) -> list[tuple[int, int]]: ...
    @overload
    def chunk(
        self,
        text: str,
        *,
        max_length: int,
        length_fn: LengthFunction = len,
        unit: Literal["sentence", "paragraph"] = "sentence",
        overlap_sentences: int = 0,
        return_spans: bool,
        threshold: float | None = None,
        paragraph_threshold: float = 0.5,
        **inference_options: Any,
    ) -> list[str] | list[tuple[int, int]]: ...

    mode: Literal["bitwise", "equivalent", "sparse"]
    trim_padding: bool
    trim_subword_padding: bool
    pack_sequences: bool
    character_preprocessing: Literal["reference", "exact"]
    precision: Literal["fp32", "fp16", "bf16", "fp8"] | None
    precision_info: dict[str, Any] | None
    quantization: Literal["int8", "combined_int8", "embedding_int8", "linear_int8"] | None
    quantization_info: dict[str, Any]
    input_line_breaks: Literal["preserve", "space"]
    def __init__(
        self,
        model_name_or_model: str | PathLike[str],
        tokenizer_name_or_path: str | PathLike[str] | None = ...,
        from_pretrained_kwargs: dict[str, Any] | None = ...,
        ort_providers: list[str] | None = ...,
        ort_kwargs: dict[str, Any] | None = ...,
        domain: str | None = ...,
        language: str | None = ...,
        lora_path: str | PathLike[str] | None = ...,
        hub_prefix: str | None = ...,
        merge_lora: bool = ...,
        device: str | TorchDevice | None = ...,
        compile: bool | dict[str, Any] = ...,
        *,
        style_or_domain: str | None = ...,
        mode: Literal["bitwise", "equivalent", "sparse"] = ...,
        character_preprocessing: Literal["reference", "exact"] = ...,
        trim_padding: bool = ...,
        trim_subword_padding: bool | None = ...,
        pack_sequences: bool | None = ...,
        fuse_depthwise_delta: bool | None = ...,
        project_tokens_first: bool | None = ...,
        gate_logit_threshold: float | None = ...,
        quantization: Literal["int8", "combined_int8", "embedding_int8", "linear_int8"] | None = ...,
        precision: Literal["fp32", "fp16", "bf16", "fp8"] | None = ...,
        input_line_breaks: Literal["preserve", "space"] | None = ...,
    ) -> None: ...
    def set_mode(
        self,
        mode: Literal["bitwise", "equivalent", "sparse"],
        *,
        trim_subword_padding: bool | None = ...,
        pack_sequences: bool | None = ...,
        fuse_depthwise_delta: bool | None = ...,
        project_tokens_first: bool | None = ...,
        gate_logit_threshold: float | None = ...,
    ) -> SaT: ...
    def __getattr__(self, name: str) -> Any: ...
    def get_threshold(self, language: str | None = ..., domain: str | None = ...) -> float: ...
    @overload
    def predict_proba(  # type: ignore[overload-overlap]
        self,
        text_or_texts: str,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        return_paragraph_probabilities: bool = ...,
        verbose: bool = ...,
        lazy: bool = ...,
    ) -> ProbabilityResult: ...
    @overload
    def predict_proba(
        self,
        text_or_texts: Iterable[str],
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        return_paragraph_probabilities: bool = ...,
        verbose: bool = ...,
        lazy: bool = ...,
    ) -> list[ProbabilityResult] | Iterator[ProbabilityResult]: ...
    @overload
    def segment(  # type: ignore[overload-overlap]
        self,
        text_or_texts: str,
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        split_on_input_newlines: bool = ...,
        treat_newline_as_space: bool | None = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        lazy: bool = ...,
    ) -> Segmentation: ...
    @overload
    def segment(
        self,
        text_or_texts: Iterable[str],
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        split_on_input_newlines: bool = ...,
        treat_newline_as_space: bool | None = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        lazy: bool = ...,
    ) -> list[Segmentation] | Iterator[Segmentation]: ...
    @overload
    def split(  # type: ignore[overload-overlap]
        self,
        text_or_texts: str,
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        split_on_input_newlines: bool = ...,
        treat_newline_as_space: bool | None = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        lazy: bool = ...,
        *,
        _return_offsets: Literal[True],
    ) -> list[int]: ...
    @overload
    def split(
        self,
        text_or_texts: Iterable[str],
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        split_on_input_newlines: bool = ...,
        treat_newline_as_space: bool | None = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        lazy: bool = ...,
        *,
        _return_offsets: Literal[True],
    ) -> list[list[int]] | Iterator[list[int]]: ...
    @overload
    def split(  # type: ignore[overload-overlap]
        self,
        text_or_texts: str,
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        split_on_input_newlines: bool = ...,
        treat_newline_as_space: bool | None = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        lazy: bool = ...,
        _return_offsets: Literal[False] = ...,
    ) -> SentenceResult: ...
    @overload
    def split(
        self,
        text_or_texts: Iterable[str],
        threshold: float | None = ...,
        stride: int = ...,
        block_size: int = ...,
        batch_size: int = ...,
        pad_last_batch: bool = ...,
        weighting: Literal["uniform", "hat"] = ...,
        remove_whitespace_before_inference: bool = ...,
        outer_batch_size: int = ...,
        paragraph_threshold: float = ...,
        strip_whitespace: bool = ...,
        do_paragraph_segmentation: bool = ...,
        split_on_input_newlines: bool = ...,
        treat_newline_as_space: bool | None = ...,
        verbose: bool = ...,
        max_length: int | None = ...,
        length_fn: LengthFunction = ...,
        lazy: bool = ...,
        _return_offsets: Literal[False] = ...,
    ) -> list[SentenceResult] | Iterator[SentenceResult]: ...
