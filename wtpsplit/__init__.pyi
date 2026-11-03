from collections.abc import Iterable, Iterator
from os import PathLike
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from torch import device as TorchDevice

from wtpsplit.constants import DEFAULT_STRIDE as DEFAULT_STRIDE
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
    def split(self, text_or_texts: Any, **kwargs: Any) -> Any: ...

class SaT:
    precision: Literal['fp32', 'fp16', 'bf16'] | None
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
        character_preprocessing: Literal["reference", "exact"] = ...,
        trim_padding: bool = ...,
        quantization: Literal["int8", "combined_int8", "embedding_int8", "linear_int8"] | None = ...,
        precision: Literal['fp32', 'fp16', 'bf16'] | None = ...,
        input_line_breaks: Literal["preserve", "space"] | None = ...,
    ) -> None: ...
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
        min_length: int = ...,
        max_length: int | None = ...,
        prior_type: str = ...,
        prior_kwargs: dict[str, Any] | None = ...,
        algorithm: Literal["viterbi", "greedy"] = ...,
        use_negative_evidence: bool = ...,
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
        min_length: int = ...,
        max_length: int | None = ...,
        prior_type: str = ...,
        prior_kwargs: dict[str, Any] | None = ...,
        algorithm: Literal["viterbi", "greedy"] = ...,
        use_negative_evidence: bool = ...,
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
        min_length: int = ...,
        max_length: int | None = ...,
        prior_type: str = ...,
        prior_kwargs: dict[str, Any] | None = ...,
        algorithm: Literal["viterbi", "greedy"] = ...,
        use_negative_evidence: bool = ...,
        lazy: bool = ...,
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
        min_length: int = ...,
        max_length: int | None = ...,
        prior_type: str = ...,
        prior_kwargs: dict[str, Any] | None = ...,
        algorithm: Literal["viterbi", "greedy"] = ...,
        use_negative_evidence: bool = ...,
        lazy: bool = ...,
    ) -> list[SentenceResult] | Iterator[SentenceResult]: ...
