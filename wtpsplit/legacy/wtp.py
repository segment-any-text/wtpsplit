from __future__ import annotations

import importlib
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Literal

from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.utils.hub import cached_file

from wtpsplit._inference import (
    decode_probabilities,
    iter_probabilities,
    resolve_text_input,
    validate_segmentation_options,
)
from wtpsplit.constants import DEFAULT_STRIDE
from wtpsplit.extract import BertCharORTWrapper, PyTorchWrapper, extract
from wtpsplit.model_registry import register_legacy_configs, register_legacy_models
from wtpsplit.utils import Constants, sigmoid


def _load_skops_io():
    """Import skops without traversing transformers' optional lazy modules."""
    if "skops.io" in sys.modules:
        return sys.modules["skops.io"]

    transformers_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "transformers" or name.startswith("transformers.")
    }
    for name in transformers_modules:
        del sys.modules[name]
    sys.modules["transformers"] = types.ModuleType("transformers")
    try:
        return importlib.import_module("skops.io")
    finally:
        del sys.modules["transformers"]
        sys.modules.update(transformers_modules)


def _resolve_deprecated_alias(value, alias, old_name: str, new_name: str):
    if alias is None:
        return value
    if value is not None:
        raise TypeError(f"Pass only one of `{new_name}` or the deprecated `{old_name}` alias.")
    warnings.warn(
        f"`{old_name}` is deprecated; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return alias


class WtP:
    def __init__(
        self,
        model_name_or_model,
        from_pretrained_kwargs=None,
        ort_providers=None,
        ort_kwargs=None,
        mixtures=None,
        hub_prefix="benjamin",
        ignore_legacy_warning=False,
        language: str = None,
    ):
        self.model_name_or_model = model_name_or_model
        self.ort_providers = ort_providers
        self.ort_kwargs = ort_kwargs
        self.language = language  # Store for language-aware prior defaults

        mixture_path = None

        if not ignore_legacy_warning:
            # WtP is deprecated!
            warnings.warn(
                "You are using WtP, the old sentence segmentation model. "
                "It is highly encouraged to use SaT instead due to strongly improved performance and efficiency. "
                "See https://github.com/segment-any-text/wtpsplit for more info. "
                "To ignore this warning, set ignore_legacy_warning=True.",
                DeprecationWarning,
            )

        if isinstance(model_name_or_model, (str, Path)):
            model_name = str(model_name_or_model)
            is_local = os.path.isdir(model_name)

            if not is_local and hub_prefix is not None:
                model_name_to_fetch = f"{hub_prefix}/{model_name}"
            else:
                model_name_to_fetch = model_name

            if is_local:
                model_path = Path(model_name)
                mixture_path = model_path / "mixtures.skops"
                if not mixture_path.exists():
                    mixture_path = None
                onnx_path = model_path / "model.onnx"
                if not onnx_path.exists():
                    onnx_path = None
            else:
                try:
                    mixture_path = cached_file(model_name_to_fetch, "mixtures.skops", **(from_pretrained_kwargs or {}))
                except OSError:
                    mixture_path = None

                # no need to load if no ort_providers set
                if ort_providers is not None:
                    onnx_path = cached_file(model_name_to_fetch, "model.onnx", **(from_pretrained_kwargs or {}))
                else:
                    onnx_path = None

            if ort_providers is not None:
                if onnx_path is None:
                    raise ValueError(
                        "Could not find an ONNX model in the model directory. Try `use_ort=False` to run with PyTorch."
                    )

                try:
                    import onnxruntime as ort  # noqa
                except ModuleNotFoundError:
                    raise ValueError("Please install `onnxruntime` to use WtP with an ONNX model.")

                register_legacy_configs()

                self.model = BertCharORTWrapper(
                    AutoConfig.from_pretrained(model_name_to_fetch, **(from_pretrained_kwargs or {})),
                    ort.InferenceSession(str(onnx_path), providers=ort_providers, **(ort_kwargs or {})),
                )
            else:
                # to register models for AutoConfig
                try:
                    import torch  # noqa
                except ModuleNotFoundError:
                    raise ValueError("Please install `torch` to use WtP with a PyTorch model.")

                register_legacy_models()

                self.model = PyTorchWrapper(
                    AutoModelForTokenClassification.from_pretrained(
                        model_name_to_fetch, **(from_pretrained_kwargs or {})
                    )
                )
        else:
            if ort_providers is not None:
                raise ValueError("You can only use onnxruntime with a model directory, not a model object.")

            self.model = model_name_or_model

        if mixtures is not None:
            self.mixtures = mixtures
        elif mixture_path is not None:
            sio = _load_skops_io()
            self.mixtures = sio.load(
                mixture_path,
                ["numpy.float32", "numpy.float64", "sklearn.linear_model._logistic.LogisticRegression"],
            )
        else:
            self.mixtures = None

    def __getattr__(self, name):
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def predict_proba(
        self,
        text_or_texts,
        language: str = None,
        domain: str = None,
        stride=DEFAULT_STRIDE,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        return_paragraph_probabilities=False,
        verbose: bool = False,
        *,
        lang_code: str = None,
        style: str = None,
        lazy: bool = False,
    ):
        language = _resolve_deprecated_alias(language, lang_code, "lang_code", "language")
        domain = _resolve_deprecated_alias(domain, style, "style", "domain")
        return resolve_text_input(
            text_or_texts,
            lambda texts: self._predict_proba(
                texts,
                lang_code=language,
                style=domain,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=return_paragraph_probabilities,
                verbose=verbose,
            ),
            lazy=lazy,
        )

    def _predict_proba(
        self,
        texts,
        lang_code: str,
        style: str,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        return_paragraph_probabilities: bool,
        verbose: bool,
    ):
        mixture = self._resolve_mixture(lang_code, style)
        clf = mixture[0] if mixture is not None else None

        def extract_logits(input_texts):
            return extract(
                input_texts,
                self.model,
                lang_code=lang_code,
                stride=stride,
                max_block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                verbose=verbose,
            )[0]

        def probability_fn(logits):
            newline_probs = sigmoid(logits[:, Constants.NEWLINE_INDEX])
            sentence_probs = clf.predict_proba(logits)[:, 1] if clf is not None else newline_probs
            return sentence_probs, newline_probs

        yield from iter_probabilities(
            texts,
            outer_batch_size=outer_batch_size,
            remove_whitespace_before_inference=remove_whitespace_before_inference,
            extract_logits=extract_logits,
            probability_fn=probability_fn,
            return_paragraph_probabilities=return_paragraph_probabilities,
        )

    def split(
        self,
        text_or_texts,
        language: str = None,
        domain: str = None,
        threshold: float = None,  # ignored when max_length is set
        stride=DEFAULT_STRIDE,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        paragraph_threshold: float = 0.5,
        strip_whitespace: bool = False,
        do_paragraph_segmentation=False,
        verbose: bool = False,
        min_length: int = 1,
        max_length: int = None,  # when set, segments may contain newlines; use ''.join(segments)
        prior_type: str = "uniform",
        prior_kwargs: dict = None,
        algorithm: str = "viterbi",
        *,
        lang_code: str = None,
        style: str = None,
        lazy: bool = False,
    ):
        language = _resolve_deprecated_alias(language, lang_code, "lang_code", "language")
        domain = _resolve_deprecated_alias(domain, style, "style", "domain")
        validate_segmentation_options(
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
            prior_type=prior_type,
            algorithm=algorithm,
            split_on_input_newlines=None,
        )

        return resolve_text_input(
            text_or_texts,
            lambda texts: self._split(
                texts,
                lang_code=language,
                style=domain,
                threshold=threshold,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                verbose=verbose,
                min_length=min_length,
                max_length=max_length,
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                algorithm=algorithm,
            ),
            lazy=lazy,
        )

    def get_threshold(
        self,
        language: str = None,
        domain: str = None,
        return_punctuation_threshold: bool = False,
        *,
        lang_code: str = None,
        style: str = None,
    ):
        language = _resolve_deprecated_alias(language, lang_code, "lang_code", "language")
        domain = _resolve_deprecated_alias(domain, style, "style", "domain")
        if language is None or domain is None:
            raise TypeError("`language` and `domain` are required.")
        try:
            _, _, punctuation_threshold, threshold = self.mixtures[language][domain]
        except KeyError:
            raise ValueError(f"Could not find a mixture for domain '{domain}' and language '{language}'.")

        if return_punctuation_threshold:
            return punctuation_threshold

        return threshold

    def _resolve_mixture(self, language: str | None, domain: str | None):
        if domain is None:
            return None
        if language is None:
            raise ValueError("Please specify a `lang_code` when passing a `style` to adapt to.")
        if self.mixtures is None:
            raise ValueError(
                "This model does not have any associated mixtures. Maybe they are missing from the model directory?"
            )
        try:
            return self.mixtures[language][domain]
        except KeyError:
            raise ValueError(f"Could not find a mixture for the style '{domain}'.")

    def _split(
        self,
        texts,
        lang_code: str | None,
        style: str | None,
        threshold: float | None,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        paragraph_threshold: float,
        do_paragraph_segmentation: bool,
        strip_whitespace: bool,
        verbose: bool,
        min_length: int,
        max_length: int | None,
        prior_type: str,
        prior_kwargs: dict | None,
        algorithm: str,
    ):
        mixture = self._resolve_mixture(lang_code, style)
        # The established default for newline probabilities is 0.01.
        default_threshold = mixture[2] if mixture is not None else 0.01

        sentence_threshold = threshold if threshold is not None else default_threshold

        for text, probs in zip(
            texts,
            self.predict_proba(
                texts,
                language=lang_code,
                domain=style,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=do_paragraph_segmentation,
                verbose=verbose,
                lazy=True,
            ),
        ):
            decoded = decode_probabilities(
                text,
                probs,
                sentence_threshold=sentence_threshold,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                split_on_input_newlines=None,
                min_length=min_length,
                max_length=max_length,
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                algorithm=algorithm,
                language=self.language,
            )
            yield decoded.paragraphs if decoded.paragraphs is not None else decoded.sentences
