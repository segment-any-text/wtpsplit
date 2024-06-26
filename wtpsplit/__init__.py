import contextlib
import math
import os
import warnings
from pathlib import Path

# avoid the "None of PyTorch, TensorFlow, etc. have been found" warning.
with contextlib.redirect_stderr(open(os.devnull, "w")):
    import transformers  # noqa

import adapters  # noqa
import numpy as np
import skops.io as sio
from adapters.models import MODEL_MIXIN_MAPPING
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.utils.hub import cached_file

from wtpsplit.evaluation import token_to_char_probs
from wtpsplit.extract import BertCharORTWrapper, PyTorchWrapper, extract
from wtpsplit.utils import Constants, indices_to_sentences, sigmoid

__version__ = "2.0.3"

warnings.simplefilter("default", DeprecationWarning)  # show by default


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
    ):
        self.model_name_or_model = model_name_or_model
        self.ort_providers = ort_providers
        self.ort_kwargs = ort_kwargs

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

                    ort.set_default_logger_severity(0)
                except ModuleNotFoundError:
                    raise ValueError("Please install `onnxruntime` to use WtP with an ONNX model.")

                # to register models for AutoConfig
                import wtpsplit.configs  # noqa

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

                import wtpsplit.models  # noqa

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
        lang_code: str = None,
        style: str = None,
        stride=256,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        return_paragraph_probabilities=False,
        verbose: bool = False,
    ):
        if isinstance(text_or_texts, str):
            return next(
                self._predict_proba(
                    [text_or_texts],
                    lang_code=lang_code,
                    style=style,
                    stride=stride,
                    block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    remove_whitespace_before_inference=remove_whitespace_before_inference,
                    outer_batch_size=outer_batch_size,
                    return_paragraph_probabilities=return_paragraph_probabilities,
                    verbose=verbose,
                )
            )
        else:
            return self._predict_proba(
                text_or_texts,
                lang_code=lang_code,
                style=style,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=return_paragraph_probabilities,
                verbose=verbose,
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
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        return_paragraph_probabilities: bool,
        verbose: bool,
    ):
        if style is not None:
            if lang_code is None:
                raise ValueError("Please specify a `lang_code` when passing a `style` to adapt to.")

            if self.mixtures is None:
                raise ValueError(
                    "This model does not have any associated mixtures. Maybe they are missing from the model directory?"
                )

            try:
                clf, _, _, _ = self.mixtures[lang_code][style]
            except KeyError:
                raise ValueError(f"Could not find a mixture for the style '{style}'.")
        else:
            clf = None

        n_outer_batches = math.ceil(len(texts) / outer_batch_size)

        for outer_batch_idx in range(n_outer_batches):
            start, end = outer_batch_idx * outer_batch_size, min((outer_batch_idx + 1) * outer_batch_size, len(texts))

            outer_batch_texts = texts[start:end]
            input_texts = []
            space_positions = []

            for text in outer_batch_texts:
                if remove_whitespace_before_inference:
                    text_space_positions = []
                    input_text = ""

                    for c in text:
                        if c == " ":
                            text_space_positions.append(len(input_text) + len(text_space_positions))
                        else:
                            input_text += c

                    space_positions.append(text_space_positions)
                else:
                    input_text = text

                input_texts.append(input_text)

            outer_batch_logits = extract(
                input_texts,
                self.model,
                lang_code=lang_code,
                stride=stride,
                max_block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                verbose=verbose,
            )[0]

            def newline_probability_fn(logits):
                return sigmoid(logits[:, Constants.NEWLINE_INDEX])

            for i, (text, logits) in enumerate(zip(outer_batch_texts, outer_batch_logits)):
                if style is not None:
                    sentence_probs = clf.predict_proba(logits)[:, 1]
                    newline_probs = newline_probability_fn(logits)
                else:
                    sentence_probs = newline_probs = newline_probability_fn(logits)

                if remove_whitespace_before_inference:
                    full_newline_probs, full_sentence_probs = list(newline_probs), list(sentence_probs)

                    for j in space_positions[i]:
                        full_newline_probs.insert(j, np.zeros_like(newline_probs[0]))
                        full_sentence_probs.insert(j, np.zeros_like(sentence_probs[0]))

                    newline_probs = np.array(full_newline_probs)
                    sentence_probs = np.array(full_sentence_probs)

                if return_paragraph_probabilities:
                    yield sentence_probs, newline_probs
                else:
                    yield sentence_probs

    def split(
        self,
        text_or_texts,
        lang_code: str = None,
        style: str = None,
        threshold: float = None,
        stride=64,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        paragraph_threshold: float = 0.5,
        strip_whitespace: bool = False,
        do_paragraph_segmentation=False,
        verbose: bool = False,
    ):
        if isinstance(text_or_texts, str):
            return next(
                self._split(
                    [text_or_texts],
                    lang_code=lang_code,
                    style=style,
                    threshold=threshold,
                    stride=stride,
                    block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    remove_whitespace_before_inference=remove_whitespace_before_inference,
                    outer_batch_size=outer_batch_size,
                    paragraph_threshold=paragraph_threshold,
                    strip_whitespace=strip_whitespace,
                    do_paragraph_segmentation=do_paragraph_segmentation,
                    verbose=verbose,
                )
            )
        else:
            return self._split(
                text_or_texts,
                lang_code=lang_code,
                style=style,
                threshold=threshold,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                verbose=verbose,
            )

    def get_threshold(self, lang_code: str, style: str, return_punctuation_threshold: bool = False):
        try:
            _, _, punctuation_threshold, threshold = self.mixtures[lang_code][style]
        except KeyError:
            raise ValueError(f"Could not find a mixture for the style '{style}' and language '{lang_code}'.")

        if return_punctuation_threshold:
            return punctuation_threshold

        return threshold

    def _split(
        self,
        texts,
        lang_code: str,
        style: str,
        threshold: float,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        paragraph_threshold: float,
        do_paragraph_segmentation: bool,
        strip_whitespace: bool,
        verbose: bool,
    ):
        if style is not None:
            if lang_code is None:
                raise ValueError("Please specify a `lang_code` when passing a `style` to adapt to.")

            if self.mixtures is None:
                raise ValueError(
                    "This model does not have any associated mixtures. Maybe they are missing from the model directory?"
                )

            try:
                _, _, default_threshold, _ = self.mixtures[lang_code][style]
            except KeyError:
                raise ValueError(f"Could not find a mixture for the style '{style}'.")
        else:
            # the established default for newline prob threshold is 0.01
            default_threshold = 0.01

        sentence_threshold = threshold if threshold is not None else default_threshold

        for text, probs in zip(
            texts,
            self.predict_proba(
                texts,
                lang_code=lang_code,
                style=style,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=do_paragraph_segmentation,
                verbose=verbose,
            ),
        ):
            if do_paragraph_segmentation:
                sentence_probs, newline_probs = probs

                offset = 0

                paragraphs = []

                for paragraph in indices_to_sentences(text, np.where(newline_probs > paragraph_threshold)[0]):
                    sentences = []

                    for sentence in indices_to_sentences(
                        paragraph,
                        np.where(
                            sentence_probs[offset : offset + len(paragraph)] > sentence_threshold,
                        )[0],
                        strip_whitespace=strip_whitespace,
                    ):
                        sentences.append(sentence)

                    paragraphs.append(sentences)
                    offset += len(paragraph)

                yield paragraphs
            else:
                sentences = indices_to_sentences(
                    text, np.where(probs > sentence_threshold)[0], strip_whitespace=strip_whitespace
                )
                yield sentences


class SaT:
    def __init__(
        self,
        model_name_or_model,
        from_pretrained_kwargs=None,
        ort_providers=None,
        ort_kwargs=None,
        style_or_domain: str = None,
        language: str = None,
        lora_path: str = None,  # local
        hub_prefix="segment-any-text",
    ):
        self.model_name_or_model = model_name_or_model
        self.ort_providers = ort_providers
        self.ort_kwargs = ort_kwargs

        self.use_lora = False

        if isinstance(model_name_or_model, (str, Path)):
            model_name = str(model_name_or_model)
            is_local = os.path.isdir(model_name)

            if not is_local and hub_prefix is not None:
                model_name_to_fetch = f"{hub_prefix}/{model_name}"
            else:
                model_name_to_fetch = model_name

            if is_local:
                model_path = Path(model_name)
                onnx_path = model_path / "model.onnx"
                if not onnx_path.exists():
                    onnx_path = None
            else:
                # no need to load if no ort_providers set
                if ort_providers is not None:
                    onnx_path = cached_file(model_name_to_fetch, "model.onnx", **(from_pretrained_kwargs or {}))
                else:
                    onnx_path = None

            if ort_providers is not None:
                raise NotImplementedError("ONNX is not supported for SaT *yet*.")
                # if onnx_path is None:
                #     raise ValueError(
                #         "Could not find an ONNX model in the model directory. Try `use_ort=False` to run with PyTorch."
                #     )

                # try:
                #     import onnxruntime as ort  # noqa

                #     ort.set_default_logger_severity(0)
                # except ModuleNotFoundError:
                #     raise ValueError("Please install `onnxruntime` to use WtP with an ONNX model.")

                # # to register models for AutoConfig
                # import wtpsplit.configs  # noqa

                # # TODO: ONNX integration
                # self.model = SaTORTWrapper(
                #     AutoConfig.from_pretrained(model_name_to_fetch, **(from_pretrained_kwargs or {})),
                #     ort.InferenceSession(str(onnx_path), providers=ort_providers, **(ort_kwargs or {})),
                # )
            else:
                # to register models for AutoConfig
                try:
                    import torch  # noqa
                except ModuleNotFoundError:
                    raise ValueError("Please install `torch` to use WtP with a PyTorch model.")

                import wtpsplit.models  # noqa

                self.model = PyTorchWrapper(
                    AutoModelForTokenClassification.from_pretrained(
                        model_name_to_fetch, **(from_pretrained_kwargs or {})
                    )
                )
            # LoRA LOADING
            # TODO: LoRA + ONNX ?
            if (style_or_domain and not language) or (language and not style_or_domain):
                raise ValueError("Please specify both language and style_or_domain!")
            if style_or_domain and language:
                # monkey patch mixin to avoid forking whole adapters library
                MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin
                model_type = self.model.model.config.model_type
                # adapters need xlm-roberta as model type.
                self.model.model.config.model_type = "xlm-roberta"
                adapters.init(self.model.model)
                # reset model type (used later)
                self.model.model.config.model_type = model_type
                try:
                    if not lora_path:
                        for file in [
                            "adapter_config.json",
                            "head_config.json",
                            "pytorch_adapter.bin",
                            "pytorch_model_head.bin",
                        ]:
                            hf_hub_download(
                                repo_id=model_name_to_fetch,
                                subfolder=f"loras/{style_or_domain}/{language}",
                                filename=file,
                                local_dir=Constants.CACHE_DIR,
                            )
                        lora_load_path = str(Constants.CACHE_DIR / "loras" / style_or_domain / language)
                    else:
                        lora_load_path = lora_path

                    self.model.model.load_adapter(
                        lora_load_path,
                        set_active=True,
                        with_head=True,
                        load_as="sat-lora",
                    )
                    # merge lora weights into transformer for 0 efficiency overhead
                    self.model.model.merge_adapter("sat-lora")
                    self.use_lora = True
                except:  # noqa
                    if lora_path:
                        print(f"LoRA at {lora_path} not found, using base model...")
                    else:
                        print(f"LoRA {style_or_domain}/{language} not found, using base model...")
        else:
            if ort_providers is not None:
                raise ValueError("You can only use onnxruntime with a model directory, not a model object.")

            self.model = model_name_or_model

    def __getattr__(self, name):
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def predict_proba(
        self,
        text_or_texts,
        stride=256,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        return_paragraph_probabilities=False,
        verbose: bool = False,
    ):
        if isinstance(text_or_texts, str):
            return next(
                self._predict_proba(
                    [text_or_texts],
                    stride=stride,
                    block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    remove_whitespace_before_inference=remove_whitespace_before_inference,
                    outer_batch_size=outer_batch_size,
                    return_paragraph_probabilities=return_paragraph_probabilities,
                    verbose=verbose,
                )
            )
        else:
            return self._predict_proba(
                text_or_texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=return_paragraph_probabilities,
                verbose=verbose,
            )

    def _predict_proba(
        self,
        texts,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        return_paragraph_probabilities: bool,
        verbose: bool,
    ):
        def newline_probability_fn(logits):
            return sigmoid(logits[:, Constants.NEWLINE_INDEX])

        n_outer_batches = math.ceil(len(texts) / outer_batch_size)

        for outer_batch_idx in range(n_outer_batches):
            start, end = outer_batch_idx * outer_batch_size, min((outer_batch_idx + 1) * outer_batch_size, len(texts))

            outer_batch_texts = texts[start:end]
            input_texts = []
            space_positions = []

            for text in outer_batch_texts:
                if remove_whitespace_before_inference:
                    text_space_positions = []
                    input_text = ""

                    for c in text:
                        if c == " ":
                            text_space_positions.append(len(input_text) + len(text_space_positions))
                        else:
                            input_text += c

                    space_positions.append(text_space_positions)
                else:
                    input_text = text

                input_texts.append(input_text)

            outer_batch_logits, _, tokenizer, tokenizer_output = extract(
                input_texts,
                self.model,
                stride=stride,
                max_block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                verbose=verbose,
            )

            # convert token probabilities to character probabilities for the entire array
            outer_batch_logits = [
                token_to_char_probs(
                    input_texts[i],
                    tokenizer_output["input_ids"][i],
                    outer_batch_logits[i],
                    tokenizer,
                    tokenizer_output["offset_mapping"][i],
                )
                for i in range(len(input_texts))
            ]

            for i, (text, logits) in enumerate(zip(outer_batch_texts, outer_batch_logits)):
                sentence_probs = newline_probs = newline_probability_fn(logits)

                if remove_whitespace_before_inference:
                    full_newline_probs, full_sentence_probs = list(newline_probs), list(sentence_probs)

                    for j in space_positions[i]:
                        full_newline_probs.insert(j, np.zeros_like(newline_probs[0]))
                        full_sentence_probs.insert(j, np.zeros_like(sentence_probs[0]))

                    newline_probs = np.array(full_newline_probs)
                    sentence_probs = np.array(full_sentence_probs)

                if return_paragraph_probabilities:
                    yield sentence_probs, newline_probs
                else:
                    yield sentence_probs

    def split(
        self,
        text_or_texts,
        threshold: float = None,
        stride=64,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        paragraph_threshold: float = 0.5,
        strip_whitespace: bool = False,
        do_paragraph_segmentation=False,
        verbose: bool = False,
    ):
        if isinstance(text_or_texts, str):
            return next(
                self._split(
                    [text_or_texts],
                    threshold=threshold,
                    stride=stride,
                    block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    remove_whitespace_before_inference=remove_whitespace_before_inference,
                    outer_batch_size=outer_batch_size,
                    paragraph_threshold=paragraph_threshold,
                    strip_whitespace=strip_whitespace,
                    do_paragraph_segmentation=do_paragraph_segmentation,
                    verbose=verbose,
                )
            )
        else:
            return self._split(
                text_or_texts,
                threshold=threshold,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                verbose=verbose,
            )

    def _split(
        self,
        texts,
        threshold: float,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        paragraph_threshold: float,
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        do_paragraph_segmentation: bool,
        strip_whitespace: bool,
        verbose: bool,
    ):
        def get_default_threshold(model_str: str):
            if "sm" in model_str:
                return 0.25
            if self.use_lora:
                return 0.5
            if "no-limited-lookahead" in model_str and "sm" not in model_str:
                return 0.01
            return 0.025

        default_threshold = get_default_threshold(self.model_name_or_model)
        sentence_threshold = threshold if threshold is not None else default_threshold

        for text, probs in zip(
            texts,
            self.predict_proba(
                texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=do_paragraph_segmentation,
                verbose=verbose,
            ),
        ):
            if do_paragraph_segmentation:
                sentence_probs, newline_probs = probs

                offset = 0
                paragraphs = []

                for paragraph in indices_to_sentences(text, np.where(newline_probs > paragraph_threshold)[0]):
                    sentences = []

                    for sentence in indices_to_sentences(
                        paragraph,
                        np.where(
                            sentence_probs[offset : offset + len(paragraph)] > sentence_threshold,
                        )[0],
                        strip_whitespace=strip_whitespace,
                    ):
                        sentences.append(sentence)

                    paragraphs.append(sentences)
                    offset += len(paragraph)

                yield paragraphs
            else:
                sentences = indices_to_sentences(
                    text, np.where(probs > sentence_threshold)[0], strip_whitespace=strip_whitespace
                )
                yield sentences
