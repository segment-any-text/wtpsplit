import os
from pathlib import Path

# to register models for AutoModel
import wtpsplit.models  # noqa


import math
import numpy as np

import torch
from transformers import AutoModelForTokenClassification
from transformers.utils.hub import cached_file
import skops.io as sio

from wtpsplit.extract import extract
from wtpsplit.utils import Constants, encode, indices_to_sentences


class ORTWrapper:
    def __init__(self, model, ort_session):
        self.model = model
        self.ort_session = ort_session

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    def __call__(self, input_ids, attention_mask, position_ids):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        logits = self.ort_session.run(
            ["logits"],
            {
                "attention_mask": attention_mask.detach().numpy(),
                "position_ids": position_ids.detach().numpy(),
                "inputs_embeds": inputs_embeds.detach().numpy(),
            },
        )[0]

        return {"logits": torch.from_numpy(logits).to(self.model.device)}


class WtP:
    def __init__(self, model_name_or_model, ort_providers=None, ort_kwargs=None, mixtures=None):
        self.model_name_or_model = model_name_or_model
        self.ort_providers = ort_providers
        self.ort_kwargs = ort_kwargs

        mixture_path = None

        if isinstance(model_name_or_model, (str, Path)):
            model_name_or_model = str(model_name_or_model)
            is_local = os.path.isdir(model_name_or_model)

            model = AutoModelForTokenClassification.from_pretrained(model_name_or_model)

            if is_local:
                model_path = Path(model_name_or_model)
                mixture_path = model_path / "mixture.skops"
                if not mixture_path.exists():
                    mixture_path = None
                onnx_path = model_path / "model.onnx"
                if not onnx_path.exists():
                    onnx_path = None
            else:
                mixture_path = cached_file(model_name_or_model, "mixture.skops")
                onnx_path = cached_file(model_name_or_model, "model.onnx")

            if ort_providers is not None:
                if onnx_path is None:
                    raise ValueError(
                        "Could not find an ONNX model in the model directory. Try `use_ort=False` to run with PyTorch."
                    )

                import onnxruntime as ort

                self.model = ORTWrapper(
                    model, ort.InferenceSession(onnx_path, providers=ort_providers, **(ort_kwargs or {}))
                )
            else:
                self.model = model
        else:
            if ort_providers is not None:
                raise ValueError("You can only use onnxruntime with a model directory, not a model object.")

            self.model = model_name_or_model

        if mixtures is not None:
            self.mixtures = mixtures
        elif mixture_path is not None:
            self.mixtures = sio.load(
                mixture_path,
                ["numpy.float32", "numpy.float64"],
            )
        else:
            self.mixtures = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict_proba(
        self,
        text_or_texts,
        lang_code: str = None,
        style: str = None,
        stride=64,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        do_paragraph_segmentation=False,
        verbose: bool = True,
    ):
        texts = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts

        if style is not None:
            if lang_code is None:
                raise ValueError("Please specify a `lang_code` when passing a `style` to adapt to.")

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

            for text in input_texts:
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
                [encode(text) for text in outer_batch_texts],
                self.model,
                lang_code=lang_code,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                verbose=verbose,
            )

            def newline_probability_fn(logits):
                return torch.sigmoid(logits.float()[:, Constants.NEWLINE_INDEX]).numpy()

            for i, (text, logits) in enumerate(zip(outer_batch_texts, outer_batch_logits)):
                if style is not None:
                    sentence_probs = clf.predict_proba(logits)[:, 1]
                    newline_probs = newline_probability_fn(logits)
                else:
                    sentence_probs = newline_probs = newline_probability_fn(logits)

                if remove_whitespace_before_inference:
                    newline_probs, sentence_probs = list(newline_probs), list(sentence_probs)

                    for i in space_positions:
                        newline_probs.insert(i, np.zeros_like(newline_probs[0]))
                        sentence_probs.insert(i, np.zeros_like(sentence_probs[0]))

                    newline_probs = np.array(newline_probs)
                    sentence_probs = np.array(sentence_probs)

                if do_paragraph_segmentation:
                    yield sentence_probs, newline_probs
                else:
                    yield sentence_probs

    def segment(
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
        do_paragraph_segmentation=False,
        verbose: bool = True,
    ):
        texts = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts

        if style is not None:
            if lang_code is None:
                raise ValueError("Please specify a `lang_code` when passing a `style` to adapt to.")

            try:
                _, _, sentence_threshold, _ = self.mixtures[lang_code][style]
            except KeyError:
                raise ValueError(f"Could not find a mixture for the style '{style}'.")
        else:
            # the established default for newline prob threshold is 0.01
            sentence_threshold = threshold if threshold is not None else 0.01

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
                do_paragraph_segmentation=do_paragraph_segmentation,
                verbose=verbose,
            ),
        ):
            if do_paragraph_segmentation:
                sentence_probs, newline_probs = probs

                offset = 0

                paragraphs = []

                # TODO: indices_to_sentences should not be in evaluation module?
                for paragraph in indices_to_sentences(text, np.where(newline_probs > paragraph_threshold)[0]):
                    sentences = []

                    for sentence in indices_to_sentences(
                        paragraph, np.where(sentence_probs[offset : offset + len(paragraph)] > sentence_threshold)[0]
                    ):
                        sentences.append(sentence)

                    paragraphs.append(sentences)
                    offset += len(paragraph)

                yield paragraphs
            else:
                sentences = indices_to_sentences(text, np.where(probs > sentence_threshold)[0])
                yield sentences
