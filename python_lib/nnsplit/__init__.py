__version__ = "0.2.1"
__all__ = ["NNSplit"]

import numpy as np
import torch
from .defaults import CUT_LENGTH, DEVICE
from .utils import load_provided_model, text_to_id
from .tokenizer import Tokenizer, Token


def _get_token(text):
    last_char_index = 0
    index = len(text) - 1

    while index >= 0:
        if not text[index].isspace():
            last_char_index = index + 1
            break

        index -= 1

    return Token(text[:last_char_index], text[last_char_index:])


class NNSplit(Tokenizer):
    """
    An NNSplit sentencizer and tokenizer.

    Parameters:
        model_or_model_name (str or torch.nn.Module or torch.jit.TracedModule):
            If str (either `de` or `en`) a pretrained model will be loaded.
            If torch.nn.Module or torch.jit.TracedModule, the given module will be used instead.
        threshold (float, 0 < x < 1): Cutoff above which predictions will be considered as 1.
        stride (int):
            How much to move the window after each prediction.
            Comparable to stride in a 1d convolution.
        cut_length (int):
            The number of characters in each cut.
        device (torch.device):
            On which device to run the model.
    """

    def __init__(
        self,
        model_or_model_name,
        threshold=0.5,
        stride=90,
        cut_length=CUT_LENGTH,
        device=DEVICE,
    ):
        self.threshold = threshold
        self.stride = stride
        self.cut_length = cut_length
        self.device = device

        if isinstance(model_or_model_name, (torch.nn.Module, torch.jit.TracedModule)):
            self.model = model_or_model_name
        else:
            self.model = load_provided_model(model_or_model_name, device)

    def split(self, texts, batch_size=128):
        """
        Split texts into sentences and tokens.

        Parameters:
            texts (List[str]):
                A list of texts to split.
                Passing multiple texts at once allows for parallelization of the model.
            batch_size (int, Optional):
                Batch size with which cuts are processed by the model.

        Returns:
            A list with the same length as `texts`.
            - Each element is a list of sentences.
            - Each sentence is a list of tokens.
            - Each token is a namedtuple with `text` and `whitespace`.
        """
        all_inputs = []
        all_idx = []
        n_cuts_per_text = []

        for text in texts:
            inputs = [text_to_id(x) for x in text]

            while len(inputs) < self.cut_length:
                inputs.append(0)

            start = 0
            end = -1
            i = 0

            # split the inputs into partially overlapping slices which all have the same length
            # allows efficient batching
            while end != len(inputs):
                end = min(start + self.cut_length, len(inputs))
                start = end - self.cut_length

                idx = slice(start, end)
                all_inputs.append(inputs[idx])
                all_idx.append(idx)

                start += self.stride
                i += 1

            n_cuts_per_text.append(i)

        batched_inputs = torch.tensor(all_inputs, dtype=torch.int64)
        preds = np.zeros((len(batched_inputs), self.cut_length, 2), dtype=np.float32)

        for start in range(0, len(batched_inputs), batch_size):
            end = start + batch_size
            preds[start:end] = torch.sigmoid(
                self.model(batched_inputs[start:end].to(self.device))
                .detach()
                .float()
                .cpu()
            ).numpy()

        all_avg_preds = [np.zeros((len(text), 3), dtype=np.float32) for text in texts]

        current_text = 0
        current_i = 0
        for pred, idx in zip(preds, all_idx):
            current_preds = all_avg_preds[current_text]

            # add current slice to average preds
            current_preds[idx, :2] += pred[: len(current_preds)]
            current_preds[idx, 2] += 1

            current_i += 1

            if current_i == n_cuts_per_text[current_text]:
                # divide predictions by number of predictions so they are on the same scale
                all_avg_preds[current_text] = (
                    (current_preds[:, :2] / current_preds[:, [2]]) > self.threshold
                ).astype(np.bool)
                # for better handling with np.where, so that each index is only interated over once
                all_avg_preds[current_text][:, 0] &= ~all_avg_preds[current_text][:, 1]

                current_text += 1
                current_i = 0

        tokenized_texts = []

        for text, avg_preds in zip(texts, all_avg_preds):
            sentences = []
            tokens = []

            prev_index = 0
            index = 0

            for index, kind in zip(*np.where(avg_preds)):
                index = index + 1

                tokens.append(_get_token(text[prev_index:index]))

                if kind == 1:
                    sentences.append(tokens)
                    tokens = []

                prev_index = index

            if index != len(text):
                tokens.append(_get_token(text[index:]))

            if len(tokens) > 0:
                sentences.append(tokens)

            tokenized_texts.append(sentences)

        return tokenized_texts
