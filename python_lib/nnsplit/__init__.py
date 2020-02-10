__version__ = "0.1.3"
__all__ = ["NNSplit"]

import numpy as np
import torch
from .defaults import CUT_LENGTH, DEVICE
from .utils import load_provided_model, text_to_id
from .tokenizer import Tokenizer, Token


def _get_token(text):
    try:
        last_char_index = next(i for i, x in enumerate(text[::-1]) if not x.isspace())
    except StopIteration:  # only spaces in text
        last_char_index = len(text)
    last_char_index = len(text) - last_char_index

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
        from tqdm.auto import tqdm

        for start in tqdm(range(0, len(batched_inputs), batch_size)):
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
                current_preds[:, :2] /= current_preds[:, [2]]

                current_text += 1
                current_i = 0

        tokenized_texts = []

        for i, avg_preds in enumerate(all_avg_preds):
            sentences = []
            tokens = []
            token = ""

            for char, pred in zip(texts[i], avg_preds):
                token += char

                if pred[0] > self.threshold or pred[1] > self.threshold:
                    tokens.append(_get_token(token))
                    token = ""

                if pred[1] > self.threshold:
                    sentences.append(tokens)
                    tokens = []

            if len(token) > 0:
                tokens.append(_get_token(token))

            if len(tokens) > 0:
                sentences.append(tokens)

            tokenized_texts.append(sentences)

        return tokenized_texts
