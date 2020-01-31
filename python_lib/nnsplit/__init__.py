__version__ = "0.1.0"
__all__ = ["NNSplit"]

import numpy as np
import torch
from .defaults import CUT_LENGTH, DEVICE
from .utils import load_model, text_to_id
from .tokenizer import Tokenizer


class NNSplit(Tokenizer):
    def __init__(
        self,
        model_name_or_path,
        threshold=0.5,
        stride=50,
        cut_length=CUT_LENGTH,
        device=DEVICE,
    ):
        self.threshold = threshold
        self.stride = stride
        self.cut_length = cut_length
        self.device = device

        self.model = load_model(model_name_or_path)

    def split(self, texts, batch_size=32, max_length=4000):
        all_inputs = []
        all_idx = []
        n_cuts_per_text = []

        raw_max_length = max(map(len, texts))
        max_effective_length = min(max_length, raw_max_length)
        max_effective_length = max(
            self.cut_length, max_effective_length
        )  # has to be at least 1 cut long

        for text in texts:
            inputs = [text_to_id(x) for x in text]

            while len(inputs) < max_effective_length:
                inputs.append(0)

            start = 0
            end = -1
            i = 0

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
        preds = torch.sigmoid(self.model(batched_inputs).detach().cpu()).numpy()

        all_avg_preds = np.zeros(
            (len(texts), max_effective_length, 3), dtype=np.float32
        )

        current_text = 0
        current_i = 0
        for pred, idx in zip(preds, all_idx):
            current_i += 1

            if current_i == n_cuts_per_text[current_text]:
                all_avg_preds[current_text, idx, :2] += pred
                all_avg_preds[current_text, idx, 2] += 1

                current_text += 1
                current_i = 0

        all_avg_preds = all_avg_preds[:, :, :2] / all_avg_preds[:, :, [2]]

        tokenized_texts = []

        for i, avg_preds in enumerate(all_avg_preds):
            sentences = []
            tokens = []
            token = ""

            for char, pred in zip(texts[i], avg_preds):
                token += char

                if pred[0] > self.threshold:
                    tokens.append(token)
                    token = ""

                if pred[1] > self.threshold:
                    sentences.append(tokens)
                    tokens = []

            if len(token) > 0:
                tokens.append(token)

            if len(tokens) > 0:
                sentences.append(tokens)

            tokenized_texts.append(sentences)

        return tokenized_texts
