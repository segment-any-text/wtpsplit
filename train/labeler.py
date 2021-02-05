from typing import List
from fractions import Fraction
from abc import ABC, abstractmethod
import spacy
import string
import random
import pandas as pd
import numpy as np
import diskcache
import sys
from somajo import SoMaJo
from spacy.lang.tr import Turkish
from spacy.lang.sv import Swedish
from spacy.lang.uk import Ukrainian

NO_MODEL_LANGUAGE_LOOKUP = {
    "turkish": Turkish,
    "swedish": Swedish,
    "ukrainian": Ukrainian,
}


def noise(text, insert_chance, delete_chance, repeat_chance):
    assert insert_chance == delete_chance == repeat_chance

    chances = np.random.random(len(text) * 3)
    if (chances < insert_chance).all():
        return text

    out = ""

    for i, char in enumerate(text):
        if chances[i * 3] >= delete_chance:
            out += char
        if chances[(i * 3) + 1] < repeat_chance:
            out += char
        if chances[(i * 3) + 2] < insert_chance:
            out += random.choice(string.ascii_letters)

    return out


def get_model(name):
    try:
        nlp = spacy.load(name, disable=["tagger", "parser", "ner"])
    except OSError:
        nlp = NO_MODEL_LANGUAGE_LOOKUP[name]()

    return nlp


def has_space(text: str) -> bool:
    return any(x.isspace() for x in text)


class Tokenizer(ABC):
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


def remove_last_punct(text: str, punctuation) -> str:
    for i in range(len(text))[::-1]:
        if text[i] in punctuation:
            return text[:i] + text[i + 1 :]
        elif not text[i].isspace():
            return text

    return text


class SpacySentenceTokenizer(Tokenizer):
    def __init__(
        self,
        model_name: str,
        lower_start_prob: Fraction,
        remove_end_punct_prob: Fraction,
        punctuation: str,
    ):
        super().__init__()
        self.nlp = get_model(model_name)
        self.nlp.add_pipe("sentencizer")

        self.lower_start_prob = lower_start_prob
        self.remove_end_punct_prob = remove_end_punct_prob
        self.punctuation = punctuation

    def tokenize(self, text: str) -> List[str]:
        out_sentences = []
        current_sentence = ""
        end_sentence = False

        for token in self.nlp(text):
            text = token.text
            whitespace = token.whitespace_

            if token.is_sent_start:
                end_sentence = True

            if end_sentence and not text.isspace():
                if self.training and random.random() < self.remove_end_punct_prob:
                    current_sentence = remove_last_punct(current_sentence, self.punctuation)

                out_sentences.append(current_sentence)

                current_sentence = ""
                end_sentence = False

            if (
                self.training
                and len(current_sentence) == 0
                and random.random() < self.lower_start_prob
            ):
                text = text.lower()

            current_sentence += text + whitespace

        out_sentences.append(current_sentence)

        return [x for x in out_sentences if len(x) > 0]


class SpacyWordTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        super().__init__()

        self.tokenizer = get_model(model_name).tokenizer

    def tokenize(self, text: str) -> List[str]:
        out_tokens = []
        current_token = ""

        for token in self.tokenizer(text):
            if not token.text.isspace():
                out_tokens.append(current_token)
                current_token = ""

            current_token += token.text + token.whitespace_

        out_tokens.append(current_token)

        return [x for x in out_tokens if len(x) > 0]


class SoMaJoSentenceTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = SoMaJo(model_name)

    def tokenize(self, text: str) -> List[str]:
        out_sentences = []
        sentences = list(self.tokenizer.tokenize_text([text]))

        for i, sentence in enumerate(sentences):
            text = ""

            for token in sentence:
                if "SpaceAfter=No" in token.extra_info:
                    whitespace = ""
                else:
                    whitespace = " "

                text += token.text + whitespace

            if i == len(sentences) - 1:
                text = text.rstrip()

            out_sentences.append(text)

        return out_sentences


class SoMaJoWordTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = SoMaJo(model_name, split_sentences=False)

    def tokenize(self, text: str) -> List[str]:
        out_tokens = []
        tokens = next(self.tokenizer.tokenize_text([text]))

        for i, token in enumerate(tokens):
            if "SpaceAfter=No" in token.extra_info or i == len(tokens) - 1:
                whitespace = ""
            else:
                whitespace = " "

            # sometimes sample more spaces than one space so the model learns to deal with it
            while random.random() < 0.05:
                whitespace += " "

            out_tokens.append(token.text + whitespace)

        return [x for x in out_tokens if len(x) > 0]


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        out = None

        for i in range(len(text))[::-1]:
            if not text[i].isspace():
                out = [text[: i + 1], text[i + 1 :]]
                break

        if out is None:
            out = [text, ""]

        return out


class SECOSCompoundTokenizer(Tokenizer):
    def __init__(self, secos_path: str):
        super().__init__()
        sys.path.append(secos_path)
        import decompound_server

        self.decompound = decompound_server.make_decompounder(
            [
                "decompound_server.py",
                f"{secos_path}data/denews70M_trigram__candidates",
                f"{secos_path}data/denews70M_trigram__WordCount",
                "50",
                "3",
                "3",
                "5",
                "3",
                "upper",
                "0.01",
                "2020",
            ]
        )

        self.disk_cache = diskcache.Index("secos_cache")
        self.cache = {}

        for key in self.disk_cache:
            self.cache[key] = self.disk_cache[key]

    def tokenize(self, text: str) -> List[str]:
        if text.isspace():
            return [text]

        text_bytes = text.encode("utf-8")

        compounds = self.cache.get(text_bytes)

        if compounds is None:
            assert not has_space(text), text

            compounds = self.decompound(text)

            if len(compounds) == 0:
                compounds = text

            compound_bytes = compounds.encode("utf-8")

            self.disk_cache[text_bytes] = compound_bytes
            self.cache[text_bytes] = compound_bytes
        else:
            compounds = compounds.decode("utf-8")

        compounds = compounds.split()
        compounds = [noise(x, 0.001, 0.001, 0.001) for x in compounds]

        return compounds if len(compounds) > 0 else [noise(text, 0.001, 0.001, 0.001)]


class Labeler:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers

    def _annotate(self, text: str, tok_index=0):
        if tok_index >= len(self.tokenizers):
            return [(text, set())]

        out = []

        for token in self.tokenizers[tok_index].tokenize(text):
            out += self._annotate(token, tok_index=tok_index + 1)
            out[-1][1].add(tok_index)

        return out

    def _to_dense_label(self, annotations):
        input_bytes = []
        label = []

        all_zeros = [0] * len(self.tokenizers)

        for (token, annotation) in annotations:
            token_bytes = token.encode("utf-8")
            input_bytes += token_bytes
            label += [all_zeros.copy() for _ in range(len(token_bytes))]

            if len(label) > 0:
                for idx in annotation:
                    label[-1][idx] = 1

        return input_bytes, label

    def label(self, text):
        return self._to_dense_label(self._annotate(text))

    def visualize(self, text):
        text, label = self.label(text)

        data = []
        for char, label_col in zip(text, label):
            data.append([char, *label_col])

        df = pd.DataFrame(
            data, columns=["byte", *[x.__class__.__name__ for x in self.tokenizers]]
        ).T
        df.columns = ["" for _ in range(len(df.columns))]

        with pd.option_context(
            "display.max_columns",
            len(text),
        ):
            print(df)


if __name__ == "__main__":
    labeler = Labeler(
        [
            SpacySentenceTokenizer(
                "de_core_news_sm", lower_start_prob=0.7, remove_end_punct_prob=0.7, punctuation=".?!"
            ),
            SpacyWordTokenizer("de_core_news_sm"),
            WhitespaceTokenizer(),
            SECOSCompoundTokenizer("../../../Experiments/SECOS/"),
        ]
    )
    labeler.visualize("KNN (ANN).")
