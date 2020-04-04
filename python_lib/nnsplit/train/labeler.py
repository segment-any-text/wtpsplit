from typing import List
from fractions import Fraction
from abc import ABC, abstractmethod
import spacy
import string
import random
import requests
from functools import lru_cache
import numpy as np
import pandas as pd


def has_space(text: str) -> bool:
    return any(x.isspace() for x in text)


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


class SpacySentenceTokenizer(Tokenizer):
    @staticmethod
    def remove_last_punct(text: str) -> str:
        for i in range(len(text))[::-1]:
            if text[i] in string.punctuation:
                return text[:i] + text[i + 1 :]
            elif not text[i].isspace():
                return text

        return text

    def __init__(
        self,
        model_name: str,
        lower_start_prob: Fraction,
        remove_end_punct_prob: Fraction,
    ):
        self.nlp = spacy.load(model_name, disable=["tagger", "parser", "ner"])
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

        self.lower_start_prob = lower_start_prob
        self.remove_end_punct_prob = remove_end_punct_prob

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
                if random.random() < self.remove_end_punct_prob:
                    current_sentence = self.remove_last_punct(current_sentence)

                out_sentences.append(current_sentence)

                current_sentence = ""
                end_sentence = False

            if len(current_sentence) == 0 and random.random() < self.lower_start_prob:
                text = text.lower()

            current_sentence += text + whitespace

        out_sentences.append(current_sentence)

        return [x for x in out_sentences if len(x) > 0]


class SpacyWordTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self.nlp = spacy.load(model_name, disable=["tagger", "parser", "ner"])

    def tokenize(self, text: str) -> List[str]:
        out_tokens = []
        current_token = ""

        for token in self.nlp(text):
            if not token.text.isspace():
                out_tokens.append(current_token)
                current_token = ""

            current_token += token.text + token.whitespace_

        out_tokens.append(current_token)

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

        assert not has_space(out[0])
        return out


class SECOSCompoundTokenizer(Tokenizer):
    def __init__(self, server_url: str):
        self.server_url = server_url

    @lru_cache(maxsize=2 ** 16)
    def tokenize(self, text: str) -> List[str]:
        if text.isspace():
            return [text]

        assert not has_space(text)

        response = requests.get(self.server_url, params={"sentence": text})
        return response.text.split()


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
        text = ""
        label = []

        for (token, annotation) in annotations:
            text += token
            label += [[0] * len(self.tokenizers) for _ in range(len(token))]

            for idx in annotation:
                label[-1][idx] = 1

        return text, np.array(label)

    def label(self, text):
        return self._to_dense_label(self._annotate(text))

    def visualize(self, text):
        text, label = self.label(text)

        data = []
        for char, label_col in zip(text, label):
            data.append([char, *label_col])

        df = pd.DataFrame(
            data, columns=["char", *[x.__class__.__name__ for x in self.tokenizers]]
        ).T
        df.columns = ["" for _ in range(len(df.columns))]

        with pd.option_context(
            "display.max_columns", len(text),
        ):
            print(df)


labeler = Labeler(
    [
        SpacySentenceTokenizer(
            "de_core_news_sm", lower_start_prob=0.5, remove_end_punct_prob=0.5
        ),
        SpacyWordTokenizer("de_core_news_sm"),
        WhitespaceTokenizer(),
        SECOSCompoundTokenizer("http://localhost:2020"),
    ]
)

labeler.visualize("Das ist ein Dampfschiff.   Das ist noch ein Test.")
