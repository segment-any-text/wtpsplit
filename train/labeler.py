from typing import List
from fractions import Fraction
from abc import ABC, abstractmethod
import spacy
import string
import random
import requests
import pandas as pd
import diskcache
from somajo import SoMaJo


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


def remove_last_punct(text: str) -> str:
    for i in range(len(text))[::-1]:
        if text[i] in string.punctuation:
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
    ):
        super().__init__()
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
                if self.training and random.random() < self.remove_end_punct_prob:
                    current_sentence = remove_last_punct(current_sentence)

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
        self.tokenizer = spacy.load(
            model_name, disable=["tagger", "parser", "ner"]
        ).tokenizer

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
    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url

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

            response = requests.get(self.server_url, params={"sentence": text})
            compounds = response.text

            if len(compounds) == 0:
                compounds = text

            compound_bytes = compounds.encode("utf-8")

            self.disk_cache[text_bytes] = compound_bytes
            self.cache[text_bytes] = compound_bytes
        else:
            compounds = compounds.decode("utf-8")

        return compounds.split()


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
            "display.max_columns", len(text),
        ):
            print(df)


if __name__ == "__main__":
    labeler = Labeler(
        [
            SpacySentenceTokenizer(
                "de_core_news_sm", lower_start_prob=0.7, remove_end_punct_prob=0.7
            ),
            SpacyWordTokenizer("de_core_news_sm"),
            # WhitespaceTokenizer(),
            # SECOSCompoundTokenizer("http://localhost:2020"),
        ]
    )
    labeler.visualize("Die erste Million Jahre vergeht schnell, die zweite Million...")
