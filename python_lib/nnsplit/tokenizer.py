__all__ = ["Token", "Tokenizer", "SoMaJoTokenizer"]

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Token:
    text: str
    whitespace: str


class Tokenizer(ABC):
    @abstractmethod
    def split(self, texts):
        """
        `split` returns an iterable over texts. Every text is an iterable over sentences.
        Every sentence is an iterable over `Token` instances.
        """
        pass


class SoMaJoTokenizer(Tokenizer):
    def __init__(self, language):
        from somajo import SoMaJo

        tokenizer_type = {"de": "de_CMC", "en": "en_PTB"}[language]
        self.tokenizer = SoMaJo(
            tokenizer_type, split_camel_case=True, split_sentences=True
        )

    def split(self, texts):
        tokenized_texts = []
        for text in texts:
            sentences = []
            for sentence in self.tokenizer.tokenize_text([text]):
                sentences.append(
                    [
                        Token(token.text, " " if token.space_after else "")
                        for token in sentence
                    ]
                )

            if not text[-1].isspace():
                sentences[-1][-1].whitespace = ""

            tokenized_texts.append(sentences)

        return tokenized_texts
