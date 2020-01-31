__all__ = ["Tokenizer", "SoMaJoTokenizer"]

from abc import ABC, abstractmethod
from somajo import SoMaJo


class Tokenizer(ABC):
    @abstractmethod
    def split(self, texts):
        """
        `split` returns an iterable over texts. Every text is an iterable over sentences.
        Every sentence is an iterable over tokens. A token must have a `space_after` boolean
        attribute and a `text` string attribute.
        """
        pass


class SoMaJoTokenizer(Tokenizer):
    def __init__(self, language):

        tokenizer_type = {"de": "de_CMC", "en": "en_PTB"}[language]
        self.tokenizer = SoMaJo(
            tokenizer_type, split_camel_case=True, split_sentences=True
        )

    def split(self, texts):
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(self.tokenizer.tokenize_text([text]))

        return tokenized_texts
