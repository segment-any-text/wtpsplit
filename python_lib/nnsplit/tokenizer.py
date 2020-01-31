__all__ = ["Tokenizer", "SoMaJoTokenizer"]

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def split(self, texts):
        """
        `split` returns an iterable over texts. Every text is an iterable over sentences.
        Every sentence is an iterable over string tokens.
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
            tokenized_texts.append(self.tokenizer.tokenize_text([text]))

        return tokenized_texts
