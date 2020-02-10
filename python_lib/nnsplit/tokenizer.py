__all__ = ["Token", "Tokenizer", "SoMaJoTokenizer"]

from abc import ABC, abstractmethod
from collections import namedtuple

Token = namedtuple("Token", ["text", "whitespace"])


class Tokenizer(ABC):
    @abstractmethod
    def split(self, texts):
        """
        `split` returns an iterable over texts. Every text is an iterable over sentences.
        Every sentence is an iterable over `Token` namedtuple instances.
        """
        pass


class SoMaJoTokenizer(Tokenizer):
    def __init__(self, language, processes=None):
        from somajo import SoMaJo

        tokenizer_type = {"de": "de_CMC", "en": "en_PTB"}[language]
        self.tokenizer = SoMaJo(
            tokenizer_type, split_camel_case=True, split_sentences=True
        )

    def _tokenize_text(self, text):
        sentences = []

        if len(text) == 0:
            return sentences

        for sentence in self.tokenizer.tokenize_text([text]):
            sentences.append(
                [
                    Token(token.text, " " if token.space_after else "")
                    for token in sentence
                ]
            )

        if not text[-1].isspace():
            sentences[-1][-1] = Token(sentences[-1][-1].text, "")

        return sentences

    def split(self, texts, verbose=False):
        bar = None
        if verbose:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(texts))

        # pool.imap leaks memory for some reason
        for sentences in map(self._tokenize_text, texts):
            yield sentences

            if verbose:
                bar.update(1)
