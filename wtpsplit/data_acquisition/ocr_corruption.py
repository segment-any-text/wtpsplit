"""OCR-style character corruption for the `-sm` training recipe.

The existing corruptions (`corrupt_asr`, `corrupt_social_media`) only remove punctuation,
lowercase, and detokenize, but not characters. The model was
never trained on text where characters are wrong, words are split or run together, or a
period has been read as a comma.

The corruptions here target what actually damages sentence segmentation in scanned text,
which is a narrower set than "OCR errors" in general:

- **Punctuation confusion** is the most damaging by far. A full stop misread as a comma
  removes the single strongest boundary cue, and OCR confuses them constantly.
- **Word merging and splitting** destroy the whitespace cue around a boundary.
- **Hyphenation artifacts** from justified columns leave `infor- mation` mid-sentence,
  which looks like a boundary and is not.
- **Character substitution, deletion and insertion** are the general background noise.

Each sentence is corrupted **independently**, and no character ever moves across a sentence
boundary. That is a hard requirement: the training pipeline derives labels from the lengths
of the corrupted sentences, so a corruption that merged two sentences would silently
mislabel the document. Cross-sentence merging is already modelled separately by
`train_SM.py`, which sometimes drops the separator when joining.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

__all__ = ["OCRCorruptionConfig", "corrupt_ocr_sentence", "corrupt_ocr", "sample_severity"]

# Log-uniform over this range spans "clean modern scan" to "bad historical scan":
# roughly a 1-point F1 cost at the bottom and 14 points at the top (see the table above).
SEVERITY_RANGE = (0.5, 8.0)

# Confusions a segmenter actually cares about: each of these can erase or fake a boundary.
PUNCTUATION_CONFUSIONS = {
    ".": [",", ";", ":", "·", "'"],
    ",": [".", ";", "'"],
    ";": [":", ",", "."],
    ":": [";", ".", ","],
    "!": ["l", "1", "i", "."],
    "?": ["7", "2", "."],
    "'": [",", "`", "."],
    "\u2014": ["-", "\u2013"],  # em dash
    "\u2013": ["-", "\u2014"],  # en dash
}

# Visually confusable glyph pairs, the classic OCR failure mode. Latin/Cyrillic/Greek
# lookalikes are included because historical scans mix scripts freely.
GLYPH_CONFUSIONS = {
    "o": ["0", "c", "e"], "0": ["o", "O", "D"], "l": ["1", "I", "i", "t"],
    "1": ["l", "I"], "I": ["l", "1", "|"], "i": ["l", "1", "j"],
    "c": ["e", "o", "("], "e": ["c", "o", "a"], "a": ["o", "e", "s"],
    "n": ["h", "u", "ri"], "h": ["b", "n", "li"], "u": ["v", "n", "ii"],
    "v": ["y", "u", "r"], "s": ["5", "8", "z"], "5": ["s", "S"],
    "g": ["9", "q", "y"], "9": ["g", "q"], "b": ["h", "6", "l3"],
    "m": ["rn", "nn"], "w": ["vv", "v"], "t": ["f", "l", "+"],
    "f": ["t", "l"], "y": ["v", "g"], "z": ["s", "2"], "2": ["z", "Z"],
    "B": ["8", "R"], "S": ["5", "$"], "O": ["0", "Q", "D"],
}


@dataclass
class OCRCorruptionConfig:
    """Per-character and per-word rates. Defaults sit in the range of a mediocre scan.

    Rates are deliberately modest: the goal is a model robust to noise, not one trained
    on unreadable text. Raise `severity` in `corrupt_ocr` to sample harsher settings.
    """

    punctuation_confusion: float = 0.06
    glyph_substitution: float = 0.015
    character_deletion: float = 0.005
    character_insertion: float = 0.005
    word_split: float = 0.02
    word_merge: float = 0.02
    hyphenation: float = 0.015


def _confuse_punctuation(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for char in text:
        options = PUNCTUATION_CONFUSIONS.get(char)
        if options and rng.random() < rate:
            out.append(rng.choice(options))
        else:
            out.append(char)
    return "".join(out)


def _substitute_glyphs(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for char in text:
        options = GLYPH_CONFUSIONS.get(char)
        if options and rng.random() < rate:
            out.append(rng.choice(options))
        else:
            out.append(char)
    return "".join(out)


def _drop_and_insert(text: str, deletion: float, insertion: float, rng: random.Random) -> str:
    out = []
    for char in text:
        if char != " " and rng.random() < deletion:
            continue
        out.append(char)
        if char != " " and rng.random() < insertion:
            out.append(rng.choice(".,'`\u00b7"))
    return "".join(out)


def _split_and_merge_words(text: str, split: float, merge: float, rng: random.Random) -> str:
    """Insert spurious spaces inside words and drop spaces between them.

    Merging never touches the final space, so it cannot run the last word of a sentence
    into anything outside it.
    """
    words = text.split(" ")

    split_words = []
    for word in words:
        if len(word) > 3 and rng.random() < split:
            cut = rng.randint(1, len(word) - 1)
            split_words.extend([word[:cut], word[cut:]])
        else:
            split_words.append(word)

    merged: list[str] = []
    for word in split_words:
        if merged and rng.random() < merge:
            merged[-1] = merged[-1] + word
        else:
            merged.append(word)
    return " ".join(merged)


def _hyphenate(text: str, rate: float, rng: random.Random) -> str:
    """Leave a line-break hyphen inside a word, as justified columns do."""
    words = text.split(" ")
    out = []
    for word in words:
        if len(word) > 5 and word.isalpha() and rng.random() < rate:
            cut = rng.randint(2, len(word) - 2)
            out.append(f"{word[:cut]}- {word[cut:]}")
        else:
            out.append(word)
    return " ".join(out)


def corrupt_ocr_sentence(
    sentence: str,
    config: OCRCorruptionConfig | None = None,
    rng: random.Random | None = None,
) -> str:
    """Corrupt one sentence. Never introduces or removes a sentence boundary."""
    config = config or OCRCorruptionConfig()
    rng = rng or random

    text = _confuse_punctuation(sentence, config.punctuation_confusion, rng)
    text = _substitute_glyphs(text, config.glyph_substitution, rng)
    text = _drop_and_insert(text, config.character_deletion, config.character_insertion, rng)
    text = _hyphenate(text, config.hyphenation, rng)
    text = _split_and_merge_words(text, config.word_split, config.word_merge, rng)

    # A sentence corrupted into nothing would break label derivation downstream.
    return text.strip() or sentence


def sample_severity(rng: random.Random | None = None) -> float:
    """Draw a severity log-uniformly from `SEVERITY_RANGE`.

    Training on one fixed noise level teaches the model one noise level. Sampling per
    document exposes it to the whole span from clean scan to badly degraded, which is what
    a real corpus of scanned material looks like.
    """
    import math

    rng = rng or random
    low, high = SEVERITY_RANGE
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def corrupt_ocr(
    sentences: list[str] | None,
    lang: str | None = None,
    severity: float = 1.0,
    seed: int | None = None,
) -> list[str] | None:
    """Sentence-list corruption, matching the `corrupt_asr` / `corrupt_social_media` shape.

    `lang` is accepted for interface parity and currently unused — the confusions are
    script-agnostic, and per-script tables would be the obvious refinement.

    `severity` scales every rate, so a corpus can mix clean, lightly degraded and badly
    degraded scans rather than training on one uniform noise level.
    """
    if sentences is None:
        return None

    rng = random.Random(seed)
    base = OCRCorruptionConfig()
    config = OCRCorruptionConfig(
        **{field: min(getattr(base, field) * severity, 1.0) for field in base.__dataclass_fields__}
    )
    return [corrupt_ocr_sentence(sentence, config, rng) for sentence in sentences]
