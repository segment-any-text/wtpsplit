"""Request-local character features for the staged extractor.

`DocumentFeatures` builds one document at a time (about 20 small numpy calls each) and `batch` copies window
slices one by one. Here all documents of a request are processed in flat arrays and every window batch is a
single gather. Documents with empty, out-of-range or non-monotone offsets, and windows that cannot reuse the
document features, keep the library implementation, so odd inputs behave exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import cast
from itertools import chain

from numpy.typing import NDArray

import numpy as np

from wtpsplit._char_preprocessing import build_char_inputs_vectorized
from wtpsplit.char_head import CharInputs
from wtpsplit.utils import PRIMES

MAX_POSITION = 15
NUM_BUCKETS = 8192


@dataclass(frozen=True)
class CharacterWindow:
    text_index: int
    token_start: int
    char_start: int
    char_end: int
    text: str
    offsets: list[tuple[int, int]]
    reuse: bool
    input_ids: list[int]


class BatchFeatures:
    def __init__(self, texts: list[str], offsets: list[list[tuple[int, int]]]) -> None:
        n = len(texts)
        self.texts, self.offsets = texts, offsets
        lengths = np.fromiter(map(len, texts), dtype=np.int64, count=n)
        self.base = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        total = int(lengths.sum())
        self.ordinals = np.frombuffer("".join(texts).encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
        counts = np.fromiter(map(len, offsets), dtype=np.int64, count=n)
        flat = np.fromiter(
            chain.from_iterable(chain.from_iterable(offsets)), dtype=np.int64, count=int(counts.sum()) * 2
        ).reshape(-1, 2)
        token_doc = np.repeat(np.arange(n), counts)
        token_local = np.arange(len(flat)) - np.repeat(np.cumsum(counts) - counts, counts)
        valid = flat[:, 1] > flat[:, 0]
        doc, local, start, end = token_doc[valid], token_local[valid], flat[valid, 0], flat[valid, 1]
        unsafe = np.bincount(doc, minlength=n) == 0
        unsafe |= np.bincount(doc[(start < 0) | (end > lengths[doc])], minlength=n) > 0
        same = doc[1:] == doc[:-1]
        unsafe |= np.bincount(doc[1:][same & ((start[1:] < start[:-1]) | (end[1:] < end[:-1]))], minlength=n) > 0
        self.unsafe = unsafe
        keep = ~unsafe[doc]
        doc, local, start, end = doc[keep], local[keep], start[keep], end[keep]
        gstart, gend = self.base[doc] + start, self.base[doc] + end
        chars = np.arange(total)
        owner = np.searchsorted(gstart, chars, side="right") - 1
        safe_owner = np.maximum(owner, 0)
        if len(gstart):
            covered = (owner >= 0) & (chars < gend[safe_owner])
        else:
            covered = np.zeros(total, dtype=bool)
        picked = owner[covered]
        self.covered = covered
        self.mapping = np.zeros(total, dtype=np.int64)
        self.mapping[covered] = local[picked]
        self.position = np.zeros(total, dtype=np.int64)
        self.position[covered] = np.minimum(chars[covered] - gstart[picked], MAX_POSITION)
        self.final = np.zeros(total, dtype=np.int64)
        self.final[gend - 1] = 1
        self.primes = np.asarray(PRIMES[:8], dtype=np.int64)
        self._edges: dict[int, tuple[NDArray[np.int64], NDArray[np.int64]]] = {}

    def can_slice(self, index: int, token_start: int, token_end: int, char_start: int, char_end: int) -> bool:
        if bool(self.unsafe[index]):
            return False
        if not token_start and token_end >= len(self.offsets[index]):
            return True  # the whole document: no excluded token can touch the window
        if index not in self._edges:
            spans = np.asarray(self.offsets[index], dtype=np.int64).reshape(-1, 2)
            valid = spans[:, 1] > spans[:, 0]
            self._edges[index] = (
                np.maximum.accumulate(np.where(valid, spans[:, 1], -1)),
                np.minimum.accumulate(np.where(valid, spans[:, 0], len(self.texts[index]) + 1)[::-1])[::-1],
            )
        prefix, suffix = self._edges[index]
        return not (
            (token_start != 0 and prefix[token_start - 1] > char_start)
            or (token_end < len(suffix) and suffix[token_end] < char_end)
        )

    def batch(self, windows: list[CharacterWindow]) -> dict[str, NDArray[np.int64]]:
        """Char arrays (int64, numpy) for one batch of windows, as `DocumentFeatures.batch` plus `char_to_token + 1`."""
        starts = np.array([w.char_end - w.char_start for w in windows], dtype=np.int64)
        width = int(starts.max())
        fast = np.array([w.reuse for w in windows], dtype=bool)
        index = np.array([self.base[w.text_index] + w.char_start for w in windows], dtype=np.int64)
        token_start = np.array([w.token_start for w in windows], dtype=np.int64)
        position = np.arange(width)
        inside = (position[None, :] < starts[:, None]) & fast[:, None]
        gather = np.minimum(index[:, None] + position[None, :], max(len(self.ordinals) - 1, 0))
        mapped = np.where(inside & self.covered[gather], self.mapping[gather] - token_start[:, None], 0)
        out = dict(
            char_to_token=mapped,
            char_is_token_final=np.where(inside, self.final[gather], 0),
            char_position_in_token=np.where(inside, self.position[gather], 0),
            char_mask=inside.astype(np.int64),
        )
        hashes = ((np.where(inside, self.ordinals[gather], 0)[..., None] + 1) * self.primes) % NUM_BUCKETS
        for row in np.flatnonzero(~fast):
            window = windows[row]
            fallback = cast(
                Callable[[list[str], list[list[tuple[int, int]]]], CharInputs], build_char_inputs_vectorized
            )([window.text], [window.offsets])
            length = int(starts[row])
            out["char_to_token"][row, :length] = fallback.char_to_token[0].numpy()
            out["char_is_token_final"][row, :length] = fallback.is_token_final[0].numpy()
            out["char_position_in_token"][row, :length] = fallback.position_in_token[0].numpy()
            out["char_mask"][row, :length] = fallback.mask[0].numpy()
            hashes[row, :length] = fallback.hashes[0].numpy()
        out["char_to_token"] = out["char_to_token"] + 1
        out["char_hashes"] = hashes
        return {key: np.ascontiguousarray(value, dtype=np.int64) for key, value in out.items()}
