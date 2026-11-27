"""Grapheme-safe source slices with verified length budgets, independent of models."""

from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property, wraps
from heapq import heappop, heappush
from inspect import Parameter, signature
from itertools import accumulate
from numbers import Integral, Real
from typing import Any, ParamSpec, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
import regex  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerBase

Span = tuple[int, int]
LengthFunction = Callable[[str], int]
_P = ParamSpec("_P")
_R = TypeVar("_R")
_SCORE_TOLERANCE = 1e-9
_RUNS = regex.compile(r"[\s\x1c-\x1f]+|[^\s\x1c-\x1f]+")
__all__ = ["Span", "LengthFunction", "Tokenizer", "TokenLength", "UnsplittableUnitError", "split_spans", "chunk_spans"]


class Tokenizer(Protocol):
    """Caller tokenizer with an explicit special-token policy.

    Only Hugging Face PreTrainedTokenizerBase receives truncation=False and
    verbose=False. Other implementations receive add_special_tokens only and
    must encode the complete input without truncation themselves.
    """

    def encode(self, text: str, *, add_special_tokens: bool = True) -> Sequence[int]:
        """Encode the complete input, without truncation."""
        ...


class UnsplittableUnitError(ValueError):
    """The leftmost indivisible legal atom exceeds the cap.

    position is its source start, independent of probabilities and threshold
    for a fixed text, budget, counter and legal-position policy.
    """

    def __init__(self, position: int, measured_length: int, max_length: int) -> None:
        """Store source position, measured atom length and requested cap."""
        self.max_length = max_length
        self.position = position
        self.measured_length = measured_length
        super().__init__(
            f"No legal piece fits at position {position}: measured length {measured_length}, cap {max_length}"
        )


def measure(text: str, length_fn: LengthFunction) -> int:
    """Validate a count on the actual slice."""
    value = length_fn(text)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"length_fn must return a nonnegative integer; received {value!r}")
    return int(value)


def validate_budget(max_length: int, name: str = "max_length") -> None:
    """Reject nonpositive/noninteger budgets with the received value."""
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length <= 0:
        raise ValueError(f"{name} must be a positive integer; received {max_length!r}")


def validate_overlap(value: int) -> None:
    """Validate overlap before model inference."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"overlap_sentences must be a nonnegative integer; received {value!r}")


_RETIRED_HINTS = {
    "min_length": "no minimum-length replacement exists; use chunk(max_length=...) for packing",
    "prior_type": "no prior replacement exists; use split(max_length=..., threshold=...) for capped sentences",
    "prior_kwargs": "no prior-parameter replacement exists; use split(max_length=..., threshold=...)",
    "algorithm": "use split(max_length=...) with the shared decoder",
    "use_negative_evidence": "use split(max_length=..., threshold=...) for the fixed objective",
    "max_tokens": "use max_length=N, length_fn=TokenLength(tokenizer, ...)",
    "tokenizer": "use length_fn=TokenLength(tokenizer, ...) with max_length=N",
    "add_special_tokens": "set add_special_tokens on TokenLength(tokenizer, ...) and pass it as length_fn",
    "reserve_tokens": "set reserve_tokens on TokenLength(tokenizer, ...) and pass it as length_fn",
}


def reject_retired(options: dict[str, Any]) -> None:
    """Explain removal individually; unrecognised keys remain errors."""
    for key in options:
        if key in _RETIRED_HINTS:
            raise TypeError(f"{key} was removed in 3.0; {_RETIRED_HINTS[key]}.")
        raise TypeError(f"Unexpected keyword argument: {key}")


def reject_legacy_keywords(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Keep typed signatures while explaining removed runtime keywords."""

    positional = {
        name: i
        for i, (name, p) in enumerate(signature(function).parameters.items())
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    }
    length_index, cap_index = positional.get("length_fn"), positional.get("max_length")

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        reject_retired({k: v for k, v in kwargs.items() if k in _RETIRED_HINTS})
        supplied = "length_fn" in kwargs or (length_index is not None and len(args) > length_index)
        cap = kwargs.get("max_length", args[cap_index] if cap_index is not None and len(args) > cap_index else None)
        counter = kwargs.get(
            "length_fn", args[length_index] if length_index is not None and len(args) > length_index else len
        )
        if supplied and counter is not len and cap is None:
            raise TypeError("length_fn requires max_length; use max_length=N, length_fn=...")
        return function(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class TokenLength:
    """True embedding count, including optional special tokens and reserved prefix tokens."""

    tokenizer: Tokenizer
    _: KW_ONLY
    add_special_tokens: bool = True
    reserve_tokens: int = 0

    def __post_init__(self) -> None:
        if not callable(getattr(self.tokenizer, "encode", None)):
            raise TypeError("tokenizer must supply encode(text, add_special_tokens=...)")
        if not isinstance(self.add_special_tokens, bool):
            raise TypeError("add_special_tokens must be bool")
        if (
            isinstance(self.reserve_tokens, bool)
            or not isinstance(self.reserve_tokens, Integral)
            or self.reserve_tokens < 0
        ):
            raise ValueError(f"reserve_tokens must be a nonnegative integer; received {self.reserve_tokens!r}")

    @cached_property
    def overhead(self) -> int:
        """Fixed empty-input count, measured once on first budget validation."""
        return self("")

    def __call__(self, text: str) -> int:
        """Count without truncation or model-length warnings."""
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            tokenizer: Any = self.tokenizer
            ids = tokenizer.encode(text, add_special_tokens=self.add_special_tokens, truncation=False, verbose=False)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
        return len(ids) + self.reserve_tokens


def resolve_budget(max_length: int | None, length_fn: LengthFunction = len) -> tuple[int, LengthFunction]:
    """Validate the single budget spelling, including fixed token overhead."""
    if max_length is None:
        raise ValueError("max_length must be a positive integer; received None")
    validate_budget(max_length)
    if not callable(length_fn):
        raise TypeError(f"length_fn must be callable; received {length_fn!r}")
    if isinstance(length_fn, TokenLength):
        overhead = length_fn.overhead
        if max_length <= overhead:
            raise ValueError(f"max_length ({max_length}) must exceed fixed token overhead ({overhead})")
    return int(max_length), length_fn


def grapheme_ends(text: str) -> list[int]:
    """Return exclusive extended-grapheme ends (regex Unicode tables)."""
    return [m.end() for m in regex.finditer(r"\X", text)]


def forced_newline_ends(text: str) -> list[int]:
    """Snap forced LF cuts without building whole-document grapheme arrays.

    UAX29 GB4 guarantees a grapheme boundary after LF, including CRLF. Scan
    only its following whitespace and the cluster at that run's end, retaining
    the shared decoder's combining-mark ownership. Long blank runs are scanned
    once. See https://www.unicode.org/reports/tr29/tr29-49.html#GB4.
    """
    if '\n' not in text:
        return []
    if text.isascii():
        return [match.end() for match in regex.finditer(r"\n", text)]
    result = []
    advanced = 0
    for match in regex.finditer(r"\n", text):
        start = match.end()
        if start <= advanced:
            continue
        end = start
        while end < len(text) and text[end].isspace():
            end += 1
        if end > start:
            for cluster in regex.finditer(r"\X", text, pos=start):
                if cluster.end() >= end:
                    end = cluster.end()
                    break
        result.append(end)
        advanced = end
    return result


def evidence_indices(text: str, cuts: Sequence[int] | npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Map each cut to the last non-whitespace character before it, or -1.

    This is the single evidence-index policy for ordinary and capped decoding.
    Leading/only whitespace has no model evidence and receives probability zero.
    """
    nonspace = np.fromiter((not c.isspace() for c in text), dtype=np.bool_, count=len(text))
    previous = np.maximum.accumulate(np.where(nonspace, np.arange(len(text)), -1))
    positions = np.asarray(cuts, dtype=np.int64)
    return previous[positions - 1] if len(positions) else np.empty(0, dtype=np.int64)


def source_spans(text: str, ends: Sequence[int]) -> list[Span]:
    """Snap predicted ends forward across whitespace and graphemes in O(n + k log n)."""
    if not text:
        return []
    boundaries = np.asarray(grapheme_ends(text), dtype=np.int64)
    # Each whitespace run is visited once, even with evidence at every position.
    next_nonspace = np.empty(len(text) + 1, dtype=np.int64)
    next_nonspace[-1] = len(text)
    for match in _RUNS.finditer(text):
        a, b = match.span()
        if text[a].isspace():
            next_nonspace[a:b] = b
        else:
            next_nonspace[a:b] = np.arange(a, b)
    supplied = np.asarray(ends, dtype=np.int64)
    supplied = supplied[(supplied > 0) & (supplied <= len(text))]
    evidence = evidence_indices(text, supplied)
    anchors = np.where(evidence >= 0, evidence + 1, supplied)
    snapped = boundaries[np.searchsorted(boundaries, next_nonspace[anchors])]
    cuts = [0, *np.unique(np.append(snapped, len(text))).tolist()]
    return list(zip(cuts, cuts[1:]))


def _probabilities(text: str, probabilities: npt.ArrayLike) -> npt.NDArray[np.float64]:
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must have len(text) finite values in [0, 1]")
    if not text.strip() and p.shape == (1,) and len(text) != 1:
        p = np.zeros(len(text))  # Established whitespace-only inference sentinel.
    if p.shape != (len(text),):
        raise ValueError("probabilities must have len(text) finite values in [0, 1]")
    return p


def _legal_cuts(text: str, ends: list[int], cap: int, count: Callable[[int, int], int]) -> list[int]:
    boundaries = np.asarray([0, *ends], dtype=np.int64)
    flags = np.fromiter(
        (
            text[a].isspace() or (b - a > 1 and any(c.isspace() for c in text[a:b]))
            for a, b in zip(boundaries, boundaries[1:])
        ),
        dtype=np.bool_,
        count=len(ends),
    )
    cuts = {0, len(text), *boundaries[1:][flags].tolist()}
    # Whitespace and its combining/spacing marks are one indivisible cluster.
    changes = np.flatnonzero(np.diff(np.r_[True, flags, True].astype(np.int8)))
    for lo, hi in changes.reshape(-1, 2):
        a, b = int(boundaries[lo]), int(boundaries[hi])
        if count(a, b) > cap:
            cuts.update(ends[lo:hi])
        elif hi < len(ends) and count(a, int(boundaries[hi + 1])) > cap:
            cuts.add(b)
    return sorted(cuts)


def _decode(
    cuts: list[int],
    lengths: npt.NDArray[Any],
    rewards: npt.NDArray[np.float64],
    cap: int,
    max_atoms: int | None = None,
    ownership: tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]] | None = None,
) -> list[Span]:
    """Exact suffix recurrence with heap maximum, secondary deque and tolerance window."""
    n = len(cuts) - 1
    have_ownership = ownership is not None
    interior, white_end = ownership if ownership is not None else (np.zeros(n + 1, dtype=bool), np.arange(n + 1))
    differences = np.diff(lengths)
    bad = np.flatnonzero(differences > cap)
    if len(bad):
        first = int(bad[0])
        raise UnsplittableUnitError(cuts[first], int(differences[first]), cap)
    cap = min(cap, int(lengths[-1] - lengths[0]))
    targets = (lengths.astype(object) if int(lengths[-1]) + cap > np.iinfo(np.int64).max else lengths) + cap
    rights = np.searchsorted(lengths, targets, side="right") - 1
    if max_atoms is not None:
        rights = np.minimum(rights, np.arange(n + 1) + max_atoms)
    if len(rewards) <= 2 or (rewards[1] <= 0 and np.all(rewards[1:-1] == rewards[1])):
        result = []
        i = 0
        while i < n:
            j = int(rights[i])
            result.append((cuts[i], cuts[j]))
            i = j
        return result
    # Unambiguous maxima take only heapq's C operations. Near ties use one
    # NumPy reduction over the eligible window, preserving the reference rule
    # even though tolerance is not a transitive ordering relation.
    heap: list[tuple[float, int]] = []
    secondary: deque[int] = deque()
    scores = [0.0] * (n + 1)
    pieces = [0] * (n + 1)
    following = [0] * (n + 1)
    candidate_scores = np.full(n + 1, -np.inf)
    candidate_keys = np.zeros(n + 1, dtype=np.int64)
    reward = rewards.tolist()
    right_limits = rights.tolist()
    for i in range(n - 1, -1, -1):
        j = i + 1
        value = scores[j] + reward[j]
        candidate_scores[j] = value
        candidate_keys[j] = pieces[j] * (n + 1) - j
        if not have_ownership or not interior[j]:
            heappush(heap, (-value, j))
        # Independent monotonic queue: the best secondary key in the window.
        # Equal counts retain the earlier (longer-prefix) endpoint.
        if not have_ownership or not interior[j]:
            while secondary and pieces[secondary[-1]] > pieces[j]:
                secondary.pop()
            secondary.append(j)
        right = right_limits[i]
        while secondary and secondary[0] > right:
            secondary.popleft()
        while heap and heap[0][1] > right:
            heappop(heap)
        if have_ownership and (white_end[i] > i or interior[right]):
            values = candidate_scores[i + 1 : right + 1].copy()
            blocked = interior[i + 1 : right + 1] & (np.abs(white_end[i + 1 : right + 1]) <= right)
            values[blocked] = -np.inf
            end = int(white_end[i])
            if i < end <= right:
                values[end - i - 1] = scores[end] + min(
                    reward[end], 0.0
                )  # No positive reward for a whitespace-only piece.
            floor = float(np.max(values)) - _SCORE_TOLERANCE
            keys = np.where(values >= floor, candidate_keys[i + 1 : right + 1], (n + 1) ** 2)
            j = i + 1 + int(np.argmin(keys))
        else:
            maximum = -heap[0][0]
            j = heap[0][1]
            floor = maximum - _SCORE_TOLERANCE
            ambiguous = (len(heap) > 1 and -heap[1][0] >= floor) or (len(heap) > 2 and -heap[2][0] >= floor)
            if candidate_scores[secondary[0]] >= floor:
                j = secondary[0]
            elif ambiguous:
                # The integer key orders fewer pieces then longer prefix. A
                # contiguous argmin avoids copying a reversed window on broad ties.
                # Expired heap children can only cause extra, harmless reductions.
                choices = np.where(
                    ((candidate_scores[i + 1 : right + 1] >= floor) & ~interior[i + 1 : right + 1])
                    if have_ownership
                    else candidate_scores[i + 1 : right + 1] >= floor,
                    candidate_keys[i + 1 : right + 1],
                    (n + 1) ** 2,
                )
                j = i + 1 + int(np.argmin(choices))
        following[i] = j
        if not have_ownership:
            scores[i] = scores[j] + reward[j]
        else:
            scores[i] = scores[j] + (min(reward[j], 0.0) if i < j <= white_end[i] else reward[j])
        pieces[i] = pieces[j] + 1
    result = []
    i = 0
    while i < n:
        j = following[i]
        result.append((cuts[i], cuts[j]))
        i = j
    return result


def split_spans(
    text: str,
    probabilities: npt.ArrayLike,
    *,
    max_length: int,
    length_fn: LengthFunction = len,
    legal_positions: Sequence[int] | None = None,
    threshold: float = 0.5,
) -> list[Span]:
    """Plan evidence-driven cuts and verify every actual emitted source slice.

    Measure legal atoms, maximize logit(p)-logit(threshold) on their summed
    counts, then verify and tighten any over-cap piece. If measured context is
    cheaper than the sum, re-plan once with that ratio and verify again. Finally
    remove feasible nonpositive cuts from lowest evidence upward, verifying
    every merge. Exact for additive counters; others guarantee the cap and
    evidence-driven planning, not global optimality or optimal fill. Fixed
    TokenLength overhead is counted once per piece, not once per atom.
    Callbacks must be deterministic nonnegative counts.

    At most 10A counter calls for A legal atoms, plus 2R default-lattice probes
    for R non-whitespace runs (none with explicit legal_positions), and one
    TokenLength overhead probe. Source-span caching reuses identical probes.
    Each of at most two plans verifies at most 3A new slices: two tighter levels
    precede atomic fallback. The merge heap schedules at most 3A probes and
    each successful merge removes a piece, so recovery always terminates.

    Discrete ties at 500k can take several seconds on comps0. Timing tests
    use a shared 10-second ceiling to catch growth regressions, not as a
    universal latency target on a shared host.

    Absolute 1e-9 ties at each suffix state prefer fewer pieces then longest
    prefix. At cumulative magnitudes above about 4e6, float64 spacing can exceed
    that tolerance; rounding then limits score resolution. Ordinary sentence
    cuts are irredundant, not necessarily the fewest. position on impossibility
    is the leftmost over-cap legal atom, independent of evidence and threshold.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string; received {type(text).__name__}")
    cap, count = resolve_budget(max_length, length_fn)
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError(f"threshold must be a real number in [0,1]; received {threshold!r}")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be finite in [0,1]; received {threshold!r}")
    try:
        p = _probabilities(text, probabilities)
    except (TypeError, ValueError) as error:
        raise ValueError("probabilities must have len(text) finite numeric values in [0,1]") from error
    ends = grapheme_ends(text)
    if legal_positions is not None:
        try:
            supplied = list(legal_positions)
        except TypeError as error:
            raise TypeError("legal_positions must be a sequence of exclusive grapheme ends") from error
        allowed = set(ends)
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x not in allowed for x in supplied):
            raise ValueError("legal_positions must be exclusive grapheme ends in the text")
        cuts = sorted({0, len(text), *(int(x) for x in supplied)})
    else:
        cuts = []
    if not text:
        return []
    cache: dict[Span, int] = {}

    def measured(a: int, b: int) -> int:
        if count is len:
            return b - a
        key = (a, b)
        if key not in cache:
            cache[key] = measure(text[a:b], count)
        return cache[key]

    if legal_positions is None:
        cuts = _legal_cuts(text, ends, cap, measured)
    cut_array = np.asarray(cuts, dtype=np.int64)
    interior = np.zeros(len(cuts), dtype=bool)
    white_end = np.arange(len(cuts), dtype=np.int64)
    white_starts = np.zeros(len(cuts), dtype=bool)
    if any(c.isspace() for c in text):
        clusters = [0, *ends]
        flags = [any(c.isspace() for c in text[a:b]) for a, b in zip(clusters, clusters[1:])]
        k = 0
        while k < len(flags):
            if not flags[k]:
                k += 1
                continue
            j = k + 1
            while j < len(flags) and flags[j]:
                j += 1
            a, b = clusters[k], clusters[j]
            lo, hi = np.searchsorted(cut_array, [a, b])
            white_starts[lo:hi] = True
            white_end[lo:hi] = hi if hi < len(cuts) and cuts[int(hi)] == b else -hi
            # A run-start cut also detaches trailing whitespace. It is legal
            # only when the current piece cannot hold the entire run.
            interior[lo:hi] = True
            k = j
    ownership = (interior, white_end) if interior.any() or np.any(white_end > np.arange(len(cuts))) else None
    raw = [measured(a, b) for a, b in zip(cuts, cuts[1:])]
    for i, value in enumerate(raw):
        if value > cap:
            raise UnsplittableUnitError(cuts[i], value, cap)
    overhead = count.overhead if isinstance(count, TokenLength) else 0
    weights = [max(0, value - overhead) for value in raw]
    total = sum(weights)
    prefix = np.asarray([0, *accumulate(weights)], dtype=np.int64 if total <= np.iinfo(np.int64).max else object)
    indices = evidence_indices(text, cuts[1:])
    q = np.clip(np.where(indices >= 0, p[np.maximum(indices, 0)], 0.0), 1e-12, 1 - 1e-12)
    theta = float(np.clip(threshold, 1e-12, 1 - 1e-12))
    rewards = np.r_[0.0, np.log(q) - np.log1p(-q) - (np.log(theta) - np.log1p(-theta))]
    # Cap-forced whitespace cuts retain only the nonpositive boundary cost;
    # a free cut must not reward shifting a word to detach its whitespace.
    rewards[white_starts] = np.minimum(rewards[white_starts], 0)
    rewards[interior] = np.minimum(rewards[interior], 0)
    rewards[-1] = 0
    base_budget = cap - overhead
    index = {value: i for i, value in enumerate(cuts)}

    def verified_plan(budget: int) -> list[Span]:
        pending: list[tuple[int, int, int, int | None, int]] = [(0, len(cuts) - 1, budget, None, 0)]
        output: list[Span] = []
        while pending:
            lo, hi, budget, atom_limit, depth = pending.pop()
            local_rewards = rewards[lo : hi + 1].copy()
            local_rewards[-1] = 0
            local_ownership = (
                None
                if ownership is None
                else (
                    interior[lo : hi + 1],
                    np.where(
                        white_end[lo : hi + 1] >= 0,
                        np.minimum(white_end[lo : hi + 1], hi) - lo,
                        -(np.minimum(-white_end[lo : hi + 1], hi) - lo),
                    ),
                )
            )
            proposed = _decode(
                cuts[lo : hi + 1], prefix[lo : hi + 1] - prefix[lo], local_rewards, budget, atom_limit, local_ownership
            )
            for a, b in proposed:
                value = measured(a, b)
                if value <= cap:
                    output.append((a, b))
                    continue
                x, y = index[a], index[b]
                if depth >= 2:
                    pending.append((x, y, base_budget, 1, depth + 1))
                    continue
                estimate = int(prefix[y] - prefix[x])
                largest = max(weights[x:y])
                tighter = min(budget - 1, int(estimate * cap / value))
                if largest <= tighter < estimate:
                    pending.append((x, y, tighter, None, depth + 1))
                else:
                    # Zero/one-positive-atom nonadditive estimates cannot force
                    # progress by lowering the scalar sum. Tighten the atom window.
                    pending.append((x, y, base_budget, max(1, (y - x) // 2), depth + 1))
        return sorted(output)

    output = verified_plan(base_budget)
    payload = sum(max(0, measured(a, b) - overhead) for a, b in output)
    if count is not len and len(output) > 1 and payload < total:
        # One measured context calibration. Planning still uses the same
        # evidence program; every larger proposal goes through true verification.
        relaxed = total if payload == 0 else max(base_budget + 1, base_budget * total // payload)
        candidate = verified_plan(relaxed)

        def objective(spans: list[Span]) -> tuple[float, int, tuple[int, ...]]:
            return (sum(float(rewards[index[b]]) for _, b in spans), -len(spans), tuple(b for _, b in spans))

        prior, proposed = objective(output), objective(candidate)
        if proposed[0] > prior[0] + _SCORE_TOLERANCE or (
            proposed[0] >= prior[0] - _SCORE_TOLERANCE and proposed[1:] > prior[1:]
        ):
            output = candidate
        output = _merge_verified(output, rewards, index, cap, measured)
    return output


def _merge_verified(
    spans: list[Span],
    rewards: npt.NDArray[np.float64],
    index: dict[int, int],
    cap: int,
    measured: Callable[[int, int], int],
) -> list[Span]:
    """Remove feasible nonpositive cuts in evidence order, at most 3A probes.

    Each successful merge deletes one piece and schedules at most two neighbors.
    The heap has O(A) entries and the linked source partition always advances.
    """
    n = len(spans)
    starts, ends = [a for a, _ in spans], [b for _, b in spans]
    previous, following = list(range(-1, n - 1)), list(range(1, n + 1))
    versions = [0] * n
    live = [True] * n
    heap: list[tuple[float, int, int, int, int, int]] = []

    def schedule(i: int) -> None:
        if i < 0 or not live[i]:
            return
        j = following[i]
        if j == n:
            return
        reward = float(rewards[index[ends[i]]])
        if reward <= 0:
            heappush(heap, (reward, ends[i], i, j, versions[i], versions[j]))

    for i in range(n - 1):
        schedule(i)
    while heap:
        _, _, i, j, vi, vj = heappop(heap)
        if not live[i] or not live[j] or versions[i] != vi or versions[j] != vj:
            continue
        if measured(starts[i], ends[j]) > cap:
            continue
        ends[i] = ends[j]
        following[i] = following[j]
        if following[j] < n:
            previous[following[j]] = i
        live[j] = False
        versions[i] += 1
        schedule(previous[i])
        schedule(i)
    return [(starts[i], ends[i]) for i in range(n) if live[i]]


def chunk_spans(
    text: str,
    probabilities: npt.ArrayLike,
    units: Sequence[Span],
    *,
    max_length: int,
    length_fn: LengthFunction = len,
    sentence_spans: Sequence[Span] | None = None,
    overlap_sentences: int = 0,
    threshold: float = 0.5,
) -> list[Span]:
    """Next-fit packing, with verified counts and up to k trailing whole sentences.

    Sentence suffix lookup uses bisect, never a scan over earlier chunks. No
    final re-count pass: a chunk is emitted only after its actual slice fits.
    """
    resolve_budget(max_length, length_fn)
    validate_overlap(overlap_sentences)
    ends = set(grapheme_ends(text)) | {0}

    def count_span(a: int, b: int) -> int:
        return b - a if length_fn is len else measure(text[a:b], length_fn)

    sentences = list(units if sentence_spans is None else sentence_spans)
    for label, spans in [("units", units), ("sentence_spans", sentences)]:
        cursor = 0
        for a, b in spans:
            if a != cursor or b <= a or a not in ends or b not in ends:
                raise ValueError(f"{label} must partition text at grapheme boundaries")
            cursor = b
        if cursor != len(text):
            raise ValueError(f"{label} must partition the entire text")
    p = _probabilities(text, probabilities)
    fragments: list[Span] = []
    for a, b in units:
        if count_span(a, b) <= max_length:
            fragments.append((a, b))
        else:
            try:
                fragments.extend(
                    (a + x, a + y)
                    for x, y in split_spans(
                        text[a:b], p[a:b], max_length=max_length, length_fn=length_fn, threshold=threshold
                    )
                )
            except UnsplittableUnitError as error:
                raise UnsplittableUnitError(a + error.position, error.measured_length, max_length) from error
    result: list[Span] = []
    sentence_ends = [b for _, b in sentences]
    start = end = 0
    for a, b in fragments:
        if end > start and count_span(start, b) > max_length:
            result.append((start, end))
            index = bisect_right(sentence_ends, end)
            trailing = []
            cursor = end
            for j in range(index - 1, max(-1, index - overlap_sentences - 1), -1):
                x, y = sentences[j]
                if y != cursor or x < start:
                    break
                trailing.append(x)
                cursor = x
            start = a
            for x in reversed(trailing):
                if count_span(x, b) <= max_length:
                    start = x
                    break
        end = b
    if end > start:
        result.append((start, end))
    return result
