"""Slow frozen AVL recurrence from 23e7ee42a and a cut-subset oracle."""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import itertools
from wtpsplit.chunking import UnsplittableUnitError

Span = tuple[int, int]
_SCORE_TOLERANCE = 1e-9


@dataclass(slots=True)
class _Node:
    key: tuple[float, int]
    pieces: int
    left: "_Node | None" = None
    right: "_Node | None" = None
    best: tuple[int, int] = (0, 0)
    height: int = 1


def _height(node: _Node | None) -> int:
    return node.height if node else 0


def _update(node: _Node) -> _Node:
    node.best = min(
        (node.pieces, -node.key[1]),
        node.left.best if node.left else (10**18, 0),
        node.right.best if node.right else (10**18, 0),
    )
    node.height = 1 + max(_height(node.left), _height(node.right))
    return node


def _rotate_left(node: _Node) -> _Node:
    child = node.right
    assert child is not None
    node.right = child.left
    child.left = _update(node)
    return _update(child)


def _rotate_right(node: _Node) -> _Node:
    child = node.left
    assert child is not None
    node.left = child.right
    child.right = _update(node)
    return _update(child)


def _balance(node: _Node) -> _Node:
    _update(node)
    if _height(node.left) - _height(node.right) > 1:
        assert node.left is not None
        if _height(node.left.right) > _height(node.left.left):
            node.left = _rotate_left(node.left)
        return _rotate_right(node)
    if _height(node.right) - _height(node.left) > 1:
        assert node.right is not None
        if _height(node.right.left) > _height(node.right.right):
            node.right = _rotate_right(node.right)
        return _rotate_left(node)
    return node


def _insert(root: _Node | None, node: _Node) -> _Node:
    if root is None:
        return _update(node)
    if node.key < root.key:
        root.left = _insert(root.left, node)
    else:
        root.right = _insert(root.right, node)
    return _balance(root)


def _erase(root: _Node | None, key: tuple[float, int]) -> _Node | None:
    if root is None:
        return None
    if key < root.key:
        root.left = _erase(root.left, key)
    elif key > root.key:
        root.right = _erase(root.right, key)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        successor = root.right
        while successor.left is not None:
            successor = successor.left
        root.key, root.pieces = successor.key, successor.pieces
        root.right = _erase(root.right, successor.key)
    return _balance(root)


def _best_above(root: _Node | None, floor: float) -> tuple[int, int]:
    if root is None:
        return (10**18, 0)
    if root.key[0] < floor:
        return _best_above(root.right, floor)
    return min(
        (root.pieces, -root.key[1]), root.right.best if root.right else (10**18, 0), _best_above(root.left, floor)
    )


def reference_decode(
    cuts: list[int], lengths: npt.NDArray[np.int64], rewards: npt.NDArray[np.float64], cap: int
) -> list[Span]:
    # Right-to-left DP. Sliding AVL score index: O(k log k), O(k) memory.
    # Range maximum then a score-suffix query gives exact tolerance / piece /
    # longest-prefix tie order without scanning character pairs.
    if len(rewards) <= 2 or (rewards[1] <= 0 and np.all(rewards[1:-1] == rewards[1])):
        # Uniform nonpositive evidence: optimum is exactly next-fit, including
        # both tie keys. Vectorized reachability avoids building a score index.
        furthest = np.searchsorted(lengths, lengths + cap, side="right") - 1
        result = []
        i = 0
        while i < len(cuts) - 1:
            j = int(furthest[i])
            if j <= i:
                raise UnsplittableUnitError(cuts[i], int(lengths[i + 1] - lengths[i]), cap)
            result.append((cuts[i], cuts[j]))
            i = j
        return result
    n = len(cuts) - 1
    scores = [0.0] * (n + 1)
    pieces = [0] * (n + 1)
    following = [0] * (n + 1)
    keys: list[tuple[float, int]] = [(0.0, 0)] * (n + 1)
    root: _Node | None = None
    right = n
    for i in range(n - 1, -1, -1):
        j = i + 1
        key = (scores[j] + float(rewards[j]), j)
        keys[j] = key
        root = _insert(root, _Node(key, pieces[j]))
        while right > i and lengths[right] - lengths[i] > cap:
            root = _erase(root, keys[right])
            right -= 1
        if root is None:
            raise UnsplittableUnitError(cuts[i], int(lengths[i + 1] - lengths[i]), cap)
        maximum = root
        while maximum.right:
            maximum = maximum.right
        _, negative_j = _best_above(root, maximum.key[0] - _SCORE_TOLERANCE)
        j = -negative_j
        following[i] = j
        scores[i] = scores[j] + float(rewards[j])
        pieces[i] = pieces[j] + 1
    result = []
    i = 0
    while i < n:
        j = following[i]
        result.append((cuts[i], cuts[j]))
        i = j
    return result


def oracle(text, p, cap, count=len, threshold=0.5):
    """Independent contract lattice, score and full-path tie enumeration."""
    import regex

    clusters = [m.span() for m in regex.finditer(r"\X", text)]
    ends = [b for _, b in clusters]
    legal = {len(text)} if text else set()
    white = [any(c.isspace() for c in text[a:b]) for a, b in clusters]
    for (a, b), flag in zip(clusters, white):
        if flag:
            legal.add(b)
    i = 0
    while i < len(clusters):
        if white[i]:
            i += 1
            continue
        j = i + 1
        while j < len(clusters) and not white[j]:
            j += 1
        a, b = clusters[i][0], clusters[j - 1][1]
        if count(text[a:b]) > cap:
            legal.update(ends[i:j])
        elif j < len(clusters) and count(text[a : clusters[j][1]]) > cap:
            legal.add(b)
        i = j
    internal = sorted(legal - {len(text)})
    theta = np.clip(threshold, 1e-12, 1 - 1e-12)
    shift = np.log(theta) - np.log1p(-theta)

    def score(cuts):
        result = 0.0
        for start, e in zip(cuts, cuts[1:-1]):
            i = e - 1
            while i >= 0 and text[i].isspace():
                i -= 1
            q = np.clip(p[i] if i >= 0 else 0.0, 1e-12, 1 - 1e-12)
            value = np.log(q) - np.log1p(-q) - shift
            result += min(value, 0.0) if text[start:e].isspace() or (e < len(text) and text[e].isspace()) else value
        return result

    candidates = []
    for mask in itertools.product([False, True], repeat=len(internal)):
        cuts = [0, *[e for e, yes in zip(internal, mask) if yes], len(text)]
        valid = all(count(text[a:b]) <= cap for a, b in zip(cuts, cuts[1:]))
        for a, b in zip(cuts, cuts[1:-1]):
            if text[b].isspace():
                end = b
                while end < len(text) and text[end].isspace():
                    end += 1
                if count(text[a:end]) <= cap:
                    valid = False
        if valid:
            candidates.append((score(cuts), cuts))
    if not candidates:
        return None
    maximum = max(value for value, _ in candidates)
    tied = [cuts for value, cuts in candidates if maximum - value <= 1e-9]
    cuts = min(tied, key=lambda cuts: (len(cuts), tuple(-x for x in cuts[1:-1])))
    return list(zip(cuts, cuts[1:]))
