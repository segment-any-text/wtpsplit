"""Bounded ordered CPU preparation with one owner of mutable tokenizer caches."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

_T = TypeVar('_T')


def _prefetch_ordered(prepare: Callable[[int], _T], count: int) -> Generator[_T, None, None]:
    """Yield chunks in order while preparing up to two ahead on a single thread.

    A single worker preserves tokenizer cache update order. At most two pending
    results plus the consumer's current chunk are retained; closing the iterator
    cancels queued work and joins any running preparation before returning.
    """
    if count <= 1:
        for index in range(count):
            yield prepare(index)
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque(executor.submit(prepare, i) for i in range(min(2, count)))
        next_index = len(pending)
        try:
            while pending:
                result = pending.popleft().result()
                if next_index < count:
                    pending.append(executor.submit(prepare, next_index))
                    next_index += 1
                yield result
        finally:
            for future in pending:
                future.cancel()


_R = TypeVar("_R")


def _overlap_ordered(
    items: Iterable[_T], dispatch: Callable[[_T], Callable[[], _R]],
) -> Generator[tuple[_T, _R], None, None]:
    """Dispatch on the caller and finish on one worker, one chunk ahead."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending: tuple[_T, Future[_R]] | None = None
        for item in items:
            finish = dispatch(item)
            next_pending = item, executor.submit(finish)
            if pending is not None:
                previous, result = pending
                yield previous, result.result()
            pending = next_pending
        if pending is not None:
            previous, result = pending
            yield previous, result.result()
