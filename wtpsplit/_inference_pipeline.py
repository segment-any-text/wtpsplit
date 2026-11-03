"""Bounded process staging with file-backed array transport and ordered output.

Each spawned worker owns its tokenizer and spec. No initializer globals, runtime
monkey patches or global switch intervals are used. Thread-limit environment
variables are set in spawned workers only; the parent environment is unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from wtpsplit._inference_stages import (
    Array,
    InferenceSpec,
    ModelBatch,
    RequestState,
    Tokenizer,
    TokenizerEmptyError,
    finish,
    prepare,
)

T = TypeVar("T")
_ALIGNMENT = 64


@dataclass(frozen=True)
class ArrayLayout:
    batch: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int


@dataclass(frozen=True)
class Transport:
    path: str
    layout: tuple[ArrayLayout, ...]
    count: int


def pack(batches: list[ModelBatch], directory: str) -> Transport:
    """Write one private, aligned file for all model inputs in a request."""
    layout: list[ArrayLayout] = []
    offset = 0
    for index, batch in enumerate(batches):
        for name, array in batch.items():
            if array.dtype.hasobject:
                raise ValueError("Object arrays cannot be transported")
            layout.append(ArrayLayout(index, name, array.dtype.str, array.shape, offset))
            offset += math_align(array.nbytes)
    fd, path = tempfile.mkstemp(prefix="request-", suffix=".bin", dir=directory)
    try:
        os.ftruncate(fd, max(offset, _ALIGNMENT))
    finally:
        os.close(fd)
    mapped = np.memmap(path, dtype=np.uint8, mode="r+", shape=(max(offset, _ALIGNMENT),))
    for entry, array in zip(layout, (a for batch in batches for a in batch.values()), strict=True):
        mapped[entry.offset : entry.offset + array.nbytes].view(array.dtype).reshape(array.shape)[...] = array
    mapped.flush()
    del mapped
    return Transport(path, tuple(layout), len(batches))


def math_align(size: int) -> int:
    return ((size + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT


def unpack(transport: Transport) -> list[ModelBatch]:
    """Map copy-on-write and unlink the private file; views retain the mapping."""
    mapped = np.memmap(transport.path, dtype=np.uint8, mode="c")
    os.unlink(transport.path)
    batches: list[ModelBatch] = [{} for _ in range(transport.count)]
    for entry in transport.layout:
        size = int(np.prod(entry.shape)) * np.dtype(entry.dtype).itemsize
        batches[entry.batch][entry.name] = (
            mapped[entry.offset : entry.offset + size].view(entry.dtype).reshape(entry.shape)
        )
    return batches


@dataclass(frozen=True)
class PrepareTask:
    sequence: int
    texts: list[str]


@dataclass(frozen=True)
class FinishTask:
    sequence: int
    state: bytes
    transport: Transport


@dataclass(frozen=True)
class PreparedTransport:
    state: bytes
    transport: Transport


@dataclass(frozen=True)
class StageResult:
    sequence: int
    value: PreparedTransport | list[NDArray[np.float32]] | None
    seconds: float
    error: str | None = None
    input_error: str | None = None


class _StageFuture(Future[StageResult]):
    """Drain results in the caller, avoiding a competing Python reader thread."""

    def __init__(self, pool: StagedPool) -> None:
        super().__init__()
        self._pool = pool

    def done(self) -> bool:
        self._pool._receive(0.0)
        return super().done()

    def result(self, timeout: float | None = None) -> StageResult:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not super().done():
            remaining = 0.2 if deadline is None else min(0.2, max(0.0, deadline - time.monotonic()))
            self._pool._receive(remaining)
            if deadline is not None and time.monotonic() >= deadline:
                break
        return super().result(timeout=0)


@dataclass
class _Worker:
    tokenizer: Tokenizer
    spec: InferenceSpec
    directory: str
    tasks: Queue[PrepareTask | FinishTask | None]
    results: Queue[StageResult]

    def run(self) -> None:
        # Local runtime objects only: changing worker thread limits cannot affect
        # another inference session or the parent's Torch/Rayon configuration.
        import torch

        os.environ["RAYON_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(1)
        while True:
            task = self.tasks.get()
            if task is None:
                return
            start = time.perf_counter()
            try:
                if isinstance(task, PrepareTask):
                    prepared = prepare(task.texts, self.spec, self.tokenizer)
                    value: PreparedTransport | list[NDArray[np.float32]] | None = (
                        None
                        if prepared is None
                        else PreparedTransport(
                            pickle.dumps(prepared.state, protocol=5), pack(prepared.batches, self.directory)
                        )
                    )
                else:
                    # State is opaque in the caller: large tokenizer lists are
                    # decoded only by a worker, not repeatedly on the GPU thread.
                    state = pickle.loads(task.state)
                    if not isinstance(state, RequestState):
                        raise RuntimeError("Invalid internal inference state")
                    logits = [batch["logits"] for batch in unpack(task.transport)]
                    value = finish(state, logits, self.spec)
                self.results.put(StageResult(task.sequence, value, time.perf_counter() - start))
            except TokenizerEmptyError as error:
                self.results.put(StageResult(task.sequence, None, time.perf_counter() - start, input_error=str(error)))
            except Exception:
                self.results.put(StageResult(task.sequence, None, time.perf_counter() - start, traceback.format_exc()))


class StagedPool:
    """An explicitly closed pool; callers bound pending work through ``results``."""

    def __init__(
        self, tokenizer: Tokenizer, spec: InferenceSpec, workers: int, *, transport_root: str | None = None
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be positive")
        context = mp.get_context("spawn")
        root = transport_root
        if root is None and Path("/dev/shm").is_dir():
            root = "/dev/shm"
        self._directory = tempfile.TemporaryDirectory(prefix="wtpsplit-staging-", dir=root)
        self._tasks: Queue[PrepareTask | FinishTask | None] = context.Queue()
        self._results: Queue[StageResult] = context.Queue()
        self._pending: dict[int, Future[StageResult]] = {}
        self._lock = threading.Lock()
        self._failure: Exception | None = None
        self._sequence = 0
        self._closed = threading.Event()
        self._processes = [
            context.Process(
                target=_Worker(tokenizer, spec, self._directory.name, self._tasks, self._results).run, daemon=True
            )
            for _ in range(workers)
        ]
        try:
            for process in self._processes:
                process.start()
        except BaseException:
            self.close()
            raise

    def _receive(self, wait: float) -> None:
        while not self._closed.is_set():
            try:
                result = self._results.get(timeout=wait)
            except queue.Empty:
                if any(p.exitcode is not None for p in self._processes):
                    self._fail(RuntimeError("Inference worker exited before completion"))
                return
            with self._lock:
                future = self._pending.pop(result.sequence, None)
            if future is not None:
                if result.input_error is not None:
                    future.set_exception(TokenizerEmptyError(f"Request {result.sequence}: {result.input_error}"))
                elif result.error is not None:
                    future.set_exception(RuntimeError("Inference stage failed:\n" + result.error))
                else:
                    future.set_result(result)
            wait = 0.0  # drain available results without waiting for another task

    def _fail(self, error: Exception) -> None:
        with self._lock:
            self._failure = error
            for future in self._pending.values():
                if not Future.done(future):
                    future.set_exception(error)
            self._pending.clear()

    def _submit(self, build: Callable[[int], PrepareTask | FinishTask]) -> Future[StageResult]:
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("Inference pool is closed")
            if self._failure is not None:
                raise RuntimeError("Inference pool failed") from self._failure
            sequence = self._sequence
            self._sequence += 1
            future: Future[StageResult] = _StageFuture(self)
            self._pending[sequence] = future
        try:
            self._tasks.put(build(sequence))
        except BaseException:
            with self._lock:
                self._pending.pop(sequence, None)
            raise
        return future

    def prepare(self, texts: list[str]) -> Future[StageResult]:
        return self._submit(lambda sequence: PrepareTask(sequence, texts))

    def finish(self, state: bytes, logits: list[Array]) -> Future[StageResult]:
        return self._submit(
            lambda sequence: FinishTask(
                sequence, state, pack([dict(logits=value) for value in logits], self._directory.name)
            )
        )

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._fail(RuntimeError("Inference pool closed"))
        for process in self._processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        self._tasks.cancel_join_thread()
        self._tasks.close()
        self._results.cancel_join_thread()
        self._results.close()
        self._directory.cleanup()

    def __enter__(self) -> StagedPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class InferenceTiming:
    """Request timings distinguish forward submission and final CPU drain."""

    request_wall_seconds: float
    prepare_worker_seconds: float
    finish_worker_seconds: float
    model_inference_seconds: float
    input_wait_seconds: float
    tokens: int | None
    fallback: bool


def results(
    items: Iterable[tuple[T, list[str]]],
    pool: StagedPool,
    forward: Callable[[list[ModelBatch]], list[Array]],
    fallback: Callable[[list[str]], list[NDArray[np.float32]]],
    *,
    ahead: int = 6,
    max_unfinished: int = 8,
) -> Iterator[tuple[T, list[NDArray[np.float32]], InferenceTiming]]:
    """Overlap CPU work and forward calls, yielding strictly in source order.

    The number of prepared requests and unfinished reductions is bounded. Source
    IO should be supplied by a process-backed iterator for CPU-heavy readers.
    """
    if ahead < 1 or max_unfinished < 1:
        raise ValueError("Pipeline bounds must be positive")
    source = iter(items)
    prepared: deque[tuple[T, list[str], Future[StageResult]]] = deque()
    unfinished: deque[tuple[T, Future[StageResult], tuple[float, float, float, float, int | None, bool]]] = deque()
    exhausted = False
    mark = time.monotonic()
    while prepared or not exhausted:
        while not exhausted and len(prepared) < ahead:
            try:
                meta, texts = next(source)
            except StopIteration:
                exhausted = True
                break
            prepared.append((meta, texts, pool.prepare(texts)))
        if not prepared:
            break
        meta, texts, future = prepared.popleft()
        wait_start = time.monotonic()
        stage = future.result()
        input_wait = time.monotonic() - wait_start
        forward_start = time.monotonic()
        if stage.value is None:
            values = fallback(texts)
            complete: Future[StageResult] = Future()
            complete.set_result(StageResult(-1, values, 0.0))
            tokens: int | None = None
        elif isinstance(stage.value, PreparedTransport):
            batches = unpack(stage.value.transport)
            tokens = sum(batch["input_ids"].size for batch in batches)
            logits = forward(batches)
            complete = pool.finish(stage.value.state, logits)
        else:
            raise RuntimeError("Unexpected preparation result")
        model_seconds = time.monotonic() - forward_start
        now = time.monotonic()
        unfinished.append(
            (meta, complete, (now - mark, stage.seconds, model_seconds, input_wait, tokens, stage.value is None))
        )
        mark = now
        while unfinished and (len(unfinished) > max_unfinished or unfinished[0][1].done()):
            entry_meta, handle, timing = unfinished.popleft()
            yield _complete(entry_meta, handle, timing)
    while unfinished:
        entry_meta, handle, timing = unfinished.popleft()
        yield _complete(entry_meta, handle, timing)


def _complete(
    meta: T, handle: Future[StageResult], timing: tuple[float, float, float, float, int | None, bool]
) -> tuple[T, list[NDArray[np.float32]], InferenceTiming]:
    stage = handle.result()
    if not isinstance(stage.value, list):
        raise RuntimeError("Unexpected reduction result")
    wall, prepare_seconds, forward_seconds, input_wait, tokens, fallback = timing
    return (
        meta,
        stage.value,
        InferenceTiming(wall, prepare_seconds, stage.seconds, forward_seconds, input_wait, tokens, fallback),
    )
