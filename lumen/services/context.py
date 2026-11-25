"""Shared helpers for the session services: time stamps, CPU gate, caches."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Hashable, Iterator, TypeVar

from ..errors import LimitExceededError, OperationCancelled, OperationTimeout

T = TypeVar("T")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CpuGate:
    """Bounds the number of concurrent CPU-heavy computations in this process.

    Vision work holds the GIL only partly (NumPy/OpenCV release it), so a small
    number of parallel jobs is useful; an unbounded number just thrashes a
    worker and starves the health endpoint.
    """

    def __init__(self, slots: int) -> None:
        self._sem = threading.BoundedSemaphore(max(1, slots))

    @contextlib.contextmanager
    def slot(self, wait_seconds: float) -> Iterator[None]:
        if not self._sem.acquire(timeout=max(wait_seconds, 0.0)):
            raise LimitExceededError(
                "server_busy",
                "The server is busy with other measurements. Try again in a moment.",
                http_status=503,
            )
        try:
            yield
        finally:
            self._sem.release()


class Deadline:
    """Cooperative time budget and cancellation flag checked between stages."""

    def __init__(self, seconds: float, cancelled: Callable[[], bool] | None = None) -> None:
        self._end = time.monotonic() + seconds
        self._seconds = seconds
        self._cancelled = cancelled

    def check(self, stage: str = "") -> None:
        if self._cancelled is not None and self._cancelled():
            raise OperationCancelled("cancelled", "The operation was cancelled.")
        if time.monotonic() > self._end:
            raise OperationTimeout(
                "timeout",
                f"The computation exceeded its {self._seconds:.0f} s budget" + (f" during {stage}." if stage else "."),
                suggestions=["Use fewer or smaller frames, or retry."],
            )

    @property
    def remaining(self) -> float:
        return max(self._end - time.monotonic(), 0.0)


class LruCache:
    """Tiny thread-safe LRU for decoded images and solver results."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._d: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self._cap:
                self._d.popitem(last=False)

    def drop_prefix(self, prefix: Hashable) -> None:
        with self._lock:
            for k in [k for k in self._d if isinstance(k, tuple) and k and k[0] == prefix]:
                del self._d[k]

    def get_or_compute(self, key: Hashable, fn: Callable[[], T]) -> T:
        v = self.get(key)
        if v is None:
            v = fn()
            self.put(key, v)
        return v
