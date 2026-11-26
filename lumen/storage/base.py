"""Session storage adapters.

A session is one JSON document (``state``) plus named binary blobs (the frames'
image bytes).  Two adapters ship with LUMEN:

``FileSessionStore`` (default)
    One directory per session under ``LUMEN_DATA_DIR`` (default: a
    per-user directory in the system temp dir).  Writes are atomic and guarded
    by an inter-process file lock, so several worker processes of the same
    host — which is how Posit Connect scales a content item — can serve the
    same session.  It is a *cache with a time-to-live*, not durable storage:
    Connect may replace the directory on redeploy or reboot.

``MemorySessionStore``
    Process-local; for tests and single-process development only.

Hosts without a shared filesystem (several Connect nodes, off-host execution)
need either sticky routing or a shared-volume ``LUMEN_DATA_DIR``; a
database/object-store adapter only has to implement :class:`SessionStore`.

Images are never kept beyond the session: deleting a session, or its expiry,
removes every blob.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from ..errors import LimitExceededError, NotFoundError

_SID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_BLOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


def new_session_id() -> str:
    """128 bits of randomness, URL-safe; the id is the only capability needed
    to read a session, so it must be unguessable."""
    return secrets.token_urlsafe(18)


def check_session_id(sid: str) -> str:
    if not isinstance(sid, str) or not _SID_RE.match(sid):
        raise NotFoundError("session_not_found", "Unknown session.")
    return sid


def check_blob_name(name: str) -> str:
    if not isinstance(name, str) or not _BLOB_RE.match(name) or ".." in name:
        raise NotFoundError("blob_not_found", "Unknown resource.")
    return name


class SessionStore(ABC):
    """Interface every storage adapter implements."""

    @abstractmethod
    def create(self, sid: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def load(self, sid: str) -> dict[str, Any]: ...

    @abstractmethod
    def save(self, sid: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def delete(self, sid: str) -> bool: ...

    @abstractmethod
    def exists(self, sid: str) -> bool: ...

    @abstractmethod
    def put_blob(self, sid: str, name: str, data: bytes) -> None: ...

    @abstractmethod
    def get_blob(self, sid: str, name: str) -> bytes: ...

    @abstractmethod
    def delete_blob(self, sid: str, name: str) -> None: ...

    @abstractmethod
    def list_ids(self) -> list[tuple[str, float]]:
        """``(session id, last-modified unix time)`` for every stored session."""

    @abstractmethod
    def lock(self, sid: str) -> contextlib.AbstractContextManager[None]:
        """Exclusive lock for read-modify-write cycles on one session."""

    # ------------------------------------------------------------ helpers
    def purge_expired(self, ttl_seconds: float, now: float | None = None) -> int:
        now = time.time() if now is None else now
        n = 0
        for sid, mtime in self.list_ids():
            if now - mtime > ttl_seconds:
                if self.delete(sid):
                    n += 1
        return n

    def enforce_capacity(self, max_sessions: int) -> None:
        """Refuse new sessions when full (after expiry has been applied)."""
        if len(self.list_ids()) >= max_sessions:
            raise LimitExceededError(
                "too_many_sessions",
                "The server is at its session limit. Finish or delete another session, or retry later.",
                http_status=429,
            )


class MemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._states: dict[str, str] = {}
        self._blobs: dict[str, dict[str, bytes]] = {}
        self._mtime: dict[str, float] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.RLock()

    def create(self, sid: str, state: dict[str, Any]) -> None:
        check_session_id(sid)
        with self._guard:
            self._states[sid] = json.dumps(state)
            self._blobs[sid] = {}
            self._mtime[sid] = time.time()
            self._locks[sid] = threading.RLock()

    def load(self, sid: str) -> dict[str, Any]:
        check_session_id(sid)
        with self._guard:
            if sid not in self._states:
                raise NotFoundError("session_not_found", "Unknown or expired session.")
            return json.loads(self._states[sid])

    def save(self, sid: str, state: dict[str, Any]) -> None:
        check_session_id(sid)
        with self._guard:
            if sid not in self._states:
                raise NotFoundError("session_not_found", "Unknown or expired session.")
            self._states[sid] = json.dumps(state)
            self._mtime[sid] = time.time()

    def delete(self, sid: str) -> bool:
        with self._guard:
            existed = sid in self._states
            self._states.pop(sid, None)
            self._blobs.pop(sid, None)
            self._mtime.pop(sid, None)
            self._locks.pop(sid, None)
            return existed

    def exists(self, sid: str) -> bool:
        with self._guard:
            return sid in self._states

    def put_blob(self, sid: str, name: str, data: bytes) -> None:
        check_blob_name(name)
        with self._guard:
            if sid not in self._blobs:
                raise NotFoundError("session_not_found", "Unknown or expired session.")
            self._blobs[sid][name] = bytes(data)
            self._mtime[sid] = time.time()

    def get_blob(self, sid: str, name: str) -> bytes:
        check_blob_name(name)
        with self._guard:
            try:
                return self._blobs[sid][name]
            except KeyError:
                raise NotFoundError("blob_not_found", "Unknown resource.") from None

    def delete_blob(self, sid: str, name: str) -> None:
        with self._guard:
            self._blobs.get(sid, {}).pop(name, None)

    def list_ids(self) -> list[tuple[str, float]]:
        with self._guard:
            return list(self._mtime.items())

    @contextlib.contextmanager
    def lock(self, sid: str) -> Iterator[None]:
        with self._guard:
            lk = self._locks.setdefault(sid, threading.RLock())
        with lk:
            yield


class FileSessionStore(SessionStore):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.root, 0o700)
        self._thread_locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def _dir(self, sid: str) -> Path:
        return self.root / check_session_id(sid)

    def _state_path(self, sid: str) -> Path:
        return self._dir(sid) / "state.json"

    def create(self, sid: str, state: dict[str, Any]) -> None:
        d = self._dir(sid)
        d.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._write(sid, state)

    def _write(self, sid: str, state: dict[str, Any]) -> None:
        path = self._state_path(sid)
        tmp = path.with_name(f"state.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, separators=(",", ":"))
            fh.flush()
        os.replace(tmp, path)

    def load(self, sid: str) -> dict[str, Any]:
        path = self._state_path(sid)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            raise NotFoundError("session_not_found", "Unknown or expired session.") from None

    def save(self, sid: str, state: dict[str, Any]) -> None:
        if not self._dir(sid).is_dir():
            raise NotFoundError("session_not_found", "Unknown or expired session.")
        self._write(sid, state)

    def delete(self, sid: str) -> bool:
        try:
            d = self._dir(sid)
        except NotFoundError:
            return False
        if not d.is_dir():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    def exists(self, sid: str) -> bool:
        try:
            return self._state_path(sid).is_file()
        except NotFoundError:
            return False

    def put_blob(self, sid: str, name: str, data: bytes) -> None:
        d = self._dir(sid)
        if not d.is_dir():
            raise NotFoundError("session_not_found", "Unknown or expired session.")
        path = d / check_blob_name(name)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)

    def get_blob(self, sid: str, name: str) -> bytes:
        path = self._dir(sid) / check_blob_name(name)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise NotFoundError("blob_not_found", "Unknown resource.") from None

    def delete_blob(self, sid: str, name: str) -> None:
        with contextlib.suppress(FileNotFoundError, NotFoundError):
            (self._dir(sid) / check_blob_name(name)).unlink()

    def list_ids(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        try:
            entries = list(os.scandir(self.root))
        except FileNotFoundError:
            return out
        for e in entries:
            if e.is_dir() and _SID_RE.match(e.name):
                try:
                    st = os.stat(os.path.join(e.path, "state.json"))
                    out.append((e.name, st.st_mtime))
                except FileNotFoundError:
                    # half-created or half-deleted directory: treat as stale
                    out.append((e.name, e.stat().st_mtime))
        return out

    @contextlib.contextmanager
    def lock(self, sid: str) -> Iterator[None]:
        d = self._dir(sid)
        with self._guard:
            tl = self._thread_locks.setdefault(sid, threading.RLock())
        with tl:
            if not d.is_dir():
                raise NotFoundError("session_not_found", "Unknown or expired session.")
            fh = open(d / ".lock", "a+")
            try:
                try:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                except ImportError:  # pragma: no cover - non-POSIX hosts
                    pass
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover
                    pass
                fh.close()


_STORE: SessionStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> SessionStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            from ..config import get_settings

            s = get_settings()
            if s.storage == "client":
                from .carried import CarriedStore

                _STORE = CarriedStore()
            else:
                _STORE = MemorySessionStore() if s.storage == "memory" else FileSessionStore(s.data_dir)
        return _STORE


def reset_store() -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = None
