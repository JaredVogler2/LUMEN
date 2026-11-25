"""Background jobs (video processing).

A job runs on a thread of the worker process that received the upload.  Its
state lives *in the session record*, not in process memory, so a status poll
may be answered by any worker process that shares the session store — which is
how Posit Connect runs a content item with more than one process.

A job whose process died stops updating its heartbeat; the next poll reports
it as failed (``worker_lost``) instead of leaving the client waiting forever.
"""

from __future__ import annotations

import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..errors import ConflictError, LumenError, NotFoundError, OperationCancelled
from .context import utcnow

ACTIVE = ("queued", "running")
STALE_AFTER_S = 45.0
MAX_FINISHED_KEPT = 6


class JobHandle:
    """What a running job uses to report progress and to notice cancellation."""

    def __init__(self, runner: "JobRunner", sid: str, jid: str) -> None:
        self._runner = runner
        self.sid = sid
        self.id = jid
        self._last_write = 0.0
        self._last_poll = 0.0
        self._cancel = False

    def progress(self, fraction: float, stage: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < 0.7:
            return
        self._last_write = now
        self._runner._update(self.sid, self.id, {"progress": float(max(0.0, min(1.0, fraction))), "stage": stage or None, "state": "running"})

    def cancelled(self) -> bool:
        if self._cancel:
            return True
        now = time.monotonic()
        if now - self._last_poll < 0.5:
            return False
        self._last_poll = now
        try:
            self._cancel = bool(self._runner.peek(self.sid, self.id).get("cancel_requested"))
        except LumenError:
            self._cancel = True  # the session is gone: stop working for it
        return self._cancel


class JobRunner:
    def __init__(self, load: Callable[[str], Any], mutate: Callable[..., Any], max_workers: int = 1) -> None:
        self._load = load
        self._mutate = mutate
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="lumen-job")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state
    def _update(self, sid: str, jid: str, patch: dict[str, Any]) -> None:
        def fn(s: Any) -> None:
            j = s.jobs.get(jid)
            if j is None:
                return
            for k, v in patch.items():
                if v is not None or k in ("result", "error"):
                    j[k] = v
            j["updated"] = utcnow()
            j["heartbeat"] = time.time()

        self._mutate(sid, fn)

    def peek(self, sid: str, jid: str) -> dict[str, Any]:
        s = self._load(sid)
        j = s.jobs.get(jid)
        if j is None:
            raise NotFoundError("job_not_found", "Unknown job.")
        return dict(j)

    def get(self, sid: str, jid: str) -> dict[str, Any]:
        j = self.peek(sid, jid)
        if j["state"] in ACTIVE and time.time() - float(j.get("heartbeat") or 0.0) > STALE_AFTER_S:
            err = {
                "code": "worker_lost", "message": "The server process working on this stopped before it finished.",
                "details": {}, "suggestions": ["Upload the video again."],
            }
            self._update(sid, jid, {"state": "failed", "error": err})
            j = self.peek(sid, jid)
        return self.public(j)

    @staticmethod
    def public(j: dict[str, Any]) -> dict[str, Any]:
        return {k: j.get(k) for k in ("id", "kind", "state", "stage", "progress", "created", "updated", "result", "error")}

    def cancel(self, sid: str, jid: str) -> dict[str, Any]:
        self.peek(sid, jid)
        self._update(sid, jid, {"cancel_requested": True})
        return self.get(sid, jid)

    # ------------------------------------------------------------------- run
    def submit(self, sid: str, kind: str, fn: Callable[[JobHandle], dict[str, Any]], *, exclusive: bool = True) -> dict[str, Any]:
        jid = "j" + secrets.token_hex(6)
        now = utcnow()
        job = {
            "id": jid, "kind": kind, "state": "queued", "stage": "Waiting", "progress": 0.0, "created": now, "updated": now,
            "heartbeat": time.time(), "result": None, "error": None, "cancel_requested": False,
        }

        def add(s: Any) -> None:
            if exclusive:
                for o in s.jobs.values():
                    if o.get("kind") == kind and o.get("state") in ACTIVE and time.time() - float(o.get("heartbeat") or 0.0) <= STALE_AFTER_S:
                        raise ConflictError("job_running", "A video of this session is already being processed.", suggestions=["Wait for it to finish or cancel it."])
            # keep the record small: drop old finished jobs
            done = sorted((o for o in s.jobs.values() if o.get("state") not in ACTIVE), key=lambda o: o.get("updated") or "")
            for o in done[: max(0, len(done) - MAX_FINISHED_KEPT + 1)]:
                s.jobs.pop(o["id"], None)
            s.jobs[jid] = job

        self._mutate(sid, add)
        handle = JobHandle(self, sid, jid)

        def run() -> None:
            try:
                self._update(sid, jid, {"state": "running", "stage": "Starting"})
                result = fn(handle)
                self._update(sid, jid, {"state": "done", "progress": 1.0, "stage": "Done", "result": result})
            except OperationCancelled:
                self._safe(sid, jid, {"state": "cancelled", "stage": "Cancelled"})
            except LumenError as exc:
                self._safe(sid, jid, {"state": "failed", "error": exc.to_dict()})
            except Exception as exc:  # unexpected: report without leaking internals
                traceback.print_exc()
                self._safe(sid, jid, {"state": "failed", "error": {"code": "internal_error", "message": f"Processing failed ({type(exc).__name__}).", "details": {}, "suggestions": ["Try again, or use photos instead."]}})

        self._pool.submit(run)
        return self.public(job)

    def _safe(self, sid: str, jid: str, patch: dict[str, Any]) -> None:
        try:
            self._update(sid, jid, patch)
        except LumenError:
            pass  # session deleted while the job was running

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
