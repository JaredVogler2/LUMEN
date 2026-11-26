"""The carrier: one address through which a browser sends a request together with the job it belongs to.

``POST api/carry`` takes an envelope (``storage/carried.py``): the request that is meant (method, path, body), the
state of the job and whatever pictures of it the step needs.  The request that is meant is then served by the
ordinary routes, in this process, with a store that holds exactly what came.  The answer goes back in an envelope
too, with the state as the request left it and any picture that was made here.

Nothing else in the application knows about this: routes, services and the measurement code are the same whether
the server keeps a job or the phone carries it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from ..config import Settings
from ..errors import LumenError
from ..storage.base import SessionStore
from ..storage.carried import CarriedStore, begin, end, read_request, write_response

CARRY_PATH = "/api/carry"
#: the envelope is bytes and is declared as nothing more: every platform and proxy passes that type untouched
CARRY_TYPE = "application/octet-stream"
#: what may be asked for through the carrier: the routes of a session or of a calibration workspace, nothing else
_INNER_PATH = re.compile(r"^api/(sessions|calibration)(/[A-Za-z0-9_.\-]+)*$")
_INNER_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_DROP_HEADERS = {b"content-type", b"content-length", b"transfer-encoding", b"x-lumen-meta", b"accept-encoding"}

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


def _error(status: int, code: str, message: str, suggestions: list[str] | None = None, details: dict[str, Any] | None = None) -> tuple[int, bytes]:
    return status, json.dumps({"error": {"code": code, "message": message, "details": details or {}, "suggestions": suggestions or []}}).encode("utf-8")


class CarryMiddleware:
    def __init__(self, app: Any, *, store: SessionStore, settings: Settings) -> None:
        self.app = app
        self.store = store
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != CARRY_PATH:
            await self.app(scope, receive, send)
            return
        status, body, ctype = await self._serve(scope, receive)
        await send({
            "type": "http.response.start", "status": status,
            "headers": [
                (b"content-type", ctype.encode("latin-1")), (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"no-referrer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def _serve(self, scope: Scope, receive: Receive) -> tuple[int, bytes, str]:
        js = "application/json"
        if scope.get("method") != "POST":
            return (*_error(405, "http_405", "The carrier takes POST only."), js)
        if not isinstance(self.store, CarriedStore):
            return (*_error(409, "job_kept_by_server", "This server keeps the job itself: requests go to it directly.", ["Reload the page."]), js)
        lim = self.settings.limits
        # ---- the whole envelope, and not more than this host can take
        chunks: list[bytes] = []
        n = 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return (*_error(400, "client_gone", "The request was not completed."), js)
            chunk = msg.get("body", b"")
            n += len(chunk)
            if n > lim.carry_max_request_bytes:
                return (*_error(
                    413, "request_too_large",
                    f"The job is larger than one request may be on this host ({lim.carry_max_request_bytes / 1e6:.1f} MB).",
                    ["Use fewer photos, or remove one and analyse again."],
                ), js)
            chunks.append(chunk)
            if not msg.get("more_body"):
                break
        try:
            header, carried, body = read_request(b"".join(chunks), self.settings.state_secret)
        except LumenError as exc:
            return exc.http_status, json.dumps({"error": exc.to_dict()}).encode("utf-8"), js
        method = str(header.get("method", "")).upper()
        target = str(header.get("path", ""))
        path, _, query = target.partition("?")
        if method not in _INNER_METHODS or ".." in path or not _INNER_PATH.match(path) or len(query) > 4000:
            return (*_error(400, "not_carried", "This request cannot be sent with a job."), js)
        headers = [(k, v) for k, v in scope.get("headers", []) if k not in _DROP_HEADERS]
        ctype = str(header.get("content_type") or "")
        if ctype:
            headers.append((b"content-type", ctype.encode("latin-1", "replace")))
        headers.append((b"content-length", str(len(body)).encode("ascii")))
        meta = header.get("meta")
        if isinstance(meta, str) and meta and len(meta) < 8000:
            headers.append((b"x-lumen-meta", meta.encode("latin-1", "replace")))
        inner: Scope = dict(scope)
        inner.update({"method": method, "path": "/" + path, "raw_path": ("/" + path).encode("ascii"), "query_string": query.encode("ascii", "replace"), "headers": headers})

        sent = False

        async def inner_receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # nothing more will come but the disconnect

        got: dict[str, Any] = {"status": 500, "ctype": js, "parts": [], "size": 0, "headers": {}}

        async def inner_send(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                got["status"] = int(msg["status"])
                for k, v in msg.get("headers", []):
                    name = k.lower()
                    if name == b"content-type":
                        got["ctype"] = v.decode("latin-1")
                    elif name.startswith(b"x-lumen-"):
                        # what the application tells the browser beside the body (the geometry of a rectified picture)
                        got["headers"][name.decode("latin-1")] = v.decode("latin-1")
            elif msg["type"] == "http.response.body":
                part = msg.get("body", b"")
                got["size"] += len(part)
                if got["size"] <= 4 * lim.carry_max_response_bytes:
                    got["parts"].append(part)

        token = begin(carried)
        try:
            await self.app(inner, inner_receive, inner_send)
        finally:
            end(token)
        answer = b"".join(got["parts"])
        if got["status"] >= 400:
            # an error is an error: it goes back as it is, and nothing the request changed is taken over
            return got["status"], answer, got["ctype"]
        out = write_response(got["status"], got["ctype"], answer, carried, self.settings.state_secret, got["headers"])
        if got["size"] > 4 * lim.carry_max_response_bytes or len(out) > lim.carry_max_response_bytes:
            return (*_error(
                413, "answer_too_large",
                f"The answer is larger than one answer may be on this host ({lim.carry_max_response_bytes / 1e6:.1f} MB).",
                ["Use fewer or smaller photos."],
            ), js)
        return 200, out, CARRY_TYPE
