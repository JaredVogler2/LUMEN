"""What a browser does where it carries the job, in Python, for tests (the browser's own code is static/js/carry.js).

``StatelessHost`` is the host such a browser talks to, at its worst: every request is served by a newly made
application that has never seen another request, and a request or an answer above the platform's size is refused
as the platform refuses it.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import replace
from typing import Any, Callable, Iterator
from urllib.parse import urlencode

from lumen.storage.carried import pack, unpack

CARRY_TYPE = "application/octet-stream"


def carried_app(**settings_kw: Any) -> Any:
    """One application whose jobs are carried by the client (what one warm instance of a function is)."""
    from lumen.api.app import create_app
    from lumen.operations.library import OperationLibrary
    from lumen.services.sessions import SessionService
    from lumen.storage.carried import CarriedStore
    from tests.conftest import BUNDLED_OPS, make_settings

    settings = replace(make_settings(**settings_kw), storage="client")
    return create_app(settings, SessionService(CarriedStore(), settings, OperationLibrary(BUNDLED_OPS)))


def warm_client(**settings_kw: Any) -> Iterator["CarryClient"]:
    """A client that carries its jobs, talking to one application that stays (a warm instance)."""
    from fastapi.testclient import TestClient

    with TestClient(carried_app(**settings_kw)) as http:
        yield CarryClient(http)


def forgetful_client(**settings_kw: Any) -> Iterator["CarryClient"]:
    """A client that carries its jobs, talking to a host that serves every request with a new application."""
    from fastapi.testclient import TestClient

    host = StatelessHost(lambda: carried_app(**settings_kw))
    with TestClient(host) as http:
        c = CarryClient(http)
        c.host = host
        yield c


class StatelessHost:
    """ASGI application: a new application for every request, and the size limits of a function platform."""

    def __init__(self, make_app: Callable[[], Any], max_body: int = 4_500_000) -> None:
        self.make_app = make_app
        self.max_body = max_body
        self.requests = 0
        self.largest_request = 0
        self.largest_answer = 0

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.make_app()(scope, receive, send)
            return
        self.requests += 1
        chunks: list[bytes] = []
        n = 0
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                break
            n += len(msg.get("body", b""))
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        self.largest_request = max(self.largest_request, n)
        if n > self.max_body:
            await self._refuse(send)
            return
        body = b"".join(chunks)
        sent = False

        async def replay() -> dict:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        start: dict | None = None
        parts: list[bytes] = []

        async def hold(msg: dict) -> None:
            nonlocal start
            if msg["type"] == "http.response.start":
                start = msg
            elif msg["type"] == "http.response.body":
                parts.append(msg.get("body", b""))

        await self.make_app()(scope, replay, hold)
        answer = b"".join(parts)
        self.largest_answer = max(self.largest_answer, len(answer))
        if len(answer) > self.max_body or start is None:
            await self._refuse(send)
            return
        await send(start)
        await send({"type": "http.response.body", "body": answer})

    @staticmethod
    async def _refuse(send: Any) -> None:
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"FUNCTION_PAYLOAD_TOO_LARGE"})


class Answer:
    """What a test reads from a response (the part of httpx's that the tests use)."""

    def __init__(self, status: int, content_type: str, content: bytes, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.content_type = content_type
        self.content = content
        self.headers = {"content-type": content_type, **(headers or {})}

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")


class CarryClient:
    """Keeps jobs as a browser does (state as the bytes the server wrote, pictures by name) and sends them along."""

    def __init__(self, http: Any, prefix: str = "", deflate: bool = True) -> None:
        self.http = http            # a TestClient / httpx client
        self.prefix = prefix
        self.deflate = deflate
        self.states: dict[str, tuple[bytes, str]] = {}      # id -> (state bytes, signature)
        self.blobs: dict[str, dict[str, bytes]] = {}        # id -> name -> picture
        self.sent_pictures = 0
        self.round_trips = 0
        self.asked_again = 0
        self.host: StatelessHost | None = None

    # ------------------------------------------------------------ requests
    def request(
        self, method: str, path: str, *, json_body: Any = None, content: bytes | None = None, content_type: str = "", meta: dict | None = None,
        raw_meta: str | None = None, attach: list[str] | None = None, tamper: Callable[[dict, list[bytes]], None] | None = None,
    ) -> Answer:
        sid = self._id_of(path)
        body = b""
        if json_body is not None:
            body, content_type = json.dumps(json_body).encode("utf-8"), "application/json"
        elif content is not None:
            body = content
        # the pictures a step is known to look at go with the first request (static/js/carry.js picturesFor)
        names = list(attach) if attach is not None else self._pictures_for(method, path, sid, json_body)
        if meta is not None:
            from urllib.parse import quote

            raw_meta = quote(json.dumps(meta))
        for _ in range(12):
            out = self._send(method, path, sid, body, content_type, raw_meta, names, tamper)
            if out.status_code == 428:
                err = out.json()["error"]
                assert err["code"] == "pictures_needed", err
                # a step that was not known to need a picture names it: that one is sent, and the step asked again
                self.asked_again += 1
                want = [n for n in err["details"]["blobs"] if n in self.blobs.get(err["details"]["id"], {}) and n not in names]
                assert want, f"asked for {err['details']['blobs']} which the client does not hold or has sent"
                names = names + want
                continue
            return out
        raise AssertionError("the server kept asking for pictures")

    _ALL_PHOTOS = ("/datums/propose", "/datums/select", "/datums/floor", "/datums/side", "/calibration", "/research/photometric")

    def _pictures_for(self, method: str, path: str, sid: str | None, json_body: Any) -> list[str]:
        if sid is None or sid not in self.states:
            return []
        p = path.strip("/").split("?")[0]
        doc = self.state(sid)
        have = self.blobs.get(sid, {})
        names: list[str] = []
        frames = doc.get("frames") or []
        if p.startswith("api/sessions/"):
            if method not in ("GET", "DELETE") and p.endswith(self._ALL_PHOTOS):
                names += [f["blob"] for f in frames if f["role"] == "registration"]
            parts = p.split("/")
            if len(parts) == 6 and parts[3] == "frames" and parts[5] in ("rectified", "image"):
                names += [f["blob"] for f in frames if f["id"] == parts[4]]
            if method == "POST" and p.endswith("/verify") and isinstance(json_body, dict):
                names += [f["blob"] for f in frames if f["id"] == json_body.get("frame_id")]
        else:
            parts = p.split("/")
            if len(parts) == 6 and parts[3] == "photos" and parts[5] == "image":
                names += [v["blob"] for v in doc.get("views") or [] if v["observation"]["view_id"] == parts[4]]
        return [n for n in dict.fromkeys(names) if n in have]

    def _send(self, method: str, path: str, sid: str | None, body: bytes, content_type: str, meta: str | None, names: list[str], tamper: Any) -> Answer:
        header: dict[str, Any] = {"method": method, "path": path.lstrip("/"), "content_type": content_type, "states": [], "blobs": [], "body": len(body)}
        if meta:
            header["meta"] = meta
        sections: list[bytes] = []
        if sid is not None and sid in self.states:
            raw, sig = self.states[sid]
            sent = zlib.compress(raw, 6) if self.deflate else raw
            entry: dict[str, Any] = {"id": sid, "size": len(sent), "encoding": "deflate" if self.deflate else "identity"}
            if sig:
                entry["sig"] = sig
            header["states"].append(entry)
            sections.append(sent)
            for name in names:
                data = self.blobs[sid][name]
                header["blobs"].append({"id": sid, "name": name, "size": len(data)})
                sections.append(data)
                self.sent_pictures += 1
        sections.append(body)
        if tamper is not None:
            tamper(header, sections)
        self.round_trips += 1
        r = self.http.post(self.prefix + "/api/carry", content=pack(header, sections), headers={"content-type": CARRY_TYPE})
        ctype = r.headers.get("content-type", "")
        if r.status_code != 200 or not r.content.startswith(b"LUMC"):
            return Answer(r.status_code, ctype, r.content)
        head, rest = unpack(r.content)
        pos = 0

        def take(n: int) -> bytes:
            nonlocal pos
            out = bytes(rest[pos : pos + n])
            pos += n
            return out

        for st in head["states"]:
            self.states[st["id"]] = (take(st["size"]), st.get("sig", ""))
            self.blobs.setdefault(st["id"], {})
        for b in head["blobs"]:
            self.blobs.setdefault(b["id"], {})[b["name"]] = take(b["size"])
        for k in head["kept"]:
            self.blobs.setdefault(k["id"], {})[k["name"]] = body
        for d in head["dropped"]:
            self.blobs.get(d["id"], {}).pop(d["name"], None)
        for gone in head["removed"]:
            self.states.pop(gone, None)
            self.blobs.pop(gone, None)
        inner = take(head["body"])
        assert pos == len(rest)
        return Answer(head["status"], head["content_type"], inner, head.get("headers"))

    @staticmethod
    def _id_of(path: str) -> str | None:
        parts = path.strip("/").split("?")[0].split("/")
        # api/sessions/<id>/..., api/calibration/<id>/...
        return parts[2] if len(parts) > 2 and parts[0] == "api" and parts[1] in ("sessions", "calibration") else None

    # ------------------------------------------- the calls of an HTTP client, as the tests make them
    def _call(self, method: str, path: str, json: Any = None, content: bytes | None = None, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, **kw: Any) -> Answer:
        if not self._is_job(path):
            # not a request of a job (the configuration, the library, a scan): it goes as it is
            r = self.http.request(method, self.prefix + path, json=json, content=content, headers=headers, params=params)
            return Answer(r.status_code, r.headers.get("content-type", ""), r.content, dict(r.headers))
        if params:
            path = path + ("&" if "?" in path else "?") + urlencode({k: v for k, v in params.items() if v is not None})
        h = {k.lower(): v for k, v in (headers or {}).items()}
        meta = h.get("x-lumen-meta")
        out = self.request(method, path, json_body=json, content=content, content_type=h.get("content-type", ""), raw_meta=meta, **kw)
        return out

    @staticmethod
    def _is_job(path: str) -> bool:
        p = path.lstrip("/")
        return (p.startswith("api/sessions") or p.startswith("api/calibration")) and not p.startswith("api/calibration/board.png")

    def get(self, path: str, **kw: Any) -> Answer:
        return self._call("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Answer:
        return self._call("POST", path, **kw)

    def put(self, path: str, **kw: Any) -> Answer:
        return self._call("PUT", path, **kw)

    def patch(self, path: str, **kw: Any) -> Answer:
        return self._call("PATCH", path, **kw)

    def delete(self, path: str, **kw: Any) -> Answer:
        return self._call("DELETE", path, **kw)

    def state(self, sid: str) -> dict:
        return json.loads(self.states[sid][0].decode("utf-8"))
