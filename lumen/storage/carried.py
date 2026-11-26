"""A session store that keeps nothing: the browser carries the job.

On a host that forgets everything between two requests (a function platform such as Vercel) a job cannot live
with the server: the request that analyses the photos and the request that measures a live frame against that
analysis may be served by two processes that have never heard of each other.

Here the job lives with the browser.  Every request brings the job's state and the pictures the step needs, and
the answer takes the changed state back.  Between two requests this process holds nothing that it needs.

For the rest of the application nothing changes: this is a :class:`SessionStore`.  What it "stores" is the
content of one request, kept in a context variable while that request is served (``api/carry.py`` puts it there).

What a client sends is not trusted more than any other input: the state is validated by the models that read
it, a picture must have the checksum its frame record names (``SessionService._image``), and where a secret is
configured (``LUMEN_STATE_SECRET``) a state must carry the signature this server gave it.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import struct
import zlib
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..errors import ConflictError, LimitExceededError, LumenError, LumenValidationError, NotFoundError
from .base import SessionStore, check_blob_name, check_session_id

MAGIC = b"LUMC\x01"
#: the largest header of an envelope (the list of what follows; never the content)
MAX_HEADER_BYTES = 256 * 1024
#: a state is a JSON document of some tens of kilobytes; nothing near this is one
MAX_STATE_BYTES = 8 * 1024 * 1024


class PicturesNeeded(LumenError):
    """The step needs pictures of the job that did not come with the request.  The client sends them and asks again."""

    http_status = 428
    default_code = "pictures_needed"


@dataclass
class Carried:
    """What one request brought, and what serving it changed."""

    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    blobs: dict[tuple[str, str], bytes] = field(default_factory=dict)
    #: checksum of the request's own body: a picture stored from it need not travel back
    body_sha256: str = ""
    changed: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)
    added: dict[tuple[str, str], bytes] = field(default_factory=dict)
    dropped: set[tuple[str, str]] = field(default_factory=set)


_CURRENT: ContextVar[Carried | None] = ContextVar("lumen_carried", default=None)


def begin(carried: Carried) -> Token:
    return _CURRENT.set(carried)


def end(token: Token) -> None:
    _CURRENT.reset(token)


def current() -> Carried | None:
    return _CURRENT.get()


class CarriedStore(SessionStore):
    """Sessions and calibration workspaces that exist for the length of one request."""

    @staticmethod
    def _here() -> Carried:
        c = _CURRENT.get()
        if c is None:
            # a request that did not come through the carrier: there is no job here, and none can be made
            raise NotFoundError(
                "session_not_found",
                "This server keeps no job between requests: the job travels with the phone, and this request came without it.",
                suggestions=["Reload the page and start the job again."],
            )
        return c

    def create(self, sid: str, state: dict[str, Any]) -> None:
        check_session_id(sid)
        c = self._here()
        if sid in c.states:
            raise FileExistsError(sid)
        c.states[sid] = json.loads(json.dumps(state))
        c.changed.add(sid)
        c.removed.discard(sid)

    def load(self, sid: str) -> dict[str, Any]:
        check_session_id(sid)
        c = self._here()
        if sid not in c.states or sid in c.removed:
            raise NotFoundError(
                "session_not_found",
                "The job did not come with the request.",
                suggestions=["Reload the page. If the job is gone from this phone, start it again."],
            )
        return json.loads(json.dumps(c.states[sid]))

    def save(self, sid: str, state: dict[str, Any]) -> None:
        check_session_id(sid)
        c = self._here()
        if sid not in c.states or sid in c.removed:
            raise NotFoundError("session_not_found", "The job did not come with the request.")
        c.states[sid] = json.loads(json.dumps(state))
        c.changed.add(sid)

    def delete(self, sid: str) -> bool:
        check_session_id(sid)
        c = self._here()
        if sid not in c.states or sid in c.removed:
            return False
        c.removed.add(sid)
        c.changed.discard(sid)
        for key in [k for k in c.blobs if k[0] == sid]:
            del c.blobs[key]
        for key in [k for k in c.added if k[0] == sid]:
            del c.added[key]
        return True

    def exists(self, sid: str) -> bool:
        c = _CURRENT.get()
        return bool(c is not None and sid in c.states and sid not in c.removed)

    def put_blob(self, sid: str, name: str, data: bytes) -> None:
        check_session_id(sid)
        check_blob_name(name)
        c = self._here()
        if sid not in c.states:
            raise NotFoundError("session_not_found", "The job did not come with the request.")
        c.blobs[(sid, name)] = data
        c.added[(sid, name)] = data
        c.dropped.discard((sid, name))

    def get_blob(self, sid: str, name: str) -> bytes:
        check_session_id(sid)
        check_blob_name(name)
        c = self._here()
        data = c.blobs.get((sid, name))
        if data is None:
            raise PicturesNeeded(
                "pictures_needed", "This step needs a picture of the job that was not sent with the request.",
                details={"id": sid, "blobs": [name]},
            )
        return data

    def delete_blob(self, sid: str, name: str) -> None:
        check_session_id(sid)
        check_blob_name(name)
        c = self._here()
        c.blobs.pop((sid, name), None)
        c.added.pop((sid, name), None)
        c.dropped.add((sid, name))

    def list_ids(self) -> list[tuple[str, float]]:
        return []

    @contextlib.contextmanager
    def lock(self, sid: str) -> Iterator[None]:
        # one request, one copy of the job: there is nobody to keep out
        yield

    def purge_expired(self, ttl_seconds: float, now: float | None = None) -> int:
        return 0

    def enforce_capacity(self, max_sessions: int) -> None:
        return None


# ------------------------------------------------------------------ envelope
def _state_bytes(state: dict[str, Any]) -> bytes:
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign(secret: str, sid: str, raw: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), sid.encode("utf-8") + b"\n" + raw, hashlib.sha256).hexdigest()


def pack(header: dict[str, Any], sections: list[bytes]) -> bytes:
    h = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return b"".join([MAGIC, struct.pack(">I", len(h)), h, *sections])


def unpack(data: bytes) -> tuple[dict[str, Any], memoryview]:
    """Header and the bytes after it.  Raises a validation error for anything that is not an envelope."""
    if len(data) < len(MAGIC) + 4 or data[: len(MAGIC)] != MAGIC:
        raise LumenValidationError("not_a_carried_request", "The request is not in the form this server expects.")
    (n,) = struct.unpack(">I", data[len(MAGIC) : len(MAGIC) + 4])
    start = len(MAGIC) + 4
    if n > MAX_HEADER_BYTES or start + n > len(data):
        raise LumenValidationError("not_a_carried_request", "The request is not in the form this server expects.")
    try:
        header = json.loads(data[start : start + n].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LumenValidationError("not_a_carried_request", "The request is not in the form this server expects.") from exc
    if not isinstance(header, dict):
        raise LumenValidationError("not_a_carried_request", "The request is not in the form this server expects.")
    return header, memoryview(data)[start + n :]


def read_request(data: bytes, secret: str = "") -> tuple[dict[str, Any], Carried, bytes]:
    """Take a carried request apart: what is asked (method, path, ...), what came along, and the request's own body."""
    header, rest = unpack(data)
    pos = 0

    def take(size: Any) -> bytes:
        nonlocal pos
        if not isinstance(size, int) or size < 0 or pos + size > len(rest):
            raise LumenValidationError("not_a_carried_request", "The request is shorter than it says.")
        out = bytes(rest[pos : pos + size])
        pos += size
        return out

    carried = Carried()
    for st in header.get("states") or []:
        sid = check_session_id(str(st.get("id", "")))
        raw = take(st.get("size"))
        if st.get("encoding") == "deflate":
            try:
                d = zlib.decompressobj()
                raw = d.decompress(raw, MAX_STATE_BYTES + 1)
                if len(raw) > MAX_STATE_BYTES or d.unconsumed_tail:
                    raise LimitExceededError("state_too_large", "The job's state is larger than any job's can be.")
            except zlib.error as exc:
                raise LumenValidationError("invalid_state", "The job's state could not be read.") from exc
        elif len(raw) > MAX_STATE_BYTES:
            raise LimitExceededError("state_too_large", "The job's state is larger than any job's can be.")
        if secret and not hmac.compare_digest(sign(secret, sid, raw), str(st.get("sig", ""))):
            raise LumenValidationError(
                "state_not_ours", "The job's state was not written by this server, or was changed since.",
                suggestions=["Start the job again."], http_status=409,
            )
        try:
            state = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise LumenValidationError("invalid_state", "The job's state could not be read.") from exc
        if not isinstance(state, dict) or state.get("id") != sid:
            raise LumenValidationError("invalid_state", "The job's state does not belong to the job it was sent for.")
        carried.states[sid] = state
    for b in header.get("blobs") or []:
        sid = check_session_id(str(b.get("id", "")))
        name = check_blob_name(str(b.get("name", "")))
        if sid not in carried.states:
            raise LumenValidationError("not_a_carried_request", "A picture came for a job that did not come with it.")
        carried.blobs[(sid, name)] = take(b.get("size"))
    body = take(header.get("body", 0))
    if pos != len(rest):
        raise LumenValidationError("not_a_carried_request", "The request is longer than it says.")
    carried.body_sha256 = hashlib.sha256(body).hexdigest() if body else ""
    return header, carried, body


def write_response(status: int, content_type: str, body: bytes, carried: Carried, secret: str = "", headers: dict[str, str] | None = None) -> bytes:
    """The answer to a carried request: the inner answer (with those of its headers that the browser reads), the
    states that changed, and the pictures made here."""
    header: dict[str, Any] = {
        "status": status, "content_type": content_type, "headers": headers or {}, "body": len(body),
        "states": [], "removed": sorted(carried.removed), "blobs": [], "kept": [], "dropped": [],
    }
    sections: list[bytes] = []
    for sid in sorted(carried.changed - carried.removed):
        raw = _state_bytes(carried.states[sid])
        entry: dict[str, Any] = {"id": sid, "size": len(raw), "encoding": "identity"}
        if secret:
            entry["sig"] = sign(secret, sid, raw)
        header["states"].append(entry)
        sections.append(raw)
    for (sid, name), data in sorted(carried.added.items()):
        if sid in carried.removed:
            continue
        sha = hashlib.sha256(data).hexdigest()
        if carried.body_sha256 and sha == carried.body_sha256:
            # the picture is the one the request brought as its body: the client has it and only learns its name
            header["kept"].append({"id": sid, "name": name, "sha256": sha})
        else:
            header["blobs"].append({"id": sid, "name": name, "size": len(data), "sha256": sha})
            sections.append(data)
    header["dropped"] = [{"id": sid, "name": name} for sid, name in sorted(carried.dropped) if sid not in carried.removed]
    sections.append(body)
    return pack(header, sections)


def require_carrier(store: SessionStore) -> None:
    """For steps that cannot work when the job is carried (they outlive the request)."""
    if isinstance(store, CarriedStore):
        raise ConflictError(
            "not_available_here", "This step needs a server that keeps the job while it works on it. On this host the phone carries the job.",
            suggestions=["Take photos instead of a video."],
        )
