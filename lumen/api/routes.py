"""HTTP API (all paths are relative to the application base).

Conventions
-----------
* JSON in, JSON out; errors are ``{"error": {code, message, details, suggestions}}``.
* Images are uploaded as the raw request body (``Content-Type: image/jpeg``),
  with optional capture metadata as URL-encoded JSON in ``X-Lumen-Meta``.
  Raw bodies give exact size control and upload progress without multipart.
* Pixel coordinates in requests and responses use the pixel-centre convention
  of the frame they belong to (see ``lumen/__init__.py``).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from typing import Any
from urllib.parse import unquote

import cv2
import numpy as np
import scipy
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .. import ALGORITHM_VERSION, RESEARCH_NOTICE, __version__
from ..errors import LimitExceededError, LumenValidationError, NotFoundError
from ..operations.library import parse_operation
from ..schemas import Calibration, Session
from ..services.sessions import SessionService

router = APIRouter()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _svc(request: Request) -> SessionService:
    return request.app.state.service


def _user(request: Request) -> str:
    """Username supplied by Posit Connect, if any (never trusted for geometry)."""
    raw = request.headers.get("rstudio-connect-credentials") or request.headers.get("posit-connect-credentials")
    if not raw:
        return ""
    try:
        return str(json.loads(raw).get("user", ""))[:120]
    except Exception:
        return ""


def _session(request: Request, sid: str) -> Session:
    s = _svc(request).get(sid)
    user = _user(request)
    if s.operator and user and s.operator != user:
        # a session id leaked to another authenticated user must not open it
        raise NotFoundError("session_not_found", "Unknown or expired session.")
    return s


async def _read_body(request: Request, limit: int) -> bytes:
    chunks: list[bytes] = []
    n = 0
    async for chunk in request.stream():
        n += len(chunk)
        if n > limit:
            raise LimitExceededError("payload_too_large", f"Upload exceeds the {limit / 1e6:.0f} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _meta(request: Request) -> dict[str, Any]:
    raw = request.headers.get("x-lumen-meta")
    if not raw:
        return {}
    try:
        d = json.loads(unquote(raw))
        return d if isinstance(d, dict) else {}
    except Exception:
        raise LumenValidationError("invalid_meta", "X-Lumen-Meta is not valid URL-encoded JSON.") from None


def public_session(s: Session) -> dict[str, Any]:
    """Session JSON for the client, without bulky numeric internals."""
    d = s.model_dump(mode="json")
    for f in d["frames"]:
        for m in f["markers"]:
            m.pop("cov", None)
            m.pop("corners_ideal", None)
    op = d.get("operation")
    if op:
        from ..operations.library import OperationDefinition

        o = OperationDefinition.model_validate(op["operation"])
        op["summary"] = o.summary()
        op["placards_resolved"] = [p.resolved() for p in o.placards]
    d["events"] = d["events"][-30:]
    d["notice"] = RESEARCH_NOTICE
    return d


# --------------------------------------------------------------------------
# health / config
# --------------------------------------------------------------------------
_HEALTH_CACHE: dict[str, Any] = {}


def _self_test() -> dict[str, Any]:
    """Real dependency check: render an AprilTag, detect it, decode a JPEG."""
    out: dict[str, Any] = {}
    try:
        from ..synth.tags import tag_image
        from ..vision.markers import detect_tags

        img = tag_image(17, cell_px=12, quiet_cells=2.0)
        img = (30 + (img.astype(np.float32) / 255.0) * 195).astype(np.uint8)  # black 30, white 225
        canvas = np.full((240, 320), 170, np.uint8)
        y0, x0 = 40, 90
        canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
        canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
        dets = detect_tags(canvas)
        # the black square spans pixel-centre coordinates 113.5 .. 209.5 horizontally
        exact = bool(len(dets) == 1 and abs(float(dets[0].corners_px[0][0]) - 113.5) < 0.1)
        out["apriltag_36h11"] = bool(len(dets) == 1 and dets[0].tag_id == 17 and dets[0].accepted and exact)
        ok, buf = cv2.imencode(".jpg", canvas)
        out["jpeg_codec"] = bool(ok and cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) is not None)
    except Exception as exc:  # pragma: no cover - reported, not raised
        out["apriltag_36h11"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import zxingcpp  # noqa: F401

        out["zxing"] = True
    except Exception:
        out["zxing"] = False
    return out


@router.get("/health")
def health(request: Request) -> JSONResponse:
    svc = _svc(request)
    if "self_test" not in _HEALTH_CACHE:
        _HEALTH_CACHE["self_test"] = _self_test()
    st = _HEALTH_CACHE["self_test"]
    storage_ok = True
    try:
        n_sessions = len(svc.store.list_ids())
    except Exception:
        storage_ok = False
        n_sessions = -1
    ok = bool(st.get("apriltag_36h11") and st.get("jpeg_codec") and storage_ok and not svc.library.load_errors)
    body = {
        "status": "ok" if ok else "degraded",
        "version": __version__,
        "algorithm_version": ALGORITHM_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {"opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "checks": {**st, "storage": storage_ok, "operation_library": not svc.library.load_errors},
        "library": {"revision": svc.library.library_revision, "operations": len(svc.library.all()), "errors": svc.library.load_errors, "editable": svc.library.editable},
        "storage": {"adapter": type(svc.store).__name__, "sessions": n_sessions},
        "uptime_s": round(time.time() - request.app.state.started, 1),
        "notice": RESEARCH_NOTICE,
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/config")
def config(request: Request) -> dict[str, Any]:
    svc = _svc(request)
    s = svc.settings
    lim = s.limits
    return {
        "version": __version__,
        "algorithm_version": ALGORITHM_VERSION,
        "notice": RESEARCH_NOTICE,
        "limits": {
            "max_image_mb": lim.max_image_bytes / 1e6,
            "max_image_megapixels": lim.max_image_pixels / 1e6,
            "max_video_mb": lim.max_video_bytes / 1e6,
            "max_video_seconds": lim.max_video_seconds,
            "max_frames": lim.max_frames_per_session,
            "session_ttl_seconds": lim.session_ttl_seconds,
            "max_track_long_edge": lim.max_track_long_edge,
        },
        "features": {
            "demo": s.demo_enabled,
            "assistant": bool(s.llm_enabled),
            "photometric_research": s.photometric_enabled,
            "library_editable": svc.library.editable,
        },
        "policy": svc.policy.to_dict(),
        "validated_domain": None if svc._domain is None else {"id": svc._domain.get("id"), "evidence": svc._domain.get("evidence")},
        "user": _user(request),
        "capture_hints": {"min_views": 3, "max_views": 9},
    }


# --------------------------------------------------------------------------
# operation library
# --------------------------------------------------------------------------
@router.get("/operations")
def operations(request: Request) -> dict[str, Any]:
    lib = _svc(request).library
    return {
        "library_revision": lib.library_revision,
        "editable": lib.editable,
        "errors": lib.load_errors,
        "operations": [{**op.summary(), "origin": lib.origin(op)} for op in lib.all()],
    }


@router.get("/operations/{operation_id:path}")
def operation(request: Request, operation_id: str, revision: str | None = None) -> dict[str, Any]:
    lib = _svc(request).library
    op = lib.get(operation_id, revision)
    return {
        "operation": op.model_dump(mode="json"),
        "summary": op.summary(),
        "origin": lib.origin(op),
        "placards_resolved": [p.resolved() for p in op.placards],
        "revisions": [{"revision": o.revision, "status": o.status} for o in lib.revisions(operation_id)],
    }


class ResolveBody(BaseModel):
    code: str = Field(max_length=256)


@router.post("/operations/resolve")
def resolve(request: Request, body: ResolveBody) -> dict[str, Any]:
    return _svc(request).library.resolve(body.code)


class SaveOperationBody(BaseModel):
    operation: dict[str, Any]


@router.post("/operations")
def save_operation(request: Request, body: SaveOperationBody) -> dict[str, Any]:
    lib = _svc(request).library
    op = parse_operation(body.operation)
    saved = lib.save(op, author=_user(request))
    return {"summary": saved.summary(), "origin": lib.origin(saved)}


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------
class CreateSessionBody(BaseModel):
    mode: str = "real"


@router.post("/sessions")
def create_session(request: Request, body: CreateSessionBody | None = None) -> dict[str, Any]:
    mode = (body.mode if body else "real").lower()
    if mode not in ("real", "demo"):
        raise LumenValidationError("invalid_mode", "mode must be 'real' or 'demo'.")
    s = _svc(request).create(mode, _user(request))
    return public_session(s)


@router.get("/sessions/{sid}")
def get_session(request: Request, sid: str) -> dict[str, Any]:
    return public_session(_session(request, sid))


@router.delete("/sessions/{sid}")
def delete_session(request: Request, sid: str) -> dict[str, Any]:
    _session(request, sid)
    return {"deleted": _svc(request).delete(sid)}


class OperationBody(BaseModel):
    operation_id: str = Field(max_length=64)
    revision: str | None = Field(default=None, max_length=32)
    placard_id: str | None = Field(default=None, max_length=64)


@router.put("/sessions/{sid}/operation")
def set_operation(request: Request, sid: str, body: OperationBody) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).set_operation(sid, body.operation_id, body.revision, body.placard_id))


class PlacardBody(BaseModel):
    placard_id: str = Field(max_length=64)


@router.put("/sessions/{sid}/placard")
def set_placard(request: Request, sid: str, body: PlacardBody) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).set_active_placard(sid, body.placard_id))


class TagSizesBody(BaseModel):
    nominal_mm: float = 50.0
    measured_mm: dict[str, float] = Field(default_factory=dict)
    measured_u_mm: float = 0.05
    nominal_tolerance_mm: float = 1.0


@router.put("/sessions/{sid}/tags")
def set_tags(request: Request, sid: str, body: TagSizesBody) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).set_tag_sizes(sid, body.model_dump()))


class CalibrationBody(BaseModel):
    profile: Calibration | None = None


@router.put("/sessions/{sid}/calibration")
def set_calibration(request: Request, sid: str, body: CalibrationBody) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).set_calibration(sid, body.profile))


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
@router.post("/sessions/{sid}/frames")
async def add_frame(request: Request, sid: str, role: str = "registration", source: str = "photo_upload") -> dict[str, Any]:
    _session(request, sid)
    svc = _svc(request)
    data = await _read_body(request, svc.settings.limits.max_image_bytes)
    meta = _meta(request)
    frame = await run_in_threadpool(svc.add_frame, sid, data, role=role, source=source, client=meta)
    d = frame.model_dump(mode="json")
    for m in d["markers"]:
        m.pop("cov", None)
        m.pop("corners_ideal", None)
    return {"frame": d}


@router.delete("/sessions/{sid}/frames/{fid}")
def remove_frame(request: Request, sid: str, fid: str) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).remove_frame(sid, fid))


class FramePatch(BaseModel):
    selected: bool


@router.patch("/sessions/{sid}/frames/{fid}")
def patch_frame(request: Request, sid: str, fid: str, body: FramePatch) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).set_frame_selected(sid, fid, body.selected))


@router.get("/sessions/{sid}/frames/{fid}/image")
def frame_image(request: Request, sid: str, fid: str, max: int | None = None) -> Response:
    _session(request, sid)
    if max is not None and not (64 <= max <= 8192):
        raise LumenValidationError("invalid_size", "max must be between 64 and 8192.")
    data, mime = _svc(request).frame_image(sid, fid, max)
    return Response(data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/sessions/{sid}/frames/{fid}/rectified")
def rectified(request: Request, sid: str, fid: str, max: int = 1600, oriented: str = "auto") -> Response:
    _session(request, sid)
    if not (128 <= max <= 4096):
        raise LumenValidationError("invalid_size", "max must be between 128 and 4096.")
    data, spec = _svc(request).rectified(sid, fid, oriented, max)
    return Response(data, media_type="image/jpeg", headers={"X-Lumen-Rect": json.dumps(spec), "Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
@router.post("/sessions/{sid}/register")
def register(request: Request, sid: str) -> dict[str, Any]:
    _session(request, sid)
    svc = _svc(request)
    reg = svc.register(sid)
    return {"registration": reg, "session": public_session(svc.get(sid))}


class DatumBody(BaseModel):
    frame_id: str = Field(max_length=16)
    p0: list[float] = Field(min_length=2, max_length=2)
    p1: list[float] = Field(min_length=2, max_length=2)
    space: str = "image"
    rect: dict[str, Any] | None = None
    snap: bool = True
    search_px: float = Field(default=18.0, ge=2.0, le=80.0)


@router.post("/sessions/{sid}/datums/confirm")
def confirm_datums(request: Request, sid: str, confirmed: bool = True) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).confirm_datums(sid, confirmed))


@router.post("/sessions/{sid}/datums/propose")
def propose_datums(request: Request, sid: str, apply: bool = True) -> dict[str, Any]:
    _session(request, sid)
    svc = _svc(request)
    out = svc.propose_datums(sid, apply=apply)
    return {"proposals": out, "session": public_session(svc.get(sid))}


class SelectCandidateBody(BaseModel):
    kind: str = Field(max_length=8)
    candidate_id: str = Field(max_length=24)


@router.post("/sessions/{sid}/datums/select")
def select_datum(request: Request, sid: str, body: SelectCandidateBody) -> dict[str, Any]:
    _session(request, sid)
    svc = _svc(request)
    obs = svc.select_datum_candidate(sid, body.kind, body.candidate_id)
    return {"datum": obs.model_dump(mode="json"), "session": public_session(svc.get(sid))}


@router.post("/sessions/{sid}/datums/{kind}")
def propose_datum(request: Request, sid: str, kind: str, body: DatumBody) -> dict[str, Any]:
    _session(request, sid)
    svc = _svc(request)
    obs = svc.propose_datum(sid, kind, body.frame_id, body.p0, body.p1, space=body.space, rect=body.rect, snap=body.snap, search_px=body.search_px)
    return {"datum": obs.model_dump(mode="json"), "session": public_session(svc.get(sid))}


@router.delete("/sessions/{sid}/datums/{kind}")
def clear_datum(request: Request, sid: str, kind: str) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).clear_datum(sid, kind))


@router.get("/sessions/{sid}/overlay")
def overlay(request: Request, sid: str, frame: str, view: str = "image", placards: str | None = None, uncertainty: bool = True, rect_max: int = 1600, candidates: bool = False) -> dict[str, Any]:
    _session(request, sid)
    if view not in ("image", "rect"):
        raise LumenValidationError("invalid_view", "view must be 'image' or 'rect'.")
    if not (128 <= rect_max <= 4096):
        raise LumenValidationError("invalid_size", "rect_max must be between 128 and 4096.")
    pids = [p for p in placards.split(",") if p] if placards else None
    return _svc(request).overlay(sid, frame, view=view, placard_ids=pids, with_uncertainty=uncertainty, rect_max_edge=rect_max, candidates=candidates)


@router.get("/sessions/{sid}/report")
def report(request: Request, sid: str, frame: str | None = None, placard: str | None = None) -> dict[str, Any]:
    _session(request, sid)
    return _svc(request).gate_report(sid, frame, placard).model_dump(mode="json")


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
class DemoBody(BaseModel):
    scenario: str = Field(default="flat", max_length=32)


@router.get("/demo/scenarios")
def demo_scenarios(request: Request) -> dict[str, Any]:
    from ..services.demo import scenarios

    return {"enabled": _svc(request).settings.demo_enabled, "scenarios": scenarios()}


@router.post("/sessions/{sid}/demo/load")
def demo_load(request: Request, sid: str, body: DemoBody) -> dict[str, Any]:
    from ..services.demo import load_demo

    _session(request, sid)
    return public_session(load_demo(_svc(request), sid, body.scenario))


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
@router.get("/sessions/{sid}/export")
def export_session(request: Request, sid: str) -> dict[str, Any]:
    from ..services.export import build_export

    _session(request, sid)
    return build_export(_svc(request), sid)
