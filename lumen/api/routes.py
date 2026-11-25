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

import builtins
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
        "storage": lib.storage(),
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


@router.post("/scan")
async def scan(request: Request, thorough: bool = True) -> dict[str, Any]:
    """Read barcodes / QR codes from an uploaded picture and look each one up.

    The picture is used for this one answer and is not stored.  Code geometry
    is returned for display only and is never used as a scale reference.
    """
    from ..vision.barcode import decoders_available, read_codes
    from ..vision.imaging import decode_image

    svc = _svc(request)
    lim = svc.settings.limits
    data = await _read_body(request, lim.max_image_bytes)

    def work() -> dict[str, Any]:
        img = decode_image(data, max_bytes=lim.max_image_bytes, max_pixels=lim.max_image_pixels, max_long_edge=2400)
        with svc.cpu.slot(lim.request_timeout_seconds):
            codes = read_codes(img.bgr, thorough=thorough)
        return {
            "codes": [{**c.to_dict(), "resolved": svc.library.resolve(c.text)} for c in codes],
            "frame": {"width": img.width, "height": img.height},
            "decoders": decoders_available(),
            "geometry_used_for_scale": False,
        }

    return await run_in_threadpool(work)


class SaveOperationBody(BaseModel):
    operation: dict[str, Any]


def _record_view(lib: Any, op: Any) -> dict[str, Any]:
    return {"summary": op.summary(), "placards_resolved": [p.resolved() for p in op.placards]}


@router.post("/operations/validate")
def validate_operation(request: Request, body: SaveOperationBody) -> dict[str, Any]:
    """Check a record as typed so far, without storing it.

    Answers with what cannot be read (``problems``), what is missing for a
    release (``summary.issues``) and every length as the app will use it (mm).
    """
    from ..operations.library import analyse_record

    lib = _svc(request).library
    op, problems = analyse_record(body.operation)
    out: dict[str, Any] = {"problems": problems, "summary": None, "placards_resolved": [], "existing": None, "can_save": False, "can_release": False}
    if op is None:
        return out
    out.update(_record_view(lib, op))
    if not problems:
        existing = next((o for o in lib.revisions(op.operation_id) if o.revision == op.revision), None)
        if existing is not None:
            out["existing"] = {"status": existing.status, "origin": lib.origin(existing), "same_content": existing.model_copy(update={"status": op.status}).content_hash() == op.content_hash()}
        fixed = existing is not None and (lib.origin(existing) == "bundled" or existing.status in ("released", "superseded"))
        out["can_save"] = lib.editable and not fixed
        out["can_release"] = out["can_save"] and not op.issues()
    return out


@router.post("/operations")
def save_operation(request: Request, body: SaveOperationBody) -> dict[str, Any]:
    lib = _svc(request).library
    op = parse_operation(body.operation)
    if op.synthetic and op.key() not in {o.key() for o in lib.all() if lib.origin(o) == "bundled"}:
        # only the bundled sample is "synthetic sample data"; a record entered here is the operator's own
        op = op.model_copy(update={"synthetic": False})
    saved = lib.save(op, author=_user(request))
    return {**_record_view(lib, saved), "origin": lib.origin(saved)}


class RevisionRef(BaseModel):
    operation_id: str = Field(max_length=64)
    revision: str = Field(max_length=64)


@router.post("/operations/supersede")
def supersede_operation(request: Request, body: RevisionRef) -> dict[str, Any]:
    lib = _svc(request).library
    op = lib.supersede(body.operation_id, body.revision, author=_user(request))
    return {**_record_view(lib, op), "origin": lib.origin(op)}


@router.post("/operations/delete")
def delete_operation(request: Request, body: RevisionRef) -> dict[str, Any]:
    _svc(request).library.delete(body.operation_id, body.revision, author=_user(request))
    return {"deleted": True}


@router.get("/operations-audit")
def operations_audit(request: Request, operation_id: str | None = None) -> dict[str, Any]:
    return {"audit": _svc(request).library.audit(operation_id)[-200:]}


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
# camera calibration from a printed board
# --------------------------------------------------------------------------
def _cal(request: Request) -> Any:
    return request.app.state.calibration


@router.get("/calibration/board.png")
def calibration_board(request: Request, kind: str = "charuco", squares_x: int = 9, squares_y: int = 6, square_mm: float = 30.0, marker_mm: float | None = 21.0, dictionary: str | None = "DICT_5X5_100", dpi: int = 300) -> Response:
    """A printable calibration target.  Measure the print: printers do not keep the requested size."""
    from ..services.calibration import board_png
    from ..vision.calibration import BoardSpec

    spec = BoardSpec.from_dict({"kind": kind, "squares_x": squares_x, "squares_y": squares_y, "square_mm": square_mm, "marker_mm": marker_mm if kind == "charuco" else None, "dictionary": dictionary if kind == "charuco" else None, "size_provenance": "nominal"})
    return Response(board_png(spec, dpi), media_type="image/png", headers={"Content-Disposition": f'attachment; filename="lumen-{kind}-{squares_x}x{squares_y}.png"'})


class CalibrationCreateBody(BaseModel):
    board: dict[str, Any]


@router.post("/calibration")
def calibration_create(request: Request, body: CalibrationCreateBody) -> dict[str, Any]:
    return _cal(request).create(body.board, _user(request))


@router.get("/calibration/{cid}")
def calibration_get(request: Request, cid: str) -> dict[str, Any]:
    return _cal(request).get(cid, _user(request))


@router.delete("/calibration/{cid}")
def calibration_delete(request: Request, cid: str) -> dict[str, Any]:
    return {"deleted": _cal(request).delete(cid, _user(request))}


@router.post("/calibration/{cid}/photos")
async def calibration_add_photo(request: Request, cid: str) -> dict[str, Any]:
    svc = _cal(request)
    svc.get(cid, _user(request))
    data = await _read_body(request, svc.settings.limits.max_image_bytes)
    return await run_in_threadpool(svc.add_photo, cid, data, _meta(request), _user(request))


@router.delete("/calibration/{cid}/photos/{vid}")
def calibration_remove_photo(request: Request, cid: str, vid: str) -> dict[str, Any]:
    return _cal(request).remove_photo(cid, vid, _user(request))


@router.get("/calibration/{cid}/photos/{vid}/image")
def calibration_photo(request: Request, cid: str, vid: str, max: int = 480) -> Response:
    if not (64 <= max <= 4096):
        raise LumenValidationError("invalid_size", "max must be between 64 and 4096.")
    import cv2 as _cv2

    from ..vision.imaging import decode_image, encode_jpeg

    svc = _cal(request)
    lim = svc.settings.limits
    img = decode_image(svc.photo(cid, vid, _user(request)), max_bytes=lim.max_image_bytes, max_pixels=lim.max_image_pixels, max_long_edge=lim.max_analysis_long_edge).bgr
    k = max / float(builtins.max(img.shape[:2]))
    if k < 1.0:
        img = _cv2.resize(img, (int(round(img.shape[1] * k)), int(round(img.shape[0] * k))), interpolation=_cv2.INTER_AREA)
    return Response(encode_jpeg(img, 85), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


class CalibrationSolveBody(BaseModel):
    label: str = Field(default="", max_length=80)


@router.post("/calibration/{cid}/solve")
async def calibration_solve(request: Request, cid: str, body: CalibrationSolveBody | None = None) -> dict[str, Any]:
    svc = _cal(request)
    return await run_in_threadpool(lambda: svc.solve(cid, label=(body.label if body else ""), operator=_user(request)))


@router.post("/calibration/{cid}/validate")
async def calibration_validate(request: Request, cid: str) -> dict[str, Any]:
    svc = _cal(request)
    svc.get(cid, _user(request))
    data = await _read_body(request, svc.settings.limits.max_image_bytes)
    return await run_in_threadpool(svc.validate, cid, data, _meta(request), _user(request))


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
    selected: bool | None = None
    torch: str | None = Field(default=None, max_length=8)
    pair_id: str | None = Field(default=None, max_length=40)
    clear_pair: bool = False


@router.patch("/sessions/{sid}/frames/{fid}")
def patch_frame(request: Request, sid: str, fid: str, body: FramePatch) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).set_frame_attributes(sid, fid, selected=body.selected, torch=body.torch, pair_id=body.pair_id, clear_pair=body.clear_pair))


# --------------------------------------------------------------------------
# video and capture coaching
# --------------------------------------------------------------------------
_VIDEO_EXT = {"mp4": ".mp4", "webm": ".webm", "avi": ".avi"}


@router.post("/sessions/{sid}/video")
async def add_video(request: Request, sid: str, source: str = "video_upload") -> dict[str, Any]:
    """Upload a video (raw body); frames are chosen by a background job."""
    from ..errors import UnsupportedMediaError
    from ..services.capture import temp_video_path
    from ..vision.video import sniff_video_type

    _session(request, sid)
    svc = _svc(request)
    limit = svc.settings.limits.max_video_bytes
    path = None
    n = 0
    fh = None
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            if fh is None:
                kind = sniff_video_type(chunk[:16])
                if kind is None:
                    raise UnsupportedMediaError(
                        "unsupported_video", "This file is not an MP4/MOV, WebM or AVI video.",
                        suggestions=["Record with the phone camera (MOV/MP4) or use the in-app recorder."],
                    )
                path = temp_video_path(_VIDEO_EXT[kind])
                fh = open(path, "wb")
            n += len(chunk)
            if n > limit:
                raise LimitExceededError("payload_too_large", f"Video exceeds the {limit / 1e6:.0f} MB limit.", suggestions=["Record a shorter sweep (10–20 s is enough)."])
            fh.write(chunk)
        if fh is None or path is None:
            raise LumenValidationError("empty_upload", "The upload was empty.")
        fh.close()
        fh = None
        job = svc.start_video_job(sid, path, source=source, meta=_meta(request))
        path = None  # the job owns (and deletes) the file now
        return {"job": job}
    finally:
        if fh is not None:
            fh.close()
        if path is not None:
            import os

            try:
                os.remove(path)
            except OSError:
                pass


@router.get("/sessions/{sid}/jobs/{jid}")
def get_job(request: Request, sid: str, jid: str) -> dict[str, Any]:
    _session(request, sid)
    return _svc(request).jobs.get(sid, jid)


@router.delete("/sessions/{sid}/jobs/{jid}")
def cancel_job(request: Request, sid: str, jid: str) -> dict[str, Any]:
    _session(request, sid)
    return _svc(request).jobs.cancel(sid, jid)


@router.get("/sessions/{sid}/capture/advice")
def capture_advice(request: Request, sid: str) -> dict[str, Any]:
    _session(request, sid)
    return _svc(request).capture_advice(sid)


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
def overlay(request: Request, sid: str, frame: str, view: str = "image", placards: str | None = None, uncertainty: bool = True, rect_max: int = 1600, candidates: bool = False, references: bool = False) -> dict[str, Any]:
    _session(request, sid)
    if view not in ("image", "rect"):
        raise LumenValidationError("invalid_view", "view must be 'image' or 'rect'.")
    if not (128 <= rect_max <= 4096):
        raise LumenValidationError("invalid_size", "rect_max must be between 128 and 4096.")
    pids = [p for p in placards.split(",") if p] if placards else None
    return _svc(request).overlay(sid, frame, view=view, placard_ids=pids, with_uncertainty=uncertainty, rect_max_edge=rect_max, candidates=candidates, references=references)


class CornerRef(BaseModel):
    tag_id: int = Field(ge=0, le=586)
    corner: int = Field(ge=0, le=3)


class ReferenceBody(BaseModel):
    a: CornerRef
    b: CornerRef
    measured_mm: float
    u_mm: float = 0.5
    instrument: str = Field(default="", max_length=80)


@router.get("/sessions/{sid}/reference")
def reference_suggestion(request: Request, sid: str) -> dict[str, Any]:
    """Which two tag corners to measure between for an independent scale check."""
    s = _session(request, sid)
    return {"suggestion": _svc(request).reference_suggestion(sid), "references": s.reference_distances}


@router.post("/sessions/{sid}/reference")
def add_reference(request: Request, sid: str, body: ReferenceBody) -> dict[str, Any]:
    _session(request, sid)
    svc = _svc(request)
    rec = svc.add_reference_distance(sid, body.a.model_dump(), body.b.model_dump(), body.measured_mm, body.u_mm, body.instrument)
    return {"reference": rec, "session": public_session(svc.get(sid))}


@router.delete("/sessions/{sid}/reference/{rid}")
def remove_reference(request: Request, sid: str, rid: str) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).remove_reference_distance(sid, rid))


class VerifyBody(BaseModel):
    frame_id: str = Field(max_length=16)
    placard_id: str | None = Field(default=None, max_length=64)
    #: operator-marked corners in image pixels (any order); omit for automatic detection
    corners_px: list[list[float]] | None = Field(default=None, min_length=4, max_length=4)
    snap: bool = True


@router.post("/sessions/{sid}/verify")
def verify(request: Request, sid: str, body: VerifyBody) -> dict[str, Any]:
    """Measure the installed placard in a verification photo (evidence, not acceptance)."""
    _session(request, sid)
    svc = _svc(request)
    out = svc.verify(sid, body.frame_id, placard_id=body.placard_id, manual_corners_px=body.corners_px, snap=body.snap)
    out["session"] = public_session(svc.get(sid))
    return out


class ConfirmBody(BaseModel):
    confirmed: bool = True


@router.post("/sessions/{sid}/verifications/{vid}/confirm")
def confirm_verification(request: Request, sid: str, vid: str, body: ConfirmBody | None = None) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).confirm_verification(sid, vid, True if body is None else body.confirmed))


@router.delete("/sessions/{sid}/verifications/{vid}")
def delete_verification(request: Request, sid: str, vid: str) -> dict[str, Any]:
    _session(request, sid)
    return public_session(_svc(request).delete_verification(sid, vid))


@router.post("/sessions/{sid}/track")
async def track(request: Request, sid: str, placards: str | None = None, full: bool = False) -> dict[str, Any]:
    """Live guidance: overlay for the uploaded camera frame.  The frame is not stored."""
    _session(request, sid)
    svc = _svc(request)
    data = await _read_body(request, svc.settings.limits.max_image_bytes)
    pids = [p for p in placards.split(",") if p] if placards else None
    return await run_in_threadpool(svc.track, sid, data, placard_ids=pids, meta=_meta(request), force_full=full)


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


@router.post("/sessions/{sid}/assistant")
async def assistant(request: Request, sid: str) -> dict[str, Any]:
    """Optional explanation of LUMEN's own results (disabled unless an administrator enables it)."""
    from ..assistant.service import AssistantRequest, AssistantService
    from ..errors import FeatureDisabledError

    _session(request, sid)
    svc = _svc(request)
    if not svc.settings.llm_enabled:
        raise FeatureDisabledError("assistant_disabled", "The assistant is not enabled on this deployment.")
    raw = await _read_body(request, 64 * 1024)
    try:
        body = AssistantRequest.model_validate_json(raw or b"{}")
    except Exception:
        raise LumenValidationError("invalid_request", "The assistant request is not valid.") from None
    adapter = getattr(request.app.state, "assistant_adapter", None)
    return await run_in_threadpool(AssistantService(svc, adapter).ask, sid, body)


@router.post("/sessions/{sid}/research/photometric")
async def research_photometric(request: Request, sid: str) -> dict[str, Any]:
    """Experimental torch-shading analysis of a rounded edge.  Research evidence only: nothing uses the result."""
    from ..services.photometric import photometric_research

    _session(request, sid)
    svc = _svc(request)
    record = await run_in_threadpool(photometric_research, svc, sid)
    return {"photometric": record, "session": public_session(svc.get(sid))}


@router.post("/sessions/{sid}/demo/verification")
def demo_verification(request: Request, sid: str) -> dict[str, Any]:
    from ..services.demo import add_demo_verification

    _session(request, sid)
    svc = _svc(request)
    frame = add_demo_verification(svc, sid)
    return {"frame": {"id": frame.id}, "session": public_session(svc.get(sid))}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
@router.get("/sessions/{sid}/export")
def export_session(request: Request, sid: str) -> dict[str, Any]:
    from ..services.export import build_export

    _session(request, sid)
    return build_export(_svc(request), sid)
