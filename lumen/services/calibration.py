"""Camera calibration workspaces.

A workspace collects photos of a printed board taken with *one* camera
configuration, calibrates the lens from them (``lumen.vision.calibration``)
and returns a profile.  The profile belongs to the phone, so the interface
keeps it in the browser of that phone and attaches it to measuring sessions;
the server keeps the workspace (points, photos, result) only as long as a
session, in the same store and under the same time limit.

Photos used for a calibration are preserved in the workspace until it expires
or is deleted, so a result can be re-examined; each view records the hash of
its picture.

Geometry bookkeeping
--------------------
Photos larger than the analysis limit are reduced before analysis, exactly as
measuring photos are.  The profile is therefore expressed for the *upright
native* pixel size of the photos (the calibrated model scaled back by the
recorded reduction), which is what ``camera_for_frame`` matches frames against.
"""

from __future__ import annotations

import secrets
from typing import Any

import cv2
import numpy as np

from .. import RESEARCH_NOTICE
from ..config import Settings
from ..errors import CalibrationError, ConflictError, LimitExceededError, LumenError, LumenValidationError, NotFoundError
from ..schemas import Calibration
from ..storage.base import SessionStore
from ..vision.calibration import WEB_API_NOTE, BoardObservation, BoardSpec, CaptureConfig, calibrate_camera, calibration_summary, detect_board, validate_planar
from ..vision.imaging import decode_image
from .context import CpuGate, utcnow

MAX_PHOTOS = 30
KIND = "calibration"


def new_workspace_id() -> str:
    return "cal" + secrets.token_urlsafe(18)


def board_png(spec: BoardSpec, dpi: int = 300, margin_mm: float = 12.0) -> bytes:
    """A printable picture of the board (white margin, caption) at ``dpi``.

    Printers rarely reproduce a requested size exactly: the caption tells the
    operator to measure the print.
    """
    if not (72 <= dpi <= 600):
        raise LumenValidationError("invalid_dpi", "dpi must be between 72 and 600.")
    p = max(8, int(round(spec.square_mm * dpi / 25.4)))
    if p * max(spec.squares_x, spec.squares_y) > 9000:
        raise LumenValidationError("board_too_large", "The board picture would be too large; lower the dpi.")
    if spec.kind == "checkerboard":
        i, j = np.meshgrid(np.arange(spec.squares_x), np.arange(spec.squares_y))
        pat = np.kron(np.where((i + j) % 2 == 0, 0, 255).astype(np.uint8), np.ones((p, p), np.uint8))
    else:
        pat = spec.cv_board().generateImage((spec.squares_x * p, spec.squares_y * p), marginSize=0, borderBits=1)
    m = int(round(margin_mm * dpi / 25.4))
    foot = int(round(9.0 * dpi / 25.4))
    img = np.full((pat.shape[0] + 2 * m + foot, pat.shape[1] + 2 * m), 255, np.uint8)
    img[m : m + pat.shape[0], m : m + pat.shape[1]] = pat
    text = f"LUMEN calibration target  {spec.kind} {spec.squares_x} x {spec.squares_y}  -  measure the printed squares and enter the measured size"
    scale = max(0.4, dpi / 300.0 * 1.1)
    cv2.putText(img, text, (m, img.shape[0] - int(0.35 * foot)), cv2.FONT_HERSHEY_SIMPLEX, scale, 60, max(1, int(round(2 * dpi / 300.0))), cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise LumenError("encode_failed", "The board picture could not be encoded.", http_status=500)
    return buf.tobytes()


class CalibrationService:
    def __init__(self, store: SessionStore, settings: Settings, cpu: CpuGate) -> None:
        self.store = store
        self.settings = settings
        self.cpu = cpu

    # ----------------------------------------------------------------- state
    def _load(self, cid: str, operator: str = "") -> dict[str, Any]:
        st = self.store.load(cid)
        if st.get("kind") != KIND:
            raise NotFoundError("calibration_not_found", "Unknown or expired calibration.")
        if st.get("operator") and operator and st["operator"] != operator:
            raise NotFoundError("calibration_not_found", "Unknown or expired calibration.")
        return st

    def _save(self, st: dict[str, Any]) -> None:
        st["updated"] = utcnow()
        self.store.save(st["id"], st)

    @staticmethod
    def _board(st: dict[str, Any]) -> BoardSpec:
        return BoardSpec.from_dict(st["board"])

    @staticmethod
    def public(st: dict[str, Any]) -> dict[str, Any]:
        """Workspace for the client: no corner lists (they stay on the server)."""
        views = []
        for v in st["views"]:
            o = v["observation"]
            views.append({
                "view_id": o["view_id"], "ok": o["ok"], "reasons": o["reasons"], "warnings": o["warnings"], "suggestions": o["suggestions"],
                "n_points": o["n_points"], "image_size": o["image_size"], "native_size": v["native_size"], "quality": o["quality"],
                "image_id": o["image_id"], "bbox": o["quality"].get("bbox"),
            })
        return {
            "id": st["id"], "created": st["created"], "updated": st["updated"], "board": st["board"], "capture": st.get("capture"),
            "views": views, "n_usable": sum(v["ok"] for v in views), "result": st.get("result"), "profile": st.get("profile"),
            "validations": st.get("validations", []), "limits": {"max_photos": MAX_PHOTOS},
            "web_api_note": WEB_API_NOTE, "notice": RESEARCH_NOTICE,
        }

    # ------------------------------------------------------------- lifecycle
    def create(self, board: dict[str, Any], operator: str = "") -> dict[str, Any]:
        spec = BoardSpec.from_dict(board)  # validates
        self.store.enforce_capacity(self.settings.limits.max_sessions)
        now = utcnow()
        st = {
            "kind": KIND, "id": new_workspace_id(), "created": now, "updated": now, "operator": operator[:120],
            "board": spec.to_dict(), "capture": None, "views": [], "validations": [], "result": None, "profile": None, "counter": 0,
        }
        self.store.create(st["id"], st)
        return self.public(st)

    def get(self, cid: str, operator: str = "") -> dict[str, Any]:
        return self.public(self._load(cid, operator))

    def delete(self, cid: str, operator: str = "") -> bool:
        self._load(cid, operator)
        return self.store.delete(cid)

    # ---------------------------------------------------------------- photos
    def _decode(self, data: bytes) -> Any:
        lim = self.settings.limits
        return decode_image(data, max_bytes=lim.max_image_bytes, max_pixels=lim.max_image_pixels, max_long_edge=lim.max_analysis_long_edge)

    @staticmethod
    def _native(img: Any) -> tuple[int, int]:
        m = img.meta
        return (m.native_height, m.native_width) if m.exif_orientation in (5, 6, 7, 8) else (m.native_width, m.native_height)

    def add_photo(self, cid: str, data: bytes, meta: dict[str, Any] | None = None, operator: str = "") -> dict[str, Any]:
        meta = meta or {}
        with self.store.lock(cid):
            st = self._load(cid, operator)
            if len(st["views"]) >= MAX_PHOTOS:
                raise LimitExceededError("too_many_photos", f"A calibration holds at most {MAX_PHOTOS} photos.", suggestions=["Remove photos that were refused, then add better ones."])
        img = self._decode(data)
        cap = CaptureConfig(
            device=str(meta.get("camera_label") or img.meta.model or "")[:120],
            lens=str(meta.get("lens") or img.meta.lens_model or "")[:120],
            zoom=_opt_float(meta.get("zoom")),
            capture_mode=str(meta.get("capture_mode") or "")[:40],
        )
        native = self._native(img)
        with self.store.lock(cid):
            st = self._load(cid, operator)
            st["counter"] = int(st.get("counter", 0)) + 1
            vid = f"c{st['counter']:02d}"
            self._save(st)
        with self.cpu.slot(self.settings.limits.request_timeout_seconds):
            obs = detect_board(img.bgr, self._board(st), view_id=vid, capture=cap)
        blob = f"{vid}.jpg" if img.meta.mime == "image/jpeg" else f"{vid}.img"
        with self.store.lock(cid):
            st = self._load(cid, operator)
            self.store.put_blob(cid, blob, data)
            st["views"].append({"observation": obs.to_dict(points=True), "native_size": list(native), "scale": float(img.processing_scale), "blob": blob})
            st["result"], st["profile"] = None, None  # a new photo makes any earlier result stale
            self._save(st)
        return {"view": self.public(st)["views"][-1], "calibration": self.public(st)}

    def remove_photo(self, cid: str, view_id: str, operator: str = "") -> dict[str, Any]:
        with self.store.lock(cid):
            st = self._load(cid, operator)
            keep = [v for v in st["views"] if v["observation"]["view_id"] != view_id]
            if len(keep) == len(st["views"]):
                raise NotFoundError("photo_not_found", "Unknown calibration photo.")
            for v in st["views"]:
                if v["observation"]["view_id"] == view_id:
                    self.store.delete_blob(cid, v["blob"])
            st["views"], st["result"], st["profile"] = keep, None, None
            self._save(st)
            return self.public(st)

    def photo(self, cid: str, view_id: str, operator: str = "") -> bytes:
        st = self._load(cid, operator)
        v = next((v for v in st["views"] if v["observation"]["view_id"] == view_id), None)
        if v is None:
            raise NotFoundError("photo_not_found", "Unknown calibration photo.")
        return self.store.get_blob(cid, v["blob"])

    # ----------------------------------------------------------------- solve
    def _observations(self, st: dict[str, Any]) -> list[BoardObservation]:
        board = self._board(st)
        return [BoardObservation.from_dict(board, v["observation"]) for v in st["views"]]

    def solve(self, cid: str, *, label: str = "", operator: str = "") -> dict[str, Any]:
        st = self._load(cid, operator)
        if not st["views"]:
            raise ConflictError("no_photos", "Add photos of the board first.")
        natives = {tuple(v["native_size"]) for v in st["views"] if v["observation"]["ok"]}
        scales = {round(float(v["scale"]), 6) for v in st["views"] if v["observation"]["ok"]}
        obs = self._observations(st)
        with self.cpu.slot(self.settings.limits.job_timeout_seconds):
            res = calibrate_camera(obs)  # raises CalibrationError for mixed configurations
        doc = calibration_summary(res, observations=obs)
        profile = None
        if res.ok:
            if len(natives) != 1 or len(scales) != 1:
                raise CalibrationError("mixed_camera_configurations", "The photos do not all have the same pixel size.", suggestions=["Use one camera, one resolution and one orientation for all calibration photos."])
            (nw, nh), s = next(iter(natives)), next(iter(scales))
            cap = res.capture
            pid = "cal-" + secrets.token_hex(5)
            prof = res.to_profile(pid, label=label[:80] or f"{cap.device or 'camera'} {nw}x{nh}", created=utcnow())
            if abs(s - 1.0) > 1e-9:
                # the photos were analysed reduced: express the model for the native pixel size
                cam = res.camera.scaled(1.0 / s, 1.0 / s)  # type: ignore[union-attr]
                k = 1.0 / s
                prof.update({"width": int(nw), "height": int(nh), "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy})
                sig = list(prof.get("param_sigma") or [])
                for i in range(min(4, len(sig))):
                    if sig[i] is not None:
                        sig[i] = float(sig[i]) * k
                prof["param_sigma"] = sig
                prof["notes"] = (prof.get("notes", "") + f" Photos were analysed at {s:.4f} of their native size; the profile is expressed for {nw}x{nh}.").strip()
            profile = Calibration.model_validate(prof).model_dump(mode="json")
        with self.store.lock(cid):
            st = self._load(cid, operator)
            st["result"], st["profile"], st["capture"] = doc, profile, res.capture.to_dict()
            self._save(st)
        return self.public(st)

    def validate(self, cid: str, data: bytes, meta: dict[str, Any] | None = None, operator: str = "") -> dict[str, Any]:
        """Check the calibrated model on a board photo that was not used for it."""
        st = self._load(cid, operator)
        if not st.get("profile") or not st.get("result"):
            raise ConflictError("not_calibrated", "Calibrate first, then check the result with a photo that was not used.")
        img = self._decode(data)
        board = self._board(st)
        with self.cpu.slot(self.settings.limits.request_timeout_seconds):
            obs = detect_board(img.bgr, board, view_id="check")
            if not obs.ok:
                raise CalibrationError(obs.reasons[0] if obs.reasons else "board_not_found", "The board could not be measured in this photo.", suggestions=list(obs.suggestions))
            used = {v["observation"]["image_id"] for v in st["views"]}
            used_points = {v["observation"]["fingerprint"] for v in st["views"]}
            if (obs.image_id and obs.image_id in used) or obs.fingerprint in used_points:
                raise ConflictError("photo_was_used", "This photo was used for the calibration: take a new one for the check.")
            # the model at the analysed size of this photo
            prof = Calibration.model_validate(st["profile"])
            from ..geometry.camera import CameraModel

            cam = CameraModel(prof.width, prof.height, prof.fx, prof.fy, prof.cx, prof.cy, tuple((list(prof.dist) + [0.0] * 5)[:5]), source=prof.method)  # type: ignore[arg-type]
            nat = self._native(img)
            if tuple(nat) != (prof.width, prof.height):
                raise CalibrationError("mixed_camera_configurations", f"The profile is for {prof.width}x{prof.height} but this photo is {nat[0]}x{nat[1]}.")
            if (img.width, img.height) != (prof.width, prof.height):
                cam = cam.scaled(img.width / prof.width, img.height / prof.height)
                cam = CameraModel(img.width, img.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist, source=prof.method)
            val = validate_planar(cam, obs)
        rec = val.to_dict()
        # independence was established above: neither this picture nor these points are among the calibration photos
        rec["independent"] = True
        with self.store.lock(cid):
            st = self._load(cid, operator)
            st.setdefault("validations", []).append(rec)
            st["validations"] = st["validations"][-5:]
            self._save(st)
        return {"validation": rec, "calibration": self.public(st)}


def _opt_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
