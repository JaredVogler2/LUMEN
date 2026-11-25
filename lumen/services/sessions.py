"""Session orchestration: the workflow behind the API.

Stages (each is separately callable and idempotent):

    configure (operation, tag sizes, calibration)
      -> add frames (decode, quality, tag detection)
      -> register (per-view mapping + tag layout, cross-validation)
      -> datums (snap / manual, confirm) -> local frame
      -> overlay + gate report for any registered frame
      -> track (live frame against the layout), verify (see verification.py)

State lives in the session store as JSON; heavy objects (decoded images, solver
results) are rebuilt on demand and cached per process, so any worker process
can serve any request.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

from .. import ALGORITHM_VERSION, RESEARCH_NOTICE, __version__
from ..config import Settings
from ..errors import ConflictError, GeometryError, LimitExceededError, LumenValidationError, NotFoundError
from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, local_scale_px_per_mm
from ..geometry.lines import Line2D
from ..geometry.localframe import LocalFrame, build_local_frame, target_rectangle
from ..operations.library import OperationDefinition, OperationLibrary, snapshot
from ..overlay.geometry import build_overlay, footprint_for_image, image_mapper, rect_mapper
from ..policy.gates import GateInputs, GatePolicy, check_validated_domain, evaluate_gates, load_policy
from ..policy.uncertainty import DatumEvidence, EvalContext, FrameConfig, TargetSpec, UncertaintyConfig, evaluate_uncertainty
from ..schemas import (
    Calibration,
    DatumObservation,
    Frame,
    FrameRole,
    FrameSource,
    GateReport,
    GuidanceStatus,
    LocalFrameModel,
    PlacementSpec,
    Provenance,
    Session,
    SessionEvent,
    SessionMode,
)
from ..storage.base import SessionStore, new_session_id
from ..vision.edgefit import snap_line
from ..vision.imaging import DecodedImage, encode_jpeg, to_gray
from ..vision.markers import MarkerConfig, TagSizeTable
from ..vision.planar import PlanarSolution, PlanarView, ViewLocalization, localize_view, solve_planar
from ..vision.rectify import RectSpec, plan_rectification, rectify_image
from .context import CpuGate, LruCache, utcnow
from .frames import build_frame, camera_for_frame, decode_for_session, detect_in_image, sizes_from_state, tag_obs

import cv2
import json
import os


class SessionService:
    def __init__(self, store: SessionStore, settings: Settings, library: OperationLibrary) -> None:
        self.store = store
        self.settings = settings
        self.library = library
        self.cpu = CpuGate(settings.limits.max_concurrent_jobs)
        self._images = LruCache(6)
        self._solutions = LruCache(16)
        self._policy = load_policy(os.environ.get("LUMEN_POLICY_FILE") or None)
        self._ucfg = UncertaintyConfig()
        self._marker_cfg = MarkerConfig()
        self._domain: dict[str, Any] | None = None
        if settings.validated_domain_file:
            try:
                with open(settings.validated_domain_file, "r", encoding="utf-8") as fh:
                    self._domain = json.load(fh)
            except Exception:
                self._domain = None
        self._last_purge = 0.0
        self._purge_lock = threading.Lock()

    # ------------------------------------------------------------------ util
    @property
    def policy(self) -> GatePolicy:
        return self._policy

    def _housekeeping(self) -> None:
        import time

        now = time.time()
        with self._purge_lock:
            if now - self._last_purge < 60.0:
                return
            self._last_purge = now
        self.store.purge_expired(self.settings.limits.session_ttl_seconds, now)

    def _load(self, sid: str) -> Session:
        return Session.model_validate(self.store.load(sid))

    def _save(self, s: Session) -> None:
        s.updated = utcnow()
        self.store.save(s.id, s.model_dump(mode="json"))

    def _mutate(self, sid: str, fn: Callable[[Session], Any], action: str | None = None, detail: dict[str, Any] | None = None) -> Any:
        with self.store.lock(sid):
            s = self._load(sid)
            out = fn(s)
            if action:
                s.events.append(SessionEvent(at=utcnow(), action=action, detail=detail or {}))
                if len(s.events) > 400:
                    s.events = s.events[-400:]
            self._save(s)
            return out

    @staticmethod
    def _frame(s: Session, frame_id: str) -> Frame:
        for f in s.frames:
            if f.id == frame_id:
                return f
        raise NotFoundError("frame_not_found", f"Frame {frame_id!r} does not exist in this session.")

    def _invalidate(self, s: Session, *, registration: bool = True, datums: bool = False) -> None:
        if registration:
            s.registration = None
            self._solutions.drop_prefix(s.id)
        if datums:
            s.datums = {}
            s.local_frame = None

    # ------------------------------------------------------------- lifecycle
    def create(self, mode: str = "real", operator: str = "") -> Session:
        self._housekeeping()
        self.store.enforce_capacity(self.settings.limits.max_sessions)
        if mode == "demo" and not self.settings.demo_enabled:
            raise ConflictError("demo_disabled", "Demo mode is disabled on this deployment.")
        now = utcnow()
        s = Session(
            id=new_session_id(),
            mode=SessionMode(mode),
            created=now,
            updated=now,
            algorithm_version=ALGORITHM_VERSION,
            app_version=__version__,
            operator=operator[:120],
            tag_sizes=TagSizeTable().to_dict(),
        )
        s.events.append(SessionEvent(at=now, action="session_created", detail={"mode": mode}))
        self.store.create(s.id, s.model_dump(mode="json"))
        return s

    def get(self, sid: str) -> Session:
        return self._load(sid)

    def delete(self, sid: str) -> bool:
        self._images.drop_prefix(sid)
        self._solutions.drop_prefix(sid)
        return self.store.delete(sid)

    # --------------------------------------------------------- configuration
    def set_operation(self, sid: str, operation_id: str, revision: str | None, placard_id: str | None) -> Session:
        op = self.library.get(operation_id, revision)
        if placard_id is not None:
            op.placard(placard_id)

        def fn(s: Session) -> Session:
            prev = (s.operation or {}).get("content_hash")
            s.operation = snapshot(op, self.library)
            s.active_placard_id = placard_id or (op.placards[0].placard_id if op.placards else None)
            if prev != s.operation["content_hash"]:
                # datum semantics may differ between records
                s.datums = {}
                s.local_frame = None
            return s

        return self._mutate(sid, fn, "operation_selected", {"operation_id": op.operation_id, "revision": op.revision, "placard_id": placard_id})

    def set_active_placard(self, sid: str, placard_id: str) -> Session:
        def fn(s: Session) -> Session:
            op = self._operation(s)
            op.placard(placard_id)
            s.active_placard_id = placard_id  # registration and datums stay valid (prompt 08)
            return s

        return self._mutate(sid, fn, "placard_selected", {"placard_id": placard_id})

    def set_tag_sizes(self, sid: str, table: dict[str, Any]) -> Session:
        try:
            t = TagSizeTable.from_dict(table)
        except Exception as exc:
            raise LumenValidationError("invalid_tag_sizes", "Tag size table is not valid.") from exc
        for name, v in [("nominal", t.nominal_mm)] + [(f"tag {k}", v) for k, v in t.measured_mm.items()]:
            if not (5.0 <= v <= 500.0):
                raise LumenValidationError("tag_size_out_of_range", f"Size of {name} must be between 5 and 500 mm (got {v}).")
        for k in t.measured_mm:
            if not (0 <= k <= 586):
                raise LumenValidationError("tag_id_out_of_range", f"Tag id {k} is not in AprilTag 36h11 (0–586).")

        def fn(s: Session) -> Session:
            s.tag_sizes = t.to_dict()
            for f in s.frames:
                for m in f.markers:
                    size, prov = t.size(m.tag_id)
                    m.size_mm = size
                    m.size_provenance = Provenance.MEASURED if prov == "measured" else Provenance.NOMINAL
            self._invalidate(s, registration=True, datums=True)
            return s

        return self._mutate(sid, fn, "tag_sizes_set", {"n_measured": len(t.measured_mm)})

    def set_calibration(self, sid: str, cal: Calibration | None) -> Session:
        if cal is not None:
            if cal.width < 64 or cal.height < 64 or cal.fx <= 0 or cal.fy <= 0:
                raise LumenValidationError("invalid_calibration", "Calibration profile has implausible values.")
            if not (0.2 * cal.width < cal.fx < 20 * cal.width):
                raise LumenValidationError("invalid_calibration", "Focal length is implausible for this image size.")

        def fn(s: Session) -> Session:
            s.calibration = cal
            # corner refinement depends on the lens model: frames are re-measured lazily
            for f in s.frames:
                f.notes = [n for n in f.notes if not n.startswith("needs re-detection")]
                f.notes.append("needs re-detection: calibration changed")
            self._invalidate(s, registration=True, datums=True)
            return s

        out = self._mutate(sid, fn, "calibration_set", {"id": None if cal is None else cal.id})
        self._redetect_stale(sid)
        return self._load(sid)

    def _operation(self, s: Session) -> OperationDefinition:
        if not s.operation:
            raise ConflictError("no_operation", "No operation is selected for this session.", suggestions=["Choose an operation on the Identify step."])
        return OperationDefinition.model_validate(s.operation["operation"])

    # ----------------------------------------------------------------- frames
    def _image(self, sid: str, frame: Frame) -> DecodedImage:
        key = (sid, frame.id)
        img = self._images.get(key)
        if img is None:
            img = decode_for_session(self.store.get_blob(sid, frame.blob), self.settings)
            self._images.put(key, img)
        return img

    def add_frame(
        self,
        sid: str,
        data: bytes,
        *,
        role: str = "registration",
        source: str = "photo_upload",
        client: dict[str, Any] | None = None,
    ) -> Frame:
        s0 = self._load(sid)
        active = [f for f in s0.frames]
        if len(active) >= self.settings.limits.max_frames_per_session:
            raise LimitExceededError(
                "too_many_frames",
                f"This session already holds {len(active)} frames (limit {self.settings.limits.max_frames_per_session}).",
                suggestions=["Remove frames you do not need before adding more."],
            )
        try:
            role_e = FrameRole(role)
            source_e = FrameSource(source)
        except ValueError as exc:
            raise LumenValidationError("invalid_frame_kind", "Unknown frame role or source.") from exc
        img = decode_for_session(data, self.settings)
        sizes = sizes_from_state(s0.tag_sizes)
        with self.cpu.slot(self.settings.limits.request_timeout_seconds):
            dets, cam, note = detect_in_image(img, s0.calibration, sizes, self._marker_cfg)
        ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/bmp": "bmp"}.get(img.meta.mime, "bin")

        def fn(s: Session) -> Frame:
            if len(s.frames) >= self.settings.limits.max_frames_per_session:
                raise LimitExceededError("too_many_frames", "Frame limit reached.")
            s.frame_counter += 1
            fid = f"f{s.frame_counter:03d}"
            blob = f"{fid}.{ext}"
            self.store.put_blob(sid, blob, data)
            frame = build_frame(
                frame_id=fid, index=s.frame_counter, blob=blob, data_len=len(data), img=img, dets=dets,
                cal=s.calibration, cal_note=note, cam=cam, role=role_e, source=source_e, client=client,
            )
            if s.mode == SessionMode.DEMO:
                frame.source = FrameSource.DEMO
            s.frames.append(frame)
            if role_e == FrameRole.REGISTRATION:
                self._invalidate(s, registration=True, datums=False)
            self._images.put((sid, fid), img)
            return frame

        return self._mutate(sid, fn, "frame_added", {"role": role, "source": source, "bytes": len(data), "sha256": img.meta.sha256[:16]})

    def remove_frame(self, sid: str, frame_id: str) -> Session:
        def fn(s: Session) -> Session:
            f = self._frame(s, frame_id)
            s.frames = [x for x in s.frames if x.id != frame_id]
            self.store.delete_blob(sid, f.blob)
            self._images.drop_prefix(sid)
            self._invalidate(s, registration=True, datums=False)
            for k in [k for k, d in s.datums.items() if d.frame_id == frame_id]:
                del s.datums[k]
                s.local_frame = None
            s.verifications = [v for v in s.verifications if v.frame_id != frame_id]
            return s

        return self._mutate(sid, fn, "frame_removed", {"frame_id": frame_id})

    def set_frame_selected(self, sid: str, frame_id: str, selected: bool) -> Session:
        def fn(s: Session) -> Session:
            self._frame(s, frame_id).selected = bool(selected)
            self._invalidate(s, registration=True, datums=False)
            return s

        return self._mutate(sid, fn, "frame_selection", {"frame_id": frame_id, "selected": selected})

    def frame_image(self, sid: str, frame_id: str, max_edge: int | None = None) -> tuple[bytes, str]:
        s = self._load(sid)
        f = self._frame(s, frame_id)
        if max_edge is None:
            data = self.store.get_blob(sid, f.blob)
            mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "bmp": "image/bmp"}.get(f.blob.rsplit(".", 1)[-1], "application/octet-stream")
            if f.processing_scale == 1.0 and not f.orientation_applied and mime == "image/jpeg":
                return data, mime
        img = self._image(sid, f)
        bgr = img.bgr
        if max_edge is not None and max(bgr.shape[:2]) > max_edge:
            sc = max_edge / float(max(bgr.shape[:2]))
            bgr = cv2.resize(bgr, (int(round(bgr.shape[1] * sc)), int(round(bgr.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        return encode_jpeg(bgr, 88), "image/jpeg"

    def _redetect_stale(self, sid: str) -> None:
        s = self._load(sid)
        stale = [f for f in s.frames if any(n.startswith("needs re-detection") for n in f.notes)]
        if not stale:
            return
        sizes = sizes_from_state(s.tag_sizes)
        results: dict[str, tuple[Any, Any, str]] = {}
        for f in stale:
            img = self._image(sid, f)
            with self.cpu.slot(self.settings.limits.request_timeout_seconds):
                results[f.id] = detect_in_image(img, s.calibration, sizes, self._marker_cfg)

        def fn(s2: Session) -> None:
            from .frames import marker_from_detection

            for f in s2.frames:
                if f.id in results:
                    dets, cam, note = results[f.id]
                    f.markers = [marker_from_detection(d) for d in dets]
                    f.calibration_id = s2.calibration.id if (s2.calibration is not None and cam is not None) else ""
                    f.calibration_note = note
                    f.notes = [n for n in f.notes if not n.startswith("needs re-detection")]

        self._mutate(sid, fn)

    # ------------------------------------------------------------ registration
    def _camera(self, s: Session, f: Frame) -> CameraModel | None:
        if not f.calibration_id or s.calibration is None or s.calibration.id != f.calibration_id:
            return None
        cam, _ = camera_for_frame(s.calibration, f.width, f.height, f.native_width, f.native_height, f.processing_scale)
        return cam

    def _views(self, s: Session, frames: list[Frame]) -> list[PlanarView]:
        out = []
        for f in frames:
            tags = [tag_obs(m) for m in f.markers if m.accepted]
            if tags:
                out.append(PlanarView(f.id, tags, self._camera(s, f), "auto"))
        return out

    @staticmethod
    def _registration_frames(s: Session) -> list[Frame]:
        return [f for f in s.frames if f.role == FrameRole.REGISTRATION.value and f.selected]

    def _solution_key(self, s: Session) -> tuple[Any, ...]:
        frames = self._registration_frames(s)
        return (
            s.id,
            tuple((f.id, f.sha256[:12], f.calibration_id) for f in frames),
            json.dumps(s.tag_sizes, sort_keys=True),
        )

    def _solve(self, s: Session) -> tuple[PlanarSolution, list[PlanarView]]:
        key = self._solution_key(s)
        hit = self._solutions.get(key)
        if hit is not None:
            return hit
        frames = self._registration_frames(s)
        views = self._views(s, frames)
        if not views:
            raise GeometryError(
                "no_tags",
                "No accepted AprilTag in the selected frames: there is no metric scale.",
                suggestions=["Place a measured AprilTag on the flat panel, fully in view and in focus, and capture again."],
            )
        sizes = sizes_from_state(s.tag_sizes)
        size_u = {}
        for v in views:
            for t in v.tags:
                _, prov = sizes.size(t.tag_id)
                size_u[t.tag_id] = sizes.measured_u_mm if prov == "measured" else sizes.nominal_tolerance_mm / np.sqrt(3.0)
        with self.cpu.slot(self.settings.limits.request_timeout_seconds):
            sol = solve_planar(views, size_u_mm=size_u)
        used = [v for v in views if v.frame_id in sol.views]
        # keep only tags that survived the consistency checks
        used = [PlanarView(v.frame_id, [t for t in v.tags if t.tag_id in sol.layout.poses], v.camera, v.mode) for v in used]
        self._solutions.put(key, (sol, used))
        return sol, used

    def register(self, sid: str) -> dict[str, Any]:
        s = self._load(sid)
        self._redetect_stale(sid)
        s = self._load(sid)
        sol, views = self._solve(s)
        frames = {f.id: f for f in s.frames}
        ref = self._pick_reference(sol, frames)
        summary = sol.to_dict()
        summary["reference_frame_id"] = ref
        summary["frames_used"] = sorted(sol.views)
        summary["computed_at"] = utcnow()
        summary["method"] = "planar_joint"
        px_per_mm = {}
        for fid, v in sol.views.items():
            px_per_mm[fid] = float(np.median(local_scale_px_per_mm(v.H, sol.layout.all_corners())))
        summary["px_per_mm"] = px_per_mm

        def fn(s2: Session) -> None:
            s2.registration = summary
            # datums measured in frames that are no longer registered are dropped
            for k in [k for k, d in s2.datums.items() if d.frame_id not in sol.views]:
                del s2.datums[k]
            self._rebuild_local_frame(s2, sol)

        self._mutate(sid, fn, "registered", {"n_tags": sol.n_tags, "n_views": len(sol.views), "flags": sol.flags})
        return summary

    @staticmethod
    def _pick_reference(sol: PlanarSolution, frames: dict[str, Frame]) -> str:
        def score(fid: str) -> tuple[float, float, float]:
            v = sol.views[fid]
            f = frames[fid]
            return (float(v.n_tags), -float(v.rms_px), float(f.quality.sharpness))

        return max(sorted(sol.views), key=score)

    def _require_registration(self, s: Session) -> tuple[PlanarSolution, list[PlanarView]]:
        if not s.registration:
            raise ConflictError("not_registered", "The session has not been registered yet.", suggestions=["Run the analysis after capturing photos."])
        return self._solve(s)

    def _frame_mapping(self, s: Session, sol: PlanarSolution, f: Frame) -> tuple[np.ndarray, ViewLocalization | None, PlanarView | None]:
        """Homography of a frame: from the joint solution, or localised against the layout."""
        if f.id in sol.views:
            return sol.views[f.id].H, None, None
        tags = [tag_obs(m) for m in f.markers if m.accepted]
        view = PlanarView(f.id, tags, self._camera(s, f), "auto")
        loc = localize_view(sol.layout, view)
        return loc.H, loc, PlanarView(f.id, [t for t in tags if t.tag_id in sol.layout.poses], view.camera, view.mode)

    # ---------------------------------------------------------------- rectified
    def rect_spec(self, s: Session, sol: PlanarSolution, f: Frame, H: np.ndarray, oriented: str = "auto", max_edge: int = 1600) -> RectSpec:
        lf = self._local_frame(s) if oriented in ("auto", "local") else None
        include = None
        kw: dict[str, Any] = {}
        if lf is not None and lf.valid:
            ex_o, ez_o = lf.ortho_axes
            # never mirror the photo: keep the handedness of the plane frame
            right = np.array([ez_o[1], -ez_o[0]])
            kw = {"up_hint": ez_o, "right_hint": right}
            include = lf.origin[None]
        return plan_rectification(
            H, f.width, f.height, sol.layout.all_corners(), max_long_edge_px=max_edge, include_plane_pts=include, **kw
        )

    def rectified(self, sid: str, frame_id: str, oriented: str = "auto", max_edge: int = 1600) -> tuple[bytes, dict[str, Any]]:
        s = self._load(sid)
        sol, _ = self._require_registration(s)
        f = self._frame(s, frame_id)
        H, _, _ = self._frame_mapping(s, sol, f)
        spec = self.rect_spec(s, sol, f, H, oriented, max_edge)
        img = self._image(sid, f)
        with self.cpu.slot(self.settings.limits.request_timeout_seconds):
            rect, _valid = rectify_image(img.bgr, self._camera(s, f), H, spec)
        return encode_jpeg(rect, 88), spec.to_dict()

    # -------------------------------------------------------------------- datums
    def _datum_def(self, s: Session, kind: str) -> Any:
        op = self._operation(s)
        d = op.datums.floor if kind == "floor" else op.datums.side
        return d

    def _to_ideal(self, s: Session, sol: PlanarSolution, f: Frame, H: np.ndarray, pts: np.ndarray, space: str, rect: dict[str, Any] | None) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if space == "image":
            cam = self._camera(s, f)
            return cam.undistort_pixels(pts) if cam is not None else pts
        if space == "rect":
            if rect is None:
                raise LumenValidationError("rect_spec_required", "Rectified-view coordinates need the view specification they refer to.")
            spec = RectSpec.from_dict(rect)
            return apply_h(H, spec.rect_to_plane(pts))
        if space == "plane":
            return apply_h(H, pts)
        raise LumenValidationError("unknown_space", f"Unknown coordinate space {space!r}.")

    def propose_datum(
        self,
        sid: str,
        kind: str,
        frame_id: str,
        p0: list[float],
        p1: list[float],
        *,
        space: str = "image",
        rect: dict[str, Any] | None = None,
        snap: bool = True,
        search_px: float = 18.0,
    ) -> DatumObservation:
        """Turn an operator-drawn segment into a datum line (not yet confirmed)."""
        if kind not in ("floor", "side"):
            raise LumenValidationError("unknown_datum", "Datum kind must be 'floor' or 'side'.")
        s = self._load(sid)
        sol, _ = self._require_registration(s)
        f = self._frame(s, frame_id)
        if f.id not in sol.views:
            raise ConflictError("frame_not_registered", "Datums must be drawn on a frame that is part of the registration.")
        H = sol.views[f.id].H
        cam = self._camera(s, f)
        seg = self._to_ideal(s, sol, f, H, np.array([p0, p1]), space, rect)
        if float(np.linalg.norm(seg[1] - seg[0])) < 30.0:
            raise LumenValidationError("datum_too_short", "The drawn line is too short; draw along at least a few centimetres of the edge.")
        notes: list[str] = []
        feature_kind = "manual"
        provenance = Provenance.MANUAL
        distinct = None
        n_cand = None
        fit_d: dict[str, Any]
        if snap:
            gray = to_gray(self._image(sid, f).bgr)
            with self.cpu.slot(self.settings.limits.request_timeout_seconds):
                r = snap_line(gray, seg[0], seg[1], camera=cam, search_px=search_px)
            if r.ok and r.fit is not None and r.fit.mid is not None and r.fit.n is not None:
                e = r.fit
                feature_kind = r.kind
                provenance = Provenance.MEASURED
                distinct, n_cand = r.distinctness, r.n_candidates
                fit_d = {
                    "mid": e.mid.tolist(), "n": e.n.tolist(), "length": e.length,
                    "var_d": e.var_d * self._marker_cfg.covariance_inflation + 0.02**2,
                    "var_theta": e.var_theta * self._marker_cfg.covariance_inflation,
                    "cov_d_theta": e.cov_d_theta * self._marker_cfg.covariance_inflation,
                    "sigma_blur_px": e.sigma_blur, "contrast": e.contrast, "shift_px": r.shift_px, "tilt_deg": r.tilt_deg,
                }
            else:
                notes.append(f"no edge found near the drawn line ({r.reason}); kept as drawn")
                snap = False
        if not snap:
            t = seg[1] - seg[0]
            L = float(np.linalg.norm(t))
            t /= L
            fit_d = {"mid": (0.5 * (seg[0] + seg[1])).tolist(), "n": [-t[1], t[0]], "length": L, "var_d": 0.0, "var_theta": 0.0, "cov_d_theta": 0.0}

        ev = self._evidence(kind, f.id, fit_d, self._datum_def(s, kind), manual=(provenance == Provenance.MANUAL))
        interior = sol.layout.centroid()
        Hinv = np.linalg.inv(H)
        feature_line = DatumEvidence(kind, f.id, ev.mid, ev.n, ev.length, ev.cov).plane_line(Hinv, interior)
        line = ev.plane_line(Hinv, interior)
        ends = apply_h(Hinv, ev.points())
        mm_per_px = 1.0 / max(float(local_scale_px_per_mm(H, ends.mean(axis=0)[None])[0]), 1e-9)
        ddef = self._datum_def(s, kind)
        declared = None if ddef is None else ddef.on_target_plane
        obs = DatumObservation(
            kind=kind,
            n=line.n.tolist(),
            d=line.d,
            provenance=provenance,
            confirmed=False,
            feature_kind=feature_kind,
            frame_id=f.id,
            endpoints=ends.tolist(),
            sigma_perp_mm=float(np.sqrt(max(fit_d["var_d"], 0.0)) * mm_per_px),
            sigma_angle_rad=float(np.sqrt(max(fit_d["var_theta"], 0.0))),
            offset_applied_mm=ev.offset_mm,
            on_plane_evidence="configured" if declared else "unverified",
            distinctness=distinct,
            n_candidates=n_cand,
            notes=notes,
            image_fit=fit_d,
            feature_n=feature_line.n.tolist(),
            feature_d=feature_line.d,
        )

        def fn(s2: Session) -> None:
            s2.datums[kind] = obs
            self._rebuild_local_frame(s2, sol)

        self._mutate(sid, fn, "datum_proposed", {"kind": kind, "frame_id": f.id, "feature": feature_kind, "provenance": provenance.value})
        return obs

    @staticmethod
    def _evidence(kind: str, frame_id: str, fit: dict[str, Any], ddef: Any, manual: bool) -> DatumEvidence:
        offset = 0.0 if (ddef is None or ddef.offset is None) else float(ddef.offset.mm)
        positive = "toward_panel" if (ddef is None or ddef.positive is None) else str(ddef.positive)
        return DatumEvidence(
            kind=kind,
            frame_id=frame_id,
            mid=np.asarray(fit["mid"], dtype=float),
            n=np.asarray(fit["n"], dtype=float),
            length=float(fit["length"]),
            cov=np.array([[fit.get("var_d", 0.0), fit.get("cov_d_theta", 0.0)], [fit.get("cov_d_theta", 0.0), fit.get("var_theta", 0.0)]], dtype=float),
            offset_mm=offset,
            positive=positive,
            manual=manual,
        )

    def confirm_datums(self, sid: str, confirmed: bool = True) -> Session:
        def fn(s: Session) -> Session:
            if confirmed and not all(k in s.datums for k in ("floor", "side")):
                raise ConflictError("datums_missing", "Both the floor and the side datum must be defined before confirming.")
            for d in s.datums.values():
                d.confirmed = bool(confirmed)
            return s

        return self._mutate(sid, fn, "datums_confirmed" if confirmed else "datums_unconfirmed", {})

    def clear_datum(self, sid: str, kind: str) -> Session:
        def fn(s: Session) -> Session:
            s.datums.pop(kind, None)
            s.local_frame = None
            return s

        return self._mutate(sid, fn, "datum_cleared", {"kind": kind})

    def _frame_config(self, s: Session) -> FrameConfig | None:
        if not s.operation:
            return None
        op = self._operation(s)
        d = op.datums
        if d.axis_policy is None or d.floor is None or d.side is None:
            return None
        return FrameConfig(
            policy=d.axis_policy,
            nominal_angle_deg=float(np.degrees(d.nominal_angle.rad)) if d.nominal_angle is not None else 90.0,
            max_axis_disagreement_deg=None if d.max_axis_disagreement is None else float(np.degrees(d.max_axis_disagreement.rad)),
            x_positive=d.side.positive or "unset",
            z_positive=d.floor.positive or "unset",
            expected_side=d.side.side_of_panel,
        )

    def _rebuild_local_frame(self, s: Session, sol: PlanarSolution) -> None:
        s.local_frame = None
        if "floor" not in s.datums or "side" not in s.datums:
            return
        fcfg = self._frame_config(s)
        if fcfg is None or fcfg.policy not in ("floor_primary", "side_primary", "perpendicular_offsets"):
            return
        fl = Line2D(np.asarray(s.datums["floor"].n), s.datums["floor"].d)
        sd = Line2D(np.asarray(s.datums["side"].n), s.datums["side"].d)
        lf = build_local_frame(
            fl, sd, sol.layout.centroid(), policy=fcfg.policy, nominal_angle_deg=fcfg.nominal_angle_deg,
            max_axis_disagreement_deg=fcfg.max_axis_disagreement_deg, x_positive=fcfg.x_positive, z_positive=fcfg.z_positive,
            expected_side=fcfg.expected_side,
        )
        prov = Provenance.MEASURED if all(d.provenance == Provenance.MEASURED.value for d in s.datums.values()) else Provenance.MANUAL
        s.local_frame = LocalFrameModel(**lf.to_dict(), provenance=prov)

    @staticmethod
    def _local_frame(s: Session) -> LocalFrame | None:
        if s.local_frame is None:
            return None
        return LocalFrame.from_dict(s.local_frame.model_dump())

    # ------------------------------------------------------------ placement
    def placement_spec(self, s: Session, placard_id: str | None = None) -> PlacementSpec | None:
        if not s.operation:
            return None
        pid = placard_id or s.active_placard_id
        if pid is None:
            return None
        op = self._operation(s)
        return PlacementSpec(**op.placard(pid).resolved())

    @staticmethod
    def _spec_complete(spec: PlacementSpec | None) -> bool:
        return spec is not None and None not in (spec.x_mm, spec.z_mm, spec.width_mm, spec.height_mm, spec.angle_rad, spec.reference_point)

    def _target(self, lf: LocalFrame, spec: PlacementSpec) -> Any:
        return target_rectangle(lf, spec.x_mm, spec.z_mm, spec.width_mm, spec.height_mm, spec.angle_rad or 0.0, spec.reference_point or "center")  # type: ignore[arg-type]

    def gate_report(self, sid: str, frame_id: str | None = None, placard_id: str | None = None, with_uncertainty: bool = True) -> GateReport:
        s = self._load(sid)
        return self._gate_report(s, frame_id, placard_id, with_uncertainty)

    def _gate_report(
        self,
        s: Session,
        frame_id: str | None,
        placard_id: str | None,
        with_uncertainty: bool,
        ext: tuple[Frame, ViewLocalization, PlanarView] | None = None,
    ) -> GateReport:
        inp = GateInputs(demo=(s.mode == SessionMode.DEMO.value))
        pid = placard_id or s.active_placard_id
        spec: PlacementSpec | None = None
        op: OperationDefinition | None = None
        if s.operation:
            op = self._operation(s)
            inp.operation_selected = True
            inp.surface_target = op.surface.target
            if pid:
                try:
                    spec = PlacementSpec(**op.placard(pid).resolved())
                    inp.placard_selected = True
                except NotFoundError:
                    spec = None
            iss = op.issues()
            inp.spec_issues = list(iss.get("datums", [])) + list(iss.get(f"placard:{pid}", [])) + [i for i in iss.get("operation", []) if i != "curved_target_experimental"]
            for i, d in enumerate((op.datums.floor, op.datums.side)):
                if d is not None:
                    feat = list(inp.datum_feature)
                    decl = list(inp.datum_declared_on_plane)
                    feat[i] = d.feature
                    decl[i] = d.on_target_plane
                    inp.datum_feature = (feat[0], feat[1])
                    inp.datum_declared_on_plane = (decl[0], decl[1])
        inp.spec_geometry_complete = self._spec_complete(spec)
        if spec is not None:
            inp.tolerance_position_mm = spec.tol_position_mm
            inp.tolerance_rotation_rad = spec.tol_rotation_rad

        sol: PlanarSolution | None = None
        views: list[PlanarView] = []
        if s.registration:
            try:
                sol, views = self._solve(s)
            except GeometryError:
                sol = None
        frames = {f.id: f for f in s.frames}
        sizes = sizes_from_state(s.tag_sizes)
        f: Frame | None = None
        H: np.ndarray | None = None
        loc: ViewLocalization | None = None
        ext_view: PlanarView | None = None
        if sol is not None:
            inp.registered = True
            inp.registration_flags = list(sol.flags)
            inp.n_tags = sol.n_tags
            inp.tags_consistent = sol.consistent
            inp.inconsistent_reasons = sorted({r for c in sol.checks.values() if not c.consistent for r in c.reasons})
            inp.excluded_tags = dict(sol.excluded_tags)
            inp.independently_checked = sol.independently_checked
            sig = [c.size_ratio_sigma for c in sol.checks.values() if c.checked and np.isfinite(c.size_ratio_sigma)]
            inp.crossval_sigma_frac = float(max(sig)) if sig else None
            inp.reduced_chi2 = sol.reduced_chi2 if sol.dof > 0 else None
            inp.rms_px = sol.rms_px
            inp.all_tags_measured = all(sizes.size(i)[1] == "measured" for i in sol.layout.poses)
            inp.all_frames_calibrated = all(v.camera is not None for v in views)
            fid = frame_id or (s.registration or {}).get("reference_frame_id")
            if ext is not None:
                f, loc, ext_view = ext
                H = loc.H
            elif fid in frames:
                f = frames[fid]
                try:
                    H, loc, ext_view = self._frame_mapping(s, sol, f)
                except GeometryError:
                    H = None
            if f is not None and H is not None:
                inp.frame_localized = True
                used = [m for m in f.markers if m.accepted and m.tag_id in sol.layout.poses]
                inp.frame_n_tags = len(used)
                inp.frame_min_tag_edge_px = float(min(min(m.edge_px) for m in used)) if used else None
                inp.frame_calibrated = self._camera(s, f) is not None
                vflags = list(sol.views[f.id].flags) if f.id in sol.views else list(loc.flags if loc else [])
                inp.frame_flags = vflags
                inp.pose_ambiguous = "pose_ambiguous" in vflags

        inp.datums_present = ("floor" in s.datums, "side" in s.datums)
        inp.datums_confirmed = (bool(s.datums.get("floor") and s.datums["floor"].confirmed), bool(s.datums.get("side") and s.datums["side"].confirmed))
        inp.datum_distinctness = (
            s.datums["floor"].distinctness if "floor" in s.datums else None,
            s.datums["side"].distinctness if "side" in s.datums else None,
        )
        inp.datum_manual = (
            bool("floor" in s.datums and s.datums["floor"].provenance == Provenance.MANUAL.value),
            bool("side" in s.datums and s.datums["side"].provenance == Provenance.MANUAL.value),
        )
        inp.datum_plane_evidence = (
            s.datums["floor"].on_plane_evidence if "floor" in s.datums else "unverified",
            s.datums["side"].on_plane_evidence if "side" in s.datums else "unverified",
        )
        lf = self._local_frame(s)
        if lf is not None:
            inp.local_frame_valid = lf.valid
            inp.local_frame_reasons = list(lf.reasons)
            inp.axis_disagreement_deg = lf.axis_disagreement_deg
            fc = self._frame_config(s)
            inp.axis_limit_deg = None if fc is None else fc.max_axis_disagreement_deg
        elif all(inp.datums_present):
            inp.local_frame_reasons = ["axis policy or datum signs are not configured"]

        unc = None
        if sol is not None and lf is not None and lf.valid and inp.spec_geometry_complete and f is not None and H is not None and spec is not None:
            rect = self._target(lf, spec)
            uv = apply_h(H, rect.corners)
            cam = self._camera(s, f)
            raw = cam.distort_pixels(uv) if cam is not None else uv
            ph = np.concatenate([rect.corners, np.ones((4, 1))], axis=1) @ H.T
            ref_w = float((np.append(sol.layout.centroid(), 1.0)) @ H[2])
            in_front = bool(np.all(ph[:, 2] * np.sign(ref_w) > 0))
            inp.target_in_view = in_front and bool(np.all((raw[:, 0] >= 0) & (raw[:, 0] <= f.width - 1) & (raw[:, 1] >= 0) & (raw[:, 1] <= f.height - 1)))
            ctr = sol.layout.centroid()
            radius = max(sol.layout.extent_radius(), 1e-6)
            inp.extrapolation_ratio = float(np.max(np.linalg.norm(rect.corners - ctr, axis=1)) / radius)
            if with_uncertainty and inp.target_in_view and sol.consistent:
                try:
                    unc = self._uncertainty(s, sol, views, f, loc, ext_view, spec)
                except Exception as exc:  # uncertainty must never crash the report; it then fails its gate
                    unc = None
                    inp.spec_issues = (inp.spec_issues or [])
                    inp.registration_flags = (inp.registration_flags or []) + [f"uncertainty_failed:{type(exc).__name__}"]
        inp.uncertainty = unc
        inp.validated_domain = self._domain
        if self._domain and unc is not None and f is not None:
            inp.in_validated_domain = check_validated_domain(
                self._domain,
                {
                    "min_tag_edge_px": inp.frame_min_tag_edge_px,
                    "calibrated": inp.frame_calibrated and inp.all_frames_calibrated,
                    "n_tags": inp.n_tags,
                    "expanded_mm": unc.expanded_mm,
                    "target_offset_mm": float(np.hypot(spec.x_mm or 0.0, spec.z_mm or 0.0)) if spec else None,
                },
            )
        report = evaluate_gates(inp, self._policy, placard_id=pid)
        if with_uncertainty and unc is None and report.status != GuidanceStatus.BLOCKED and sol is not None:
            # geometry is computable but no uncertainty could be attached: stay at preview
            from ..schemas import GateResult, Severity

            report.gates.append(GateResult(code="uncertainty_available", passed=False, severity=Severity.WARNING, message="No uncertainty could be evaluated for this target.", suggestion="Check the datum lines and retake the photo."))
            report.reasons.append("No uncertainty could be evaluated for this target.")
            if report.status in (GuidanceStatus.PROTOTYPE_GUIDANCE.value, GuidanceStatus.VALIDATED.value):
                report.status = GuidanceStatus.RESEARCH_PREVIEW
        return report

    def _uncertainty(
        self,
        s: Session,
        sol: PlanarSolution,
        views: list[PlanarView],
        f: Frame,
        loc: ViewLocalization | None,
        ext_view: PlanarView | None,
        spec: PlacementSpec,
    ) -> Any:
        fcfg = self._frame_config(s)
        assert fcfg is not None
        sizes = sizes_from_state(s.tag_sizes)
        frames = {x.id: x for x in s.frames}
        ev = {}
        for kind in ("floor", "side"):
            d = s.datums[kind]
            assert d.image_fit is not None and d.frame_id is not None
            ev[kind] = self._evidence(kind, d.frame_id, d.image_fit, self._datum_def(s, kind), manual=(d.provenance == Provenance.MANUAL.value))
        info = {}
        for i in sol.layout.poses:
            _, prov = sizes.size(i)
            info[i] = (prov, sizes.measured_u_mm, sizes.nominal_tolerance_mm)
        ids = set(sol.views) | {f.id}
        ctx = EvalContext(
            sol=sol,
            views=views,
            frame_id=f.id,
            floor=ev["floor"],
            side=ev["side"],
            fcfg=fcfg,
            target=TargetSpec(spec.x_mm, spec.z_mm, spec.width_mm, spec.height_mm, spec.angle_rad or 0.0, spec.reference_point or "center"),  # type: ignore[arg-type]
            ext_loc=loc,
            ext_view=ext_view,
            image_sizes={i: (frames[i].width, frames[i].height) for i in ids if i in frames} | ({f.id: (f.width, f.height)}),
            focal_prior_px={i: float(frames[i].exif["focal_px_prior"]) for i in ids if i in frames and "focal_px_prior" in frames[i].exif},
            size_info=info,
            calibration_sigma=(s.calibration.param_sigma if s.calibration is not None else None),
            all_bias_corrected=all(m.bias_applied for x in s.frames if x.id in ids for m in x.markers if m.accepted and m.tag_id in sol.layout.poses),
            out_of_plane_bound_mm=None,
        )
        with self.cpu.slot(self.settings.limits.request_timeout_seconds):
            return evaluate_uncertainty(ctx, self._ucfg)

    # ---------------------------------------------------------------- overlay
    def overlay(
        self,
        sid: str,
        frame_id: str,
        *,
        view: str = "image",
        placard_ids: list[str] | None = None,
        rect_max_edge: int = 1600,
        with_uncertainty: bool = True,
    ) -> dict[str, Any]:
        s = self._load(sid)
        sol, _ = self._require_registration(s)
        f = self._frame(s, frame_id)
        try:
            H, loc, _ = self._frame_mapping(s, sol, f)
        except GeometryError as exc:
            return {
                "frame_id": f.id, "view": view, "primitives": [], "grid_step_mm": 0.0,
                "report": self._gate_report(s, f.id, None, False).model_dump(mode="json"),
                "error": exc.to_dict(), "notice": RESEARCH_NOTICE,
            }
        return self._overlay_for(s, sol, f, H, view=view, placard_ids=placard_ids, rect_max_edge=rect_max_edge, with_uncertainty=with_uncertainty)

    def _overlay_for(
        self,
        s: Session,
        sol: PlanarSolution,
        f: Frame,
        H: np.ndarray,
        *,
        view: str,
        placard_ids: list[str] | None,
        rect_max_edge: int,
        with_uncertainty: bool,
        ext: tuple[Frame, ViewLocalization, PlanarView] | None = None,
    ) -> dict[str, Any]:
        cam = self._camera(s, f)
        lf = self._local_frame(s)
        ctr = sol.layout.centroid()
        footprint = footprint_for_image(H, f.width, f.height, ctr)
        px_per_mm = float(np.median(local_scale_px_per_mm(H, sol.layout.all_corners())))
        rect_spec = None
        if view == "rect":
            rect_spec = self.rect_spec(s, sol, f, H, "auto", rect_max_edge)
            mapper = rect_mapper(rect_spec)
            vw, vh, scale = rect_spec.width, rect_spec.height, 1.0 / rect_spec.mm_per_px
            corners = np.array([[0, 0], [vw - 1, 0], [vw - 1, vh - 1], [0, vh - 1]], dtype=float)
            visible = rect_spec.rect_to_plane(corners)
        else:
            mapper = image_mapper(H, cam, f.width, f.height, ctr)
            vw, vh, scale = f.width, f.height, px_per_mm
            visible = footprint

        op = self._operation(s) if s.operation else None
        pids = placard_ids if placard_ids is not None else ([s.active_placard_id] if s.active_placard_id else [])
        reports: dict[str, GateReport] = {}
        targets = []
        for pid in pids:
            if pid is None or op is None:
                continue
            rep = self._gate_report(s, f.id, pid, with_uncertainty and pid == s.active_placard_id, ext=ext)
            reports[pid] = rep
            spec = PlacementSpec(**op.placard(pid).resolved())
            if rep.status == GuidanceStatus.BLOCKED.value or lf is None or not lf.valid or not self._spec_complete(spec):
                continue  # fail closed: no target outline without a passing geometry
            rect = self._target(lf, spec)
            targets.append(
                (
                    pid,
                    rect,
                    {
                        "kind": "target" if rep.status != GuidanceStatus.RESEARCH_PREVIEW.value else "target_preview",
                        "status": rep.status,
                        "reference_point": spec.reference_point,
                        "active": pid == s.active_placard_id,
                        "dimensions": pid == s.active_placard_id,
                        "width_mm": spec.width_mm,
                        "height_mm": spec.height_mm,
                        "expanded_mm": rep.uncertainty.expanded_mm if rep.uncertainty else None,
                    },
                )
            )
        tags = []
        for m in f.markers:
            if m.tag_id in sol.layout.poses and m.accepted:
                tags.append({"tag_id": m.tag_id, "accepted": True, "corners_plane": sol.layout.corners(m.tag_id).tolist()})
            elif view != "rect":
                tags.append({"tag_id": m.tag_id, "accepted": False, "corners_px": m.corners_px})
        floor = Line2D(np.asarray(s.datums["floor"].n), s.datums["floor"].d) if "floor" in s.datums else None
        side = Line2D(np.asarray(s.datums["side"].n), s.datums["side"].d) if "side" in s.datums else None
        names = {}
        if op is not None:
            if op.datums.floor is not None:
                names["floor"] = f"{op.datums.floor.id} · {op.datums.floor.name}"
            if op.datums.side is not None:
                names["side"] = f"{op.datums.side.id} · {op.datums.side.name}"
        ov = build_overlay(
            mapper=mapper, view_w=vw, view_h=vh, px_per_mm=scale, visible_plane_pts=visible,
            frame=lf, floor=floor, side=side, targets=targets, tags=tags, labels=names,
        )
        ov.update(
            {
                "frame_id": f.id,
                "view": view,
                "rect": None if rect_spec is None else rect_spec.to_dict(),
                "px_per_mm": scale,
                "reports": {k: v.model_dump(mode="json") for k, v in reports.items()},
                "local_frame": None if s.local_frame is None else s.local_frame.model_dump(mode="json"),
                "frozen": True,
                "notice": RESEARCH_NOTICE,
                "simulated": s.mode == SessionMode.DEMO.value,
            }
        )
        return ov
