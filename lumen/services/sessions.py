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
from ..vision.datums import (
    DatumConfig, FrameContext, LineCandidate, choose_pairs, fit_cylinder_from_silhouettes, fit_plane_line, fit_to_dict, line_width_limit_px,
    match_and_classify, propose_in_frame, step_beside,
)
from ..vision.edgefit import snap_line
from ..vision.imaging import DecodedImage, encode_jpeg, to_gray
from ..vision.markers import MarkerConfig, TagSizeTable
from ..vision.multiview import MVConfig, MVSolution, MVView, solve_multiview
from ..vision.planar import PlanarSolution, PlanarView, ViewLocalization, localize_view, solve_planar
from ..vision.rectify import RectSpec, plan_rectification, rectify_image
from .context import CpuGate, LruCache, utcnow
from .frames import EdgeShiftPool, build_frame, camera_for_frame, decode_for_session, detect_in_image, edge_shift_pool, sizes_from_state, tag_obs
from .jobs import JobRunner

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
        self._mv_cfg = MVConfig()
        self._datum_cfg = DatumConfig()
        self._domain: dict[str, Any] | None = None
        if settings.validated_domain_file:
            try:
                with open(settings.validated_domain_file, "r", encoding="utf-8") as fh:
                    self._domain = json.load(fh)
            except Exception:
                self._domain = None
        self._last_purge = 0.0
        self._purge_lock = threading.Lock()
        self.jobs = JobRunner(self._load, self._mutate, max_workers=1)
        self._live = LruCache(8)
        from ..vision.placard import PlacardConfig

        self._placard_cfg = PlacardConfig()

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
        from .capture import purge_stale_temp

        purge_stale_temp()

    def log_event(self, sid: str, action: str, detail: dict[str, Any] | None = None) -> None:
        self._mutate(sid, lambda s: None, action, detail or {})

    def _load(self, sid: str) -> Session:
        d = self.store.load(sid)
        if d.get("kind") == "calibration":  # a calibration workspace shares the store but is not a session
            raise NotFoundError("session_not_found", "Unknown or expired session.")
        return Session.model_validate(d)

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
            # verification results were measured against the old registration
            s.verifications = []
            s.research.pop("verification_detail", None)
            s.research.pop("photometric", None)  # research evidence computed from the old registration
            # ruler measurements are kept; they are compared again at the next registration
            from .reference import reevaluate_all

            reevaluate_all(s, None)
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
        self._housekeeping()  # expired sessions (and their photos) go even when nobody starts a new one
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
            dets, cam, note = detect_in_image(img, s0.calibration, sizes, self._marker_cfg, client)
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

    def set_frame_attributes(self, sid: str, frame_id: str, *, selected: bool | None = None, torch: str | None = None, pair_id: str | None = None, clear_pair: bool = False) -> Session:
        """Operator edits of a frame: include/exclude, light state, flash pairing."""
        from ..schemas import TorchState

        if torch is not None and torch not in ("on", "off", "unknown"):
            raise LumenValidationError("invalid_torch", "torch must be 'on', 'off' or 'unknown'.")

        def fn(s: Session) -> Session:
            f = self._frame(s, frame_id)
            if selected is not None and bool(selected) != f.selected:
                f.selected = bool(selected)
                self._invalidate(s, registration=True, datums=False)
            if torch is not None:
                f.torch = TorchState(torch)
            if clear_pair:
                f.pair_id = None
            elif pair_id is not None:
                f.pair_id = str(pair_id)[:40]
            return s

        return self._mutate(sid, fn, "frame_attributes", {"frame_id": frame_id, "selected": selected, "torch": torch, "pair": None if clear_pair else pair_id})

    # ------------------------------------------------------------------ video
    def start_video_job(self, sid: str, path: str, source: str = "video_upload", meta: dict[str, Any] | None = None) -> dict[str, Any]:
        from .capture import process_video

        if source not in (FrameSource.VIDEO_UPLOAD.value, FrameSource.LIVE_SWEEP.value):
            raise LumenValidationError("invalid_frame_kind", "Video source must be 'video_upload' or 'live_sweep'.")
        s = self._load(sid)
        if len(s.frames) >= self.settings.limits.max_frames_per_session:
            raise LimitExceededError("too_many_frames", "This session already holds the maximum number of frames.", suggestions=["Remove frames you do not need before adding a video."])
        m = dict(meta or {})
        return self.jobs.submit(sid, "video", lambda job: process_video(self, sid, path, source, m, job))

    def capture_advice(self, sid: str) -> dict[str, Any]:
        from .capture import capture_advice

        return capture_advice(self, self._load(sid))

    # ------------------------------------------------- independent scale check
    def reference_suggestion(self, sid: str) -> dict[str, Any] | None:
        from .reference import suggest_pair

        s = self._load(sid)
        sol, _ = self._require_registration(s)
        return suggest_pair(sol)

    def add_reference_distance(self, sid: str, a: dict[str, Any], b: dict[str, Any], measured_mm: float, u_mm: float = 0.5, instrument: str = "") -> dict[str, Any]:
        from .reference import add_reference

        return add_reference(self, sid, a, b, measured_mm, u_mm, instrument)

    def remove_reference_distance(self, sid: str, rid: str) -> Session:
        from .reference import remove_reference

        return remove_reference(self, sid, rid)

    # ----------------------------------------------------------- verification
    def verify(self, sid: str, frame_id: str, *, placard_id: str | None = None, manual_corners_px: list[list[float]] | None = None, snap: bool = True) -> dict[str, Any]:
        from .verification import verify_frame

        return verify_frame(self, sid, frame_id, placard_id=placard_id, manual_corners_px=manual_corners_px, snap=snap)

    def confirm_verification(self, sid: str, vid: str, confirmed: bool = True) -> Session:
        from .verification import confirm_verification

        return confirm_verification(self, sid, vid, confirmed)

    def delete_verification(self, sid: str, vid: str) -> Session:
        from .verification import delete_verification

        return delete_verification(self, sid, vid)

    def track(self, sid: str, data: bytes, *, placard_ids: list[str] | None = None, meta: dict[str, Any] | None = None, force_full: bool = False) -> dict[str, Any]:
        """Overlay for one live camera frame (never stored); see services/live.py."""
        from .live import track_frame

        return track_frame(self, sid, data, placard_ids=placard_ids, meta=meta, force_full=force_full)

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
                results[f.id] = detect_in_image(img, s.calibration, sizes, self._marker_cfg, {"camera_label": f.camera_label, "lens": f.lens, "zoom": f.zoom})

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
        cam, _ = camera_for_frame(s.calibration, f.width, f.height, f.native_width, f.native_height, f.processing_scale, device=f.camera_label, lens=f.lens, zoom=f.zoom)
        return cam

    def _pool_for(self, s: Session, f: Frame) -> EdgeShiftPool | None:
        """Edge shift to use on the tags of ``f`` that are too small for their own estimate: measured on the
        larger tags of the same photo if there are any, else on those of all registration photos together."""
        return edge_shift_pool(f.markers) or edge_shift_pool(m for x in self._registration_frames(s) for m in x.markers)

    def _views(self, s: Session, frames: list[Frame]) -> list[PlanarView]:
        out = []
        for f in frames:
            pool = self._pool_for(s, f)
            tags = [tag_obs(m, pool) for m in f.markers if m.accepted]
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
        sol, views, _ = self._solve_full(s)
        return sol, views

    def _solve_full(self, s: Session) -> tuple[PlanarSolution, list[PlanarView], dict[str, Any] | None]:
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
            # 3-D bundle adjustment first: it is the only thing that can see a
            # tag that moved between photos or sits off the common plane.
            mv = self._multiview(s, frames, views)
            excluded: dict[int, list[str]] = {}
            excluded_obs: dict[str, set[int]] = {}
            off_all: set[int] = set()
            for attempt in range(3):
                if mv is None or "skipped" in mv:
                    break
                drop = {int(k) for k in mv["excluded_tags"]}
                off = {int(k) for k in mv["checks"].get("coplanarity", {}).get("off_plane_tags", [])}
                # single observations that fit no other photo of the same tag (the tag itself is kept)
                drop_obs = {(fid, int(t)) for fid, ts in (mv.get("excluded_observations") or {}).items() for t in ts}
                for k, reasons in mv["excluded_tags"].items():
                    excluded.setdefault(int(k), list(reasons))
                for k in off:
                    excluded.setdefault(k, ["off_the_common_plane"])
                for fid, t in drop_obs:
                    excluded_obs.setdefault(fid, set()).add(t)
                if drop or off or drop_obs:
                    views = [PlanarView(v.frame_id, [t for t in v.tags if t.tag_id not in (drop | off) and (v.frame_id, t.tag_id) not in drop_obs], v.camera, v.mode) for v in views]
                    views = [v for v in views if v.tags]
                    if not views:
                        raise GeometryError("no_consistent_tags", "After removing inconsistent tags none is left to establish scale.")
                if not off or attempt == 2:
                    break
                # The bundle adjustment sets aside a tag that moved and solves again by itself.  A tag off the
                # common plane it only reports: its solution still contains that tag, and so do the plane fitted
                # through "all" tags, the flatness figures and the heights of the datum lines measured from that
                # plane.  What is used further on has to describe the tags that carry the result: solve again
                # without it.
                off_all |= off
                mv = self._multiview(s, frames, views)
                if mv is None:
                    mv = {"skipped": "after setting aside the tags off the common plane, fewer than two photos show the remaining tags", "excluded_tags": {}, "checks": {}, "flags": []}
            if mv is not None and off_all:
                mv["excluded_tags"] = {str(k): v for k, v in sorted(excluded.items())}
                mv["excluded_observations"] = {fid: sorted(ts) for fid, ts in sorted(excluded_obs.items())}
                mv["flags"] = [*mv.get("flags", []), "off_plane_tags_excluded"]
                cop = mv["checks"].setdefault("coplanarity", {})
                # tags found off the common plane in an earlier pass (set aside), in front of any the last pass still names
                cop["off_plane_tags"] = sorted(off_all | {int(k) for k in cop.get("off_plane_tags", [])})
                cop["set_aside_before_this_solution"] = sorted(off_all)
            sol = solve_planar(views, size_u_mm=size_u)
            for k, reasons in excluded.items():
                sol.excluded_tags.setdefault(int(k), list(reasons))
        used = [v for v in views if v.frame_id in sol.views]
        # keep only tags that survived the consistency checks
        used = [PlanarView(v.frame_id, [t for t in v.tags if t.tag_id in sol.layout.poses], v.camera, v.mode) for v in used]
        out = (sol, used, mv)
        self._solutions.put(key, out)
        return out

    def _multiview(self, s: Session, frames: list[Frame], views: list[PlanarView]) -> dict[str, Any] | None:
        """3-D diagnostics of the capture (needs at least two views).

        With a calibration the intrinsics are fixed and, when the views are
        diverse enough, an independent focal length is self-calibrated from the
        tags and compared with the profile.  Without one, the focal length is
        self-calibrated (labelled *inferred*) just to obtain the 3-D checks.
        """
        if len(views) < 2:
            return None
        by_id = {f.id: f for f in frames}
        cams = [v.camera for v in views]
        calibrated = all(c is not None for c in cams)
        same_geom = len({(by_id[v.frame_id].width, by_id[v.frame_id].height) for v in views}) == 1
        if not same_geom:
            return {"skipped": "photos have different pixel sizes; 3-D checks need one camera configuration", "excluded_tags": {}, "checks": {}, "flags": ["mixed_resolutions"]}
        mvviews = []
        for v in views:
            f = by_id[v.frame_id]
            raw = {m.tag_id: np.asarray(m.corners_px, dtype=float) for m in f.markers if m.accepted}
            mvviews.append(MVView(v.frame_id, list(v.tags), v.camera, raw, (f.width, f.height)))
        try:
            if calibrated:
                sol = solve_multiview(mvviews, camera=cams[0], cfg=self._mv_cfg)
            else:
                sol = solve_multiview(mvviews, refine_intrinsics=("f",), cfg=self._mv_cfg)
        except GeometryError as exc:
            return {"skipped": exc.message, "excluded_tags": {}, "checks": {}, "flags": ["multiview_failed"]}
        out = sol.to_dict()
        out["intrinsics_provenance"] = "measured" if calibrated else "inferred"
        out["calibrated"] = calibrated
        # independent focal-length check of an attached profile
        if calibrated and sol.checks.get("diversity", {}).get("diverse") and len(sol.cameras) >= 3:
            try:
                keep = [v for v in mvviews if v.frame_id in sol.cameras]
                keep = [MVView(v.frame_id, [t for t in v.tags if t.tag_id in sol.tags], v.camera, v.raw, v.image_size) for v in keep]
                sc = solve_multiview(keep, camera=cams[0], refine_intrinsics=("f",), cfg=self._mv_cfg, checks=False)
                a = sc.param_index.get("intr:f")
                sf = float(np.sqrt(max(sc.cov[a[0], a[0]], 0.0))) if a else float("inf")
                f_cal = 0.5 * (cams[0].fx + cams[0].fy)
                rel = (sc.intrinsics.fx - f_cal) / f_cal
                ok = abs(rel) <= max(4.0 * sf / f_cal, 0.004)
                out["calibration_check"] = {
                    "f_profile": f_cal, "f_from_tags": sc.intrinsics.fx, "sigma_f": sf, "relative_difference": rel, "consistent": bool(ok),
                    "note": "Focal length re-estimated from the tags with the profile's distortion and principal point held fixed.",
                }
                if not ok:
                    out["flags"].append("calibration_inconsistent")
            except GeometryError:
                pass
        return out

    def register(self, sid: str) -> dict[str, Any]:
        s = self._load(sid)
        self._redetect_stale(sid)
        s = self._load(sid)
        sol, views, mv = self._solve_full(s)
        frames = {f.id: f for f in s.frames}
        ref = self._pick_reference(sol, frames)
        summary = sol.to_dict()
        summary["reference_frame_id"] = ref
        summary["frames_used"] = sorted(sol.views)
        summary["computed_at"] = utcnow()
        summary["method"] = "planar_joint" + ("+multiview_ba" if mv and "skipped" not in mv else "")
        summary["multiview"] = mv
        # how the dark-side edge shift of the tag borders was handled, per observation used
        obs = [t for v in views for t in v.tags]
        pool_all = edge_shift_pool(m for x in self._registration_frames(s) for m in x.markers)
        summary["edge_shift"] = {
            "own_estimate": sum(1 for t in obs if t.bias_corrected),
            "pooled": sum(1 for t in obs if t.bias_pooled),
            "uncorrected": sum(1 for t in obs if not t.bias_corrected and not t.bias_pooled),
            "pool": None if pool_all is None else pool_all.to_dict(),
        }
        surface = self._surface_from(mv, sol)
        px_per_mm = {}
        for fid, v in sol.views.items():
            px_per_mm[fid] = float(np.median(local_scale_px_per_mm(v.H, sol.layout.all_corners())))
        summary["px_per_mm"] = px_per_mm

        def fn(s2: Session) -> None:
            from .reference import reevaluate_all

            s2.registration = summary
            s2.surface = surface
            reevaluate_all(s2, sol)  # the operator's ruler measurements are kept and checked against the new layout
            # datums measured in frames that are no longer registered are dropped
            for k in [k for k, d in s2.datums.items() if d.frame_id not in sol.views]:
                del s2.datums[k]
            self._rebuild_local_frame(s2, sol)

        self._mutate(sid, fn, "registered", {"n_tags": sol.n_tags, "n_views": len(sol.views), "flags": sol.flags})
        return summary

    @staticmethod
    def _surface_from(mv: dict[str, Any] | None, sol: PlanarSolution) -> Any:
        from ..schemas import SurfaceModel

        notes: list[str] = []
        if mv is None or "skipped" in (mv or {}):
            how = "crossvalidated_single_view" if sol.independently_checked else "assumed"
            if mv and mv.get("skipped"):
                notes.append(f"3-D check skipped: {mv['skipped']}")
            return SurfaceModel(kind="plane", planarity=how, notes=notes)
        cop = mv["checks"].get("coplanarity", {})
        verdict = cop.get("verdict")
        keep = [str(i) for i in sol.layout.poses]
        offs = [abs(cop["tags"][k]["offset_mm"]) + 2.0 * cop["tags"][k]["offset_sigma_mm"] for k in keep if k in cop.get("tags", {})]
        set_aside = cop.get("set_aside_before_this_solution") or []
        if verdict == "coplanar" and set_aside:
            # the remaining tags were solved again without the ones off the plane, and agree on one plane
            how = "verified_multiview_after_exclusion"
            notes.append("tags off the common plane were excluded: " + ", ".join(f"ID {i}" for i in set_aside))
        elif verdict == "coplanar":
            how = "verified_multiview"
        elif verdict == "not_coplanar":
            # also after setting tags aside the rest does not lie on one plane, or no majority could be identified
            how = "violated"
            if set_aside:
                notes.append("tags off the common plane were excluded (" + ", ".join(f"ID {i}" for i in set_aside) + "), and the remaining tags still do not lie on one plane")
        else:
            how = "crossvalidated_single_view" if sol.independently_checked else "assumed"
            notes.append("views are too similar to establish coplanarity in 3-D")
        return SurfaceModel(
            kind="plane",
            planarity=how,
            out_of_plane_rms_mm=cop.get("rms_mm"),
            out_of_plane_max_mm=(float(max(offs)) if offs and how.startswith("verified") else None),
            notes=notes,
        )

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
        pool = self._pool_for(s, f)
        tags = [tag_obs(m, pool) for m in f.markers if m.accepted]
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
        sol, _views, mv = self._require_registration_full(s)
        f = self._frame(s, frame_id)
        if f.id not in sol.views:
            raise ConflictError("frame_not_registered", "Datums must be drawn on a frame that is part of the registration.")
        H = sol.views[f.id].H
        cam = self._camera(s, f)
        seg = self._to_ideal(s, sol, f, H, np.array([p0, p1]), space, rect)
        if float(np.linalg.norm(seg[1] - seg[0])) < 30.0:
            raise LumenValidationError("datum_too_short", "The drawn line is too short; draw along at least a few centimetres of the edge.")
        notes: list[str] = []
        cand: LineCandidate | None = None
        if snap:
            gray = to_gray(self._image(sid, f).bgr)
            with self.cpu.slot(self.settings.limits.request_timeout_seconds):
                seg_plane = apply_h(np.linalg.inv(H), seg)
                scale = float(local_scale_px_per_mm(H, seg_plane.mean(axis=0)[None])[0])
                r = snap_line(gray, seg[0], seg[1], camera=cam, search_px=search_px, max_bar_px=line_width_limit_px(self._datum_cfg, scale))
                if r.ok and r.fit is not None:
                    line, ends = fit_plane_line(r.fit, np.linalg.inv(H))
                    cand = LineCandidate(
                        id=f"{f.id}:drawn", frame_id=f.id, kind=r.kind, fit=r.fit, line=line, endpoints=ends,
                        length_mm=float(np.linalg.norm(ends[1] - ends[0])), contrast=r.fit.contrast,
                        sigma_blur_px=r.fit.sigma_blur, distinctness=r.distinctness,
                    )
                    cand.notes.append(f"{r.n_candidates} feature(s) in the search band") if r.n_candidates > 1 else None
                    ctxs = self._frame_contexts(sid, s, sol, mv)
                    if len(ctxs) > 1:
                        match_and_classify(cand, ctxs, self._datum_cfg, self._tag_plane(sol, mv)[1])
                else:
                    notes.append(f"no edge found near the drawn line ({r.reason}); kept as drawn")
        if cand is not None:
            obs = self._datum_from_candidate(s, sol, kind, cand, n_candidates=None)
        else:
            t = seg[1] - seg[0]
            L = float(np.linalg.norm(t))
            t /= L
            fit_d = {"mid": (0.5 * (seg[0] + seg[1])).tolist(), "n": [-t[1], t[0]], "length": L, "var_d": 0.0, "var_theta": 0.0, "cov_d_theta": 0.0}
            obs = self._datum_from_fit(s, sol, kind, f.id, fit_d, feature_kind="manual", provenance=Provenance.MANUAL, distinct=None, n_cand=None, notes=notes, plane_evidence=None)

        def fn(s2: Session) -> None:
            s2.datums[kind] = obs
            self._rebuild_local_frame(s2, sol)

        self._mutate(sid, fn, "datum_proposed", {"kind": kind, "frame_id": f.id, "feature": obs.feature_kind, "provenance": obs.provenance})
        return obs

    def _require_registration_full(self, s: Session) -> tuple[PlanarSolution, list[PlanarView], dict[str, Any] | None]:
        if not s.registration:
            raise ConflictError("not_registered", "The session has not been registered yet.", suggestions=["Run the analysis after capturing photos."])
        return self._solve_full(s)

    def _frame_contexts(self, sid: str, s: Session, sol: PlanarSolution, mv: dict[str, Any] | None) -> dict[str, FrameContext]:
        """Per-photo data for multi-view datum reasoning (pixels, mapping, 3-D pose)."""
        from ..geometry.se3 import Pose

        frames = {f.id: f for f in s.frames}
        poses: dict[str, Pose] = {}
        K = None
        T_w_p, _ = self._tag_plane(sol, mv)
        if T_w_p is not None and mv is not None:
            for fid, d in mv["cameras"].items():
                poses[fid] = Pose.from_dict(d) @ T_w_p  # T_cam_plane
            i = mv["intrinsics"]
            K = np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]])
        out: dict[str, FrameContext] = {}
        for fid, v in sol.views.items():
            f = frames[fid]
            blur = [m.sigma_blur_px for m in f.markers if m.accepted and m.sigma_blur_px > 0]
            out[fid] = FrameContext(
                frame_id=fid,
                gray=to_gray(self._image(sid, f).bgr),
                camera=self._camera(s, f),
                H=v.H,
                pose=poses.get(fid),
                K=K if fid in poses else None,
                tag_blur_px=float(np.median(blur)) if blur else 1.0,
            )
        return out

    @staticmethod
    def _tag_plane(sol: PlanarSolution, mv: dict[str, Any] | None) -> tuple[Any, np.ndarray | None]:
        """Reference plane for 3-D reasoning about datum features, and how well the tags define it.

        The bundle adjustment gives every tag its own position and tilt.  Heights are measured from the plane
        fitted through *all* of them, not from the face of one 50 mm tag: the tilt error of a single tag grows
        to a millimetre a few hundred millimetres away.  In-plane coordinates stay those of the gauge tag.

        Returns ``(T_world_plane, covariance of (a, b, c) of the plane z = a x + b y + c in that frame)``;
        ``(None, None)`` without a multi-view solution.  With a single tag the plane is that tag's face and no
        covariance is given.
        """
        from ..geometry.se3 import Pose

        g = str(sol.layout.gauge_id)
        if not mv or "skipped" in mv or g not in mv.get("tags", {}):
            return None, None
        T_w_g = Pose.from_dict(mv["tags"][g])
        cop = (mv.get("checks") or {}).get("coplanarity") or {}
        per = cop.get("tags") or {}
        if cop.get("normal") is None or len(mv["tags"]) < 2:
            return T_w_g, None
        z = np.asarray(cop["normal"], dtype=float)
        z = z / np.linalg.norm(z)
        d = float(cop["d"])
        if z @ T_w_g.R[:, 2] < 0:
            z, d = -z, -d
        x = T_w_g.R[:, 0] - (T_w_g.R[:, 0] @ z) * z
        x = x / np.linalg.norm(x)
        T_w_p = Pose(np.stack([x, np.cross(z, x), z], axis=1), T_w_g.t - (z @ T_w_g.t - d) * z)
        # how well the tags pin that plane down: each tag gives a height at its centre and a slope
        so = [float(v.get("offset_sigma_mm") or 0.0) for v in per.values()]
        st = [float(np.radians(v.get("tilt_sigma_deg") or 0.0)) for v in per.values()]
        so_d = float(np.median([v for v in so if v > 0])) if any(v > 0 for v in so) else 0.3
        st_d = float(np.median([v for v in st if v > 0])) if any(v > 0 for v in st) else float(np.radians(0.3))
        rows, wts = [], []
        T_p_w = T_w_p.inv()
        for k, td in mv["tags"].items():
            c = T_p_w.apply(Pose.from_dict(td).t)
            v = per.get(str(k), {})
            s_off = max(float(v.get("offset_sigma_mm") or 0.0) or so_d, 0.02)  # the gauge tag is given the typical value
            s_tilt = max(float(np.radians(v.get("tilt_sigma_deg") or 0.0)) or st_d, 1e-4)
            rows += [[c[0], c[1], 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            wts += [1.0 / s_off**2, 1.0 / s_tilt**2, 1.0 / s_tilt**2]
        A = np.asarray(rows)
        N = A.T @ (A * np.asarray(wts)[:, None])
        return T_w_p, (np.linalg.inv(N) if np.linalg.cond(N) < 1e12 else None)

    @staticmethod
    def _datum_cfg_on_plane_mm() -> float:
        from ..vision.datums import DatumConfig

        return DatumConfig().on_plane_mm

    def _datum_from_candidate(self, s: Session, sol: PlanarSolution, kind: str, c: LineCandidate, n_candidates: int | None) -> DatumObservation:
        ddef = self._datum_def(s, kind)
        declared = None if ddef is None else ddef.on_target_plane
        evidence = {"on_plane": "verified_multiview", "off_plane": "off_plane", "view_dependent": "off_plane"}.get(c.classification)
        if evidence is None:
            evidence = "configured" if declared else "unverified"
        notes = list(c.notes)
        if c.classification == "on_plane":
            notes.append(f"stays fixed on the panel face across {1 + len(c.matches)} photos (spread {c.spread_mm:.2f} mm; height {c.height_mm:+.1f} mm, resolved to ±{c.height_margin_mm:.1f} mm)")
        fit_d = fit_to_dict(c.fit, self._marker_cfg.covariance_inflation)
        fit_d["classification"] = c.classification
        fit_d["height_mm"] = c.height_mm
        fit_d["n_views"] = 1 + len(c.matches)
        return self._datum_from_fit(
            s, sol, kind, c.frame_id, fit_d, feature_kind=c.kind, provenance=Provenance.MEASURED,
            distinct=c.distinctness, n_cand=n_candidates, notes=notes, plane_evidence=evidence,
        )

    def _datum_from_fit(
        self,
        s: Session,
        sol: PlanarSolution,
        kind: str,
        frame_id: str,
        fit_d: dict[str, Any],
        *,
        feature_kind: str,
        provenance: Provenance,
        distinct: float | None,
        n_cand: int | None,
        notes: list[str],
        plane_evidence: str | None,
    ) -> DatumObservation:
        ddef = self._datum_def(s, kind)
        declared = None if ddef is None else ddef.on_target_plane
        ev = self._evidence(kind, frame_id, fit_d, ddef, manual=(provenance == Provenance.MANUAL))
        interior = sol.layout.centroid()
        H = sol.views[frame_id].H
        Hinv = np.linalg.inv(H)
        feature_line = DatumEvidence(kind, frame_id, ev.mid, ev.n, ev.length, ev.cov, in_plane=ev.in_plane).plane_line(Hinv, interior)
        line = ev.plane_line(Hinv, interior)
        ends = ev.points() if ev.in_plane else apply_h(Hinv, ev.points())
        if ev.in_plane:
            sigma_perp = float(np.sqrt(max(fit_d.get("var_d", 0.0), 0.0)))
        else:
            mm_per_px = 1.0 / max(float(local_scale_px_per_mm(H, ends.mean(axis=0)[None])[0]), 1e-9)
            sigma_perp = float(np.sqrt(max(fit_d.get("var_d", 0.0), 0.0)) * mm_per_px)
            half_px = float(fit_d.get("line_half_width_px") or 0.0)
            if half_px > 0.0:
                notes = [*notes, f"a line about {2.0 * half_px * mm_per_px:.1f} mm wide, resolved into its two edges: the datum is its centre line"]
        return DatumObservation(
            kind=kind,
            n=line.n.tolist(),
            d=line.d,
            provenance=provenance,
            confirmed=False,
            feature_kind=feature_kind,
            frame_id=frame_id,
            endpoints=np.asarray(ends).tolist(),
            sigma_perp_mm=sigma_perp,
            sigma_angle_rad=float(np.sqrt(max(fit_d.get("var_theta", 0.0), 0.0))),
            offset_applied_mm=ev.offset_mm,
            on_plane_evidence=plane_evidence or ("configured" if declared else "unverified"),
            distinctness=distinct,
            n_candidates=n_cand,
            notes=notes,
            image_fit=fit_d,
            feature_n=feature_line.n.tolist(),
            feature_d=feature_line.d,
        )

    # -------------------------------------------------- automatic proposals
    def propose_datums(self, sid: str, apply: bool = True) -> dict[str, Any]:
        """Find floor / side datum candidates automatically (prompt 06).

        Candidates are measured in one or two well-placed photos, re-found in
        all others and classified in 3-D.  Unambiguous best candidates are set
        as *unconfirmed* datums; the operator still has to review and confirm.
        """
        s = self._load(sid)
        sol, _views, mv = self._require_registration_full(s)
        op = self._operation(s)
        frames = {f.id: f for f in s.frames}
        with self.cpu.slot(self.settings.limits.job_timeout_seconds):
            ctxs = self._frame_contexts(sid, s, sol, mv)
            home = self._home_frames(sol, ctxs, frames)
            tag_c = [sol.layout.corners(t) for t in sol.layout.poses]
            cands: list[LineCandidate] = []
            for fid in home:
                f = frames[fid]
                H = sol.views[fid].H
                spec = self.rect_spec(s, sol, f, H, "photo", 1400)
                rect, _ = rectify_image(self._image(sid, f).bgr, self._camera(s, f), H, spec)
                for c in propose_in_frame(fid, self._image(sid, f).bgr, self._camera(s, f), H, rect, spec, tag_c, self._datum_cfg):
                    c.id = f"c{len(cands)}"
                    if len(ctxs) > 1:
                        match_and_classify(c, ctxs, self._datum_cfg, self._tag_plane(sol, mv)[1])
                    # the same feature seen from a second home photo is one candidate
                    dup = False
                    for o in cands:
                        if o.classification != "view_dependent" and c.classification != "view_dependent" and o.line.angle_to(c.line) < np.deg2rad(1.0) and abs(float(o.line.signed_distance(c.endpoints.mean(axis=0)))) < 1.5:
                            dup = True
                            break
                        if o.frame_id in c.matches and c.kind == o.kind and o.line.angle_to(c.matches[o.frame_id][1]) < np.deg2rad(1.0) and abs(float(o.line.signed_distance(c.matches[o.frame_id][2].mean(axis=0)))) < 1.5:
                            dup = True
                            break
                    if not dup:
                        cands.append(c)
            centers = np.array([[p[0], p[1]] for p in sol.layout.poses.values()])
            d = op.datums
            nominal = float(np.degrees(d.nominal_angle.rad)) if d.nominal_angle is not None else 90.0
            ref = sol.views[home[0]]
            up = apply_h(np.linalg.inv(ref.H), (apply_h(ref.H, centers.mean(axis=0)[None])[0] + np.array([0.0, -5.0]))[None])[0] - centers.mean(axis=0)
            up = up / max(np.linalg.norm(up), 1e-9)
            floor_feat = d.floor.feature if d.floor else None
            side_feat = d.side.feature if d.side else None
            curve_side = side_feat in ("curve_tangent", "curve_outer_extreme")
            prop = choose_pairs(
                cands, centers, nominal_angle_deg=nominal, expected_side=(d.side.side_of_panel if d.side else None),
                floor_on_plane=(d.floor.on_target_plane if d.floor else None),
                side_on_plane=(None if curve_side else (d.side.on_target_plane if d.side else None)),
                image_up_plane=up, cfg=self._datum_cfg,
            )
            cyl = None
            cyl_obs = None
            if curve_side:
                cyl, cyl_obs = self._cylinder_side(s, sol, mv, ctxs, cands, centers, d.side)
        out = prop.to_dict()
        out["candidates"] = [c.to_dict() for c in cands]
        out["home_frames"] = home
        out["cylinder"] = None if cyl is None else cyl.to_dict()
        out["floor_feature"] = floor_feat
        out["side_feature"] = side_feat
        by_id = {c.id: c for c in cands}
        applied: dict[str, str] = {}
        new_obs: dict[str, DatumObservation] = {}
        if apply:
            if prop.floor:
                c = by_id[prop.floor[0].id]
                o = self._datum_from_candidate(s, sol, "floor", c, n_candidates=len(prop.floor))
                if prop.ambiguous_floor:
                    o.notes.append("another candidate of similar strength exists: choose the intended feature")
                    o.distinctness = min(o.distinctness or 1.0, 1.0)
                self._flag_step_beside(o, c, cands, centers)
                new_obs["floor"] = o
                applied["floor"] = c.id
            if curve_side:
                if cyl_obs is not None:
                    new_obs["side"] = cyl_obs
                    applied["side"] = "cylinder_model"
            elif prop.side:
                c = by_id[prop.side[0].id]
                o = self._datum_from_candidate(s, sol, "side", c, n_candidates=len(prop.side))
                if prop.ambiguous_side:
                    o.notes.append("another candidate of similar strength exists: choose the intended feature")
                    o.distinctness = min(o.distinctness or 1.0, 1.0)
                self._flag_step_beside(o, c, cands, centers)
                new_obs["side"] = o
                applied["side"] = c.id
        out["applied"] = applied
        missing = [k for k in ("floor", "side") if k not in applied]
        hints = []
        fixed = [c for c in cands if c.classification != "view_dependent"]
        if not curve_side and len(missing) == 2 and fixed:
            # Lines were found, but not two that belong together.  Which of the two datums is the missing one
            # cannot be told from a single line, so the hint must not claim that it is the floor (or the side).
            found = "Only one straight line was found in the photos" if len(fixed) == 1 else f"{len(fixed)} straight lines were found in the photos, but no two of them fit together"
            hints.append(
                f"{found}. A floor line and a side line that meet at the angle the record gives, with the tags between them, are both needed: "
                "one of them is too faint, hidden or outside the photos. Retake with both clearly in view, or draw the lines by hand (with snap on they are fitted to the edge)."
            )
        else:
            if "floor" in missing:
                hints.append("No floor line was found: include the floor seam in at least one photo (tilt the phone down).")
            if "side" in missing:
                hints.append(
                    "The rounded edge could not be modelled: capture 5–9 photos while moving sideways around the edge." if curve_side
                    else "No side line was found: include the side seam/edge in at least one photo."
                )
        out["hints"] = hints

        def fn(s2: Session) -> None:
            s2.research["datum_proposals"] = out
            if apply:
                for k, o in new_obs.items():
                    s2.datums[k] = o
                if cyl is not None and cyl.ok:
                    s2.surface.kind = "plane_with_cylindrical_edge"
                    s2.surface.cylinder = cyl.to_dict()
                self._rebuild_local_frame(s2, sol)

        self._mutate(sid, fn, "datums_auto_proposed", {"applied": applied, "n_candidates": len(cands)})
        return out

    @staticmethod
    def _home_frames(sol: PlanarSolution, ctxs: dict[str, FrameContext], frames: dict[str, Frame]) -> list[str]:
        """Photos in which candidates are searched: the reference photo and,
        when camera positions are known, the one seen from the most different
        direction (features hidden in one are often visible in the other)."""
        ref = SessionService._pick_reference(sol, frames)
        out = [ref]
        if ctxs.get(ref) is not None and ctxs[ref].pose is not None:
            c0 = ctxs[ref].pose.center  # type: ignore[union-attr]
            others = [(float(np.linalg.norm(c.pose.center - c0)), fid) for fid, c in ctxs.items() if fid != ref and c.pose is not None]
            if others:
                dist, fid = max(others)
                if dist > 80.0:
                    out.append(fid)
        return out

    def _flag_step_beside(self, o: DatumObservation, c: LineCandidate, cands: list[LineCandidate], centers: np.ndarray) -> None:
        """Mark an automatically applied datum that has a fixed edge at another depth beside it."""
        nb = step_beside(c, cands, centers.mean(axis=0), self._datum_cfg)
        if nb is None:
            return
        o.depth_neighbour = nb
        hh = float(nb["height_mm"])
        where = "right beside it" if nb["relation"] == "beside" else f"{abs(float(nb['gap_mm'])):.0f} mm beyond it"
        o.notes.append(
            f"a fixed edge about {abs(hh):.0f} mm {'in front of' if hh > 0 else 'behind'} the panel face runs {where}: "
            "this line may be the edge of a trim, a moulding or a recess — choose the intended feature yourself"
        )

    def select_datum_candidate(self, sid: str, kind: str, candidate_id: str) -> DatumObservation:
        """Operator picks one of the automatically found alternatives."""
        s = self._load(sid)
        sol, _views, _mv = self._require_registration_full(s)
        props = s.research.get("datum_proposals") or {}
        cd = next((c for c in props.get("candidates", []) if c["id"] == candidate_id), None)
        if cd is None or kind not in ("floor", "side"):
            raise NotFoundError("candidate_not_found", "That datum candidate is no longer available; search again.")
        evidence = {"on_plane": "verified_multiview", "off_plane": "off_plane", "view_dependent": "off_plane"}.get(cd["classification"])
        fit_d = dict(cd["image_fit"])
        fit_d.update(classification=cd["classification"], height_mm=cd["height_mm"], n_views=cd["n_views"])
        obs = self._datum_from_fit(
            s, sol, kind, cd["frame_id"], fit_d, feature_kind=cd["kind"], provenance=Provenance.MEASURED,
            distinct=None, n_cand=None, notes=list(cd.get("notes", [])) + ["chosen by the operator among automatic candidates"], plane_evidence=evidence,
        )

        def fn(s2: Session) -> None:
            s2.datums[kind] = obs
            self._rebuild_local_frame(s2, sol)

        self._mutate(sid, fn, "datum_candidate_selected", {"kind": kind, "candidate": candidate_id})
        return obs

    def _cylinder_side(
        self,
        s: Session,
        sol: PlanarSolution,
        mv: dict[str, Any] | None,
        ctxs: dict[str, FrameContext],
        cands: list[LineCandidate],
        centers: np.ndarray,
        ddef: Any,
    ) -> tuple[Any, DatumObservation | None]:
        """Side datum of a rounded edge from its silhouettes (3-D evidence)."""
        interior = centers.mean(axis=0)
        radius_cfg = None if (ddef is None or ddef.radius is None) else float(ddef.radius.mm)
        # silhouettes: view-dependent edges, plus edges that triangulate to a
        # fixed line *behind* the face (a tight bend looks almost stationary)
        sil = [
            c for c in cands
            if c.kind == "step" and not c.soft and len(c.matches) >= 1
            and (
                c.classification == "view_dependent"
                or (c.classification == "off_plane" and (c.height_mm or 0.0) < 0.0)
                # ... also when the photos do not resolve the height well enough to call it off the plane;
                # the cylinder fit below has its own acceptance tests
                or (c.classification == "unverified" and c.height_mm is not None and c.height_mm < -self._datum_cfg_on_plane_mm())
            )
        ]
        # also allow a single calibrated view when the radius is configured
        if not sil and radius_cfg is not None:
            sil = [c for c in cands if c.classification == "unverified" and c.kind == "step" and ctxs[c.frame_id].pose is not None]
        best = None
        best_c = None
        for c in sorted(sil, key=lambda c: -c.score):
            obs = []
            for fid in [c.frame_id, *c.matches]:
                fc = ctxs[fid]
                if fc.pose is None:
                    continue
                ends = c.endpoints if fid == c.frame_id else c.matches[fid][2]
                e = c.fit if fid == c.frame_id else c.matches[fid][0]
                mm_per_px = 1.0 / max(float(local_scale_px_per_mm(fc.H, ends.mean(axis=0)[None])[0]), 1e-9)
                sig = max(float(np.sqrt(max(e.var_d, 0.0))) * mm_per_px * 2.0, 0.08)
                obs.append((fc.pose.center, ends, sig))
            fit = fit_cylinder_from_silhouettes(obs, interior, radius=None)
            if radius_cfg is not None and (not fit.ok or fit.n_views < 3):
                fit = fit_cylinder_from_silhouettes(obs, interior, radius=radius_cfg)
            if fit.ok and (best is None or fit.rms_mm < best.rms_mm):
                best, best_c = fit, c
        if best is None or best_c is None or best.tangent is None:
            from ..vision.datums import CylinderFit

            return CylinderFit(False, "no_silhouette_found"), None
        if radius_cfg is not None and not best.radius_fixed:
            dev = abs(best.radius - radius_cfg)
            if dev > max(3.0 * best.sigma_radius_mm, 0.1 * radius_cfg):
                best.notes.append(f"measured radius {best.radius:.1f} mm differs from the configured {radius_cfg:.1f} mm")
        line = best.tangent.oriented_towards(interior)
        feature = ddef.feature if ddef is not None else "curve_tangent"
        if feature == "curve_outer_extreme":
            line = line.offset(-best.radius)  # outermost ruling projected onto the face (bend of 90° or more)
        ends = best.endpoints if best.endpoints is not None else np.stack([line.point - 100 * line.direction, line.point + 100 * line.direction])
        ends = line.project(ends)
        L = float(np.linalg.norm(ends[1] - ends[0]))
        var_d = best.sigma_tangent_mm**2 + (best.sigma_radius_mm**2 if feature == "curve_outer_extreme" else 0.0) + best.rms_mm**2
        fit_d = {
            "in_plane": True, "mid": ends.mean(axis=0).tolist(), "n": line.n.tolist(), "length": L,
            "var_d": float(var_d), "var_theta": float(np.radians(best.sigma_angle_deg) ** 2), "cov_d_theta": 0.0,
            "radius_mm": best.radius, "radius_fixed": best.radius_fixed, "n_views": best.n_views, "rms_mm": best.rms_mm,
        }
        inferred = bool(mv and mv.get("intrinsics_provenance") == "inferred")
        notes = [f"tangent line of a cylindrical edge fitted to {best.n_views} silhouettes: radius {best.radius:.1f} mm" + (" (configured)" if best.radius_fixed else f" ± {best.sigma_radius_mm:.1f} mm")] + best.notes
        by_id = {f.id: f for f in s.frames}
        lit = [fid for fid in [best_c.frame_id, *best_c.matches] if fid in by_id and getattr(by_id[fid].torch, "value", by_id[fid].torch) == "on"]
        if lit:
            # With the light next to the lens the surface darkens toward its outline, so the outline is found
            # too far in (about +0.4 to +0.9 mm on simulated photos, more in a dark room).
            notes.append(f"{len(lit)} of the outline photos were taken with the phone light on: the outline of a lit rounded edge is found too far in. Use photos without the light for this datum.")
        if inferred:
            notes.append("camera positions come from a self-calibrated focal length (inferred)")
        obs = self._datum_from_fit(
            s, sol, "side", best_c.frame_id, fit_d, feature_kind="cylinder_model",
            provenance=(Provenance.INFERRED if inferred else Provenance.MEASURED), distinct=None, n_cand=None, notes=notes, plane_evidence="cylinder_model",
        )
        return best, obs

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
            manual=manual and not bool(fit.get("in_plane")),
            in_plane=bool(fit.get("in_plane")),
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
        unc_override: Any | None = None,
    ) -> GateReport:
        inp = GateInputs(demo=(s.mode == SessionMode.DEMO.value))
        pid = placard_id or s.active_placard_id
        spec: PlacementSpec | None = None
        op: OperationDefinition | None = None
        if s.operation:
            op = self._operation(s)
            inp.operation_selected = True
            inp.record_status = op.status
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
            mv = (s.registration or {}).get("multiview")
            if mv and "skipped" not in mv:
                inp.multiview = True
                inp.multiview_flags = list(mv.get("flags", []))
                inp.excluded_observations = [f"tag {t} in photo {fid}" for fid, ts in sorted((mv.get("excluded_observations") or {}).items()) for t in ts]
                inp.coplanarity_verdict = mv["checks"].get("coplanarity", {}).get("verdict")
                inp.coplanarity_power_mm = mv["checks"].get("coplanarity", {}).get("resolving_power_mm")
                cc = mv.get("calibration_check")
                inp.calibration_consistent = None if cc is None else bool(cc["consistent"])
                ho = mv["checks"].get("held_out", {})
                inp.held_out_consistent = ho.get("consistent")
                inp.plane_violated = s.surface.planarity == "violated"
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

        from .reference import summary as _ref_summary

        inp.reference_consistent, inp.reference_message = _ref_summary(s) if sol is not None else (None, "")
        inp.datums_present = ("floor" in s.datums, "side" in s.datums)
        inp.datums_confirmed = (bool(s.datums.get("floor") and s.datums["floor"].confirmed), bool(s.datums.get("side") and s.datums["side"].confirmed))
        inp.datum_distinctness = (
            s.datums["floor"].distinctness if "floor" in s.datums else None,
            s.datums["side"].distinctness if "side" in s.datums else None,
        )
        inp.datum_depth_neighbour = (
            s.datums["floor"].depth_neighbour if "floor" in s.datums else None,
            s.datums["side"].depth_neighbour if "side" in s.datums else None,
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
            if unc_override is not None and inp.target_in_view and sol.consistent:
                # evaluated moments ago for a nearly identical view of the same tags (live guidance)
                unc = unc_override
            elif with_uncertainty and inp.target_in_view and sol.consistent:
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
            # evidence at the tags only: the panel between them is not measured,
            # so the bound never drops below a small floor
            out_of_plane_bound_mm=(max(0.15, float(s.surface.out_of_plane_max_mm)) if s.surface.out_of_plane_max_mm is not None else None),
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
        candidates: bool = False,
        references: bool = False,
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
        return self._overlay_for(s, sol, f, H, view=view, placard_ids=placard_ids, rect_max_edge=rect_max_edge, with_uncertainty=with_uncertainty, candidates=candidates, references=references)

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
        candidates: bool = False,
        unc_override: Any | None = None,
        references: bool = False,
    ) -> dict[str, Any]:
        cam = self._camera(s, f) if ext is None else ext[2].camera
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
            rep = self._gate_report(s, f.id, pid, with_uncertainty and pid == s.active_placard_id, ext=ext, unc_override=(unc_override if pid == s.active_placard_id else None))
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
        if references:
            from .reference import suggest_pair

            pairs = [(r["a"], r["b"], r.get("consistent"), f"{r['measured_mm']:.1f} mm measured") for r in s.reference_distances if r.get("evaluable")]
            sug = suggest_pair(sol)
            if sug and not pairs:
                pairs = [(sug["a"], sug["b"], None, "measure between these two corners")]
            for a, b, okk, label in pairs:
                if a["tag_id"] in sol.layout.poses and b["tag_id"] in sol.layout.poses:
                    P = np.array([sol.layout.corners(a["tag_id"])[a["corner"]], sol.layout.corners(b["tag_id"])[b["corner"]]])
                    uv, good = mapper(P)
                    if good.all():
                        col = "#ffffff" if okk is None else ("#3ddc97" if okk else "#ff5d5d")
                        ov["primitives"].append({"kind": "segment", "points": uv.tolist(), "color": col, "width": 2.5, "dash": [8, 6]})
                        ov["primitives"].append({"kind": "handle", "points": [uv[0].tolist()], "color": col, "label": f"ID {a['tag_id']}"})
                        ov["primitives"].append({"kind": "handle", "points": [uv[1].tolist()], "color": col, "label": f"ID {b['tag_id']} · {label}"})
        if s.verifications and ext is None:
            from .verification import verification_primitives

            ov["primitives"].extend(verification_primitives(s, f.id, mapper, vw, vh))
        if candidates:
            # automatically found straight features, so the operator can pick one
            for c in (s.research.get("datum_proposals") or {}).get("candidates", []):
                ends = np.asarray(c["endpoints"], dtype=float)
                q = ends[0] + np.linspace(0.0, 1.0, 24)[:, None] * (ends[1] - ends[0])
                uv, ok = mapper(q)
                pts = [[float(a), float(b)] for (a, b), k in zip(uv, ok) if k and -0.5 <= a <= vw - 0.5 and -0.5 <= b <= vh - 0.5]
                if len(pts) >= 2:
                    ov["primitives"].append({
                        "kind": "candidate", "points": pts, "closed": False, "id": c["id"], "label": None,
                        "classification": c["classification"], "feature": c["kind"], "height_mm": c["height_mm"], "length_mm": c["length_mm"],
                    })
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
