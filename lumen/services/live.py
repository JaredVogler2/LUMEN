"""Live guidance: register one camera frame against the session and return its overlay.

How "live" stays honest
-----------------------
The client sends a frame, the server measures the tags *in that frame*,
localises it against the registered tag layout and returns the overlay *for
that frame*.  The client shows the frame it sent together with the overlay it
got back, so a target outline is never drawn on a picture it was not computed
from.  Latency only lowers the update rate.

When a frame cannot be localised the response carries no geometry at all and
says why (and, where possible, which way to move).  Frames are processed in
memory and are never stored.

Uncertainty for a live frame costs about half a second, so it is evaluated
for a frame and reused only for later frames that show the same tags from a
nearly identical viewpoint (and never for longer than a few seconds); any
other frame is evaluated afresh.  Gates that depend on the frame itself (tags
in view, tag size, target in view, residuals) are evaluated for every frame.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import RESEARCH_NOTICE
from ..errors import GeometryError, LumenError
from ..geometry.homography import apply_h
from ..schemas import FrameRole, FrameSource, PlacementSpec, SessionMode
from ..vision.imaging import decode_image
from ..vision.markers import detect_tags
from ..vision.planar import PlanarView, localize_view
from .frames import build_frame, camera_for_frame, sizes_from_state, tag_obs

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

REUSE_MAX_AGE_S = 6.0
REUSE_MAX_SHIFT = 0.02  # of the frame diagonal


def _lost(code: str, message: str, *, arrow: str | None = None, suggestions: list[str] | None = None, tags: list[dict[str, Any]] | None = None, frame: dict[str, Any] | None = None, simulated: bool = False) -> dict[str, Any]:
    return {
        "frame_id": "live", "view": "image", "primitives": [], "grid_step_mm": 0.0, "reports": {}, "frozen": False,
        "tracking": {"state": "lost", "code": code, "message": message, "arrow": arrow, "suggestions": suggestions or [], "tags": tags or []},
        "frame": frame or {}, "notice": RESEARCH_NOTICE, "simulated": simulated,
    }


_REJECT = {
    "too_small": ("tags_too_small", "The tags are too small in the picture: move closer.", "closer"),
    "clipped": ("tags_clipped", "A tag is cut off at the edge of the picture: move back or re-aim.", "back"),
    "too_oblique": ("tags_oblique", "The tags are seen at too steep an angle: face the panel more directly.", None),
    "glare": ("glare", "Glare on the tags: change the angle slightly.", None),
}


def _client_native(meta: dict[str, Any], w: int, h: int) -> tuple[int, int]:
    """Pixel size of the camera stream this (possibly reduced) frame came from."""
    try:
        nw, nh = int(meta.get("native_width", 0)), int(meta.get("native_height", 0))
    except (TypeError, ValueError):
        return w, h
    if nw >= w and nh >= h and nw > 0 and nh > 0 and abs(nw / w - nh / h) <= 3e-3 * max(nw / w, 1.0):
        return nw, nh
    return w, h


def _zoom(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def track_frame(svc: "SessionService", sid: str, data: bytes, *, placard_ids: list[str] | None = None, meta: dict[str, Any] | None = None, force_full: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    meta = meta or {}
    lim = svc.settings.limits
    s = svc.get(sid)
    sol, views = svc._require_registration(s)
    simulated = s.mode == SessionMode.DEMO.value
    img = decode_image(data, max_bytes=lim.max_image_bytes, max_pixels=lim.max_image_pixels, max_long_edge=lim.max_track_long_edge)
    nat_w, nat_h = _client_native(meta, int(img.meta.native_width), int(img.meta.native_height))
    scale = img.width / float(nat_w)
    cam, note = camera_for_frame(s.calibration, img.width, img.height, nat_w, nat_h, scale, device=str(meta.get("camera_label") or ""), zoom=_zoom(meta.get("zoom")))
    sizes = sizes_from_state(s.tag_sizes)
    t1 = time.perf_counter()
    with svc.cpu.slot(lim.request_timeout_seconds):
        dets = detect_tags(img.bgr, camera=cam, sizes=sizes, cfg=svc._marker_cfg)
    t2 = time.perf_counter()
    f = build_frame(
        frame_id="live", index=0, blob="", data_len=len(data), img=img, dets=dets, cal=s.calibration, cal_note=note, cam=cam,
        role=FrameRole.VERIFICATION, source=FrameSource.CAMERA_STILL, client={k: meta.get(k) for k in ("camera_label", "facing_mode", "zoom", "torch")},
    )
    f.native_width, f.native_height, f.processing_scale = nat_w, nat_h, scale
    fr = {"width": f.width, "height": f.height, "calibrated": cam is not None, "sharpness": round(f.quality.sharpness, 1)}
    tag_list = [{"tag_id": m.tag_id, "accepted": m.accepted, "in_layout": m.tag_id in sol.layout.poses, "reasons": m.reject_reasons, "corners_px": m.corners_px} for m in f.markers]
    accepted = [m for m in f.markers if m.accepted]
    if not f.markers:
        return _lost("no_tags_in_view", "No tag in view: aim the phone at the tags.", tags=tag_list, frame=fr, simulated=simulated)
    if not accepted:
        reasons = [r for m in f.markers for r in m.reject_reasons]
        for r, (code, text, arrow) in _REJECT.items():
            if r in reasons:
                return _lost(code, text, arrow=arrow, tags=tag_list, frame=fr, simulated=simulated)
        return _lost("tags_not_usable", "The tags cannot be measured in this frame: hold still and check focus and light.", tags=tag_list, frame=fr, simulated=simulated)
    if not any(m.tag_id in sol.layout.poses for m in accepted):
        return _lost("unknown_tags", "These tags are not part of this session's registration.", suggestions=["Aim at the tags that were on the panel during capture, or analyse again with the new tags."], tags=tag_list, frame=fr, simulated=simulated)
    pool = svc._pool_for(s, f)  # small tags take the edge shift measured on larger ones (this frame, else the registration photos)
    view = PlanarView("live", [tag_obs(m, pool) for m in accepted], cam, "auto")
    try:
        with svc.cpu.slot(lim.request_timeout_seconds):
            loc = localize_view(sol.layout, view)
    except GeometryError as exc:
        return _lost(exc.code, exc.message, suggestions=exc.suggestions, tags=tag_list, frame=fr, simulated=simulated)
    ext_view = PlanarView("live", [t for t in view.tags if t.tag_id in sol.layout.poses], cam, view.mode)
    ext = (f, loc, ext_view)
    t3 = time.perf_counter()

    # ---- uncertainty: reuse only for a nearly identical view of the same tags
    pid = s.active_placard_id
    lf = svc._local_frame(s)
    key = (
        sid, "live", pid, (s.registration or {}).get("computed_at"),
        tuple((k, tuple(round(x, 6) for x in d.n), round(d.d, 4), d.confirmed) for k, d in sorted(s.datums.items())),
        (s.operation or {}).get("content_hash"), None if s.calibration is None else s.calibration.id, (f.width, f.height),
    )
    probe_pts = sol.layout.all_corners()
    if lf is not None and lf.valid and pid and s.operation:
        try:
            spec = PlacementSpec(**svc._operation(s).placard(pid).resolved())
            if svc._spec_complete(spec):
                probe_pts = np.concatenate([probe_pts, svc._target(lf, spec).corners])
        except LumenError:
            pass
    sig = apply_h(loc.H, probe_pts) / float(np.hypot(f.width, f.height))
    used = tuple(sorted(loc.used_ids))
    bias = tuple(sorted((t.tag_id, bool(t.bias_corrected), bool(t.bias_pooled)) for t in ext_view.tags))
    hit = svc._live.get(key)
    reuse = None
    if (
        not force_full and hit is not None and hit["used"] == used and hit["bias"] == bias and hit["sig"].shape == sig.shape
        and time.time() - hit["t"] <= REUSE_MAX_AGE_S and float(np.max(np.linalg.norm(hit["sig"] - sig, axis=1))) <= REUSE_MAX_SHIFT
    ):
        reuse = hit["unc"]
    ov = svc._overlay_for(s, sol, f, loc.H, view="image", placard_ids=placard_ids, rect_max_edge=1600, with_uncertainty=True, ext=ext, unc_override=reuse)
    rep = ov["reports"].get(pid) if pid else None
    if reuse is None and rep is not None and rep.get("uncertainty") is not None:
        from ..schemas import Uncertainty

        svc._live.put(key, {"used": used, "bias": bias, "sig": sig, "t": time.time(), "unc": Uncertainty.model_validate(rep["uncertainty"])})
    t4 = time.perf_counter()

    # ---- what the operator should do about this frame
    msg, code, arrow = "Tracking.", "tracking", None
    if "high_residual" in loc.flags:
        code, msg = "tag_moved", "A registered tag appears to have moved or is misread: do not use this view."
    elif rep is not None and rep["status"] != "blocked":
        tgt = next((p for p in ov["primitives"] if p["kind"].startswith("target") and p.get("active") is not False and p.get("corners")), None)
        if tgt is None:
            code, msg = "no_target", "No target outline for this frame."
    if rep is not None and not next((g["passed"] for g in rep["gates"] if g["code"] == "target_in_view"), True) and lf is not None and lf.valid and pid:
        # say which way the target lies
        try:
            spec = PlacementSpec(**svc._operation(s).placard(pid).resolved())
            c = svc._target(lf, spec).corners.mean(axis=0)
            ph = np.append(c, 1.0) @ loc.H.T
            ref_w = float(np.append(sol.layout.centroid(), 1.0) @ loc.H[2])
            if ph[2] * np.sign(ref_w) <= 0:
                code, msg = "target_behind", "The target location is not in front of the camera: step back and face the panel."
            else:
                u, v = ph[0] / ph[2], ph[1] / ph[2]
                dx, dy = u - (f.width - 1) / 2.0, v - (f.height - 1) / 2.0
                if abs(dx) / f.width >= abs(dy) / f.height:
                    arrow = "right" if dx > 0 else "left"
                else:
                    arrow = "down" if dy > 0 else "up"
                code, msg = "target_out_of_view", f"The target is outside the picture: aim further {arrow}, or move back."
        except LumenError:
            pass
    elif len(used) == 1 and code == "tracking":
        code, msg = "one_tag", "Only one registered tag is in view: include more tags for a steadier, checked result."
    elif f.quality.sharpness < 40 and code == "tracking":
        code, msg = "blurred", "The picture is soft: hold still."

    ov.update(
        {
            "frozen": False,
            "tracking": {
                "state": "tracking", "code": code, "message": msg, "arrow": arrow, "suggestions": [],
                "tags": tag_list, "used_tags": list(used), "rms_px": round(float(loc.rms_px), 4), "flags": list(loc.flags),
                "uncertainty_reused": reuse is not None,
                "H": np.asarray(loc.H, dtype=float).tolist(),
            },
            "frame": fr,
            "timing_ms": {
                "decode": round((t1 - t0) * 1e3, 1), "tags": round((t2 - t1) * 1e3, 1), "localise": round((t3 - t2) * 1e3, 1),
                "overlay": round((t4 - t3) * 1e3, 1), "total": round((t4 - t0) * 1e3, 1),
            },
        }
    )
    return ov
