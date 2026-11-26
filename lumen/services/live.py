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

Two things make a stream of frames cheaper than as many photos, and both are
said in the answer.  The dark-side shift of the tag borders is a property of
the stream (blur, tone processing), so it is measured on one frame and the
client may hand it back with the next ones (``edge_shift`` in the frame's
metadata): those frames skip that measurement, which is two thirds of the tag
time, and their tags are treated like tags too small to measure it themselves.
And with ``placement`` the placard is followed from where it was in the
previous frame (``placement_hint``) before it is searched for.

Uncertainty for a live frame costs about half a second, so it is evaluated
for a frame and reused only for later frames that show the same tags from a
nearly identical viewpoint (and never for longer than a few seconds); any
other frame is evaluated afresh.  Gates that depend on the frame itself (tags
in view, tag size, target in view, residuals) are evaluated for every frame.

A phone that stands still
-------------------------
During placement the phone is on a stand, and arms reach across the tags.
Measuring the tags again in every frame then costs most of the frame's time
and gives a different registration whenever a tag is covered.  So the
registration of one frame (the *reference*) is kept, together with the picture
around each tag, and a later frame is first asked whether that picture is
still in the same place (``vision/steady.py``: the shift of each tag's
surroundings, to a few hundredths of a pixel).  If at least two tags say so,
the kept registration is used: the frame is *held*.  Tags that cannot be
compared (an arm over them) are simply not asked; a tag that was moved does
not compare either.  If the tags have all moved a little and agree on how (a
stand that gives when the panel is touched: up to half a pixel), the
registration is moved along with them.  Anything else is measured afresh, and
so is a held registration from time to time.  The answer says that it was
held, by which tags, and how far they had moved; the reading carries a term
for what the check cannot see.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .. import RESEARCH_NOTICE
from ..errors import GeometryError, LumenError
from ..geometry.homography import apply_h
from ..schemas import FrameRole, FrameSource, PlacementSpec, SessionMode
from ..vision.imaging import decode_image
from ..vision.markers import detect_tags
from ..vision.planar import PlanarView, localize_view
from ..vision.steady import make_patch, patch_shift
from .frames import EdgeShiftPool, build_frame, camera_for_frame, edge_shift_pool, sizes_from_state, tag_obs

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

REUSE_MAX_AGE_S = 6.0
REUSE_MAX_SHIFT = 0.02  # of the frame diagonal

#: share of the picture around a tag that the reference must explain for the tag to count as seen, unmoved
HOLD_MIN_R2 = 0.97
#: up to this shift of the tags (px) the kept registration is used as it is
HOLD_STILL_PX = 0.1
#: up to this shift it is moved along with the tags, if at least three of them agree on the movement ...
HOLD_MAX_PX = 0.5
#: ... to this (px)
HOLD_AGREE_PX = 0.06
#: a held registration is measured afresh this often while no tag is covered (seconds); sooner when the
#: reference itself did not have all the tags of the layout
HOLD_REFRESH_S = 5.0
HOLD_REFRESH_PARTIAL_S = 2.0
#: with a single tag left to compare (an arm over the others) the registration is held for this long after the last
#: frame in which two tags agreed (seconds).  One tag alone cannot show a movement of the camera that leaves that
#: tag where it is; for a moment after two tags said "nothing moved", that is not what is happening.
HOLD_ONE_TAG_S = 3.0


def _steady(held: dict[str, Any], gray: np.ndarray) -> dict[str, Any] | None:
    """Is the picture around the tags where it was in the reference frame?

    Returns which tags say so, which cannot be compared (covered, moved, changed), the largest shift, and, when
    the tags have moved a little and agree on how, the similarity that carries reference pixels to this frame.
    ``None``: the frame has to be measured afresh.
    """
    seen, covered, pts, sh = [], [], [], []
    for p in held["patches"]:
        dx, dy, r2 = patch_shift(p, gray)
        if r2 >= HOLD_MIN_R2 and np.isfinite(dx) and np.isfinite(dy):
            seen.append(p.tag_id)
            pts.append([p.x0 + 0.5 * p.shape[1], p.y0 + 0.5 * p.shape[0]])
            sh.append([dx, dy])
        else:
            covered.append(p.tag_id)
    if not seen:
        return None
    S = np.asarray(sh, dtype=float)
    worst = float(np.max(np.linalg.norm(S, axis=1)))
    if worst > HOLD_MAX_PX:
        return None
    if len(seen) < 2:
        # one tag cannot show a turn of the camera about itself: accepted only for a moment after two tags agreed
        if worst > HOLD_STILL_PX or time.time() - held.get("t2", 0.0) > HOLD_ONE_TAG_S:
            return None
        return {"seen": seen, "covered": covered, "shift_px": worst, "A": None}
    held["t2"] = time.time()
    A = None
    if worst > HOLD_STILL_PX:
        if len(seen) < 3:
            return None  # two tags fix a similarity exactly: nothing would check it
        P = np.asarray(pts, dtype=float)
        # x' = a x - b y + tx,  y' = b x + a y + ty  through the tag centres and where they are now
        M = np.zeros((2 * len(P), 4))
        M[0::2] = np.c_[P[:, 0], -P[:, 1], np.ones(len(P)), np.zeros(len(P))]
        M[1::2] = np.c_[P[:, 1], P[:, 0], np.zeros(len(P)), np.ones(len(P))]
        rhs = (P + S).ravel()
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        if float(np.max(np.linalg.norm((M @ sol - rhs).reshape(-1, 2), axis=1))) > HOLD_AGREE_PX:
            return None  # the tags do not agree: it is not the camera that moved
        a, b, tx, ty = (float(v) for v in sol)
        A = np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])
    return {"seen": seen, "covered": covered, "shift_px": worst, "A": A}


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


def _carried_pool(meta: dict[str, Any], width: int, height: int) -> EdgeShiftPool | None:
    """The edge shift measured on an earlier frame of the same stream, handed back by the client."""
    e = meta.get("edge_shift")
    if not isinstance(e, dict):
        return None
    try:
        delta, sigma, n = float(e["delta_px"]), float(e["sigma_px"]), int(e.get("n", 1))
        size = [int(v) for v in e.get("for", [])]
    except (KeyError, TypeError, ValueError):
        return None
    if size != [width, height] or not (np.isfinite(delta) and np.isfinite(sigma)) or abs(delta) > 2.0 or not (0.0 < sigma <= 1.0) or n < 1:
        return None
    return EdgeShiftPool(delta, sigma, n)


def track_frame(
    svc: "SessionService", sid: str, data: bytes, *, placard_ids: list[str] | None = None, meta: dict[str, Any] | None = None, force_full: bool = False,
    placement: bool = False,
) -> dict[str, Any]:
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
    pid = s.active_placard_id
    lf = svc._local_frame(s)
    # everything a live answer depends on besides the frame itself
    key = (
        sid, "live", pid, (s.registration or {}).get("computed_at"),
        tuple((k, tuple(round(x, 6) for x in d.n), round(d.d, 4), d.confirmed) for k, d in sorted(s.datums.items())),
        (s.operation or {}).get("content_hash"), None if s.calibration is None else s.calibration.id, (img.width, img.height),
    )
    hold_key = (sid, "hold")
    hold_sig = (key, tuple(placard_ids or ()), nat_w, nat_h)
    gray: np.ndarray | None = None
    steady: dict[str, Any] | None = None  # the reference this frame was found unmoved against, when it is measured again all the same
    t1 = time.perf_counter()
    if placement and not force_full:
        held = svc._live.get(hold_key)
        if held is not None and held["sig"] == hold_sig:
            gray = cv2.cvtColor(img.bgr, cv2.COLOR_BGR2GRAY) if img.bgr.ndim == 3 else img.bgr
            chk = _steady(held, gray)
            age = time.time() - held["t"]
            due = age > (HOLD_REFRESH_S if held["whole"] else HOLD_REFRESH_PARTIAL_S)
            if chk is not None and not (due and not chk["covered"]):
                return _held_answer(svc, s, held, chk, img, cam, pid, meta, t0, t1, age)
            if chk is not None and chk["A"] is None:
                steady = held
    carried = None if force_full else _carried_pool(meta, img.width, img.height)
    with svc.cpu.slot(lim.request_timeout_seconds):
        dets = detect_tags(img.bgr, camera=cam, sizes=sizes, cfg=svc._marker_cfg if carried is None else replace(svc._marker_cfg, bias_correction=False))
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
    # small tags take the edge shift measured on larger ones: this frame's, else the stream's (handed back), else the registration photos'
    own = edge_shift_pool(f.markers)
    pool = own or carried or svc._pool_for(s, f)
    shift = None if pool is None else {**pool.to_dict(), "for": [f.width, f.height], "source": "this_frame" if own is not None else ("stream" if carried is not None else "registration_photos")}
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
    if reuse is None and not force_full and steady is not None and steady.get("unc") is not None and steady["used"] == used and steady["bias"] == bias:
        # The phone stands still and this frame is measured again as a matter of routine: the same tags, in the
        # same pixels, measured the same way.  The uncertainty of that view has not changed with the clock.
        reuse = steady["unc"]
    ov = svc._overlay_for(s, sol, f, loc.H, view="image", placard_ids=placard_ids, rect_max_edge=1600, with_uncertainty=True, ext=ext, unc_override=reuse)
    rep = ov["reports"].get(pid) if pid else None
    unc_now = reuse
    if reuse is None and rep is not None and rep.get("uncertainty") is not None:
        from ..schemas import Uncertainty

        unc_now = Uncertainty.model_validate(rep["uncertainty"])
        svc._live.put(key, {"used": used, "bias": bias, "sig": sig, "t": time.time(), "unc": unc_now})
    t4 = time.perf_counter()
    if placement:
        from .placement import live_placement

        with svc.cpu.slot(lim.request_timeout_seconds):
            ov["placement"] = live_placement(svc, s, f, img.bgr, cam, loc, rep, pid, meta)
    t5 = time.perf_counter()

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
                "edge_shift": shift,
            },
            "frame": fr,
            "timing_ms": {
                "decode": round((t1 - t0) * 1e3, 1), "tags": round((t2 - t1) * 1e3, 1), "localise": round((t3 - t2) * 1e3, 1),
                "overlay": round((t4 - t3) * 1e3, 1), "placard": round((t5 - t4) * 1e3, 1), "hold": 0.0, "total": round((t5 - t0) * 1e3, 1),
            },
        }
    )
    if placement and "high_residual" not in loc.flags:
        # the reference for the frames that follow, should the phone turn out to stand still
        if gray is None:
            gray = cv2.cvtColor(img.bgr, cv2.COLOR_BGR2GRAY) if img.bgr.ndim == 3 else img.bgr
        patches = [p for p in (make_patch(gray, m.tag_id, np.asarray(m.corners_px, dtype=float)) for m in accepted if m.tag_id in used) if p is not None]
        if len(patches) >= 2:
            svc._live.put(hold_key, {
                "sig": hold_sig, "patches": patches, "loc": loc, "f": f, "t": time.time(), "t2": time.time(),
                "unc": unc_now, "used": used, "bias": bias,
                # nothing more to gain from measuring again soon: every tag of the layout was used, or this frame was
                # itself a re-measurement from the same place and found no more tags than the one before (the others
                # are outside the picture)
                "whole": len(used) >= len(sol.layout.poses) or (steady is not None and len(used) >= len(steady["used"])),
                "ov": {k: v for k, v in ov.items() if k not in ("placement", "timing_ms", "tracking", "frame")},
                "tracking": ov["tracking"], "frame": fr,
            })
        else:
            svc._live.put(hold_key, None)
    return ov


def _held_answer(
    svc: "SessionService", s: Any, held: dict[str, Any], chk: dict[str, Any], img: Any, cam: Any, pid: str | None, meta: dict[str, Any],
    t0: float, t1: float, age: float,
) -> dict[str, Any]:
    """The answer for a frame whose registration is the reference's, checked on the tags that can be seen."""
    from .placement import live_placement

    t2 = time.perf_counter()
    loc = held["loc"]
    if chk["A"] is not None:
        loc = replace(loc, H=chk["A"] @ np.asarray(loc.H, dtype=float))
    ov = dict(held["ov"])
    rep = ov["reports"].get(pid) if pid else None
    with svc.cpu.slot(svc.settings.limits.request_timeout_seconds):
        ov["placement"] = live_placement(
            svc, s, held["f"], img.bgr, cam, loc, rep, pid, meta,
            held_px=HOLD_STILL_PX if chk["A"] is None else HOLD_AGREE_PX,
        )
    t3 = time.perf_counter()
    shift = held["tracking"].get("edge_shift")
    ov["tracking"] = {
        **held["tracking"], "H": np.asarray(loc.H, dtype=float).tolist(),
        # no tag was measured in this frame; the stream's edge shift is the one the reference frame used
        "edge_shift": None if shift is None else {**shift, "source": "stream"},
        "held": {
            "tags_seen": chk["seen"], "tags_covered": chk["covered"], "shift_px": round(float(chk["shift_px"]), 3),
            "moved_along": chk["A"] is not None, "age_s": round(float(age), 1),
        },
    }
    ov["frame"] = held["frame"]
    ov["frozen"] = False
    ov["timing_ms"] = {
        "decode": round((t1 - t0) * 1e3, 1), "tags": 0.0, "localise": 0.0, "overlay": 0.0,
        "placard": round((t3 - t2) * 1e3, 1), "hold": round((t2 - t1) * 1e3, 1), "total": round((t3 - t0) * 1e3, 1),
    }
    return ov
