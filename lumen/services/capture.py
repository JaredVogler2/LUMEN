"""Capture-side services: video ingestion, flash pairs and capture coaching.

These functions take the :class:`SessionService` as their first argument; they
are kept out of ``sessions.py`` so that file stays about registration,
datums and guidance.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import LumenError
from ..geometry.homography import dlt_homography, apply_h, h_local_jacobian
from ..schemas import Frame, FrameRole, Session
from ..vision.video import SelectionConfig, VideoCandidate, analyse_video, estimate_viewpoints, select_frames
from .jobs import JobHandle

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

TMP_PREFIX = "lumen-video-"


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------
def temp_video_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=TMP_PREFIX, suffix=suffix)
    os.close(fd)
    return path


def purge_stale_temp(max_age_s: float = 3600.0) -> int:
    """Remove upload temp files left behind by a crashed worker."""
    n = 0
    root = tempfile.gettempdir()
    now = time.time()
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(TMP_PREFIX):
            continue
        p = os.path.join(root, name)
        try:
            if now - os.path.getmtime(p) > max_age_s:
                os.remove(p)
                n += 1
        except OSError:
            pass
    return n


def process_video(svc: "SessionService", sid: str, path: str, source: str, meta: dict[str, Any], job: JobHandle) -> dict[str, Any]:
    """Job body: choose frames from an uploaded video and add them as photos.

    The video file is deleted when this returns, whatever happens.
    """
    lim = svc.settings.limits
    t0 = time.monotonic()
    try:
        with svc.cpu.slot(lim.video_job_timeout_seconds):
            an = analyse_video(
                path,
                max_candidates=lim.max_video_candidates,
                max_seconds=lim.max_video_seconds,
                marker_cfg=svc._marker_cfg,
                progress=lambda p, st: job.progress(0.55 * p, st),
                cancelled=job.cancelled,
            )
            job.progress(0.58, "Choosing frames", force=True)
            s = svc.get(sid)
            capacity = lim.max_frames_per_session - len(s.frames)
            sel = select_frames(an, SelectionConfig(), capacity=capacity)
        cands = an.candidates
        added: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for n, k in enumerate(sel.selected):
            if job.cancelled():
                break
            job.progress(0.6 + 0.4 * n / max(len(sel.selected), 1), f"Measuring tags in frame {n + 1} of {len(sel.selected)}", force=True)
            c = cands[k]
            client = {
                "video_time_s": c.time_s,
                "captured_at": str(meta.get("captured_at", ""))[:40],
                "camera_label": str(meta.get("camera_label", ""))[:120],
                "facing_mode": str(meta.get("facing_mode", ""))[:24],
                "torch": meta.get("torch", "unknown"),
                "zoom": meta.get("zoom"),
            }
            try:
                f = svc.add_frame(sid, c.jpeg, role="registration", source=source, client=client)
            except LumenError as exc:
                dropped.append({"time_s": round(c.time_s, 3), "reason": exc.code})
                continue
            if not any(m.accepted for m in f.markers) and any(cands[j].tags for j in sel.selected):
                # the careful measurement rejected every tag the quick look had found
                reasons = sorted({r for m in f.markers for r in m.reject_reasons}) or ["no_tag_measurable"]
                svc.remove_frame(sid, f.id)
                dropped.append({"time_s": round(c.time_s, 3), "reason": ", ".join(reasons)})
                continue
            added.append({"frame_id": f.id, "time_s": round(c.time_s, 3), "why": sel.notes.get(k, ""), "tags": sorted(m.tag_id for m in f.markers if m.accepted)})
        out = sel.to_dict(cands)
        out.update(
            {
                "video": an.info.to_dict(),
                "candidates": len(cands),
                "window_s": round(an.window_s, 3),
                "selected": added,
                "dropped_after_measurement": dropped,
                "seconds": round(time.monotonic() - t0, 2),
                "cancelled": bool(job.cancelled()),
                "note": "Frames were chosen for sharpness, tag visibility, exposure and viewpoint spread. The video itself was not stored.",
            }
        )
        svc.log_event(sid, "video_processed", {"source": source, "candidates": len(cands), "selected": len(added), "flags": sel.flags, "video": an.info.to_dict()})
        return out
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# flash on/off pairs
# --------------------------------------------------------------------------
def _exposure_product(f: Frame) -> float | None:
    """Relative exposure (time x ISO / N^2) from EXIF, when the photo has it."""
    e = f.exif
    t, iso, n = e.get("exposure_time_s"), e.get("iso"), e.get("f_number")
    if not t or not iso:
        return None
    try:
        return float(t) * float(iso) / (float(n) ** 2 if n else 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def flash_pairs(s: Session) -> list[dict[str, Any]]:
    """Check every flash on/off pair for camera motion and exposure.

    A pair is only useful for separating flash light from ambient light when
    the camera did not move between the two photos and neither is clipped.
    Browsers give no exposure lock, so unless both photos carry EXIF exposure
    data the brightness ratio between them is unknown and is reported as such.
    """
    groups: dict[str, list[Frame]] = {}
    for f in s.frames:
        if f.pair_id:
            groups.setdefault(f.pair_id, []).append(f)
    out = []
    for pid, fs in sorted(groups.items()):
        on = [f for f in fs if f.torch == "on"]
        off = [f for f in fs if f.torch == "off"]
        rec: dict[str, Any] = {"pair_id": pid, "frames": [f.id for f in fs], "usable": False, "reasons": []}
        if len(on) != 1 or len(off) != 1:
            rec["status"] = "incomplete"
            rec["reasons"].append("a pair needs exactly one photo with the light off and one with it on")
            out.append(rec)
            continue
        a, b = off[0], on[0]
        rec.update({"off": a.id, "on": b.id})
        if (a.width, a.height) != (b.width, b.height):
            rec["status"] = "different_geometry"
            rec["reasons"].append("the two photos have different pixel sizes")
            out.append(rec)
            continue
        ma = {m.tag_id: np.asarray(m.corners_px, dtype=float) for m in a.markers if "clipped" not in m.reject_reasons}
        mb = {m.tag_id: np.asarray(m.corners_px, dtype=float) for m in b.markers if "clipped" not in m.reject_reasons}
        common = sorted(set(ma) & set(mb))
        rec["common_tags"] = common
        if not common:
            rec["status"] = "no_common_tags"
            rec["reasons"].append("no tag is visible in both photos, so camera motion between them cannot be checked")
            out.append(rec)
            continue
        P = np.concatenate([ma[i] for i in common])
        Q = np.concatenate([mb[i] for i in common])
        d = np.linalg.norm(Q - P, axis=1)
        long_edge = float(max(a.width, a.height))
        tol = max(1.0, 0.0008 * long_edge)
        edge_px = float(np.mean([np.mean(np.linalg.norm(np.roll(ma[i], -1, axis=0) - ma[i], axis=1)) for i in common]))
        size_mm = float(np.mean([m.size_mm for m in a.markers if m.tag_id in common]))
        rec["max_shift_px"] = round(float(d.max()), 3)
        rec["rms_shift_px"] = round(float(np.sqrt(np.mean(d**2))), 3)
        rec["shift_on_panel_mm"] = round(float(d.max()) * size_mm / max(edge_px, 1e-6), 3)
        rec["tolerance_px"] = round(tol, 3)
        try:
            H = dlt_homography(P, Q)
            rec["plane_alignment_rms_px"] = round(float(np.sqrt(np.mean(np.sum((apply_h(H, P) - Q) ** 2, axis=1)))), 3)
        except Exception:
            rec["plane_alignment_rms_px"] = None
        moved = float(d.max()) > tol
        ea, eb = _exposure_product(a), _exposure_product(b)
        if ea and eb:
            rec["exposure"] = {"source": "exif", "on_over_off": round(eb / ea, 4)}
        else:
            rec["exposure"] = {"source": "unknown", "on_over_off": None, "note": "no exposure data: the browser cannot lock or report exposure"}
        rec["brightness_on_over_off"] = round(float(b.quality.mean) / max(float(a.quality.mean), 1e-6), 3)
        over = float(b.quality.clipped_high) > 0.10
        if moved:
            rec["reasons"].append(f"the camera moved {rec['max_shift_px']:.1f} px between the two photos (limit {tol:.1f} px): rest the phone on something and retake the pair")
        if over:
            rec["reasons"].append("the photo with the light on is overexposed: move back or angle the phone")
        rec["status"] = "moved" if moved else ("overexposed" if over else "aligned")
        rec["usable"] = not moved and not over
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# coaching
# --------------------------------------------------------------------------
def _frame_K(s: Session, f: Frame, assumed_hfov_deg: float) -> np.ndarray:
    cal = s.calibration
    if cal is not None and f.calibration_id == cal.id:
        sx = f.width / float(cal.width)
        return np.array([[cal.fx * sx, 0.0, (cal.cx + 0.5) * sx - 0.5], [0.0, cal.fy * sx, (cal.cy + 0.5) * sx - 0.5], [0.0, 0.0, 1.0]])
    fpx = f.exif.get("focal_px_prior")
    if not fpx:
        fpx = 0.5 * max(f.width, f.height) / math.tan(math.radians(assumed_hfov_deg) / 2.0)
    return np.array([[float(fpx), 0.0, (f.width - 1) / 2.0], [0.0, float(fpx), (f.height - 1) / 2.0], [0.0, 0.0, 1.0]])


_TAG_FIX = {
    "clipped": ("tag_clipped", "back", "A tag is cut off at the edge of the picture: move back or re-aim."),
    "too_small": ("tag_small", "closer", "The tags are too small in the picture: move closer."),
    "too_oblique": ("tag_oblique", None, "A tag is seen at too steep an angle: face the panel more directly."),
    "glare": ("glare", None, "Glare on a tag: change the angle so the reflection moves off the tags."),
    "edge_disturbed": ("tag_disturbed", None, "A tag edge is hidden or dirty: clear it and retake."),
    "bent_or_distorted": ("tag_bent", None, "A tag does not look flat: press it down or replace it."),
    "duplicate_id": ("duplicate_id", None, "The same tag ID appears twice: every tag on the panel needs a different ID."),
}


def capture_advice(svc: "SessionService", s: Session) -> dict[str, Any]:
    """What the operator should do next to get a usable set of photos.

    Viewpoints are estimated roughly from the tags (assumed or prior focal
    length): good enough to say "you have not moved" and which way there is
    room to move.  Nothing here is used for measurement.
    """
    frames = [f for f in s.frames if f.role == FrameRole.REGISTRATION.value and f.selected]
    usable = [f for f in frames if any(m.accepted for m in f.markers)]
    items: list[dict[str, Any]] = []

    def add(code: str, level: str, text: str, arrow: str | None = None) -> None:
        items.append({"code": code, "level": level, "text": text, "arrow": arrow})

    seen = sorted({m.tag_id for f in usable for m in f.markers if m.accepted})
    common = sorted(set.intersection(*[{m.tag_id for m in f.markers if m.accepted} for f in usable])) if usable else []
    cands: list[VideoCandidate] = []
    for f in usable:
        c = VideoCandidate(index=f.index, time_s=0.0, sharpness=f.quality.sharpness, mean=f.quality.mean, clipped_high=f.quality.clipped_high, clipped_low=f.quality.clipped_low)
        c.tags = {m.tag_id: np.asarray(m.corners_px, dtype=float) for m in f.markers if m.accepted}
        c.min_tag_edge_px = float(min(min(m.edge_px) for m in f.markers if m.accepted))
        c.K = _frame_K(s, f, 66.0)
        cands.append(c)
    spread = None
    side_bias = 0.0
    if cands:
        estimate_viewpoints(cands, usable[-1].width, usable[-1].height)
        posed = [c for c in cands if c.center is not None]
        if len(posed) >= 2:
            dirs = np.array([c.center / np.linalg.norm(c.center) for c in posed])
            spread = float(np.degrees(np.arccos(np.clip((dirs @ dirs.T).min(), -1.0, 1.0))))
        last = cands[-1]
        if last.H is not None and posed:
            # plane direction that appears as "right" in the newest photo
            J = h_local_jacobian(last.H, np.zeros((1, 2)))[0]
            try:
                e_r = np.linalg.solve(J, np.array([1.0, 0.0]))
                e_r /= max(np.linalg.norm(e_r), 1e-9)
                side_bias = float(np.mean([float(c.center[:2] @ e_r) / max(np.linalg.norm(c.center), 1e-9) for c in posed]))
            except np.linalg.LinAlgError:
                side_bias = 0.0
    reg = s.registration
    mv = (reg or {}).get("multiview") or {}
    reg_spread = (mv.get("checks", {}).get("diversity", {}) or {}).get("max_view_angle_deg") if "skipped" not in mv else None

    # ---- what is still missing, most important first
    if not frames:
        add("first_photo", "todo", "Take the first photo: all tags in view, phone roughly facing the panel.")
    elif not usable:
        add("no_tags", "todo", "No usable tag yet: get the tags fully in view, in focus and not washed out.")
    newest = frames[-1] if frames else None
    if newest is not None:
        told = set()
        for m in newest.markers:
            for r in m.reject_reasons:
                if r in _TAG_FIX and r not in told:
                    told.add(r)
                    code, arrow, text = _TAG_FIX[r]
                    add(code, "warn", text, arrow)
        if newest.quality.sharpness < 40:
            add("blurred", "warn", "The newest photo is soft: hold still, or add light.")
        if newest.quality.clipped_high > 0.08:
            add("overexposed", "warn", "Large overexposed area: change the angle or reduce the light.")
        if seen and usable and newest in usable:
            missing = [i for i in seen if i not in {m.tag_id for m in newest.markers if m.accepted}]
            if missing:
                add("tag_missing", "warn", "Not in the newest photo: " + ", ".join(f"ID {i}" for i in missing) + ". Keep every tag in view.")
    min_views, max_views = 3, 9
    if usable and len(usable) < min_views:
        n = min_views - len(usable)
        add("more_views", "todo", f"Take {n} more photo{'s' if n > 1 else ''} from a different position.", "left" if side_bias > 0.05 else "right")
    eff = reg_spread if reg_spread is not None else spread
    if len(usable) >= 2 and eff is not None and eff < 12.0:
        add("narrow_baseline", "todo", f"The photos are taken from almost the same place ({eff:.0f}° apart): step sideways and take another.", "left" if side_bias > 0.05 else "right")
    if len(usable) >= max_views:
        add("enough", "info", f"{len(usable)} photos is plenty: more will not help much.")
    if seen and len(seen) < 2:
        add("one_tag", "warn", "Only one tag is seen: a second tag lets LUMEN check the scale independently.")
    unmeasured = [i for i in seen if str(i) not in (s.tag_sizes.get("measured_mm") or {})]
    if unmeasured:
        add("unmeasured_tags", "warn", "Tags without a measured size: " + ", ".join(f"ID {i}" for i in unmeasured) + ".")
    # Tags that are small in the picture cannot measure their own edge shift, and every pixel is a larger part
    # of their size: the stated uncertainty grows.  (About 6.5 px per tag cell are needed, i.e. 52 px per edge.)
    obs = [m for f in usable for m in f.markers if m.accepted]
    if obs:
        small = [m for m in obs if not m.bias_applied]
        smallest = min(min(m.edge_px) for m in obs)
        if len(small) > len(obs) / 2:
            add("tags_low_resolution", "warn", f"The tags are small in the photos (smallest {smallest:.0f} px): move closer, or use Full-res photo. Small tags make the result less certain.", "closer")
    if usable and s.calibration is None:
        add("no_lens_profile", "info", "No lens profile is attached: results stay a research preview. Make one once per phone under Settings.")

    # ---- constraints that only the datum step can report
    props = s.research.get("datum_proposals") if reg else None
    if props:
        applied = props.get("applied", {})
        side_def = None
        if s.operation:
            side_def = ((s.operation.get("operation") or {}).get("datums") or {}).get("side") or {}
        if "floor" not in applied and "floor" not in s.datums:
            add("floor_missing", "todo", "The floor line is not in any photo: tilt the phone down and take one that includes it.", "down")
        if "side" not in applied and "side" not in s.datums:
            where = (side_def or {}).get("side_of_panel")
            curved = (props.get("side_feature") or "").startswith("curve")
            if curved:
                add("curve_needs_views", "todo", "The rounded edge needs 5–9 photos taken while moving sideways around it.", where if where in ("left", "right") else None)
            else:
                add("side_missing", "todo", "The side edge is not in any photo: take one that includes it.", where if where in ("left", "right") else None)
        if props.get("ambiguous_side") or props.get("ambiguous_floor"):
            add("ambiguous_datum", "warn", "Two similar lines were found for a datum. More photos cannot decide between identical features: choose the intended one on the Datums step.")
    pairs = flash_pairs(s)
    for p in pairs:
        if not p["usable"] and p.get("reasons"):
            add("flash_pair", "warn", "Flash pair: " + p["reasons"][0])

    ready = bool(usable)
    blocking = [i for i in items if i["level"] == "todo"]
    nxt = (blocking or [i for i in items if i["level"] == "warn"] or [None])[0]
    if nxt is None and ready:
        nxt = {"code": "analyse", "level": "ok", "text": "Analyse the photos." if not reg else "The photos are analysed.", "arrow": None}
    return {
        "ready": ready,
        "recommended": bool(ready and not blocking and len(usable) >= min_views),
        "n_photos": len(frames),
        "n_usable": len(usable),
        "tags_seen": seen,
        "common_tags": common,
        "spread_deg": None if eff is None else round(float(eff), 1),
        "spread_source": None if eff is None else ("registered" if reg_spread is not None else "approximate"),
        "items": items,
        "next": nxt,
        "pairs": pairs,
        "views": {"min": min_views, "max": max_views},
    }
