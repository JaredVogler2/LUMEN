"""Datum geometry: floor seam, side boundary, rounded-edge model (prompt 06).

Three questions are answered here, in this order:

1. **Where are straight features?**  Long straight edges and thin lines are
   proposed in the rectified view of each photo (line-segment detection,
   collinear merging) and then *measured in the original pixels* with the
   edge-spread / thin-line fits of :mod:`lumen.vision.edgefit`.

2. **What kind of thing is each feature?**  With several calibrated views the
   same feature is found again in every photo and its viewing planes are
   intersected (:func:`lumen.vision.multiview.triangulate_line`):

   * all planes meet in one 3-D line **on** the target plane  ->  a physical
     seam or edge of the panel face (``on_plane``);
   * they meet in one 3-D line **off** the plane  ->  a physical edge that is
     proud of or behind the face, e.g. a trim strip (``off_plane``, with the
     measured height);
   * they do not meet in one line  ->  the feature is a different 3-D line in
     every photo: the occluding silhouette of a curved surface, a shadow, a
     highlight (``view_dependent``).

   A soft edge (much wider than the tag edges of the same photo) is flagged as
   a possible lighting gradient.  With a single view nothing of this can be
   known and the feature stays ``unverified``.

3. **Where is the rounded edge?**  Silhouette lines of a cylindrical bend are
   tangent planes of the cylinder.  Given the camera positions, the tangent
   line (where flat becomes curved), the axis direction and the radius are
   fitted to them.  A uniform cylinder fixes the direction and the position
   *across* the axis but nothing *along* it — the floor datum has to supply
   that.

Nothing here assumes that "down" in a photo is the floor, that the side edge
is vertical, or that the two datums are perpendicular; candidate pairs are
matched to the operation's configured nominal angle and handedness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, local_scale_px_per_mm
from ..geometry.lines import Line2D
from ..geometry.optimize import lm_solve
from ..geometry.se3 import Pose
from .edgefit import EdgeFit, fit_bar, fit_edge, snap_line
from .imaging import linear_luminance, to_gray
from .multiview import triangulate_line, view_plane_of_image_line
from .rectify import RectSpec


@dataclass(frozen=True)
class DatumConfig:
    min_length_mm: float = 90.0
    #: collinear merging tolerances in the rectified view
    merge_angle_deg: float = 1.2
    merge_offset_px: float = 2.5
    max_candidates_per_frame: int = 14
    #: a feature is "the same" in another photo if found within this band (px)
    match_search_px: float = 26.0
    #: stationarity: residual of the common 3-D line (mm) and height limits.
    #: Real seams triangulate to 0.002–0.02 mm on synthetic sweeps while the
    #: silhouette of a 35 mm radius gives 0.7–1.4 mm, so the threshold sits
    #: between them with margin for pose error on real captures.
    stationary_rms_mm: float = 0.25
    on_plane_mm: float = 1.0
    #: an edge this many times softer than the tag edges may be a lighting gradient
    soft_edge_factor: float = 3.5
    #: pair search: tolerance around the configured nominal datum angle
    pair_angle_tol_deg: float = 8.0
    #: competing candidate threshold (fraction of the best score)
    ambiguity_fraction: float = 0.6
    tag_margin_mm: float = 12.0


@dataclass
class LineCandidate:
    id: str
    frame_id: str
    kind: str  # "step" | "bar"
    fit: EdgeFit  # in ideal pixels of ``frame_id``
    line: Line2D  # plane frame
    endpoints: np.ndarray  # (2, 2) plane mm
    length_mm: float
    contrast: float
    sigma_blur_px: float
    distinctness: float
    score: float = 0.0
    #: matched fits in other frames: frame_id -> (EdgeFit, plane line, plane endpoints)
    matches: dict[str, tuple[EdgeFit, Line2D, np.ndarray]] = field(default_factory=dict)
    classification: str = "unverified"  # on_plane | off_plane | view_dependent | unverified
    height_mm: float | None = None
    stationarity_rms_mm: float | None = None
    spread_mm: float | None = None  # spread of the plane-projected position across views
    soft: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "frame_id": self.frame_id,
            "kind": self.kind,
            "n": self.line.n.round(8).tolist(),
            "d": round(self.line.d, 5),
            "endpoints": self.endpoints.round(3).tolist(),
            "length_mm": round(self.length_mm, 2),
            "contrast": round(self.contrast, 4),
            "sigma_blur_px": round(self.sigma_blur_px, 3),
            "distinctness": round(self.distinctness, 3),
            "score": round(self.score, 4),
            "n_views": 1 + len(self.matches),
            "classification": self.classification,
            "height_mm": None if self.height_mm is None else round(self.height_mm, 3),
            "stationarity_rms_mm": None if self.stationarity_rms_mm is None else round(self.stationarity_rms_mm, 3),
            "spread_mm": None if self.spread_mm is None else round(self.spread_mm, 3),
            "soft": self.soft,
            "notes": list(self.notes),
            "image_fit": fit_to_dict(self.fit),
        }


def fit_to_dict(e: EdgeFit, inflation: float = 1.6) -> dict[str, Any]:
    assert e.mid is not None and e.n is not None
    return {
        "mid": e.mid.tolist(),
        "n": e.n.tolist(),
        "length": e.length,
        "var_d": e.var_d * inflation + 0.02**2,
        "var_theta": e.var_theta * inflation,
        "cov_d_theta": e.cov_d_theta * inflation,
        "sigma_blur_px": e.sigma_blur,
        "contrast": e.contrast,
    }


def fit_plane_line(e: EdgeFit, Hinv: np.ndarray) -> tuple[Line2D, np.ndarray]:
    """Plane line and end points of an image-space fit."""
    assert e.mid is not None and e.n is not None
    t = e.t
    h = 0.5 * max(e.length, 2.0)
    q = apply_h(Hinv, np.stack([e.mid - h * t, e.mid + h * t]))
    return Line2D.from_points(q[0], q[1]), q


# --------------------------------------------------------------------------
# 1. proposals in one frame
# --------------------------------------------------------------------------
def straight_support(gray: np.ndarray, camera: CameraModel | None, fit: EdgeFit, kind: str, tol_px: float = 0.6) -> tuple[np.ndarray, np.ndarray] | None:
    """Longest stretch of a fitted feature that really is one straight line.

    A long merged segment can run from a true seam onto something else (past
    the start of a bend, onto the background).  The feature is re-measured in
    short windows; the longest contiguous run of windows that agree with a
    common line is returned as ``(p0, p1)`` in ideal pixels, or ``None`` if no
    such run exists.
    """
    assert fit.mid is not None and fit.n is not None
    t = fit.t
    L = fit.length
    m = int(np.clip(L // 45, 4, 16))
    if L < 4 * 30:
        return fit.mid - 0.5 * L * t, fit.mid + 0.5 * L * t
    edges = np.linspace(-0.5 * L, 0.5 * L, m + 1)
    offs = np.full(m, np.nan)
    for k in range(m):
        a = fit.mid + edges[k] * t
        b = fit.mid + edges[k + 1] * t
        if kind == "bar":
            f = fit_bar(gray, a, b, half_width=5.0, trim=1.0, camera=camera, min_contrast=0.015)
        else:
            f = fit_edge(gray, a, b, fit.n, half_width=5.0, trim=1.0, camera=camera, fix_theta=True, fix_sigma=max(fit.sigma_blur, 0.4), level_slopes=False, passes=1, min_contrast=0.015)
        if f.ok and f.mid is not None and f.inlier_frac > 0.5:
            offs[k] = float(fit.offset_at(f.mid))
    ok = np.isfinite(offs)
    if ok.sum() < 3:
        return None
    # robust line through the window offsets: allow a small residual tilt
    x = 0.5 * (edges[:-1] + edges[1:])
    best_run: tuple[int, int] = (0, -1)
    for i in range(m):
        for j in range(i + 2, m):
            if not (ok[i] and ok[j]):
                continue
            slope = (offs[j] - offs[i]) / (x[j] - x[i])
            if abs(slope) > np.tan(np.deg2rad(0.6)):
                continue
            pred = offs[i] + slope * (x - x[i])
            good = ok & (np.abs(offs - pred) <= tol_px)
            # longest contiguous run of good windows
            k = 0
            while k < m:
                if good[k]:
                    e = k
                    while e + 1 < m and good[e + 1]:
                        e += 1
                    if e - k > best_run[1] - best_run[0]:
                        best_run = (k, e)
                    k = e + 1
                else:
                    k += 1
    i, j = best_run
    if j - i + 1 < 3:
        return None
    return fit.mid + edges[i] * t, fit.mid + edges[j + 1] * t


def _merge_segments(segs: np.ndarray, cfg: DatumConfig) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Greedy collinear merge.  ``segs`` is (N, 4) rect px.  Returns
    ``(p0, p1, covered_length_px)`` per merged line."""
    if len(segs) == 0:
        return []
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lens)
    clusters: list[dict[str, Any]] = []
    for i in order:
        p0, p1 = segs[i, :2], segs[i, 2:]
        L = float(lens[i])
        if L < 6.0:
            continue
        t = (p1 - p0) / L
        placed = False
        for c in clusters:
            ct = c["t"]
            cross = abs(float(t[0] * ct[1] - t[1] * ct[0]))
            if cross > np.sin(np.deg2rad(cfg.merge_angle_deg)):
                continue
            n = np.array([-ct[1], ct[0]])
            if abs(float(n @ (0.5 * (p0 + p1) - c["p"]))) > cfg.merge_offset_px:
                continue
            s0, s1 = sorted((float(ct @ (p0 - c["p"])), float(ct @ (p1 - c["p"]))))
            # only join if it overlaps or is close to the existing extent
            if s0 > c["s1"] + 0.35 * (c["s1"] - c["s0"]) + 30 or s1 < c["s0"] - 0.35 * (c["s1"] - c["s0"]) - 30:
                continue
            c["pts"].append((p0, p1, L))
            c["s0"], c["s1"] = min(c["s0"], s0), max(c["s1"], s1)
            c["cov"] += L
            placed = True
            break
        if not placed:
            clusters.append({"p": 0.5 * (p0 + p1), "t": t, "s0": -0.5 * L, "s1": 0.5 * L, "cov": L, "pts": [(p0, p1, L)]})
    out = []
    for c in clusters:
        # refit through all end points, weighted by segment length
        P = np.array([p for a, b, _ in c["pts"] for p in (a, b)])
        wts = np.array([L for _, _, L in c["pts"] for _ in (0, 1)])
        ctr = (P * wts[:, None]).sum(0) / wts.sum()
        _, _, Vt = np.linalg.svd((P - ctr) * np.sqrt(wts)[:, None])
        t = Vt[0]
        s = (P - ctr) @ t
        out.append((ctr + s.min() * t, ctr + s.max() * t, float(min(c["cov"], s.max() - s.min()))))
    return out


def propose_in_frame(
    frame_id: str,
    bgr: np.ndarray,
    camera: CameraModel | None,
    H: np.ndarray,
    rect_bgr: np.ndarray,
    spec: RectSpec,
    tag_corners_plane: list[np.ndarray],
    cfg: DatumConfig | None = None,
) -> list[LineCandidate]:
    """Straight-feature candidates of one photo, measured in its own pixels."""
    cfg = cfg or DatumConfig()
    lum = linear_luminance(rect_bgr)
    g8 = np.clip(np.sqrt(np.clip(lum, 0, 1)) * 255.0, 0, 255).astype(np.uint8)  # perceptual-ish for the detector
    g8 = cv2.GaussianBlur(g8, (0, 0), 1.0)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(g8)[0]
    if lines is None or len(lines) == 0:
        return []
    segs = lines.reshape(-1, 4).astype(float)
    merged = _merge_segments(segs, cfg)
    # drop lines that are part of a tag (their own edges and cell boundaries)
    tag_boxes = []
    for c in tag_corners_plane:
        r = spec.plane_to_rect(c)
        m = cfg.tag_margin_mm / spec.mm_per_px
        tag_boxes.append((r[:, 0].min() - m, r[:, 1].min() - m, r[:, 0].max() + m, r[:, 1].max() + m))
    keep = []
    for p0, p1, covered in merged:
        L_mm = float(np.linalg.norm(p1 - p0)) * spec.mm_per_px
        if L_mm < cfg.min_length_mm or covered * spec.mm_per_px < 0.45 * L_mm:
            continue
        inside = False
        for x0, y0, x1, y1 in tag_boxes:
            if all(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in (p0, p1)):
                inside = True
        if not inside:
            keep.append((p0, p1, covered))
    keep.sort(key=lambda s: -s[2])
    keep = keep[: cfg.max_candidates_per_frame * 2]

    gray = to_gray(bgr)
    Hinv = np.linalg.inv(H)
    out: list[LineCandidate] = []
    px_per_mm = float(np.median(local_scale_px_per_mm(H, np.concatenate(tag_corners_plane)))) if tag_corners_plane else 1.0
    for k, (p0, p1, _cov) in enumerate(keep):
        q = spec.rect_to_plane(np.stack([p0, p1]))
        uv = apply_h(H, q)
        if float(np.linalg.norm(uv[1] - uv[0])) < 40.0:
            continue
        r = snap_line(gray, uv[0], uv[1], camera=camera, search_px=6.0, max_tilt_deg=2.0)
        if not r.ok or r.fit is None:
            continue
        sup = straight_support(gray, camera, r.fit, r.kind)
        if sup is None:
            continue
        if float(np.linalg.norm(sup[1] - sup[0])) < 0.97 * r.fit.length:
            # only part of the merged segment is one straight feature: measure that part
            r2 = snap_line(gray, sup[0], sup[1], camera=camera, search_px=4.0, max_tilt_deg=1.5)
            if not r2.ok or r2.fit is None:
                continue
            r = r2
        line, ends = fit_plane_line(r.fit, Hinv)
        if float(np.linalg.norm(ends[1] - ends[0])) < cfg.min_length_mm:
            continue
        # duplicates: the two flanks of a thin line, or the same edge found twice
        if any(c.line.angle_to(line) < np.deg2rad(1.0) and abs(float(c.line.signed_distance(ends.mean(axis=0)))) < 1.5 for c in out):
            continue
        L_mm = float(np.linalg.norm(ends[1] - ends[0]))
        cand = LineCandidate(
            id=f"{frame_id}:{len(out)}",
            frame_id=frame_id,
            kind=r.kind,
            fit=r.fit,
            line=line,
            endpoints=ends,
            length_mm=L_mm,
            contrast=r.fit.contrast,
            sigma_blur_px=r.fit.sigma_blur,
            distinctness=r.distinctness,
        )
        cand.score = float(min(L_mm, 600.0) / 600.0 * min(cand.contrast / 0.15, 1.0))
        out.append(cand)
        if len(out) >= cfg.max_candidates_per_frame:
            break
    return out


# --------------------------------------------------------------------------
# 2. multi-view classification
# --------------------------------------------------------------------------
@dataclass
class FrameContext:
    frame_id: str
    gray: np.ndarray
    camera: CameraModel | None
    H: np.ndarray
    #: T_cam_world of the multi-view solution (world = gauge tag frame = plane frame with z out of the panel)
    pose: Pose | None = None
    K: np.ndarray | None = None
    tag_blur_px: float = 1.0


def match_and_classify(cand: LineCandidate, frames: dict[str, FrameContext], cfg: DatumConfig | None = None) -> None:
    """Find ``cand`` again in the other photos and decide what it is."""
    cfg = cfg or DatumConfig()
    home = frames[cand.frame_id]
    cand.soft = bool(cand.kind == "step" and cand.sigma_blur_px > cfg.soft_edge_factor * max(home.tag_blur_px, 0.5))
    if cand.soft:
        cand.notes.append("edge is much softer than the tag edges: possibly a lighting gradient or a rounded transition")
    for fid, fc in frames.items():
        if fid == cand.frame_id:
            continue
        uv = apply_h(fc.H, cand.endpoints)
        Himg, Wimg = fc.gray.shape[:2]
        # clip the seed to the image
        seg = _clip_to_image(uv[0], uv[1], Wimg, Himg)
        if seg is None or float(np.linalg.norm(seg[1] - seg[0])) < 60.0:
            continue
        r = snap_line(fc.gray, seg[0], seg[1], camera=fc.camera, search_px=cfg.match_search_px, max_tilt_deg=3.0)
        if not r.ok or r.fit is None or r.kind != cand.kind:
            continue
        line, ends = fit_plane_line(r.fit, np.linalg.inv(fc.H))
        if line.angle_to(cand.line) > np.deg2rad(4.0):
            continue
        cand.matches[fid] = (r.fit, line, ends)
    if not cand.matches:
        return
    # spread of the feature's plane position across photos (zero for an on-plane feature)
    mid = cand.endpoints.mean(axis=0)
    offs = [0.0] + [float(l.oriented_towards(mid + cand.line.n).signed_distance(mid)) * (1 if l.n @ cand.line.n > 0 else -1) for _, l, _ in cand.matches.values()]
    cand.spread_mm = float(np.ptp(offs))
    posed = [fid for fid in [cand.frame_id, *cand.matches] if frames[fid].pose is not None and frames[fid].K is not None]
    if len(posed) < 2:
        cand.notes.append("camera positions unknown: cannot tell an on-plane seam from an off-plane edge")
        return
    planes = []
    for fid in posed:
        fc = frames[fid]
        e = cand.fit if fid == cand.frame_id else cand.matches[fid][0]
        assert e.mid is not None and fc.pose is not None and fc.K is not None
        h = 0.5 * e.length
        planes.append(view_plane_of_image_line(fc.pose, fc.K, e.mid - h * e.t, e.mid + h * e.t))
    tri = triangulate_line(planes)
    cand.stationarity_rms_mm = tri["rms_offset_mm"]
    if tri["baseline_quality"] < 0.02:
        cand.notes.append("the photos look along this feature from almost the same side: height above the plane is not observable")
        return
    # height of the 3-D line above the plane z = 0 (world z points out of the panel), at the feature's two ends
    P, d = tri["point"], tri["direction"]
    heights = []
    for q in cand.endpoints:
        s = float((np.array([q[0], q[1], 0.0]) - P) @ d)
        heights.append(float((P + s * d)[2]))
    stationary = tri["rms_offset_mm"] <= cfg.stationary_rms_mm and tri["max_angle_deg"] <= 1.0
    if not stationary and len(posed) >= 3:
        cand.classification = "view_dependent"
        cand.notes.append("its position changes with the viewpoint in a way no fixed line explains: silhouette of a curved surface, shadow or reflection")
    elif stationary or len(posed) == 2:
        cand.height_mm = float(np.mean(heights))
        if max(abs(h) for h in heights) <= cfg.on_plane_mm:
            cand.classification = "on_plane"
        else:
            cand.classification = "off_plane"
            cand.notes.append(f"a fixed edge about {abs(cand.height_mm):.1f} mm {'in front of' if cand.height_mm > 0 else 'behind'} the panel face")
        if len(posed) == 2:
            cand.notes.append("only two views: a silhouette cannot be told from a fixed edge yet")


def _clip_to_image(p0: np.ndarray, p1: np.ndarray, w: int, h: int, margin: float = 6.0) -> np.ndarray | None:
    x0, y0 = p0
    dx, dy = p1 - p0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - margin), (dx, w - 1 - margin - x0), (-dy, y0 - margin), (dy, h - 1 - margin - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return np.stack([p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0)])


# --------------------------------------------------------------------------
# pair selection
# --------------------------------------------------------------------------
@dataclass
class DatumProposal:
    floor: list[LineCandidate]
    side: list[LineCandidate]
    pairs: list[dict[str, Any]]
    ambiguous_floor: bool
    ambiguous_side: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "floor": [c.to_dict() for c in self.floor],
            "side": [c.to_dict() for c in self.side],
            "pairs": self.pairs,
            "ambiguous_floor": self.ambiguous_floor,
            "ambiguous_side": self.ambiguous_side,
            "notes": list(self.notes),
        }


def side_of_panel(floor: Line2D, side: Line2D, interior: np.ndarray) -> str:
    f = floor.oriented_towards(interior)
    s = side.oriented_towards(interior)
    return "left" if float(s.n[0] * f.n[1] - s.n[1] * f.n[0]) > 0 else "right"


def choose_pairs(
    cands: list[LineCandidate],
    tag_centers: np.ndarray,
    *,
    nominal_angle_deg: float = 90.0,
    expected_side: str | None = None,
    floor_on_plane: bool | None = True,
    side_on_plane: bool | None = True,
    image_up_plane: np.ndarray | None = None,
    cfg: DatumConfig | None = None,
) -> DatumProposal:
    """Rank (floor, side) assignments among the candidates.

    A pair qualifies when the two lines meet at the configured nominal angle
    (within a tolerance) and every tag lies in one quadrant between them.  The
    configured handedness (``expected_side``) decides which line is the floor
    and which the side; without it both assignments are offered and the
    direction that is "up" in the photo is used only to *order* them.
    """
    cfg = cfg or DatumConfig()
    interior = tag_centers.mean(axis=0)
    notes: list[str] = []
    usable = []
    for c in cands:
        d = c.line.signed_distance(tag_centers)
        if np.all(d > 2.0) or np.all(d < -2.0):  # all tags strictly on one side
            usable.append(c)

    def role_ok(c: LineCandidate, want_on_plane: bool | None) -> float:
        if want_on_plane and c.classification == "off_plane":
            return 0.15
        if want_on_plane and c.classification == "view_dependent":
            return 0.1
        if c.classification == "on_plane":
            return 1.25
        return 1.0

    pairs: list[dict[str, Any]] = []
    for a in usable:
        for b in usable:
            if a is b:
                continue
            ang = np.degrees(a.line.angle_to(b.line))
            # acute angle between undirected lines vs nominal folded into [0, 90]
            nominal = nominal_angle_deg if nominal_angle_deg <= 90 else 180 - nominal_angle_deg
            if abs(ang - nominal) > cfg.pair_angle_tol_deg:
                continue
            side_is = side_of_panel(a.line, b.line, interior)
            if expected_side in ("left", "right") and side_is != expected_side:
                continue
            score = a.score * role_ok(a, floor_on_plane) * b.score * role_ok(b, side_on_plane)
            up_bonus = 1.0
            if image_up_plane is not None:
                n_floor = a.line.oriented_towards(interior).n
                up_bonus = 1.0 + 0.25 * float(n_floor @ image_up_plane)  # ordering hint only
            pairs.append({"floor": a.id, "side": b.id, "score": float(score * up_bonus), "angle_deg": float(ang), "side_of_panel": side_is})
    pairs.sort(key=lambda p: -p["score"])
    by_id = {c.id: c for c in cands}
    floor_ids = list(dict.fromkeys(p["floor"] for p in pairs))
    side_ids = list(dict.fromkeys(p["side"] for p in pairs))

    def ambiguous(ids: list[str], key: str) -> bool:
        if len(ids) < 2:
            return False
        best = max(p["score"] for p in pairs if p[key] == ids[0])
        second = max(p["score"] for p in pairs if p[key] == ids[1])
        return bool(second >= cfg.ambiguity_fraction * best)

    if not pairs:
        notes.append("no pair of straight features meets at the configured angle with the tags between them")
    if expected_side is None and pairs:
        notes.append("the record does not say on which side the side datum lies; the order of suggestions follows the photo orientation only")
    return DatumProposal(
        floor=[by_id[i] for i in floor_ids],
        side=[by_id[i] for i in side_ids],
        pairs=pairs[:12],
        ambiguous_floor=ambiguous(floor_ids, "floor"),
        ambiguous_side=ambiguous(side_ids, "side"),
        notes=notes,
    )


# --------------------------------------------------------------------------
# 3. rounded edge from silhouettes
# --------------------------------------------------------------------------
def silhouette_offset(c_e: float, c_h: float, R: float) -> tuple[float, float]:
    """Where the silhouette of a cylindrical bend appears on the plane.

    Cross-section coordinates: ``e`` is the in-plane distance from the tangent
    line (positive into the flat region), ``h`` the height above the panel
    face.  The bend curves away below the face with radius ``R``; the camera
    is at ``(c_e, c_h)``, ``c_h > 0``.  Returns ``(e_obs, phi)``: the in-plane
    coordinate where the silhouette ray meets the face (negative: beyond the
    tangent line) and the bend angle of the silhouette ruling.
    """
    # Surface: p(phi) = (-R sin phi, -R (1 - cos phi)), outward normal
    # n(phi) = (-sin phi, cos phi).  Tangency (c - p) . n = 0 gives
    #     (c_h + R) cos phi - c_e sin phi = R
    # whose solution on the visible side is  phi = atan2(c_h + R, c_e) - asin(R / rho).
    B = R + c_h
    rho = float(np.hypot(c_e, B))
    if rho <= R or c_h <= 0:
        return float("nan"), float("nan")
    phi = float(np.arctan2(B, c_e) - np.arcsin(R / rho))
    if phi <= 0:
        return float("nan"), float("nan")
    p_e = -R * np.sin(phi)
    p_h = -R * (1.0 - np.cos(phi))
    e_obs = c_e + (p_e - c_e) * c_h / (c_h - p_h)
    return float(e_obs), phi


@dataclass
class CylinderFit:
    ok: bool
    reason: str = ""
    tangent: Line2D | None = None  # plane frame; normal points into the flat region
    radius: float = 0.0
    radius_fixed: bool = False
    rms_mm: float = 0.0
    n_views: int = 0
    sigma_tangent_mm: float = float("nan")
    sigma_radius_mm: float = float("nan")
    sigma_angle_deg: float = float("nan")
    bend_angles_deg: list[float] = field(default_factory=list)
    condition: float = float("nan")
    endpoints: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def f(x: float) -> float | None:
            return None if not np.isfinite(x) else round(float(x), 4)

        return {
            "ok": self.ok,
            "reason": self.reason,
            "tangent": None if self.tangent is None else {"n": self.tangent.n.tolist(), "d": self.tangent.d},
            "radius_mm": f(self.radius),
            "radius_fixed": self.radius_fixed,
            "rms_mm": f(self.rms_mm),
            "n_views": self.n_views,
            "sigma_tangent_mm": f(self.sigma_tangent_mm),
            "sigma_radius_mm": f(self.sigma_radius_mm),
            "sigma_angle_deg": f(self.sigma_angle_deg),
            "bend_angles_deg": [round(a, 1) for a in self.bend_angles_deg],
            "condition": f(self.condition),
            "endpoints": None if self.endpoints is None else self.endpoints.round(3).tolist(),
            "notes": list(self.notes),
        }


def fit_cylinder_from_silhouettes(
    observations: list[tuple[np.ndarray, np.ndarray, float]],
    interior: np.ndarray,
    *,
    radius: float | None = None,
    radius_bounds: tuple[float, float] = (3.0, 400.0),
    max_bend_deg: float = 87.0,
) -> CylinderFit:
    """Robust wrapper around :func:`_fit_cylinder`.

    A view from beyond the edge would need the bend to continue past
    ``max_bend_deg`` for its silhouette to lie on the cylinder; what such a
    view really shows is whatever comes after the bend.  Those views, and
    views that disagree with the common model by more than three times its
    scatter, are removed and the fit is repeated.
    """
    obs = list(observations)
    fit = _fit_cylinder(obs, interior, radius=radius, radius_bounds=radius_bounds)
    # never whittle a multi-view fit down to a trivial one
    need = max(1 if radius is not None else 3, (len(observations) + 1) // 2)
    dropped = 0
    for _ in range(len(observations)):
        if not fit.ok or fit.tangent is None or len(obs) <= need:
            break
        res = []
        for C, ends, _sig in obs:
            e_obs, _ = silhouette_offset(float(fit.tangent.n @ C[:2] - fit.tangent.d), float(C[2]), fit.radius)
            res.append(float(np.max(np.abs(ends @ fit.tangent.n - fit.tangent.d - e_obs))) if np.isfinite(e_obs) else np.inf)
        over = [k for k, a in enumerate(fit.bend_angles_deg) if not np.isfinite(a) or a > max_bend_deg]
        worst = int(np.argmax(res))
        outlier = res[worst] > max(3.0 * fit.rms_mm, 0.3) and fit.rms_mm > 0.15
        if over:
            k = max(over, key=lambda i: (fit.bend_angles_deg[i] if np.isfinite(fit.bend_angles_deg[i]) else 1e9))
        elif outlier:
            k = worst
        else:
            break
        obs.pop(k)
        dropped += 1
        fit = _fit_cylinder(obs, interior, radius=radius, radius_bounds=radius_bounds)
    if dropped and fit.ok:
        fit.notes.append(f"{dropped} view(s) not used: seen from beyond the edge or inconsistent with a cylinder")
    return fit


def _fit_cylinder(
    observations: list[tuple[np.ndarray, np.ndarray, float]],
    interior: np.ndarray,
    *,
    radius: float | None = None,
    radius_bounds: tuple[float, float] = (3.0, 400.0),
) -> CylinderFit:
    """Fit the tangent line, axis direction and radius of a cylindrical bend.

    ``observations``: per view ``(camera centre in plane coordinates (x, y, h)
    with h = height above the face, silhouette end points on the plane (2, 2),
    sigma_mm)``.  ``interior`` is a plane point on the flat side (tag
    centroid).  With ``radius`` given it is held fixed (configuration-defined
    radius); then a single view is enough.
    """
    n_obs = len(observations)
    if n_obs < (1 if radius is not None else 2):
        return CylinderFit(False, "too_few_views", n_views=n_obs)
    # initial tangent: the mean silhouette line shifted by ~R into the panel
    P = np.concatenate([o[1] for o in observations])
    ctr = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - ctr)
    t0 = Vt[0]
    n0 = np.array([-t0[1], t0[0]])
    if float(n0 @ (interior - ctr)) < 0:
        n0, t0 = -n0, -t0
    R0 = float(radius) if radius is not None else 30.0

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        ang = x[1]
        c, s = np.cos(ang), np.sin(ang)
        n = np.array([c * n0[0] - s * n0[1], s * n0[0] + c * n0[1]])
        d = float(n @ ctr) + x[0]
        R = float(radius) if radius is not None else float(np.exp(x[2]))
        return n, np.array([n[1], -n[0]]), d, R

    def residuals(x: np.ndarray) -> np.ndarray:
        n, _t, d, R = unpack(x)
        out = []
        for C, ends, sig in observations:
            c_e = float(n @ C[:2] - d)
            e_obs, _phi = silhouette_offset(c_e, float(C[2]), R)
            for q in ends:
                meas = float(n @ q - d)
                out.append((meas - e_obs) / max(sig, 1e-3) if np.isfinite(e_obs) else 1e3)
        return np.asarray(out)

    def fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = residuals(x)
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            h = 1e-4 if k != 1 else 1e-6
            xp, xm = x.copy(), x.copy()
            xp[k] += h
            xm[k] -= h
            J[:, k] = (residuals(xp) - residuals(xm)) / (2 * h)
        return r, J

    x0 = np.array([R0, 0.0] + ([] if radius is not None else [np.log(R0)]))
    best = None
    seeds = [R0] if radius is not None else [10.0, 30.0, 80.0]
    for Rs in seeds:
        xs = x0.copy()
        xs[0] = Rs
        if radius is None:
            xs[2] = np.log(Rs)
        sol = lm_solve(fn, xs, max_iter=60)
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    n, t, d, R = unpack(best.x)
    if radius is None and not (radius_bounds[0] <= R <= radius_bounds[1]):
        return CylinderFit(False, "radius_out_of_range", radius=R, n_views=n_obs)
    r_mm = residuals(best.x) * np.repeat([o[2] for o in observations], 2)
    dof = max(r_mm.size - best.x.size, 1)
    rms = float(np.sqrt(np.sum(r_mm**2) / dof)) if r_mm.size > best.x.size else float(np.sqrt(np.mean(r_mm**2)))
    _, J = fn(best.x)
    JTJ = J.T @ J
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-300))
    scale = max(1.0, float(best.cost) / dof)
    try:
        cov = np.linalg.inv(JTJ) * scale
    except np.linalg.LinAlgError:
        return CylinderFit(False, "degenerate", n_views=n_obs, condition=cond)
    phis = []
    for C, _ends, _sig in observations:
        _, phi = silhouette_offset(float(n @ C[:2] - d), float(C[2]), R)
        phis.append(float(np.degrees(phi)))
    line = Line2D(n, d)
    s = [(q - line.point) @ t for q in P]
    ends = np.stack([line.point + min(s) * t, line.point + max(s) * t])
    fit = CylinderFit(
        ok=True,
        tangent=line,
        radius=R,
        radius_fixed=radius is not None,
        rms_mm=rms,
        n_views=n_obs,
        sigma_tangent_mm=float(np.sqrt(max(cov[0, 0], 0.0))),
        sigma_angle_deg=float(np.degrees(np.sqrt(max(cov[1, 1], 0.0)))),
        sigma_radius_mm=(float(R * np.sqrt(max(cov[2, 2], 0.0))) if radius is None else 0.0),
        bend_angles_deg=phis,
        condition=cond,
        endpoints=ends,
    )
    if radius is None and fit.sigma_radius_mm > 0.5 * R:
        fit.ok = False
        fit.reason = "radius_not_identifiable"
        fit.notes.append("the views do not differ enough across the edge to separate radius and position")
    return fit
