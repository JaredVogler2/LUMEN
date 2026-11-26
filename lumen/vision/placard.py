"""Finding and measuring the perimeter of an installed placard (prompt 11).

Inputs are a photo, its plane-to-image mapping ``H`` (from the tags) and the
*expected* rectangle (the target from the operation record, plane millimetres).
The expected rectangle only says where to look and how big the thing is; the
measured perimeter comes from the picture.

Two stages
----------
1. **Proposal.**  A rectangle of the nominal size is slid and rotated over a
   metric (rectified) patch around the expected position.  A pose scores by
   the brightness step across all four sides, taken *with one sign for the
   whole perimeter*: a real placard is uniformly lighter or darker than the
   panel around it, while printed text and borders inside it change sign and
   cancel.  The proposal is rejected when no pose stands out, and flagged
   ambiguous when two different poses score alike.
2. **Measurement.**  Each side is then measured in the original pixels: many
   short profiles across the side locate the *outermost* edge of the right
   sign (so a printed border a few millimetres inside is not mistaken for the
   edge); a robust straight line through them tells how much of the side is
   really visible and whether it is straight; an edge-spread fit refines the
   line and gives its covariance.  Corners are intersections of neighbouring
   side lines, so a corner hidden by a finger or tape is still measured if the
   two sides are visible elsewhere — and is reported as not directly seen.

Conventions: pixel-centre coordinates; "ideal" pixels are lens-corrected;
sides are indexed 0 top (TL-TR), 1 right (TR-BR), 2 bottom (BR-BL), 3 left
(BL-TL), matching the corner order TL, TR, BR, BL of the target rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, local_scale_px_per_mm
from .edgefit import EdgeFit, corner_jacobian, fit_edge, intersect_lines
from .imaging import SRGB_TO_LINEAR_LUT, to_gray

SIDE_NAMES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class PlacardConfig:
    #: how far from the expected position the proposal stage looks (mm)
    search_mm: float = 45.0
    angle_search_deg: float = 12.0
    coarse_step_mm: float = 1.5
    coarse_angle_step_deg: float = 1.0
    #: a proposal must stand this many robust sigmas above the other poses
    min_proposal_z: float = 6.0
    #: a second, different pose scoring at least this fraction of the best is an ambiguity
    ambiguity_ratio: float = 0.8
    #: half-length of the profiles across a side (mm)
    side_scan_mm: float = 7.0
    #: fraction of a side left out at each end (rounded corners, curl)
    trim_frac: float = 0.14
    #: fraction of the profiles that must lie on one straight edge
    min_side_support: float = 0.55
    #: a side is "partly hidden" below this
    full_side_support: float = 0.85
    max_sagitta_px: float = 1.0
    max_saturated: float = 0.3
    #: standard deviation given to an operator-marked corner that was not snapped (px)
    manual_sigma_px: float = 1.5
    #: bound on the edge-definition error of a side (dark-side shift, sticker thickness), px
    edge_definition_bound_px: float = 0.25
    #: look three times further for a side that is not found near the outline (a part of another size).  Switched
    #: off for a placard in a hand: there, a side that is not where it should be is hidden, and the next straight
    #: edge further in is the hand's.
    wide_side_retry: bool = True


@dataclass
class SideFit:
    index: int
    ok: bool = False
    reason: str = ""
    n: np.ndarray | None = None  # unit normal pointing outward (ideal px)
    mid: np.ndarray | None = None
    length_px: float = 0.0
    var_d: float = 0.0
    var_theta: float = 0.0
    cov_d_theta: float = 0.0
    support: float = 0.0
    saturated: float = 0.0
    contrast: float = 0.0
    sagitta_px: float = 0.0
    sigma_blur_px: float = 0.0
    polarity: int = 0  # +1: outside brighter than the placard edge, -1: darker
    method: str = ""  # "esf" | "profile" | "manual"
    flags: list[str] = field(default_factory=list)
    #: which ends of the side had no edge (0 = start corner, 1 = end corner)
    hidden_ends: list[int] = field(default_factory=list)
    #: unit vector along the side (start corner to end corner) and where on it the edge was seen (px from ``mid``)
    t: np.ndarray | None = None
    along: np.ndarray | None = None

    @property
    def name(self) -> str:
        return SIDE_NAMES[self.index]

    def as_edgefit(self) -> EdgeFit:
        return EdgeFit(ok=self.ok, n=self.n, mid=self.mid, length=self.length_px, var_d=self.var_d, var_theta=self.var_theta, cov_d_theta=self.cov_d_theta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.name, "ok": self.ok, "reason": self.reason, "method": self.method,
            "support": round(self.support, 3), "saturated": round(self.saturated, 3), "contrast": round(self.contrast, 4),
            "sagitta_px": round(self.sagitta_px, 3), "sigma_px": round(float(np.sqrt(max(self.var_d, 0.0))), 4),
            "sigma_blur_px": round(self.sigma_blur_px, 3), "polarity": self.polarity, "flags": list(self.flags), "hidden_ends": list(self.hidden_ends),
        }


@dataclass
class PlacardMeasurement:
    ok: bool
    reason: str = ""
    #: TL, TR, BR, BL in ideal pixels (NaN rows where a corner could not be formed)
    corners_ideal: np.ndarray = field(default_factory=lambda: np.full((4, 2), np.nan))
    corners_cov: np.ndarray = field(default_factory=lambda: np.zeros((8, 8)))
    sides: list[SideFit] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    proposal: dict[str, Any] = field(default_factory=dict)
    provenance: str = "measured"  # "measured" | "manual"
    #: distance from each corner to where the edge is first seen (mm); large = rounded, folded or covered
    corner_gap_mm: list[float | None] = field(default_factory=lambda: [None] * 4)

    @property
    def complete(self) -> bool:
        return bool(self.ok and np.all(np.isfinite(self.corners_ideal)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "reason": self.reason, "complete": self.complete, "provenance": self.provenance,
            "corners_ideal": [[None if not np.isfinite(v) else float(v) for v in row] for row in self.corners_ideal],
            "sides": [s.to_dict() for s in self.sides], "flags": list(self.flags), "proposal": self.proposal,
            "corner_gap_mm": list(self.corner_gap_mm),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _remap(img: np.ndarray, x: np.ndarray, y: np.ndarray, border: float = float("nan")) -> np.ndarray:
    """Bilinear samples of a float image at raw pixel positions of any shape.

    ``cv2.remap`` limits each map dimension to 32767, so the positions are
    laid out as rows of a fixed width.
    """
    shape = x.shape
    xf = np.ascontiguousarray(x, dtype=np.float32).ravel()
    yf = np.ascontiguousarray(y, dtype=np.float32).ravel()
    n = xf.size
    wdt = 2048
    rows = int(np.ceil(n / wdt))
    pad = rows * wdt - n
    if pad:
        xf = np.concatenate([xf, np.full(pad, -10.0, np.float32)])
        yf = np.concatenate([yf, np.full(pad, -10.0, np.float32)])
    out = cv2.remap(img, xf.reshape(rows, wdt), yf.reshape(rows, wdt), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return out.ravel()[:n].reshape(shape)


def _sample(img: np.ndarray, camera: CameraModel | None, ideal_xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of a float image at ideal pixel positions (NaN outside)."""
    flat = np.asarray(ideal_xy, dtype=float).reshape(-1, 2)
    raw = camera.distort_pixels(flat) if (camera is not None and camera.has_distortion) else flat
    return _remap(img, raw[:, 0], raw[:, 1]).reshape(np.asarray(ideal_xy).shape[:-1])


def _rect_corners(center: np.ndarray, w: float, h: float, ang: float, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL of a rectangle; ``ang`` rotates the width axis from ``ex`` toward ``ey``."""
    c, s = np.cos(ang), np.sin(ang)
    wd = c * ex + s * ey
    hd = -s * ex + c * ey
    return np.stack([center - 0.5 * w * wd + 0.5 * h * hd, center + 0.5 * w * wd + 0.5 * h * hd, center + 0.5 * w * wd - 0.5 * h * hd, center - 0.5 * w * wd - 0.5 * h * hd])


# --------------------------------------------------------------------------
# stage 1: proposal
# --------------------------------------------------------------------------
def propose_rectangle(
    gray: np.ndarray,
    camera: CameraModel | None,
    H: np.ndarray,
    expected_plane: np.ndarray,
    cfg: PlacardConfig | None = None,
) -> dict[str, Any]:
    """Best-supported pose of a nominal-size rectangle near the expected one.

    Returns ``{"found", "corners_plane", "offset_mm", "angle_deg", "score",
    "z", "polarity", "ambiguous", "alternatives", ...}``; plane millimetres.
    The pose only has to be good to a few millimetres: the sides are measured
    afterwards from the pixels.
    """
    from scipy import ndimage

    cfg = cfg or PlacardConfig()
    E = np.asarray(expected_plane, dtype=float).reshape(4, 2)
    center0 = E.mean(axis=0)
    wvec, hvec = E[1] - E[0], E[0] - E[3]
    W, Hh = float(np.linalg.norm(wvec)), float(np.linalg.norm(hvec))
    ex, ey = wvec / W, hvec / Hh
    ppm = float(np.median(local_scale_px_per_mm(H, E)))
    mmpp = 1.0 / max(ppm, 1e-6)
    half = 0.5 * float(np.hypot(W, Hh)) + cfg.search_mm + 6.0
    n = int(min(max(int(np.ceil(2 * half / mmpp)), 64), 1400))
    mmpp = 2 * half / n
    # metric patch aligned with the expected rectangle: column = along width, row = downward (-height)
    cols = (np.arange(n) + 0.5) * mmpp - half
    rows = half - (np.arange(n) + 0.5) * mmpp
    P = center0 + cols[None, :, None] * ex + rows[:, None, None] * ey
    lin = SRGB_TO_LINEAR_LUT[to_gray(gray)]
    patch = _sample(lin, camera, apply_h(H, P.reshape(-1, 2)).reshape(n, n, 2))
    valid = np.isfinite(patch)
    if valid.mean() < 0.5:
        return {"found": False, "reason": "expected_area_outside_image", "px_per_mm": ppm}
    patch = np.where(valid, patch, float(np.nanmedian(patch)))
    # smooth to about the coarse step, so that a pose half a step off an edge still responds
    sm = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), max(0.8, 0.55 * cfg.coarse_step_mm / mmpp))
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8.0  # d/d col  (+ex)
    gy = -cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8.0  # d/d(-row) (+ey)
    unit_x, unit_y = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    def score(ox: np.ndarray, oy: np.ndarray, ang: float, K: int) -> np.ndarray:
        """Outward brightness step averaged over the perimeter (signed), per offset."""
        t = (np.arange(K) + 0.5) / K
        t = cfg.trim_frac * 0.7 + t * (1 - 2 * cfg.trim_frac * 0.7)
        c = _rect_corners(np.zeros(2), W, Hh, ang, unit_x, unit_y)
        acc = np.zeros(ox.shape, np.float32)
        for k in range(4):
            p0, p1 = c[k], c[(k + 1) % 4]
            d = p1 - p0
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)  # corners run clockwise in (ex, ey): outward
            pts = p0[None, :] + t[:, None] * d[None, :]
            X = (ox[..., None] + pts[:, 0] + half) / mmpp - 0.5
            Y = (half - (oy[..., None] + pts[:, 1])) / mmpp - 0.5
            acc += (_remap(gx, X, Y, 0.0) * nrm[0] + _remap(gy, X, Y, 0.0) * nrm[1]).mean(axis=-1)
        return acc / 4.0

    offs = np.arange(-cfg.search_mm, cfg.search_mm + 1e-6, cfg.coarse_step_mm)
    angs = np.deg2rad(np.arange(-cfg.angle_search_deg, cfg.angle_search_deg + 1e-6, cfg.coarse_angle_step_deg))
    ox, oy = np.meshgrid(offs, offs)
    S = np.stack([score(ox, oy, float(a), 18) for a in angs])  # (angle, y, x)
    A = np.abs(S)
    med = float(np.median(A))
    mad = float(np.median(np.abs(A - med))) * 1.4826 + 1e-9
    # distinct maxima: strongest pose and its rivals
    rad = max(1, int(round(4.0 / cfg.coarse_step_mm)))
    mx = ndimage.maximum_filter(A, size=(2 * int(round(3.0 / cfg.coarse_angle_step_deg)) + 1, 2 * rad + 1, 2 * rad + 1), mode="nearest")
    peaks = np.argwhere((A == mx) & (A > med + 3.0 * mad))
    order = np.argsort([-A[tuple(i)] for i in peaks])
    peaks = peaks[order][:6]
    if not len(peaks):
        return {"found": False, "reason": "no_rectangle_stands_out", "z": float((A.max() - med) / mad), "px_per_mm": ppm}
    ai, yi, xi = (int(v) for v in peaks[0])
    best = float(A[ai, yi, xi])
    z = (best - med) / mad
    pol = 1 if S[ai, yi, xi] > 0 else -1
    alts = []
    second = 0.0
    for i in peaks[1:]:
        da = abs(int(i[0]) - ai) * cfg.coarse_angle_step_deg
        dist = float(np.hypot(offs[i[2]] - offs[xi], offs[i[1]] - offs[yi]))
        sc = float(A[tuple(i)])
        alts.append({"offset_mm": [float(offs[i[2]]), float(offs[i[1]])], "angle_deg": float(np.degrees(angs[i[0]])), "score": sc, "ratio": sc / best})
        # a rival is a clearly different placement, not the same placard a border-width away
        if dist > max(9.0, 0.2 * min(W, Hh)) or da > 5.0:
            second = max(second, sc)
    # refine around the best pose
    fstep = max(0.25, 0.5 * mmpp)
    fo = np.arange(-cfg.coarse_step_mm, cfg.coarse_step_mm + 1e-6, fstep)
    fx, fy = np.meshgrid(offs[xi] + fo, offs[yi] + fo)
    fine = (best, float(angs[ai]), float(offs[xi]), float(offs[yi]))
    for a in angs[ai] + np.deg2rad(np.arange(-cfg.coarse_angle_step_deg, cfg.coarse_angle_step_deg + 1e-6, 0.25)):
        acc = pol * score(fx, fy, float(a), 28)
        j = np.unravel_index(int(np.argmax(acc)), acc.shape)
        if float(acc[j]) > fine[0]:
            fine = (float(acc[j]), float(a), float(fx[j]), float(fy[j]))
    sc, ang, dxm, dym = fine
    corners = _rect_corners(center0 + dxm * ex + dym * ey, W, Hh, ang, ex, ey)
    found = bool(z >= cfg.min_proposal_z)
    lim = cfg.search_mm - cfg.coarse_step_mm
    return {
        "found": found, "reason": "" if found else "no_rectangle_stands_out",
        "corners_plane": corners, "offset_mm": [float(dxm), float(dym)], "angle_deg": float(np.degrees(ang)),
        "score": float(sc), "z": float(z), "polarity": int(pol),
        "ambiguous": bool(second >= cfg.ambiguity_ratio * best), "second_score_ratio": float(second / max(best, 1e-12)),
        "alternatives": alts, "px_per_mm": ppm,
        "at_search_limit": bool(abs(dxm) >= lim or abs(dym) >= lim or abs(np.degrees(ang)) >= cfg.angle_search_deg - 0.5),
    }


# --------------------------------------------------------------------------
# stage 2: sides
# --------------------------------------------------------------------------
def measure_side(
    gray_u8: np.ndarray,
    lin: np.ndarray,
    camera: CameraModel | None,
    p0: np.ndarray,
    p1: np.ndarray,
    outward: np.ndarray,
    index: int,
    scan_px: float,
    polarity: int,
    cfg: PlacardConfig,
) -> SideFit:
    """Measure one side between the predicted corners ``p0`` -> ``p1`` (ideal px)."""
    sf = SideFit(index=index, polarity=polarity)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 12.0:
        sf.reason = "side_too_short"
        return sf
    t = d / L
    nrm = outward / np.linalg.norm(outward)
    a0, a1 = cfg.trim_frac * L, (1.0 - cfg.trim_frac) * L
    M = int(np.clip((a1 - a0) / 2.5, 12, 80))
    st = a0 + (np.arange(M) + 0.5) * (a1 - a0) / M
    step = 0.5
    ss = np.arange(-scan_px, scan_px + 1e-6, step)
    pts = p0[None, None, :] + st[:, None, None] * t + ss[None, :, None] * nrm
    prof = _sample(lin, camera, pts)  # (M, S)
    ok_rows = np.isfinite(prof).all(axis=1)
    if ok_rows.sum() < max(6, M // 3):
        sf.reason = "side_outside_image"
        return sf
    prof = np.where(np.isfinite(prof), prof, 0.0).astype(np.float32)
    k = cv2.getGaussianKernel(9, 1.6).ravel().astype(np.float32)
    sm = cv2.filter2D(prof, -1, k[None, :], borderType=cv2.BORDER_REPLICATE)
    g = polarity * np.gradient(sm, step, axis=1)  # positive at an edge of the placard's sign
    core = g[ok_rows]
    # Noise from an end of the profiles, not from the edge itself: the outer end is bare panel when the side is
    # where it was expected.  A side that lies further out (a placard followed from where it was a moment ago)
    # puts its edge into that end; then the inner end is the calm one.  The calmer of the two is taken.
    q = max(4, core.shape[1] // 5)
    noise = min(float(np.median(np.abs(e - np.median(e)))) for e in (core[:, -q:], core[:, :q])) * 1.4826 + 1e-6
    peak_all = float(np.percentile(core.max(axis=1), 70))
    thr = max(5.0 * noise, 0.3 * peak_all)
    pos = np.full(M, np.nan)
    inner_gap = []
    for i in range(M):
        if not ok_rows[i]:
            continue
        gi = g[i]
        idx = np.nonzero((gi[1:-1] > thr) & (gi[1:-1] >= gi[:-2]) & (gi[1:-1] >= gi[2:]))[0] + 1
        if not len(idx):
            continue
        j = int(idx[-1])  # outermost edge of the right sign
        den = gi[j - 1] - 2 * gi[j] + gi[j + 1]
        off = 0.5 * (gi[j - 1] - gi[j + 1]) / den if abs(den) > 1e-12 else 0.0
        pos[i] = ss[j] + float(np.clip(off, -1, 1)) * step
        # nearest other strong gradient further inside (either sign): limits the fitting band
        inner = np.nonzero(np.abs(gi[:j - 2]) > thr)[0] if j > 3 else np.array([], dtype=int)
        if len(inner):
            # walk inward from the edge until the gradient of this edge has died away
            jj = j
            while jj > 1 and gi[jj - 1] > 0.3 * gi[j] and gi[jj - 1] <= gi[jj]:
                jj -= 1
            far = inner[inner < jj]
            if len(far):
                inner_gap.append((j - int(far[-1])) * step)
    have = np.isfinite(pos)
    if have.sum() < max(6, int(0.25 * M)):
        sf.reason = "edge_not_found"
        sf.support = float(have.sum()) / M
        return sf
    # robust straight line through the edge positions
    x = st[have] - 0.5 * L
    y = pos[have]
    w = np.ones_like(y)
    a = b = 0.0
    sig = 1.0
    for _ in range(8):
        Aw = np.stack([np.ones_like(x), x], axis=1) * w[:, None]
        sol, *_ = np.linalg.lstsq(Aw, y * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = y - (a + b * x)
        sig = max(float(np.median(np.abs(r))) * 1.4826, 0.12)
        u = r / (4.0 * sig)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() < 4:
            break
    r = y - (a + b * x)
    inl = np.abs(r) < max(0.6, 3.0 * sig)
    sf.support = float(inl.sum()) / M
    if sf.support < cfg.min_side_support * 0.6 or inl.sum() < 6:
        sf.reason = "edge_not_found"
        return sf
    # which stretch of the side is unsupported?
    good = np.zeros(M, dtype=bool)
    good[np.nonzero(have)[0][inl]] = True
    q = max(2, M // 5)
    if good[:q].mean() < 0.34:
        sf.hidden_ends.append(0)
    if good[-q:].mean() < 0.34:
        sf.hidden_ends.append(1)
    xi, yi = x[inl], y[inl]
    if len(xi) >= 8:
        c2 = np.polyfit(xi, yi, 2)[0]
        sf.sagitta_px = float(abs(c2) * (0.5 * (a1 - a0)) ** 2)
    # refined endpoints of the supported, trimmed stretch
    lo, hi = float(xi.min()), float(xi.max())
    mid = p0 + 0.5 * L * t
    q0 = mid + lo * t + (a + b * lo) * nrm
    q1 = mid + hi * t + (a + b * hi) * nrm
    hw = float(np.clip(0.45 * np.median(inner_gap), 2.0, 6.0)) if len(inner_gap) >= max(3, int(0.3 * inl.sum())) else 5.0
    sf.length_px = float(np.linalg.norm(q1 - q0))
    ef = fit_edge(gray_u8, q0, q1, polarity * nrm, half_width=hw, trim=0.0, camera=camera, min_contrast=0.02)
    line_n = nrm - b * t
    line_n = line_n / np.linalg.norm(line_n)
    line_mid = 0.5 * (q0 + q1)
    use_esf = bool(ef.ok and ef.n is not None and ef.mid is not None and abs(float((ef.mid - line_mid) @ line_n)) < 0.75 and ef.inlier_frac > 0.6)
    if use_esf:
        assert ef.n is not None and ef.mid is not None
        n_out = ef.n if float(ef.n @ nrm) > 0 else -ef.n
        sf.n, sf.mid = n_out, ef.mid
        sf.var_d, sf.var_theta, sf.cov_d_theta = float(ef.var_d), float(ef.var_theta), float(ef.cov_d_theta)
        if float(ef.n @ nrm) < 0:
            sf.cov_d_theta = -sf.cov_d_theta
        sf.saturated, sf.contrast, sf.sigma_blur_px = float(ef.saturated_frac), float(ef.contrast), float(ef.sigma_blur)
        sf.method = "esf"
    else:
        nn = int(inl.sum())
        s2 = float(np.mean(r[inl] ** 2))
        sxx = float(np.sum((xi - xi.mean()) ** 2)) + 1e-9
        sf.n, sf.mid = line_n, line_mid
        sf.var_d, sf.var_theta = max(s2 / nn, 0.05**2), max(s2 / sxx, 1e-8)
        sf.method = "profile"
        sf.flags.append("edge_fit_fallback")
        lv_out = float(np.nanmedian(prof[ok_rows][:, -4:]))
        lv_in = float(np.nanmedian(prof[ok_rows][:, :4]))
        sf.contrast = abs(lv_out - lv_in)
        sf.saturated = float(np.mean(gray_u8 >= 254)) if gray_u8.size < 4e6 else 0.0
    sf.ok = True
    assert sf.mid is not None
    sf.t = t
    sf.along = xi + float((mid - sf.mid) @ t)
    if sf.support < cfg.min_side_support:
        sf.ok = False
        sf.reason = "edge_mostly_hidden"
    elif sf.support < cfg.full_side_support:
        sf.flags.append("edge_partly_hidden")
    if sf.sagitta_px > cfg.max_sagitta_px:
        sf.flags.append("edge_not_straight")
    if sf.saturated > cfg.max_saturated:
        sf.flags.append("glare_on_edge")
    return sf


def _corners_from_sides(sides: list[SideFit]) -> tuple[np.ndarray, np.ndarray]:
    """Corner k is where side k-1 and side k meet (TL = left x top, ...)."""
    C = np.full((4, 2), np.nan)
    cov = np.zeros((8, 8))
    for k in range(4):
        a, b = sides[(k - 1) % 4], sides[k]
        if not (a.ok and b.ok):
            continue
        ea, eb = a.as_edgefit(), b.as_edgefit()
        x = intersect_lines(ea, eb)
        if x is None:
            continue
        C[k] = x
        J = corner_jacobian(ea, eb, x)
        S = np.zeros((4, 4))
        S[0, 0], S[1, 1], S[0, 1], S[1, 0] = a.var_d, a.var_theta, a.cov_d_theta, a.cov_d_theta
        S[2, 2], S[3, 3], S[2, 3], S[3, 2] = b.var_d, b.var_theta, b.cov_d_theta, b.cov_d_theta
        cov[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = J @ S @ J.T
    return C, cov


def corner_gaps(
    lin: np.ndarray,
    camera: CameraModel | None,
    sides: list[SideFit],
    corners: np.ndarray,
    polarity: int,
) -> np.ndarray:
    """How far from each corner (pixels along the sides) the edge is first found.

    The corners are intersections of the side lines, i.e. where the corner
    *would* be.  A corner that is rounded, folded over, torn or covered shows
    as a stretch next to it where no edge lies on the side line.  Returns one
    value per corner (TL, TR, BR, BL), NaN where a side is missing.
    """
    out = np.full(4, np.nan)
    if not np.all(np.isfinite(corners)):
        return out
    ss = np.arange(-4.0, 4.0 + 1e-6, 0.5)
    k1 = cv2.getGaussianKernel(7, 1.2).ravel().astype(np.float32)
    start_gap = np.full(4, np.nan)
    end_gap = np.full(4, np.nan)
    for sd in sides:
        if not sd.ok or sd.n is None:
            continue
        p0, p1 = corners[sd.index], corners[(sd.index + 1) % 4]
        L = float(np.linalg.norm(p1 - p0))
        if L < 16:
            continue
        t = (p1 - p0) / L
        st = np.arange(1.0, L - 1.0 + 1e-6, 1.0)
        prof = _sample(lin, camera, p0[None, None, :] + st[:, None, None] * t + ss[None, :, None] * sd.n)
        prof = np.where(np.isfinite(prof), prof, 0.0).astype(np.float32)
        g = polarity * np.gradient(cv2.filter2D(prof, -1, k1[None, :], borderType=cv2.BORDER_REPLICATE), 0.5, axis=1)
        j = np.argmax(g, axis=1)
        ref = float(np.median(g[len(st) // 4 : 3 * len(st) // 4].max(axis=1)))
        present = (g.max(axis=1) > 0.45 * ref) & (np.abs(ss[j]) <= 1.25)
        # tolerate single misses: smooth the presence along the side
        pres = np.convolve(present.astype(float), np.ones(3) / 3.0, mode="same") > 0.5
        idx = np.nonzero(pres)[0]
        if not len(idx):
            continue
        start_gap[sd.index] = float(st[idx[0]])
        end_gap[sd.index] = float(L - st[idx[-1]])
    for k in range(4):
        a, b = end_gap[(k - 1) % 4], start_gap[k]
        if np.isfinite(a) and np.isfinite(b):
            out[k] = max(a, b)
    return out


def _linear_planes(image: np.ndarray) -> list[np.ndarray]:
    """The picture's colour channels in linear light (one plane for a grey picture)."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return [SRGB_TO_LINEAR_LUT[arr]]
    return [SRGB_TO_LINEAR_LUT[np.ascontiguousarray(arr[..., c])] for c in range(arr.shape[2])]


def _levels_across(
    planes: list[np.ndarray], camera: CameraModel | None, sd: SideFit, frac: float = 0.8,
    inner: tuple[float, float] = (3.0, 6.0), outer: tuple[float, float] = (3.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Colour inside and outside a side line, at points along it: arrays ``(points, channels)``, linear light.

    ``inner`` / ``outer``: from how many to how many pixels off the line the colour is taken.  For a measured side
    the points are where its edge was seen (so a hand over the rest of the side does not enter); for a side that
    was placed, its middle ``frac``.
    """
    assert sd.n is not None and sd.mid is not None
    if sd.along is not None and sd.t is not None and len(sd.along) >= 4:
        t = sd.t
        st = np.asarray(sd.along, dtype=float)
        st = st[:: max(1, len(st) // 60)]
    else:
        t = np.array([sd.n[1], -sd.n[0]])
        K = int(np.clip(frac * sd.length_px / 2.0, 6, 60))
        st = (np.arange(K) + 0.5) / K * frac * sd.length_px - 0.5 * frac * sd.length_px
    base = sd.mid[None, None, :] + st[:, None, None] * t
    p_in = base - np.arange(inner[0], inner[1] + 1e-6, 0.5)[None, :, None] * sd.n
    p_out = base + np.arange(outer[0], outer[1] + 1e-6, 0.5)[None, :, None] * sd.n
    with np.errstate(all="ignore"):
        inside = np.stack([np.nanmedian(_sample(pl, camera, p_in), axis=1) for pl in planes], axis=1)
        outside = np.stack([np.nanmedian(_sample(pl, camera, p_out), axis=1) for pl in planes], axis=1)
    return inside, outside


#: Around a measured side the panel is sampled 3 to 6 px outside it, and the placard 6 to 10 px inside: a printed
#: border along the edge is a few pixels wide, and in a video frame or a JPEG its colour is smeared over its
#: neighbours (chroma is stored at half resolution), so the first pixels inside an edge say little about the placard.
_REF_INNER = (6.0, 10.0)


def _side_colours(planes: list[np.ndarray], camera: CameraModel | None, sides: list[SideFit]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Median colour of the placard inside, and of whatever is outside, every measured side."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sd in sides:
        if sd.ok and sd.method not in ("record_size", "manual") and sd.n is not None and sd.mid is not None and sd.length_px > 8:
            a, b = _levels_across(planes, camera, sd, inner=_REF_INNER)
            ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if ok.sum() >= 4:
                out[sd.index] = (np.median(a[ok], axis=0), np.median(b[ok], axis=0))
    return out


def _same_surface(c: np.ndarray, ref: np.ndarray, chroma_tol: float, factor: float) -> np.ndarray:
    """Do colours ``c`` (..., channels) look like the surface with colour ``ref``?

    Hue is compared closely (the share of each channel in the sum, which light and shade leave alone); brightness
    only coarsely (within ``factor`` either way), because a hand also throws a shadow on what lies next to it.
    """
    c = np.asarray(c, dtype=float)
    y, yr = c.mean(axis=-1), float(np.mean(ref))
    q = c / np.maximum(c.sum(axis=-1, keepdims=True), 1e-6)
    qr = np.asarray(ref, dtype=float) / max(float(np.sum(ref)), 1e-6)
    return (np.abs(q - qr).max(axis=-1) < chroma_tol) & (y < factor * yr) & (y * factor > yr)


def _colour_distance(c: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance between colours ``c`` (..., channels) and ``ref`` in units of what is plainly visible: a hue shift
    of 0.03 (share of a channel in the sum) or a brightness step of 10 % count as one."""
    c = np.asarray(c, dtype=float)
    ref = np.asarray(ref, dtype=float)
    q = c / np.maximum(c.sum(axis=-1, keepdims=True), 1e-6)
    qr = ref / max(float(ref.sum()), 1e-6)
    hue = np.abs(q - qr).max(axis=-1) / 0.03
    bright = np.abs(np.log(np.maximum(c.mean(axis=-1), 1e-6) / max(float(ref.mean()), 1e-6))) / 0.10
    return np.hypot(hue, bright)


def edge_fraction(planes: list[np.ndarray], camera: CameraModel | None, placed: SideFit, measured: list[SideFit]) -> float | None:
    """How much of a side that was *placed* shows the placard's own edge: placard inside, panel outside.

    Then the side is not hidden at all; it was only not found where it was looked for (a placard that moved
    further between two frames than a side is followed), and it can be measured where it was placed.
    """
    ref = _side_colours(planes, camera, measured)
    if not ref or placed.n is None or placed.mid is None:
        return None
    c_in = np.median(np.stack([v[0] for v in ref.values()]), axis=0)
    c_out = np.median(np.stack([v[1] for v in ref.values()]), axis=0)
    apart = float(_colour_distance(c_in, c_out))
    if apart < 1.5:
        return None
    a, b = _levels_across(planes, camera, placed, inner=_REF_INNER)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if ok.sum() < 4:
        return None
    tol = 0.35 * apart
    return float(np.mean((_colour_distance(a[ok], c_in) < tol) & (_colour_distance(b[ok], c_out) < tol)))


def plain_fraction(planes: list[np.ndarray], camera: CameraModel | None, placed: SideFit, measured: list[SideFit]) -> float | None:
    """How much of a side that was *placed*, not measured, lies in plain panel or plain placard.

    A side can only be hidden where something hides it.  Where the picture shows undisturbed panel on both sides
    of the placed line, the placard does not reach that far; where it shows undisturbed placard on both sides, the
    placard goes on beyond it.  Either way the outline is not where it was placed (a hand's edge was taken for a
    side, or the part has another size).  "Undisturbed" is judged against the colours outside and inside the sides
    that were measured.  Returns the fraction of the line that looks plain, or ``None`` when that cannot be judged
    (panel and placard look alike, or there is nothing to compare with).
    """
    ref = _side_colours(planes, camera, measured)
    if not ref or placed.n is None or placed.mid is None:
        return None
    c_in = np.median(np.stack([v[0] for v in ref.values()]), axis=0)
    c_out = np.median(np.stack([v[1] for v in ref.values()]), axis=0)
    # how far apart panel and placard are; "looks like the panel" means: within a third of that from the panel
    apart = float(_colour_distance(c_in, c_out))
    if apart < 1.5:
        return None
    a, b = _levels_across(planes, camera, placed)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if ok.sum() < 4:
        return None
    a, b = a[ok], b[ok]
    tol = 0.35 * apart
    panel = (_colour_distance(a, c_out) < tol) & (_colour_distance(b, c_out) < tol)
    placard = (_colour_distance(a, c_in) < tol) & (_colour_distance(b, c_in) < tol)
    return float(np.mean(panel | placard))


def foreign_sides(image: np.ndarray, camera: CameraModel | None, sides: list[SideFit]) -> list[int]:
    """Measured sides that do not have, outside them, what the other sides have.

    All sides of a placard on a panel have the panel outside.  A straight edge with something else outside is not
    a side of the placard.  Two cases are told apart from the colour a few pixels outside each measured side:

    * it is not the panel's (compared with the median of all sides: by hue closely, by brightness only within a
      factor of three, because what covers the placard also shades the panel next to it): the edge of a hand or a
      tool lying over the placard;
    * it is nearer to the placard's own colour than to the panel's: an edge *inside* the placard, such as the
      ends of a block of print, which a third of a side's length is enough to pass for a side while a hand
      covers the real one.

    With fewer than three measured sides there is no majority: if two differ from each other, both are returned.

    What this cannot see: something of the panel's own hue and of similar brightness (a grey glove on a grey
    panel).  And on a placard that lies partly against a different background (another label, a painted stripe)
    it points at the side that has it; it is meant for a placard that is being handled, not for records.
    """
    both = _side_colours(_linear_planes(image), camera, sides)
    cols = {k: v[1] for k, v in both.items()}
    if len(cols) < 2:
        return []
    keys = sorted(cols)
    if len(keys) == 2:
        return keys if not bool(_same_surface(cols[keys[0]], cols[keys[1]], 0.05, 3.0)) else []
    panel = np.median(np.stack([cols[k] for k in keys]), axis=0)
    placard = np.median(np.stack([both[k][0] for k in keys]), axis=0)
    apart = float(_colour_distance(placard, panel))
    out = []
    for k in keys:
        not_panel = not bool(_same_surface(cols[k], panel, 0.05, 3.0))
        inside_placard = apart >= 1.5 and float(_colour_distance(cols[k], placard)) < float(_colour_distance(cols[k], panel))
        if not_panel or inside_placard:
            out.append(k)
    return out


def complete_from_size(
    m: PlacardMeasurement, H: np.ndarray, width_mm: float, height_mm: float, *, allow_two: bool = False,
    image: np.ndarray | None = None, camera: CameraModel | None = None, max_plain: float = 0.5,
) -> list[int]:
    """Place the sides that could not be measured by the placard's known size (plane millimetres).

    For following a placard that a hand partly covers.  A hidden side lies parallel to the opposite, measured
    side, at the placard's height (top / bottom) or width (left / right) from it *on the panel*; it is computed
    there and carried back into the picture.  One hidden side is placed from the three measured ones.  Two
    adjacent hidden sides are placed from the other two only when ``allow_two`` is set, which is sensible only for
    a placard that was seen whole a moment ago.  Two opposite hidden sides cannot be placed: nothing says where
    the placard is along them.

    The sides placed carry the method ``"record_size"``, the variances of the side they were derived from and the
    flag ``side_from_record_size``; the corners next to them are flagged as not directly seen.  This is an
    assumption (the part has the size the record gives), so it serves guidance only: the caller has to add what
    the assumption may be wrong by to the uncertainty, and a record needs the whole outline.

    With ``image`` the placement is checked against the picture (:func:`plain_fraction`): a side is placed only
    where something is seen to cover it.  If more than ``max_plain`` of a placed side lies in plain panel or
    plain placard, or if that cannot be judged (panel and placard look too much alike), nothing is changed.

    Returns the indices of the sides placed (empty when nothing was changed).
    """
    if len(m.sides) != 4:
        return []
    missing = [k for k in range(4) if not m.sides[k].ok]
    if not missing or len(missing) > 2:
        return []
    if len(missing) == 2 and (not allow_two or (missing[1] - missing[0]) % 4 == 2):
        return []
    Hinv = np.linalg.inv(H)
    made: list[SideFit] = []
    for j in missing:
        src = m.sides[(j + 2) % 4]
        if not src.ok or src.n is None or src.mid is None or src.length_px <= 0:
            return []
        dist = float(height_mm if j in (0, 2) else width_mm)
        t = np.array([src.n[1], -src.n[0]])
        ends = np.stack([src.mid - 0.5 * src.length_px * t, src.mid + 0.5 * src.length_px * t])
        A, B = apply_h(Hinv, ends)
        d = B - A
        n_p = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-12)
        out_dir = apply_h(Hinv, (src.mid + 2.0 * src.n)[None, :])[0] - apply_h(Hinv, src.mid[None, :])[0]
        if float(n_p @ out_dir) < 0:
            n_p = -n_p  # outward for the measured side: the hidden side lies the other way
        a2, b2 = apply_h(H, np.stack([A - dist * n_p, B - dist * n_p]))
        d2 = b2 - a2
        L2 = float(np.linalg.norm(d2))
        if not np.isfinite(L2) or L2 < 4.0:
            return []
        n2 = np.array([-d2[1], d2[0]]) / L2
        mid2 = 0.5 * (a2 + b2)
        if float(n2 @ (mid2 - src.mid)) < 0:
            n2 = -n2  # outward for the hidden side: away from the side it was placed from
        made.append(SideFit(
            index=j, ok=True, n=n2, mid=mid2, length_px=L2, var_d=src.var_d, var_theta=src.var_theta, cov_d_theta=0.0,
            support=0.0, polarity=src.polarity, method="record_size", flags=["side_from_record_size"],
        ))
    if image is not None:
        planes = _linear_planes(image)
        for sd in made:
            plain = plain_fraction(planes, camera, sd, m.sides)
            m.proposal.setdefault("placed_side_plain", {})[SIDE_NAMES[sd.index]] = None if plain is None else round(plain, 3)
            seen = edge_fraction(planes, camera, sd, m.sides)
            m.proposal.setdefault("placed_side_edge", {})[SIDE_NAMES[sd.index]] = None if seen is None else round(seen, 3)
            if plain is None or plain > max_plain:
                return []  # nothing covers it there (or that cannot be told): the outline is not placed
    trial = list(m.sides)
    for sd in made:
        trial[sd.index] = sd
    C, cov = _corners_from_sides(trial)
    if not np.all(np.isfinite(C)):
        return []
    m.sides = trial
    m.corners_ideal, m.corners_cov = C, cov
    m.ok, m.reason = True, ""
    names = {SIDE_NAMES[sd.index] for sd in made}
    m.flags = [f for f in m.flags if f not in ("partial_perimeter", "placard_not_found") and not (f.startswith("side_not_measured:") and f.split(":")[1] in names)]
    for sd in made:
        m.flags.append(f"side_from_record_size:{sd.name}")
        for corner in (sd.index, (sd.index + 1) % 4):
            tag = f"corner_not_directly_seen:{('TL', 'TR', 'BR', 'BL')[corner]}"
            if tag not in m.flags:
                m.flags.append(tag)
    return [sd.index for sd in made]


def measure_placard(
    image: np.ndarray,
    camera: CameraModel | None,
    H: np.ndarray,
    expected_plane: np.ndarray,
    *,
    manual_corners_ideal: np.ndarray | None = None,
    snap: bool = True,
    cfg: PlacardConfig | None = None,
    seed_corners_ideal: np.ndarray | None = None,
    seed_polarity: int = 0,
) -> PlacardMeasurement:
    """Measure the placard near ``expected_plane`` (or near operator-marked corners).

    ``seed_corners_ideal`` (with ``seed_polarity``, the sign of its edges if known) says where the placard was a
    moment ago, for following it from one video frame to the next: the search for a rectangle is skipped and the
    sides are measured near the seed.  A seed is a place to look, nothing more: sides that are not found there are
    reported as not measured, never filled in from the seed.
    """
    cfg = cfg or PlacardConfig()
    gray = to_gray(image)
    lin = SRGB_TO_LINEAR_LUT[gray]
    E = np.asarray(expected_plane, dtype=float).reshape(4, 2)
    ppm = float(np.median(local_scale_px_per_mm(H, E)))
    out = PlacardMeasurement(ok=False)
    Himg, Wimg = gray.shape[:2]

    seed = None if seed_corners_ideal is None else np.asarray(seed_corners_ideal, dtype=float).reshape(4, 2)
    if seed is not None and not np.all(np.isfinite(seed)):
        seed = None
    if manual_corners_ideal is not None:
        start = np.asarray(manual_corners_ideal, dtype=float).reshape(4, 2)
        out.provenance = "manual"
        out.proposal = {"found": True, "manual": True}
        scan = float(np.clip(3.0 * ppm, 6.0, 26.0))
        pol = 0
    elif seed is not None:
        start = seed
        out.proposal = {"found": True, "seeded": True}
        scan = float(np.clip(cfg.side_scan_mm * ppm, 6.0, 40.0))
        pol = int(np.sign(seed_polarity))
    else:
        prop = propose_rectangle(gray, camera, H, E, cfg)
        out.proposal = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in prop.items()}
        if not prop.get("found"):
            out.reason = prop.get("reason", "placard_not_found")
            out.flags.append("placard_not_found")
            return out
        if prop["ambiguous"]:
            out.flags.append("ambiguous_proposal")
        if prop["at_search_limit"]:
            out.flags.append("at_search_limit")
        start = apply_h(H, prop["corners_plane"])
        scan = float(np.clip(cfg.side_scan_mm * ppm, 6.0, 40.0))
        pol = int(prop["polarity"])

    ctr = start.mean(axis=0)
    wn, hn = float(np.linalg.norm(E[1] - E[0])), float(np.linalg.norm(E[0] - E[3]))
    Hinv = np.linalg.inv(H)

    def outward_normal(p0: np.ndarray, p1: np.ndarray, centre: np.ndarray) -> np.ndarray:
        d = p1 - p0
        nrm = np.array([d[1], -d[0]]) / max(float(np.linalg.norm(d)), 1e-9)
        return -nrm if float(nrm @ (0.5 * (p0 + p1) - centre)) < 0 else nrm

    def pair_mismatch(sides: list[SideFit]) -> float:
        """Largest relative difference between the distance of two measured opposite sides and the record's
        size in that direction (plane millimetres); infinite when no opposite pair was measured."""
        vals = []
        for ia, ib, nominal in ((0, 2, hn), (1, 3, wn)):
            sa, sb = sides[ia], sides[ib]
            if not (sa.ok and sb.ok) or sa.mid is None or sb.mid is None or sb.n is None:
                continue
            tb = np.array([sb.n[1], -sb.n[0]])
            A = apply_h(Hinv, sa.mid[None, :])[0]
            Q = apply_h(Hinv, np.stack([sb.mid - 0.5 * sb.length_px * tb, sb.mid + 0.5 * sb.length_px * tb]))
            d = Q[1] - Q[0]
            nrm = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-12)
            vals.append(abs(abs(float((A - Q[0]) @ nrm)) - nominal) / nominal)
        return max(vals) if vals else float("inf")

    def manual_side(k: int) -> SideFit:
        p0, p1 = start[k], start[(k + 1) % 4]
        L = float(np.linalg.norm(p1 - p0))
        return SideFit(
            index=k, ok=True, n=outward_normal(p0, p1, start.mean(axis=0)), mid=0.5 * (p0 + p1), length_px=L,
            var_d=cfg.manual_sigma_px**2, var_theta=2.0 * cfg.manual_sigma_px**2 / max(L, 1.0) ** 2, support=1.0, method="manual", flags=["manual_side"],
        )

    def run(polarity: int) -> tuple[list[SideFit], np.ndarray]:
        """Two passes: along the starting rectangle, then along the measured sides with a tighter scan."""
        cur, centre = start.copy(), ctr.copy()
        sides: list[SideFit] = []
        Cn = np.full((4, 2), np.nan)
        for it in range(2):
            sides = []
            for k in range(4):
                p0, p1 = cur[k], cur[(k + 1) % 4]
                nrm = outward_normal(p0, p1, centre)
                sd = measure_side(gray, lin, camera, p0, p1, nrm, k, scan if it == 0 else max(0.6 * scan, 8.0), polarity, cfg)
                if cfg.wide_side_retry and not sd.ok and it == 0 and sd.reason == "edge_not_found":
                    # a part of another size has sides well away from the nominal outline: look further
                    wide = measure_side(gray, lin, camera, p0, p1, nrm, k, min(3.0 * scan, 70.0), polarity, cfg)
                    if wide.ok:
                        sd = wide
                sides.append(sd)
            Cn, _ = _corners_from_sides(sides)
            if it == 0 and np.all(np.isfinite(Cn)):
                cur, centre = Cn, Cn.mean(axis=0)
            else:
                break
        return sides, Cn

    if manual_corners_ideal is not None and not snap:
        out.sides = [manual_side(k) for k in range(4)]
    else:
        # The placard's edge may be lighter or darker than the panel, and print
        # inside it has edges of both signs.  Both signs are tried; the one that
        # gives a complete perimeter of the recorded size is the placard.
        cands = []
        # A placard followed from the previous frame keeps the sign of its edges: the other sign is print inside
        # it, and with a side hidden by a hand that print would otherwise offer "more sides" than the placard.
        signs = (pol,) if (seed is not None and pol) else ((pol, -pol) if pol else (1, -1))
        for polarity in signs:
            sides, Cn = run(polarity)
            n_ok = sum(sd.ok for sd in sides)
            rel, area = float("inf"), 0.0
            if np.all(np.isfinite(Cn)):
                Pm = apply_h(Hinv, Cn)
                wm = 0.5 * (np.linalg.norm(Pm[1] - Pm[0]) + np.linalg.norm(Pm[2] - Pm[3]))
                hm = 0.5 * (np.linalg.norm(Pm[0] - Pm[3]) + np.linalg.norm(Pm[1] - Pm[2]))
                rel = float(max(abs(wm - wn) / wn, abs(hm - hn) / hn))
                area = float(wm * hm)
            cands.append({
                "polarity": polarity, "sides": sides, "n_ok": n_ok, "rel": rel, "area": area, "support": float(np.mean([sd.support for sd in sides])),
                "pair_rel": pair_mismatch(sides),
            })
        full = [c for c in cands if c["n_ok"] == 4 and np.isfinite(c["rel"])]
        if manual_corners_ideal is not None:
            # the operator's quadrilateral decides which edges are meant: stay with the nearest ones
            pick = max(cands, key=lambda c: (c["n_ok"], c["support"]))
        elif full:
            match = [c for c in full if c["rel"] <= 0.06]
            pick = min(match, key=lambda c: c["rel"]) if match else max(full, key=lambda c: c["area"])
            if not match:
                out.flags.append("size_differs_from_record")
        else:
            # No complete outline.  What decides first is whether two opposite sides, where both were measured, are
            # the record's distance apart: a printed border a few millimetres inside the edge has sides too, of
            # the other sign and closer together, and with a hand's straight edge of that sign it can even offer
            # more "sides" than the placard.  A candidate whose opposite sides fit comes before one where none
            # was measured, and that before one whose opposite sides do not fit.  Then the number of sides, then
            # the sign the whole perimeter voted for.
            def fit_class(c: dict[str, Any]) -> int:
                return 1 if not np.isfinite(c["pair_rel"]) else (2 if c["pair_rel"] <= 0.03 else 0)

            pick = max(cands, key=lambda c: (fit_class(c), c["n_ok"], bool(pol) and c["polarity"] == pol, c["support"]))
        out.sides = pick["sides"]
        out.proposal["edge_polarity"] = int(pick["polarity"])
        out.proposal["polarities_tried"] = [
            {"polarity": c["polarity"], "sides_ok": c["n_ok"], "size_mismatch": None if not np.isfinite(c["rel"]) else round(c["rel"], 4),
             "opposite_sides_mismatch": None if not np.isfinite(c["pair_rel"]) else round(c["pair_rel"], 4)} for c in cands
        ]
        if manual_corners_ideal is not None:
            for sd in out.sides:
                if not sd.ok:
                    # keep the operator's side where the picture gives no edge
                    m = manual_side(sd.index)
                    sd.ok, sd.n, sd.mid, sd.length_px = True, m.n, m.mid, m.length_px
                    sd.var_d, sd.var_theta, sd.cov_d_theta, sd.method = m.var_d, m.var_theta, 0.0, "manual"
                    sd.flags.append("manual_side")
    C, cov = _corners_from_sides(out.sides)
    out.corners_ideal, out.corners_cov = C, cov
    n_ok = sum(s.ok for s in out.sides)
    gaps_px = corner_gaps(lin, camera, out.sides, C, int(out.proposal.get("edge_polarity") or pol or 1)) if not (manual_corners_ideal is not None and not snap) else np.full(4, np.nan)
    out.corner_gap_mm = [None if not np.isfinite(g) else round(float(g) / ppm, 2) for g in gaps_px]
    if np.isfinite(gaps_px).sum() >= 3:
        med = float(np.nanmedian(gaps_px))
        for k in range(4):
            if np.isfinite(gaps_px[k]) and gaps_px[k] / ppm > 5.0 and gaps_px[k] > 2.0 * med + 2.0 * ppm:
                out.flags.append(f"corner_lifted_or_hidden:{('TL', 'TR', 'BR', 'BL')[k]}")
    for s in out.sides:
        for fl in s.flags:
            tag = f"{fl}:{s.name}"
            if tag not in out.flags:
                out.flags.append(tag)
        if not s.ok:
            out.flags.append(f"side_not_measured:{s.name}")
        # a corner is "not directly seen" when the side has no edge near it
        for e in s.hidden_ends:
            corner = (s.index + e) % 4
            name = ("TL", "TR", "BR", "BL")[corner]
            if f"corner_not_directly_seen:{name}" not in out.flags:
                out.flags.append(f"corner_not_directly_seen:{name}")
    inside = np.isfinite(C).all(axis=1) & (C[:, 0] >= 0) & (C[:, 0] <= Wimg - 1) & (C[:, 1] >= 0) & (C[:, 1] <= Himg - 1)
    for k in range(4):
        if np.isfinite(C[k]).all() and not inside[k]:
            out.flags.append(f"corner_outside_image:{('TL', 'TR', 'BR', 'BL')[k]}")
    if n_ok < 3:
        if n_ok <= 1 and manual_corners_ideal is None:
            # the proposal was a weak pattern in the panel, not a placard
            out.reason = "no_rectangle_stands_out"
            out.flags = ["placard_not_found"]
        else:
            out.reason = "too_few_sides_measured"
        return out
    out.ok = True
    if n_ok == 3:
        out.flags.append("partial_perimeter")
    return out
