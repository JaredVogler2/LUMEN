"""AprilTag 36h11 detection with metrology-grade corner localisation.

Conventions
-----------
* **Tag size** means the *detection edge*: the side of the black square (the
  boundary between the black border and the white quiet zone), exactly as the
  AprilTag library defines ``tagsize``.  It is **not** the sticker outline and
  not the 10x10-cell image including the white border.  Operators measure this
  edge with a caliper; the packaging size is only a nominal fallback.
* **Corner order** is top-left, top-right, bottom-right, bottom-left of the tag
  in its own upright orientation (clockwise in the image).
* **Tag frame**: origin at the tag centre, ``+x`` to the tag's right, ``+y`` to
  the tag's top, ``+z`` out of the printed face.  Corners are therefore at
  ``(-s/2, +s/2)``, ``(+s/2, +s/2)``, ``(+s/2, -s/2)``, ``(-s/2, -s/2)``.

Pipeline
--------
1. OpenCV ``ArucoDetector`` (``DICT_APRILTAG_36h11``) proposes quads and IDs.
2. Each of the four border edges is localised with the edge-spread fit of
   :mod:`lumen.vision.edgefit` (linear light, lens model applied); corners are
   the intersections of adjacent lines.
3. The decoded pattern is verified by resampling the cells through the refined
   homography; this also yields a decoding margin.
4. *Polarity-balanced bias estimation*: interior cell boundaries have both
   polarities (black→white and white→black).  A residual tone-curve
   non-linearity moves every edge toward its dark side by the same amount
   ``delta``; fitting all boundaries jointly makes ``delta`` observable and
   lets us remove it from the border (otherwise it is a pure scale error).
5. Quality gates reject clipped, tiny, oblique, bent, glare-damaged, ambiguous
   and duplicate tags.  Nothing rejected here can contribute scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, dlt_homography
from ..synth.tags import CELLS, FAMILY_SIZE, tag_bits
from .edgefit import EdgeFit, corner_jacobian, fit_edge, intersect_lines
from .imaging import SRGB_TO_LINEAR_LUT, to_gray

#: Corner signs in the tag frame (x right, y up) for TL, TR, BR, BL.
CORNER_SIGNS = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])

#: Corners in "cell" coordinates (u right, v down, origin at the TL corner).
_CELL_CORNERS = np.array([[0.0, 0.0], [CELLS, 0.0], [CELLS, CELLS], [0.0, CELLS]])


def tag_object_points(size_mm: float) -> np.ndarray:
    """3-D corners of a tag in its own frame, ``(4, 3)``, order TL TR BR BL."""
    h = 0.5 * float(size_mm)
    return np.concatenate([CORNER_SIGNS * h, np.zeros((4, 1))], axis=1)


def tag_plane_points(size_mm: float) -> np.ndarray:
    """2-D corners of a tag in its own frame, ``(4, 2)``."""
    return CORNER_SIGNS * (0.5 * float(size_mm))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarkerConfig:
    #: Smallest accepted tag edge in pixels (below this corner noise dominates).
    min_edge_px: float = 28.0
    #: Tags whose shortest/longest edge ratio is below this are too oblique.
    min_edge_ratio: float = 0.28
    #: Detection is run at most at this long-edge size, then refined at full size.
    detect_long_edge: int = 2000
    #: Maximum decoded bit errors after refinement.
    max_bit_errors: int = 2
    #: Edge must be this straight (sagitta, px) or the tag is treated as bent.
    max_sagitta_px: float = 0.6
    #: Fraction of clipped (code >= 254) pixels on an edge band that marks
    #: over-exposure of the white margin.
    max_edge_saturation: float = 0.15
    #: Minimum fraction of pixels consistent with the edge model.
    min_edge_inliers: float = 0.6
    #: Minimum linear-light contrast between black and white.
    min_contrast: float = 0.05
    #: Estimate and remove the dark-side edge bias from interior boundaries.
    bias_correction: bool = True
    #: Smallest cell size (px) for which interior boundaries are measured.
    bias_min_cell_px: float = 6.5
    #: Fraction of an estimated edge bias retained as (systematic) uncertainty.
    bias_model_fraction: float = 0.35
    #: Floor of the edge-bias uncertainty (px), also used when it is not estimable.
    bias_floor_px: float = 0.01
    #: Variance inflation applied to formal corner covariances.  The formal
    #: covariance of the edge fits matches Monte-Carlo scatter for white noise
    #: (ratio 0.8–1.0) and under-predicts by up to 1.6x in variance for JPEG
    #: quality 75, whose noise is spatially correlated (docs/BUILD_LOG.md).
    covariance_inflation: float = 1.6
    #: Never report a corner standard deviation below this (px); this is the
    #: deterministic model error observed on noise-free synthetic images.
    sigma_floor_px: float = 0.012


@dataclass
class TagSizeTable:
    """Measured detection-edge size per tag ID, with provenance.

    ``nominal_mm`` is used for IDs without a measurement and is always
    labelled as nominal: nominal scale can support a research preview but never
    validated guidance.
    """

    nominal_mm: float = 50.0
    measured_mm: dict[int, float] = field(default_factory=dict)
    #: Standard uncertainty of a caliper measurement of the black edge.
    measured_u_mm: float = 0.05
    #: Half-width of the assumed tolerance of an unmeasured (nominal) tag.
    nominal_tolerance_mm: float = 1.0

    def size(self, tag_id: int) -> tuple[float, str]:
        if int(tag_id) in self.measured_mm:
            return float(self.measured_mm[int(tag_id)]), "measured"
        return float(self.nominal_mm), "nominal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_mm": self.nominal_mm,
            "measured_mm": {str(k): v for k, v in sorted(self.measured_mm.items())},
            "measured_u_mm": self.measured_u_mm,
            "nominal_tolerance_mm": self.nominal_tolerance_mm,
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "TagSizeTable":
        d = d or {}
        return TagSizeTable(
            nominal_mm=float(d.get("nominal_mm", 50.0)),
            measured_mm={int(k): float(v) for k, v in (d.get("measured_mm") or {}).items()},
            measured_u_mm=float(d.get("measured_u_mm", 0.05)),
            nominal_tolerance_mm=float(d.get("nominal_tolerance_mm", 1.0)),
        )


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class TagDetection:
    tag_id: int
    #: corners as recorded by the lens (raw pixels), order TL TR BR BL
    corners_px: np.ndarray
    #: corners in ideal (undistorted) pixels; equals ``corners_px`` without calibration
    corners_ideal: np.ndarray
    #: 8x8 covariance of ``corners_ideal`` flattened as (x0, y0, x1, y1, ...)
    cov: np.ndarray
    size_mm: float
    size_provenance: str
    accepted: bool = True
    reject_reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    refined: bool = False
    edge_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    sigma_blur_px: float = 0.0
    contrast: float = 0.0
    edge_rms: float = 0.0
    bit_errors: int = 0
    decode_margin: float = 0.0
    #: estimated dark-side edge shift (px) removed from the border, and its std
    bias_px: float = 0.0
    bias_sigma_px: float = float("nan")
    bias_applied: bool = False
    n_interior_edges: int = 0
    max_sagitta_px: float = 0.0
    saturation: float = 0.0

    @property
    def mean_edge_px(self) -> float:
        return float(np.mean(self.edge_px))

    @property
    def min_edge_px(self) -> float:
        return float(np.min(self.edge_px))

    @property
    def center_ideal(self) -> np.ndarray:
        """Projective centre (intersection of the diagonals)."""
        c = self.corners_ideal
        d1 = c[2] - c[0]
        d2 = c[3] - c[1]
        A = np.stack([d1, -d2], axis=1)
        try:
            s = np.linalg.solve(A, c[1] - c[0])
            return c[0] + s[0] * d1
        except np.linalg.LinAlgError:
            return c.mean(axis=0)

    @property
    def corner_sigma_px(self) -> float:
        """RMS per-coordinate standard deviation of the corners."""
        return float(np.sqrt(np.trace(self.cov) / 8.0))

    def quality(self) -> float:
        """0..1 summary used only for ranking (never as a confidence)."""
        size = np.clip((self.min_edge_px - 20.0) / 100.0, 0.0, 1.0)
        sharp = np.clip(1.6 / max(self.sigma_blur_px, 0.4), 0.0, 1.0)
        contrast = np.clip(self.contrast / 0.5, 0.0, 1.0)
        ok = 1.0 if self.accepted else 0.25
        return float(ok * (0.45 * size + 0.3 * sharp + 0.25 * contrast))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "corners_px": np.asarray(self.corners_px).round(3).tolist(),
            "corners_ideal": np.asarray(self.corners_ideal).round(3).tolist(),
            "corner_sigma_px": round(self.corner_sigma_px, 4),
            "size_mm": self.size_mm,
            "size_provenance": self.size_provenance,
            "accepted": self.accepted,
            "reject_reasons": list(self.reject_reasons),
            "flags": list(self.flags),
            "refined": self.refined,
            "edge_px": [round(float(e), 2) for e in self.edge_px],
            "sigma_blur_px": round(self.sigma_blur_px, 3),
            "contrast": round(self.contrast, 4),
            "bit_errors": self.bit_errors,
            "decode_margin": round(self.decode_margin, 3),
            "bias_px": round(self.bias_px, 4),
            "bias_sigma_px": None if not np.isfinite(self.bias_sigma_px) else round(self.bias_sigma_px, 4),
            "bias_applied": self.bias_applied,
            "max_sagitta_px": round(self.max_sagitta_px, 3),
            "saturation": round(self.saturation, 4),
            "quality": round(self.quality(), 3),
        }


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
_DICT = None


def _dictionary():
    global _DICT
    if _DICT is None:
        _DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return _DICT


def _detector(min_edge_px: float, long_edge: int) -> "cv2.aruco.ArucoDetector":
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    p.adaptiveThreshWinSizeMin = 5
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 12
    p.minMarkerPerimeterRate = float(np.clip(4.0 * min_edge_px * 0.6 / long_edge, 0.005, 0.2))
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.05
    p.minDistanceToBorder = 1
    p.errorCorrectionRate = 0.4
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    return cv2.aruco.ArucoDetector(_dictionary(), p)


def _raw_detect(gray: np.ndarray, cfg: MarkerConfig) -> list[tuple[int, np.ndarray]]:
    """Candidate (id, corners[4,2]) in full-resolution raw pixel coordinates."""
    H, W = gray.shape[:2]
    long_edge = max(H, W)
    out: dict[int, list[np.ndarray]] = {}
    scales = [1.0]
    if long_edge > cfg.detect_long_edge:
        scales = [cfg.detect_long_edge / float(long_edge), 1.0]
    for i, s in enumerate(scales):
        if s < 1.0:
            small = cv2.resize(gray, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
            sx = small.shape[1] / float(W)
            sy = small.shape[0] / float(H)
        else:
            small, sx, sy = gray, 1.0, 1.0
        det = _detector(cfg.min_edge_px * min(sx, sy), max(small.shape[:2]))
        corners, ids, _rej = det.detectMarkers(small)
        if ids is not None:
            for c, tid in zip(corners, ids.ravel()):
                q = c.reshape(4, 2).astype(np.float64)
                # map pixel centres of the reduced image back to full resolution
                q[:, 0] = (q[:, 0] + 0.5) / sx - 0.5
                q[:, 1] = (q[:, 1] + 0.5) / sy - 0.5
                lst = out.setdefault(int(tid), [])
                # the same physical tag found at two scales is one candidate
                if not any(np.max(np.linalg.norm(q - e, axis=1)) < 0.25 * _mean_edge(q) for e in lst):
                    lst.append(q)
        if i == 0 and len(scales) > 1 and len(out) >= 4:
            break  # enough found at reduced size; skip the expensive full-size pass
    return [(tid, q) for tid, qs in sorted(out.items()) for q in qs]


def _mean_edge(q: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)))


def _edges(q: np.ndarray) -> tuple[float, float, float, float]:
    e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    return (float(e[0]), float(e[1]), float(e[2]), float(e[3]))


def _sample_cells(gray: np.ndarray, camera: CameraModel | None, Hc: np.ndarray, sub: int = 3) -> np.ndarray:
    """Mean linear intensity of each cell (8x8), sampled in the cell interior."""
    offs = (np.arange(sub) + 0.5) / sub * 0.5 + 0.25  # central half of the cell
    uu, vv = np.meshgrid(np.arange(CELLS), np.arange(CELLS))
    pts = []
    for oy in offs:
        for ox in offs:
            pts.append(np.stack([uu.ravel() + ox, vv.ravel() + oy], axis=1))
    P = np.concatenate(pts, axis=0)
    xy = apply_h(Hc, P)
    if camera is not None and camera.has_distortion:
        xy = camera.distort_pixels(xy)
    Himg, Wimg = gray.shape[:2]
    xi = np.clip(np.rint(xy[:, 0]).astype(np.intp), 0, Wimg - 1)
    yi = np.clip(np.rint(xy[:, 1]).astype(np.intp), 0, Himg - 1)
    vals = SRGB_TO_LINEAR_LUT[gray[yi, xi]].reshape(sub * sub, CELLS * CELLS)
    return vals.mean(axis=0).reshape(CELLS, CELLS)


def _interior_runs(bits: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Maximal straight boundary runs between cells of different colour.

    Returns ``(p0, p1, normal_to_white)`` in cell coordinates (u right, v down).
    Only boundaries strictly inside the tag are returned (the outer border is
    measured separately).
    """
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    # horizontal boundaries between row r-1 and row r (v = r), r = 1..7
    for r in range(1, CELLS):
        c = 0
        while c < CELLS:
            up, dn = bits[r - 1, c], bits[r, c]
            if up == dn:
                c += 1
                continue
            c0 = c
            while c < CELLS and bits[r - 1, c] == up and bits[r, c] == dn:
                c += 1
            nrm = np.array([0.0, 1.0]) if dn else np.array([0.0, -1.0])
            runs.append((np.array([float(c0), float(r)]), np.array([float(c), float(r)]), nrm))
    for cc in range(1, CELLS):
        r = 0
        while r < CELLS:
            lf, rt = bits[r, cc - 1], bits[r, cc]
            if lf == rt:
                r += 1
                continue
            r0 = r
            while r < CELLS and bits[r, cc - 1] == lf and bits[r, cc] == rt:
                r += 1
            nrm = np.array([1.0, 0.0]) if rt else np.array([-1.0, 0.0])
            runs.append((np.array([float(cc), float(r0)]), np.array([float(cc), float(r)]), nrm))
    return runs


def _refine(
    gray: np.ndarray,
    camera: CameraModel | None,
    q_raw: np.ndarray,
    tag_id: int,
    cfg: MarkerConfig,
) -> tuple[np.ndarray, np.ndarray, list[EdgeFit], dict[str, Any]] | str:
    """Edge-fit refinement.  Returns ``(corners_ideal, cov8, edge_fits, info)``
    or a rejection reason."""
    q = camera.undistort_pixels(q_raw) if (camera is not None and camera.has_distortion) else q_raw.copy()
    centre = q.mean(axis=0)
    cell_px = _mean_edge(q) / CELLS
    fits: list[EdgeFit] = []
    half_width = float(np.clip(0.42 * cell_px, 2.0, 9.0))
    for it in range(2):
        fits = []
        for k in range(4):
            p0, p1 = q[k], q[(k + 1) % 4]
            mid = 0.5 * (p0 + p1)
            L = float(np.linalg.norm(p1 - p0))
            trim = float(np.clip(0.75 * cell_px, 2.5, 0.2 * L))
            f = fit_edge(
                gray,
                p0,
                p1,
                mid - centre,
                half_width=half_width if it else float(np.clip(0.7 * cell_px, 3.0, 12.0)),
                trim=trim,
                camera=camera,
                min_contrast=cfg.min_contrast,
            )
            if not f.ok:
                return f"edge_{f.reason}"
            fits.append(f)
        new_q = np.empty((4, 2))
        for k in range(4):
            x = intersect_lines(fits[(k - 1) % 4], fits[k])
            if x is None:
                return "degenerate_quad"
            new_q[k] = x
        if float(np.max(np.linalg.norm(new_q - q, axis=1))) > 0.6 * _mean_edge(q):
            return "refinement_inconsistent"
        q = new_q
        centre = q.mean(axis=0)
        cell_px = _mean_edge(q) / CELLS

    # covariance of the four corners from the four independent line fits
    Jc = np.zeros((8, 8))
    S = np.zeros((8, 8))
    for k in range(4):
        f = fits[k]
        S[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = [[f.var_d, f.cov_d_theta], [f.cov_d_theta, f.var_theta]]
    for k in range(4):
        kp = (k - 1) % 4
        J = corner_jacobian(fits[kp], fits[k], q[k])
        Jc[2 * k : 2 * k + 2, 2 * kp : 2 * kp + 2] = J[:, 0:2]
        Jc[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = J[:, 2:4]
    cov = Jc @ S @ Jc.T

    # straightness: compare a centre-third fit with the full line
    sag = 0.0
    for k in range(4):
        f = fits[k]
        p0, p1 = q[k], q[(k + 1) % 4]
        L = float(np.linalg.norm(p1 - p0))
        if L < 36.0:
            continue
        a = p0 + (p1 - p0) * (1.0 / 3.0)
        b = p0 + (p1 - p0) * (2.0 / 3.0)
        assert f.n is not None
        fm = fit_edge(
            gray, a, b, f.n, half_width=half_width, trim=0.5, camera=camera,
            fix_theta=True, fix_sigma=f.sigma_blur, level_slopes=False, passes=1, min_contrast=cfg.min_contrast,
        )
        if fm.ok and fm.mid is not None:
            sag = max(sag, abs(float(f.offset_at(fm.mid))))
    info = {"sagitta": sag, "half_width": half_width}
    return q, cov, fits, info


def _bias_from_interior(
    gray: np.ndarray,
    camera: CameraModel | None,
    q: np.ndarray,
    fits: list[EdgeFit],
    tag_id: int,
    cfg: MarkerConfig,
) -> dict[str, Any] | None:
    """Estimate the dark-side edge shift ``delta`` (px) from interior boundaries.

    Every boundary (border and interior) is observed displaced toward its dark
    side by ``delta``.  The border corners used to build the cell homography
    are themselves displaced, which shrinks the predicted grid about the tag
    centre by ``delta`` per half-size.  An interior boundary at signed distance
    ``rho`` (in half-sizes, measured from the centre along the boundary normal
    that points to its white side) is therefore *predicted* at
    ``true - rho * delta`` and *observed* at ``true - delta``: the measured
    offset along the white-pointing normal is ``m = -(1 - rho) * delta``.
    Because the pattern contains both polarities at many positions, a
    weighted robust fit of ``m`` against ``-(1 - rho)`` isolates ``delta``.

    Boundaries are split into two families (those parallel to the tag's top
    edge and those parallel to its left edge) because directional blur makes
    the shift direction dependent; with too little support in one family the
    pooled estimate is used for both.

    Returns ``{"delta": (d_h, d_v), "sigma": (s_h, s_v), "n": count}`` where
    index 0 applies to the top/bottom border lines and 1 to left/right.
    """
    cell_px = _mean_edge(q) / CELLS
    if cell_px < cfg.bias_min_cell_px:
        return None
    Hc = dlt_homography(_CELL_CORNERS, q)
    bits = tag_bits(tag_id)
    sig = float(np.median([f.sigma_blur for f in fits]))
    half_width = float(np.clip(0.42 * cell_px, 2.0, 9.0))
    half = CELLS / 2.0
    fam_rows: tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]] = ([], [])
    for p0c, p1c, nrm_c in _interior_runs(bits):
        Lc = float(np.linalg.norm(p1c - p0c))
        # stay 0.3 cell away from both ends (corners of neighbouring cells)
        a = p0c + (p1c - p0c) * (0.3 / Lc)
        b = p1c - (p1c - p0c) * (0.3 / Lc)
        pa, pb = apply_h(Hc, np.stack([a, b]))
        midc = 0.5 * (p0c + p1c)
        pm, pn = apply_h(Hc, np.stack([midc, midc + 0.25 * nrm_c]))
        f = fit_edge(
            gray, pa, pb, pn - pm, half_width=half_width, trim=0.0, camera=camera,
            fix_theta=True, fix_sigma=sig, level_slopes=False, passes=2, min_contrast=cfg.min_contrast,
        )
        if not f.ok or f.mid is None or f.n is None or f.inlier_frac < 0.5:
            continue
        m = float((f.mid - pm) @ f.n)
        if abs(m) > 0.45 * cell_px:
            continue
        rho = float((midc - np.array([half, half])) @ nrm_c) / half
        var = max(f.var_d, 1e-6) * cfg.covariance_inflation + 1e-4
        fam = 0 if abs(nrm_c[1]) > 0.5 else 1  # 0: boundary parallel to the top edge
        fam_rows[fam].append((-(1.0 - rho), m, var))

    def solve(rows: list[tuple[float, float, float]]) -> tuple[float, float] | None:
        if len(rows) < 5:
            return None
        A = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        w = 1.0 / np.array([r[2] for r in rows])
        delta = float(np.sum(w * A * y) / np.sum(w * A * A))
        wr = w
        for _ in range(3):
            res = y - A * delta
            s = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-6
            wr = w * np.where(np.abs(res) <= 2.5 * s, 1.0, 2.5 * s / np.maximum(np.abs(res), 1e-12))
            delta = float(np.sum(wr * A * y) / np.sum(wr * A * A))
        res = y - A * delta
        chi2 = float(np.sum(wr * res * res) / max(len(rows) - 1, 1))
        var = max(chi2, 1.0) / float(np.sum(wr * A * A))
        # The constant-shift model is itself approximate (halos and local tone
        # mapping are not a pure shift): keep a fraction of the correction as
        # uncertainty so a large correction is never reported as exact.
        var += (cfg.bias_model_fraction * delta) ** 2 + cfg.bias_floor_px**2
        return delta, float(np.sqrt(var))

    pooled = solve(fam_rows[0] + fam_rows[1])
    if pooled is None:
        return None
    sh = solve(fam_rows[0]) or pooled
    sv = solve(fam_rows[1]) or pooled
    return {
        "delta": (sh[0], sv[0]),
        "sigma": (sh[1], sv[1]),
        "pooled": pooled,
        "n": len(fam_rows[0]) + len(fam_rows[1]),
    }


def detect_tags(
    image: np.ndarray,
    *,
    camera: CameraModel | None = None,
    sizes: TagSizeTable | None = None,
    cfg: MarkerConfig | None = None,
    refine: bool = True,
) -> list[TagDetection]:
    """Detect and measure AprilTag 36h11 markers in an 8-bit image.

    Returns every decoded tag, accepted or not; rejected tags carry their
    reasons so the UI can tell the operator what to fix.
    """
    cfg = cfg or MarkerConfig()
    sizes = sizes or TagSizeTable()
    gray = to_gray(image)
    Himg, Wimg = gray.shape[:2]
    if camera is not None and (camera.width != Wimg or camera.height != Himg):
        raise ValueError(
            f"camera model is for {camera.width}x{camera.height} but the image is {Wimg}x{Himg}"
        )
    cands = _raw_detect(gray, cfg)
    counts: dict[int, int] = {}
    for tid, _ in cands:
        counts[tid] = counts.get(tid, 0) + 1

    out: list[TagDetection] = []
    for tid, q_raw in cands:
        size_mm, prov = sizes.size(tid)
        det = TagDetection(
            tag_id=tid,
            corners_px=q_raw.copy(),
            corners_ideal=(camera.undistort_pixels(q_raw) if camera is not None else q_raw.copy()),
            cov=np.eye(8) * 1.0,
            size_mm=size_mm,
            size_provenance=prov,
            edge_px=_edges(q_raw),
        )
        if not (0 <= tid < FAMILY_SIZE):  # pragma: no cover - dictionary guarantees range
            det.accepted = False
            det.reject_reasons.append("unknown_id")
        if counts[tid] > 1:
            det.accepted = False
            det.reject_reasons.append("duplicate_id")
        if det.min_edge_px < cfg.min_edge_px:
            det.accepted = False
            det.reject_reasons.append("too_small")
        if min(det.edge_px) / max(det.edge_px) < cfg.min_edge_ratio:
            det.accepted = False
            det.reject_reasons.append("too_oblique")
        # the quiet zone (about one cell) must be inside the image as well
        cell = det.mean_edge_px / CELLS
        margin = max(3.0, 0.9 * cell)
        if (
            q_raw[:, 0].min() < margin
            or q_raw[:, 1].min() < margin
            or q_raw[:, 0].max() > Wimg - 1 - margin
            or q_raw[:, 1].max() > Himg - 1 - margin
        ):
            det.accepted = False
            det.reject_reasons.append("clipped")

        if refine and "clipped" not in det.reject_reasons and "too_small" not in det.reject_reasons:
            r = _refine(gray, camera, q_raw, tid, cfg)
            if isinstance(r, str):
                det.accepted = False
                det.reject_reasons.append(r)
            else:
                q, cov, fits, info = r
                det.refined = True
                det.sigma_blur_px = float(np.median([f.sigma_blur for f in fits]))
                det.contrast = float(np.median([f.contrast for f in fits]))
                det.edge_rms = float(np.sqrt(np.mean([f.rms**2 for f in fits])))
                det.saturation = float(np.max([f.saturated_frac for f in fits]))
                det.max_sagitta_px = float(info["sagitta"])

                bias = _bias_from_interior(gray, camera, q, fits, tid, cfg) if cfg.bias_correction else None
                if bias is not None:
                    deltas, sdeltas = bias["delta"], bias["sigma"]
                    det.bias_px = float(bias["pooled"][0])
                    det.bias_sigma_px = float(bias["pooled"][1])
                    det.n_interior_edges = int(bias["n"])
                    if max(abs(deltas[0]), abs(deltas[1])) < 0.45 * cell:
                        # Border lines 0 (top) and 2 (bottom) belong to family 0,
                        # lines 1 (right) and 3 (left) to family 1.  Move each
                        # outward (toward white) by its family's delta and add the
                        # fully correlated uncertainty of that common shift.
                        for fam in (0, 1):
                            g = np.zeros(8)
                            for k in range(4):
                                n_prev, n_next = fits[(k - 1) % 4].n, fits[k].n
                                assert n_prev is not None and n_next is not None
                                rhs = np.array(
                                    [1.0 if ((k - 1) % 4) % 2 == fam else 0.0, 1.0 if k % 2 == fam else 0.0]
                                )
                                g[2 * k : 2 * k + 2] = np.linalg.solve(np.stack([n_prev, n_next]), rhs)
                            cov = cov + np.outer(g, g) * sdeltas[fam] ** 2 / max(cfg.covariance_inflation, 1e-9)
                        for k, f in enumerate(fits):
                            assert f.mid is not None and f.n is not None
                            f.mid = f.mid + deltas[k % 2] * f.n
                        qc = np.empty((4, 2))
                        for k in range(4):
                            x = intersect_lines(fits[(k - 1) % 4], fits[k])
                            assert x is not None
                            qc[k] = x
                        q = qc
                        det.bias_applied = True
                    else:
                        det.flags.append("bias_estimate_unreasonable")
                else:
                    det.flags.append("bias_not_estimated")

                cov = cov * cfg.covariance_inflation + np.eye(8) * cfg.sigma_floor_px**2
                det.corners_ideal = q
                det.corners_px = camera.distort_pixels(q) if camera is not None else q.copy()
                det.cov = cov
                det.edge_px = _edges(det.corners_px)

                # verify the decoded pattern through the refined homography
                Hc = dlt_homography(_CELL_CORNERS, q)
                vals = _sample_cells(gray, camera, Hc)
                black = float(np.median([f.black for f in fits]))
                white = float(np.median([f.white for f in fits]))
                thr = 0.5 * (black + white)
                expect = tag_bits(tid)
                got = vals > thr
                det.bit_errors = int(np.sum(got != expect))
                span = max(white - black, 1e-6)
                det.decode_margin = float(np.min(np.abs(vals - thr)) / (0.5 * span))
                if det.bit_errors > cfg.max_bit_errors:
                    det.accepted = False
                    det.reject_reasons.append("decode_mismatch")
                if det.max_sagitta_px > cfg.max_sagitta_px:
                    det.accepted = False
                    det.reject_reasons.append("bent_or_distorted")
                if det.saturation > cfg.max_edge_saturation:
                    # Clipped highlights move every edge toward its dark side.
                    # That is exactly the shift the interior-boundary estimate
                    # measures and removes; without that estimate the tag cannot
                    # be trusted for scale.
                    det.flags.append("highlights_clipped")
                    if not det.bias_applied:
                        det.accepted = False
                        det.reject_reasons.append("glare")
                if min(f.inlier_frac for f in fits) < cfg.min_edge_inliers:
                    det.accepted = False
                    det.reject_reasons.append("edge_disturbed")
                if det.min_edge_px < cfg.min_edge_px and "too_small" not in det.reject_reasons:
                    det.accepted = False
                    det.reject_reasons.append("too_small")
                if min(det.edge_px) / max(det.edge_px) < cfg.min_edge_ratio and "too_oblique" not in det.reject_reasons:
                    det.accepted = False
                    det.reject_reasons.append("too_oblique")
        elif refine:
            det.flags.append("not_refined")
        out.append(det)
    return out


def accepted_by_id(dets: list[TagDetection]) -> dict[int, TagDetection]:
    return {d.tag_id: d for d in dets if d.accepted}
