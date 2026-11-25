"""Experimental photometric (phone torch) analysis of a rounded panel edge (prompt 07).

RESEARCH EVIDENCE ONLY.  Nothing in this module produces guidance; every
result is labelled ``research_evidence_only`` and the planar workflow never
depends on it.  No CAD, no calibrated light, no calibrated radiometry.

What is measured
----------------
A phone sweep with the torch on moves the camera *and* the light together
(the LED sits a centimetre or two from the lens), so this is not fixed-camera
photometric stereo.  With tag-based camera poses every pixel is a known ray
in the frame of the panel face ("plane frame": x, y in the face, z out of
it, millimetres).  For a surface point ``P`` with normal ``N``::

    radiance = albedo * (ambient + S * (d_ref / d)^2 * beam(theta) * max(N . l, 0)) + gloss

On the flat face ``N`` is constant, so the torch shading is a smooth function
of the pose with a handful of nuisance parameters.  Past the tangent line the
normal turns with the arc (``phi = arc / R``): the measured brightness
departs from the flat prediction starting *at* the tangent line, and the rate
of departure encodes ``R``.  With the light next to the lens the shading
``N . l`` falls to zero exactly at the occluding contour, so a torch photo has
a soft, dark limb (the same effect that degrades the edge-only silhouette
method on torch photos).

Method
------
1. *Aligned observations.*  Cross-sections perpendicular to the hinted edge
   are laid out on the plane at many heights.  Every cell is a fixed plane
   point; each view samples its own pixels of that point through its pose, so
   observations of the flat face are aligned across views by construction.
   On the bend the pixel-to-surface correspondence depends on the shape, which
   is why the fit is model based (below).
2. *Normalised brightness and its gradients* (model free).  Each profile is
   divided by the flat-face prediction.  For a light at the lens this ratio is
   a function of the cross-section coordinate only, so its iso-lines are
   parallel to the cylinder axis in *every* view: the gradient field of the
   normalised brightness gives the axis direction, and the start of the
   plateau, walking in from the dark limb, gives a first guess of the
   transition.
3. *Joint model fit.*  Tangent line (offset, direction) and radius are fitted
   to all views together with nuisance parameters: per-view level (exposure
   is unknown and changes), radial beam shape, lens fall-off, ambient/torch
   ratio, torch position relative to the lens (weak prior), gloss lobe, an
   effective tone exponent (weak prior), optionally an albedo gradient, a
   beam tilt and first-harmonic ambient shading.  The loss is robust;
   saturated, dark, grazing, glare, excluded and outlying cells are masked
   and reported.
4. *Flash-off partners.*  A pair is used only if the two camera centres agree
   (poses are given) and the off photo is usable.  The browser gives no
   exposure lock, so the exposure ratio of a pair is unknown; a flash-off
   photo therefore never fixes the ambient *level* by itself.  Two
   formulations are available (``pair_mode``):

   * ``"difference"``: ``on = k * off + torch term``.  The off photo is the
     measured ambient image, whatever its shape (uneven room light, fixed
     shadows); the paint is still taken to be uniform.  The flash-off
     photos all see the same static scene, so their exposures are made
     comparable with each other from the brightness of the flat face; one
     shared ambient/torch scale then remains.
   * ``"ratio"``: ``on / off = k * (1 + torch / ambient)``.  Albedo cancels
     cell by cell (paint boundaries, texture); the ambient light is then
     taken to be uniform (or first-harmonic) over the bend, and ``k`` is one
     unknown per pair.
5. *Evidence for identifiability.*  Singular values and correlations of the
   Jacobian, three variance estimates (formal, row-block sandwich, delete-one
   view jackknife; the largest is reported), an allowance for brightness
   patterns fixed to the surface, a held-out-view prediction error and the
   position each view alone would give the transition (a geometric transition
   is stationary on the panel; a shadow or reflection moves with the phone).
   The status is decided from those numbers.

Shadows are never used as geometry: cast shadows move with the light and are
handled as outliers; the attached-shadow terminator is excluded by the
grazing/dark masks.

Not modelled (and not present in the synthetic photos either): inter-
reflection, structured torch beams, non-Lambertian paint, local tone mapping.
See docs/PHOTOMETRIC_ABLATION.md for what was measured and what was not.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.lines import Line2D
from ..geometry.se3 import Pose, rotation_angle
from .imaging import linear_luminance

EVIDENCE_LABEL = "research_evidence_only"

#: reference distance of the inverse-square term (absorbed by the per-view level)
_REF_MM = 500.0
#: the bend is modelled as a circular arc up to this angle; beyond it nothing is assumed
_PHI_CAP = np.deg2rad(100.0)

# shared parameter vector
(_I_D, _I_PSI, _I_LOGR, _I_M, _I_M2, _I_V1, _I_V2, _I_RHO, _I_OX, _I_OY, _I_BX, _I_BY, _I_KS, _I_LOGSH, _I_GAMMA, _I_A1, _I_AL1, _I_AL2) = range(18)
_N_SHARED = 18
_PARAM_NAMES = (
    "offset_mm", "angle_rad", "log_radius", "beam_exponent", "beam_shape2", "lens_falloff", "lens_falloff2", "ambient_ratio",
    "light_x_mm", "light_y_mm", "beam_tilt_x", "beam_tilt_y", "specular", "log_shininess", "tone_gamma",
    "albedo_gradient_per_mm", "ambient_cos", "ambient_sin",
)
_FD_STEP = np.array([0.02, 2e-5, 1e-3, 2e-3, 5e-3, 2e-3, 5e-3, 2e-4, 0.05, 0.05, 2e-4, 2e-4, 2e-4, 2e-3, 2e-4, 2e-6, 2e-4, 2e-4])
_FD_RELATIVE = (_I_M, _I_M2, _I_V1, _I_V2, _I_RHO, _I_KS)
#: observation types of a view
_ALONE, _RATIO, _DIFF = 0, 1, 2


# --------------------------------------------------------------------------
# Public data types
# --------------------------------------------------------------------------
@dataclass
class FlashView:
    """One photo of the sweep with its tag-based pose."""

    image_bgr: np.ndarray  # 8-bit sRGB photo, upright
    camera: CameraModel  # intrinsics (+ distortion) of this photo
    T_cam_plane: Pose  # plane frame -> camera frame
    torch: str  # "on" | "off" | "unknown"
    pair: int | None = None  # index of the flash-off/on partner, if the operator took a pair
    frame_id: str = ""
    #: lens blur (Gaussian sigma, px) if known, e.g. the median tag-edge blur of this photo
    blur_sigma_px: float | None = None


@dataclass
class EdgeHint:
    """Where to look (plane frame, mm): from the datum step or the operator."""

    point: tuple[float, float]  # a point near the expected tangent line
    direction: tuple[float, float]  # approximate direction of the edge
    inward: tuple[float, float]  # unit vector from the edge toward the flat panel
    radius_mm: float | None = None  # configured nominal radius, if the operation defines one
    #: usable stretch of the edge, measured from ``point`` along ``direction`` (mm)
    extent_mm: tuple[float, float] = (-200.0, 200.0)


@dataclass(frozen=True)
class PhotometricConfig:
    # ---- sampling ---------------------------------------------------------
    flat_extent_mm: float = 70.0  # sections reach this far onto the flat face
    outward_extent_mm: float = 110.0  # ... and this far beyond the hint, over the bend
    hint_tolerance_mm: float = 25.0  # the hint may be this far from the tangent line
    step_mm: float = 1.0  # sample spacing along a section
    section_spacing_mm: float = 25.0  # one section (height) every this many mm along the edge
    subsample_mm: float = 0.6
    max_sections: int = 40
    # ---- masks ------------------------------------------------------------
    saturation_code: int = 250  # 8-bit code at or above which a pixel is treated as clipped
    dark_linear: float = 0.008  # linear luminance below which a cell is too dark to trust
    min_view_cos: float = 0.40  # bend cells seen more obliquely than this are not used
    min_light_cos: float = 0.15  # ... nor cells this close to the terminator
    max_bend_deg: float = 70.0  # bend angle up to which the arc model is used
    glare_fraction: float = 0.10  # cells whose predicted gloss exceeds this share are not used
    exclude_margin_mm: float = 3.0
    outlier_z: float = 4.0
    outlier_min_rel: float = 0.04
    stationary_fraction: float = 0.6  # an outlier at the same surface point in this share of the views is "fixed"
    # ---- flash-off partners ------------------------------------------------
    pair_mode: str = "difference"  # "difference" | "ratio" (see the module docstring)
    pair_max_center_mm: float = 15.0
    pair_max_rotation_deg: float = 6.0
    pair_exposure_range: tuple[float, float] = (0.125, 8.0)
    pair_min_off_level: float = 0.03  # median linear luminance of the off photo on the face
    # ---- model ---------------------------------------------------------------
    radius_bounds_mm: tuple[float, float] = (3.0, 400.0)
    fit_beam_exponent: bool = True
    fit_beam_shape2: bool = True  # second-order term of the radial beam model
    fit_lens_falloff: bool = True  # radial fall-off of the lens (multiplies ambient and torch light)
    fit_lens_falloff2: bool = True
    lens_falloff_prior: float = 2.0
    fit_light_offset: bool = True  # torch position relative to the lens (camera x, y)
    light_offset_prior_mm: float = 20.0
    light_offset_max_mm: float = 60.0
    fit_beam_axis: bool = False  # tilt of the beam axis against the optical axis (not exercised by the synthetic photos)
    beam_axis_prior_rad: float = 0.15
    fit_specular: bool = True
    fit_tone_gamma: bool = True  # effective tone exponent: the browser gives no radiometric calibration
    tone_gamma_prior: float = 0.15
    fit_albedo_gradient: bool = False
    fit_ambient_harmonics: bool = False  # first-harmonic dependence of the ambient light on the bend angle
    ambient_harmonics_prior: float = 0.5
    default_blur_px: float = 0.7
    huber_k: float = 2.0
    sigma_rel: float = 0.01  # starting noise model; rescaled from the residuals
    sigma_abs: float = 0.002
    max_iterations: int = 30
    rounds: int = 3
    # ---- decisions ------------------------------------------------------------
    min_views: int = 2
    max_sigma_tangent_mm: float = 1.0
    max_rel_sigma_radius: float = 0.15
    max_condition: float = 3000.0
    max_heldout_ratio: float = 1.8
    max_heldout_rms_rel: float = 0.05
    max_stationarity_mm: float = 0.5
    max_surface_allowance_mm: float = 0.5
    max_material_fraction: float = 0.08
    min_flash_fraction: float = 0.15
    min_bend_coverage_deg: float = 30.0
    min_bend_cells: int = 60
    max_rms_rel: float = 0.06
    #: floors of the stated standard deviations.  The poses are taken as exact;
    #: the tangent floor covers the tag-pose error seen on synthetic sweeps.
    sigma_floor_tangent_mm: float = 0.10
    sigma_floor_angle_deg: float = 0.01
    sigma_floor_rel_radius: float = 0.003
    n_profile_sections: int = 3


@dataclass
class ViewDiagnostics:
    frame_id: str
    torch: str
    role: str = "unused"  # "torch" | "torch_minus_partner" | "torch_over_partner" | "partner" | "unused"
    used: bool = False
    reasons: list[str] = field(default_factory=list)
    pair_frame_id: str | None = None
    pair_center_distance_mm: float | None = None
    pair_rotation_deg: float | None = None
    pair_exposure_ratio: float | None = None  # fitted exposure of the on photo relative to the off photo
    gain: float | None = None
    camera_across_mm: float | None = None  # camera position across the edge (+ over the flat face)
    camera_height_mm: float | None = None
    view_angle_deg: float | None = None
    bend_coverage_deg: float | None = None
    flash_fraction: float | None = None  # torch share of the brightness of the flat face
    n_cells: int = 0
    n_used: int = 0
    mask_fractions: dict[str, float] = field(default_factory=dict)
    rms_rel: float | None = None
    heldout_rms_rel: float | None = None
    tangent_offset_mm: float | None = None  # where this view alone would move the transition

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id, "torch": self.torch, "role": self.role, "used": self.used, "reasons": list(self.reasons),
            "pair_frame_id": self.pair_frame_id, "pair_center_distance_mm": _r(self.pair_center_distance_mm, 2),
            "pair_rotation_deg": _r(self.pair_rotation_deg, 3), "pair_exposure_ratio": _r(self.pair_exposure_ratio, 4),
            "gain": _r(self.gain, 5), "camera_across_mm": _r(self.camera_across_mm, 1), "camera_height_mm": _r(self.camera_height_mm, 1),
            "view_angle_deg": _r(self.view_angle_deg, 2), "bend_coverage_deg": _r(self.bend_coverage_deg, 1),
            "flash_fraction": _r(self.flash_fraction, 3), "n_cells": self.n_cells, "n_used": self.n_used,
            "mask_fractions": {k: round(float(v), 4) for k, v in self.mask_fractions.items()},
            "rms_rel": _r(self.rms_rel, 5), "heldout_rms_rel": _r(self.heldout_rms_rel, 5), "tangent_offset_mm": _r(self.tangent_offset_mm, 3),
        }


@dataclass
class SectionProfile:
    """One cross-section for inspection: measured vs modelled normalised brightness."""

    frame_id: str
    height_mm: float  # position of the section along the edge (from the tangent point)
    e_mm: list[float]  # across the edge from the fitted tangent line (+ on the flat face)
    measured: list[float | None]  # measured brightness / predicted flat-face brightness
    model: list[float | None]
    used: list[bool]

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "height_mm": round(self.height_mm, 2), "e_mm": [round(x, 2) for x in self.e_mm],
                "measured": [_r(x, 4) for x in self.measured], "model": [_r(x, 4) for x in self.model], "used": list(self.used)}


@dataclass
class PhotometricResult:
    status: str  # "supported" | "needs_view" | "unsupported"
    reasons: list[str] = field(default_factory=list)  # machine readable
    suggestions: list[str] = field(default_factory=list)  # operator readable
    requests: list[dict[str, Any]] = field(default_factory=list)  # specific additional views
    #: tangent line in the plane frame; the normal points into the flat region
    tangent: Line2D | None = None
    tangent_point: np.ndarray | None = None  # point of the line at the middle of the supported stretch
    direction: np.ndarray | None = None  # unit direction of the tangent line = cylinder axis direction
    endpoints: np.ndarray | None = None
    sigma_tangent_mm: float | None = None  # at ``tangent_point``, perpendicular to the line
    sigma_angle_deg: float | None = None
    radius_mm: float | None = None
    sigma_radius_mm: float | None = None
    radius_source: str | None = None  # "fitted" | "hint_fixed"
    conditioning: dict[str, Any] = field(default_factory=dict)
    heldout: dict[str, Any] = field(default_factory=dict)
    nuisance: dict[str, Any] = field(default_factory=dict)
    gradient_axis: dict[str, Any] = field(default_factory=dict)
    views: list[ViewDiagnostics] = field(default_factory=list)
    profiles: list[SectionProfile] = field(default_factory=list)
    n_views_used: int = 0
    runtime_s: float = 0.0
    label: str = EVIDENCE_LABEL
    guidance_enabled: bool = False  # this module never enables guidance

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "guidance_enabled": False,
            "status": self.status,
            "reasons": list(self.reasons),
            "suggestions": list(self.suggestions),
            "requests": _jsonable(self.requests),
            "tangent": None if self.tangent is None else {"n": [float(x) for x in self.tangent.n], "d": float(self.tangent.d)},
            "tangent_point": None if self.tangent_point is None else [round(float(x), 4) for x in self.tangent_point],
            "axis_direction": None if self.direction is None else [round(float(x), 7) for x in self.direction],
            "endpoints": None if self.endpoints is None else np.asarray(self.endpoints).round(3).tolist(),
            "sigma_tangent_mm": _r(self.sigma_tangent_mm, 4),
            "sigma_angle_deg": _r(self.sigma_angle_deg, 4),
            "radius_mm": _r(self.radius_mm, 3),
            "sigma_radius_mm": _r(self.sigma_radius_mm, 3),
            "radius_source": self.radius_source,
            "conditioning": _jsonable(self.conditioning),
            "heldout": _jsonable(self.heldout),
            "nuisance": _jsonable(self.nuisance),
            "gradient_axis": _jsonable(self.gradient_axis),
            "views": [v.to_dict() for v in self.views],
            "profiles": [p.to_dict() for p in self.profiles],
            "n_views_used": self.n_views_used,
            "runtime_s": round(self.runtime_s, 3),
        }


def _r(x: Any, nd: int) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(np.round(o, 6)) if np.isfinite(o) else None
    return o


# --------------------------------------------------------------------------
# Section grid and sampling
# --------------------------------------------------------------------------
@dataclass
class _Grid:
    """Cross-sections on the plane: cell (i, j) is ``origin + e[j] n + w[i] t``."""

    origin: np.ndarray
    n: np.ndarray  # unit, toward the flat face
    t: np.ndarray  # unit, along the edge (n is 90 degrees counter-clockwise from t)
    e: np.ndarray
    w: np.ndarray
    step: float
    spacing: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.w.size, self.e.size

    def points(self) -> np.ndarray:
        return self.origin + self.e[None, :, None] * self.n + self.w[:, None, None] * self.t


def _hint_frame(hint: EdgeHint) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(hint.point, dtype=float).reshape(2)
    t = np.asarray(hint.direction, dtype=float).reshape(2)
    n = np.asarray(hint.inward, dtype=float).reshape(2)
    if not (np.all(np.isfinite(p)) and np.linalg.norm(t) > 1e-9 and np.linalg.norm(n) > 1e-9):
        raise ValueError("edge hint needs a finite point and non-zero direction / inward vectors")
    t = t / np.linalg.norm(t)
    n = n - (n @ t) * t
    if np.linalg.norm(n) < 0.2:
        raise ValueError("edge hint: 'inward' must not be parallel to 'direction'")
    n = n / np.linalg.norm(n)
    # keep the convention n = rot90ccw(t) (as Line2D); flip the direction if needed
    if float(-t[1] * n[0] + t[0] * n[1]) < 0:
        t = -t
    return p, t, n


@dataclass
class _Image:
    """Linear luminance and clipped-pixel map of the part of a photo that the sections cover."""

    lum: np.ndarray
    sat: np.ndarray
    u0: int
    v0: int


def _prepare_image(bgr: np.ndarray, sat_code: int, cam: CameraModel, T_cam_plane: Pose, grid: _Grid, pad: float = 40.0) -> _Image:
    H, W = bgr.shape[:2]
    q = grid.points()
    border = np.concatenate([q[0], q[-1], q[:, 0], q[:, -1]])
    lo = border - (0.5 * grid.spacing + pad) * np.abs(grid.t) - pad * np.abs(grid.n)
    hi = border + (0.5 * grid.spacing + pad) * np.abs(grid.t) + pad * np.abs(grid.n)
    X = np.concatenate([np.concatenate([border, lo, hi]), np.zeros((3 * border.shape[0], 1))], axis=1)
    uv, front = cam.project(T_cam_plane.apply(X))
    if np.any(front):
        uv = uv[front]
        u0 = int(np.clip(np.floor(uv[:, 0].min()) - 8, 0, W - 1))
        u1 = int(np.clip(np.ceil(uv[:, 0].max()) + 9, u0 + 1, W))
        v0 = int(np.clip(np.floor(uv[:, 1].min()) - 8, 0, H - 1))
        v1 = int(np.clip(np.ceil(uv[:, 1].max()) + 9, v0 + 1, H))
    else:
        u0, u1, v0, v1 = 0, W, 0, H
    roi = np.ascontiguousarray(bgr[v0:v1, u0:u1])
    lum = linear_luminance(roi)
    mx = roi.max(axis=2) if roi.ndim == 3 else roi
    sat = cv2.dilate((mx >= sat_code).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    return _Image(lum, sat, u0, v0)


def _sample(im: _Image, cam: CameraModel, T_cam_plane: Pose, grid: _Grid, cfg: PhotometricConfig) -> dict[str, np.ndarray]:
    """Mean linear luminance of every cell of ``grid`` in one photo.

    A cell averages bilinear samples over ``step x spacing`` millimetres of the
    plane (pixel centres are at integer coordinates, as everywhere in LUMEN).
    """
    nr, nc = grid.shape
    n_al = int(np.clip(np.ceil(grid.spacing / cfg.subsample_mm), 1, 64))
    al = ((np.arange(n_al) + 0.5) / n_al - 0.5) * grid.spacing
    # camera-frame coordinates of a plane point are affine in its (across, along) coordinates
    A = T_cam_plane.apply(np.append(grid.origin, 0.0))
    Bn = T_cam_plane.R @ np.append(grid.n, 0.0)
    Ct = T_cam_plane.R @ np.append(grid.t, 0.0)

    def pixels(e: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = A + e[None, :, None] * Bn + w[:, None, None] * Ct
        z = Xc[..., 2]
        front = z > 1e-9
        zs = np.where(front, z, 1.0)
        return cam.normalized_to_pixels(Xc[..., :2] / zs[..., None]), front

    acc = np.zeros((nr, nc))
    satmax = np.zeros((nr, nc), dtype=np.float32)
    inside = np.ones((nr, nc), dtype=bool)
    h_roi, w_roi = im.lum.shape[:2]
    w_sub = (al[:, None] + grid.w[None, :]).ravel()  # sub-rows, grouped by sub-sample
    for a in (-0.25 * grid.step, 0.25 * grid.step):
        uv, front = pixels(grid.e + a, w_sub)
        mx_ = uv[..., 0] - im.u0
        my_ = uv[..., 1] - im.v0
        ok = front & cam.in_image(uv, margin=1.0) & (mx_ >= 0) & (mx_ <= w_roi - 1) & (my_ >= 0) & (my_ <= h_roi - 1)
        mapx = np.ascontiguousarray(mx_, dtype=np.float32)
        mapy = np.ascontiguousarray(my_, dtype=np.float32)
        val = cv2.remap(im.lum, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sv = cv2.remap(im.sat, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        acc += val.reshape(n_al, nr, nc).mean(axis=0)
        satmax = np.maximum(satmax, sv.reshape(n_al, nr, nc).max(axis=0))
        inside &= ok.reshape(n_al, nr, nc).all(axis=0)
    # image scale across the edge at each cell (px per mm), for the blur term
    u0, _ = pixels(grid.e - 0.5, grid.w)
    u1, _ = pixels(grid.e + 0.5, grid.w)
    scale = np.linalg.norm(u1 - u0, axis=-1)
    return {"mean": acc / 2.0, "saturated": satmax > 0.0, "inside": inside, "px_per_mm": scale}


def _in_polygons(q: np.ndarray, polygons: Sequence[np.ndarray], margin: float) -> np.ndarray:
    """Cells whose centre lies in (or within ``margin`` of) any polygon; ``q`` is (..., 2)."""
    pts = q.reshape(-1, 2)
    out = np.zeros(pts.shape[0], dtype=bool)
    for poly in polygons:
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        if p.shape[0] < 3:
            continue
        lo, hi = p.min(axis=0) - margin, p.max(axis=0) + margin
        cand = np.nonzero(np.all((pts >= lo) & (pts <= hi), axis=1))[0]
        if cand.size == 0:
            continue
        x, y = pts[cand, 0], pts[cand, 1]
        ins = np.zeros(cand.size, dtype=bool)
        near = np.zeros(cand.size, dtype=bool)
        for k in range(p.shape[0]):
            a, b = p[k], p[(k + 1) % p.shape[0]]
            cross = (a[1] > y) != (b[1] > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            ins ^= cross & (x < xc)
            ab = b - a
            L2 = float(ab @ ab)
            s = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(L2, 1e-12), 0.0, 1.0)
            near |= np.hypot(x - (a[0] + s * ab[0]), y - (a[1] + s * ab[1])) <= margin
        out[cand] |= ins | near
    return out.reshape(q.shape[:-1])


def _denoise_ambient(B: np.ndarray, ok: np.ndarray, w: np.ndarray, step: float) -> np.ndarray:
    """Noise reduction of a flash-off image that serves as the measured ambient light.

    Its noise would act as noise in a regressor, which biases the ambient
    scale low and, through it, the tone exponent (errors in variables).  Two
    steps that keep what matters sharp:

    * along the edge the ambient light varies slowly: each column of the
      section grid (cells at the same distance from the edge) is replaced by a
      robust quadratic in the along-edge coordinate;
    * across the edge a running median over about 7 mm: for a monotone profile
      the median of a window is its centre value, so the start of the bend (a
      kink) and a fixed shadow edge (a step) stay where they are.
    """
    out = B.copy()
    wn = (w - w.mean()) / max(float(np.ptp(w)), 1e-9)
    for j in range(B.shape[1]):
        m = ok[:, j] & np.isfinite(B[:, j])
        if m.sum() < 6:
            continue
        A = np.stack([np.ones(int(m.sum())), wn[m], wn[m] ** 2], axis=1)
        y = B[m, j]
        wt = np.ones(y.size)
        coef = np.zeros(3)
        for _ in range(3):
            coef = np.linalg.lstsq(A * wt[:, None], y * wt, rcond=None)[0]
            r = y - A @ coef
            sc = 1.4826 * float(np.median(np.abs(r))) + 1e-9
            wt = np.sqrt(np.where(np.abs(r) <= 2.5 * sc, 1.0, 2.5 * sc / np.maximum(np.abs(r), 1e-12)))
        out[m, j] = A @ coef
    k = max(int(round(3.5 / max(step, 1e-9))), 1)
    pad = np.pad(out, ((0, 0), (k, k)), mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1, axis=1), axis=2)
    return np.where(ok, med, out)


# --------------------------------------------------------------------------
# Shading model
# --------------------------------------------------------------------------
@dataclass
class _Geom:
    """Cells and cameras expressed in the reference frame of the fit.

    The reference line passes through ``pivot`` with inward normal ``n``; the
    fitted line is that line rotated by ``psi`` about the pivot and shifted by
    ``d`` along its own normal.
    """

    pivot: np.ndarray
    n: np.ndarray
    t: np.ndarray
    shape: tuple[int, int]
    qe0: np.ndarray  # (M,)
    qw0: np.ndarray
    cam: np.ndarray  # (V, 3): camera centre (across, along, height)
    X: np.ndarray  # (V, 3) camera x axis in the same components
    Y: np.ndarray
    Z: np.ndarray  # optical axis
    mode: np.ndarray  # (V,) observation type: torch alone, on/off ratio, on with off as ambient image
    aref: np.ndarray  # (V, M) flash-off image of the "difference" views, scaled to 1 on the flat face
    blur: np.ndarray  # (V, nr, nc) weight of the second difference along a section

    @property
    def n_views(self) -> int:
        return int(self.cam.shape[0])


def _make_geom(grid: _Grid, poses: list[Pose], mode: Sequence[int], blur: np.ndarray, aref: np.ndarray | None = None) -> _Geom:
    q = grid.points().reshape(-1, 2)
    B = np.stack([np.append(grid.n, 0.0), np.append(grid.t, 0.0), np.array([0.0, 0.0, 1.0])])  # rows: across, along, up
    cams, Xs, Ys, Zs = [], [], [], []
    for T in poses:
        Tpc = T.inv()
        C = Tpc.t - np.append(grid.origin, 0.0)
        cams.append(B @ C)
        Xs.append(B @ Tpc.R[:, 0])
        Ys.append(B @ Tpc.R[:, 1])
        Zs.append(B @ Tpc.R[:, 2])
    V = len(poses)
    return _Geom(
        pivot=grid.origin.copy(), n=grid.n.copy(), t=grid.t.copy(), shape=grid.shape,
        qe0=(q - grid.origin) @ grid.n, qw0=(q - grid.origin) @ grid.t,
        cam=np.asarray(cams), X=np.asarray(Xs), Y=np.asarray(Ys), Z=np.asarray(Zs),
        mode=np.asarray(mode, dtype=np.int8), aref=np.ones((V, q.shape[0])) if aref is None else np.asarray(aref, dtype=float).reshape(V, -1), blur=blur,
    )


def _predict(th: np.ndarray, lin: np.ndarray, G: _Geom, *, flat_only: bool = False, aux: bool = False, blur: bool = True) -> Any:
    """Predicted observation of every cell in every view, shape ``(V, nr, nc)``.

    ``th`` are the shared parameters (``_PARAM_NAMES``), ``lin`` the per-view
    log gains.  With ``flat_only`` the surface is taken to be the plane
    everywhere (the "predicted flat" reference of a profile).
    """
    V = G.n_views
    nr, nc = G.shape
    R = float(np.exp(th[_I_LOGR]))
    cp, sp = np.cos(th[_I_PSI]), np.sin(th[_I_PSI])
    d0 = th[_I_D]
    eq = cp * G.qe0 - sp * G.qw0 - d0
    wq = cp * G.qw0 + sp * G.qe0

    def rot(a: np.ndarray) -> np.ndarray:
        return np.stack([cp * a[:, 0] - sp * a[:, 1], cp * a[:, 1] + sp * a[:, 0], a[:, 2]], axis=1)

    cam = rot(G.cam)
    ce = (cam[:, 0] - d0)[:, None]
    cw = cam[:, 1][:, None]
    ch = cam[:, 2][:, None]
    X, Y, Z = rot(G.X), rot(G.Y), rot(G.Z)

    # ---- where does the pixel ray of each cell meet the surface -----------
    de = eq[None, :] - ce
    if flat_only:
        tau = np.ones((V, eq.size))
        es = np.broadcast_to(eq[None, :], (V, eq.size))
        hs = np.zeros((V, eq.size))
        phi = np.zeros((V, eq.size))
        valid = np.ones((V, eq.size), dtype=bool)
    else:
        # cross-section: the bend is the circle of radius R centred at (0, -R), bend angle phi from the tangent point
        dh = -ch
        qa = de * de + dh * dh
        qb = 2.0 * (ce * de + (ch + R) * dh)
        qc = ce * ce + (ch + R) ** 2 - R * R
        disc = qb * qb - 4.0 * qa * qc
        ok = disc > 0.0
        tau_c = (-qb - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * qa)
        curved = np.broadcast_to((eq < 0.0)[None, :], (V, eq.size))
        tau = np.where(curved, tau_c, 1.0)
        es = ce + tau * de
        hs = np.where(curved, ch + tau * dh, 0.0)
        phi = np.where(curved, np.arctan2(-es, hs + R), 0.0)
        valid = ~curved | (ok & (phi >= 0.0) & (phi <= _PHI_CAP) & (tau > 0.0))
        phi = np.clip(phi, 0.0, _PHI_CAP)
    ws = cw + tau * (wq[None, :] - cw)
    sphi, cphi = np.sin(phi), np.cos(phi)  # surface normal = (-sin phi, 0, cos phi)

    # ---- torch: inverse square, incidence, radial beam -----------------------
    ox, oy = th[_I_OX], th[_I_OY]
    le = ce + ox * X[:, 0:1] + oy * Y[:, 0:1] - es
    lw = cw + ox * X[:, 1:2] + oy * Y[:, 1:2] - ws
    lh = ch + ox * X[:, 2:3] + oy * Y[:, 2:3] - hs
    dist = np.sqrt(le * le + lw * lw + lh * lh)
    le, lw, lh = le / dist, lw / dist, lh / dist
    ndl = -sphi * le + cphi * lh
    Bx = Z + th[_I_BX] * X + th[_I_BY] * Y
    Bx = Bx / np.linalg.norm(Bx, axis=1, keepdims=True)
    # log beam = m u + m2 u^2 with u = log cos(angle between the ray from the torch and the beam axis)
    lc = np.log(np.clip(-(le * Bx[:, 0:1] + lw * Bx[:, 1:2] + lh * Bx[:, 2:3]), 1e-4, None))
    beam = np.exp(th[_I_M] * lc + th[_I_M2] * lc * lc)
    fall = (_REF_MM / dist) ** 2
    E = fall * beam * np.clip(ndl, 0.0, None)

    ve, vw, vh = ce - es, cw - ws, ch - hs
    vn = np.sqrt(ve * ve + vw * vw + vh * vh)
    ve, vw, vh = ve / vn, vw / vn, vh / vn
    ndv = -sphi * ve + cphi * vh
    # lens fall-off acts on everything the camera sees (ambient and torch light alike):
    # log vig = v1 u + v2 u^2 with u = log cos(field angle of the viewing ray)
    if th[_I_V1] != 0.0 or th[_I_V2] != 0.0:
        lv_ = np.log(np.clip(-(ve * Z[:, 0:1] + vw * Z[:, 1:2] + vh * Z[:, 2:3]), 1e-4, None))
        vig = np.exp(th[_I_V1] * lv_ + th[_I_V2] * lv_ * lv_)
    else:
        vig = 1.0

    ks = th[_I_KS]
    if ks > 0.0:  # gloss: Blinn-Phong lobe about the half vector, not tinted by the paint
        hx, hy, hz = le + ve, lw + vw, lh + vh
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        ndh = np.clip((-sphi * hx + cphi * hz) / np.maximum(hn, 1e-12), 0.0, 1.0)
        spec = ks * fall * beam * ndh ** float(np.exp(th[_I_LOGSH])) * (ndl > 0.0)
    else:
        spec = np.zeros_like(E)

    amb = np.clip(th[_I_RHO] * (1.0 + th[_I_AL1] * (cphi - 1.0) + th[_I_AL2] * sphi), 0.0, None)
    if th[_I_A1] != 0.0:
        s_arc = np.where(phi > 0.0, -R * phi, np.clip(es, 0.0, None))
        alb = np.clip(1.0 + th[_I_A1] * s_arc, 0.05, None)
    else:
        alb = 1.0
    gam = float(th[_I_GAMMA])
    torch = (E * alb + spec) * vig
    mode = G.mode[:, None]
    f = (amb * alb) * vig + torch  # torch photo alone
    if np.any(G.mode == _RATIO):  # on / off: albedo and lens fall-off cancel
        f = np.where(mode == _RATIO, 1.0 + (E + spec) / np.maximum(amb, 1e-6), f)
    amb_meas = None
    if np.any(G.mode == _DIFF):  # on = scale * (measured ambient image) + torch term, in linear radiance
        aref = np.clip(G.aref, 0.0, None)
        amb_meas = max(float(th[_I_RHO]), 0.0) * (aref if gam == 1.0 else aref ** (1.0 / gam))
        f = np.where(mode == _DIFF, amb_meas + torch, f)
    if gam != 1.0:
        f = np.clip(f, 1e-9, None) ** gam
    y3 = (np.exp(lin)[:, None] * f).reshape(V, nr, nc)
    v3 = valid.reshape(V, nr, nc)
    if blur and nc >= 3:  # lens blur and cell footprint: second-order term along the section
        lap = np.zeros_like(y3)
        lap[:, :, 1:-1] = y3[:, :, :-2] + y3[:, :, 2:] - 2.0 * y3[:, :, 1:-1]
        okn = np.zeros_like(v3)
        okn[:, :, 1:-1] = v3[:, :, :-2] & v3[:, :, 2:] & v3[:, :, 1:-1]
        y3 = y3 + np.where(okn, G.blur * lap, 0.0)
    if not aux:
        return y3
    shape = (V, nr, nc)
    if amb_meas is None:
        denom, e_part, s_part = amb + E + spec, E, spec
    else:
        dm = mode == _DIFF
        denom = np.where(dm, amb_meas + torch, amb + E + spec)
        e_part = np.where(dm, E * alb * vig, E)
        s_part = np.where(dm, spec * vig, spec)
    denom = np.maximum(denom, 1e-12)
    return y3, {
        "valid": v3, "phi": phi.reshape(shape), "ndv": ndv.reshape(shape), "ndl": ndl.reshape(shape),
        "spec_frac": (s_part / denom).reshape(shape), "flash_frac": (e_part / denom).reshape(shape),
        "e": np.broadcast_to(eq.reshape(1, nr, nc), shape),
        "arc": np.where(phi > 0.0, -R * phi, np.broadcast_to(eq[None, :], phi.shape)).reshape(shape),
        "ce": ce[:, 0], "ch": ch[:, 0],
    }


# --------------------------------------------------------------------------
# Model-free stage: normalised brightness, its gradients, first transition guess
# --------------------------------------------------------------------------
def _robust_plane(e: np.ndarray, w: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Huber plane ``v ~ a + b e + c w`` (IRLS); returns the coefficients."""
    if v.size < 12:
        return None
    A = np.stack([np.ones_like(e), e - e.mean(), w - w.mean()], axis=1)
    wt = np.ones_like(v)
    coef = np.zeros(3)
    for _ in range(4):
        Aw = A * wt[:, None]
        try:
            coef = np.linalg.lstsq(Aw.T @ A, Aw.T @ v, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        r = v - A @ coef
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-9
        wt = np.where(np.abs(r) <= 2.0 * s, 1.0, 2.0 * s / np.maximum(np.abs(r), 1e-12))
    return np.array([coef[0] - coef[1] * e.mean() - coef[2] * w.mean(), coef[1], coef[2]])


def _flat_trend(base: np.ndarray, ok: np.ndarray, grid: _Grid, e_far: float) -> tuple[np.ndarray, float] | None:
    """Residual log-linear trend of ``observation / flat prediction`` where the
    face is surely flat, and the robust scatter about it."""
    E_, W_ = np.meshgrid(grid.e, grid.w)
    far = ok & (E_ >= e_far) & np.isfinite(base) & (base > 0)
    if far.sum() < 30:
        return None
    v = np.log(base[far])
    coef = _robust_plane(E_[far], W_[far], v)
    if coef is None:
        return None
    res = v - (coef[0] + coef[1] * E_[far] + coef[2] * W_[far])
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return np.exp(coef[0] + coef[1] * E_ + coef[2] * W_), mad


def _normalised_brightness(obs: np.ndarray, ok: np.ndarray, G: _Geom, grid: _Grid, e_far: float) -> tuple[list[np.ndarray | None], dict[str, float]]:
    """Brightness of every view divided by its predicted flat-face brightness.

    The flat prediction uses the torch model with a beam exponent and an
    ambient level picked from a small grid so that the far flat face is
    explained best; a per-view log-linear trend takes up what is left.
    """
    V = G.n_views
    best: tuple[float, float, float, list[Any]] | None = None
    th = np.zeros(_N_SHARED)
    th[_I_LOGR], th[_I_GAMMA], th[_I_LOGSH] = np.log(30.0), 1.0, np.log(40.0)
    zeros = np.zeros(V)
    for m in (0.5, 1.5, 3.0, 5.0):
        for rho in (0.02, 0.15, 0.5, 1.5, 4.0):
            th[_I_M], th[_I_RHO] = m, rho
            flat = _predict(th, zeros, G, flat_only=True, blur=False)
            score, trends = 0.0, []
            for v in range(V):
                base = np.where(ok[v], obs[v] / np.maximum(flat[v], 1e-12), np.nan)
                ft = _flat_trend(base, ok[v], grid, e_far)
                trends.append(None if ft is None else base / ft[0])
                score += 0.0 if ft is None else ft[1] ** 2
            if best is None or score < best[0]:
                best = (score, m, rho, trends)
    assert best is not None
    return best[3], {"beam_exponent": best[1], "ambient_ratio": best[2]}


def _gradient_axis(r: np.ndarray, grid: _Grid) -> dict[str, float] | None:
    """Tilt of the iso-brightness lines of the normalised brightness on the bend.

    ``r(e, w) = f(e - w tan(delta))`` gives ``dr/dw = -tan(delta) dr/de``:
    a (robust) regression of the along-edge gradient on the across-edge one.
    """
    nr, nc = grid.shape
    if nr < 3 or nc < 7:
        return None
    k = 2
    ge = np.full_like(r, np.nan)
    ge[:, k:-k] = (r[:, 2 * k:] - r[:, :-2 * k]) / (2 * k * grid.step)
    gw = np.full_like(r, np.nan)
    gw[1:-1] = (r[2:] - r[:-2]) / (grid.w[2:] - grid.w[:-2])[:, None]
    with np.errstate(invalid="ignore"):
        sel = np.isfinite(ge) & np.isfinite(gw) & (r > 0.3) & (r < 0.93) & (np.abs(ge) > 0.004)
    n = int(sel.sum())
    if n < 20:
        return None
    a, b = ge[sel], gw[sel]
    wt = np.ones(n)
    slope = 0.0
    for _ in range(4):
        slope = -float(np.sum(wt * a * b) / max(np.sum(wt * a * a), 1e-18))
        res = b + slope * a
        s = 1.4826 * float(np.median(np.abs(res))) + 1e-12
        wt = np.where(np.abs(res) <= 2.5 * s, 1.0, 2.5 * s / np.maximum(np.abs(res), 1e-18))
    res = b + slope * a
    sig = float(np.sqrt(np.sum(wt * res * res) / max(np.sum(wt) - 1.0, 1.0) / max(np.sum(wt * a * a), 1e-18)))
    return {"slope": slope, "sigma": sig, "n": n}


def _stacked_profile(r: np.ndarray, grid: _Grid, slope: float) -> np.ndarray:
    """Median profile over the sections after removing the tilt ``slope``."""
    nr, nc = grid.shape
    wm = float(np.mean(grid.w))
    out = np.full((nr, nc), np.nan)
    for i in range(nr):
        ok = np.isfinite(r[i])
        if ok.sum() < 8:
            continue
        shift = slope * (grid.w[i] - wm)
        out[i] = np.interp(grid.e + shift, grid.e[ok], r[i, ok], left=np.nan, right=np.nan)
        # do not interpolate across masked stretches
        gap = np.interp(grid.e + shift, grid.e, (~ok).astype(float), left=1.0, right=1.0)
        out[i, gap > 0.5] = np.nan
    cnt = np.sum(np.isfinite(out), axis=0)
    prof = np.full(nc, np.nan)
    use = cnt >= max(2, nr // 4)
    if use.any():
        prof[use] = np.nanmedian(out[:, use], axis=0)
    return prof


def _onset(prof: np.ndarray, grid: _Grid, e_max: float) -> dict[str, float] | None:
    """First guess of the transition on a stacked normalised-brightness profile.

    Walking from the darkest place (the limb of the bend, or whatever lies
    beyond it) toward the flat face, the transition is where the brightness
    stops changing: the start of the first plateau.  Walking in this direction
    a paint boundary further in on the flat face is never reached, and a
    plateau is a plateau whatever its level, so a different paint next to the
    edge does not matter here.  Only positions up to ``e_max`` (the hint
    tolerance) are considered.
    """
    e = grid.e
    ok = np.isfinite(prof)
    if ok.sum() < 20:
        return None
    win = max(int(round(9.0 / grid.step)), 5)  # slope window, about 9 mm
    idx = np.nonzero(ok & (e <= e_max + win * grid.step))[0]
    if idx.size < 2 * win:
        return None
    j_dark = int(idx[np.argmin(prof[idx])])
    slope = np.full(e.size, np.nan)
    level = np.full(e.size, np.nan)
    for j in range(j_dark, e.size - win):
        seg = slice(j, j + win)
        m = ok[seg]
        if m.sum() < win - 2:
            continue
        x, y = e[seg][m], prof[seg][m]
        xm = x - x.mean()
        slope[j] = float(np.sum(xm * (y - y.mean())) / max(np.sum(xm * xm), 1e-12))
        level[j] = float(np.median(y))
    far = ok & (e >= e_max)
    noise = 1.4826 * float(np.median(np.abs(prof[far] - np.median(prof[far])))) if far.sum() >= 6 else 0.005
    thr = max(0.0012, 3.0 * noise * np.sqrt(12.0) / (win ** 1.5 * grid.step))  # relative brightness per mm
    dark = float(prof[j_dark])
    for j in range(j_dark, e.size - win - 1):
        if e[j] > e_max:
            break
        if not (np.isfinite(slope[j]) and np.isfinite(slope[j + 1])):
            continue
        lv = level[j]
        if abs(slope[j]) < thr * lv and abs(slope[j + 1]) < thr * lv and lv > dark + 0.08 and lv > 0.3:
            return {"e": float(e[j]), "noise": noise, "depth": float(1.0 - dark / lv), "level": float(lv)}
    return None


# --------------------------------------------------------------------------
# Joint fit
# --------------------------------------------------------------------------
class _Fit:
    """State of the joint fit: cells, masks, parameters and the solver."""

    def __init__(self, G: _Geom, obs: np.ndarray, base_ok: np.ndarray, cfg: PhotometricConfig, hint_radius: float | None) -> None:
        self.G = G
        self.obs = obs  # (V, nr, nc)
        self.base_ok = base_ok
        self.cfg = cfg
        V = G.n_views
        model_ambient = bool(np.any(G.mode != _DIFF))  # some view relies on a model of the ambient shading
        # Lens fall-off can only be told from the torch beam where both the ambient and the torch
        # light are modelled (a torch photo alone).  In a ratio it cancels; in a difference view the
        # ambient part is measured, so the fall-off of the torch part is absorbed by the beam shape.
        sees_lens = bool(np.any(G.mode == _ALONE))
        self.any_ratio = bool(np.any(G.mode == _RATIO))
        self.th = np.zeros(_N_SHARED)
        self.th[_I_LOGR] = np.log(hint_radius if hint_radius else 30.0)
        self.th[_I_M] = 1.5
        self.th[_I_RHO] = 0.3 if self.any_ratio else 0.05
        self.th[_I_LOGSH] = np.log(40.0)
        self.th[_I_GAMMA] = 1.0
        self.lin = np.zeros(V)
        lo = np.full(_N_SHARED, -np.inf)
        hi = np.full(_N_SHARED, np.inf)
        lo[_I_D], hi[_I_D] = -60.0, 60.0
        lo[_I_PSI], hi[_I_PSI] = -0.35, 0.35
        lo[_I_LOGR], hi[_I_LOGR] = np.log(cfg.radius_bounds_mm[0] * 0.8), np.log(cfg.radius_bounds_mm[1] * 1.25)
        lo[_I_M], hi[_I_M] = -1.0, 12.0
        lo[_I_M2], hi[_I_M2] = -30.0, 30.0
        lo[_I_V1], hi[_I_V1] = -6.0, 12.0
        lo[_I_V2], hi[_I_V2] = -30.0, 30.0
        lo[_I_RHO], hi[_I_RHO] = (1e-3 if self.any_ratio else 0.0), 200.0
        lo[_I_OX], hi[_I_OX] = -cfg.light_offset_max_mm, cfg.light_offset_max_mm
        lo[_I_OY], hi[_I_OY] = -cfg.light_offset_max_mm, cfg.light_offset_max_mm
        lo[_I_BX], hi[_I_BX] = -0.5, 0.5
        lo[_I_BY], hi[_I_BY] = -0.5, 0.5
        lo[_I_KS], hi[_I_KS] = 0.0, 50.0
        lo[_I_LOGSH], hi[_I_LOGSH] = np.log(8.0), np.log(600.0)  # a broader lobe is not gloss but diffuse shading
        lo[_I_GAMMA], hi[_I_GAMMA] = 0.5, 2.0
        lo[_I_A1], hi[_I_A1] = -0.01, 0.01
        lo[_I_AL1], hi[_I_AL1] = -2.0, 2.0
        lo[_I_AL2], hi[_I_AL2] = -2.0, 2.0
        self.lo, self.hi = lo, hi
        free = np.zeros(_N_SHARED, dtype=bool)
        free[[_I_D, _I_PSI, _I_LOGR, _I_RHO]] = True
        free[_I_M] = cfg.fit_beam_exponent
        free[_I_M2] = cfg.fit_beam_exponent and cfg.fit_beam_shape2
        free[_I_V1] = cfg.fit_lens_falloff and sees_lens
        free[_I_V2] = free[_I_V1] and cfg.fit_lens_falloff2
        free[[_I_OX, _I_OY]] = cfg.fit_light_offset
        free[[_I_BX, _I_BY]] = cfg.fit_beam_axis
        free[[_I_KS, _I_LOGSH]] = cfg.fit_specular
        if cfg.fit_specular:
            self.th[_I_KS] = 0.02
        free[_I_GAMMA] = cfg.fit_tone_gamma
        free[_I_A1] = cfg.fit_albedo_gradient
        free[[_I_AL1, _I_AL2]] = cfg.fit_ambient_harmonics and model_ambient
        self.free = free
        # weak priors (mean, sigma) keep nuisance parameters finite where the photos do not determine them
        self.prior: dict[int, tuple[float, float]] = {
            _I_OX: (0.0, cfg.light_offset_prior_mm), _I_OY: (0.0, cfg.light_offset_prior_mm),
            _I_BX: (0.0, cfg.beam_axis_prior_rad), _I_BY: (0.0, cfg.beam_axis_prior_rad),
            _I_KS: (0.0, 0.5), _I_LOGSH: (np.log(40.0), 1.5), _I_GAMMA: (1.0, cfg.tone_gamma_prior),
            _I_M: (2.0, 4.0), _I_M2: (0.0, 6.0), _I_A1: (0.0, 0.003),
            _I_V1: (0.0, cfg.lens_falloff_prior), _I_V2: (0.0, 2.0 * cfg.lens_falloff_prior),
            _I_AL1: (0.0, cfg.ambient_harmonics_prior), _I_AL2: (0.0, cfg.ambient_harmonics_prior),
        }
        self.active = base_ok.copy()
        self.ref = base_ok.copy()  # cells that define the level of a view (fixed during a solve)
        self.sigma = np.ones_like(obs)
        self.noise_scale = 1.0
        self.flags: dict[str, np.ndarray] = {}
        self._cache: tuple[bytes, np.ndarray] | None = None
        self.nfev = 0

    # ---------------------------------------------------------------- helpers
    def predict(self, th: np.ndarray | None = None, lin: np.ndarray | None = None, **kw: Any) -> Any:
        self.nfev += 1
        return _predict(self.th if th is None else th, self.lin if lin is None else lin, self.G, **kw)

    def solve_gains(self, th: np.ndarray | None = None, mask: np.ndarray | None = None, robust: bool = False, f: np.ndarray | None = None) -> np.ndarray:
        """Per-view log gains in closed form (weighted least squares, or the median ratio).

        ``f`` may pass the unit-gain prediction for ``th``.
        """
        th = self.th if th is None else th
        V = self.G.n_views
        if f is None:
            f = self.predict(th, np.zeros(V))
        m = self.active if mask is None else mask
        lin = np.zeros(V)
        for v in range(V):
            mk = m[v] & np.isfinite(f[v]) & (f[v] > 0)
            if mk.sum() < 5:
                continue
            ob, fv = self.obs[v][mk], f[v][mk]
            if robust:
                g = float(np.median(ob / fv))
            else:
                w = 1.0 / np.maximum(self.sigma[v][mk], 1e-12) ** 2
                g = float(np.sum(w * ob * fv) / max(np.sum(w * fv * fv), 1e-300))
            lin[v] = np.log(max(g, 1e-12))
        return lin

    def select(self) -> dict[str, np.ndarray]:
        """Cells that the arc model may be asked to explain, for the current shape."""
        cfg = self.cfg
        _, a = self.predict(aux=True)
        bend = a["e"] < 0.0
        grazing = bend & a["valid"] & ((a["ndv"] < cfg.min_view_cos) | (a["ndl"] < cfg.min_light_cos))
        beyond = bend & (~a["valid"] | (a["phi"] > np.deg2rad(cfg.max_bend_deg)))
        glare = a["spec_frac"] > cfg.glare_fraction
        act = self.base_ok & ~grazing & ~beyond & ~glare
        for k in ("material", "shadow", "bright"):
            if k in self.flags:
                act &= ~self.flags[k]
        self.active = act
        self.flags.update(grazing=grazing & self.base_ok, beyond=beyond & self.base_ok & ~grazing, glare=glare & self.base_ok & ~grazing & ~beyond)
        return a

    def set_sigma(self) -> None:
        y = self.predict()
        self.sigma = self.noise_scale * np.sqrt(self.cfg.sigma_abs ** 2 + (self.cfg.sigma_rel * np.abs(np.where(np.isfinite(y), y, 0.0))) ** 2)

    # ------------------------------------------------------------ optimisation
    # Solver variables: [free shared parameters | level of each view].  The
    # *level* of a view is the log of its mean modelled brightness over a fixed
    # set of cells; with it a change of beam shape, ambient level or tone
    # exponent only changes the shape of a view's brightness field, not its
    # level, which removes the strong correlation between those parameters and
    # the gains (the marginal covariance of the shared parameters is
    # unaffected by this choice).
    def _free_shared(self) -> np.ndarray:
        return np.nonzero(self.free)[0]

    def variable_names(self) -> list[str]:
        return [_PARAM_NAMES[i] for i in self._free_shared()] + [f"level:{v}" for v in range(self.G.n_views)]

    def _level(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros(self.G.n_views)
        for v in range(self.G.n_views):
            m = self.ref[v] & np.isfinite(f[v])
            if m.any():
                out[v] = np.log(max(float(np.mean(f[v][m])), 1e-300))
        return out

    def _pack(self) -> np.ndarray:
        f = self.predict(self.th, np.zeros(self.G.n_views))
        return np.concatenate([self.th[self._free_shared()], self.lin + self._level(f)])

    def _eval(self, x: np.ndarray, th: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared parameters, gains and prediction for solver variables ``x``."""
        n = self._free_shared().size
        if th is None:
            th = self.th.copy()
            th[self._free_shared()] = x[:n]
        f = self.predict(th, np.zeros(self.G.n_views))
        lin = x[n:] - self._level(f)
        return th, lin, np.exp(lin)[:, None, None] * f

    def _residuals(self, th: np.ndarray, y: np.ndarray) -> np.ndarray:
        yy = np.where(np.isfinite(y), y, 0.0)
        z = ((self.obs - yy) / self.sigma)[self.active]
        pri = [(th[i] - mu) / sg for i, (mu, sg) in self.prior.items() if self.free[i]]
        return np.concatenate([z, np.asarray(pri, dtype=float)])

    def _fun(self, x: np.ndarray) -> np.ndarray:
        th, _lin, y = self._eval(x)
        self._cache = (x.tobytes(), y)
        return self._residuals(th, y)

    def _jac(self, x: np.ndarray, central: bool = False) -> np.ndarray:
        idx = self._free_shared()
        V = self.G.n_views
        th = self.th.copy()
        th[idx] = x[: idx.size]
        y0 = self._cache[1] if (self._cache is not None and self._cache[0] == x.tobytes()) else self._eval(x)[2]
        act = self.active
        n_act = int(act.sum())
        n_pri = sum(1 for i in self.prior if self.free[i])
        J = np.zeros((n_act + n_pri, x.size))
        inv_sig = (1.0 / self.sigma)[act]

        def at(y: np.ndarray) -> np.ndarray:
            return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)[act]

        y0a = at(y0)

        def shifted(i: int, h: float) -> np.ndarray:
            tp = th.copy()
            tp[i] = th[i] + h
            return at(self._eval(x, tp)[2])

        for k, i in enumerate(idx):  # shared parameters: finite differences
            h = _FD_STEP[i] * (1.0 + abs(th[i]) if i in _FD_RELATIVE else 1.0)
            if th[i] + h > self.hi[i]:
                h = -h
            yp = shifted(i, h)
            if central and self.lo[i] <= th[i] - h <= self.hi[i]:
                J[:n_act, k] = -(yp - shifted(i, -h)) / (2.0 * h) * inv_sig
            else:
                J[:n_act, k] = -(yp - y0a) / h * inv_sig
        view_of = np.broadcast_to(np.arange(V)[:, None, None], act.shape)[act]
        for v in range(V):  # levels: the prediction of a view is proportional to exp(level)
            J[:n_act, idx.size + v] = np.where(view_of == v, -y0a * inv_sig, 0.0)
        r = n_act
        for i, (_mu, sg) in self.prior.items():
            if self.free[i]:
                J[r, int(np.nonzero(idx == i)[0][0])] = 1.0 / sg
                r += 1
        return J

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        idx, V = self._free_shared(), self.G.n_views
        return np.concatenate([self.lo[idx], np.full(V, -60.0)]), np.concatenate([self.hi[idx], np.full(V, 60.0)])

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th, lin, _y = self._eval(x)
        return th, lin

    def run(self, max_iter: int | None = None) -> bool:
        """Bounded Levenberg-Marquardt with Huber weights; returns ``converged``."""
        self.ref = self.active.copy()
        lo, hi = self.bounds()
        x, ok = _bounded_lm(self._fun, self._jac, self._pack(), lo, hi, int(self.active.sum()), self.cfg.huber_k, max_iter or self.cfg.max_iterations)
        self.th, self.lin = self.unpack(x)
        return ok

    def run_staged(self, only: Sequence[int], cells: np.ndarray | None, max_iter: int) -> bool:
        """Optimise a subset of the free shared parameters (and all levels), optionally on a subset of cells."""
        keep = (self.free.copy(), self.active, self.ref)
        sub = np.zeros_like(self.free)
        sub[list(only)] = True
        self.free = self.free & sub
        if cells is not None:
            self.active = self.active & cells
        try:
            return self.run(max_iter)
        finally:
            self.free, self.active, self.ref = keep

    def without_view(self, v: int, max_iter: int) -> np.ndarray:
        """Shared parameters of a short refit that does not see view ``v``."""
        keep = (self.active, self.th.copy(), self.lin.copy(), self.ref)
        act = self.active.copy()
        act[v] = False
        self.active = act
        try:
            self.run(max_iter)
            return self.th.copy()
        finally:
            self.active, self.th, self.lin, self.ref = keep

    def crop(self, j0: int) -> None:
        """Drop the first ``j0`` columns of every section (cells far beyond the bend that are never used)."""
        G = self.G
        nr, nc = G.shape
        V = G.n_views

        def cut(a: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(a[..., j0:])

        self.G = _Geom(
            pivot=G.pivot, n=G.n, t=G.t, shape=(nr, nc - j0), qe0=cut(G.qe0.reshape(nr, nc)).ravel(), qw0=cut(G.qw0.reshape(nr, nc)).ravel(),
            cam=G.cam, X=G.X, Y=G.Y, Z=G.Z, mode=G.mode, aref=cut(G.aref.reshape(V, nr, nc)).reshape(V, -1), blur=cut(G.blur),
        )
        for name in ("obs", "base_ok", "active", "ref", "sigma"):
            setattr(self, name, cut(getattr(self, name)))
        self.flags = {k: cut(v) for k, v in self.flags.items()}
        self._cache = None

    def rescale_noise(self) -> None:
        """Set the noise scale from the robust scatter of the whitened residuals."""
        y = self.predict()
        z = ((self.obs - np.where(np.isfinite(y), y, 0.0)) / self.sigma)[self.active]
        if z.size < 30:
            return
        s = float(np.clip(1.4826 * np.median(np.abs(z - np.median(z))), 0.05, 50.0))
        self.noise_scale *= s
        self.sigma = self.sigma * s


def _huber_cost(r: np.ndarray, n_data: int, k: float) -> float:
    a = np.abs(r[:n_data])
    return float(np.sum(np.where(a <= k, a * a, 2.0 * k * a - k * k)) + np.sum(r[n_data:] ** 2))


def _bounded_lm(fun: Any, jac: Any, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_data: int, k: float, max_iter: int) -> tuple[np.ndarray, bool]:
    """Levenberg-Marquardt on the normal equations with box bounds and Huber weights.

    The first ``n_data`` residuals get Huber weights (IRLS); the rest are
    prior rows.  Parameters on a bound whose gradient points outward, and
    parameters without any data, are held for the iteration.  Dense and tiny:
    about twenty unknowns.
    """
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    r = fun(x)
    cost = _huber_cost(r, n_data, k)
    lam = 1e-3
    converged = False
    eps = 1e-9 * np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    for _ in range(max_iter):
        J = jac(x)
        a = np.abs(r[:n_data])
        w = np.concatenate([np.where(a <= k, 1.0, k / np.maximum(a, 1e-300)), np.ones(r.size - n_data)])
        Jw = J * w[:, None]
        A = J.T @ Jw
        g = Jw.T @ r
        dA = np.diag(A)
        held = ((x <= lo + eps) & (g > 0.0)) | ((x >= hi - eps) & (g < 0.0)) | (dA <= 1e-14 * max(float(dA.max()), 1e-300))
        f = np.nonzero(~held)[0]
        if f.size == 0:
            converged = True
            break
        Af, gf = A[np.ix_(f, f)], g[f]
        dg = np.clip(np.diag(Af), 1e-12, None)
        improved = False
        for _try in range(10):
            try:
                dx = -np.linalg.solve(Af + lam * np.diag(dg), gf)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = x.copy()
            xn[f] = np.clip(x[f] + dx, lo[f], hi[f])
            rn = fun(xn)
            cn = _huber_cost(rn, n_data, k)
            if np.isfinite(cn) and cn < cost:
                small = (cost - cn) <= 1e-6 * max(cost, 1e-300) or float(np.max(np.abs(xn - x) / (np.abs(x) + 1e-3))) < 1e-7
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-9)
                improved = True
                converged = small
                break
            lam *= 10.0
        if not improved:
            converged = True  # no descent direction left at this damping
            break
        if converged:
            break
    return x, converged


def _coarse_start(fit: _Fit, hint_radius: float | None) -> None:
    """Pick a starting radius / offset / ambient level by a small grid search.

    The gains are solved in closed form for each candidate; the score is a
    robust relative misfit in a window around the first transition guess
    (which is the origin of the section coordinate).
    """
    cfg = fit.cfg
    V = fit.G.n_views
    radii = [6.0, 10.0, 16.0, 25.0, 40.0, 65.0, 110.0, 180.0]
    if hint_radius:
        radii.append(float(hint_radius))
    radii = [r for r in radii if cfg.radius_bounds_mm[0] <= r <= cfg.radius_bounds_mm[1]]
    rhos = [0.3, 1.5] if fit.any_ratio else [0.0, 0.4]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    e = fit.G.qe0.reshape(fit.G.shape)[None]
    win = fit.base_ok & (e > -30.0) & (e < 25.0)
    for R in radii:
        for d in (-2.0, 4.0, 10.0):
            for rho in rhos:
                th = fit.th.copy()
                th[_I_LOGR], th[_I_D], th[_I_RHO] = np.log(R), d, rho
                f, a = fit.predict(th, np.zeros(V), aux=True, blur=False)
                usable = fit.base_ok & a["valid"] & (a["ndv"] > cfg.min_view_cos) & (a["phi"] < np.deg2rad(cfg.max_bend_deg))
                lin = fit.solve_gains(th, usable, robust=True, f=f)
                y = np.exp(lin)[:, None, None] * f
                rel = np.where(usable & np.isfinite(y), fit.obs / np.maximum(y, 1e-12) - 1.0, 0.3)
                score, cnt = 0.0, 0
                for v in range(V):
                    if usable[v].sum() < 20:
                        continue
                    rv = np.clip(np.abs(rel[v][win[v]]), 0.0, 0.3)
                    score += float(np.sum(rv ** 2))
                    cnt += int(rv.size)
                if cnt and (best is None or score / cnt < best[0]):
                    best = (score / cnt, th, lin)
    if best is not None:
        fit.th, fit.lin = best[1], best[2]


def _classify_outliers(fit: _Fit, aux: dict[str, np.ndarray]) -> None:
    """Split residual outliers into surface-fixed ones (material, fixed shadow,
    a shape that is not a flat face joined to a circular bend) and
    view-dependent ones (cast torch shadow, occluder, reflection).

    The same surface point is the same grid cell on the flat face; on the bend
    it is the same (section, arc length) bin under the current shape.
    Recomputed from scratch on every call.
    """
    cfg = fit.cfg
    V = fit.G.n_views
    nr, nc = fit.G.shape
    y = fit.predict()
    yy = np.where(np.isfinite(y), y, 0.0)
    cand = fit.base_ok & np.isfinite(y) & ~fit.flags["grazing"] & ~fit.flags["beyond"] & ~fit.flags["glare"]
    z = np.where(cand, (fit.obs - yy) / fit.sigma, 0.0)
    rel = np.where(cand, fit.obs / np.maximum(yy, 1e-12) - 1.0, 0.0)
    out = cand & (np.abs(z) > cfg.outlier_z) & (np.abs(rel) > cfg.outlier_min_rel)
    ker = np.ones((1, 3), np.uint8)
    neg = np.stack([cv2.dilate(m.astype(np.uint8), ker) > 0 for m in out & (z < 0)]) & fit.base_ok
    pos = np.stack([cv2.dilate(m.astype(np.uint8), ker) > 0 for m in out & (z > 0)]) & fit.base_ok
    material = np.zeros_like(out)
    if V >= 2:
        arc = aux["arc"]
        lo = float(np.nanmin(arc))
        nb = int(np.ceil((float(np.nanmax(arc)) - lo) / 2.0)) + 1
        b = np.clip(((arc - lo) / 2.0).astype(np.intp), 0, nb - 1)  # 2 mm bins of arc length
        rows = np.broadcast_to(np.arange(nr)[None, :, None], (V, nr, nc))
        seen = np.zeros((V, nr, nb), dtype=bool)
        hit_n = np.zeros((V, nr, nb), dtype=bool)
        hit_p = np.zeros((V, nr, nb), dtype=bool)
        for v in range(V):
            seen[v][rows[v][cand[v]], b[v][cand[v]]] = True
            hit_n[v][rows[v][neg[v]], b[v][neg[v]]] = True
            hit_p[v][rows[v][pos[v]], b[v][pos[v]]] = True
        n_seen = seen.sum(axis=0)
        need = np.maximum(2, np.ceil(cfg.stationary_fraction * n_seen))
        stat = ((hit_n.sum(axis=0) >= need) | (hit_p.sum(axis=0) >= need)) & (n_seen >= 2)
        for v in range(V):
            material[v] = stat[rows[v], b[v]] & fit.base_ok[v]
    fit.flags["material"] = material
    fit.flags["shadow"] = neg & ~material
    fit.flags["bright"] = pos & ~material


# --------------------------------------------------------------------------
# Statistics: conditioning, variances, held-out views, stationarity
# --------------------------------------------------------------------------
def _covariances(fit: _Fit) -> dict[str, Any]:
    """Jacobian-based conditioning and two covariance estimates of the solver variables."""
    cfg = fit.cfg
    fit.ref = fit.active.copy()
    x = fit._pack()
    idx = fit._free_shared()
    J = fit._jac(x, central=True)
    th, _lin, y = fit._eval(x)
    r = fit._residuals(th, y)
    n_act = int(fit.active.sum())
    a = np.abs(r[:n_act])  # Huber weights of the data rows
    wt = np.concatenate([np.where(a <= cfg.huber_k, 1.0, cfg.huber_k / np.maximum(a, 1e-300)), np.ones(r.size - n_act)])
    # variables sitting on a bound carry no first-order information
    lo, hi = fit.bounds()
    span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    at_bound = (x - lo < 1e-6 * span) | (hi - x < 1e-6 * span)
    if fit.th[_I_KS] <= 1e-5 and fit.free[_I_KS]:
        at_bound[: idx.size] |= np.isin(idx, [_I_KS, _I_LOGSH])
    keep = np.nonzero(~at_bound)[0]
    Jk = J[:, keep] * np.sqrt(wt)[:, None]
    rk = r * np.sqrt(wt)
    A = Jk.T @ Jk
    norms = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    # singular values of the column-scaled Jacobian = sqrt(eigenvalues of the scaled normal matrix)
    sv = np.sqrt(np.clip(np.linalg.eigvalsh(A / np.outer(norms, norms)), 0.0, None))[::-1]
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A)
    chi2 = float(np.sum(rk[:n_act] ** 2) / max(n_act - keep.size, 1))
    C_formal = Ainv * max(chi2, 1.0)
    V = fit.G.n_views
    nr, nc = fit.G.shape
    view_of = np.broadcast_to(np.arange(V)[:, None, None], fit.active.shape)[fit.active]
    row_of = np.broadcast_to(np.arange(nr)[None, :, None], fit.active.shape)[fit.active]
    # sandwich estimate with blocks of neighbouring sections as clusters: residuals that are
    # correlated along a section or between views of the same piece of surface (texture) count once
    C_rows = None
    if nr >= 3:
        n_blocks = int(np.clip(nr // 2, 3, 8))
        labels = (row_of * n_blocks) // nr
        ids = np.unique(labels)
        if ids.size >= 3:
            meat = np.zeros_like(A)
            for c in ids:
                g = Jk[:n_act][labels == c].T @ rk[:n_act][labels == c]
                meat += np.outer(g, g)
            C_rows = Ainv @ meat @ Ainv * (ids.size / (ids.size - 1.0))

    def sig(C: np.ndarray | None) -> np.ndarray:
        return np.full(keep.size, np.nan) if C is None else np.sqrt(np.clip(np.diag(C), 0.0, None))

    names = fit.variable_names()
    dstd = np.sqrt(np.clip(np.diag(C_formal), 1e-300, None))
    return {
        "keep": keep, "names": [names[k] for k in keep], "all_names": names, "J": Jk, "r": rk, "n_act": n_act, "A": A, "chi2": chi2, "sv": sv,
        "sigma_formal": sig(C_formal), "sigma_rows": sig(C_rows), "corr": C_formal / np.outer(dstd, dstd),
        "view_of": view_of, "n_shared": idx.size, "at_bound": [names[k] for k in np.nonzero(at_bound)[0]],
    }


def _leave_one_out(fit: _Fit, cov: dict[str, Any]) -> list[np.ndarray | None]:
    """Shared parameters of solutions that did not see one view each.

    With few views, or when the problem is not well conditioned, the problem
    is re-solved (a few Levenberg-Marquardt iterations from the joint
    optimum); otherwise one Gauss-Newton step from the joint optimum is used.
    """
    V = fit.G.n_views
    out: list[np.ndarray | None] = [None] * V
    if V < 2:
        return out
    keep, A, names = cov["keep"], cov["A"], cov["names"]
    Jk, rk, n_act, view_of = cov["J"], cov["r"], cov["n_act"], cov["view_of"]
    lo, hi = fit.bounds()
    refit = V <= 4 or cov["sv"][0] / max(cov["sv"][-1], 1e-300) > 1500.0
    for v in range(V):
        m = view_of == v
        if m.sum() < 10:
            continue
        if refit:
            out[v] = fit.without_view(v, 4)
            continue
        cols = np.array([c for c in range(keep.size) if names[c] != f"level:{v}"])
        Jv, rv = Jk[:n_act][m], rk[:n_act][m]
        Av = A[np.ix_(cols, cols)] - Jv[:, cols].T @ Jv[:, cols]
        try:
            step = np.linalg.solve(Av, Jv[:, cols].T @ rv)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Av, Jv[:, cols].T @ rv, rcond=None)[0]
        x = fit._pack()
        x[keep[cols]] += step
        out[v] = fit.unpack(np.clip(x, lo, hi))[0]
    return out


def _heldout(fit: _Fit, loo: list[np.ndarray | None]) -> dict[str, Any]:
    """Predict each view from a solution that did not see it; the held-out view
    only gets its own gain."""
    V = fit.G.n_views
    if V < 2:
        return {"available": False, "note": "one view: nothing to hold out"}
    per: list[dict[str, float]] = []
    y_in = fit.predict()
    for v in range(V):
        if loo[v] is None:
            continue
        th = loo[v]
        f = fit.predict(th, np.zeros(V))
        y_out = (np.exp(fit.solve_gains(th, f=f))[:, None, None] * f)[v]
        mk = fit.active[v] & np.isfinite(y_out) & (y_out > 0)
        if mk.sum() < 10:
            continue
        rel_out = fit.obs[v][mk] / np.maximum(y_out[mk], 1e-12) - 1.0
        rel_in = fit.obs[v][mk] / np.maximum(y_in[v][mk], 1e-12) - 1.0
        rms_out = float(np.sqrt(np.mean(np.clip(rel_out, -0.5, 0.5) ** 2)))
        rms_in = float(np.sqrt(np.mean(np.clip(rel_in, -0.5, 0.5) ** 2)))
        per.append({"view": v, "heldout_rms_rel": rms_out, "in_sample_rms_rel": rms_in, "ratio": rms_out / max(rms_in, 1e-4),
                    "shift_offset_mm": float(th[_I_D] - fit.th[_I_D]), "shift_radius_rel": float(th[_I_LOGR] - fit.th[_I_LOGR])})
    if not per:
        return {"available": False, "note": "no view had enough cells to hold out"}
    return {
        "available": True,
        "method": "leave-one-view-out (re-solved for up to four views or a poorly conditioned problem, otherwise one Gauss-Newton step from the joint solution); the held-out view only gets its own gain",
        "per_view": per, "max_ratio": float(max(p["ratio"] for p in per)), "median_ratio": float(np.median([p["ratio"] for p in per])),
        "max_heldout_rms_rel": float(max(p["heldout_rms_rel"] for p in per)),
        "max_offset_shift_mm": float(max(abs(p["shift_offset_mm"]) for p in per)),
    }


def _jackknife(loo: list[np.ndarray | None]) -> dict[str, float]:
    """Delete-one-view jackknife standard deviations of the three shape parameters."""
    sols = [s_ for s_ in loo if s_ is not None]
    n = len(sols)
    if n < 3:
        return {}
    D = np.asarray(sols)[:, [_I_D, _I_PSI, _I_LOGR]]
    D = D - D.mean(axis=0)
    sd = np.sqrt((n - 1.0) / n * np.sum(D * D, axis=0))
    return {"offset_mm": float(sd[0]), "angle_rad": float(sd[1]), "log_radius": float(sd[2])}


def _per_view_offsets(fit: _Fit, cov: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tangent-line offset each view would prefer on its own, and its standard
    deviation (one Gauss-Newton step with only the offset and that view's level free).

    A geometric transition is the same line on the panel in every view; the
    edge of a cast shadow or a reflection is not.
    """
    names = cov["names"]
    V = fit.G.n_views
    out = np.full(V, np.nan)
    sig = np.full(V, np.nan)
    if "offset_mm" not in names:
        return out, sig
    kd = names.index("offset_mm")
    Jk, rk, n_act, view_of = cov["J"], cov["r"], cov["n_act"], cov["view_of"]
    for v in range(V):
        m = view_of == v
        if m.sum() < 10 or f"level:{v}" not in names:
            continue
        Jv = Jk[:n_act][m][:, [kd, names.index(f"level:{v}")]]
        try:
            Ninv = np.linalg.inv(Jv.T @ Jv)
        except np.linalg.LinAlgError:
            continue
        o = float(-(Ninv @ (Jv.T @ rk[:n_act][m]))[0])
        if abs(o) < 1e3:
            out[v] = o
            sig[v] = float(np.sqrt(max(Ninv[0, 0], 0.0) * max(cov["chi2"], 1.0)))
    return out, sig


def _fixed_pattern(fit: _Fit, aux: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Brightness pattern that stays on the flat face in every view, and what it
    could do to the transition.

    The model takes the paint to be uniform (except for ratio views, where the
    albedo cancels).  Averaging the relative residual of the other views cell
    by cell keeps what is fixed to the surface (texture, smooth albedo
    changes, a faint line).  Two parts matter:

    * a *coherent* part, the same in every section (a band parallel to the
      edge): the largest section-averaged anomaly of a 15 mm band within
      about 30 mm of the transition;
    * an *incoherent* part: how well the mean of such a band is known given
      its section-to-section scatter (texture averages down with the number
      of sections; a pattern that is smooth along the edge does not).

    An unexplained brightness offset ``A`` next to the bend moves the apparent
    start of the bend by about ``A / slope`` with ``slope`` the relative
    brightness slope on the first part of the bend.  That allowance is added
    to the stated uncertainty: a physically motivated bound, not a fit.
    """
    nr, nc = fit.G.shape
    un = fit.G.mode != _RATIO
    out = {"amplitude": 0.0, "coherent": 0.0, "incoherent": 0.0, "slope_per_mm": float("nan"), "offset_mm": 0.0}
    yy = np.where(np.isfinite(y) & (y > 0), y, np.nan)
    step = max(float(fit.cfg.step_mm), 1e-9)
    with np.errstate(all="ignore"):
        ld = np.abs(np.diff(np.log(yy), axis=2)) / step
        ok = fit.active[:, :, 1:] & fit.active[:, :, :-1] & (aux["e"][:, :, 1:] < 0.0) & (aux["phi"][:, :, :-1] < np.deg2rad(25.0))
        if ok.sum() >= 10:
            out["slope_per_mm"] = float(np.median(ld[ok]))
    if not np.any(un):
        return out
    flat = fit.active & (aux["e"] > 2.0) & un[:, None, None]
    rel = np.where(flat, fit.obs / np.where(np.isfinite(yy), yy, 1.0) - 1.0, np.nan)
    cnt = np.sum(np.isfinite(rel), axis=0)
    tot = np.nansum(rel, axis=0)
    mean = np.where(cnt >= max(1, int(np.ceil(0.5 * un.sum()))), tot / np.maximum(cnt, 1), np.nan)
    width = max(int(round(15.0 / step)), 4)
    e_grid = fit.G.qe0.reshape(nr, nc)[0] - fit.th[_I_D]
    sig_cell = fit.noise_scale * fit.cfg.sigma_rel
    bands: list[tuple[float, float, bool, float]] = []  # (mean anomaly, its noise variance, near the transition, std of that mean)
    for j0 in range(0, nc - width + 1, max(width // 2, 1)):
        blk = mean[:, j0:j0 + width]
        fin = np.isfinite(blk)
        k = fin.sum(axis=1)
        good = k >= max(width // 2, 2)
        if good.sum() < max(3, nr // 4):
            continue
        m = np.where(fin[good], blk[good], 0.0).sum(axis=1) / k[good]
        n_obs = float(np.sum(np.where(fin[good], cnt[good, j0:j0 + width], 0)))
        # how well is the band mean known?  Group neighbouring sections into blocks of growing
        # length: for texture the answer does not depend on the block length, for a pattern
        # that is smooth along the edge it grows until the blocks are longer than the pattern.
        u = 0.0
        for L in (1, 2, 4, 8):
            nb = m.size // L
            if nb < 3:
                break
            bm = m[: nb * L].reshape(nb, L).mean(axis=1)
            u = max(u, float(np.std(bm, ddof=1) / np.sqrt(nb)))
        bands.append((float(np.mean(m)), sig_cell ** 2 / max(n_obs, 1.0), bool(e_grid[j0] < 30.0), u))
    if len(bands) < 2:
        return out
    c = np.array([b_[0] for b_ in bands])
    c = c - float(np.median(c))
    vn = np.array([b_[1] for b_ in bands])
    near = np.array([b_[2] for b_ in bands])
    u_ = np.array([b_[3] for b_ in bands])
    sel = near if near.any() else np.ones_like(near)
    coh = float(np.max(np.sqrt(np.clip(c[sel] ** 2 - 6.0 * vn[sel], 0.0, None))))
    inc = float(np.sqrt(np.mean(np.clip(u_[sel] ** 2 - vn[sel], 0.0, None))))
    out.update(coherent=coh, incoherent=inc, amplitude=float(np.hypot(coh, inc)))
    if np.isfinite(out["slope_per_mm"]) and out["slope_per_mm"] > 1e-5:
        out["offset_mm"] = out["amplitude"] / out["slope_per_mm"]
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def analyse_flash_sweep(
    views: list[FlashView],
    hint: EdgeHint,
    cfg: PhotometricConfig | None = None,
    exclude_polygons: Sequence[np.ndarray] = (),
) -> PhotometricResult:
    """Estimate the tangent line, axis direction and radius of a rounded edge
    from torch-on photos with known poses, or say why that is not possible.

    ``exclude_polygons`` are plane-frame polygons (tags, placards, anything
    that is not bare panel).  The result is research evidence only.
    """
    t_start = time.perf_counter()
    cfg = cfg or PhotometricConfig()
    if cfg.pair_mode not in ("difference", "ratio"):
        raise ValueError("pair_mode must be 'difference' or 'ratio'")
    res = PhotometricResult(status="unsupported")
    diags = [ViewDiagnostics(frame_id=v.frame_id or f"view{i}", torch=str(v.torch)) for i, v in enumerate(views)]
    res.views = diags

    def finish(status: str, reason: str | None = None, suggestion: str | None = None, request: dict[str, Any] | None = None) -> PhotometricResult:
        res.status = status
        if reason and reason not in res.reasons:
            res.reasons.insert(0, reason)
        if suggestion and suggestion not in res.suggestions:
            res.suggestions.insert(0, suggestion)
        if request:
            res.requests.insert(0, request)
        res.runtime_s = time.perf_counter() - t_start
        return res

    def note(reason: str, suggestion: str | None = None, request: dict[str, Any] | None = None) -> None:
        if reason not in res.reasons:
            res.reasons.append(reason)
            if suggestion:
                res.suggestions.append(suggestion)
            if request:
                res.requests.append(request)

    p_h, t_h, n_h = _hint_frame(hint)

    # ---- which photos can be used, and how -------------------------------
    on = [i for i, v in enumerate(views) if v.torch == "on"]
    for i, v in enumerate(views):
        if v.torch not in ("on", "off"):
            diags[i].reasons.append("torch_state_unknown")
    if any(v.torch not in ("on", "off") for v in views):
        note("torch_state_unknown", "The torch state of some photos is not known: mark each photo as torch on or torch off; photos of unknown state are not used.")
    if not on:
        return finish(
            "needs_view", "no_torch_on_views",
            "Photometric evidence needs torch-on photos: switch the torch on and take 3 to 5 photos while moving sideways across the rounded edge. The planar workflow does not need this.",
            {"code": "torch_on_sweep", "torch": "on", "n_photos": 3},
        )
    partner: dict[int, int] = {}
    for i in on:
        j = views[i].pair
        if j is None:
            j = next((k for k, o in enumerate(views) if o.torch == "off" and o.pair == i), None)
        if j is None or not (0 <= j < len(views)) or views[j].torch != "off":
            continue
        dist = float(np.linalg.norm(views[i].T_cam_plane.center - views[j].T_cam_plane.center))
        ang = float(np.degrees(rotation_angle(views[i].T_cam_plane.R @ views[j].T_cam_plane.R.T)))
        diags[i].pair_frame_id = diags[j].frame_id
        diags[i].pair_center_distance_mm, diags[i].pair_rotation_deg = dist, ang
        if dist > cfg.pair_max_center_mm or ang > cfg.pair_max_rotation_deg:
            diags[i].reasons.append("pair_misaligned")
            diags[j].reasons.append("pair_misaligned")
            continue
        partner[i] = j

    # ---- pass 1: sections on the hinted line ------------------------------
    w0, w1 = float(min(hint.extent_mm)), float(max(hint.extent_mm))
    n_sec = int(np.clip(np.floor((w1 - w0) / cfg.section_spacing_mm), 1, cfg.max_sections))
    spacing = (w1 - w0) / n_sec
    step1 = max(cfg.step_mm, 1.5)
    outward = max(cfg.outward_extent_mm, cfg.hint_tolerance_mm + 1.3 * float(hint.radius_mm or 0.0))
    e1 = np.arange(-outward, cfg.flat_extent_mm + cfg.hint_tolerance_mm + 1e-9, step1)
    grid1 = _Grid(p_h, n_h, t_h, e1, w0 + (np.arange(n_sec) + 0.5) * spacing, step1, spacing)
    excl1 = _in_polygons(grid1.points(), exclude_polygons, cfg.exclude_margin_mm)
    images: dict[int, _Image] = {}

    def img(i: int) -> _Image:
        if i not in images:  # one crop per photo, large enough for both passes
            images[i] = _prepare_image(views[i].image_bgr, cfg.saturation_code, views[i].camera, views[i].T_cam_plane, grid1)
        return images[i]

    def observe(i: int, grid: _Grid, excl: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int, np.ndarray]:
        """Observation of view ``i`` on ``grid``: the torch photo, with its flash-off partner if that is usable."""
        s_on = _sample(img(i), views[i].camera, views[i].T_cam_plane, grid, cfg)
        masks = {
            "off_image": ~s_on["inside"], "excluded": excl & s_on["inside"],
            "saturated": s_on["saturated"] & s_on["inside"] & ~excl,
            "dark": (s_on["mean"] < cfg.dark_linear) & s_on["inside"] & ~excl & ~s_on["saturated"],
        }
        ok = s_on["inside"] & ~excl & ~s_on["saturated"] & (s_on["mean"] >= cfg.dark_linear)
        obs, mode, aref = s_on["mean"], _ALONE, np.zeros_like(s_on["mean"])
        if i in partner:
            j = partner[i]
            s_off = _sample(img(j), views[j].camera, views[j].T_cam_plane, grid, cfg)
            ok_off = s_off["inside"] & ~s_off["saturated"] & (s_off["mean"] >= cfg.dark_linear)
            both = ok & ok_off
            face = both & (grid.e[None, :] >= cfg.hint_tolerance_mm)
            level = float(np.median(s_off["mean"][face])) if face.sum() >= 20 else 0.0
            ratio = np.where(both, s_on["mean"] / np.maximum(s_off["mean"], 1e-9), np.nan)
            # where the torch adds least (the limb, whatever lies beyond it) the on/off ratio is the exposure ratio
            k_pre = float(np.nanpercentile(ratio, 5)) if both.sum() >= 20 else float("nan")
            if level < cfg.pair_min_off_level:
                diags[i].reasons.append("pair_off_too_dark")
                partner.pop(i)
            elif not (cfg.pair_exposure_range[0] <= k_pre <= cfg.pair_exposure_range[1]):
                diags[i].reasons.append("pair_exposure_not_comparable")
                diags[i].pair_exposure_ratio = k_pre if np.isfinite(k_pre) else None
                partner.pop(i)
            else:
                masks["saturated"] |= s_off["saturated"] & s_on["inside"] & ~excl
                masks["dark"] |= (s_off["mean"] < cfg.dark_linear) & s_off["inside"] & s_on["inside"] & ~excl & ~masks["saturated"]
                masks["off_image"] |= ~s_off["inside"]
                ok = both
                if cfg.pair_mode == "ratio":
                    mode, obs = _RATIO, np.where(both, ratio, 0.0)
                else:
                    mode, aref = _DIFF, np.where(both, s_off["mean"], 0.0)
        return obs, ok, {**masks, "px_per_mm": s_on["px_per_mm"]}, mode, aref

    e_far = cfg.hint_tolerance_mm + 5.0
    cand: list[int] = []
    obs1, ok1, mode1 = [], [], []
    for i in on:
        obs, ok, masks, mode, _aref = observe(i, grid1, excl1)
        if int((ok & (grid1.e[None, :] >= e_far)).sum()) < 30:
            frac = {m: float(masks[m].mean()) for m in ("off_image", "excluded", "saturated", "dark")}
            why = max(frac, key=lambda m: frac[m])
            diags[i].reasons.append({"off_image": "edge_region_not_in_photo", "excluded": "edge_region_excluded", "saturated": "overexposed", "dark": "underexposed"}[why])
            diags[i].mask_fractions = frac
            continue
        cand.append(i)
        obs1.append(obs)
        ok1.append(ok)
        mode1.append(_RATIO if mode == _RATIO else _ALONE)  # for the first guess a difference view is a torch photo
    if not cand:
        for r_ in sorted({r_ for d in diags for r_ in d.reasons}):
            note(r_)
        return finish("needs_view", "edge_region_not_usable", "No torch-on photo shows the flat face next to the edge well exposed: retake with the edge region in the middle of the frame, without clipping.", {"code": "reframe_edge", "torch": "on"})

    # ---- model-free estimate: axis from normalised-brightness gradients ----
    G1 = _make_geom(grid1, [views[i].T_cam_plane for i in cand], mode1, np.zeros((len(cand),) + grid1.shape))
    norm, flat_guess = _normalised_brightness(np.asarray(obs1), np.asarray(ok1), G1, grid1, e_far)
    stage: list[dict[str, Any]] = []
    for k, i in enumerate(cand):
        if norm[k] is None:
            diags[i].reasons.append("flat_face_not_usable")
            continue
        ga = _gradient_axis(norm[k], grid1)
        on_ = _onset(_stacked_profile(norm[k], grid1, ga["slope"] if ga else 0.0), grid1, cfg.hint_tolerance_mm)
        if on_ is None or on_["depth"] < 0.08:
            diags[i].reasons.append("no_transition_in_this_photo")
            continue
        stage.append({"view": i, "gradient": ga, "onset": on_})
    if not stage:
        return finish("unsupported", "no_shading_transition", "No torch shading transition was found near the hinted edge: the surface looks flat there, or the hint is more than about %d mm off." % int(cfg.hint_tolerance_mm))
    slopes = np.array([s_["gradient"]["slope"] for s_ in stage if s_["gradient"]])
    wts = np.array([s_["gradient"]["n"] for s_ in stage if s_["gradient"]], dtype=float)
    onset0 = float(np.median([s_["onset"]["e"] for s_ in stage]))
    delta = float(np.arctan(_weighted_median(slopes, wts))) if slopes.size else 0.0
    # iso-lines run along t + tan(delta) n: rotate the hint frame by delta
    t_a = np.cos(delta) * t_h + np.sin(delta) * n_h
    n_a = np.cos(delta) * n_h - np.sin(delta) * t_h
    pivot = p_h + onset0 * n_h + 0.5 * (w0 + w1) * t_h
    res.gradient_axis = {
        "method": "regression of the along-edge on the across-edge gradient of the normalised brightness (its iso-lines are parallel to the cylinder axis in every view)",
        "direction": t_a.tolist(),
        "inward": n_a.tolist(),
        "angle_from_hint_deg": float(np.degrees(delta)),
        "per_view_angle_deg": [float(np.degrees(np.arctan(s_["gradient"]["slope"]))) if s_["gradient"] else None for s_ in stage],
        "spread_deg": float(np.degrees(np.ptp(np.arctan(slopes)))) if slopes.size >= 2 else None,
        "onset_point": pivot.tolist(),
        "onset_per_view_mm": [float(s_["onset"]["e"] - onset0) for s_ in stage],
        "onset_note": "start of the brightness plateau walking in from the limb: a starting value for the fit, not an estimate",
        "flat_field_guess": flat_guess,
        "n_views": len(stage),
    }

    # ---- pass 2: sections aligned with that estimate -----------------------
    used = [s_["view"] for s_ in stage]
    w2 = (np.arange(n_sec) + 0.5) * spacing - 0.5 * (w1 - w0)
    e2 = np.arange(-(outward - cfg.hint_tolerance_mm), cfg.flat_extent_mm + 1e-9, cfg.step_mm)
    grid2 = _Grid(pivot, n_a, t_a, e2, w2, cfg.step_mm, spacing)
    excl2 = _in_polygons(grid2.points(), exclude_polygons, cfg.exclude_margin_mm)
    obs_l, ok_l, mask_l, mode_l, blur_l, aref_l = [], [], [], [], [], []
    for i in used:
        obs, ok, masks, mode, aref = observe(i, grid2, excl2)
        obs_l.append(obs)
        ok_l.append(ok)
        mask_l.append(masks)
        mode_l.append(mode)
        aref_l.append(aref)
        sb = views[i].blur_sigma_px if views[i].blur_sigma_px is not None else cfg.default_blur_px
        # lens blur + pixel footprint + bilinear sampling (px^2), on the plane, plus the two sub-samples across a cell
        var_mm = (float(sb) ** 2 + 0.25) / np.maximum(masks["px_per_mm"], 1e-3) ** 2 + (cfg.step_mm / 4.0) ** 2
        blur_l.append(np.clip(0.5 * var_mm / cfg.step_mm ** 2, 0.0, 0.4))
        diags[i].role = {_ALONE: "torch", _RATIO: "torch_over_partner", _DIFF: "torch_minus_partner"}[mode]
        if mode != _ALONE:
            diags[partner[i]].role = "partner"
            diags[partner[i]].used = True
    # Flash-off photos of "difference" views show the same static scene: scale each to 1 on the
    # part of the flat face that all of them see, which makes their exposures comparable.
    aref_arr = np.asarray(aref_l, dtype=float)
    off_level = np.ones(len(used))
    is_diff = np.array([m == _DIFF for m in mode_l])
    if is_diff.any():
        face = grid2.e[None, :] > 5.0
        common = np.all(np.asarray(ok_l)[is_diff], axis=0) & face
        for k in np.nonzero(is_diff)[0]:
            sel = common if common.sum() >= 50 else (ok_l[k] & face)
            off_level[k] = float(np.median(aref_arr[k][sel])) if sel.sum() >= 10 else float(np.median(aref_arr[k][ok_l[k]]))
            aref_arr[k] = _denoise_ambient(aref_arr[k], ok_l[k], grid2.w, cfg.step_mm) / max(off_level[k], 1e-9)
    aref_arr[~is_diff] = 1.0
    G = _make_geom(grid2, [views[i].T_cam_plane for i in used], mode_l, np.asarray(blur_l), aref_arr)
    fit = _Fit(G, np.asarray(obs_l), np.asarray(ok_l), cfg, hint.radius_mm)
    V = G.n_views
    res.n_views_used = V

    # ---- joint robust fit ---------------------------------------------------
    fit.sigma = np.sqrt(cfg.sigma_abs ** 2 + (cfg.sigma_rel * np.abs(fit.obs)) ** 2)
    fit.th[_I_M] = flat_guess["beam_exponent"]
    _coarse_start(fit, hint.radius_mm)
    aux = fit.select()
    fit.lin = fit.solve_gains(robust=True)
    fit.set_sigma()
    # staged start: (1) lighting on the flat face only, where the shape does not matter,
    # (2) shape and ambient level with that lighting, then (3) everything together
    fit.run_staged((_I_M, _I_M2, _I_V1, _I_V2, _I_BX, _I_BY, _I_RHO), aux["e"] > 8.0, 15)
    fit.run_staged((_I_D, _I_PSI, _I_LOGR, _I_RHO), None, 12)
    fit.select()
    fit.set_sigma()
    # cells far beyond the bend are never used: drop those columns (keeps a margin for a growing radius)
    cols = np.nonzero(fit.active.any(axis=(0, 1)))[0]
    if cols.size and cols[0] > 8:
        j0 = int(cols[0]) - 8
        fit.crop(j0)
        mask_l = [{k_: v_[:, j0:] for k_, v_ in m_.items()} for m_ in mask_l]
        fit.select()
    converged = True
    changed = 1.0
    for rnd in range(max(cfg.rounds, 1)):
        before = fit.active.copy()
        converged = fit.run(cfg.max_iterations if rnd == 0 else max(cfg.max_iterations // 2, 8))
        if rnd == 0:
            fit.rescale_noise()
        aux = fit.select()
        _classify_outliers(fit, aux)
        aux = fit.select()
        if fit.free[_I_KS] and float(np.nanmax(np.where(fit.base_ok, aux["spec_frac"], 0.0))) < 0.005:
            fit.th[_I_KS] = 0.0  # no gloss worth modelling
            fit.free[[_I_KS, _I_LOGSH]] = False
            aux = fit.select()
        fit.set_sigma()
        changed = float(np.mean(before != fit.active))
        if rnd >= 1 and changed < 0.005:
            break
    if changed >= 0.001:  # the set of cells changed after the last solve
        converged = fit.run(max(cfg.max_iterations // 2, 8))
    if fit.free[_I_KS] and fit.th[_I_LOGSH] <= fit.lo[_I_LOGSH] + 0.02:
        # A lobe at the broad end of the range is not gloss: it is a second diffuse-like term that
        # trades against the ambient level and the tone exponent.  Drop it and solve again.
        fit.th[_I_KS] = 0.0
        fit.free[[_I_KS, _I_LOGSH]] = False
        fit.select()
        fit.set_sigma()
        converged = fit.run(cfg.max_iterations)
        aux = fit.select()
        _classify_outliers(fit, aux)
        fit.select()
        fit.set_sigma()
        converged = fit.run(max(cfg.max_iterations // 2, 8)) and converged
    if not converged:  # a flat valley can need more steps than the round budget
        converged = fit.run(cfg.max_iterations)

    # ---- uncertainty: formal, row-block sandwich, delete-one-view jackknife, fixed pattern ----
    def statistics() -> tuple[dict[str, Any], list[Any], dict[str, float], dict[str, float]]:
        cov_ = _covariances(fit)
        loo_ = _leave_one_out(fit, cov_)
        y_, aux_ = fit.predict(aux=True)
        return cov_, loo_, _jackknife(loo_), _fixed_pattern(fit, aux_, y_)

    cov, loo, jack, fixed = statistics()

    def sigma_of(name: str, which: str = "sigma_formal") -> float:
        return float(cov[which][cov["names"].index(name)]) if name in cov["names"] else float("nan")

    def stated(name: str) -> float:
        """Largest of the available standard-deviation estimates of a shape parameter."""
        c = [x for x in (sigma_of(name), sigma_of(name, "sigma_rows"), jack.get(name, float("nan"))) if np.isfinite(x)]
        return max(c) if c else float("nan")

    # configured radius as a fallback when the photos do not determine the radius
    radius_source = "fitted"
    R_fit = float(np.exp(fit.th[_I_LOGR]))
    if hint.radius_mm and (not np.isfinite(stated("log_radius")) or stated("log_radius") > cfg.max_rel_sigma_radius):
        fit.th[_I_LOGR] = np.log(float(hint.radius_mm))
        fit.free[_I_LOGR] = False
        fit.select()
        fit.set_sigma()
        converged = fit.run()
        cov, loo, jack, fixed = statistics()
        radius_source = "hint_fixed"
        R_fit = float(hint.radius_mm)
    names = cov["names"]

    # ---- numbers the decision is based on ----------------------------------
    y, aux = fit.predict(aux=True)
    flat_ref = fit.predict(flat_only=True)
    rel = np.where(fit.active, fit.obs / np.maximum(np.where(np.isfinite(y), y, 1.0), 1e-12) - 1.0, np.nan)
    held = _heldout(fit, loo)
    offsets, offsets_sigma = _per_view_offsets(fit, cov)
    psi, d_fit = float(fit.th[_I_PSI]), float(fit.th[_I_D])
    n_fit = np.cos(psi) * n_a - np.sin(psi) * t_a
    t_fit = np.cos(psi) * t_a + np.sin(psi) * n_a
    p_fit = pivot + d_fit * n_fit
    rows_used = np.nonzero(fit.active.any(axis=(0, 2)))[0]
    bend_cells = fit.active & (aux["e"] < 0.0)
    flat_cells = fit.active & (aux["e"] >= 0.0)
    cover = [float(np.degrees(np.max(aux["phi"][v][bend_cells[v]]))) if bend_cells[v].any() else 0.0 for v in range(V)]
    flash_frac = [float(np.median(aux["flash_frac"][v][flat_cells[v]])) if flat_cells[v].any() else float("nan") for v in range(V)]
    sv = cov["sv"]
    cond = float(min(sv[0] / max(sv[-1], 1e-300), 1e12))
    shape_names = [n for n in ("offset_mm", "angle_rad", "log_radius") if n in names]
    k_shape = [names.index(n) for n in shape_names]
    corr = cov["corr"]
    worst: dict[str, Any] = {}
    for n_, k in zip(shape_names, k_shape):
        others = [(abs(float(corr[k, j])), names[j]) for j in range(len(names)) if j != k]
        if others:
            c, who = max(others)
            worst[n_] = {"with": who, "abs_correlation": c}
    length = float((rows_used[-1] - rows_used[0] + 1) * spacing) if rows_used.size else 0.0
    # allowance for brightness that is fixed to the surface (see _fixed_pattern)
    fp_t = float(fixed["offset_mm"])
    fp_a = float(np.degrees(0.5 * np.sqrt(12.0) * fp_t / max(length, 1.0)))
    fp_r = 1.5 * float(fixed["amplitude"])
    finite_off = np.isfinite(offsets)
    res.conditioning = {
        "n_cells_used": int(cov["n_act"]), "n_parameters": int(cov["keep"].size), "reduced_chi2": cov["chi2"],
        "noise_scale_rel": float(fit.noise_scale * cfg.sigma_rel),
        "condition_number_scaled": cond, "singular_values_scaled": [float(s_) for s_ in sv],
        "shape_parameters": shape_names,
        "shape_correlation": corr[np.ix_(k_shape, k_shape)].tolist(),
        "strongest_correlation": worst,
        "sigma_formal": {n_: float(cov["sigma_formal"][k]) for n_, k in zip(shape_names, k_shape)},
        "sigma_row_blocks": {n_: _r(cov["sigma_rows"][k], 6) for n_, k in zip(shape_names, k_shape)},
        "sigma_view_jackknife": {n_: _r(jack.get(n_), 6) for n_ in shape_names},
        "fixed_pattern": {"amplitude_rel": fixed["amplitude"], "coherent_rel": fixed["coherent"], "incoherent_rel": fixed["incoherent"],
                          "bend_slope_per_mm": _r(fixed["slope_per_mm"], 6), "allowance_tangent_mm": fp_t,
                          "allowance_angle_deg": fp_a, "allowance_radius_rel": fp_r},
        "parameters_at_bound": cov["at_bound"],
        "converged": converged, "model_evaluations": int(fit.nfev),
        "per_view_tangent_offset_mm": [_r(o, 4) for o in offsets],
        "per_view_tangent_offset_sigma_mm": [_r(o, 4) for o in offsets_sigma],
        "stationarity_mm": _r(float(np.max(np.abs(offsets[finite_off]))) if finite_off.any() else None, 4),
    }
    res.heldout = held
    gam = float(fit.th[_I_GAMMA])
    res.nuisance = {
        "beam_exponent": {"value": float(fit.th[_I_M]), "sigma": _r(sigma_of("beam_exponent"), 4), "fitted": bool(fit.free[_I_M]),
                          "second_order": float(fit.th[_I_M2]), "second_order_sigma": _r(sigma_of("beam_shape2"), 4),
                          "note": "log beam = m u + m2 u^2 with u = log cos(angle from the optical axis)"},
        "lens_falloff": {"value": [float(fit.th[_I_V1]), float(fit.th[_I_V2])], "sigma": [_r(sigma_of("lens_falloff"), 4), _r(sigma_of("lens_falloff2"), 4)], "fitted": bool(fit.free[_I_V1]),
                         "note": "log fall-off = v1 u + v2 u^2 with u = log cos(field angle); separable from the beam only through the ambient share"},
        "ambient_to_torch_ratio": {"value": float(fit.th[_I_RHO]), "sigma": _r(sigma_of("ambient_ratio"), 5),
                                   "note": "ambient light on the flat face / torch irradiance at 500 mm and normal incidence"},
        "light_offset_cam_mm": {
            "value": [float(fit.th[_I_OX]), float(fit.th[_I_OY])], "sigma": [_r(sigma_of("light_x_mm"), 3), _r(sigma_of("light_y_mm"), 3)],
            "fitted": bool(fit.free[_I_OX]), "prior_sigma_mm": cfg.light_offset_prior_mm,
            "determined_by_data": bool(fit.free[_I_OX] and np.isfinite(sigma_of("light_x_mm")) and max(sigma_of("light_x_mm"), sigma_of("light_y_mm")) < 0.5 * cfg.light_offset_prior_mm),
        },
        "beam_tilt": {"value": [float(fit.th[_I_BX]), float(fit.th[_I_BY])], "fitted": bool(fit.free[_I_BX])},
        "specular": {"strength": float(fit.th[_I_KS]), "shininess": float(np.exp(fit.th[_I_LOGSH])), "fitted": bool(fit.free[_I_KS])},
        "tone_gamma": {"value": gam, "sigma": _r(sigma_of("tone_gamma"), 4), "fitted": bool(fit.free[_I_GAMMA]), "prior_sigma": cfg.tone_gamma_prior},
        "albedo_gradient_per_mm": {"value": float(fit.th[_I_A1]), "fitted": bool(fit.free[_I_A1])},
        "ambient_harmonics": {"value": [float(fit.th[_I_AL1]), float(fit.th[_I_AL2])], "fitted": bool(fit.free[_I_AL1])},
        "pair_mode": cfg.pair_mode if bool(np.any(G.mode != _ALONE)) else None,
    }

    # ---- per-view diagnostics ------------------------------------------------
    n_cells = int(np.prod(G.shape))
    held_by_view = {p["view"]: p for p in held.get("per_view", [])}
    none = np.zeros_like(fit.active)
    for k, i in enumerate(used):
        dg = diags[i]
        dg.used = bool(fit.active[k].sum() >= 20)
        dg.n_cells, dg.n_used = n_cells, int(fit.active[k].sum())
        mf = {m: float(mask_l[k][m].mean()) for m in ("off_image", "excluded", "saturated", "dark")}
        mf["grazing_or_terminator"] = float(fit.flags["grazing"][k].mean())
        mf["beyond_modelled_bend"] = float(fit.flags["beyond"][k].mean())
        mf["glare"] = float((fit.flags["glare"][k] | fit.flags.get("bright", none)[k]).mean())
        mf["shadow_or_occlusion"] = float(fit.flags.get("shadow", none)[k].mean())
        mf["material_or_fixed_pattern"] = float(fit.flags.get("material", none)[k].mean())
        mf["used"] = float(fit.active[k].mean())
        dg.mask_fractions = mf
        if G.mode[k] == _RATIO:
            dg.pair_exposure_ratio = float(np.exp(fit.lin[k]))
        else:
            dg.gain = float(np.exp(fit.lin[k]))
            if G.mode[k] == _DIFF:  # ambient part of the on photo relative to the off photo, in linear radiance
                dg.pair_exposure_ratio = float(np.exp(fit.lin[k] / gam) * max(fit.th[_I_RHO], 0.0) / max(off_level[k], 1e-9) ** (1.0 / gam))
        dg.camera_across_mm, dg.camera_height_mm = float(aux["ce"][k]), float(aux["ch"][k])
        dg.view_angle_deg = float(np.degrees(np.arctan2(aux["ce"][k], aux["ch"][k])))
        dg.bend_coverage_deg = cover[k]
        dg.flash_fraction = flash_frac[k] if np.isfinite(flash_frac[k]) else None
        rv = rel[k][np.isfinite(rel[k])]
        dg.rms_rel = float(np.sqrt(np.mean(rv ** 2))) if rv.size else None
        dg.heldout_rms_rel = held_by_view.get(k, {}).get("heldout_rms_rel")
        dg.tangent_offset_mm = float(offsets[k]) if np.isfinite(offsets[k]) else None
        if not dg.used:
            dg.reasons.append("too_few_usable_cells")

    # ---- profiles for inspection ----------------------------------------------
    if rows_used.size:
        pick = rows_used[np.unique(np.linspace(0, rows_used.size - 1, min(cfg.n_profile_sections, rows_used.size)).round().astype(int))]
        for k, i in enumerate(used):
            for row in pick:
                fr = np.maximum(flat_ref[k, row], 1e-12)
                meas = np.where(fit.base_ok[k, row], fit.obs[k, row] / fr, np.nan)
                mod = np.where(aux["valid"][k, row] & np.isfinite(y[k, row]), y[k, row] / fr, np.nan)
                res.profiles.append(SectionProfile(
                    frame_id=diags[i].frame_id, height_mm=float(grid2.w[row]), e_mm=[float(x) for x in aux["e"][k, row]],
                    measured=[None if not np.isfinite(x) else float(x) for x in meas], model=[None if not np.isfinite(x) else float(x) for x in mod],
                    used=[bool(b) for b in fit.active[k, row]],
                ))

    # ---- estimates ----------------------------------------------------------
    sig_d = float(np.sqrt(stated("offset_mm") ** 2 + fp_t ** 2 + cfg.sigma_floor_tangent_mm ** 2))
    sig_a = float(np.sqrt(np.degrees(stated("angle_rad")) ** 2 + fp_a ** 2 + cfg.sigma_floor_angle_deg ** 2))
    rel_sig_R = float(np.sqrt(stated("log_radius") ** 2 + fp_r ** 2 + cfg.sigma_floor_rel_radius ** 2)) if radius_source == "fitted" else float("nan")
    res.tangent, res.tangent_point, res.direction = Line2D(n_fit, float(n_fit @ p_fit)), p_fit, t_fit
    if rows_used.size:
        res.endpoints = np.stack([p_fit + (grid2.w[rows_used[0]] - 0.5 * spacing) * t_fit, p_fit + (grid2.w[rows_used[-1]] + 0.5 * spacing) * t_fit])
    res.radius_mm, res.radius_source = R_fit, radius_source
    res.sigma_tangent_mm, res.sigma_angle_deg = sig_d, sig_a
    res.sigma_radius_mm = None if radius_source != "fitted" else float(R_fit * rel_sig_R)

    # ---- decision: is the result identified by the photos? ------------------
    blockers: list[tuple[str, str | None]] = []
    needs: list[tuple[str, str, dict[str, Any]]] = []
    n_good = sum(1 for k in range(V) if fit.active[k].sum() >= 20 and bend_cells[k].sum() >= 10)
    tot_bend = int(bend_cells.sum())
    max_cover = max(cover) if cover else 0.0
    view_ang = np.degrees(np.arctan2(aux["ce"], aux["ch"]))
    toward_edge = {"code": "move_toward_edge", "torch": "on", "direction_plane": (-n_fit).round(4).tolist(), "distance_mm": 150}
    crit = fit.base_ok & (aux["e"] < 8.0) & ~fit.flags["beyond"] & ~fit.flags["grazing"]
    mat = fit.flags.get("material", none)
    mat_frac = float((mat & crit).sum() / max(crit.sum(), 1))
    rms_all = float(np.sqrt(np.nanmean(np.clip(rel, -0.5, 0.5) ** 2))) if np.any(np.isfinite(rel)) else float("nan")
    res.conditioning.update(
        rms_rel=rms_all, material_fraction_near_edge=mat_frac, bend_cells_used=tot_bend, max_bend_coverage_deg=max_cover,
        shadow_fraction=float((fit.flags.get("shadow", none) & fit.base_ok).sum() / max(fit.base_ok.sum(), 1)),
    )
    if any("pair_misaligned" in d.reasons for d in diags):
        note("pair_misaligned", f"A flash-off partner was not taken from the same place (the camera centres must agree within {cfg.pair_max_center_mm:.0f} mm): retake the pair from a rest. That torch-on photo was analysed without its partner.",
             {"code": "retake_pair_from_rest", "max_center_distance_mm": cfg.pair_max_center_mm})
    if any("pair_off_too_dark" in d.reasons for d in diags):
        note("pair_off_too_dark", "A flash-off partner is too dark to be useful (the room is dark enough that none is needed): its torch-on photo was analysed alone.")
    if any("pair_exposure_not_comparable" in d.reasons for d in diags):
        note("pair_exposure_not_comparable", "The exposure of a flash-off partner differs too much from its torch-on photo: the pair was not used as a pair.")
    if tot_bend < cfg.min_bend_cells or max_cover < 10.0:
        needs.append(("bend_not_covered", "The rounded part is not visible well enough in the torch-on photos (seen too obliquely or cut off): take a photo from further toward the edge side, torch on, with the edge near the middle of the frame.", toward_edge))
    if n_good < cfg.min_views:
        needs.append(("too_few_views", f"Only {n_good} torch-on photo(s) show the rounded edge usefully; at least {cfg.min_views} from clearly different positions are needed so that the result can be cross-checked: add a torch-on photo after moving about 150 mm sideways across the edge.",
                      {"code": "add_view", "torch": "on", "direction_plane": (-n_fit).round(4).tolist(), "distance_mm": 150}))
    ff = [f_ for f_ in flash_frac if np.isfinite(f_)]
    if ff and float(np.median(ff)) < cfg.min_flash_fraction:
        needs.append(("ambient_dominates", "Room light dominates the torch on the panel (torch share %.0f %%): move closer (about 400 to 500 mm), dim the room light, or take aligned flash-off partners from a rest." % (100.0 * float(np.median(ff))),
                      {"code": "increase_torch_share", "torch": "on", "max_distance_mm": 500}))
    if mat_frac > cfg.max_material_fraction:
        blockers.append(("surface_fixed_pattern_at_edge", "Brightness that stays fixed on the surface (a paint or material boundary, a fixed shadow, or a shape that is not a flat face joined to a circular bend of at least %d degrees) lies on or next to the rounded edge: torch shading cannot be separated from it. Use the edge-only silhouette method%s." % (int(cfg.max_bend_deg), "" if bool(np.any(G.mode == _RATIO)) else ", or aligned flash-off partners with pair_mode='ratio' if it is paint")))
    if not converged:
        blockers.append(("fit_not_converged", None))
    if "log_radius" in cov["at_bound"] or not (cfg.radius_bounds_mm[0] <= R_fit <= cfg.radius_bounds_mm[1]):
        blockers.append(("radius_out_of_range", None))
    if np.isfinite(rms_all) and rms_all > cfg.max_rms_rel:
        blockers.append(("model_does_not_explain_brightness", "The torch shading model does not explain these photos (non-linear tone rendering, strong gloss, textured or patterned surface, or a shape that is not a flat face joined to a circular bend)."))
    # a view whose own transition lies clearly (in mm and in its own standard deviations) off the common line
    moved = finite_off & (np.abs(np.nan_to_num(offsets)) > cfg.max_stationarity_mm) & (np.abs(np.nan_to_num(offsets)) > 4.0 * np.nan_to_num(offsets_sigma, nan=np.inf))
    if V >= 2 and bool(np.any(moved)):
        blockers.append(("transition_not_stationary", "The brightness transition does not stay on one line of the panel when the phone moves: it behaves like a shadow, a reflection or uneven room light, not like the start of the bend."))
    if held.get("available") and (held["max_ratio"] > cfg.max_heldout_ratio or held["max_heldout_rms_rel"] > cfg.max_heldout_rms_rel):
        blockers.append(("heldout_inconsistent", "A photo that was left out is not predicted by the others: something differs between photos (moving shadow, reflection, uneven room light, tone rendering) or a pose is wrong; retake the sweep, or add aligned flash-off partners."))
    if fp_t > cfg.max_surface_allowance_mm:
        blockers.append(("surface_not_uniform", "The brightness of the flat face has a pattern that stays on the surface (texture, uneven paint, dirt: about %.1f %%): it limits the photometric estimate to roughly %.1f mm and more torch photos will not help; aligned flash-off partners with pair_mode='ratio' would." % (100.0 * fixed["amplitude"], fp_t)))
    weak_t = (not np.isfinite(sig_d)) or sig_d > cfg.max_sigma_tangent_mm
    weak_R = radius_source == "fitted" and ((not np.isfinite(rel_sig_R)) or rel_sig_R > cfg.max_rel_sigma_radius)
    if cond > cfg.max_condition or weak_t or weak_R:
        why = "radius_not_identifiable" if (weak_R and not weak_t) else ("tangent_not_identifiable" if weak_t else "ill_conditioned")
        if max_cover < cfg.min_bend_coverage_deg or float(np.min(view_ang)) > 12.0:
            text, req = "The photos see the bend only from over the flat face: take a photo from further toward the edge side (phone in front of or slightly beyond the rounded edge), torch on.", toward_edge
        elif float(np.ptp(view_ang)) < 12.0:
            text = "The torch-on photos were all taken from nearly the same place across the edge: add one after moving at least 150 mm sideways (perpendicular to the edge), torch on."
            req = {"code": "add_view", "torch": "on", "direction_plane": n_fit.round(4).tolist(), "distance_mm": 150}
        else:
            text = "The photos do not determine the transition well enough: add torch-on photos closer to the edge (400 to 500 mm) with the rounded edge near the middle of the frame."
            req = {"code": "closer_view", "torch": "on", "max_distance_mm": 500}
        needs.append((why, text, req))
    if radius_source == "hint_fixed":
        note("radius_fixed_to_configured_value")
    if any(d.mask_fractions.get("saturated", 0.0) > 0.25 for d in diags):
        note("overexposed_views", "Parts of the edge region are clipped in some torch-on photos: step back a little or lower the exposure.")
    if any(d.mask_fractions.get("dark", 0.0) > 0.25 for d in diags):
        note("underexposed_views", "Parts of the edge region are too dark in some torch-on photos to be measured: move closer or raise the exposure.")

    res.runtime_s = time.perf_counter() - t_start
    if blockers:
        res.reasons = [b_[0] for b_ in blockers] + [r_ for r_ in res.reasons if r_ not in [b_[0] for b_ in blockers]]
        res.suggestions = [b_[1] for b_ in blockers if b_[1]] + res.suggestions
        res.status = "unsupported"
        _withhold(res)
    elif needs:
        res.reasons = [n_[0] for n_ in needs] + [r_ for r_ in res.reasons if r_ not in [n_[0] for n_ in needs]]
        res.suggestions = [n_[1] for n_ in needs] + res.suggestions
        res.requests = [n_[2] for n_ in needs] + res.requests
        res.status = "needs_view"
        _withhold(res, keep_tangent=all(n_[0] == "radius_not_identifiable" for n_ in needs))
    else:
        res.status = "supported"
    return res


def _withhold(res: PhotometricResult, keep_tangent: bool = False) -> None:
    """Remove estimates that the evidence does not support.  They stay in the
    diagnostics for a researcher (``conditioning["withheld_estimate"]``), never as a result."""
    res.conditioning["withheld_estimate"] = {
        "note": "not identified or not cross-checked: for inspection only",
        "tangent": None if res.tangent is None else {"n": [float(x) for x in res.tangent.n], "d": float(res.tangent.d)},
        "tangent_point": None if res.tangent_point is None else [float(x) for x in res.tangent_point],
        "radius_mm": res.radius_mm, "sigma_tangent_mm": res.sigma_tangent_mm, "sigma_angle_deg": res.sigma_angle_deg, "sigma_radius_mm": res.sigma_radius_mm,
    }
    res.radius_mm = None
    res.sigma_radius_mm = None
    res.radius_source = None
    if not keep_tangent:
        res.tangent = None
        res.tangent_point = None
        res.direction = None
        res.endpoints = None
        res.sigma_tangent_mm = None
        res.sigma_angle_deg = None


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(int(np.searchsorted(c, 0.5 * c[-1])), x.size - 1)])


# --------------------------------------------------------------------------
# Combination with the edge-only (silhouette) estimate
# --------------------------------------------------------------------------
def combine_with_silhouette(
    photo: PhotometricResult,
    tangent: Line2D | None,
    sigma_tangent_mm: float,
    sigma_angle_deg: float,
    radius_mm: float | None = None,
    sigma_radius_mm: float | None = None,
    reference_point: np.ndarray | None = None,
    z_inconsistent: float = 3.0,
) -> dict[str, Any]:
    """Inverse-variance combination of the photometric and the silhouette estimate.

    Both tangent lines are compared at ``photo.tangent_point`` (offset along
    the photometric normal) and by direction.  ``reference_point`` is where
    ``sigma_tangent_mm`` of the silhouette fit applies (default: the same
    point); its direction uncertainty is carried to the comparison point.
    The two estimates use the same photos and poses, so they are not fully
    independent; when they disagree by more than ``z_inconsistent`` combined
    standard deviations nothing is combined and the disagreement is reported
    (the photometric estimate then serves as a consistency check only).
    """
    out: dict[str, Any] = {"label": EVIDENCE_LABEL, "method": "inverse_variance", "available": False}
    if photo.status != "supported" or photo.tangent is None or photo.tangent_point is None or photo.direction is None:
        out["reason"] = "photometric_not_supported"
        return out
    if tangent is None or not (np.isfinite(sigma_tangent_mm) and np.isfinite(sigma_angle_deg)):
        out["reason"] = "silhouette_not_available"
        return out
    p = photo.tangent_point
    sil = tangent.oriented_towards(p + 50.0 * photo.tangent.n)
    lever = 0.0 if reference_point is None else float((np.asarray(reference_point, dtype=float) - p) @ photo.direction)
    off_s = float(-sil.signed_distance(p))  # silhouette line relative to the photometric one, along its normal
    s_s = float(np.hypot(sigma_tangent_mm, lever * np.radians(sigma_angle_deg)))
    s_p = float(photo.sigma_tangent_mm)  # type: ignore[arg-type]
    ang_s = float(np.degrees(np.arctan2(photo.tangent.n[0] * sil.n[1] - photo.tangent.n[1] * sil.n[0], float(photo.tangent.n @ sil.n))))
    a_p, a_s = float(photo.sigma_angle_deg), max(float(sigma_angle_deg), 1e-6)  # type: ignore[arg-type]
    z_off = abs(off_s) / max(float(np.hypot(s_s, s_p)), 1e-9)
    z_ang = abs(ang_s) / max(float(np.hypot(a_s, a_p)), 1e-9)
    both_r = radius_mm is not None and photo.radius_mm is not None and bool(sigma_radius_mm) and bool(photo.sigma_radius_mm)
    z_rad = abs(radius_mm - photo.radius_mm) / max(float(np.hypot(sigma_radius_mm, photo.sigma_radius_mm)), 1e-9) if both_r else None  # type: ignore[operator,arg-type]
    out.update(offset_difference_mm=off_s, angle_difference_deg=ang_s, z_offset=float(z_off), z_angle=float(z_ang), z_radius=None if z_rad is None else float(z_rad))
    out["consistent"] = bool(z_off <= z_inconsistent and z_ang <= z_inconsistent and (z_rad is None or z_rad <= z_inconsistent))
    if not out["consistent"]:
        out["reason"] = "estimates_disagree"
        return out
    wp, ws = 1.0 / s_p ** 2, 1.0 / max(s_s, 1e-9) ** 2
    off = ws * off_s / (wp + ws)
    wa_p, wa_s = 1.0 / a_p ** 2, 1.0 / a_s ** 2
    ang = np.radians(wa_s * ang_s / (wa_p + wa_s))
    c, s = np.cos(ang), np.sin(ang)
    n = np.array([c * photo.tangent.n[0] - s * photo.tangent.n[1], s * photo.tangent.n[0] + c * photo.tangent.n[1]])
    q = p + off * photo.tangent.n
    out.update(
        available=True,
        tangent={"n": n.tolist(), "d": float(n @ q)},
        tangent_point=q.tolist(),
        sigma_tangent_mm=float(1.0 / np.sqrt(wp + ws)),
        sigma_angle_deg=float(1.0 / np.sqrt(wa_p + wa_s)),
        weight_photometric=float(wp / (wp + ws)),
    )
    if both_r:
        wr_p, wr_s = 1.0 / photo.sigma_radius_mm ** 2, 1.0 / sigma_radius_mm ** 2  # type: ignore[operator]
        out.update(radius_mm=float((wr_p * photo.radius_mm + wr_s * radius_mm) / (wr_p + wr_s)), sigma_radius_mm=float(1.0 / np.sqrt(wr_p + wr_s)))  # type: ignore[operator]
    elif photo.radius_mm is not None:
        out.update(radius_mm=float(photo.radius_mm), sigma_radius_mm=photo.sigma_radius_mm)
    return out


def tag_exclusion_polygons(tag_corners_plane: Sequence[np.ndarray], grow: float = 0.45) -> list[np.ndarray]:
    """Plane-frame polygons covering tag stickers (the black square plus its white
    film and printed label): each tag square grown by ``grow`` times its size on every side."""
    out = []
    for c in tag_corners_plane:
        c = np.asarray(c, dtype=float).reshape(-1, 2)
        ctr = c.mean(axis=0)
        out.append(ctr + (c - ctr) * (1.0 + 2.0 * grow))
    return out
