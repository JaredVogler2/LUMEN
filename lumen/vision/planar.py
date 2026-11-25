"""Planar metrology: metric image<->plane mapping from independently placed tags.

Model
-----
All accepted tags are assumed to lie on one plane ("the target plane").  Each
tag ``i`` has a measured edge ``s_i`` and an unknown in-plane rigid pose
``(c_i, theta_i)``; nothing is assumed about the spacing between tags.  The
*plane frame* is the frame of one tag, the gauge (``c = 0, theta = 0``): its
``+x`` is the gauge tag's right, ``+y`` its top, millimetres.

Each view ``v`` has a mapping from the plane to ideal image pixels:

* ``homography`` mode — a free 3x3 homography (8 dof).  It needs no camera
  calibration beyond lens-distortion removal and is immune to focus breathing
  and optical-stabiliser shifts of the principal point;
* ``pose`` mode — ``K [r1 r2 t]`` with calibrated intrinsics (6 dof).  It
  extrapolates much better from a single small tag but inherits every error in
  ``K`` and has the classic two-fold planar pose ambiguity, which is tested
  explicitly and fails closed.

All views and all tag poses are estimated jointly by minimising whitened
reprojection residuals (each tag observation carries the 8x8 covariance of its
four corners from the edge fits).  One tag fixes the mapping exactly
(8 equations, 8 unknowns); every additional tag adds 8 equations and only 3
unknowns, which is what makes the redundancy checks below possible.

Checks (all fail closed)
------------------------
* leave-one-tag-out cross-validation: a tag is predicted from the others and
  its *measured* size and squareness are compared with its declared size —
  this is what detects non-coplanar, bent, mis-measured or moved tags;
* residual and chi-square tests per tag observation;
* pose-mode ambiguity test (two IPPE branches refined and compared);
* agreement between homography and pose modes when both are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..errors import GeometryError
from ..geometry.camera import CameraModel
from ..geometry.homography import (
    apply_h,
    dlt_homography,
    fit_rigid_2d,
    h_local_jacobian,
    homography_from_pose,
    normalization_transform,
    normalize_h,
    pose_from_homography,
)
from ..geometry.optimize import lm_solve
from ..geometry.se3 import Pose
from .markers import CORNER_SIGNS, TagDetection


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class TagObs:
    """One accepted tag in one view (ideal pixel coordinates)."""

    tag_id: int
    corners: np.ndarray  # (4, 2)
    cov: np.ndarray  # (8, 8)
    size_mm: float
    #: True when the dark-side edge shift of this observation was estimated
    #: from the tag interior and removed (see lumen.vision.markers)
    bias_corrected: bool = True
    #: True when the shift could not be estimated on this tag and the value
    #: measured on larger tags of the same camera was removed instead
    bias_pooled: bool = False

    @staticmethod
    def from_detection(d: TagDetection) -> "TagObs":
        return TagObs(d.tag_id, np.asarray(d.corners_ideal, dtype=float), np.asarray(d.cov, dtype=float), float(d.size_mm), bool(d.bias_applied))


@dataclass
class PlanarView:
    frame_id: str
    tags: list[TagObs]
    camera: CameraModel | None = None
    #: "homography" | "pose" | "auto"
    mode: str = "auto"

    def ids(self) -> list[int]:
        return [t.tag_id for t in self.tags]


@dataclass
class TagLayout:
    """In-plane rigid pose of every tag in the plane (gauge-tag) frame."""

    gauge_id: int
    poses: dict[int, tuple[float, float, float]]  # id -> (cx, cy, theta)
    sizes: dict[int, float]

    def corners(self, tag_id: int) -> np.ndarray:
        cx, cy, th = self.poses[tag_id]
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        return (CORNER_SIGNS * (0.5 * self.sizes[tag_id])) @ R.T + np.array([cx, cy])

    def all_corners(self, ids: list[int] | None = None) -> np.ndarray:
        ids = sorted(self.poses) if ids is None else ids
        return np.concatenate([self.corners(i) for i in ids], axis=0)

    def centroid(self) -> np.ndarray:
        return np.mean([[p[0], p[1]] for p in self.poses.values()], axis=0)

    def extent_radius(self) -> float:
        """RMS distance of all tag corners from their centroid (mm)."""
        P = self.all_corners()
        return float(np.sqrt(np.mean(np.sum((P - P.mean(axis=0)) ** 2, axis=1))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gauge_id": self.gauge_id,
            "poses": {str(k): [float(v[0]), float(v[1]), float(v[2])] for k, v in sorted(self.poses.items())},
            "sizes": {str(k): float(v) for k, v in sorted(self.sizes.items())},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TagLayout":
        return TagLayout(
            gauge_id=int(d["gauge_id"]),
            poses={int(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in d["poses"].items()},
            sizes={int(k): float(v) for k, v in d["sizes"].items()},
        )


@dataclass(frozen=True)
class PlanarConfig:
    #: A tag observation whose RMS reprojection residual exceeds this is suspect.
    max_tag_rms_px: float = 0.8
    #: Whitened residual chi-square per degree of freedom above which the fit is suspect.
    max_reduced_chi2: float = 6.0
    #: Cross-validation: relative size error always treated as inconsistent.
    crossval_max_frac: float = 0.025
    #: ... and the smallest relative error that can be called inconsistent at all.
    crossval_min_frac: float = 0.004
    crossval_z: float = 3.5
    #: Shape (non-squareness) residual of a back-projected tag, as a fraction of its size.
    crossval_max_shape_frac: float = 0.012
    #: Pose-mode ambiguity: the alternative branch must be worse by this chi-square margin.
    ambiguity_min_delta_chi2: float = 16.0
    #: Use pose mode automatically only for views with this many tags or fewer.
    pose_mode_max_tags: int = 1
    max_iterations: int = 40


@dataclass
class TagCheck:
    tag_id: int
    n_views: int
    size_ratio: float = float("nan")  # measured / declared (leave-one-out)
    size_ratio_sigma: float = float("nan")
    shape_rms_frac: float = float("nan")
    rms_px: float = float("nan")
    consistent: bool = True
    checked: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def f(x: float) -> float | None:
            return None if not np.isfinite(x) else round(float(x), 6)

        return {
            "tag_id": self.tag_id,
            "n_views": self.n_views,
            "size_ratio": f(self.size_ratio),
            "size_ratio_sigma": f(self.size_ratio_sigma),
            "shape_rms_frac": f(self.shape_rms_frac),
            "rms_px": f(self.rms_px),
            "consistent": self.consistent,
            "checked": self.checked,
            "reasons": list(self.reasons),
        }


@dataclass
class ViewSolution:
    frame_id: str
    mode: str
    H: np.ndarray  # plane mm -> ideal px
    rms_px: float
    n_tags: int
    pose: Pose | None = None  # T_cam_plane (pose mode, or decomposed when K is known)
    ambiguity_delta_chi2: float | None = None
    #: index range of this view's parameters in the global parameter vector
    slice: tuple[int, int] = (0, 0)
    norm_img: np.ndarray | None = None
    norm_plane: np.ndarray | None = None
    flags: list[str] = field(default_factory=list)
    #: this view's parameter values at the solution, and (pose mode) what is
    #: needed to turn perturbed parameters back into a homography
    params: np.ndarray | None = None
    K: np.ndarray | None = None
    pose_ref: Pose | None = None

    def H_from_params(self, p: np.ndarray) -> np.ndarray:
        """Homography for (perturbed) view parameters ``p``."""
        if self.mode == "homography":
            assert self.norm_img is not None and self.norm_plane is not None
            Hn = np.append(np.asarray(p, dtype=float), 1.0).reshape(3, 3)
            return np.linalg.inv(self.norm_img) @ Hn @ self.norm_plane
        assert self.K is not None and self.pose_ref is not None
        return homography_from_pose(self.K, self.pose_ref.retract(np.asarray(p, dtype=float)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "mode": self.mode,
            "H": self.H.tolist(),
            "rms_px": round(self.rms_px, 4),
            "n_tags": self.n_tags,
            "pose": None if self.pose is None else self.pose.to_dict(),
            "ambiguity_delta_chi2": self.ambiguity_delta_chi2,
            "flags": list(self.flags),
        }


@dataclass
class PlanarSolution:
    layout: TagLayout
    views: dict[str, ViewSolution]
    checks: dict[int, TagCheck]
    reduced_chi2: float
    rms_px: float
    dof: int
    #: covariance of the full parameter vector (views first, then tag poses)
    param_cov: np.ndarray
    tag_slices: dict[int, tuple[int, int]]
    excluded_tags: dict[int, list[str]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    #: scale applied to the formal covariance: max(1, reduced chi-square)
    cov_scale: float = 1.0

    @property
    def n_tags(self) -> int:
        return len(self.layout.poses)

    @property
    def consistent(self) -> bool:
        return all(c.consistent for c in self.checks.values())

    @property
    def independently_checked(self) -> bool:
        """True when at least one tag was verified against the others."""
        return any(c.checked for c in self.checks.values())

    def plane_to_image(self, frame_id: str, pts: np.ndarray) -> np.ndarray:
        return apply_h(self.views[frame_id].H, pts)

    def image_to_plane(self, frame_id: str, pts: np.ndarray) -> np.ndarray:
        return apply_h(np.linalg.inv(self.views[frame_id].H), pts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.to_dict(),
            "views": {k: v.to_dict() for k, v in self.views.items()},
            "checks": {str(k): v.to_dict() for k, v in sorted(self.checks.items())},
            "reduced_chi2": round(self.reduced_chi2, 4),
            "rms_px": round(self.rms_px, 4),
            "dof": self.dof,
            "n_tags": self.n_tags,
            "consistent": self.consistent,
            "independently_checked": self.independently_checked,
            "excluded_tags": {str(k): v for k, v in self.excluded_tags.items()},
            "flags": list(self.flags),
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _tag_local(size: float) -> np.ndarray:
    return CORNER_SIGNS * (0.5 * size)


def _whitener(cov: np.ndarray) -> np.ndarray:
    """``W`` with ``W cov W^T = I`` (lower-triangular inverse Cholesky factor)."""
    c = 0.5 * (cov + cov.T)
    try:
        L = np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(c)
        w = np.clip(w, 1e-8, None)
        L = np.linalg.cholesky((V * w) @ V.T)
    return np.linalg.inv(L)


def backprojected_size(Hinv: np.ndarray, corners_img: np.ndarray, size: float) -> tuple[float, float]:
    """Similarity-fit size ratio and shape residual of a tag mapped to the plane.

    Returns ``(measured/declared, rms shape residual / declared size)``.
    """
    Q = apply_h(Hinv, corners_img)
    _, _, scale, rms = fit_rigid_2d(_tag_local(size), Q, allow_scale=True)
    return float(scale), float(rms / size)


def choose_gauge(views: list[PlanarView]) -> int:
    """Deterministic gauge choice: most views, then largest image size, then lowest id."""
    score: dict[int, tuple[int, float]] = {}
    for v in views:
        for t in v.tags:
            e = np.linalg.norm(np.roll(t.corners, -1, axis=0) - t.corners, axis=1).mean()
            n, s = score.get(t.tag_id, (0, 0.0))
            score[t.tag_id] = (n + 1, s + float(e))
    if not score:
        raise GeometryError("no_tags", "No accepted AprilTag is available to establish scale.")
    return sorted(score, key=lambda k: (-score[k][0], -score[k][1], k))[0]


def _pairwise_layout(views: list[PlanarView], gauge: int) -> tuple[dict[int, tuple[float, float, float]], list[int]]:
    """Initial in-plane poses by chaining per-view tag-to-tag rigid fits.

    Returns ``(poses, unreachable_ids)``.
    """
    rel: dict[tuple[int, int], list[tuple[float, np.ndarray, float]]] = {}
    for v in views:
        if len(v.tags) < 2:
            continue
        for a in v.tags:
            Ha = dlt_homography(_tag_local(a.size_mm), a.corners)
            Hai = np.linalg.inv(Ha)
            ea = float(np.linalg.norm(np.roll(a.corners, -1, axis=0) - a.corners, axis=1).mean())
            for b in v.tags:
                if b.tag_id == a.tag_id:
                    continue
                Q = apply_h(Hai, b.corners)
                th, t, _, rms = fit_rigid_2d(_tag_local(b.size_mm), Q)
                rel.setdefault((a.tag_id, b.tag_id), []).append((th, t, ea))
    poses: dict[int, tuple[float, float, float]] = {gauge: (0.0, 0.0, 0.0)}
    all_ids = sorted({t.tag_id for v in views for t in v.tags})
    frontier = [gauge]
    while frontier:
        nxt: list[int] = []
        for a in frontier:
            for b in all_ids:
                if b in poses or (a, b) not in rel:
                    continue
                ests = rel[(a, b)]
                w = np.array([e[2] for e in ests])
                # weighted circular mean of the angle, weighted mean of the translation
                th = float(np.arctan2(np.sum(w * np.sin([e[0] for e in ests])), np.sum(w * np.cos([e[0] for e in ests]))))
                t = np.sum(w[:, None] * np.array([e[1] for e in ests]), axis=0) / np.sum(w)
                ax, ay, ath = poses[a]
                ca, sa = np.cos(ath), np.sin(ath)
                tb = np.array([ax, ay]) + np.array([[ca, -sa], [sa, ca]]) @ t
                poses[b] = (float(tb[0]), float(tb[1]), float(ath + th))
                nxt.append(b)
        frontier = nxt
    return poses, [i for i in all_ids if i not in poses]


# --------------------------------------------------------------------------
# Joint problem
# --------------------------------------------------------------------------
class _Problem:
    def __init__(
        self,
        views: list[PlanarView],
        modes: list[str],
        layout: TagLayout,
        H0: list[np.ndarray],
        poses0: list[Pose | None],
        fixed_layout: bool = False,
    ) -> None:
        self.views = views
        self.modes = modes
        self.gauge = layout.gauge_id
        self.sizes = dict(layout.sizes)
        self.fixed_layout = fixed_layout
        self.layout0 = layout
        # --- parameter layout
        self.view_slices: list[tuple[int, int]] = []
        k = 0
        for m in modes:
            n = 8 if m == "homography" else 6
            self.view_slices.append((k, k + n))
            k += n
        self.tag_ids = [] if fixed_layout else [i for i in sorted(layout.poses) if i != self.gauge]
        self.tag_slices: dict[int, tuple[int, int]] = {}
        for i in self.tag_ids:
            self.tag_slices[i] = (k, k + 3)
            k += 3
        self.n_params = k
        # --- normalisations for homography views
        self.Ti: list[np.ndarray] = []
        self.Tp: list[np.ndarray] = []
        self.pose_ref: list[Pose | None] = []
        self.x0 = np.zeros(k)
        for vi, v in enumerate(views):
            img = np.concatenate([t.corners for t in v.tags], axis=0)
            pl = np.concatenate([layout.corners(t.tag_id) for t in v.tags], axis=0)
            Ti = normalization_transform(img)
            Tp = normalization_transform(pl)
            self.Ti.append(Ti)
            self.Tp.append(Tp)
            a, b = self.view_slices[vi]
            if modes[vi] == "homography":
                Hn = Ti @ H0[vi] @ np.linalg.inv(Tp)
                Hn = Hn / Hn[2, 2]
                self.x0[a:b] = Hn.ravel()[:8]
                self.pose_ref.append(None)
            else:
                assert poses0[vi] is not None
                self.pose_ref.append(poses0[vi])
        for i in self.tag_ids:
            a, b = self.tag_slices[i]
            self.x0[a:b] = layout.poses[i]
        # --- whitening
        self.W = [[_whitener(t.cov) for t in v.tags] for v in views]
        self.n_res = 8 * sum(len(v.tags) for v in views)

    # -- unpack ----------------------------------------------------------
    def view_H(self, x: np.ndarray, vi: int) -> np.ndarray:
        a, b = self.view_slices[vi]
        if self.modes[vi] == "homography":
            Hn = np.append(x[a:b], 1.0).reshape(3, 3)
            return np.linalg.inv(self.Ti[vi]) @ Hn @ self.Tp[vi]
        ref = self.pose_ref[vi]
        cam = self.views[vi].camera
        assert ref is not None and cam is not None
        return homography_from_pose(cam.without_distortion().K, ref.retract(x[a:b]))

    def view_pose(self, x: np.ndarray, vi: int) -> Pose | None:
        if self.modes[vi] != "pose":
            return None
        a, b = self.view_slices[vi]
        ref = self.pose_ref[vi]
        assert ref is not None
        return ref.retract(x[a:b])

    def tag_pose(self, x: np.ndarray, tag_id: int) -> tuple[float, float, float]:
        if tag_id in self.tag_slices:
            a, b = self.tag_slices[tag_id]
            return float(x[a]), float(x[a + 1]), float(x[a + 2])
        return self.layout0.poses[tag_id]

    def layout(self, x: np.ndarray) -> TagLayout:
        return TagLayout(self.gauge, {i: self.tag_pose(x, i) for i in self.layout0.poses}, dict(self.sizes))

    def layout_jacobian(self, x: np.ndarray, vi: int, tag_ids: list[int]) -> np.ndarray:
        """Whitened Jacobian of view ``vi``'s residuals with respect to the
        in-plane poses ``(cx, cy, theta)`` of ``tag_ids`` — available also
        when the layout is held fixed (sensitivity of a localisation to the
        layout it relies on)."""
        v = self.views[vi]
        H = self.view_H(x, vi)
        J = np.zeros((8 * len(v.tags), 3 * len(tag_ids)))
        col = {t: 3 * k for k, t in enumerate(tag_ids)}
        for ti, t in enumerate(v.tags):
            if t.tag_id not in col:
                continue
            cx, cy, th = self.tag_pose(x, t.tag_id)
            c, s = np.cos(th), np.sin(th)
            loc = _tag_local(t.size_mm)
            q = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
            Jq = h_local_jacobian(H, q)
            dq_dth = loc @ np.array([[-s, c], [-c, -s]])
            Jt = np.zeros((8, 3))
            for kk in range(4):
                Jt[2 * kk : 2 * kk + 2, 0:2] = Jq[kk]
                Jt[2 * kk : 2 * kk + 2, 2] = Jq[kk] @ dq_dth[kk]
            J[8 * ti : 8 * ti + 8, col[t.tag_id] : col[t.tag_id] + 3] = self.W[vi][ti] @ Jt
        return J

    # -- residuals -------------------------------------------------------
    def _view_res(self, x: np.ndarray, vi: int, H: np.ndarray | None = None) -> np.ndarray:
        H = self.view_H(x, vi) if H is None else H
        out = []
        for ti, t in enumerate(self.views[vi].tags):
            cx, cy, th = self.tag_pose(x, t.tag_id)
            c, s = np.cos(th), np.sin(th)
            q = _tag_local(t.size_mm) @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
            r = (apply_h(H, q) - t.corners).ravel()
            out.append(self.W[vi][ti] @ r)
        return np.concatenate(out)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.empty(self.n_res)
        J = np.zeros((self.n_res, self.n_params))
        row = 0
        for vi, v in enumerate(self.views):
            H = self.view_H(x, vi)
            a, b = self.view_slices[vi]
            n_v = 8 * len(v.tags)
            rv = self._view_res(x, vi, H)
            r[row : row + n_v] = rv
            # --- view parameters
            if self.modes[vi] == "homography":
                Hn = np.append(x[a:b], 1.0).reshape(3, 3)
                Tiinv = np.linalg.inv(self.Ti[vi])
                sx, sy = Tiinv[0, 0], Tiinv[1, 1]
                rr = row
                for ti, t in enumerate(v.tags):
                    cx, cy, th = self.tag_pose(x, t.tag_id)
                    c, s = np.cos(th), np.sin(th)
                    q = _tag_local(t.size_mm) @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
                    qn = apply_h(self.Tp[vi], q)
                    ph = np.concatenate([qn, np.ones((4, 1))], axis=1) @ Hn.T
                    w = ph[:, 2]
                    un = ph[:, 0] / w
                    vn = ph[:, 1] / w
                    Jt = np.zeros((8, 8))
                    for kk in range(4):
                        X, Y = qn[kk]
                        Jt[2 * kk, 0:3] = np.array([X, Y, 1.0]) / w[kk] * sx
                        Jt[2 * kk, 6:8] = -un[kk] * np.array([X, Y]) / w[kk] * sx
                        Jt[2 * kk + 1, 3:6] = np.array([X, Y, 1.0]) / w[kk] * sy
                        Jt[2 * kk + 1, 6:8] = -vn[kk] * np.array([X, Y]) / w[kk] * sy
                    J[rr : rr + 8, a:b] = self.W[vi][ti] @ Jt
                    rr += 8
            else:
                eps = 1e-6
                for j in range(6):
                    xp = x.copy()
                    xm = x.copy()
                    h = eps * (1.0 if j < 3 else 100.0)
                    xp[a + j] += h
                    xm[a + j] -= h
                    J[row : row + n_v, a + j] = (self._view_res(xp, vi) - self._view_res(xm, vi)) / (2.0 * h)
            # --- tag pose parameters
            rr = row
            for ti, t in enumerate(v.tags):
                if t.tag_id in self.tag_slices:
                    ta, tb = self.tag_slices[t.tag_id]
                    cx, cy, th = self.tag_pose(x, t.tag_id)
                    c, s = np.cos(th), np.sin(th)
                    loc = _tag_local(t.size_mm)
                    q = loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
                    Jq = h_local_jacobian(H, q)  # (4, 2, 2)
                    dq_dth = loc @ np.array([[-s, c], [-c, -s]])  # d(R loc)/dtheta
                    Jt = np.zeros((8, 3))
                    for kk in range(4):
                        Jt[2 * kk : 2 * kk + 2, 0:2] = Jq[kk]
                        Jt[2 * kk : 2 * kk + 2, 2] = Jq[kk] @ dq_dth[kk]
                    J[rr : rr + 8, ta:tb] = self.W[vi][ti] @ Jt
                rr += 8
            row += n_v
        return r, J


# --------------------------------------------------------------------------
# Pose-mode initialisation with ambiguity handling
# --------------------------------------------------------------------------
def _ippe_candidates(cam: CameraModel, plane_pts: np.ndarray, img_pts: np.ndarray) -> list[tuple[Pose, float]]:
    """Both planar-PnP branches (IPPE) with their RMS reprojection error."""
    obj = np.concatenate([plane_pts, np.zeros((plane_pts.shape[0], 1))], axis=1).astype(np.float64)
    K = cam.without_distortion().K
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img_pts.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        n, rvecs, tvecs, errs = 0, [], [], []
    out: list[tuple[Pose, float]] = []
    for i in range(int(n)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = np.asarray(tvecs[i], dtype=float).ravel()
        e = float(np.asarray(errs).ravel()[i]) if errs is not None and len(np.asarray(errs).ravel()) > i else float("nan")
        out.append((Pose(R, t), e))
    if not out:
        H = dlt_homography(plane_pts, img_pts)
        out.append((pose_from_homography(K, H), float("nan")))
    return out


def _mode_for(v: PlanarView, cfg: PlanarConfig) -> str:
    if v.mode in ("homography", "pose"):
        if v.mode == "pose" and v.camera is None:
            raise GeometryError("pose_mode_needs_calibration", "Pose mode requires a camera calibration.")
        return v.mode
    if v.camera is not None and len(v.tags) <= cfg.pose_mode_max_tags and v.camera.source not in ("exif_prior", "assumed", "unknown"):
        return "pose"
    return "homography"


def _solve_once(
    views: list[PlanarView],
    layout0: TagLayout,
    cfg: PlanarConfig,
    fixed_layout: bool = False,
) -> tuple[_Problem, Any, list[float | None]]:
    modes = [_mode_for(v, cfg) for v in views]
    H0: list[np.ndarray] = []
    poses0: list[Pose | None] = []
    alt: list[Pose | None] = []
    for v, m in zip(views, modes):
        pl = np.concatenate([layout0.corners(t.tag_id) for t in v.tags], axis=0)
        img = np.concatenate([t.corners for t in v.tags], axis=0)
        H0.append(dlt_homography(pl, img))
        if m == "pose":
            assert v.camera is not None
            cands = _ippe_candidates(v.camera, pl, img)
            poses0.append(cands[0][0])
            alt.append(cands[1][0] if len(cands) > 1 else None)
        else:
            poses0.append(None)
            alt.append(None)
    prob = _Problem(views, modes, layout0, H0, poses0, fixed_layout=fixed_layout)
    sol = lm_solve(prob, prob.x0, max_iter=cfg.max_iterations)

    # ambiguity: refine the other IPPE branch of each pose view (others fixed)
    amb: list[float | None] = [None] * len(views)
    for vi, m in enumerate(modes):
        if m != "pose" or alt[vi] is None:
            continue
        a, b = prob.view_slices[vi]
        ref_main = prob.pose_ref[vi]
        prob.pose_ref[vi] = alt[vi]
        x_alt = sol.x.copy()
        x_alt[a:b] = 0.0

        def fn(xv: np.ndarray, vi: int = vi, a: int = a, b: int = b, x_alt: np.ndarray = x_alt) -> tuple[np.ndarray, np.ndarray]:
            xx = x_alt.copy()
            xx[a:b] = xv
            r = prob._view_res(xx, vi)
            Jv = np.empty((r.size, 6))
            for j in range(6):
                h = 1e-6 * (1.0 if j < 3 else 100.0)
                xp = xx.copy()
                xm = xx.copy()
                xp[a + j] += h
                xm[a + j] -= h
                Jv[:, j] = (prob._view_res(xp, vi) - prob._view_res(xm, vi)) / (2 * h)
            return r, Jv

        s_alt = lm_solve(fn, np.zeros(6), max_iter=cfg.max_iterations)
        pose_alt = alt[vi].retract(s_alt.x)  # type: ignore[union-attr]
        prob.pose_ref[vi] = ref_main
        pose_main = prob.view_pose(sol.x, vi)
        assert pose_main is not None
        chi_main = float(np.sum(prob._view_res(sol.x, vi) ** 2))
        chi_alt = float(s_alt.cost)
        ang, _ = pose_main.distance_to(pose_alt)
        if ang < np.deg2rad(1.0):
            amb[vi] = float("inf")  # both branches converged to the same pose
        elif chi_alt < chi_main:
            # the alternative is actually better: adopt it and report the margin
            prob.pose_ref[vi] = pose_alt
            sol.x[a:b] = 0.0
            amb[vi] = chi_main - chi_alt
        else:
            amb[vi] = chi_alt - chi_main
    if any(a is not None and np.isfinite(a) for a in amb):
        sol = lm_solve(prob, sol.x, max_iter=cfg.max_iterations)
    return prob, sol, amb


def _package(
    prob: _Problem,
    sol: Any,
    amb: list[float | None],
    cfg: PlanarConfig,
) -> PlanarSolution:
    x = sol.x
    layout = prob.layout(x)
    r, J = prob(x)
    dof = max(prob.n_res - prob.n_params, 0)
    chi2 = float(r @ r)
    red = chi2 / dof if dof > 0 else 0.0
    cov_scale = max(1.0, red)
    try:
        cov = np.linalg.inv(J.T @ J) * cov_scale
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J) * cov_scale
    views: dict[str, ViewSolution] = {}
    all_sq = []
    per_tag_sq: dict[int, list[float]] = {}
    for vi, v in enumerate(prob.views):
        H = normalize_h(prob.view_H(x, vi))
        sq = []
        for t in v.tags:
            e = apply_h(H, layout.corners(t.tag_id)) - t.corners
            s2 = float(np.mean(np.sum(e * e, axis=1)))
            sq.append(s2)
            per_tag_sq.setdefault(t.tag_id, []).append(s2)
        all_sq += sq
        pose = prob.view_pose(x, vi)
        if pose is None and v.camera is not None:
            pose = pose_from_homography(v.camera.without_distortion().K, H)
        vs = ViewSolution(
            frame_id=v.frame_id,
            mode=prob.modes[vi],
            H=H,
            rms_px=float(np.sqrt(np.mean(sq))),
            n_tags=len(v.tags),
            pose=pose,
            ambiguity_delta_chi2=None if amb[vi] is None else (None if not np.isfinite(amb[vi]) else float(amb[vi])),  # type: ignore[arg-type]
            slice=prob.view_slices[vi],
            norm_img=prob.Ti[vi],
            norm_plane=prob.Tp[vi],
            params=x[prob.view_slices[vi][0] : prob.view_slices[vi][1]].copy(),
            K=(v.camera.without_distortion().K if (prob.modes[vi] == "pose" and v.camera is not None) else None),
            pose_ref=prob.pose_ref[vi],
        )
        if amb[vi] is not None and np.isfinite(amb[vi]) and amb[vi] < cfg.ambiguity_min_delta_chi2:  # type: ignore[operator]
            vs.flags.append("pose_ambiguous")
        views[v.frame_id] = vs
    checks = {
        i: TagCheck(tag_id=i, n_views=len(per_tag_sq.get(i, [])), rms_px=float(np.sqrt(np.mean(per_tag_sq[i]))) if i in per_tag_sq else float("nan"))
        for i in layout.poses
    }
    return PlanarSolution(
        layout=layout,
        views=views,
        checks=checks,
        reduced_chi2=red,
        rms_px=float(np.sqrt(np.mean(all_sq))) if all_sq else 0.0,
        dof=dof,
        param_cov=cov,
        tag_slices=dict(prob.tag_slices),
        cov_scale=cov_scale,
    )


def _crossvalidate(views: list[PlanarView], sol: PlanarSolution, cfg: PlanarConfig, size_u: dict[int, float]) -> None:
    """Leave-one-tag-out size and shape test; updates ``sol.checks`` in place."""
    ids = sorted(sol.layout.poses)
    if len(ids) < 2:
        return
    for j in ids:
        sub_views = []
        hold: list[tuple[str, TagObs]] = []
        for v in views:
            rest = [t for t in v.tags if t.tag_id != j]
            mine = [t for t in v.tags if t.tag_id == j]
            if rest:
                sub_views.append(PlanarView(v.frame_id, rest, v.camera, v.mode))
                if mine:
                    hold.append((v.frame_id, mine[0]))
        if not hold:
            continue
        remaining = sorted({t.tag_id for v in sub_views for t in v.tags})
        gauge = sol.layout.gauge_id if sol.layout.gauge_id != j else choose_gauge(sub_views)
        init, unreachable = _pairwise_layout(sub_views, gauge)
        if any(i in unreachable for i in remaining):
            sub_views = [PlanarView(v.frame_id, [t for t in v.tags if t.tag_id in init], v.camera, v.mode) for v in sub_views]
            sub_views = [v for v in sub_views if v.tags]
        lay = TagLayout(gauge, init, {i: sol.layout.sizes[i] for i in init})
        try:
            prob, s, amb = _solve_once(sub_views, lay, cfg)
        except Exception:
            continue
        sub = _package(prob, s, amb, cfg)
        ratios, sigmas, shapes = [], [], []
        for fid, obs in hold:
            if fid not in sub.views:
                continue
            vs = sub.views[fid]
            Hinv = np.linalg.inv(vs.H)
            ratio, shape = backprojected_size(Hinv, obs.corners, obs.size_mm)

            # first-order uncertainty of the ratio: view parameters + held-out corners
            vi = [k for k, v in enumerate(sub_views) if v.frame_id == fid][0]
            a, b = prob.view_slices[vi]

            def f_params(p: np.ndarray, vi: int = vi, a: int = a, b: int = b) -> float:
                xx = s.x.copy()
                xx[a:b] = p
                return backprojected_size(np.linalg.inv(prob.view_H(xx, vi)), obs.corners, obs.size_mm)[0]

            g = np.empty(b - a)
            p0 = s.x[a:b].copy()
            for k in range(b - a):
                h = 1e-6 * max(1.0, abs(p0[k]))
                pp, pm = p0.copy(), p0.copy()
                pp[k] += h
                pm[k] -= h
                g[k] = (f_params(pp) - f_params(pm)) / (2 * h)
            var = float(g @ sub.param_cov[a:b, a:b] @ g)
            gc = np.empty(8)
            c0 = obs.corners.ravel().copy()
            for k in range(8):
                cp, cm = c0.copy(), c0.copy()
                cp[k] += 1e-4
                cm[k] -= 1e-4
                gc[k] = (
                    backprojected_size(Hinv, cp.reshape(4, 2), obs.size_mm)[0]
                    - backprojected_size(Hinv, cm.reshape(4, 2), obs.size_mm)[0]
                ) / 2e-4
            var += float(gc @ obs.cov @ gc)
            ratios.append(ratio)
            sigmas.append(np.sqrt(max(var, 0.0)))
            shapes.append(shape)
        if not ratios:
            continue
        w = 1.0 / np.maximum(np.array(sigmas), 1e-9) ** 2
        ratio = float(np.sum(w * np.array(ratios)) / np.sum(w))
        sigma_stat = float(1.0 / np.sqrt(np.sum(w)))
        # declared sizes are themselves uncertain (held-out tag and the others)
        u_rel = size_u.get(j, 0.0) / sol.layout.sizes[j]
        u_others = float(np.sqrt(np.mean([(size_u.get(i, 0.0) / sol.layout.sizes[i]) ** 2 for i in remaining]))) if remaining else 0.0
        sigma = float(np.sqrt(sigma_stat**2 + u_rel**2 + u_others**2))
        chk = sol.checks[j]
        chk.checked = True
        chk.size_ratio = ratio
        chk.size_ratio_sigma = sigma
        chk.shape_rms_frac = float(np.median(shapes))
        err = abs(ratio - 1.0)
        if err > cfg.crossval_max_frac or (err > cfg.crossval_min_frac and err > cfg.crossval_z * sigma):
            chk.consistent = False
            chk.reasons.append("size_mismatch")
        if chk.shape_rms_frac > cfg.crossval_max_shape_frac:
            chk.consistent = False
            chk.reasons.append("not_square_in_plane")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def solve_planar(
    views: list[PlanarView],
    *,
    gauge: int | None = None,
    cfg: PlanarConfig | None = None,
    size_u_mm: dict[int, float] | None = None,
    crossvalidate: bool = True,
    exclude_inconsistent: bool = True,
) -> PlanarSolution:
    """Jointly estimate per-view plane mappings and the in-plane tag layout.

    ``size_u_mm`` gives the standard uncertainty of each declared tag size; it
    only widens the cross-validation test (it is propagated to results in
    :mod:`lumen.policy.uncertainty`).
    """
    cfg = cfg or PlanarConfig()
    views = [v for v in views if v.tags]
    if not views:
        raise GeometryError(
            "no_tags",
            "No accepted AprilTag is available to establish scale.",
            suggestions=["Place at least one measured AprilTag on the flat part of the panel and retake the photo."],
        )
    size_u = dict(size_u_mm or {})
    excluded: dict[int, list[str]] = {}
    flags: list[str] = []

    while True:
        g = gauge if (gauge is not None and any(gauge in v.ids() for v in views)) else choose_gauge(views)
        init, unreachable = _pairwise_layout(views, g)
        for i in unreachable:
            excluded[i] = ["not_connected_to_gauge"]
        if unreachable:
            views = [PlanarView(v.frame_id, [t for t in v.tags if t.tag_id in init], v.camera, v.mode) for v in views]
            dropped = [v.frame_id for v in views if not v.tags]
            views = [v for v in views if v.tags]
            if dropped:
                flags.append("views_without_connected_tags")
        sizes = {t.tag_id: t.size_mm for v in views for t in v.tags}
        layout0 = TagLayout(g, {i: init[i] for i in sizes}, sizes)
        prob, s, amb = _solve_once(views, layout0, cfg)
        sol = _package(prob, s, amb, cfg)
        if crossvalidate:
            _crossvalidate(views, sol, cfg, size_u)
        for c in sol.checks.values():
            if np.isfinite(c.rms_px) and c.rms_px > cfg.max_tag_rms_px:
                c.consistent = False
                c.reasons.append("high_residual")
        bad = [c for c in sol.checks.values() if not c.consistent]
        if not bad or not exclude_inconsistent or len(sol.layout.poses) < 3:
            break
        # With three or more tags the majority identifies the outlier: remove
        # the worst offender and solve again.  With two tags nobody can tell
        # which one is wrong, so the inconsistency is reported and kept.
        def badness(c: TagCheck) -> float:
            b = abs(c.size_ratio - 1.0) / cfg.crossval_min_frac if np.isfinite(c.size_ratio) else 0.0
            if np.isfinite(c.shape_rms_frac):
                b = max(b, c.shape_rms_frac / cfg.crossval_max_shape_frac)
            if np.isfinite(c.rms_px):
                b = max(b, c.rms_px / cfg.max_tag_rms_px)
            return b

        worst = max(bad, key=badness)
        excluded[worst.tag_id] = list(worst.reasons)
        views = [PlanarView(v.frame_id, [t for t in v.tags if t.tag_id != worst.tag_id], v.camera, v.mode) for v in views]
        views = [v for v in views if v.tags]
        if not views:
            raise GeometryError("no_consistent_tags", "The tags are mutually inconsistent; no reliable scale.")
        if gauge == worst.tag_id:
            gauge = None

    sol.excluded_tags = excluded
    sol.flags += flags
    if sol.dof > 0 and sol.reduced_chi2 > cfg.max_reduced_chi2:
        sol.flags.append("high_chi2")
    if not sol.consistent:
        sol.flags.append("tags_inconsistent")
    if any("pose_ambiguous" in v.flags for v in sol.views.values()):
        sol.flags.append("pose_ambiguous")
    if len(sol.layout.poses) < 2:
        sol.flags.append("single_tag_no_independent_check")
    return sol


@dataclass
class ViewLocalization:
    """Mapping of one new frame onto an existing tag layout."""

    frame_id: str
    mode: str
    H: np.ndarray
    cov: np.ndarray  # covariance of the view parameters
    rms_px: float
    reduced_chi2: float
    n_tags: int
    used_ids: list[int]
    pose: Pose | None
    flags: list[str]
    norm_img: np.ndarray
    norm_plane: np.ndarray
    params: np.ndarray
    tag_rms_px: dict[int, float] = field(default_factory=dict)
    K: np.ndarray | None = None
    pose_ref: Pose | None = None
    #: ``d params / d (cx, cy, theta)`` of each non-gauge layout tag used here
    layout_sens: dict[int, np.ndarray] = field(default_factory=dict)

    def H_from_params(self, p: np.ndarray) -> np.ndarray:
        if self.mode == "homography":
            Hn = np.append(np.asarray(p, dtype=float), 1.0).reshape(3, 3)
            return np.linalg.inv(self.norm_img) @ Hn @ self.norm_plane
        assert self.K is not None and self.pose_ref is not None
        return homography_from_pose(self.K, self.pose_ref.retract(np.asarray(p, dtype=float)))


def localize_view(
    layout: TagLayout,
    view: PlanarView,
    *,
    cfg: PlanarConfig | None = None,
) -> ViewLocalization:
    """Register one frame against a *fixed* tag layout (tracking, verification).

    Tags of the frame that are not part of the layout are ignored.  The result
    reports per-tag residuals so the caller can detect moved markers.
    """
    cfg = cfg or PlanarConfig()
    tags = [t for t in view.tags if t.tag_id in layout.poses]
    if not tags:
        raise GeometryError(
            "no_reference_tags",
            "None of the registered AprilTags is visible in this image.",
            suggestions=["Bring at least one of the session's tags fully into view."],
        )
    v = PlanarView(view.frame_id, tags, view.camera, view.mode)
    prob, s, amb = _solve_once([v], layout, cfg, fixed_layout=True)
    sol = _package(prob, s, amb, cfg)
    vs = sol.views[v.frame_id]
    a, b = prob.view_slices[0]
    tag_rms = {}
    for t in tags:
        e = apply_h(vs.H, layout.corners(t.tag_id)) - t.corners
        tag_rms[t.tag_id] = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
    flags = list(vs.flags)
    if sol.dof > 0 and sol.reduced_chi2 > cfg.max_reduced_chi2:
        flags.append("high_chi2")
    if any(r > cfg.max_tag_rms_px for r in tag_rms.values()):
        flags.append("high_residual")
    params = s.x[a:b].copy()
    # sensitivity of this localisation to the layout it was given:
    # d p / d layout = -(Jv^T Jv)^-1 Jv^T Jl
    sens: dict[int, np.ndarray] = {}
    free_ids = [t.tag_id for t in tags if t.tag_id != layout.gauge_id]
    if free_ids:
        _, Jfull = prob(s.x)
        Jv = Jfull[:, a:b]
        Jl = prob.layout_jacobian(s.x, 0, free_ids)
        try:
            G = -np.linalg.solve(Jv.T @ Jv, Jv.T @ Jl)
        except np.linalg.LinAlgError:
            G = -np.linalg.pinv(Jv.T @ Jv) @ (Jv.T @ Jl)
        for k, tid in enumerate(free_ids):
            sens[tid] = G[:, 3 * k : 3 * k + 3]
    loc = ViewLocalization(
        frame_id=v.frame_id,
        mode=vs.mode,
        H=vs.H,
        cov=sol.param_cov[a:b, a:b],
        rms_px=vs.rms_px,
        reduced_chi2=sol.reduced_chi2,
        n_tags=len(tags),
        used_ids=[t.tag_id for t in tags],
        pose=vs.pose,
        flags=flags,
        norm_img=prob.Ti[0],
        norm_plane=prob.Tp[0],
        params=params,
        tag_rms_px=tag_rms,
        K=(v.camera.without_distortion().K if (vs.mode == "pose" and v.camera is not None) else None),
        pose_ref=prob.pose_ref[0],
        layout_sens=sens,
    )
    return loc
