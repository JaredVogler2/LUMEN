"""Planar projective geometry.

A homography here always maps *plane millimetres* to *ideal image pixels*
(``x_img ~ H @ [X, Y, 1]``) unless a function says otherwise.  "Ideal" means
lens distortion has already been removed.
"""

from __future__ import annotations

import numpy as np

from .se3 import Pose, orthonormalize


def to_h(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to ``(N, 2)`` points (no sign checks)."""
    p = to_h(pts) @ np.asarray(H, dtype=float).T
    w = p[..., 2:3]
    w = np.where(np.abs(w) < 1e-300, 1e-300, w)
    return p[..., :2] / w


def apply_h_checked(H: np.ndarray, pts: np.ndarray, w_ref_sign: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``H`` and also report which points keep a consistent sign of ``w``.

    A sign flip of the homogeneous coordinate means the point lies beyond the
    plane's vanishing line (behind the camera); such results are geometrically
    meaningless and must not be drawn.
    """
    p = to_h(pts) @ np.asarray(H, dtype=float).T
    w = p[..., 2]
    if w_ref_sign is None:
        w_ref_sign = float(np.sign(np.median(w))) or 1.0
    ok = (w * w_ref_sign) > 1e-12
    ws = np.where(np.abs(w) < 1e-300, 1e-300, w)
    return p[..., :2] / ws[..., None], ok


def normalization_transform(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centroid to origin, mean distance sqrt(2)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def dlt_homography(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalised DLT estimate of ``H`` with ``dst ~ H @ src`` (N >= 4).

    Returns ``H`` scaled so that ``H[2, 2] == 1`` when that is well defined,
    otherwise unit Frobenius norm.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("dlt_homography needs >= 4 matched points")
    Ts = normalization_transform(src)
    Td = normalization_transform(dst)
    s = apply_h(Ts, src)
    d = apply_h(Td, dst)
    n = s.shape[0]
    A = np.zeros((2 * n, 9))
    x, y = s[:, 0], s[:, 1]
    u, v = d[:, 0], d[:, 1]
    A[0::2, 0:3] = np.stack([x, y, np.ones(n)], axis=1)
    A[0::2, 6:9] = -u[:, None] * np.stack([x, y, np.ones(n)], axis=1)
    A[1::2, 3:6] = np.stack([x, y, np.ones(n)], axis=1)
    A[1::2, 6:9] = -v[:, None] * np.stack([x, y, np.ones(n)], axis=1)
    if weights is not None:
        w = np.repeat(np.asarray(weights, dtype=float).reshape(-1), 2)
        A = A * w[:, None]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return normalize_h(H)


def normalize_h(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    if abs(H[2, 2]) > 1e-12 * np.linalg.norm(H):
        return H / H[2, 2]
    return H / np.linalg.norm(H)


def h_local_jacobian(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``d(u, v)/d(X, Y)`` of the homography at each point, ``(N, 2, 2)``."""
    H = np.asarray(H, dtype=float)
    p = to_h(pts) @ H.T
    w = p[..., 2]
    u = p[..., 0] / w
    v = p[..., 1] / w
    J = np.empty(p.shape[:-1] + (2, 2))
    J[..., 0, 0] = (H[0, 0] - u * H[2, 0]) / w
    J[..., 0, 1] = (H[0, 1] - u * H[2, 1]) / w
    J[..., 1, 0] = (H[1, 0] - v * H[2, 0]) / w
    J[..., 1, 1] = (H[1, 1] - v * H[2, 1]) / w
    return J


def local_scale_px_per_mm(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Geometric-mean image scale (px/mm) of the mapping at each plane point."""
    J = h_local_jacobian(H, pts)
    det = np.abs(J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0])
    return np.sqrt(det)


def homography_from_pose(K: np.ndarray, T_cam_plane: Pose) -> np.ndarray:
    """Homography of the plane ``Z = 0`` of the plane frame: ``K [r1 r2 t]``."""
    R, t = T_cam_plane.R, T_cam_plane.t
    return np.asarray(K, dtype=float) @ np.stack([R[:, 0], R[:, 1], t], axis=1)


def pose_from_homography(K: np.ndarray, H: np.ndarray) -> Pose:
    """Zhang's decomposition ``H ~ K [r1 r2 t]`` -> ``T_cam_plane``.

    The scale and sign are chosen so the plane origin is in front of the
    camera; the rotation is projected onto SO(3).  This is an *initial* value:
    callers refine it by minimising reprojection error.
    """
    B = np.linalg.inv(np.asarray(K, dtype=float)) @ np.asarray(H, dtype=float)
    n1 = np.linalg.norm(B[:, 0])
    n2 = np.linalg.norm(B[:, 1])
    lam = 2.0 / (n1 + n2)
    if B[2, 2] * lam < 0:
        lam = -lam
    r1 = B[:, 0] * lam
    r2 = B[:, 1] * lam
    t = B[:, 2] * lam
    r3 = np.cross(r1, r2)
    R = orthonormalize(np.stack([r1, r2, r3], axis=1))
    return Pose(R, t)


def vanishing_line_in_plane(H: np.ndarray) -> np.ndarray:
    """Line ``l`` (plane coordinates, ``l @ [X, Y, 1] = 0``) that maps to infinity."""
    return np.asarray(H, dtype=float)[2].copy()


def rigid_2d(theta: float, tx: float, ty: float) -> np.ndarray:
    """3x3 homogeneous matrix of a planar rigid motion."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def fit_rigid_2d(src: np.ndarray, dst: np.ndarray, allow_scale: bool = False) -> tuple[float, np.ndarray, float, float]:
    """Least-squares planar rigid (or similarity) fit ``dst ≈ s R(theta) src + t``.

    Returns ``(theta, t, scale, rms)``.  With ``allow_scale=False`` the scale is
    fixed to 1 (Kabsch); with ``True`` it is the Umeyama similarity scale.
    Reflections are never allowed.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - cs, dst - cd
    # theta maximises sum b · R a
    sxx = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    sxy = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    theta = float(np.arctan2(sxy, sxx))
    scale = 1.0
    if allow_scale:
        denom = float(np.sum(a * a))
        scale = float(np.hypot(sxx, sxy) / denom) if denom > 1e-18 else 1.0
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = cd - scale * (R @ cs)
    res = dst - (scale * (src @ R.T) + t)
    rms = float(np.sqrt(np.mean(np.sum(res * res, axis=1))))
    return theta, t, scale, rms
