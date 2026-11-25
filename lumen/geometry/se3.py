"""Rigid transforms on SO(3)/SE(3).

Naming rule used everywhere: ``T_a_b`` (a :class:`Pose`) maps coordinates from
frame ``b`` into frame ``a``:  ``x_a = T_a_b.R @ x_b + T_a_b.t``.
Composition therefore reads naturally: ``T_a_c = T_a_b @ T_b_c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

_EPS = 1e-12


def skew(w: np.ndarray) -> np.ndarray:
    """Cross-product matrix ``[w]_x``."""
    x, y, z = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues formula: rotation vector -> rotation matrix."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-8:
        # Second-order Taylor expansion keeps the result orthonormal to ~1e-16.
        return np.eye(3) + W + 0.5 * (W @ W)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * W + b * (W @ W)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (angle in [0, pi])."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-5:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I.
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        # Fix signs relative to the largest component.
        signs = np.sign(A[k])
        signs[k] = 1.0
        axis = axis * np.where(signs == 0.0, 1.0, signs)
        axis /= max(np.linalg.norm(axis), _EPS)
        # Resolve the overall sign with the (tiny) antisymmetric part.
        anti = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(anti @ axis) < 0.0:
            axis = -axis
        return axis * theta
    s = 2.0 * np.sin(theta)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    return axis * theta


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, radians in [0, pi]."""
    return float(np.arccos(np.clip((np.trace(np.asarray(R, dtype=float)) - 1.0) * 0.5, -1.0, 1.0)))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Closest rotation matrix in the Frobenius sense (det = +1)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class Pose:
    """Rigid transform ``x_a = R @ x_b + t`` (see module docstring for naming)."""

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", np.asarray(self.R, dtype=float).reshape(3, 3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(3))

    # -- constructors -----------------------------------------------------
    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3), np.zeros(3))

    @staticmethod
    def from_rvec(rvec: Iterable[float], t: Iterable[float]) -> "Pose":
        return Pose(exp_so3(np.asarray(list(rvec), dtype=float)), np.asarray(list(t), dtype=float))

    @staticmethod
    def from_matrix(M: np.ndarray) -> "Pose":
        M = np.asarray(M, dtype=float)
        return Pose(M[:3, :3], M[:3, 3])

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Pose":
        return Pose(np.asarray(d["R"], dtype=float), np.asarray(d["t"], dtype=float))

    # -- algebra ----------------------------------------------------------
    def inv(self) -> "Pose":
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Transform an ``(N, 3)`` (or ``(3,)``) array of points."""
        p = np.asarray(pts, dtype=float)
        return p @ self.R.T + self.t

    def rotate(self, vecs: np.ndarray) -> np.ndarray:
        """Rotate direction vectors (no translation)."""
        return np.asarray(vecs, dtype=float) @ self.R.T

    def retract(self, xi: np.ndarray) -> "Pose":
        """Left-multiplicative update ``exp(xi) * self`` with ``xi = (w, v)``.

        The rotation part uses the exponential map; the translation is updated
        additively in the destination frame, which keeps the Jacobian simple
        and is a valid retraction on SE(3).
        """
        xi = np.asarray(xi, dtype=float).reshape(6)
        dR = exp_so3(xi[:3])
        return Pose(dR @ self.R, dR @ self.t + xi[3:])

    # -- views ------------------------------------------------------------
    @property
    def rvec(self) -> np.ndarray:
        return log_so3(self.R)

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    @property
    def center(self) -> np.ndarray:
        """Origin of frame ``a`` expressed in frame ``b`` (e.g. camera centre in
        world coordinates when this pose is ``T_cam_world``)."""
        return -self.R.T @ self.t

    def to_dict(self) -> dict[str, Any]:
        return {"R": self.R.tolist(), "t": self.t.tolist()}

    def distance_to(self, other: "Pose") -> tuple[float, float]:
        """(rotation angle [rad], translation distance) between two poses."""
        d = self.inv() @ other
        return rotation_angle(d.R), float(np.linalg.norm(d.t))


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray, roll: float = 0.0) -> Pose:
    """Camera pose ``T_world_cam`` for an OpenCV camera (x right, y down, z forward).

    ``up`` is a hint for the direction that should appear upward in the image;
    ``roll`` rotates the camera about its optical axis (positive = clockwise
    scene rotation in the image is *not* implied; it is a right-handed rotation
    about +z of the camera).
    """
    eye = np.asarray(eye, dtype=float)
    f = np.asarray(target, dtype=float) - eye
    f /= max(np.linalg.norm(f), _EPS)
    upv = np.asarray(up, dtype=float)
    r = np.cross(f, upv)
    if np.linalg.norm(r) < 1e-9:
        # Looking along the up hint: pick any perpendicular.
        alt = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        r = np.cross(f, alt)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # image "down"
    R_wc = np.stack([r, d, f], axis=1)
    if roll:
        R_wc = R_wc @ rot_z(roll)
    return Pose(R_wc, eye)
