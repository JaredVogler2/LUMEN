"""2-D lines in the measurement plane (millimetres) or in an image (pixels).

A :class:`Line2D` is stored in Hesse normal form ``n · x = d`` with ``|n| = 1``.
The pair ``(n, d)`` and ``(-n, -d)`` describe the same undirected line; the
*direction* ``t = (-n_y, n_x)`` gives the line an orientation, which LUMEN uses
to carry sign semantics (which side is "into the panel").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Line2D:
    n: np.ndarray  # unit normal (2,)
    d: float

    def __post_init__(self) -> None:
        n = np.asarray(self.n, dtype=float).reshape(2)
        norm = float(np.linalg.norm(n))
        if norm < 1e-15:
            raise ValueError("degenerate line normal")
        object.__setattr__(self, "n", n / norm)
        object.__setattr__(self, "d", float(self.d) / norm)

    # -- constructors -----------------------------------------------------
    @staticmethod
    def from_points(p0: np.ndarray, p1: np.ndarray) -> "Line2D":
        p0 = np.asarray(p0, dtype=float).reshape(2)
        p1 = np.asarray(p1, dtype=float).reshape(2)
        t = p1 - p0
        if float(np.linalg.norm(t)) < 1e-12:
            raise ValueError("line needs two distinct points")
        n = np.array([-t[1], t[0]])
        return Line2D(n, float(n @ p0))

    @staticmethod
    def from_point_direction(p: np.ndarray, t: np.ndarray) -> "Line2D":
        p = np.asarray(p, dtype=float).reshape(2)
        t = np.asarray(t, dtype=float).reshape(2)
        n = np.array([-t[1], t[0]])
        return Line2D(n, float(n @ p))

    @staticmethod
    def from_homogeneous(l: np.ndarray) -> "Line2D":
        a, b, c = np.asarray(l, dtype=float).reshape(3)
        return Line2D(np.array([a, b]), -c)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Line2D":
        return Line2D(np.asarray(d["n"], dtype=float), float(d["d"]))

    # -- views ------------------------------------------------------------
    @property
    def direction(self) -> np.ndarray:
        """Unit direction ``t`` with ``n = rot90ccw(t)`` i.e. n is to the left of t."""
        return np.array([self.n[1], -self.n[0]])

    @property
    def angle(self) -> float:
        """Direction angle in (-pi, pi]."""
        t = self.direction
        return float(np.arctan2(t[1], t[0]))

    @property
    def homogeneous(self) -> np.ndarray:
        return np.array([self.n[0], self.n[1], -self.d])

    @property
    def point(self) -> np.ndarray:
        """Point of the line closest to the origin."""
        return self.n * self.d

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n.tolist(), "d": self.d}

    # -- geometry ---------------------------------------------------------
    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance (positive on the side the normal points to)."""
        return np.asarray(pts, dtype=float) @ self.n - self.d

    def project(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        return pts - np.multiply.outer(self.signed_distance(pts), self.n)

    def flipped(self) -> "Line2D":
        return Line2D(-self.n, -self.d)

    def oriented_towards(self, pt: np.ndarray) -> "Line2D":
        """Same line with the normal pointing to the side that contains ``pt``."""
        return self if float(self.signed_distance(np.asarray(pt, dtype=float))) >= 0 else self.flipped()

    def offset(self, dist: float) -> "Line2D":
        """Parallel line shifted by ``dist`` along the normal."""
        return Line2D(self.n, self.d + dist)

    def intersect(self, other: "Line2D", min_sin: float = 1e-9) -> np.ndarray | None:
        """Intersection point or ``None`` when the lines are (nearly) parallel."""
        A = np.stack([self.n, other.n])
        det = float(np.linalg.det(A))
        if abs(det) < min_sin:
            return None
        return np.linalg.solve(A, np.array([self.d, other.d]))

    def angle_to(self, other: "Line2D") -> float:
        """Unsigned acute angle between two undirected lines, in [0, pi/2]."""
        c = abs(float(self.n @ other.n))
        return float(np.arccos(np.clip(c, 0.0, 1.0)))

    def transformed(self, M: np.ndarray) -> "Line2D":
        """Line after applying the point transform ``M`` (3x3: affine or projective)."""
        l = np.linalg.inv(np.asarray(M, dtype=float)).T @ self.homogeneous
        return Line2D.from_homogeneous(l)

    def clip_to_box(self, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray | None:
        """Segment (2x2) of the line inside an axis-aligned box, or ``None``."""
        t = self.direction
        p = self.point
        ts: list[float] = []
        for axis, lo, hi in ((0, xmin, xmax), (1, ymin, ymax)):
            if abs(t[axis]) < 1e-12:
                if not (lo - 1e-9 <= p[axis] <= hi + 1e-9):
                    return None
                continue
            for bound in (lo, hi):
                s = (bound - p[axis]) / t[axis]
                q = p + s * t
                o = 1 - axis
                lo_o, hi_o = (ymin, ymax) if axis == 0 else (xmin, xmax)
                if lo_o - 1e-6 <= q[o] <= hi_o + 1e-6:
                    ts.append(float(s))
        if len(ts) < 2:
            return None
        s0, s1 = min(ts), max(ts)
        if s1 - s0 < 1e-9:
            return None
        return np.stack([p + s0 * t, p + s1 * t])


@dataclass(frozen=True)
class LineFit:
    """Result of a total-least-squares line fit with first-order uncertainty.

    The uncertainty is parametrised about the centroid of the support:
    ``sigma_perp`` is the standard deviation of the line's perpendicular
    position at the centroid and ``sigma_angle`` that of its direction.  At a
    point at signed along-line distance ``s`` from the centroid the
    perpendicular standard deviation is ``sqrt(sigma_perp^2 + s^2 sigma_angle^2)``
    (the two are uncorrelated in this parametrisation).
    """

    line: Line2D
    centroid: np.ndarray
    extent: tuple[float, float]  # along-line coordinate range of the support
    rms: float
    n_points: int
    sigma_perp: float
    sigma_angle: float

    def sigma_at(self, pts: np.ndarray) -> np.ndarray:
        s = (np.asarray(pts, dtype=float) - self.centroid) @ self.line.direction
        return np.sqrt(self.sigma_perp**2 + (s * self.sigma_angle) ** 2)

    @property
    def endpoints(self) -> np.ndarray:
        t = self.line.direction
        c = self.line.project(self.centroid)
        return np.stack([c + self.extent[0] * t, c + self.extent[1] * t])


def fit_line_tls(
    pts: np.ndarray,
    weights: np.ndarray | None = None,
    sigma_point: float | None = None,
) -> LineFit:
    """Weighted total-least-squares line through ``(N, 2)`` points.

    ``sigma_point`` is the a-priori perpendicular standard deviation of one
    point; when omitted the residual scatter is used (never smaller than what
    the residuals show when both are available).
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        raise ValueError("fit_line_tls needs at least two points")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
    w = np.clip(w, 0.0, None)
    wsum = float(w.sum())
    if wsum <= 0:
        raise ValueError("all weights are zero")
    c = (pts * w[:, None]).sum(axis=0) / wsum
    q = pts - c
    C = (q * w[:, None]).T @ q / wsum
    evals, evecs = np.linalg.eigh(C)
    t = evecs[:, 1]  # direction of largest spread
    nrm = np.array([-t[1], t[0]])
    line = Line2D(nrm, float(nrm @ c))
    t = line.direction
    s = q @ t
    r = q @ line.n
    dof = max(n - 2, 1)
    rms = float(np.sqrt(np.sum(w * r * r) / wsum))
    sig_res = float(np.sqrt(np.sum(w * r * r) / wsum * n / dof))
    sig = sig_res if sigma_point is None else max(float(sigma_point), sig_res)
    # Effective number of points and along-line spread for the weighted fit.
    n_eff = wsum * wsum / float(np.sum(w * w))
    s_var = float(np.sum(w * s * s) / wsum)
    sigma_perp = sig / np.sqrt(max(n_eff, 1.0))
    sigma_angle = sig / np.sqrt(max(n_eff * s_var, 1e-18))
    return LineFit(
        line=line,
        centroid=c,
        extent=(float(s.min()), float(s.max())),
        rms=rms,
        n_points=n,
        sigma_perp=float(sigma_perp),
        sigma_angle=float(sigma_angle),
    )


def polygon_area(poly: np.ndarray) -> float:
    """Signed area (positive for counter-clockwise order in a y-up frame)."""
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd rule point-in-polygon test."""
    x, y = float(pt[0]), float(pt[1])
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    inside = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside
