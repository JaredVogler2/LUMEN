"""Pinhole camera with Brown–Conrady distortion (OpenCV 5-parameter layout).

    x = X/Z, y = Y/Z                                   (normalised, undistorted)
    r2 = x^2 + y^2
    radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
    xd = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
    yd = y radial + p1 (r2 + 2 y^2) + 2 p2 x y
    u = fx xd + cx,  v = fy yd + cy                    (pixel-centre convention)

The implementation is plain NumPy so that Jacobians, inversion and uncertainty
propagation are under our control and behave identically across OpenCV builds.
``tests/unit/test_camera.py`` cross-checks it against ``cv2.projectPoints``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    #: Free-text provenance, e.g. "charuco", "self_calibration", "exif_prior".
    source: str = field(default="unknown", compare=False)

    # ------------------------------------------------------------------ basics
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def Kinv(self) -> np.ndarray:
        return np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def dist_array(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=float)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist_array) > 0.0))

    @staticmethod
    def from_fov(width: int, height: int, hfov_deg: float, dist: tuple[float, ...] | None = None) -> "CameraModel":
        f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
        d = tuple(float(x) for x in (dist or (0, 0, 0, 0, 0)))
        d = (d + (0.0,) * 5)[:5]
        # Principal point at the image centre in pixel-centre coordinates.
        return CameraModel(width, height, float(f), float(f), (width - 1) * 0.5, (height - 1) * 0.5, d)  # type: ignore[arg-type]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CameraModel":
        dist = tuple(float(x) for x in d.get("dist", (0, 0, 0, 0, 0)))
        dist = (dist + (0.0,) * 5)[:5]
        return CameraModel(
            int(d["width"]),
            int(d["height"]),
            float(d["fx"]),
            float(d["fy"]),
            float(d["cx"]),
            float(d["cy"]),
            dist,  # type: ignore[arg-type]
            str(d.get("source", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist": list(self.dist),
            "source": self.source,
        }

    def scaled(self, sx: float, sy: float | None = None) -> "CameraModel":
        """Model for the same camera after resizing the image by ``(sx, sy)``.

        With the pixel-centre convention a resize by ``s`` maps the continuous
        coordinate ``u + 0.5`` to ``s (u + 0.5)``, hence ``cx' = s (cx + 0.5) - 0.5``.
        Distortion coefficients act on normalised coordinates and are unchanged.
        """
        sy = sx if sy is None else sy
        return replace(
            self,
            width=int(round(self.width * sx)),
            height=int(round(self.height * sy)),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=sx * (self.cx + 0.5) - 0.5,
            cy=sy * (self.cy + 0.5) - 0.5,
        )

    def without_distortion(self) -> "CameraModel":
        return replace(self, dist=(0.0, 0.0, 0.0, 0.0, 0.0))

    # ------------------------------------------------------------- distortion
    def distort_normalized(self, xn: np.ndarray) -> np.ndarray:
        """Apply lens distortion to normalised points ``(N, 2)``."""
        xn = np.asarray(xn, dtype=float)
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[..., 0], xn[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([xd, yd], axis=-1)

    def _distort_jacobian(self, xn: np.ndarray) -> np.ndarray:
        """``d(xd, yd)/d(x, y)`` as an ``(N, 2, 2)`` array."""
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[..., 0], xn[..., 1]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dradial = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)  # d radial / d r2
        J = np.empty(xn.shape[:-1] + (2, 2))
        J[..., 0, 0] = radial + 2.0 * x * x * dradial + 2.0 * p1 * y + 6.0 * p2 * x
        J[..., 0, 1] = 2.0 * x * y * dradial + 2.0 * p1 * x + 2.0 * p2 * y
        J[..., 1, 0] = 2.0 * x * y * dradial + 2.0 * p1 * x + 2.0 * p2 * y
        J[..., 1, 1] = radial + 2.0 * y * y * dradial + 6.0 * p1 * y + 2.0 * p2 * x
        return J

    def undistort_normalized(self, xd: np.ndarray, iterations: int = 20, tol: float = 1e-12) -> np.ndarray:
        """Invert the distortion with Newton iterations (vectorised)."""
        xd = np.asarray(xd, dtype=float)
        if not self.has_distortion:
            return xd.copy()
        x = xd.copy()
        for _ in range(iterations):
            f = self.distort_normalized(x) - xd
            J = self._distort_jacobian(x)
            det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dx = (J[..., 1, 1] * f[..., 0] - J[..., 0, 1] * f[..., 1]) / det
            dy = (-J[..., 1, 0] * f[..., 0] + J[..., 0, 0] * f[..., 1]) / det
            x = x - np.stack([dx, dy], axis=-1)
            if float(np.max(np.abs(f))) < tol:
                break
        return x

    # ------------------------------------------------------------- projection
    def normalized_to_pixels(self, xn: np.ndarray, apply_distortion: bool = True) -> np.ndarray:
        xn = np.asarray(xn, dtype=float)
        xd = self.distort_normalized(xn) if (apply_distortion and self.has_distortion) else xn
        return np.stack([self.fx * xd[..., 0] + self.cx, self.fy * xd[..., 1] + self.cy], axis=-1)

    def pixels_to_normalized(self, uv: np.ndarray, undistort: bool = True) -> np.ndarray:
        """Pixel coordinates -> normalised image coordinates.

        With ``undistort=True`` (default) the result is the ideal pinhole
        direction ``(x, y, 1)``; this is the function used to turn a pixel into
        a viewing ray.
        """
        uv = np.asarray(uv, dtype=float)
        xd = np.stack([(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy], axis=-1)
        return self.undistort_normalized(xd) if (undistort and self.has_distortion) else xd

    def project(self, X_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points ``(N, 3)``.  Returns ``(uv, in_front)``."""
        X = np.asarray(X_cam, dtype=float)
        z = X[..., 2]
        in_front = z > 1e-9
        zs = np.where(in_front, z, 1.0)
        xn = np.stack([X[..., 0] / zs, X[..., 1] / zs], axis=-1)
        uv = self.normalized_to_pixels(xn)
        return uv, in_front

    def undistort_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Distorted pixels -> "ideal" pixels of the same K without distortion."""
        if not self.has_distortion:
            return np.asarray(uv, dtype=float).copy()
        return self.normalized_to_pixels(self.pixels_to_normalized(uv), apply_distortion=False)

    def distort_pixels(self, uv_ideal: np.ndarray) -> np.ndarray:
        """Ideal (undistorted) pixels -> pixels as recorded by the lens."""
        if not self.has_distortion:
            return np.asarray(uv_ideal, dtype=float).copy()
        uv_ideal = np.asarray(uv_ideal, dtype=float)
        xn = np.stack(
            [(uv_ideal[..., 0] - self.cx) / self.fx, (uv_ideal[..., 1] - self.cy) / self.fy], axis=-1
        )
        return self.normalized_to_pixels(xn, apply_distortion=True)

    def in_image(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        return (
            (uv[..., 0] >= -0.5 + margin)
            & (uv[..., 0] <= self.width - 0.5 - margin)
            & (uv[..., 1] >= -0.5 + margin)
            & (uv[..., 1] <= self.height - 0.5 - margin)
        )

    # --------------------------------------------------------------- queries
    @property
    def hfov(self) -> float:
        return float(2.0 * np.arctan(0.5 * self.width / self.fx))

    @property
    def vfov(self) -> float:
        return float(2.0 * np.arctan(0.5 * self.height / self.fy))

    def max_distortion_px(self, samples: int = 9) -> float:
        """Largest displacement caused by distortion over the image (pixels)."""
        if not self.has_distortion:
            return 0.0
        us = np.linspace(0, self.width - 1, samples)
        vs = np.linspace(0, self.height - 1, samples)
        g = np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2)
        return float(np.max(np.linalg.norm(self.undistort_pixels(g) - g, axis=1)))
