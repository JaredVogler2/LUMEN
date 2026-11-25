"""Rectified ("orthographic") view of the target plane.

The rectified image is a metric picture of the plane: one pixel is a fixed
number of millimetres everywhere.  Only things that really lie *on* the plane
are shown at their true position; anything off the plane (the floor, the
curved edge, a trim strip) is drawn where its viewing ray hits the plane, i.e.
displaced by parallax.  The UI says so, and :mod:`lumen.vision.datums` uses
exactly this displacement to tell on-plane seams from silhouettes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, local_scale_px_per_mm, to_h


@dataclass(frozen=True)
class RectSpec:
    """Geometry of a rectified view.

    ``A`` maps homogeneous rectified pixel coordinates ``(col, row, 1)`` (pixel
    centres at integers) to plane millimetres.  Rows grow *down* on screen
    while the view's "up" direction is ``e_up`` in the plane.
    """

    width: int
    height: int
    mm_per_px: float
    origin: tuple[float, float]  # plane coordinates of the centre of pixel (0, 0)
    e_right: tuple[float, float]
    e_up: tuple[float, float]

    @property
    def A(self) -> np.ndarray:
        s = self.mm_per_px
        er = np.asarray(self.e_right)
        eu = np.asarray(self.e_up)
        return np.array(
            [
                [s * er[0], -s * eu[0], self.origin[0]],
                [s * er[1], -s * eu[1], self.origin[1]],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def Ainv(self) -> np.ndarray:
        return np.linalg.inv(self.A)

    def rect_to_plane(self, px: np.ndarray) -> np.ndarray:
        return (to_h(px) @ self.A.T)[..., :2]

    def plane_to_rect(self, q: np.ndarray) -> np.ndarray:
        return (to_h(q) @ self.Ainv.T)[..., :2]

    @property
    def mirrored(self) -> bool:
        """True if the view shows the plane frame mirrored (it never should for
        the gauge frame; it does when a left-handed local frame is shown with
        ``+x`` to the right)."""
        er = np.asarray(self.e_right)
        eu = np.asarray(self.e_up)
        return bool(er[0] * eu[1] - er[1] * eu[0] < 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mm_per_px": self.mm_per_px,
            "origin": list(self.origin),
            "e_right": list(self.e_right),
            "e_up": list(self.e_up),
            "A": self.A.tolist(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RectSpec":
        return RectSpec(
            int(d["width"]),
            int(d["height"]),
            float(d["mm_per_px"]),
            (float(d["origin"][0]), float(d["origin"][1])),
            (float(d["e_right"][0]), float(d["e_right"][1])),
            (float(d["e_up"][0]), float(d["e_up"][1])),
        )


def image_footprint_on_plane(
    H: np.ndarray,
    width: int,
    height: int,
    ref_plane_pt: np.ndarray,
    samples: int = 24,
    min_scale_frac: float = 0.12,
) -> np.ndarray:
    """Plane points of the image border that are usably resolved.

    Border points beyond the plane's vanishing line, or where the image scale
    has dropped below ``min_scale_frac`` of the scale at ``ref_plane_pt``
    (grazing view, far away), are discarded.
    """
    Hinv = np.linalg.inv(H)
    t = np.linspace(0.0, 1.0, samples)
    xs = t * (width - 1)
    ys = t * (height - 1)
    border = np.concatenate(
        [
            np.stack([xs, np.zeros_like(xs)], axis=1),
            np.stack([np.full_like(ys, width - 1), ys], axis=1),
            np.stack([xs[::-1], np.full_like(xs, height - 1)], axis=1),
            np.stack([np.zeros_like(ys), ys[::-1]], axis=1),
        ]
    )
    # also sample the interior so a footprint cut by the horizon keeps its bulk
    gx, gy = np.meshgrid(np.linspace(0, width - 1, 9), np.linspace(0, height - 1, 9))
    pts = np.concatenate([border, np.stack([gx.ravel(), gy.ravel()], axis=1)])
    ph = to_h(pts) @ Hinv.T
    ref_w = float((np.append(apply_h(H, ref_plane_pt[None])[0], 1.0)) @ Hinv[2])
    sign = np.sign(ref_w) or 1.0
    ok = (ph[:, 2] * sign) > 1e-9
    q = ph[ok, :2] / ph[ok, 2:3]
    if q.shape[0] == 0:
        return q
    ref_scale = float(local_scale_px_per_mm(H, ref_plane_pt[None])[0])
    sc = local_scale_px_per_mm(H, q)
    return q[sc >= min_scale_frac * ref_scale]


def plan_rectification(
    H: np.ndarray,
    width: int,
    height: int,
    tag_corners_plane: np.ndarray,
    *,
    up_hint: np.ndarray | None = None,
    right_hint: np.ndarray | None = None,
    margin_mm: float = 450.0,
    max_long_edge_px: int = 1800,
    mm_per_px: float | None = None,
    include_plane_pts: np.ndarray | None = None,
) -> RectSpec:
    """Choose the extent, resolution and orientation of the rectified view.

    Without hints the view is oriented like the photo (image "up" at the tag
    centroid stays up).  ``up_hint``/``right_hint`` are plane directions used
    once a local frame exists (``+z`` up).
    """
    ctr = tag_corners_plane.mean(axis=0)
    if up_hint is None:
        u0 = apply_h(H, ctr[None])[0]
        Hinv = np.linalg.inv(H)
        q1 = apply_h(Hinv, (u0 + np.array([0.0, -5.0]))[None])[0]
        up = q1 - ctr
    else:
        up = np.asarray(up_hint, dtype=float)
    up = up / max(np.linalg.norm(up), 1e-12)
    if right_hint is None:
        right = np.array([up[1], -up[0]])  # clockwise from up: unmirrored view of a y-up frame
    else:
        right = np.asarray(right_hint, dtype=float)
        right = right - (right @ up) * up
        right /= max(np.linalg.norm(right), 1e-12)

    foot = image_footprint_on_plane(H, width, height, ctr)
    pts = tag_corners_plane if foot.shape[0] == 0 else np.concatenate([foot, tag_corners_plane])
    a = (pts - ctr) @ right
    b = (pts - ctr) @ up
    ta = (tag_corners_plane - ctr) @ right
    tb = (tag_corners_plane - ctr) @ up
    lo_a = max(a.min(), ta.min() - margin_mm)
    hi_a = min(a.max(), ta.max() + margin_mm)
    lo_b = max(b.min(), tb.min() - margin_mm)
    hi_b = min(b.max(), tb.max() + margin_mm)
    if include_plane_pts is not None and len(include_plane_pts):
        ia = (np.asarray(include_plane_pts) - ctr) @ right
        ib = (np.asarray(include_plane_pts) - ctr) @ up
        pad = 30.0
        lo_a, hi_a = min(lo_a, ia.min() - pad), max(hi_a, ia.max() + pad)
        lo_b, hi_b = min(lo_b, ib.min() - pad), max(hi_b, ib.max() + pad)

    if mm_per_px is None:
        px_per_mm = float(np.median(local_scale_px_per_mm(H, tag_corners_plane)))
        mm_per_px = 1.0 / max(px_per_mm, 1e-6)
    w_mm, h_mm = hi_a - lo_a, hi_b - lo_b
    long_px = max(w_mm, h_mm) / mm_per_px
    if long_px > max_long_edge_px:
        mm_per_px = max(w_mm, h_mm) / max_long_edge_px
    W = max(int(np.ceil(w_mm / mm_per_px)), 16)
    Hh = max(int(np.ceil(h_mm / mm_per_px)), 16)
    # centre of pixel (0, 0) is the top-left: a = lo_a + s/2, b = hi_b - s/2
    origin = ctr + (lo_a + 0.5 * mm_per_px) * right + (hi_b - 0.5 * mm_per_px) * up
    return RectSpec(W, Hh, float(mm_per_px), (float(origin[0]), float(origin[1])), (float(right[0]), float(right[1])), (float(up[0]), float(up[1])))


def rectify_image(
    bgr: np.ndarray,
    camera: CameraModel | None,
    H: np.ndarray,
    spec: RectSpec,
    fill: tuple[int, int, int] = (24, 24, 24),
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the photo into the rectified view.

    Returns ``(rectified BGR, valid mask)``; ``valid`` is False where the plane
    point is outside the photo or beyond the vanishing line.
    """
    cols, rows = np.meshgrid(np.arange(spec.width, dtype=np.float64), np.arange(spec.height, dtype=np.float64))
    px = np.stack([cols.ravel(), rows.ravel()], axis=1)
    q = spec.rect_to_plane(px)
    ph = to_h(q) @ np.asarray(H, dtype=float).T
    ctr_w = float(np.median(ph[:, 2]))
    sign = np.sign(ctr_w) or 1.0
    okw = (ph[:, 2] * sign) > 1e-9
    w = np.where(okw, ph[:, 2], 1.0)
    uv = ph[:, :2] / w[:, None]
    if camera is not None and camera.has_distortion:
        uv = camera.distort_pixels(uv)
    Himg, Wimg = bgr.shape[:2]
    inside = okw & (uv[:, 0] >= 0) & (uv[:, 0] <= Wimg - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= Himg - 1)
    mapx = np.where(inside, uv[:, 0], -1).astype(np.float32).reshape(spec.height, spec.width)
    mapy = np.where(inside, uv[:, 1], -1).astype(np.float32).reshape(spec.height, spec.width)
    out = cv2.remap(bgr, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, inside.reshape(spec.height, spec.width)
