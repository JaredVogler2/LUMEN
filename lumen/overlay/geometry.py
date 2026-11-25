"""Overlay primitives in image pixel coordinates.

The browser never invents geometry.  Every line it draws — the metric grid,
the datum lines, the axes, the placard outline, the dimension leaders — is
computed here by pushing *plane* geometry through the estimated mapping of the
frame (homography, then the lens model) and is returned as polylines in the
frame's own pixel coordinates (pixel-centre convention).  The client only
applies the display transform of its viewer.

A primitive is ``{"kind": str, "points": [[u, v], ...], "closed": bool,
"label": str | None, ...}``.  Kinds are semantic (``grid_minor``,
``grid_major``, ``datum_floor``, ``target`` ...) so styling stays in CSS/JS.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.homography import to_h
from ..geometry.lines import Line2D
from ..geometry.localframe import LocalFrame, RectTarget
from ..vision.rectify import RectSpec, image_footprint_on_plane

PlaneToView = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def image_mapper(H: np.ndarray, camera: CameraModel | None, width: int, height: int, ref_plane_pt: np.ndarray) -> PlaneToView:
    """Plane mm -> raw image pixels, with a validity mask (in front of the
    camera and inside a generous border around the image)."""
    H = np.asarray(H, dtype=float)
    ref = to_h(np.asarray(ref_plane_pt, dtype=float)[None]) @ H.T
    sign = float(np.sign(ref[0, 2])) or 1.0

    def f(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ph = to_h(np.asarray(q, dtype=float)) @ H.T
        ok = (ph[..., 2] * sign) > 1e-9
        w = np.where(ok, ph[..., 2], 1.0)
        uv = ph[..., :2] / w[..., None]
        pad = 0.25 * max(width, height)
        ok &= (uv[..., 0] > -pad) & (uv[..., 0] < width + pad) & (uv[..., 1] > -pad) & (uv[..., 1] < height + pad)
        if camera is not None and camera.has_distortion:
            uv = np.where(ok[..., None], camera.distort_pixels(np.where(ok[..., None], uv, 0.0)), uv)
        return uv, ok

    return f


def rect_mapper(spec: RectSpec) -> PlaneToView:
    def f(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = spec.plane_to_rect(np.asarray(q, dtype=float))
        return px, np.ones(px.shape[:-1], dtype=bool)

    return f


def _clip_segment(p0: np.ndarray, p1: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang–Barsky clip of a segment to the image rectangle [-0.5, w-0.5] x [-0.5, h-0.5]."""
    x0, y0 = p0
    dx, dy = p1 - p0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 + 0.5), (dx, w - 0.5 - x0), (-dy, y0 + 0.5), (dy, h - 0.5 - y0)):
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
    return p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0)


def _polyline(mapper: PlaneToView, q: np.ndarray, w: int, h: int) -> list[list[list[float]]]:
    """Map a plane polyline and split it into pieces that are valid and inside the view."""
    uv, ok = mapper(q)
    pieces: list[list[list[float]]] = []
    cur: list[list[float]] = []
    for i in range(len(uv) - 1):
        if not (ok[i] and ok[i + 1]):
            if len(cur) >= 2:
                pieces.append(cur)
            cur = []
            continue
        c = _clip_segment(uv[i], uv[i + 1], w, h)
        if c is None:
            if len(cur) >= 2:
                pieces.append(cur)
            cur = []
            continue
        a, b = c
        if cur and np.allclose(cur[-1], a, atol=1e-6):
            cur.append([float(b[0]), float(b[1])])
        else:
            if len(cur) >= 2:
                pieces.append(cur)
            cur = [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]
        if not np.allclose(b, uv[i + 1], atol=1e-6):
            pieces.append(cur)
            cur = []
    if len(cur) >= 2:
        pieces.append(cur)
    return pieces


def _nice_step(target_mm: float) -> float:
    """Round a spacing to 1/2/2.5/5 x 10^n millimetres."""
    if target_mm <= 0 or not np.isfinite(target_mm):
        return 50.0
    exp = np.floor(np.log10(target_mm))
    base = target_mm / 10**exp
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if base <= m:
            return float(m * 10**exp)
    return float(10.0 * 10**exp)


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def build_overlay(
    *,
    mapper: PlaneToView,
    view_w: int,
    view_h: int,
    px_per_mm: float,
    visible_plane_pts: np.ndarray,
    frame: LocalFrame | None,
    floor: Line2D | None = None,
    side: Line2D | None = None,
    targets: list[tuple[str, RectTarget, dict[str, Any]]] | None = None,
    tags: list[dict[str, Any]] | None = None,
    grid: bool = True,
    max_extent_mm: float = 2500.0,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble overlay primitives for one view (photo or rectified).

    ``visible_plane_pts`` are plane points known to be inside the view (image
    footprint); they bound the grid.  ``px_per_mm`` is a representative scale
    used only to pick a readable grid spacing.
    """
    prims: list[dict[str, Any]] = []
    labels = labels or {}
    step = 0.0

    def add_lines(kind: str, q: np.ndarray, **extra: Any) -> None:
        for piece in _polyline(mapper, q, view_w, view_h):
            prims.append({"kind": kind, "points": piece, "closed": False, **extra})

    # --- tags (always shown: they are the evidence of scale)
    for t in tags or []:
        uv, ok = mapper(np.asarray(t["corners_plane"], dtype=float)) if "corners_plane" in t else (np.asarray(t["corners_px"], dtype=float), np.ones(4, bool))
        if np.all(ok):
            prims.append(
                {
                    "kind": "tag" if t.get("accepted", True) else "tag_rejected",
                    "points": np.asarray(uv).round(2).tolist(),
                    "closed": True,
                    "label": f"ID {t['tag_id']}",
                }
            )

    if frame is not None and frame.valid and len(visible_plane_pts) >= 3:
        loc = frame.to_local(visible_plane_pts)
        lo = np.maximum(loc.min(axis=0), -max_extent_mm)
        hi = np.minimum(loc.max(axis=0), max_extent_mm)
        if grid:
            step = _nice_step(55.0 / max(px_per_mm, 1e-6))
            # keep the number of lines bounded
            while (hi[0] - lo[0]) / step > 60 or (hi[1] - lo[1]) / step > 60:
                step = _nice_step(step * 2.01)
            xs = np.arange(np.ceil(lo[0] / step), np.floor(hi[0] / step) + 1) * step
            zs = np.arange(np.ceil(lo[1] / step), np.floor(hi[1] / step) + 1) * step
            n = 48
            for x in xs:
                seg = np.stack([np.full(n, x), np.linspace(lo[1], hi[1], n)], axis=1)
                k = int(round(x / step))
                kind = "grid_axis" if k == 0 else ("grid_major" if k % 5 == 0 else "grid_minor")
                add_lines(kind, frame.to_plane(seg), axis="x", value=float(x), label=_fmt(x) if k % 5 == 0 else None)
            for z in zs:
                seg = np.stack([np.linspace(lo[0], hi[0], n), np.full(n, z)], axis=1)
                k = int(round(z / step))
                kind = "grid_axis" if k == 0 else ("grid_major" if k % 5 == 0 else "grid_minor")
                add_lines(kind, frame.to_plane(seg), axis="z", value=float(z), label=_fmt(z) if k % 5 == 0 else None)

        # axes arrows from the origin
        L = max(60.0, 2.0 * (step or 50.0))
        o = frame.origin
        for kind, d, name in (("axis_x", frame.ex, "+x"), ("axis_z", frame.ez, "+z")):
            add_lines(kind, np.stack([o, o + L * d / np.linalg.norm(d)]), label=name, arrow=True)
        uv, ok = mapper(o[None])
        if ok[0]:
            prims.append({"kind": "origin", "points": [uv[0].round(2).tolist()], "closed": False, "label": "0"})

    # --- datum lines (drawn even when the frame is invalid: they are the evidence)
    if len(visible_plane_pts) >= 2:
        lo_p = visible_plane_pts.min(axis=0) - 50.0
        hi_p = visible_plane_pts.max(axis=0) + 50.0
        for kind, line, key in (("datum_floor", floor, "floor"), ("datum_side", side, "side")):
            if line is None:
                continue
            seg = line.clip_to_box(lo_p[0], lo_p[1], hi_p[0], hi_p[1])
            if seg is None:
                continue
            q = seg[0] + np.linspace(0.0, 1.0, 32)[:, None] * (seg[1] - seg[0])
            add_lines(kind, q, label=labels.get(key))

    # --- targets
    for name, rect, meta in targets or []:
        ring = np.concatenate([rect.corners, rect.corners[:1]])
        # densify so lens distortion bends the outline correctly
        dense = np.concatenate([ring[i] + np.linspace(0, 1, 9)[:-1, None] * (ring[i + 1] - ring[i]) for i in range(4)])
        uv, ok = mapper(dense)
        cuv, cok = mapper(rect.corners)
        if np.all(ok):
            prims.append(
                {
                    "kind": meta.get("kind", "target"),
                    "points": uv.round(2).tolist(),
                    "corners": cuv.round(2).tolist(),
                    "closed": True,
                    "label": name,
                    **{k: v for k, v in meta.items() if k != "kind"},
                }
            )
            ruv, rok = mapper(rect.reference[None])
            if rok[0]:
                prims.append({"kind": "target_ref", "points": [ruv[0].round(2).tolist()], "closed": False, "label": meta.get("reference_point")})
        else:
            prims.append({"kind": "target_out_of_view", "points": [], "closed": False, "label": name})
        if frame is not None and frame.valid and meta.get("dimensions", True):
            ref_l = frame.to_local(rect.reference)
            # leader along +x on the floor datum, then up to the reference point
            a = frame.to_plane(np.array([0.0, ref_l[1]]))
            b = rect.reference
            c = frame.to_plane(np.array([ref_l[0], 0.0]))
            add_lines("dim", np.stack([a, b]), label=f"x {ref_l[0]:.1f} mm", axis="x")
            add_lines("dim", np.stack([c, b]), label=f"z {ref_l[1]:.1f} mm", axis="z")

    return {"primitives": prims, "grid_step_mm": step, "view": {"width": view_w, "height": view_h}}


def footprint_for_image(H: np.ndarray, width: int, height: int, ref_plane_pt: np.ndarray) -> np.ndarray:
    return image_footprint_on_plane(H, width, height, ref_plane_pt, samples=20, min_scale_frac=0.15)
