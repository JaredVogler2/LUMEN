"""Deterministic CPU ray caster for synthetic panels.

This module is deliberately *independent* of the estimation code: it has its
own pinhole/distortion maths (pixel -> ray via Newton inversion, never
``cv2.projectPoints``) and intersects analytic surfaces in 3-D.  A bug in the
estimators therefore cannot hide behind a matching bug here, and
``tests/unit/test_synth.py`` cross-checks the two implementations.

Imaging pipeline (all in linear light until the last step):

    radiance -> exposure/white balance -> vignette -> pixel integration
    (adaptive supersampling) -> PSF blur -> motion blur -> shot/read noise
    -> clip -> sRGB transfer -> optional ISP sharpening -> 8 bit -> optional JPEG
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .scene import Box, CameraSpec, CurvedEdge, Placard, Scene, TagSticker
from .tags import CELLS, tag_bits

_INF = np.float64(np.inf)
_Y = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------
# Independent camera maths
# --------------------------------------------------------------------------
def _distort(x: np.ndarray, y: np.ndarray, dist: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _undistort(xd: np.ndarray, yd: np.ndarray, dist: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not any(dist):
        return xd, yd
    x, y = xd.copy(), yd.copy()
    h = 1e-6
    for _ in range(25):
        fx0, fy0 = _distort(x, y, dist)
        ex, ey = fx0 - xd, fy0 - yd
        # Numerical 2x2 Jacobian (independent of the analytic one in lumen.geometry).
        ax, ay = _distort(x + h, y, dist)
        bx, by = _distort(x, y + h, dist)
        j00, j10 = (ax - fx0) / h, (ay - fy0) / h
        j01, j11 = (bx - fx0) / h, (by - fy0) / h
        det = j00 * j11 - j01 * j10
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        x = x - (j11 * ex - j01 * ey) / det
        y = y - (-j10 * ex + j00 * ey) / det
        if float(np.max(np.abs(ex)) + np.max(np.abs(ey))) < 1e-13:
            break
    return x, y


def camera_matrices(cam: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R_wc, C)`` as arrays."""
    return np.asarray(cam.R_wc, dtype=float), np.asarray(cam.C, dtype=float)


def project_points(cam: CameraSpec, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels with the forward distortion model.

    Returns ``(uv, depth)``; ``depth <= 0`` means behind the camera.
    """
    R_wc, C = camera_matrices(cam)
    Xc = (np.asarray(Xw, dtype=float) - C) @ R_wc  # == R_wc.T @ (X - C)
    z = Xc[..., 2]
    zs = np.where(z > 1e-9, z, 1.0)
    xd, yd = _distort(Xc[..., 0] / zs, Xc[..., 1] / zs, cam.dist)
    return np.stack([cam.fx * xd + cam.cx, cam.fy * yd + cam.cy], axis=-1), z


def pixel_rays(cam: CameraSpec, uv: np.ndarray) -> np.ndarray:
    """Unit world-space ray directions through pixel coordinates ``(N, 2)``."""
    R_wc, _ = camera_matrices(cam)
    xd = (uv[:, 0] - cam.cx) / cam.fx
    yd = (uv[:, 1] - cam.cy) / cam.fy
    x, y = _undistort(xd, yd, cam.dist)
    d = np.stack([x, y, np.ones_like(x)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d @ R_wc.T


# --------------------------------------------------------------------------
# Procedural texture
# --------------------------------------------------------------------------
class _ValueNoise:
    """Tileable smooth value noise in [-1, 1], deterministic from the seed."""

    def __init__(self, seed: int, size: int = 128) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.table = rng.uniform(-1.0, 1.0, size=(size, size))

    def __call__(self, u: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
        s = self.size
        x = u / scale
        y = v / scale
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = x - x0
        fy = y - y0
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        i0 = np.mod(x0, s).astype(np.intp)
        j0 = np.mod(y0, s).astype(np.intp)
        i1 = (i0 + 1) % s
        j1 = (j0 + 1) % s
        t = self.table
        a = t[j0, i0] * (1 - fx) + t[j0, i1] * fx
        b = t[j1, i0] * (1 - fx) + t[j1, i1] * fx
        return a * (1 - fy) + b * fy


# --------------------------------------------------------------------------
# Edge / panel geometry helpers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _EdgeGeom:
    """Derived quantities of a :class:`CurvedEdge` (see scene.py)."""

    side: float  # +1 for the left edge, -1 for the right edge
    P_t: np.ndarray  # tangent point at Z = 0
    a: np.ndarray  # axis / tangent-line direction (unit)
    e_in: np.ndarray  # in-plane unit vector pointing into the flat region
    R: float
    phi_max: float
    return_length: float

    @staticmethod
    def make(edge: CurvedEdge, side: float) -> "_EdgeGeom":
        psi = np.deg2rad(edge.taper_deg)
        a = np.array([np.sin(psi), 0.0, np.cos(psi)])
        e_in = side * np.array([np.cos(psi), 0.0, -np.sin(psi)])
        return _EdgeGeom(
            side=side,
            P_t=np.array([edge.position, 0.0, 0.0]),
            a=a,
            e_in=e_in,
            R=float(edge.radius),
            phi_max=float(np.deg2rad(edge.sweep_deg)),
            return_length=float(edge.return_length),
        )

    # in-plane coordinate of a flat point relative to the tangent line
    def inward(self, P: np.ndarray) -> np.ndarray:
        return (P - self.P_t) @ self.e_in

    def surface_point(self, s_in: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """3-D point and outward normal for unrolled coordinates.

        ``s_in`` is the signed unrolled distance from the tangent line along
        ``e_in`` (>= 0 flat, < 0 on the bend / return flange); ``w`` is the
        coordinate along the axis direction.
        """
        s_in = np.asarray(s_in, dtype=float)
        w = np.asarray(w, dtype=float)
        base = self.P_t + np.multiply.outer(w, self.a)
        arc = np.clip(-s_in, 0.0, self.R * self.phi_max)
        phi = arc / self.R
        beyond = np.clip(-s_in - self.R * self.phi_max, 0.0, None)
        flat = np.clip(s_in, 0.0, None)
        # flat part + arc + straight return
        P = (
            base
            + np.multiply.outer(flat - self.R * np.sin(phi), self.e_in)
            + np.multiply.outer(self.R * (1.0 - np.cos(phi)), _Y)
            + np.multiply.outer(-beyond * np.cos(self.phi_max), self.e_in)
            + np.multiply.outer(beyond * np.sin(self.phi_max), _Y)
        )
        N = np.multiply.outer(-np.sin(phi), self.e_in) + np.multiply.outer(-np.cos(phi), _Y)
        return P, N

    def unrolled_xz(self, s_in: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Panel surface (unrolled sheet) coordinates of ``(s_in, w)``."""
        su = self.P_t[0] + w * self.a[0] + s_in * self.e_in[0]
        sv = self.P_t[2] + w * self.a[2] + s_in * self.e_in[2]
        return su, sv

    def from_unrolled(self, su: np.ndarray, sv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = np.asarray(su, dtype=float) - self.P_t[0]
        dz = np.asarray(sv, dtype=float) - self.P_t[2]
        s_in = dx * self.e_in[0] + dz * self.e_in[2]
        w = dx * self.a[0] + dz * self.a[2]
        return s_in, w


class SceneGeometry:
    """Analytic geometry + materials of a :class:`Scene` (shared by the ray
    caster and by ground-truth queries)."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        p = scene.panel
        self.left = _EdgeGeom.make(p.left_edge, +1.0) if p.left_edge else None
        self.right = _EdgeGeom.make(p.right_edge, -1.0) if p.right_edge else None
        beta = np.deg2rad(scene.floor.dihedral_deg)
        self.floor_dir = np.array([0.0, -np.sin(beta), np.cos(beta)])
        self.floor_normal = np.array([0.0, np.cos(beta), np.sin(beta)])
        self.noise = _ValueNoise(scene.texture_seed)
        self.noise2 = _ValueNoise(scene.texture_seed + 101)
        boxes = list(scene.boxes)
        if scene.trim is not None:
            t = scene.trim
            boxes.append(Box((t.x_min, -t.thickness, 0.0), (t.x_max, 0.0, t.height), t.albedo))
        self.boxes = boxes
        self.shims = [s for s in scene.stickers if s.host == "panel" and s.lift > 0.0]
        self.panel_stickers = [s for s in scene.stickers if s.host == "panel" and s.lift <= 0.0]
        self.floor_stickers = [s for s in scene.stickers if s.host == "floor"]

    # ---------------------------------------------------------------- truth
    def panel_point(self, su: np.ndarray, sv: np.ndarray) -> np.ndarray:
        """World point of panel surface (unrolled) coordinates, any region."""
        su = np.atleast_1d(np.asarray(su, dtype=float))
        sv = np.atleast_1d(np.asarray(sv, dtype=float))
        P = np.stack([su, np.zeros_like(su), sv], axis=1)
        for eg in (self.left, self.right):
            if eg is None:
                continue
            s_in, w = eg.from_unrolled(su, sv)
            m = s_in < 0.0
            if np.any(m):
                Pm, _ = eg.surface_point(s_in[m], w[m])
                P[m] = Pm
        return P

    def sticker_corners_world(self, s: TagSticker) -> np.ndarray:
        """True 3-D positions of the four detection corners (TL, TR, BR, BL)."""
        h = 0.5 * s.size
        local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
        c, sn = np.cos(np.deg2rad(s.rotation_deg)), np.sin(np.deg2rad(s.rotation_deg))
        uv = local @ np.array([[c, sn], [-sn, c]]) + np.asarray(s.center, dtype=float)
        if s.host == "floor":
            return np.multiply.outer(uv[:, 0], np.array([1.0, 0.0, 0.0])) + np.multiply.outer(
                uv[:, 1], self.floor_dir
            )
        if s.lift > 0.0:
            return np.stack([uv[:, 0], np.full(4, -s.lift), uv[:, 1]], axis=1)
        return self.panel_point(uv[:, 0], uv[:, 1])

    def placard_corners_world(self, pl: Placard) -> np.ndarray:
        hw, hh = 0.5 * pl.width, 0.5 * pl.height
        local = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])
        c, sn = np.cos(np.deg2rad(pl.rotation_deg)), np.sin(np.deg2rad(pl.rotation_deg))
        uv = local @ np.array([[c, sn], [-sn, c]]) + np.asarray(pl.center, dtype=float)
        return self.panel_point(uv[:, 0], uv[:, 1])

    # ------------------------------------------------------------ materials
    def _sticker_paint(
        self, a: np.ndarray, b: np.ndarray, s: TagSticker
    ) -> tuple[np.ndarray, np.ndarray]:
        """(inside mask, scalar albedo) for sticker-local coordinates."""
        h = 0.5 * s.size
        inside = (np.abs(a) <= h + s.margin) & (b <= h + s.margin) & (b >= -h - s.margin_bottom)
        alb = np.full(a.shape, s.white)
        core = inside & (np.abs(a) <= h) & (np.abs(b) <= h)
        if np.any(core):
            cell = s.size / CELLS
            col = np.clip(np.floor((a[core] + h) / cell).astype(np.intp), 0, CELLS - 1)
            row = np.clip(np.floor((h - b[core]) / cell).astype(np.intp), 0, CELLS - 1)
            bits = tag_bits(s.tag_id)
            alb[core] = np.where(bits[row, col], s.white, s.black)
        cell_mm = s.size / CELLS
        if s.label and s.margin_bottom >= cell_mm + 3.0:
            # Row of small dark blocks imitating a printed ID under the tag.  It
            # stays outside the one-cell quiet zone that the tag design requires.
            y0 = -h - s.margin_bottom + 1.0
            y1 = -h - cell_mm - 0.5
            band = inside & (b >= y0) & (b <= y1) & (np.abs(a) <= h * 0.8)
            if np.any(band):
                k = np.floor((a[band] + h * 0.8) / 3.0).astype(np.int64)
                on = ((k * 2654435761 + s.tag_id * 40503) >> 3) % 3 != 0
                frac = np.mod(a[band] + h * 0.8, 3.0)
                alb[band] = np.where(on & (frac < 2.0), 0.12, s.white)
        return inside, alb

    def panel_material(
        self, su: np.ndarray, sv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Albedo ``(N, 3)``, specular strength ``(N,)`` and shininess ``(N,)``."""
        sc = self.scene
        p = sc.panel
        n = su.shape[0]
        alb = np.empty((n, 3))
        alb[:] = np.asarray(p.albedo)
        if p.texture_contrast > 0.0:
            tex = 0.6 * self.noise(su, sv, p.texture_scale) + 0.4 * self.noise2(su, sv, p.texture_scale * 0.37)
            alb *= (1.0 + p.texture_contrast * tex)[:, None]
        ks = np.full(n, p.spec_strength)
        shin = np.full(n, p.shininess)
        for seam in sc.seams:
            p0 = np.asarray(seam.p0, dtype=float)
            p1 = np.asarray(seam.p1, dtype=float)
            t = p1 - p0
            L = float(np.linalg.norm(t))
            t /= L
            rel_u = su - p0[0]
            rel_v = sv - p0[1]
            along = rel_u * t[0] + rel_v * t[1]
            perp = -rel_u * t[1] + rel_v * t[0]
            on = (along >= 0.0) & (along <= L)
            if seam.soft > 0.0:
                prof = np.exp(-0.5 * (perp / seam.soft) ** 2)
            else:
                prof = (np.abs(perp) <= 0.5 * seam.width).astype(float)
            alb *= (1.0 - seam.darkness * prof * on)[:, None]
        for pl in sc.placards:
            c, s = np.cos(np.deg2rad(pl.rotation_deg)), np.sin(np.deg2rad(pl.rotation_deg))
            du = su - pl.center[0]
            dv = sv - pl.center[1]
            a = du * c + dv * s
            b = -du * s + dv * c
            hw, hh = 0.5 * pl.width, 0.5 * pl.height
            inside = (np.abs(a) <= hw) & (np.abs(b) <= hh)
            if not np.any(inside):
                continue
            if pl.corner_radius > 0.0:
                r = pl.corner_radius
                qx = np.abs(a) - (hw - r)
                qy = np.abs(b) - (hh - r)
                inside &= ~((qx > 0) & (qy > 0) & (qx * qx + qy * qy > r * r))
            flap = np.zeros(n, dtype=bool)
            if pl.curl is not None:
                k, leg = pl.curl
                sx = (-1.0, 1.0, 1.0, -1.0)[k]
                sy = (1.0, 1.0, -1.0, -1.0)[k]
                da = hw - sx * a  # distance from the curled corner along a
                db = hh - sy * b
                peeled = inside & (da + db < leg)
                inside &= ~peeled
                # folded flap lies mirrored across the fold line da + db = leg
                flap = inside & (da + db < 2.0 * leg) & (da < leg) & (db < leg)
            ia = np.nonzero(inside)[0]
            aa, bb = a[ia], b[ia]
            col = np.empty((ia.size, 3))
            col[:] = np.asarray(pl.albedo)
            edge = (hw - np.abs(aa) < pl.border) | (hh - np.abs(bb) < pl.border)
            col[edge] = np.asarray(pl.border_albedo)
            if pl.text_lines > 0:
                usable = 2.0 * (hh - pl.border) * 0.8
                pitch = usable / pl.text_lines
                rel = bb + usable * 0.5
                li = np.floor(rel / pitch)
                within = (rel >= 0) & (rel < usable) & (np.mod(rel, pitch) < pitch * 0.45)
                stop = hw - pl.border - pl.width * (0.08 + 0.12 * np.mod(li, 3))
                text = within & (aa > -(hw - pl.border) + pl.width * 0.06) & (aa < stop) & ~edge
                col[text] = np.asarray(pl.text_albedo)
            if np.any(flap):
                col[flap[ia]] = (0.55, 0.55, 0.52)
            alb[ia] = col
            ks[ia] = pl.spec_strength
            shin[ia] = pl.shininess
        for st in self.panel_stickers:
            c, s = np.cos(np.deg2rad(st.rotation_deg)), np.sin(np.deg2rad(st.rotation_deg))
            du = su - st.center[0]
            dv = sv - st.center[1]
            a = du * c + dv * s
            b = -du * s + dv * c
            inside, val = self._sticker_paint(a, b, st)
            if np.any(inside):
                alb[inside] = val[inside, None]
                ks[inside] = st.spec_strength
                shin[inside] = st.shininess
        return alb, ks, shin

    def floor_material(self, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self.scene.floor
        n = x.shape[0]
        alb = np.empty((n, 3))
        alb[:] = np.asarray(f.albedo)
        if f.texture_contrast > 0.0:
            tex = 0.55 * self.noise(x + 311.0, t + 97.0, f.texture_scale) + 0.45 * self.noise2(
                x, t, f.texture_scale * 0.31
            )
            alb *= (1.0 + f.texture_contrast * tex)[:, None]
        ks = np.full(n, f.spec_strength)
        shin = np.full(n, f.shininess)
        for st in self.floor_stickers:
            c, s = np.cos(np.deg2rad(st.rotation_deg)), np.sin(np.deg2rad(st.rotation_deg))
            du = x - st.center[0]
            dv = t - st.center[1]
            a = du * c + dv * s
            b = -du * s + dv * c
            inside, val = self._sticker_paint(a, b, st)
            if np.any(inside):
                alb[inside] = val[inside, None]
                ks[inside] = st.spec_strength
                shin[inside] = st.shininess
        return alb, ks, shin


# --------------------------------------------------------------------------
# Ray caster
# --------------------------------------------------------------------------
_ID_VOID, _ID_PANEL, _ID_CYL_L, _ID_CYL_R, _ID_RET_L, _ID_RET_R = 0, 1, 2, 3, 4, 5
_ID_FLOOR, _ID_BG, _ID_BACK = 6, 7, 8
_ID_SHIM0 = 20
_ID_BOX0 = 60


def _plane_t(o: np.ndarray, d: np.ndarray, p0: np.ndarray, n: np.ndarray) -> np.ndarray:
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p0 - o) @ n) / denom
    t = np.where((np.abs(denom) > 1e-12) & (t > 1e-6), t, _INF)
    return t


def _box_t(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slab intersection.  Returns (t_near or inf, axis index of the entry face)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    tmin = np.where(np.isnan(tmin), -_INF, tmin)
    tmax = np.where(np.isnan(tmax), _INF, tmax)
    axis = np.argmax(tmin, axis=1)
    tn = np.max(tmin, axis=1)
    tf = np.min(tmax, axis=1)
    hit = (tf >= tn) & (tn > 1e-6)
    return np.where(hit, tn, _INF), axis


class _Tracer:
    def __init__(self, geom: SceneGeometry, cam: CameraSpec) -> None:
        self.g = geom
        self.cam = cam
        self.R_wc, self.C = camera_matrices(cam)
        sc = geom.scene
        self.flash_pos = self.C + self.R_wc @ np.asarray(sc.light.flash_offset_cam, dtype=float)
        self.optical_axis = self.R_wc[:, 2]

    # -- intersections ---------------------------------------------------
    def _cyl(self, d: np.ndarray, eg: _EdgeGeom) -> tuple[np.ndarray, np.ndarray]:
        """Nearest valid hit on the bend.  Returns (t, is_back_face)."""
        p = self.g.scene.panel
        A0 = eg.P_t + eg.R * _Y
        oc = self.C - A0
        oc_p = oc - (oc @ eg.a) * eg.a
        d_p = d - np.multiply.outer(d @ eg.a, eg.a)
        qa = np.einsum("ij,ij->i", d_p, d_p)
        qb = 2.0 * (d_p @ oc_p)
        qc = float(oc_p @ oc_p) - eg.R * eg.R
        disc = qb * qb - 4.0 * qa * qc
        ok = (disc > 0.0) & (qa > 1e-14)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        qa_s = np.where(ok, qa, 1.0)
        best = np.full(d.shape[0], _INF)
        back = np.zeros(d.shape[0], dtype=bool)
        for k, sign in enumerate((-1.0, 1.0)):
            t = (-qb + sign * sq) / (2.0 * qa_s)
            valid = ok & (t > 1e-6)
            P = self.C + t[:, None] * d
            rel = P - A0
            sin_phi = -(rel @ eg.e_in) / eg.R
            cos_phi = -(rel @ _Y) / eg.R
            phi = np.arctan2(sin_phi, cos_phi)
            valid &= (phi >= 0.0) & (phi <= eg.phi_max) & (P[:, 2] >= p.z_min) & (P[:, 2] <= p.z_max)
            take = valid & (t < best)
            best = np.where(take, t, best)
            if k == 1:
                back = np.where(take, True, back)
            else:
                back = np.where(take, False, back)
        return best, back

    def _trace(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear RGB radiance and surface ids for pixel coordinates ``(N, 2)``."""
        g = self.g
        sc = g.scene
        p = sc.panel
        d = pixel_rays(self.cam, uv)
        n = d.shape[0]
        o = self.C
        t_best = np.full(n, _INF)
        ids = np.zeros(n, dtype=np.int32)
        aux_back = np.zeros(n, dtype=bool)
        box_axis = np.zeros(n, dtype=np.int8)

        def consider(t: np.ndarray, ident: int) -> np.ndarray:
            nonlocal t_best, ids
            take = t < t_best
            t_best = np.where(take, t, t_best)
            ids = np.where(take, ident, ids)
            return take

        # flat panel
        t = _plane_t(o, d, np.zeros(3), -_Y)
        P = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
        ok = np.isfinite(t) & (P[:, 2] >= p.z_min) & (P[:, 2] <= p.z_max)
        ok &= (g.left.inward(P) >= 0.0) if g.left else (P[:, 0] >= p.x_min)
        ok &= (g.right.inward(P) >= 0.0) if g.right else (P[:, 0] <= p.x_max)
        consider(np.where(ok, t, _INF), _ID_PANEL)

        # bends and return flanges
        for eg, cid, rid in ((g.left, _ID_CYL_L, _ID_RET_L), (g.right, _ID_CYL_R, _ID_RET_R)):
            if eg is None:
                continue
            tc, back = self._cyl(d, eg)
            take = consider(tc, cid)
            aux_back = np.where(take, back, aux_back)
            if eg.return_length > 0.0:
                p0, nrm = eg.surface_point(np.array([-eg.R * eg.phi_max]), np.array([0.0]))
                tau = -np.cos(eg.phi_max) * eg.e_in + np.sin(eg.phi_max) * _Y
                tr = _plane_t(o, d, p0[0], nrm[0])
                Pr = o + np.where(np.isfinite(tr), tr, 0.0)[:, None] * d
                s = (Pr - p0[0]) @ tau
                okr = np.isfinite(tr) & (s >= 0.0) & (s <= eg.return_length)
                okr &= (Pr[:, 2] >= p.z_min) & (Pr[:, 2] <= p.z_max)
                consider(np.where(okr, tr, _INF), rid)

        # floor
        if sc.floor.present:
            f = sc.floor
            tf = _plane_t(o, d, np.zeros(3), g.floor_normal)
            Pf = o + np.where(np.isfinite(tf), tf, 0.0)[:, None] * d
            tt = Pf @ g.floor_dir
            okf = np.isfinite(tf) & (tt >= 0.0) & (tt <= f.depth) & (Pf[:, 0] >= f.x_min) & (Pf[:, 0] <= f.x_max)
            consider(np.where(okf, tf, _INF), _ID_FLOOR)

        # lifted stickers (shims)
        for k, st in enumerate(g.shims):
            ts = _plane_t(o, d, np.array([0.0, -st.lift, 0.0]), -_Y)
            Ps = o + np.where(np.isfinite(ts), ts, 0.0)[:, None] * d
            c, s = np.cos(np.deg2rad(st.rotation_deg)), np.sin(np.deg2rad(st.rotation_deg))
            du = Ps[:, 0] - st.center[0]
            dv = Ps[:, 2] - st.center[1]
            a = du * c + dv * s
            b = -du * s + dv * c
            h = 0.5 * st.size
            oks = np.isfinite(ts) & (np.abs(a) <= h + st.margin) & (b <= h + st.margin) & (b >= -h - st.margin_bottom)
            consider(np.where(oks, ts, _INF), _ID_SHIM0 + k)

        # boxes (occluders, trim)
        for k, bx in enumerate(g.boxes):
            tb, ax = _box_t(o, d, np.asarray(bx.lo, dtype=float), np.asarray(bx.hi, dtype=float))
            take = consider(tb, _ID_BOX0 + k)
            box_axis = np.where(take, ax.astype(np.int8), box_axis)

        # background wall
        tbg = _plane_t(o, d, np.array([0.0, sc.background.y, 0.0]), -_Y)
        consider(tbg, _ID_BG)

        # ---------------- shading
        rgb = np.empty((n, 3))
        rgb[:] = np.asarray(sc.background.void)
        hit = np.isfinite(t_best)
        P = o + np.where(hit, t_best, 0.0)[:, None] * d
        alb = np.zeros((n, 3))
        nrm = np.zeros((n, 3))
        ks = np.zeros(n)
        shin = np.full(n, 30.0)

        m = ids == _ID_PANEL
        if np.any(m):
            a_, k_, s_ = g.panel_material(P[m, 0], P[m, 2])
            alb[m], ks[m], shin[m] = a_, k_, s_
            nrm[m] = -_Y
        for eg, cid, rid in ((g.left, _ID_CYL_L, _ID_RET_L), (g.right, _ID_CYL_R, _ID_RET_R)):
            if eg is None:
                continue
            m = ids == cid
            if np.any(m):
                A0 = eg.P_t + eg.R * _Y
                rel = P[m] - A0
                w = rel @ eg.a
                radial = rel - np.multiply.outer(w, eg.a)
                nn = radial / eg.R
                phi = np.arctan2(-(nn @ eg.e_in), -(nn @ _Y))
                su, sv = eg.unrolled_xz(-eg.R * phi, w)
                a_, k_, s_ = g.panel_material(su, sv)
                bk = aux_back[m]
                a_[bk] = np.asarray(p.back_albedo)
                k_[bk] = 0.0
                alb[m], ks[m], shin[m] = a_, k_, s_
                nrm[m] = np.where(bk[:, None], -nn, nn)
            m = ids == rid
            if np.any(m):
                p0, nr0 = eg.surface_point(np.array([-eg.R * eg.phi_max]), np.array([0.0]))
                tau = -np.cos(eg.phi_max) * eg.e_in + np.sin(eg.phi_max) * _Y
                rel = P[m] - p0[0]
                s = rel @ tau
                w = rel @ eg.a
                su, sv = eg.unrolled_xz(-eg.R * eg.phi_max - s, w)
                a_, k_, s_ = g.panel_material(su, sv)
                alb[m], ks[m], shin[m] = a_, k_, s_
                nrm[m] = nr0[0]
        m = ids == _ID_FLOOR
        if np.any(m):
            a_, k_, s_ = g.floor_material(P[m, 0], P[m] @ g.floor_dir)
            alb[m], ks[m], shin[m] = a_, k_, s_
            nrm[m] = g.floor_normal
        for k, st in enumerate(g.shims):
            m = ids == _ID_SHIM0 + k
            if np.any(m):
                c, s = np.cos(np.deg2rad(st.rotation_deg)), np.sin(np.deg2rad(st.rotation_deg))
                du = P[m, 0] - st.center[0]
                dv = P[m, 2] - st.center[1]
                _, val = g._sticker_paint(du * c + dv * s, -du * s + dv * c, st)
                alb[m] = val[:, None]
                ks[m] = st.spec_strength
                shin[m] = st.shininess
                nrm[m] = -_Y
        for k, bx in enumerate(g.boxes):
            m = ids == _ID_BOX0 + k
            if np.any(m):
                alb[m] = np.asarray(bx.albedo)
                ax = box_axis[m].astype(np.intp)
                nb = np.zeros((ax.size, 3))
                nb[np.arange(ax.size), ax] = -np.sign(d[m][np.arange(ax.size), ax])
                nrm[m] = nb
        m = ids == _ID_BG
        if np.any(m):
            bg = sc.background
            a_ = np.empty((int(m.sum()), 3))
            a_[:] = np.asarray(bg.albedo)
            if bg.texture_contrast > 0:
                tex = g.noise(P[m, 0], P[m, 2], bg.texture_scale)
                a_ *= (1.0 + bg.texture_contrast * tex)[:, None]
            alb[m] = a_
            nrm[m] = -_Y

        lit = hit
        if np.any(lit):
            L = self._shade(P[lit], nrm[lit], -d[lit], alb[lit], ks[lit], shin[lit])
            rgb[lit] = L
        return rgb, ids

    # -- lighting --------------------------------------------------------
    def _shadow(self, P: np.ndarray, to_light: np.ndarray, max_t: np.ndarray | None) -> np.ndarray:
        """1 where the point sees the light, 0 where a box blocks it."""
        vis = np.ones(P.shape[0])
        if not self.g.boxes:
            return vis
        for bx in self.g.boxes:
            lo = np.asarray(bx.lo, dtype=float)
            hi = np.asarray(bx.hi, dtype=float)
            # per-point origins: slab test written out for varying origins
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / to_light
                t0 = (lo - P) * inv
                t1 = (hi - P) * inv
            tmin = np.where(np.isnan(t0), -_INF, np.minimum(t0, t1))
            tmax = np.where(np.isnan(t1), _INF, np.maximum(t0, t1))
            tn = np.max(tmin, axis=1)
            tf = np.min(tmax, axis=1)
            blocked = (tf >= tn) & (tf > 1e-3) & (tn > 1e-3)
            if max_t is not None:
                blocked &= tn < max_t
            vis = np.where(blocked, 0.0, vis)
        return vis

    def _shade(
        self,
        P: np.ndarray,
        N: np.ndarray,
        V: np.ndarray,
        alb: np.ndarray,
        ks: np.ndarray,
        shin: np.ndarray,
    ) -> np.ndarray:
        lt = self.g.scene.light
        out = alb * lt.ambient
        Pe = P + 1e-3 * N
        if lt.directional is not None:
            direction, strength = lt.directional
            ld = -np.asarray(direction, dtype=float)
            ld = ld / np.linalg.norm(ld)
            ndl = np.clip(N @ ld, 0.0, None)
            vis = self._shadow(Pe, np.broadcast_to(ld, P.shape).copy(), None)
            out = out + alb * (strength * ndl * vis)[:, None]
        if lt.flash:
            lv = self.flash_pos - P
            dist = np.linalg.norm(lv, axis=1)
            l = lv / dist[:, None]
            ndl = np.clip(np.einsum("ij,ij->i", N, l), 0.0, None)
            fall = (lt.flash_ref_distance / dist) ** 2
            beam = np.clip(-(l @ self.optical_axis), 0.0, None) ** lt.flash_beam_exponent
            vis = self._shadow(Pe, l, dist)
            e = lt.flash_strength * fall * beam * vis
            out = out + alb * (e * ndl)[:, None]
            h = l + V
            h /= np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-12, None)
            ndh = np.clip(np.einsum("ij,ij->i", N, h), 0.0, None)
            spec = ks * e * (ndh**shin) * (ndl > 0.0)
            out = out + spec[:, None]
        return out

    # -- image -----------------------------------------------------------
    def render_linear(self, ss: int, adaptive: bool, chunk: int = 600_000, pattern: str = "grid") -> tuple[np.ndarray, np.ndarray]:
        """Linear-light image and object ids.

        ``pattern`` is how the ``ss * ss`` samples of a pixel are placed.  ``"grid"`` is a regular grid.  It is
        cheap and exact for axis-parallel edges, but every pixel along an edge at 45 degrees (or at any slope
        that is a small ratio) meets the grid at the same phase, so all of them carry the same rounding error:
        the rendered edge is displaced by up to 0.2 px with ``ss = 2`` and 0.1 px with ``ss = 3``.
        ``"lattice"`` places the samples on a golden-ratio lattice that is shifted by a different amount in every
        pixel; the rounding error then differs from pixel to pixel and averages out along an edge.
        """
        if pattern not in ("grid", "lattice"):
            raise ValueError("pattern must be 'grid' or 'lattice'")
        cam = self.cam
        W, H = cam.width, cam.height
        us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        uv = np.stack([us.ravel(), vs.ravel()], axis=1)
        n = uv.shape[0]
        img = np.empty((n, 3))
        ids = np.empty(n, dtype=np.int32)

        def trace_all(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            col = np.empty((coords.shape[0], 3))
            idd = np.empty(coords.shape[0], dtype=np.int32)
            for s in range(0, coords.shape[0], chunk):
                c, i = self._trace(coords[s : s + chunk])
                col[s : s + chunk] = c
                idd[s : s + chunk] = i
            return col, idd

        if ss <= 1:
            img, ids = trace_all(uv)
            return img.reshape(H, W, 3), ids.reshape(H, W)

        offs = (np.arange(ss) + 0.5) / ss - 0.5
        sub = np.stack(np.meshgrid(offs, offs), axis=-1).reshape(-1, 2)
        n_sub = sub.shape[0]
        k = np.arange(n_sub)
        lattice = np.stack([(k + 0.5) / n_sub, (k * 0.6180339887498949) % 1.0], axis=1)

        def offsets(j: int, pixels: np.ndarray) -> np.ndarray:
            """Sub-pixel position of sample ``j`` for the given pixels (flat indices), in [-0.5, 0.5)."""
            if pattern == "grid":
                return np.broadcast_to(sub[j], (pixels.size, 2))
            # per-pixel shift of the lattice from a hash of the pixel index: the same picture every time
            h1 = (pixels.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
            h2 = (pixels.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(0x165667B19E3779F9)) >> np.uint64(40)
            shift = np.stack([h1, h2], axis=1).astype(np.float64) / float(1 << 24)
            return (lattice[j] + shift) % 1.0 - 0.5

        if not adaptive:
            acc = np.zeros((n, 3))
            everything = np.arange(n)
            for j in range(n_sub):
                c, i = trace_all(uv + offsets(j, everything))
                acc += c
                ids = i
            return (acc / n_sub).reshape(H, W, 3), ids.reshape(H, W)

        img, ids = trace_all(uv)
        im2 = img.reshape(H, W, 3)
        id2 = ids.reshape(H, W)
        lum = im2.mean(axis=2).astype(np.float32)
        mx = cv2.dilate(lum, np.ones((3, 3), np.uint8))
        mn = cv2.erode(lum, np.ones((3, 3), np.uint8))
        idf = id2.astype(np.float32)
        edge = ((mx - mn) > (0.012 + 0.04 * mn)) | (
            cv2.dilate(idf, np.ones((3, 3), np.uint8)) != cv2.erode(idf, np.ones((3, 3), np.uint8))
        )
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sel = np.nonzero(edge.ravel())[0]
        if sel.size:
            acc = np.zeros((sel.size, 3))
            base = uv[sel]
            for j in range(n_sub):
                c, _ = trace_all(base + offsets(j, sel))
                acc += c
            img[sel] = acc / n_sub
        return img.reshape(H, W, 3), id2


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


@dataclass
class RenderResult:
    image: np.ndarray  # uint8 BGR (OpenCV channel order)
    truth: dict[str, Any]
    exposure: float
    jpeg: bytes | None = None


def scene_truth(scene: Scene, cam: CameraSpec) -> dict[str, Any]:
    """Ground truth for a shot: camera, tag and placard corners, datum lines."""
    g = SceneGeometry(scene)
    R_wc, C = camera_matrices(cam)
    tags = []
    for s in scene.stickers:
        Xw = g.sticker_corners_world(s)
        uv, depth = project_points(cam, Xw)
        tags.append(
            {
                "id": s.tag_id,
                "size_mm": s.size,
                "host": s.host,
                "lift_mm": s.lift,
                "center": list(s.center),
                "rotation_deg": s.rotation_deg,
                "corners_world": Xw.tolist(),
                "corners_px": uv.tolist(),
                "in_front": bool(np.all(depth > 0)),
                "in_image": bool(
                    np.all((uv[:, 0] >= 0) & (uv[:, 0] <= cam.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= cam.height - 1))
                ),
            }
        )
    placards = []
    for pl in scene.placards:
        Xw = g.placard_corners_world(pl)
        uv, _ = project_points(cam, Xw)
        placards.append(
            {
                "name": pl.name,
                "center": list(pl.center),
                "width": pl.width,
                "height": pl.height,
                "rotation_deg": pl.rotation_deg,
                "corners_world": Xw.tolist(),
                "corners_px": uv.tolist(),
            }
        )
    datums: dict[str, Any] = {
        "floor_line": {"point": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        "panel_bottom_edge": {"point": [0.0, 0.0, scene.panel.z_min], "direction": [1.0, 0.0, 0.0]},
    }
    if scene.trim is not None:
        t = scene.trim
        datums["trim_top_front_edge"] = {"point": [0.0, -t.thickness, t.height], "direction": [1.0, 0.0, 0.0]}
        datums["trim_panel_junction"] = {"point": [0.0, 0.0, t.height], "direction": [1.0, 0.0, 0.0]}
    for name, eg in (("left", g.left), ("right", g.right)):
        if eg is None:
            continue
        datums[f"{name}_tangent_line"] = {"point": eg.P_t.tolist(), "direction": eg.a.tolist()}
        datums[f"{name}_axis"] = {"point": (eg.P_t + eg.R * _Y).tolist(), "direction": eg.a.tolist(), "radius": eg.R}
        datums[f"{name}_outer_extreme_on_plane"] = {
            "point": (eg.P_t - eg.R * np.sin(min(eg.phi_max, np.pi / 2)) * eg.e_in).tolist(),
            "direction": eg.a.tolist(),
        }
    return {
        "camera": {
            "width": cam.width,
            "height": cam.height,
            "fx": cam.fx,
            "fy": cam.fy,
            "cx": cam.cx,
            "cy": cam.cy,
            "dist": list(cam.dist),
            "R_wc": R_wc.tolist(),
            "C": C.tolist(),
        },
        "tags": tags,
        "placards": placards,
        "datums": datums,
        "frame": "world: +X along floor seam, +Y into panel, +Z up the panel; mm",
    }


def render(
    scene: Scene,
    cam: CameraSpec,
    *,
    ss: int = 3,
    adaptive: bool = True,
    seed: int = 0,
    return_jpeg: bool = False,
    pattern: str = "grid",
) -> RenderResult:
    """Render ``scene`` as seen by ``cam``.  Fully deterministic for given inputs.

    ``pattern="lattice"`` avoids the coherent rounding of edges at 45 degrees and similar slopes that the
    regular sub-pixel grid has (see :meth:`_Tracer.render_linear`).
    """
    geom = SceneGeometry(scene)
    tracer = _Tracer(geom, cam)
    lin, _ids = tracer.render_linear(ss=ss, adaptive=adaptive, pattern=pattern)
    lin = lin * np.asarray(cam.white_balance)[None, None, :]

    exposure = float(cam.exposure)
    if cam.auto_exposure:
        mean = float(np.mean(lin))
        exposure = cam.auto_exposure_target / max(mean, 1e-6)
    lin = lin * exposure

    if cam.vignette > 0.0:
        us, vs = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
        r2 = ((us - cam.cx) / cam.fx) ** 2 + ((vs - cam.cy) / cam.fy) ** 2
        fall = 1.0 / (1.0 + r2) ** 2  # cos^4 of the field angle
        lin = lin * (1.0 - cam.vignette + cam.vignette * fall)[..., None]

    lin = lin.astype(np.float32)
    if cam.blur_sigma > 0.0:
        lin = cv2.GaussianBlur(lin, (0, 0), cam.blur_sigma, borderType=cv2.BORDER_REPLICATE)
    if cam.motion_blur_px > 1.0:
        L = int(round(cam.motion_blur_px))
        k = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.float32)
        a = np.deg2rad(cam.motion_blur_deg)
        for s in np.linspace(-L / 2.0, L / 2.0, 4 * L + 1):
            x = int(round(L + s * np.cos(a)))
            y = int(round(L + s * np.sin(a)))
            k[y, x] += 1.0
        k /= k.sum()
        lin = cv2.filter2D(lin, -1, k, borderType=cv2.BORDER_REPLICATE)

    if cam.noise_shot > 0.0 or cam.noise_read > 0.0:
        rng = np.random.default_rng(seed)
        var = cam.noise_shot * np.clip(lin, 0.0, None) + cam.noise_read**2
        lin = lin + rng.standard_normal(lin.shape).astype(np.float32) * np.sqrt(var)

    enc = srgb_encode(lin.astype(np.float64)).astype(np.float32)
    if cam.tone_gamma != 1.0:
        enc = np.power(np.clip(enc, 0.0, 1.0), cam.tone_gamma)
    if cam.tone_contrast != 0.0:
        # smooth S-curve around mid-grey (0 = identity), like a "vivid" picture style
        k = cam.tone_contrast
        enc = np.clip(enc + k * (enc - 0.5) * (1.0 - np.abs(2.0 * enc - 1.0)) * 2.0 * enc * (1 - enc) * 4.0, 0.0, 1.0)
    if cam.sharpen > 0.0:
        soft = cv2.GaussianBlur(enc, (0, 0), 1.2, borderType=cv2.BORDER_REPLICATE)
        enc = np.clip(enc + cam.sharpen * (enc - soft), 0.0, 1.0)
    img8 = np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(img8[:, :, ::-1])

    jpeg: bytes | None = None
    if cam.jpeg_quality is not None or return_jpeg:
        q = int(cam.jpeg_quality if cam.jpeg_quality is not None else 95)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:  # pragma: no cover - OpenCV always encodes valid input
            raise RuntimeError("JPEG encoding failed")
        jpeg = buf.tobytes()
        if cam.jpeg_quality is not None:
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    truth = scene_truth(scene, cam)
    truth["render"] = {"ss": ss, "adaptive": adaptive, "pattern": pattern, "seed": seed, "exposure": exposure}
    return RenderResult(image=bgr, truth=truth, exposure=exposure, jpeg=jpeg)


TruthFn = Callable[[Scene, CameraSpec], dict[str, Any]]
