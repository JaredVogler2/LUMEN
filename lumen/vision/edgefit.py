"""Sub-pixel straight-edge localisation by edge-spread-function fitting.

Model for the pixels in a narrow band around a straight black/white edge:

    I(x) = B(a) + (W(a) - B(a)) * Phi(s / sigma)
    s = n(theta) . (x - c) - d        signed distance to the line (+ on the white side)
    a = t . (x - c) / L               normalised along-edge coordinate

``Phi`` is the Gaussian CDF (edge-spread function of a Gaussian PSF), ``B`` and
``W`` are the dark and bright levels, each allowed a linear trend along the
edge (flash fall-off, vignetting).  The fit is done

* on **linear-light** intensities (see :mod:`lumen.vision.imaging`): only there
  is a blurred edge anti-symmetric about its true position;
* on the **actual pixel samples** (no resampling), with pixel coordinates
  mapped through the lens model, so the straight line lives in ideal
  (undistorted) coordinates;
* in a band that is re-centred on the estimate, so that truncation of the
  profile is symmetric and does not bias ``d``.

The result carries the first-order covariance of the line, which later becomes
the covariance of tag corners and placard edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from ..geometry.camera import CameraModel
from ..geometry.optimize import lm_solve
from .imaging import SRGB_TO_LINEAR_LUT

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_LOGSIG_MIN, _LOGSIG_MAX = float(np.log(0.15)), float(np.log(8.0))


@dataclass
class EdgeFit:
    ok: bool
    reason: str = ""
    #: unit normal pointing from the dark side to the bright side (ideal px)
    n: np.ndarray | None = None
    #: point on the fitted line at the middle of the support (ideal px)
    mid: np.ndarray | None = None
    length: float = 0.0  # supported length along the edge, px
    sigma_blur: float = 0.0  # px
    black: float = 0.0  # linear level on the dark side (mid-edge)
    white: float = 0.0
    rms: float = 0.0  # residual RMS in linear units
    n_pix: int = 0
    inlier_frac: float = 1.0
    #: variances of the perpendicular position at ``mid`` and of the direction
    var_d: float = 0.0
    var_theta: float = 0.0
    cov_d_theta: float = 0.0
    saturated_frac: float = 0.0
    #: half the width of a line that the photo resolves into its two edges (px); 0 for an edge and for a line
    #: thinner than that.  ``mid`` is then on the centre line between the two edges.
    bar_half_px: float = 0.0

    @property
    def contrast(self) -> float:
        return self.white - self.black

    @property
    def t(self) -> np.ndarray:
        """Unit direction along the edge (n is 90° counter-clockwise from t)."""
        assert self.n is not None
        return np.array([self.n[1], -self.n[0]])

    @property
    def d(self) -> float:
        assert self.n is not None and self.mid is not None
        return float(self.n @ self.mid)

    def offset_at(self, pts: np.ndarray) -> np.ndarray:
        assert self.n is not None and self.mid is not None
        return (np.asarray(pts, dtype=float) - self.mid) @ self.n


def _band_pixels(
    gray: np.ndarray,
    camera: CameraModel | None,
    c: np.ndarray,
    t: np.ndarray,
    n: np.ndarray,
    half_len: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pixels of the band: (ideal xy, linear intensity, raw code values)."""
    H, W = gray.shape[:2]
    pad = 1.5
    corners = np.array(
        [
            c + sa * (half_len + pad) * t + sn * (half_width + pad) * n
            for sa in (-1.0, 1.0)
            for sn in (-1.0, 1.0)
        ]
    )
    raw = camera.distort_pixels(corners) if (camera is not None and camera.has_distortion) else corners
    u0 = int(max(np.floor(raw[:, 0].min()), 0))
    u1 = int(min(np.ceil(raw[:, 0].max()), W - 1))
    v0 = int(max(np.floor(raw[:, 1].min()), 0))
    v1 = int(min(np.ceil(raw[:, 1].max()), H - 1))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    us, vs = np.meshgrid(np.arange(u0, u1 + 1, dtype=float), np.arange(v0, v1 + 1, dtype=float))
    xy = np.stack([us.ravel(), vs.ravel()], axis=1)
    if camera is not None and camera.has_distortion:
        xy = camera.undistort_pixels(xy)
    codes = gray[v0 : v1 + 1, u0 : u1 + 1].ravel()
    return xy, SRGB_TO_LINEAR_LUT[codes], codes


def fit_edge(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    normal_to_white: np.ndarray,
    *,
    half_width: float,
    trim: float,
    camera: CameraModel | None = None,
    fix_theta: bool = False,
    fix_sigma: float | None = None,
    level_slopes: bool = True,
    passes: int = 2,
    min_contrast: float = 0.04,
) -> EdgeFit:
    """Fit the edge predicted between ``p0`` and ``p1`` (ideal pixel coords).

    ``normal_to_white`` only needs to indicate the bright side; ``trim`` is
    the length removed at each end (to stay clear of corners / neighbouring
    structures); ``half_width`` is the half-width of the fitting band.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    seg = p1 - p0
    L_full = float(np.linalg.norm(seg))
    if L_full < (2.4 if fix_theta else 4.0):
        return EdgeFit(False, "segment_too_short")
    t0 = seg / L_full
    n0 = np.array([-t0[1], t0[0]])
    if float(n0 @ np.asarray(normal_to_white, dtype=float)) < 0:
        n0 = -n0
        t0 = -t0
    c = 0.5 * (p0 + p1)
    half_len = 0.5 * L_full - trim
    if half_len < (1.1 if fix_theta else 1.5):
        return EdgeFit(False, "segment_too_short")

    d_cur = 0.0
    th_cur = 0.0
    result: EdgeFit | None = None
    sigma0 = fix_sigma if fix_sigma is not None else 1.0

    for p in range(max(1, passes)):
        cs, sn = np.cos(th_cur), np.sin(th_cur)
        n_cur = np.array([cs * n0[0] - sn * n0[1], sn * n0[0] + cs * n0[1]])
        t_cur = np.array([n_cur[1], -n_cur[0]])
        c_cur = c + d_cur * n_cur
        band = _band_pixels(gray, camera, c_cur, t_cur, n_cur, half_len, half_width)
        if band is None:
            return EdgeFit(False, "outside_image")
        xy, lin, codes = band
        rel = xy - c_cur
        s_all = rel @ n_cur
        a_all = rel @ t_cur
        m = (np.abs(s_all) <= half_width) & (np.abs(a_all) <= half_len)
        if int(m.sum()) < (8 if fix_theta else 12):
            return EdgeFit(False, "too_few_pixels")
        rel = rel[m]
        I = lin[m].astype(np.float64)
        sat = float(np.mean(codes[m] >= 254))
        s0 = s_all[m]
        a = a_all[m] / max(half_len, 1e-9) * 0.5  # in [-0.5, 0.5]

        # robust plateau estimates from the outer thirds of the band
        inner = I[s0 < -half_width * 0.45]
        outer = I[s0 > half_width * 0.45]
        if inner.size < 2 or outer.size < 2:
            return EdgeFit(False, "too_few_pixels")
        b_init = float(np.median(inner))
        w_init = float(np.median(outer))
        if w_init - b_init < min_contrast:
            return EdgeFit(False, "low_contrast", black=b_init, white=w_init, saturated_frac=sat)

        if p == 0:
            # coarse position from the 50 % crossing of the binned profile
            nb = max(int(half_width * 2.0), 6)
            edges = np.linspace(-half_width, half_width, nb + 1)
            idx = np.clip(np.digitize(s0, edges) - 1, 0, nb - 1)
            cnt = np.bincount(idx, minlength=nb)
            prof = np.bincount(idx, weights=I, minlength=nb) / np.maximum(cnt, 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            half = 0.5 * (b_init + w_init)
            above = prof >= half
            cross = np.nonzero(above[1:] & ~above[:-1] & (cnt[1:] > 0) & (cnt[:-1] > 0))[0]
            d_seed = 0.0
            if cross.size:
                k = int(cross[np.argmin(np.abs(centers[cross]))])
                f = (half - prof[k]) / max(prof[k + 1] - prof[k], 1e-9)
                d_seed = float(centers[k] + f * (centers[k + 1] - centers[k]))
        else:
            d_seed = 0.0

        nx, ny = n_cur
        rx, ry = rel[:, 0], rel[:, 1]
        s_theta = -ny * rx + nx * ry  # d s / d theta at theta = 0

        free_theta = not fix_theta
        free_sigma = fix_sigma is None
        # parameter vector layout
        names = ["d"]
        if free_theta:
            names.append("theta")
        if free_sigma:
            names.append("logsig")
        names += ["b0", "w0"]
        if level_slopes:
            names += ["b1", "w1"]
        ix = {k: i for i, k in enumerate(names)}
        x0 = np.zeros(len(names))
        x0[ix["d"]] = d_seed
        if free_sigma:
            x0[ix["logsig"]] = np.log(sigma0)
        x0[ix["b0"]] = b_init
        x0[ix["w0"]] = w_init

        def fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d = x[ix["d"]]
            th = x[ix["theta"]] if free_theta else 0.0
            # the blur stays within 0.15 .. 8 px; limiting the logarithm (not the result) keeps a wild trial step
            # of the solver from overflowing the exponential
            sig = float(np.exp(min(max(x[ix["logsig"]], _LOGSIG_MIN), _LOGSIG_MAX))) if free_sigma else min(max(float(sigma0), 0.15), 8.0)
            B = x[ix["b0"]] + (x[ix["b1"]] * a if level_slopes else 0.0)
            Wt = x[ix["w0"]] + (x[ix["w1"]] * a if level_slopes else 0.0)
            # small-angle rotation of the normal about c_cur
            s = s0 + th * s_theta - d
            u = s / sig
            Ph = ndtr(u)
            ph = np.exp(-0.5 * u * u) / _SQRT_2PI
            amp = Wt - B
            model = B + amp * Ph
            J = np.empty((I.size, len(names)))
            J[:, ix["d"]] = -amp * ph / sig
            if free_theta:
                J[:, ix["theta"]] = amp * ph / sig * s_theta
            if free_sigma:
                J[:, ix["logsig"]] = -amp * ph * u
            J[:, ix["b0"]] = 1.0 - Ph
            J[:, ix["w0"]] = Ph
            if level_slopes:
                J[:, ix["b1"]] = (1.0 - Ph) * a
                J[:, ix["w1"]] = Ph * a
            return model - I, J

        sol = lm_solve(fn, x0, max_iter=25, robust_k=2.5, robust_rounds=2)
        d_fit = float(sol.x[ix["d"]])
        th_fit = float(sol.x[ix["theta"]]) if free_theta else 0.0
        sig_fit = float(np.exp(min(max(sol.x[ix["logsig"]], _LOGSIG_MIN), _LOGSIG_MAX))) if free_sigma else min(max(float(sigma0), 0.15), 8.0)
        if not np.isfinite(d_fit) or abs(d_fit) > half_width or abs(th_fit) > 0.35:
            return EdgeFit(False, "fit_diverged", saturated_frac=sat)
        black = float(sol.x[ix["b0"]])
        white = float(sol.x[ix["w0"]])
        if white - black < min_contrast:
            return EdgeFit(False, "low_contrast", black=black, white=white, saturated_frac=sat)

        cov = sol.covariance()
        var_d = float(cov[ix["d"], ix["d"]])
        var_th = float(cov[ix["theta"], ix["theta"]]) if free_theta else 0.0
        cov_dt = float(cov[ix["d"], ix["theta"]]) if free_theta else 0.0

        # compose the update: new normal, new point on the line
        cs2, sn2 = np.cos(th_fit), np.sin(th_fit)
        n_new = np.array([cs2 * n_cur[0] - sn2 * n_cur[1], sn2 * n_cur[0] + cs2 * n_cur[1]])
        # the line is  n_new . (x - c_cur) = d_fit ; closest point to c_cur:
        mid = c_cur + d_fit * n_new
        w_rob = sol.weights
        result = EdgeFit(
            ok=True,
            n=n_new,
            mid=mid,
            length=2.0 * half_len,
            sigma_blur=sig_fit,
            black=black,
            white=white,
            rms=float(np.sqrt(np.mean(sol.residuals**2))),
            n_pix=int(I.size),
            inlier_frac=float(np.mean(w_rob > 0.999)),
            var_d=var_d,
            var_theta=var_th,
            cov_d_theta=cov_dt,
            saturated_frac=sat,
        )
        # prepare the re-centred pass
        th_cur = th_cur + th_fit
        # keep the reference point c fixed and express the offset along the new normal
        d_cur = float((mid - c) @ n_new)
        if free_sigma:
            sigma0 = sig_fit
    assert result is not None
    return result


def intersect_lines(e0: EdgeFit, e1: EdgeFit) -> np.ndarray | None:
    """Intersection of two fitted edges (ideal pixels)."""
    assert e0.n is not None and e1.n is not None
    A = np.stack([e0.n, e1.n])
    det = float(np.linalg.det(A))
    if abs(det) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([e0.d, e1.d]))


def corner_jacobian(e_prev: EdgeFit, e_next: EdgeFit, corner: np.ndarray) -> np.ndarray:
    """``d corner / d (d_prev, theta_prev, d_next, theta_next)`` as a 2x4 matrix.

    Each line is parametrised at its own ``mid`` point: shifting it by ``d``
    along its normal, or rotating it by ``theta`` about ``mid``.
    """
    assert e_prev.n is not None and e_next.n is not None
    assert e_prev.mid is not None and e_next.mid is not None
    A = np.stack([e_prev.n, e_next.n])
    Ainv = np.linalg.inv(A)
    # n.(x - mid) = d ; rotating n by theta changes the left side by theta * n_perp.(x - mid)
    q_prev = float(np.array([-e_prev.n[1], e_prev.n[0]]) @ (corner - e_prev.mid))
    q_next = float(np.array([-e_next.n[1], e_next.n[0]]) @ (corner - e_next.mid))
    J = np.zeros((2, 4))
    J[:, 0] = Ainv[:, 0]
    J[:, 1] = -q_prev * Ainv[:, 0]
    J[:, 2] = Ainv[:, 1]
    J[:, 3] = -q_next * Ainv[:, 1]
    return J


# --------------------------------------------------------------------------
# Thin line ("bar") features and operator-seeded snapping
# --------------------------------------------------------------------------
def fit_bar(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    half_width: float,
    trim: float = 0.0,
    camera: CameraModel | None = None,
    min_contrast: float = 0.02,
) -> EdgeFit:
    """Fit a thin dark or bright line (a seam, a scribed joint).

    Model: ``I = B(a) + A * exp(-s^2 / (2 w^2))`` with a linear base level along
    the line.  The returned :class:`EdgeFit` has ``black``/``white`` set to the
    line and base levels (whichever is darker first) and ``sigma_blur`` set to
    the Gaussian half-width ``w`` of the line profile.  ``n`` keeps the
    orientation of the left normal of ``p0 -> p1``.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    seg = p1 - p0
    L_full = float(np.linalg.norm(seg))
    if L_full < 6.0:
        return EdgeFit(False, "segment_too_short")
    t0 = seg / L_full
    n0 = np.array([-t0[1], t0[0]])
    c = 0.5 * (p0 + p1)
    half_len = 0.5 * L_full - trim
    band = _band_pixels(gray, camera, c, t0, n0, half_len, half_width)
    if band is None:
        return EdgeFit(False, "outside_image")
    xy, lin, codes = band
    rel = xy - c
    s0 = rel @ n0
    a_all = rel @ t0
    m = (np.abs(s0) <= half_width) & (np.abs(a_all) <= half_len)
    if int(m.sum()) < 20:
        return EdgeFit(False, "too_few_pixels")
    rel = rel[m]
    s0 = s0[m]
    I = lin[m].astype(np.float64)
    a = a_all[m] / max(half_len, 1e-9) * 0.5
    sat = float(np.mean(codes[m] >= 254))
    base = float(np.median(I[np.abs(s0) > 0.6 * half_width])) if np.any(np.abs(s0) > 0.6 * half_width) else float(np.median(I))
    # seed from the binned profile extremum
    nb = max(int(half_width * 2.0), 8)
    edges = np.linspace(-half_width, half_width, nb + 1)
    idx = np.clip(np.digitize(s0, edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    prof = np.bincount(idx, weights=I, minlength=nb) / np.maximum(cnt, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dev = np.where(cnt > 0, prof - base, 0.0)
    k = int(np.argmax(np.abs(dev)))
    amp0 = float(dev[k])
    if abs(amp0) < min_contrast:
        return EdgeFit(False, "low_contrast", saturated_frac=sat)
    s_theta = -n0[1] * rel[:, 0] + n0[0] * rel[:, 1]

    def fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d, th, lw, b0, b1, amp = x
        w = float(np.clip(np.exp(lw), 0.3, half_width))
        s = s0 + th * s_theta - d
        g = np.exp(-0.5 * (s / w) ** 2)
        model = b0 + b1 * a + amp * g
        J = np.empty((I.size, 6))
        J[:, 0] = amp * g * s / (w * w)
        J[:, 1] = -amp * g * s / (w * w) * s_theta
        J[:, 2] = amp * g * (s / w) ** 2
        J[:, 3] = 1.0
        J[:, 4] = a
        J[:, 5] = g
        return model - I, J

    x0 = np.array([float(centers[k]), 0.0, np.log(1.2), base, 0.0, amp0])
    sol = lm_solve(fn, x0, max_iter=30, robust_k=2.5, robust_rounds=2)
    d, th, lw, b0, b1, amp = sol.x
    if not np.isfinite(d) or abs(d) > half_width or abs(th) > 0.35 or abs(amp) < min_contrast:
        return EdgeFit(False, "fit_diverged", saturated_frac=sat)
    cov = sol.covariance()
    cs, sn = np.cos(th), np.sin(th)
    n_new = np.array([cs * n0[0] - sn * n0[1], sn * n0[0] + cs * n0[1]])
    mid = c + d * n_new
    return EdgeFit(
        ok=True,
        n=n_new,
        mid=mid,
        length=2.0 * half_len,
        sigma_blur=float(np.clip(np.exp(lw), 0.3, half_width)),
        black=float(min(b0, b0 + amp)),
        white=float(max(b0, b0 + amp)),
        rms=float(np.sqrt(np.mean(sol.residuals**2))),
        n_pix=int(I.size),
        inlier_frac=float(np.mean(sol.weights > 0.999)),
        var_d=float(cov[0, 0]),
        var_theta=float(cov[1, 1]),
        cov_d_theta=float(cov[0, 1]),
        saturated_frac=sat,
    )


def fit_trench(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    half_width: float,
    h0: float,
    trim: float = 0.0,
    camera: CameraModel | None = None,
    min_contrast: float = 0.02,
) -> EdgeFit:
    """Fit a line that the photo resolves: two opposite edges a few pixels apart, the same level outside both.

    A seam 2 mm wide is a thin line in a small photo and a trench seven pixels wide in a full-resolution one.
    Model: ``I = B(a) + A * [Phi((s + h) / sigma) - Phi((s - h) / sigma)]``, a box of half-width ``h`` blurred
    by ``sigma``, on a base level that is linear along the line.  ``p0 -> p1`` is the seed of the centre line
    and ``h0`` the seed of the half-width.  The fit is accepted only when the two edges are resolved
    (``h >= 1.2 sigma``; a thinner line is the business of :func:`fit_bar`) and both lie inside the band.

    In the returned :class:`EdgeFit`, ``mid`` is on the centre line, ``bar_half_px`` is ``h`` and ``sigma_blur``
    the width of the Gaussian with the same second moment, ``sqrt(sigma^2 + h^2 / 3)``.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    seg = p1 - p0
    L_full = float(np.linalg.norm(seg))
    if L_full < 6.0:
        return EdgeFit(False, "segment_too_short")
    t0 = seg / L_full
    n0 = np.array([-t0[1], t0[0]])
    c = 0.5 * (p0 + p1)
    half_len = 0.5 * L_full - trim
    band = _band_pixels(gray, camera, c, t0, n0, half_len, half_width)
    if band is None:
        return EdgeFit(False, "outside_image")
    xy, lin, codes = band
    rel = xy - c
    s0 = rel @ n0
    a_all = rel @ t0
    m = (np.abs(s0) <= half_width) & (np.abs(a_all) <= half_len)
    if int(m.sum()) < 40:
        return EdgeFit(False, "too_few_pixels")
    rel = rel[m]
    s0 = s0[m]
    I = lin[m].astype(np.float64)
    a = a_all[m] / max(half_len, 1e-9) * 0.5
    sat = float(np.mean(codes[m] >= 254))
    outer = np.abs(s0) > h0 + 2.0
    inner = np.abs(s0) < max(h0 - 1.5, 0.6)
    if int(outer.sum()) < 10 or int(inner.sum()) < 5:
        return EdgeFit(False, "too_few_pixels", saturated_frac=sat)
    base = float(np.median(I[outer]))
    amp0 = float(np.median(I[inner])) - base
    if abs(amp0) < min_contrast:
        return EdgeFit(False, "low_contrast", saturated_frac=sat)
    s_theta = -n0[1] * rel[:, 0] + n0[0] * rel[:, 1]
    h_max = max(half_width - 1.0, 1.0)

    def fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d, th, lh, ls, b0, b1, amp = x
        h = float(np.clip(np.exp(min(lh, 5.0)), 0.5, h_max))
        sg = float(np.clip(np.exp(min(max(ls, -3.0), 3.0)), 0.3, 4.0))
        s = s0 + th * s_theta - d
        u1 = (s + h) / sg
        u2 = (s - h) / sg
        f1 = np.exp(-0.5 * u1 * u1) / _SQRT_2PI
        f2 = np.exp(-0.5 * u2 * u2) / _SQRT_2PI
        P = ndtr(u1) - ndtr(u2)
        dP_ds = (f1 - f2) / sg
        model = b0 + b1 * a + amp * P
        J = np.empty((I.size, 7))
        J[:, 0] = -amp * dP_ds
        J[:, 1] = amp * dP_ds * s_theta
        J[:, 2] = amp * (f1 + f2) / sg * h
        J[:, 3] = -amp * (u1 * f1 - u2 * f2)
        J[:, 4] = 1.0
        J[:, 5] = a
        J[:, 6] = P
        return model - I, J

    x0 = np.array([0.0, 0.0, np.log(max(h0, 0.6)), 0.0, base, 0.0, amp0])
    sol = lm_solve(fn, x0, max_iter=40, robust_k=2.5, robust_rounds=2)
    d, th, lh, ls, b0, _b1, amp = sol.x
    h = float(np.clip(np.exp(min(lh, 5.0)), 0.5, h_max))
    sg = float(np.clip(np.exp(min(max(ls, -3.0), 3.0)), 0.3, 4.0))
    if not np.isfinite(d) or abs(d) > half_width or abs(th) > 0.35 or abs(amp) < min_contrast:
        return EdgeFit(False, "fit_diverged", saturated_frac=sat)
    if h < 1.2 * sg:
        return EdgeFit(False, "not_resolved", saturated_frac=sat)
    if abs(d) + h + 2.0 * sg > half_width:
        return EdgeFit(False, "edges_outside_band", saturated_frac=sat)
    cov = sol.covariance()
    cs, sn = np.cos(th), np.sin(th)
    n_new = np.array([cs * n0[0] - sn * n0[1], sn * n0[0] + cs * n0[1]])
    return EdgeFit(
        ok=True,
        n=n_new,
        mid=c + d * n_new,
        length=2.0 * half_len,
        sigma_blur=float(np.sqrt(sg * sg + h * h / 3.0)),
        black=float(min(b0, b0 + amp)),
        white=float(max(b0, b0 + amp)),
        rms=float(np.sqrt(np.mean(sol.residuals**2))),
        n_pix=int(I.size),
        inlier_frac=float(np.mean(sol.weights > 0.999)),
        var_d=float(cov[0, 0]),
        var_theta=float(cov[1, 1]),
        cov_d_theta=float(cov[0, 1]),
        saturated_frac=sat,
        bar_half_px=h,
    )


def _wide_line(gray: np.ndarray, camera: CameraModel | None, step: EdgeFit, max_bar_px: float, trim: float) -> EdgeFit | None:
    """The resolved line of which the fitted edge ``step`` is one flank, if it is one.

    Looked for across the edge, up to ``max_bar_px`` away on either side: a second edge of the opposite
    direction and of comparable strength, with the same level outside both.  Nothing of the kind: ``None``
    (an edge between two different surfaces has different levels on its two sides and stays an edge).
    """
    assert step.mid is not None and step.n is not None
    n, t = step.n, step.t  # n points from the dark side of the edge to its bright side
    half_len = 0.5 * step.length
    hw = max_bar_px + 6.0
    band = _band_pixels(gray, camera, step.mid, t, n, half_len, hw)
    if band is None:
        return None
    xy, lin, _codes = band
    rel = xy - step.mid
    keep = np.abs(rel @ t) <= half_len
    sdist = (rel @ n)[keep]
    I = lin[keep].astype(np.float64)
    nb = int(2 * hw) + 1
    edges = np.linspace(-hw, hw, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(sdist, edges) - 1
    ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    if not np.any(cnt > 0):
        return None
    prof = np.bincount(idx[ok], weights=I[ok], minlength=nb) / np.maximum(cnt, 1)
    valid = cnt > 0.25 * np.median(cnt[cnt > 0])
    g = np.zeros(nb)
    g[1:-1] = (prof[2:] - prof[:-2]) * 0.5
    g[~valid] = 0.0
    k0 = int(np.argmin(np.abs(centers)))
    g0 = float(np.max(np.abs(g[max(k0 - 1, 0) : k0 + 2])))
    if g0 <= 0.0:
        return None

    def level(lo: float, hi: float) -> float | None:
        sel = valid & (centers >= lo) & (centers <= hi)
        return float(np.median(prof[sel])) if int(sel.sum()) >= 2 else None

    best: tuple[float, float] | None = None  # (signed distance of the second edge, its strength)
    for k in range(2, nb - 2):
        sep = float(centers[k])
        if not (4.0 <= abs(sep) <= max_bar_px) or not valid[k]:
            continue
        # the first edge rises along n; the second one must fall, be a local extremum and be of comparable strength
        if g[k] >= 0.0 or -g[k] < 0.5 * g0 or -g[k] < -g[k - 1] or -g[k] < -g[k + 1]:
            continue
        side = 1.0 if sep > 0 else -1.0
        a_in, b_in = sorted((side * 1.5, sep - side * 1.5))
        inner = level(a_in, b_in) if b_in - a_in >= 1.0 else level(min(0.0, sep), max(0.0, sep))
        out_near = level(*sorted((-side * 1.5, -side * 5.0)))
        out_far = level(*sorted((sep + side * 1.5, sep + side * 5.0)))
        if inner is None or out_near is None or out_far is None:
            continue
        depth = 0.5 * (out_near + out_far) - inner
        if abs(depth) < 0.02 or abs(out_near - out_far) > 0.35 * abs(depth):
            continue
        if best is None or abs(sep) < abs(best[0]):
            best = (sep, float(-g[k]))
    if best is None:
        return None
    sep = best[0]
    h0 = 0.5 * abs(sep)
    c = step.mid + 0.5 * sep * n
    q0, q1 = c - 0.5 * step.length * t, c + 0.5 * step.length * t
    fit = fit_trench(gray, q0, q1, half_width=min(h0 + 5.0, hw), h0=h0, trim=trim, camera=camera, min_contrast=0.015)
    if not fit.ok or fit.inlier_frac < 0.6 or fit.rms > max(1.6 * step.rms, 0.004):
        return None
    return fit


@dataclass
class SnapResult:
    ok: bool
    reason: str = ""
    kind: str = ""  # "step" | "bar"
    fit: EdgeFit | None = None
    #: how far the feature was from the seed (px, signed along the seed's left normal) and its tilt
    shift_px: float = 0.0
    tilt_deg: float = 0.0
    #: strength of the best feature relative to the second best in the search band
    distinctness: float = 0.0
    n_candidates: int = 0
    #: the model that was not chosen, when both fitted ("step" for a chosen "bar" and the reverse).  Which of
    #: the two wins can differ between photos of the same feature, and a model that wins over the whole length
    #: need not hold piece by piece: callers that compare or re-check a feature can ask for the other one.
    other_kind: str = ""
    other: EdgeFit | None = None

    def as_kind(self, kind: str) -> "SnapResult | None":
        """This result described by the model ``kind``; ``None`` when that model did not fit."""
        if self.kind == kind:
            return self
        if self.other is not None and self.other_kind == kind:
            return SnapResult(self.ok, self.reason, kind, self.other, self.shift_px, self.tilt_deg, self.distinctness, self.n_candidates, self.kind, self.fit)
        return None


def snap_line(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    camera: CameraModel | None = None,
    search_px: float = 18.0,
    max_tilt_deg: float = 4.0,
    fit_half_width: float = 5.0,
    trim: float = 2.0,
    max_bar_px: float = 0.0,
) -> SnapResult:
    """Refine an operator-drawn line onto the nearest straight image feature.

    A coarse search over offset and tilt finds the strongest straight
    intensity transition inside the band around the seed; the edge-spread
    (step) and thin-line (bar) models are then fitted there and the better one
    is returned.  ``distinctness`` below ~1.5 means a second feature of
    similar strength lies inside the search band — the caller must then treat
    the snap as ambiguous and ask the operator.

    ``max_bar_px`` (0 = off): a seam that the photo resolves is two edges, and the search above lands on one of
    them.  When the edge has a partner within this many pixels (opposite direction, same level outside), the two
    are fitted together as one line and its centre is returned as a ``"bar"`` with ``fit.bar_half_px`` set; the
    single edge stays available as ``other``.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L < 20.0:
        return SnapResult(False, "segment_too_short")
    t0 = seg / L
    n0 = np.array([-t0[1], t0[0]])
    c = 0.5 * (p0 + p1)
    half_len = 0.5 * L - trim
    band = _band_pixels(gray, camera, c, t0, n0, half_len, search_px + fit_half_width)
    if band is None:
        return SnapResult(False, "outside_image")
    xy, lin, _codes = band
    rel = xy - c
    a_all = rel @ t0
    keep = np.abs(a_all) <= half_len
    rel = rel[keep]
    I = lin[keep].astype(np.float64)
    if I.size < 200:
        return SnapResult(False, "too_few_pixels")
    half_w = search_px + fit_half_width
    nb = int(2 * half_w) + 1
    edges = np.linspace(-half_w, half_w, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    best = (0.0, 0.0, 0.0)  # score, tilt, offset
    scores_at_best_tilt: np.ndarray | None = None
    for tilt in np.deg2rad(np.arange(-max_tilt_deg, max_tilt_deg + 1e-9, 0.5)):
        cs, sn = np.cos(tilt), np.sin(tilt)
        n = np.array([cs * n0[0] - sn * n0[1], sn * n0[0] + cs * n0[1]])
        s = rel @ n
        idx = np.digitize(s, edges) - 1
        ok = (idx >= 0) & (idx < nb)
        cnt = np.bincount(idx[ok], minlength=nb)
        prof = np.bincount(idx[ok], weights=I[ok], minlength=nb) / np.maximum(cnt, 1)
        valid = cnt > 0.25 * np.median(cnt[cnt > 0])
        # smoothed derivative (central difference over 2 bins)
        g = np.zeros(nb)
        g[1:-1] = (prof[2:] - prof[:-2]) * 0.5
        g[~valid] = 0.0
        g[:2] = 0.0
        g[-2:] = 0.0
        inside = np.abs(centers) <= search_px
        sc = np.abs(g) * inside
        k = int(np.argmax(sc))
        if sc[k] > best[0]:
            best = (float(sc[k]), float(tilt), float(centers[k]))
            scores_at_best_tilt = sc.copy()
    if best[0] < 0.004 or scores_at_best_tilt is None:
        return SnapResult(False, "no_feature_found")
    sc = scores_at_best_tilt

    cs, sn = np.cos(best[1]), np.sin(best[1])
    n = np.array([cs * n0[0] - sn * n0[1], sn * n0[0] + cs * n0[1]])
    t = np.array([n[1], -n[0]])
    cc = c + best[2] * n
    q0 = cc - (0.5 * L) * t
    q1 = cc + (0.5 * L) * t
    # polarity for the step model: brighter side
    s = rel @ n - best[2]
    near = np.abs(s) <= fit_half_width
    pos = I[near & (s > 1.0)]
    neg = I[near & (s < -1.0)]
    if pos.size < 5 or neg.size < 5:
        return SnapResult(False, "too_few_pixels")
    toward_white = n if float(np.median(pos)) >= float(np.median(neg)) else -n
    step = fit_edge(gray, q0, q1, toward_white, half_width=fit_half_width, trim=trim, camera=camera, min_contrast=0.015)
    bar = fit_bar(gray, q0, q1, half_width=fit_half_width, trim=trim, camera=camera, min_contrast=0.015)
    choice: EdgeFit | None = None
    kind = ""
    other: EdgeFit | None = None
    other_kind = ""
    if step.ok and bar.ok:
        # a bar explains a thin line with both flanks at the same level; prefer
        # it only when it fits clearly better than a step
        if bar.rms < 0.8 * step.rms:
            choice, kind, other, other_kind = bar, "bar", step, "step"
        else:
            choice, kind, other, other_kind = step, "step", bar, "bar"
    elif step.ok:
        choice, kind = step, "step"
    elif bar.ok:
        choice, kind = bar, "bar"
    if choice is None or choice.mid is None or choice.n is None:
        return SnapResult(False, "fit_failed")
    if kind == "step" and max_bar_px >= 4.0:
        wide = _wide_line(gray, camera, choice, max_bar_px, trim)
        if wide is not None:
            choice, kind, other, other_kind = wide, "bar", choice, "step"
    shift = float((choice.mid - c) @ n0)
    # Distinctness: gradient strength of the chosen feature against the
    # strongest transition elsewhere in the search band.  The chosen feature's
    # own footprint is excluded — both flanks of a thin line belong to it.
    excl = max(4.0, 3.0 * choice.sigma_blur + (3.0 if kind == "bar" else 2.0))
    s_fit = float((choice.mid - c) @ n)
    own = np.abs(centers - s_fit) <= excl
    first = float(sc[own].max()) if np.any(own) else best[0]
    second = float(sc[~own].max()) if np.any(~own) else 0.0
    distinct = first / max(second, 1e-9)
    rest = np.where(own, 0.0, sc)
    n_cand = 1 + int(np.sum((rest[1:-1] > rest[:-2]) & (rest[1:-1] >= rest[2:]) & (rest[1:-1] > 0.5 * first)))
    tilt_deg = float(np.degrees(np.arcsin(np.clip(choice.t[0] * t0[1] - choice.t[1] * t0[0], -1, 1))))
    if other is not None and (other.mid is None or other.n is None):
        other, other_kind = None, ""
    return SnapResult(True, "", kind, choice, shift, tilt_deg, float(distinct), n_cand, other_kind, other)
