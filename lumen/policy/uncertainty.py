"""Uncertainty of a placard target position (prompt 10).

What is being quantified
------------------------
The overlay tells the operator to put a placard corner at an image pixel.  The
*placement error* is the difference, in local-frame millimetres, between the
physical point under that pixel and the point the operation definition asks
for.  It has contributions from the whole chain:

    tag corners -> per-view mapping and tag layout -> datum lines mapped to
    the plane -> local frame -> target point -> pixel in the displayed frame

and its components are correlated (a datum line and the target are usually
seen through the *same* mapping, so much of the mapping error cancels near
the datums and grows with the offset from them).  Instead of deriving this by
hand, the random part is propagated by **linearised Monte Carlo**: parameter
perturbations are drawn from the solver's covariance (and from the covariance
of each datum's fitted image line), the non-linear chain is re-evaluated
exactly for every draw, and the pixel is mapped back to the plane with the
nominal solution.  The seed is fixed, so results are reproducible.

Random versus systematic
------------------------
* ``random`` — repeatability of this capture: pixel noise in tag corners and
  datum edges.  Reported as a standard deviation (1 sigma).
* ``systematic`` — effects that repeat identically if the same photo were
  taken again: tag size error, residual edge bias, uncharacterised lens
  distortion, panel non-flatness.  Each is evaluated by a *scenario*: the
  input is perturbed by its bound, the whole solution is recomputed, and the
  displacement of the target is the component.  Bounds are combined as
  rectangular distributions (variance ``b^2 / 3``); standard uncertainties as
  normal ones.  The plain sum of all components is reported too.

No probability statement is attached to the expanded value ``k * u_c``; how
often it covers the true error has only been examined on synthetic data
(``tests/integration/test_uncertainty_coverage.py``, docs/VALIDATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, local_scale_px_per_mm
from ..geometry.lines import Line2D
from ..geometry.localframe import LocalFrame, build_local_frame, target_rectangle
from ..schemas import Uncertainty, UncertaintyComponent
from ..vision.planar import PlanarSolution, PlanarView, TagObs, ViewLocalization, localize_view, solve_planar


@dataclass
class DatumEvidence:
    """A datum as observed: a fitted straight line in the ideal pixels of one frame."""

    kind: str  # "floor" | "side"
    frame_id: str
    mid: np.ndarray
    n: np.ndarray
    length: float
    cov: np.ndarray  # 2x2 covariance of (perpendicular shift [px], rotation [rad]) about ``mid``
    offset_mm: float = 0.0
    positive: str = "toward_panel"
    manual: bool = False
    #: True when ``mid``/``n``/``length``/``cov`` are already in plane
    #: millimetres (a datum derived from a 3-D model, e.g. the tangent line of
    #: a fitted rounded edge) instead of pixels of ``frame_id``.
    in_plane: bool = False

    def points(self, dd: float = 0.0, dth: float = 0.0) -> np.ndarray:
        """Two points of the (perturbed) line (pixels, or plane mm if ``in_plane``)."""
        c, s = np.cos(dth), np.sin(dth)
        n = np.array([c * self.n[0] - s * self.n[1], s * self.n[0] + c * self.n[1]])
        t = np.array([n[1], -n[0]])
        m = self.mid + dd * n
        h = 0.5 * max(self.length, 2.0)
        return np.stack([m - h * t, m + h * t])

    def plane_line(self, Hinv: np.ndarray, interior: np.ndarray, pts: np.ndarray | None = None) -> Line2D:
        """Datum line in the plane (configured offset applied)."""
        p = self.points() if pts is None else pts
        q = p if self.in_plane else apply_h(Hinv, p)
        line = Line2D.from_points(q[0], q[1]).oriented_towards(interior)
        if self.offset_mm:
            sign = 1.0 if self.positive == "toward_panel" else -1.0
            line = line.offset(sign * self.offset_mm)
        return line


@dataclass
class FrameConfig:
    policy: str
    nominal_angle_deg: float = 90.0
    max_axis_disagreement_deg: float | None = None
    x_positive: str = "toward_panel"
    z_positive: str = "toward_panel"
    expected_side: str | None = None


@dataclass
class TargetSpec:
    x: float
    z: float
    width: float
    height: float
    angle: float = 0.0
    reference_point: str = "center"


@dataclass(frozen=True)
class UncertaintyConfig:
    n_samples: int = 400
    seed: int = 20261125
    k: float = 2.0
    #: bound on the residual common-mode edge shift after bias correction (px)
    edge_bias_bound_px: float = 0.05
    #: ... and when the correction could not be estimated for some tag
    edge_bias_bound_uncorrected_px: float = 0.25
    #: assumed bound of radial distortion without a calibration, as relative
    #: displacement at the image corner (1 % = 0.01)
    uncalibrated_distortion_frac: float = 0.01
    #: assumed out-of-plane bound of the panel when nothing better is known (mm)
    flatness_bound_mm: float = 0.3
    #: bound of a common caliper error on measured tag sizes (mm)
    measured_common_bound_mm: float = 0.03
    #: std of an operator-drawn line that was not snapped to an edge (px)
    manual_datum_sigma_px: float = 2.5
    #: assumed viewing angle when no focal length is known at all
    default_tan_view: float = 1.0


@dataclass
class EvalContext:
    """Everything needed to evaluate the chain for one displayed frame."""

    sol: PlanarSolution
    views: list[PlanarView]  # the views the solution was computed from
    frame_id: str  # displayed frame
    floor: DatumEvidence
    side: DatumEvidence
    fcfg: FrameConfig
    target: TargetSpec
    #: localisation of the displayed frame when it is not part of ``sol``
    ext_loc: ViewLocalization | None = None
    ext_view: PlanarView | None = None
    #: per-frame image sizes and focal priors (px) for scenarios
    image_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    focal_prior_px: dict[str, float] = field(default_factory=dict)
    #: per-tag size info: id -> (provenance, standard uncertainty mm, nominal tolerance mm)
    size_info: dict[int, tuple[str, float, float]] = field(default_factory=dict)
    #: calibration parameter standard deviations (fx, fy, cx, cy, k1, ...) when known
    calibration_sigma: list[float] | None = None
    #: bias-corrected flag per tag observation (any False -> larger edge-bias bound)
    all_bias_corrected: bool = True
    out_of_plane_bound_mm: float | None = None


def _H(ctx: EvalContext, fid: str) -> np.ndarray:
    if fid in ctx.sol.views:
        return ctx.sol.views[fid].H
    if ctx.ext_loc is not None and fid == ctx.ext_loc.frame_id:
        return ctx.ext_loc.H
    raise KeyError(fid)


def chain(
    H_floor: np.ndarray,
    H_side: np.ndarray,
    floor: DatumEvidence,
    side: DatumEvidence,
    fcfg: FrameConfig,
    target: TargetSpec,
    interior: np.ndarray,
    floor_pts: np.ndarray | None = None,
    side_pts: np.ndarray | None = None,
) -> tuple[np.ndarray, LocalFrame, Line2D, Line2D]:
    """Datum image lines -> plane -> local frame -> target points (plane mm).

    Returns ``(points (5, 2): TL, TR, BR, BL, centre; frame; floor line; side line)``.
    """
    fl = floor.plane_line(np.linalg.inv(H_floor), interior, floor_pts)
    sl = side.plane_line(np.linalg.inv(H_side), interior, side_pts)
    frame = build_local_frame(
        fl,
        sl,
        interior,
        policy=fcfg.policy,
        nominal_angle_deg=fcfg.nominal_angle_deg,
        max_axis_disagreement_deg=fcfg.max_axis_disagreement_deg,
        x_positive=fcfg.x_positive,
        z_positive=fcfg.z_positive,
        expected_side=fcfg.expected_side,
    )
    rect = target_rectangle(frame, target.x, target.z, target.width, target.height, target.angle, target.reference_point)
    return np.concatenate([rect.corners, rect.center[None]]), frame, fl, sl


def _angle_of(local_pts: np.ndarray) -> float:
    w = 0.5 * ((local_pts[1] - local_pts[0]) + (local_pts[2] - local_pts[3]))
    return float(np.arctan2(w[1], w[0]))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _psd_sqrt(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return V * np.sqrt(np.clip(w, 0.0, None))


def _scale_quad(c: np.ndarray, shift_px: float) -> np.ndarray:
    """Move every edge of a quad outward by ``shift_px`` (first order)."""
    ctr = c.mean(axis=0)
    edge = float(np.mean(np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)))
    return ctr + (c - ctr) * (1.0 + 2.0 * shift_px / max(edge, 1e-9))


def _resolve(ctx: EvalContext, views: list[PlanarView], ext_view: PlanarView | None) -> dict[str, np.ndarray]:
    """Re-solve the registration for modified observations; H per frame id."""
    sol = solve_planar(views, gauge=ctx.sol.layout.gauge_id, crossvalidate=False, exclude_inconsistent=False)
    out = {fid: v.H for fid, v in sol.views.items()}
    if ext_view is not None:
        out[ext_view.frame_id] = localize_view(sol.layout, ext_view).H
    return out


def evaluate_uncertainty(ctx: EvalContext, cfg: UncertaintyConfig | None = None) -> Uncertainty:
    cfg = cfg or UncertaintyConfig()
    sol = ctx.sol
    interior = sol.layout.centroid()
    Hf_nom, Hs_nom, HF_nom = _H(ctx, ctx.floor.frame_id), _H(ctx, ctx.side.frame_id), _H(ctx, ctx.frame_id)
    HF_inv = np.linalg.inv(HF_nom)
    pts_nom, L_nom, _, _ = chain(Hf_nom, Hs_nom, ctx.floor, ctx.side, ctx.fcfg, ctx.target, interior)
    u_nom = apply_h(HF_nom, pts_nom)
    loc_nom = L_nom.to_local(pts_nom)
    ang_nom = _angle_of(loc_nom)

    def to_local_err(u: np.ndarray) -> np.ndarray:
        return L_nom.to_local(apply_h(HF_inv, u)) - loc_nom

    notes: list[str] = []
    comps: list[UncertaintyComponent] = []

    # ------------------------------------------------------------------ random
    rng = np.random.default_rng(cfg.seed)
    n_par = sol.param_cov.shape[0]
    Lc = _psd_sqrt(sol.param_cov)
    cov_floor = np.asarray(ctx.floor.cov, dtype=float)
    cov_side = np.asarray(ctx.side.cov, dtype=float)
    if ctx.floor.manual:
        cov_floor = cov_floor + np.diag([cfg.manual_datum_sigma_px**2, (2.0 * cfg.manual_datum_sigma_px / max(ctx.floor.length, 20.0)) ** 2])
    if ctx.side.manual:
        cov_side = cov_side + np.diag([cfg.manual_datum_sigma_px**2, (2.0 * cfg.manual_datum_sigma_px / max(ctx.side.length, 20.0)) ** 2])
    Lfl, Lsd = _psd_sqrt(cov_floor), _psd_sqrt(cov_side)
    ext = ctx.ext_loc if (ctx.ext_loc is not None and ctx.frame_id == ctx.ext_loc.frame_id) else None
    Lext = _psd_sqrt(ext.cov) if ext is not None else None

    def view_H(fid: str, dx: np.ndarray, dext: np.ndarray | None) -> np.ndarray:
        if ext is not None and fid == ext.frame_id:
            dp = np.zeros_like(ext.params) if dext is None else dext.copy()
            for tid, G in ext.layout_sens.items():
                if tid in sol.tag_slices:
                    a, b = sol.tag_slices[tid]
                    dp = dp + G @ dx[a:b]
            return ext.H_from_params(ext.params + dp)
        v = sol.views[fid]
        a, b = v.slice
        assert v.params is not None
        return v.H_from_params(v.params + dx[a:b])

    N = int(cfg.n_samples)
    E = np.empty((N, 5, 2))
    A = np.empty(N)
    for j in range(N):
        dx = Lc @ rng.standard_normal(n_par)
        dext = (Lext @ rng.standard_normal(Lext.shape[0])) if Lext is not None else None
        zf = Lfl @ rng.standard_normal(2)
        zs = Lsd @ rng.standard_normal(2)
        Hf = view_H(ctx.floor.frame_id, dx, dext)
        Hs = view_H(ctx.side.frame_id, dx, dext)
        HF = view_H(ctx.frame_id, dx, dext)
        try:
            pts, _, _, _ = chain(
                Hf, Hs, ctx.floor, ctx.side, ctx.fcfg, ctx.target, interior,
                ctx.floor.points(zf[0], zf[1]), ctx.side.points(zs[0], zs[1]),
            )
        except Exception:
            E[j] = np.nan
            A[j] = np.nan
            continue
        e = to_local_err(apply_h(HF, pts))
        E[j] = e
        A[j] = _wrap(_angle_of(loc_nom + e) - ang_nom)
    good = np.all(np.isfinite(E.reshape(N, -1)), axis=1)
    if int(good.sum()) < max(20, N // 4):
        notes.append("random propagation unstable: geometry is close to degenerate")
        sig = np.full((5, 2), np.inf)
        u_ang = float("inf")
    else:
        if int(good.sum()) < N:
            notes.append(f"{N - int(good.sum())} of {N} draws produced no valid frame")
        Eg = E[good]
        # robust per-axis spread; second moment about zero so a mean shift counts
        sig = np.sqrt(np.mean(Eg**2, axis=0))
        u_ang = float(np.sqrt(np.mean(A[good] ** 2)))
    u_rand_pts = np.max(sig, axis=1)
    worst = int(np.argmax(u_rand_pts))
    u_rand = float(u_rand_pts[worst])
    comps.append(
        UncertaintyComponent(
            name="pixel noise (tag corners, datum edges)",
            kind="random",
            distribution="normal",
            value_mm=u_rand,
            variance_mm2=u_rand**2,
            note="repeatability of this capture, linearised Monte Carlo",
        )
    )

    # -------------------------------------------------------------- scenarios
    sys_pts = np.zeros(5)  # accumulated systematic variance per point (mm^2)

    def add(name: str, disp: np.ndarray, distribution: str, note: str) -> None:
        nonlocal sys_pts
        per_pt = np.max(np.abs(disp), axis=1)
        v = float(np.max(per_pt))
        var_pt = per_pt**2 / (3.0 if distribution == "rectangular" else 1.0)
        sys_pts = sys_pts + var_pt
        comps.append(
            UncertaintyComponent(
                name=name, kind="systematic", distribution=distribution, value_mm=v,
                variance_mm2=float(np.max(var_pt)), note=note,
            )
        )

    def scenario(views: list[PlanarView], ext_view: PlanarView | None, floor: DatumEvidence, side: DatumEvidence) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """Ideal-pixel target points of the displayed frame for modified inputs."""
        try:
            Hs_all = _resolve(ctx, views, ext_view)
            pts, _, _, _ = chain(Hs_all[floor.frame_id], Hs_all[side.frame_id], floor, side, ctx.fcfg, ctx.target, interior)
            return apply_h(Hs_all[ctx.frame_id], pts), Hs_all
        except Exception:
            return None

    ext_view = ctx.ext_view if ext is not None else None

    # (1) tag size: common-mode scale.  Scaling every size by (1 + eps) scales
    # every metric offset from the *observed* datum features by the same factor.
    provs = {i: ctx.size_info.get(i, ("nominal", 0.0, 1.0)) for i in sol.layout.poses}
    sizes = sol.layout.sizes
    if any(p[0] != "measured" for p in provs.values()):
        eps = max(p[2] / sizes[i] for i, p in provs.items() if p[0] != "measured")
        label, note = "tag size: nominal (unmeasured) tags", f"assumed print tolerance, scale ±{eps * 100:.2f} %"
    else:
        eps = cfg.measured_common_bound_mm / min(sizes.values())
        label, note = "tag size: common caliper error", f"±{cfg.measured_common_bound_mm} mm on the tag edge, scale ±{eps * 100:.3f} %"
    # local coordinates count from the datums; the datums sit ``offset`` (along
    # +axis) from the observed features, so distances from the features are:
    rel = loc_nom + np.array([ctx.side.offset_mm, ctx.floor.offset_mm])
    add(label, eps * rel, "rectangular", note)

    # (2) tag size: independent measurement uncertainty per tag
    indep = np.zeros((5, 2))
    n_meas = 0
    for i, p in provs.items():
        if p[0] != "measured" or p[1] <= 0:
            continue
        n_meas += 1
        f = 1.0 + p[1] / sizes[i]

        def mod(v: PlanarView) -> PlanarView:
            return PlanarView(v.frame_id, [replace(t, size_mm=t.size_mm * f) if t.tag_id == i else t for t in v.tags], v.camera, v.mode)

        r = scenario([mod(v) for v in ctx.views], mod(ext_view) if ext_view is not None else None, ctx.floor, ctx.side)
        if r is not None:
            indep += to_local_err(r[0]) ** 2
    if n_meas:
        add("tag size: per-tag measurement", np.sqrt(indep), "normal", "independent caliper repeatability of each measured tag")

    # (3) residual edge bias, common in sign to all tags of a capture.  Each
    # observation gets the bound that matches how it was measured: small when
    # its bias was estimated from the tag interior, large when the tag was too
    # small in the image for that.
    def bound(t: TagObs) -> float:
        return cfg.edge_bias_bound_px if t.bias_corrected else cfg.edge_bias_bound_uncorrected_px

    def shift(v: PlanarView) -> PlanarView:
        return PlanarView(v.frame_id, [replace(t, corners=_scale_quad(t.corners, bound(t))) for t in v.tags], v.camera, v.mode)

    all_obs = [t for v in ctx.views for t in v.tags] + ([t for t in ext_view.tags] if ext_view is not None else [])
    n_unc = sum(1 for t in all_obs if not t.bias_corrected)
    r = scenario([shift(v) for v in ctx.views], shift(ext_view) if ext_view is not None else None, ctx.floor, ctx.side)
    if r is not None:
        add(
            "edge localisation bias (tone curve / sharpening)",
            to_local_err(r[0]),
            "rectangular",
            f"tag borders shifted by {cfg.edge_bias_bound_px} px"
            + (f"; {n_unc} of {len(all_obs)} tag observations too small for the interior correction: {cfg.edge_bias_bound_uncorrected_px} px" if n_unc else ""),
        )
    else:
        notes.append("edge-bias scenario could not be evaluated")

    # (4) lens distortion
    def distortion_scenario(make_true: Any, make_nom: Any) -> np.ndarray | None:
        """``make_nom(fid)`` / ``make_true(fid)`` give the nominal and the
        hypothesised true camera of a frame (``None`` = ideal pinhole)."""

        def reproject(pts: np.ndarray, fid: str) -> np.ndarray:
            cn, ct = make_nom(fid), make_true(fid)
            raw = cn.distort_pixels(pts) if cn is not None else pts
            return ct.undistort_pixels(raw) if ct is not None else raw

        def modv(v: PlanarView) -> PlanarView:
            return PlanarView(v.frame_id, [replace(t, corners=reproject(t.corners, v.frame_id)) for t in v.tags], v.camera, v.mode)

        def modd(d: DatumEvidence) -> DatumEvidence:
            if d.in_plane:
                return d
            p = reproject(d.points(), d.frame_id)
            t = p[1] - p[0]
            L = float(np.linalg.norm(t))
            t = t / L
            n = np.array([-t[1], t[0]])
            if float(n @ d.n) < 0:
                n = -n
            return replace(d, mid=0.5 * (p[0] + p[1]), n=n, length=L)

        r2 = scenario([modv(v) for v in ctx.views], modv(ext_view) if ext_view is not None else None, modd(ctx.floor), modd(ctx.side))
        if r2 is None:
            return None
        u_true_ideal, _ = r2
        ct, cn = make_true(ctx.frame_id), make_nom(ctx.frame_id)
        raw_true = ct.distort_pixels(u_true_ideal) if ct is not None else u_true_ideal
        back = cn.undistort_pixels(raw_true) if cn is not None else raw_true
        return to_local_err(back)

    cams = {v.frame_id: v.camera for v in ctx.views}
    if ext_view is not None:
        cams[ext_view.frame_id] = ext_view.camera
    uncal = [fid for fid, c in cams.items() if c is None]
    if uncal:
        frac = cfg.uncalibrated_distortion_frac

        def unit_cam(fid: str, k1: float) -> CameraModel | None:
            if cams.get(fid) is not None:
                return cams[fid]
            w, h = ctx.image_sizes.get(fid, (0, 0))
            if not w or not h:
                return None
            half_diag = 0.5 * float(np.hypot(w, h))
            return CameraModel(w, h, half_diag, half_diag, (w - 1) * 0.5, (h - 1) * 0.5, (k1, 0.0, 0.0, 0.0, 0.0))

        worst_d = None
        for sgn in (1.0, -1.0):
            d = distortion_scenario(lambda fid, s=sgn: unit_cam(fid, s * frac), lambda fid: cams.get(fid))
            if d is not None and (worst_d is None or np.max(np.abs(d)) > np.max(np.abs(worst_d))):
                worst_d = d
        if worst_d is not None:
            add(
                "lens distortion: not characterised",
                worst_d,
                "rectangular",
                f"no calibration; assumed radial distortion up to ±{frac * 100:.1f} % at the image corner",
            )
        else:
            notes.append("uncalibrated-distortion scenario could not be evaluated")
    cal_sigma = ctx.calibration_sigma
    if cal_sigma is not None and len(cal_sigma) >= 5 and any(c is not None for c in cams.values()):
        sk1 = float(cal_sigma[4])
        sf = float(cal_sigma[0])

        def pert(fid: str, dk1: float, df_rel: float) -> CameraModel | None:
            c = cams.get(fid)
            if c is None:
                return None
            d = list(c.dist)
            d[0] += dk1
            return CameraModel(c.width, c.height, c.fx * (1 + df_rel), c.fy * (1 + df_rel), c.cx, c.cy, tuple(d), source=c.source)  # type: ignore[arg-type]

        # k1 acts on normalised coordinates, so it is independent of resolution
        d1 = distortion_scenario(lambda fid: pert(fid, sk1, 0.0), lambda fid: cams.get(fid))
        if d1 is not None:
            add("lens distortion: calibration uncertainty", d1, "normal", f"k1 standard deviation {sk1:.2e} from the calibration")
        if any(v.mode == "pose" for v in sol.views.values()) and sf > 0:
            pass  # focal-length sensitivity of pose-mode views is covered by the mode-agreement gate
    elif any(c is not None for c in cams.values()):
        notes.append("calibration profile carries no parameter uncertainties; its distortion error is not budgeted")

    # (5) panel flatness / out-of-plane deviation seen at an angle
    h_bound = ctx.out_of_plane_bound_mm if ctx.out_of_plane_bound_mm is not None else cfg.flatness_bound_mm
    tan_view = cfg.default_tan_view
    f_px = ctx.focal_prior_px.get(ctx.frame_id)
    cam_F = cams.get(ctx.frame_id)
    if cam_F is not None:
        f_px = cam_F.fx
    if f_px:
        w, h = ctx.image_sizes.get(ctx.frame_id, (0, 0))
        cx, cy = ((w - 1) * 0.5, (h - 1) * 0.5) if cam_F is None else (cam_F.cx, cam_F.cy)
        from ..geometry.homography import pose_from_homography

        K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1.0]])
        try:
            T = pose_from_homography(K, HF_nom)
            C = T.center  # camera centre in plane coordinates (z = height above plane)
            lateral = np.linalg.norm(pts_nom - C[:2], axis=1)
            tan_view = float(np.max(lateral) / max(abs(C[2]), 1e-6))
        except Exception:
            pass
    else:
        notes.append("no focal length known: viewing angle assumed 45° for the flatness term")
    d_flat = np.full((5, 2), h_bound * tan_view)
    add(
        "panel flatness (out-of-plane x viewing angle)",
        d_flat,
        "rectangular",
        f"out-of-plane bound {h_bound:.2f} mm" + (" (assumed)" if ctx.out_of_plane_bound_mm is None else " (from evidence)") + f", tan(view) {tan_view:.2f}",
    )

    # ------------------------------------------------------------- combine
    var_pts = u_rand_pts**2 + sys_pts
    uc_pts = np.sqrt(var_pts)
    u_c = float(np.max(uc_pts))
    bias_var = float(np.max(sys_pts))
    lin = float(sum(c.value_mm for c in comps if c.kind == "systematic"))
    return Uncertainty(
        method="linearized_monte_carlo",
        seed=cfg.seed,
        n_samples=N,
        k=cfg.k,
        u_random_mm=u_rand,
        u_random_xz_mm=[float(sig[worst, 0]), float(sig[worst, 1])],
        bias_bound_mm=float(np.sqrt(bias_var)),
        bias_linear_mm=lin,
        combined_standard_mm=u_c,
        expanded_mm=float(cfg.k * u_c),
        u_angle_rad=u_ang,
        per_corner_expanded_mm=[float(cfg.k * v) for v in uc_pts[:4]],
        components=comps,
        notes=notes,
    )


def local_scale_mm_per_px(H: np.ndarray, plane_pts: np.ndarray) -> float:
    """Millimetres per pixel of a mapping near given plane points."""
    return float(1.0 / max(np.median(local_scale_px_per_mm(H, plane_pts)), 1e-9))


def tag_obs_with(t: TagObs, **kw: Any) -> TagObs:
    return replace(t, **kw)
