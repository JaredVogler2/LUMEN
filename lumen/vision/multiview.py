"""Multi-view camera and tag localisation with bundle adjustment (prompt 05).

Problem
-------
Tags are stuck anywhere, at unknown relative poses, and stay put.  Several
photos (or selected video frames) each see some of them.  We want every
camera pose and every tag pose in one metric frame, without assuming the tags
are coplanar — coplanarity is a *result* that is tested, not an input.

Formulation
-----------
* World frame = frame of the gauge tag (its pose is fixed: that removes the
  7-dof gauge freedom together with the known tag sizes).
* Unknowns: ``T_cam_world`` per view (6), ``T_world_tag`` per other tag (6),
  optionally shared intrinsics (focal length, principal point, k1, k2).
* Residuals: reprojection of the four corners of every observed tag, whitened
  by the 8x8 corner covariance from the edge fits, with a Huber loss.
* Updates are on SE(3): ``T <- exp(delta) T`` (left-multiplicative retraction),
  analytic Jacobians, Levenberg–Marquardt with dense normal equations (the
  problems are small: a handful of tags, at most a few dozen views).
* Every tag is metric (its measured edge), so each one constrains scale.

Initialisation handles the planar pose ambiguity explicitly: both IPPE
branches of every tag observation are kept, tag-to-tag transforms are chosen
by consensus across views, and after optimisation views whose two branches are
not clearly separated are re-tested from the other branch.

Checks reported to the caller (all can fail closed in the gates)
----------------------------------------------------------------
connected observation graph · positive depth · per-observation chi-square
(moved or misread tag) · free-size test per tag (wrong declared size) ·
held-out views · viewpoint diversity · conditioning / posterior sigmas ·
coplanarity test with its own power · agreement of an attached calibration
with a self-calibration from the tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..errors import GeometryError
from ..geometry.camera import CameraModel
from ..geometry.homography import dlt_homography
from ..geometry.se3 import Pose, exp_so3, skew
from .markers import tag_object_points
from .planar import TagObs, _whitener


@dataclass
class MVView:
    frame_id: str
    tags: list[TagObs]  # corners in *ideal* pixels of ``camera`` (or raw pixels when uncalibrated)
    camera: CameraModel | None = None
    #: raw (as recorded) corners per tag id, needed only when distortion is refined
    raw: dict[int, np.ndarray] = field(default_factory=dict)
    #: (width, height) of the frame; needed for self-calibration without a camera
    image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class MVConfig:
    huber_k: float = 3.0
    max_iterations: int = 60
    #: chi-square per residual of one tag observation above which it is suspect
    max_obs_chi2: float = 9.0
    #: relative size deviation always treated as a mismatch / smallest callable one
    size_max_frac: float = 0.02
    size_min_frac: float = 0.004
    size_z: float = 3.5
    #: coplanarity: tolerated out-of-plane distance and tilt beyond their uncertainty
    coplanar_max_mm: float = 1.5
    coplanar_max_deg: float = 2.5
    coplanar_z: float = 3.5
    #: the coplanarity test is only called "established" when it could have seen this
    coplanar_power_mm: float = 1.5
    coplanar_power_deg: float = 2.5
    #: minimum angular spread of viewing directions for "diverse" (degrees)
    min_view_spread_deg: float = 12.0
    ambiguity_min_delta_chi2: float = 16.0
    #: self-calibration is reported only when focal length is determined this well
    selfcal_max_rel_sigma_f: float = 0.02
    holdout_max_ratio: float = 3.0


@dataclass
class MVSolution:
    gauge_id: int
    cameras: dict[str, Pose]  # T_cam_world
    tags: dict[int, Pose]  # T_world_tag
    sizes: dict[int, float]
    intrinsics: CameraModel
    intrinsics_refined: list[str]
    rms_px: float
    reduced_chi2: float
    dof: int
    obs_chi2: dict[tuple[str, int], float]  # mean whitened squared residual per observation
    cov: np.ndarray
    param_index: dict[str, tuple[int, int]]
    excluded_tags: dict[int, list[str]] = field(default_factory=dict)
    excluded_views: dict[str, list[str]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def camera_center(self, frame_id: str) -> np.ndarray:
        return self.cameras[frame_id].center

    def tag_corners_world(self, tag_id: int) -> np.ndarray:
        return self.tags[tag_id].apply(tag_object_points(self.sizes[tag_id]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gauge_id": self.gauge_id,
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "camera_centers": {k: v.center.round(3).tolist() for k, v in self.cameras.items()},
            "tags": {str(k): v.to_dict() for k, v in sorted(self.tags.items())},
            "sizes": {str(k): v for k, v in sorted(self.sizes.items())},
            "intrinsics": self.intrinsics.to_dict(),
            "intrinsics_refined": list(self.intrinsics_refined),
            "rms_px": round(self.rms_px, 4),
            "reduced_chi2": round(self.reduced_chi2, 4),
            "dof": self.dof,
            "excluded_tags": {str(k): v for k, v in self.excluded_tags.items()},
            "excluded_views": dict(self.excluded_views),
            "flags": list(self.flags),
            "checks": self.checks,
        }


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------
def _ippe_square(K: np.ndarray, corners: np.ndarray, size: float) -> list[tuple[Pose, float]]:
    """Both IPPE branches ``T_cam_tag`` of one tag, best first, with RMS px."""
    obj = tag_object_points(size).astype(np.float64)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        n = 0
    out: list[tuple[Pose, float]] = []
    for i in range(int(n)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = np.asarray(tvecs[i], dtype=float).ravel()
        if t[2] <= 0:
            continue
        e = float(np.asarray(errs).ravel()[i])
        out.append((Pose(R, t), e))
    out.sort(key=lambda x: x[1])
    return out


def _pose_close(a: Pose, b: Pose, scale: float) -> bool:
    ang, dist = a.distance_to(b)
    return ang < np.deg2rad(6.0) and dist < 0.06 * scale + 4.0


def focal_from_tags(views: list[MVView], width: int, height: int) -> float | None:
    """Closed-form focal length from tag homographies (Zhang's constraints with
    square pixels and the principal point at the image centre).

    Returns ``None`` when the views carry no usable perspective (all tags seen
    almost frontally), in which case the focal length is not observable.
    """
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    num: list[float] = []
    den: list[float] = []
    for v in views:
        for t in v.tags:
            H = dlt_homography(tag_object_points(t.size_mm)[:, :2], t.corners - np.array([cx, cy]))
            h1, h2 = H[:, 0], H[:, 1]
            # w = diag(1/f^2, 1/f^2, 1):  h1^T w h2 = 0 ; h1^T w h1 = h2^T w h2
            a1 = h1[0] * h2[0] + h1[1] * h2[1]
            b1 = h1[2] * h2[2]
            a2 = (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)
            b2 = h1[2] ** 2 - h2[2] ** 2
            for a, b in ((a1, b1), (a2, b2)):
                # a / f^2 + b = 0  ->  f^2 = -a / b ; weight by |b| (perspective content)
                num.append(-a * b)
                den.append(b * b)
    if not den or sum(den) <= 0:
        return None
    f2 = sum(num) / sum(den)
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = float(np.sqrt(f2))
    if not (0.25 * max(width, height) < f < 8.0 * max(width, height)):
        return None
    return f


def _initialise(views: list[MVView], K: np.ndarray, gauge: int) -> tuple[dict[str, Pose], dict[int, Pose], dict[str, list[tuple[Pose, float]]], list[int], list[str]]:
    """Initial camera and tag poses by consensus over IPPE branches.

    Returns ``(T_cam_world, T_world_tag, alternatives per view, unreachable
    tags, unreachable views)``.
    """
    cands: dict[tuple[str, int], list[tuple[Pose, float]]] = {}
    size: dict[int, float] = {}
    for v in views:
        for t in v.tags:
            c = _ippe_square(K, t.corners, t.size_mm)
            if c:
                cands[(v.frame_id, t.tag_id)] = c
                size[t.tag_id] = t.size_mm
    ids = sorted(size)
    # --- pairwise tag-to-tag hypotheses T_i_j = inv(T_c_i) @ T_c_j
    hyp: dict[tuple[int, int], list[tuple[Pose, float, str]]] = {}
    for v in views:
        present = [t.tag_id for t in v.tags if (v.frame_id, t.tag_id) in cands]
        for i in present:
            for j in present:
                if i >= j:
                    continue
                for Ti, ei in cands[(v.frame_id, i)]:
                    for Tj, ej in cands[(v.frame_id, j)]:
                        hyp.setdefault((i, j), []).append((Ti.inv() @ Tj, ei + ej, v.frame_id))
    edges: dict[tuple[int, int], tuple[Pose, float]] = {}
    for (i, j), hs in hyp.items():
        scale = float(np.median([np.linalg.norm(h[0].t) for h in hs])) + max(size[i], size[j])
        best, best_score = None, (-1, 0.0)
        for h in hs:
            support = {g[2] for g in hs if _pose_close(h[0], g[0], scale)}
            score = (len(support), -h[1])
            if score > best_score:
                best, best_score = h, score
        assert best is not None
        edges[(i, j)] = (best[0], float(best_score[0]) - 0.01 * best[1])
    # --- spanning tree from the gauge over the strongest edges
    T_w_t: dict[int, Pose] = {gauge: Pose.identity()}
    frontier = [gauge]
    while frontier:
        nxt: list[int] = []
        order = sorted(((conf, i, j) for (i, j), (_, conf) in edges.items()), reverse=True)
        for _, i, j in order:
            Tij = edges[(i, j)][0]
            if i in T_w_t and j not in T_w_t and i in frontier:
                T_w_t[j] = T_w_t[i] @ Tij
                nxt.append(j)
            elif j in T_w_t and i not in T_w_t and j in frontier:
                T_w_t[i] = T_w_t[j] @ Tij.inv()
                nxt.append(i)
        frontier = nxt
    unreachable = [i for i in ids if i not in T_w_t]

    # --- camera poses: PnP on all corners of the reachable tags
    T_c_w: dict[str, Pose] = {}
    alts: dict[str, list[tuple[Pose, float]]] = {}
    lost_views: list[str] = []
    for v in views:
        tags = [t for t in v.tags if t.tag_id in T_w_t and (v.frame_id, t.tag_id) in cands]
        if not tags:
            lost_views.append(v.frame_id)
            continue
        obj = np.concatenate([T_w_t[t.tag_id].apply(tag_object_points(t.size_mm)) for t in tags])
        img = np.concatenate([t.corners for t in tags]).astype(np.float64)
        # candidates: each tag's two branches mapped to the world, scored on all corners
        pool: list[tuple[Pose, float]] = []
        for t in tags:
            for Tct, _ in cands[(v.frame_id, t.tag_id)]:
                T = Tct @ T_w_t[t.tag_id].inv()
                Xc = T.apply(obj)
                if np.any(Xc[:, 2] <= 0):
                    continue
                uv = (Xc[:, :2] / Xc[:, 2:3]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
                pool.append((T, float(np.sqrt(np.mean(np.sum((uv - img) ** 2, axis=1))))))
        if not pool:
            lost_views.append(v.frame_id)
            continue
        pool.sort(key=lambda x: x[1])
        # refine the two most distinct candidates with a few Gauss-Newton steps
        distinct = [pool[0]]
        for p in pool[1:]:
            if not _pose_close(p[0], distinct[0][0], float(np.linalg.norm(distinct[0][0].t))):
                distinct.append(p)
                break
        refined = []
        for T0, _ in distinct:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec=cv2.Rodrigues(T0.R)[0], tvec=T0.t.reshape(3, 1).copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            T = Pose(cv2.Rodrigues(rvec)[0], np.asarray(tvec).ravel())
            Xc = T.apply(obj)
            if np.any(Xc[:, 2] <= 0):
                continue
            uv = (Xc[:, :2] / Xc[:, 2:3]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
            refined.append((T, float(np.sqrt(np.mean(np.sum((uv - img) ** 2, axis=1))))))
        if not refined:
            lost_views.append(v.frame_id)
            continue
        refined.sort(key=lambda x: x[1])
        T_c_w[v.frame_id] = refined[0][0]
        alts[v.frame_id] = refined[1:]
    return T_c_w, T_w_t, alts, unreachable, lost_views


# --------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------
_INTR = ("f", "cx", "cy", "k1", "k2")


class _BA:
    def __init__(
        self,
        views: list[MVView],
        cams: dict[str, Pose],
        tags: dict[int, Pose],
        gauge: int,
        cam0: CameraModel,
        refine: list[str],
        free_scale: bool = False,
        use_raw: bool = False,
    ) -> None:
        self.views = [v for v in views if v.frame_id in cams]
        self.cams = dict(cams)
        self.tags = dict(tags)
        self.gauge = gauge
        self.refine = [r for r in _INTR if r in refine]
        self.intr = np.array([0.5 * (cam0.fx + cam0.fy), cam0.cx, cam0.cy, cam0.dist[0], cam0.dist[1]], dtype=float)
        self.aspect = cam0.fy / cam0.fx
        self.dist_rest = (cam0.dist[2], cam0.dist[3], cam0.dist[4])
        self.cam0 = cam0
        self.use_raw = use_raw  # residuals in raw pixels with distortion in the model
        self.free_scale = free_scale
        self.scale = {i: 1.0 for i in tags}
        # --- parameter layout
        self.index: dict[str, tuple[int, int]] = {}
        k = 0
        for v in self.views:
            self.index[f"cam:{v.frame_id}"] = (k, k + 6)
            k += 6
        self.free_tags = [i for i in sorted(tags) if i != gauge]
        for i in self.free_tags:
            self.index[f"tag:{i}"] = (k, k + 6)
            k += 6
        for name in self.refine:
            self.index[f"intr:{name}"] = (k, k + 1)
            k += 1
        self.scale_tags = self.free_tags if free_scale else []
        for i in self.scale_tags:
            self.index[f"scale:{i}"] = (k, k + 1)
            k += 1
        self.n_params = k
        self.obs: list[tuple[int, TagObs, np.ndarray, np.ndarray]] = []
        for vi, v in enumerate(self.views):
            for t in v.tags:
                if t.tag_id not in tags:
                    continue
                target = v.raw[t.tag_id] if (use_raw and t.tag_id in v.raw) else t.corners
                self.obs.append((vi, t, _whitener(t.cov), np.asarray(target, dtype=float)))
        self.n_res = 8 * len(self.obs)

    # ---------------------------------------------------------------- model
    def camera_model(self) -> CameraModel:
        f, cx, cy, k1, k2 = self.intr
        return CameraModel(self.cam0.width, self.cam0.height, float(f), float(f * self.aspect), float(cx), float(cy), (float(k1), float(k2), *self.dist_rest), source=self.cam0.source)

    def _project(self, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixels, d(pixel)/d(Xc) (N,2,3) and d(pixel)/d(intrinsics) (N,2,5)."""
        f, cx, cy, k1, k2 = self.intr
        fy = f * self.aspect
        X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        x, y = X / Z, Y / Z
        n = Xc.shape[0]
        Jn = np.zeros((n, 2, 3))
        Jn[:, 0, 0] = 1 / Z
        Jn[:, 0, 2] = -x / Z
        Jn[:, 1, 1] = 1 / Z
        Jn[:, 1, 2] = -y / Z
        if self.use_raw:
            p1, p2, k3 = self.dist_rest
            r2 = x * x + y * y
            rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            drad = k1 + r2 * (2 * k2 + 3 * k3 * r2)
            xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            Jd = np.empty((n, 2, 2))
            Jd[:, 0, 0] = rad + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
            Jd[:, 0, 1] = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
            Jd[:, 1, 0] = Jd[:, 0, 1]
            Jd[:, 1, 1] = rad + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
        else:
            xd, yd = x, y
            r2 = x * x + y * y
            Jd = np.broadcast_to(np.eye(2), (n, 2, 2))
        uv = np.stack([f * xd + cx, fy * yd + cy], axis=1)
        F = np.array([[f, 0.0], [0.0, fy]])
        JX = np.einsum("ab,nbc,ncd->nad", F, Jd, Jn)
        JI = np.zeros((n, 2, 5))
        JI[:, 0, 0] = xd
        JI[:, 1, 0] = yd * self.aspect
        JI[:, 0, 1] = 1.0
        JI[:, 1, 2] = 1.0
        if self.use_raw:
            JI[:, 0, 3] = f * x * r2
            JI[:, 1, 3] = fy * y * r2
            JI[:, 0, 4] = f * x * r2 * r2
            JI[:, 1, 4] = fy * y * r2 * r2
        return uv, JX, JI

    def evaluate(self, jac: bool = True) -> tuple[np.ndarray, np.ndarray | None, bool]:
        r = np.empty(self.n_res)
        J = np.zeros((self.n_res, self.n_params)) if jac else None
        front = True
        for k, (vi, t, W, target) in enumerate(self.obs):
            v = self.views[vi]
            Tcw = self.cams[v.frame_id]
            Twt = self.tags[t.tag_id]
            s = self.scale[t.tag_id]
            Xt = tag_object_points(t.size_mm) * s
            Xw = Twt.apply(Xt)
            Xc = Tcw.apply(Xw)
            if np.any(Xc[:, 2] <= 1e-6):
                front = False
                Xc = Xc.copy()
                Xc[:, 2] = np.maximum(Xc[:, 2], 1e-3)
            uv, JX, JI = self._project(Xc)
            r[8 * k : 8 * k + 8] = W @ (uv - target).ravel()
            if not jac:
                continue
            assert J is not None
            a, b = self.index[f"cam:{v.frame_id}"]
            Jc = np.zeros((8, 6))
            for c in range(4):
                Jc[2 * c : 2 * c + 2, 0:3] = JX[c] @ (-skew(Xc[c]))
                Jc[2 * c : 2 * c + 2, 3:6] = JX[c]
            J[8 * k : 8 * k + 8, a:b] = W @ Jc
            if t.tag_id != self.gauge:
                a, b = self.index[f"tag:{t.tag_id}"]
                Jt = np.zeros((8, 6))
                for c in range(4):
                    M = JX[c] @ Tcw.R
                    Jt[2 * c : 2 * c + 2, 0:3] = M @ (-skew(Xw[c]))
                    Jt[2 * c : 2 * c + 2, 3:6] = M
                J[8 * k : 8 * k + 8, a:b] = W @ Jt
            for name in self.refine:
                a, _ = self.index[f"intr:{name}"]
                J[8 * k : 8 * k + 8, a] = W @ JI[:, :, _INTR.index(name)].ravel()
            if t.tag_id in self.scale_tags:
                a, _ = self.index[f"scale:{t.tag_id}"]
                dX = Tcw.R @ Twt.R @ (tag_object_points(t.size_mm)).T  # d Xc / d s, columns per corner
                J[8 * k : 8 * k + 8, a] = W @ np.concatenate([JX[c] @ dX[:, c] for c in range(4)])
        return r, J, front

    def apply(self, dx: np.ndarray) -> "_BA":
        import copy

        new = copy.copy(self)
        new.cams = dict(self.cams)
        new.tags = dict(self.tags)
        new.intr = self.intr.copy()
        new.scale = dict(self.scale)
        for v in self.views:
            a, b = self.index[f"cam:{v.frame_id}"]
            new.cams[v.frame_id] = self.cams[v.frame_id].retract(dx[a:b])
        for i in self.free_tags:
            a, b = self.index[f"tag:{i}"]
            new.tags[i] = self.tags[i].retract(dx[a:b])
        for name in self.refine:
            a, _ = self.index[f"intr:{name}"]
            new.intr[_INTR.index(name)] += dx[a]
        for i in self.scale_tags:
            a, _ = self.index[f"scale:{i}"]
            new.scale[i] = self.scale[i] + dx[a]
        return new


def _huber_w(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-300))


def _optimise(ba: _BA, cfg: MVConfig, robust: bool = True) -> tuple[_BA, np.ndarray, np.ndarray, np.ndarray, bool]:
    """LM on the manifold.  Returns (state, residuals, jacobian, weights, converged)."""
    r, J, _ = ba.evaluate()
    assert J is not None
    w = _huber_w(r, cfg.huber_k) if robust else np.ones_like(r)
    cost = float(np.sum(w * r * r))
    lam = 1e-3
    converged = False
    for _ in range(cfg.max_iterations):
        Jw = J * w[:, None]
        A = J.T @ Jw
        g = Jw.T @ r
        diag = np.clip(np.diag(A), 1e-9, None)
        improved = False
        for _try in range(10):
            try:
                dx = -np.linalg.solve(A + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            cand = ba.apply(dx)
            rn, _, front = cand.evaluate(jac=False)
            wn = _huber_w(rn, cfg.huber_k) if robust else np.ones_like(rn)
            # compare with the Huber cost itself so re-weighting cannot fake progress
            cn = float(np.sum(np.where(np.abs(rn) <= cfg.huber_k, rn * rn, 2 * cfg.huber_k * np.abs(rn) - cfg.huber_k**2))) if robust else float(rn @ rn)
            c0 = float(np.sum(np.where(np.abs(r) <= cfg.huber_k, r * r, 2 * cfg.huber_k * np.abs(r) - cfg.huber_k**2))) if robust else cost
            if front and np.isfinite(cn) and cn < c0:
                small = float(np.linalg.norm(dx)) < 1e-9 or (c0 - cn) < 1e-10 * max(c0, 1e-12)
                ba = cand
                r, J, _ = ba.evaluate()
                assert J is not None
                w = wn
                cost = float(np.sum(w * r * r))
                lam = max(lam * 0.3, 1e-10)
                improved = True
                converged = small
                break
            lam *= 10
        if not improved:
            converged = True
            break
        if converged:
            break
    return ba, r, J, w, converged


# --------------------------------------------------------------------------
# Public solver
# --------------------------------------------------------------------------
def _plane_fit(P: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c)
    n = Vt[-1]
    return n, float(n @ c), (P - c) @ n


def _coplanarity(sol_tags: dict[int, Pose], sizes: dict[int, float], cov: np.ndarray, index: dict[str, tuple[int, int]], gauge: int, cfg: MVConfig) -> dict[str, Any]:
    """Is every tag on one plane?  Reported with the test's own resolving power.

    Each tag is compared with the plane of the *other* tags (so one raised tag
    cannot drag the reference plane toward itself): the out-of-plane offset of
    its centre and the tilt of its normal, each with a standard deviation
    propagated from the bundle-adjustment covariance.  With only two tags the
    reference is the other tag's own face.
    """
    ids = sorted(sol_tags)
    corners = {i: sol_tags[i].apply(tag_object_points(sizes[i])) for i in ids}
    P_all = np.concatenate([corners[i] for i in ids])
    n_all, d_all, res_all = _plane_fit(P_all)
    if float(n_all @ sol_tags[gauge].R[:, 2]) < 0:
        n_all, d_all, res_all = -n_all, -d_all, -res_all
    per: dict[str, Any] = {}
    worst_mm = worst_deg = 0.0
    sig_mm_all: list[float] = []
    sig_deg_all: list[float] = []
    consistent = True
    for i in ids:
        others = [j for j in ids if j != i]
        if not others:
            per[str(i)] = {"offset_mm": 0.0, "offset_sigma_mm": 0.0, "tilt_deg": 0.0, "tilt_sigma_deg": 0.0, "on_plane": True}
            continue
        if len(others) == 1:
            Tj = sol_tags[others[0]]
            n = Tj.R[:, 2]
            d = float(n @ Tj.t)
        else:
            n, d, _ = _plane_fit(np.concatenate([corners[j] for j in others]))
        T = sol_tags[i]
        if float(n @ T.R[:, 2]) < 0:
            n, d = -n, -d
        off = float(n @ T.t - d)
        tilt = float(np.degrees(np.arccos(np.clip(abs(float(n @ T.R[:, 2])), 0.0, 1.0))))
        # uncertainty of this comparison: the tag's own pose and the poses defining the plane
        var_mm = var_rad = 0.0
        for j in [i] + others:
            if j == gauge or f"tag:{j}" not in index:
                continue
            a, b = index[f"tag:{j}"]
            C = cov[a:b, a:b]
            Tj = sol_tags[j]
            wgt = 1.0 if j == i else 1.0 / len(others)
            g = np.concatenate([np.cross(Tj.t, n), n]) * wgt
            var_mm += float(max(g @ C @ g, 0.0))
            Rn = np.eye(3) - np.outer(n, n)
            var_rad += float(max(np.trace(Rn @ C[:3, :3] @ Rn), 0.0)) * wgt**2
        s_mm = float(np.sqrt(var_mm))
        s_deg = float(np.degrees(np.sqrt(var_rad)))
        sig_mm_all.append(s_mm)
        sig_deg_all.append(s_deg)
        bad = (abs(off) > cfg.coplanar_max_mm + cfg.coplanar_z * s_mm) or (tilt > cfg.coplanar_max_deg + cfg.coplanar_z * s_deg)
        consistent &= not bad
        worst_mm = max(worst_mm, abs(off))
        worst_deg = max(worst_deg, tilt)
        per[str(i)] = {"offset_mm": round(off, 4), "offset_sigma_mm": round(s_mm, 4), "tilt_deg": round(tilt, 4), "tilt_sigma_deg": round(s_deg, 4), "on_plane": not bad}
    power_mm = float(max(sig_mm_all)) * cfg.coplanar_z if sig_mm_all else float("inf")
    power_deg = float(max(sig_deg_all)) * cfg.coplanar_z if sig_deg_all else float("inf")
    established = bool(consistent and len(ids) >= 2 and power_mm <= cfg.coplanar_power_mm and power_deg <= cfg.coplanar_power_deg)
    off_plane = sorted(int(k) for k, v in per.items() if not v["on_plane"])
    return {
        "normal": n_all.round(6).tolist(),
        "d": round(d_all, 4),
        "rms_mm": round(float(np.sqrt(np.mean(res_all**2))), 4),
        "max_mm": round(float(np.max(np.abs(res_all))), 4),
        "worst_center_offset_mm": round(worst_mm, 4),
        "worst_tilt_deg": round(worst_deg, 4),
        "tags": per,
        "off_plane_tags": off_plane,
        "consistent_with_one_plane": bool(consistent),
        "resolving_power_mm": None if not np.isfinite(power_mm) else round(power_mm, 4),
        "resolving_power_deg": None if not np.isfinite(power_deg) else round(power_deg, 4),
        "established": established,
        "verdict": "coplanar" if established else ("not_coplanar" if not consistent else "not_established"),
    }


def _diversity(cams: dict[str, Pose], centroid: np.ndarray) -> dict[str, Any]:
    dirs = []
    dists = []
    for T in cams.values():
        v = T.center - centroid
        dists.append(float(np.linalg.norm(v)))
        dirs.append(v / max(np.linalg.norm(v), 1e-9))
    D = np.array(dirs)
    spread = 0.0
    if len(D) >= 2:
        cosm = np.clip(D @ D.T, -1, 1)
        spread = float(np.degrees(np.arccos(cosm.min())))
    return {"n_views": len(cams), "max_view_angle_deg": round(spread, 3), "distance_mm": [round(min(dists), 1), round(max(dists), 1)] if dists else None}


def solve_multiview(
    views: list[MVView],
    *,
    camera: CameraModel | None = None,
    gauge: int | None = None,
    refine_intrinsics: tuple[str, ...] = (),
    cfg: MVConfig | None = None,
    checks: bool = True,
) -> MVSolution:
    """Bundle-adjust all views and tags.

    ``camera`` gives the shared intrinsics (all views must use the same lens
    configuration).  Without it the focal length is initialised from the tags
    themselves and must be listed in ``refine_intrinsics`` (self-calibration).
    """
    cfg = cfg or MVConfig()
    views = [MVView(v.frame_id, list(v.tags), v.camera, dict(v.raw), v.image_size) for v in views if v.tags]
    if not views:
        raise GeometryError("no_tags", "No accepted AprilTag in any view.")
    cam0 = camera or next((v.camera for v in views if v.camera is not None), None)
    refine = list(refine_intrinsics)
    flags: list[str] = []
    if cam0 is None:
        # Self-calibration: seed the focal length from the tags themselves.
        size = next((v.image_size for v in views if v.image_size), None)
        if size is None:
            raise GeometryError("intrinsics_required", "Multi-view localisation needs camera intrinsics or the image size for self-calibration.")
        f0 = focal_from_tags(views, size[0], size[1])
        if f0 is None:
            f0 = 0.9 * max(size)
            flags.append("focal_seed_unobservable")
        cam0 = CameraModel(size[0], size[1], f0, f0, (size[0] - 1) * 0.5, (size[1] - 1) * 0.5, source="self_calibration")
        if "f" not in refine:
            refine.append("f")
    use_raw = any(n in refine for n in ("k1", "k2"))
    K = cam0.without_distortion().K

    def pick_gauge(vs: list[MVView], wanted: int | None) -> int:
        ids = sorted({t.tag_id for v in vs for t in v.tags})
        if wanted is not None and wanted in ids:
            return wanted
        counts = {i: sum(1 for v in vs if any(t.tag_id == i for t in v.tags)) for i in ids}
        return sorted(ids, key=lambda i: (-counts[i], i))[0]

    gauge = pick_gauge(views, gauge)
    excluded_tags: dict[int, list[str]] = {}
    excluded_views: dict[str, list[str]] = {}
    rounds = 0
    while True:
        rounds += 1
        cams, tags, alts, unreachable, lost = _initialise(views, K, gauge)
        for i in unreachable:
            excluded_tags.setdefault(i, ["not_connected_to_gauge"])
        for fid in lost:
            excluded_views.setdefault(fid, ["no_connected_tag"])
        if (unreachable or lost) and "graph_disconnected" not in flags:
            flags.append("graph_disconnected")
        if not cams:
            raise GeometryError("no_views_localised", "No view could be localised against the gauge tag.")
        ba = _BA(views, cams, tags, gauge, cam0, refine, use_raw=use_raw)
        if ba.n_res < ba.n_params:
            raise GeometryError("underdetermined", "Too few observations for the requested unknowns.", suggestions=["Capture more views that each show at least two tags."])
        ba, r, J, w, _ = _optimise(ba, cfg)

        # --- ambiguity: retry views from their other IPPE branch
        ambiguous: list[str] = []
        for fid, alt in alts.items():
            for T_alt, _ in alt:
                trial = _BA(views, {**ba.cams, fid: T_alt}, ba.tags, gauge, ba.camera_model(), [], use_raw=use_raw)
                a, b = trial.index[f"cam:{fid}"]
                for _it in range(15):  # optimise only this camera
                    rr, JJ, _f = trial.evaluate()
                    assert JJ is not None
                    rows = np.any(JJ[:, a:b] != 0, axis=1)
                    Jv = JJ[rows][:, a:b]
                    try:
                        dx6 = -np.linalg.solve(Jv.T @ Jv + 1e-6 * np.eye(6), Jv.T @ rr[rows])
                    except np.linalg.LinAlgError:
                        break
                    dx = np.zeros(trial.n_params)
                    dx[a:b] = dx6
                    trial = trial.apply(dx)
                    if float(np.linalg.norm(dx6)) < 1e-9:
                        break
                rr, _, front_alt = trial.evaluate(jac=False)
                rows_main = np.array([ba.views[vi].frame_id == fid for vi, *_ in ba.obs]).repeat(8)
                rows_alt = np.array([trial.views[vi].frame_id == fid for vi, *_ in trial.obs]).repeat(8)
                chi_main = float(np.sum(r[rows_main] ** 2))
                chi_alt = float(np.sum(rr[rows_alt] ** 2))
                ang, _ = trial.cams[fid].distance_to(ba.cams[fid])
                if ang < np.deg2rad(1.0) or not front_alt:
                    continue  # both branches converged to the same pose, or the other is behind the camera
                if chi_alt < chi_main:
                    ba.cams[fid] = trial.cams[fid]
                    ba, r, J, w, _ = _optimise(ba, cfg)
                if abs(chi_alt - chi_main) < cfg.ambiguity_min_delta_chi2:
                    ambiguous.append(fid)

        # --- per-observation chi-square: a moved / misread tag, or a bad view
        obs_chi2: dict[tuple[str, int], float] = {}
        for k, (vi, t, _W, _tg) in enumerate(ba.obs):
            obs_chi2[(ba.views[vi].frame_id, t.tag_id)] = float(np.mean(r[8 * k : 8 * k + 8] ** 2))
        bad = {key: c for key, c in obs_chi2.items() if c > cfg.max_obs_chi2}
        if not bad or rounds >= 5:
            if bad:
                flags.append("inconsistent_observations")
            break
        by_tag: dict[int, float] = {}
        by_view: dict[str, float] = {}
        for (fid, tid), c in bad.items():
            by_tag[tid] = by_tag.get(tid, 0.0) + c
            by_view[fid] = by_view.get(fid, 0.0) + c
        n_views_now = len({v.frame_id for v in views})
        if len(by_view) == 1 and len(by_tag) >= 2 and n_views_now >= 3:
            # everything wrong is in one photo: drop the photo, keep the tags
            fid = next(iter(by_view))
            excluded_views[fid] = ["inconsistent_with_other_views"]
            views = [v for v in views if v.frame_id != fid]
        elif len(ba.tags) >= 3:
            worst = max(by_tag, key=lambda i: by_tag[i])
            # Would a different size explain it?  Free the tag sizes and look
            # at this tag's own residuals again.
            reason = "moved_or_misread"
            try:
                fs = _BA(views, ba.cams, ba.tags, gauge, ba.camera_model(), [], free_scale=True, use_raw=use_raw)
                fs, rf, _Jf, _wf, _ = _optimise(fs, cfg)
                chi_free = [float(np.mean(rf[8 * k : 8 * k + 8] ** 2)) for k, (_vi, t, _W, _tg) in enumerate(fs.obs) if t.tag_id == worst]
                ratio = fs.scale.get(worst, 1.0)
                others = [abs(fs.scale[i] - 1.0) for i in fs.scale_tags if i != worst]
                if worst == gauge:
                    # the gauge cannot be rescaled; it is the odd one if all others moved together
                    if others and min(others) > cfg.size_min_frac and np.ptp([fs.scale[i] for i in fs.scale_tags]) < 0.5 * min(others):
                        reason = "declared_size_inconsistent"
                elif chi_free and max(chi_free) <= cfg.max_obs_chi2 and abs(ratio - 1.0) > cfg.size_min_frac:
                    reason = "declared_size_inconsistent"
            except Exception:
                pass
            excluded_tags[worst] = [reason]
            views = [MVView(v.frame_id, [t for t in v.tags if t.tag_id != worst], v.camera, v.raw, v.image_size) for v in views]
            views = [v for v in views if v.tags]
            if worst == gauge:
                gauge = pick_gauge(views, None)
        else:
            flags.append("inconsistent_observations")
            break

    r, J, front = ba.evaluate()
    assert J is not None
    w = _huber_w(r, cfg.huber_k)
    dof = max(ba.n_res - ba.n_params, 0)
    chi2 = float(np.sum(w * r * r))
    red = chi2 / dof if dof > 0 else 0.0
    cov_scale = max(1.0, red)
    JW = J * np.sqrt(w)[:, None]
    try:
        cov = np.linalg.inv(JW.T @ JW) * cov_scale
        rank_ok = True
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JW.T @ JW) * cov_scale
        rank_ok = False
    # unwhitened RMS in pixels
    sq = []
    for k, (vi, t, W, target) in enumerate(ba.obs):
        e = np.linalg.solve(W, r[8 * k : 8 * k + 8])
        sq.append(float(np.mean(e.reshape(4, 2) ** 2) * 2))
    if not front:
        flags.append("negative_depth")
    if not rank_ok:
        flags.append("rank_deficient")
    if ambiguous:
        flags.append("pose_ambiguous")
    sol = MVSolution(
        gauge_id=gauge,
        cameras=dict(ba.cams),
        tags=dict(ba.tags),
        sizes={i: next(t.size_mm for v in views for t in v.tags if t.tag_id == i) for i in ba.tags},
        intrinsics=ba.camera_model(),
        intrinsics_refined=list(ba.refine),
        rms_px=float(np.sqrt(np.mean(sq))) if sq else 0.0,
        reduced_chi2=red,
        dof=dof,
        obs_chi2=obs_chi2,
        cov=cov,
        param_index=dict(ba.index),
        excluded_tags=excluded_tags,
        excluded_views=excluded_views,
        flags=flags,
    )
    sol.checks["ambiguous_views"] = sorted(set(ambiguous))
    sol.checks["positive_depth"] = bool(front)
    sv = np.linalg.svd(JW, compute_uv=False)
    sol.checks["condition_number"] = float(sv[0] / max(sv[-1], 1e-300))
    if not checks:
        return sol

    # ------------------------------------------------------------- checks
    centroid = np.mean([T.t for T in sol.tags.values()], axis=0)
    div = _diversity(sol.cameras, centroid)
    div["diverse"] = bool(div["max_view_angle_deg"] >= cfg.min_view_spread_deg and div["n_views"] >= 2)
    sol.checks["diversity"] = div
    if not div["diverse"]:
        sol.flags.append("narrow_baseline")
    cop = _coplanarity(sol.tags, sol.sizes, sol.cov, sol.param_index, sol.gauge_id, cfg)
    if cop["verdict"] == "not_coplanar":
        sol.flags.append("tags_not_coplanar")
        # Which tags are off the common plane?  A single raised tag also tilts
        # the reference plane of the others, so the question is asked the
        # other way round: removing which tag leaves the most coplanar set?
        def score(sub: dict[int, Pose]) -> tuple[bool, float]:
            g = sol.gauge_id if sol.gauge_id in sub else next(iter(sub))
            c = _coplanarity(sub, sol.sizes, sol.cov, sol.param_index, g, cfg)
            dev = 0.0
            for v in c["tags"].values():
                dev = max(dev, abs(v["offset_mm"]) / (0.2 + 3.0 * v["offset_sigma_mm"]), v["tilt_deg"] / (0.3 + 3.0 * v["tilt_sigma_deg"]))
            return bool(c["consistent_with_one_plane"]), dev

        keep = dict(sol.tags)
        peeled: list[int] = []
        while len(keep) >= 3:
            trials = {i: score({k: v for k, v in keep.items() if k != i}) for i in keep}
            best = min(trials, key=lambda i: trials[i][1])
            peeled.append(best)
            keep.pop(best)
            if trials[best][0] and trials[best][1] < 3.0:
                break
        if peeled and len(keep) >= 2:
            cop["off_plane_tags"] = sorted(peeled)
            cop["majority_plane_tags"] = sorted(keep)
    sol.checks["coplanarity"] = cop

    # free-size test: is every declared size consistent with the others?
    size_check: dict[str, Any] = {}
    if len(sol.tags) >= 2:
        fs = _BA(views, sol.cameras, sol.tags, sol.gauge_id, sol.intrinsics, [], free_scale=True, use_raw=use_raw)
        fs, rf, Jf, wf, _ = _optimise(fs, cfg)
        JWf = Jf * np.sqrt(wf)[:, None]
        try:
            covf = np.linalg.inv(JWf.T @ JWf) * max(1.0, float(np.sum(wf * rf * rf)) / max(fs.n_res - fs.n_params, 1))
        except np.linalg.LinAlgError:
            covf = np.linalg.pinv(JWf.T @ JWf)
        ratios = {}
        for i in fs.scale_tags:
            a, _ = fs.index[f"scale:{i}"]
            ratios[i] = (float(fs.scale[i]), float(np.sqrt(max(covf[a, a], 0.0))))
        mism = []
        for i, (s, sg) in ratios.items():
            err = abs(s - 1.0)
            bad = err > cfg.size_max_frac or (err > cfg.size_min_frac and err > cfg.size_z * sg)
            size_check[str(i)] = {"ratio_to_gauge": round(s, 6), "sigma": round(sg, 6), "consistent": not bad}
            if bad:
                mism.append(i)
        if mism:
            # if every other tag disagrees with the gauge in the same way, the gauge is the odd one
            if len(mism) == len(ratios) and len(ratios) >= 2 and np.ptp([ratios[i][0] for i in mism]) < 3 * max(ratios[i][1] for i in mism) + cfg.size_min_frac:
                size_check["suspect"] = [sol.gauge_id]
            else:
                size_check["suspect"] = mism
            sol.flags.append("tag_size_inconsistent")
    sol.checks["sizes"] = size_check

    # held-out views: localise each left-out view against a map built without it
    hold: dict[str, Any] = {}
    fids = list(sol.cameras)
    if len(fids) >= 3:
        ratios_h = []
        for fid in fids:
            rest = [v for v in views if v.frame_id != fid and v.frame_id in sol.cameras]
            if use_raw:
                rest = [MVView(v.frame_id, [TagObs(t.tag_id, sol.intrinsics.undistort_pixels(v.raw.get(t.tag_id, t.corners)), t.cov, t.size_mm) for t in v.tags], None, {}, v.image_size) for v in rest]
            try:
                sub = solve_multiview(rest, camera=sol.intrinsics, gauge=sol.gauge_id, cfg=cfg, checks=False)
            except GeometryError:
                continue
            v = next(v for v in views if v.frame_id == fid)
            tags = [t for t in v.tags if t.tag_id in sub.tags]
            if not tags:
                continue
            obj = np.concatenate([sub.tags[t.tag_id].apply(tag_object_points(t.size_mm)) for t in tags])
            if use_raw:  # compare in ideal pixels of the estimated lens
                img = np.concatenate([sol.intrinsics.undistort_pixels(v.raw.get(t.tag_id, t.corners)) for t in tags]).astype(np.float64)
            else:
                img = np.concatenate([t.corners for t in tags]).astype(np.float64)
            T0 = sol.cameras[fid]
            ok, rvec, tvec = cv2.solvePnP(obj, img, sol.intrinsics.without_distortion().K, None, rvec=cv2.Rodrigues(T0.R)[0], tvec=T0.t.reshape(3, 1).copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            T = Pose(cv2.Rodrigues(rvec)[0], np.asarray(tvec).ravel())
            uv, _ = sol.intrinsics.without_distortion().project(T.apply(obj))
            rms_out = float(np.sqrt(np.mean(np.sum((uv - img) ** 2, axis=1))))
            Tin = sol.cameras[fid]
            obj_in = np.concatenate([sol.tags[t.tag_id].apply(tag_object_points(t.size_mm)) for t in tags])
            uv_in, _ = sol.intrinsics.without_distortion().project(Tin.apply(obj_in))
            rms_in = float(np.sqrt(np.mean(np.sum((uv_in - img) ** 2, axis=1))))
            hold[fid] = {"held_out_rms_px": round(rms_out, 4), "in_sample_rms_px": round(rms_in, 4)}
            ratios_h.append(rms_out / max(rms_in, 0.02))
        if ratios_h:
            hold["max_ratio"] = round(float(max(ratios_h)), 3)
            hold["consistent"] = bool(max(ratios_h) <= cfg.holdout_max_ratio)
            if not hold["consistent"]:
                sol.flags.append("held_out_inconsistent")
    sol.checks["held_out"] = hold

    if sol.intrinsics_refined:
        a = sol.param_index.get("intr:f")
        sf = float(np.sqrt(max(sol.cov[a[0], a[0]], 0.0))) if a else None
        sc: dict[str, Any] = {"f": sol.intrinsics.fx, "sigma_f": sf, "relative_sigma_f": None if sf is None else sf / sol.intrinsics.fx}
        for name in ("cx", "cy", "k1", "k2"):
            idx = sol.param_index.get(f"intr:{name}")
            if idx:
                sc[name] = float({"cx": sol.intrinsics.cx, "cy": sol.intrinsics.cy, "k1": sol.intrinsics.dist[0], "k2": sol.intrinsics.dist[1]}[name])
                sc[f"sigma_{name}"] = float(np.sqrt(max(sol.cov[idx[0], idx[0]], 0.0)))
        sc["usable"] = bool(sf is not None and sf / sol.intrinsics.fx <= cfg.selfcal_max_rel_sigma_f and div["diverse"])
        sol.checks["self_calibration"] = sc
    return sol


# --------------------------------------------------------------------------
# Triangulation of stationary features
# --------------------------------------------------------------------------
def view_plane_of_image_line(T_cam_world: Pose, K: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """World-space plane ``(n, d)`` (``n . X = d``) through the camera centre and an image line."""
    Kinv = np.linalg.inv(K)
    a = Kinv @ np.array([p0[0], p0[1], 1.0])
    b = Kinv @ np.array([p1[0], p1[1], 1.0])
    n_c = np.cross(a, b)
    n_c /= np.linalg.norm(n_c)
    n_w = T_cam_world.R.T @ n_c
    return np.append(n_w, float(n_w @ T_cam_world.center))


def triangulate_line(planes: list[np.ndarray]) -> dict[str, Any]:
    """Least-squares 3-D line common to several view planes.

    A *stationary* physical line seen in N views lies on all N view planes;
    the residual angles and offsets tell whether that is true.  An occluding
    contour of a curved surface or a shadow edge is a different 3-D line in
    every view and fails this test — which is exactly how they are told apart
    from a real seam (prompt 06).
    """
    P = np.asarray(planes, dtype=float)
    if P.shape[0] < 2:
        raise ValueError("need at least two view planes")
    N = P[:, :3]
    # direction: perpendicular to all plane normals
    _, s, Vt = np.linalg.svd(N)
    d = Vt[-1]
    # point: least squares on n_i . X = d_i, minimum norm along the direction removed
    A = np.vstack([N, d[None]])
    b = np.append(P[:, 3], 0.0)
    X0, *_ = np.linalg.lstsq(A, b, rcond=None)
    res_off = N @ X0 - P[:, 3]
    res_ang = np.degrees(np.arcsin(np.clip(np.abs(N @ d), 0, 1)))
    cond = float(s[1] / max(s[0], 1e-300)) if len(s) > 1 else 0.0
    return {
        "point": X0,
        "direction": d,
        "offset_residual_mm": res_off,
        "angle_residual_deg": res_ang,
        "rms_offset_mm": float(np.sqrt(np.mean(res_off**2))),
        "max_angle_deg": float(np.max(res_ang)),
        #: second/first singular value of the normals: ~0 means all planes are
        #: nearly the same plane (no baseline across the line)
        "baseline_quality": cond,
    }


def triangulate_point(obs: list[tuple[Pose, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """Linear triangulation of a stationary point from ``(T_cam_world, K, pixel)``.

    Returns the world point and the RMS reprojection error in pixels.
    """
    A = []
    for T, K, uv in obs:
        P = K @ np.concatenate([T.R, T.t[:, None]], axis=1)
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1][:3] / Vt[-1][3]
    err = []
    for T, K, uv in obs:
        x = K @ T.apply(X)
        err.append(np.sum((x[:2] / x[2] - uv) ** 2))
    return X, float(np.sqrt(np.mean(err)))
