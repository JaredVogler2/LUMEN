"""Small dense Levenberg–Marquardt solver with robust reweighting.

SciPy's ``least_squares`` is used for the large sparse problems (bundle
adjustment).  This compact solver exists for the many tiny, hot problems
(edge-spread fits, per-frame pose refinement) where call overhead matters and
where we want the exact Jacobian and covariance at the solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ResidualFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class LMResult:
    x: np.ndarray
    residuals: np.ndarray
    jac: np.ndarray
    weights: np.ndarray
    cost: float
    iterations: int
    converged: bool

    @property
    def dof(self) -> int:
        return max(int(self.residuals.size - self.x.size), 1)

    @property
    def sigma(self) -> float:
        """Residual standard deviation estimate (weighted, per residual)."""
        w = self.weights
        return float(np.sqrt(np.sum(w * self.residuals**2) / max(np.sum(w) - self.x.size, 1.0)))

    def covariance(self, sigma: float | None = None) -> np.ndarray:
        """``sigma^2 (J^T W J)^-1`` (pseudo-inverse for rank-deficient cases)."""
        s = self.sigma if sigma is None else float(sigma)
        JW = self.jac * np.sqrt(self.weights)[:, None]
        return s * s * np.linalg.pinv(JW.T @ JW)


def huber_weights(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-300))


def mad_sigma(r: np.ndarray) -> float:
    """Robust scale: 1.4826 * median absolute deviation about the median."""
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return 0.0
    med = np.median(r)
    return float(1.4826 * np.median(np.abs(r - med)))


def lm_solve(
    fn: ResidualFn,
    x0: np.ndarray,
    *,
    max_iter: int = 30,
    xtol: float = 1e-10,
    ftol: float = 1e-12,
    lam0: float = 1e-3,
    robust_k: float | None = None,
    robust_rounds: int = 2,
) -> LMResult:
    """Minimise ``sum w_i r_i(x)^2``.

    ``fn(x)`` returns ``(residuals, jacobian)``.  With ``robust_k`` set, Huber
    weights with threshold ``robust_k * MAD-sigma`` are recomputed for
    ``robust_rounds`` outer rounds (IRLS).
    """
    x = np.asarray(x0, dtype=float).copy()
    r, J = fn(x)
    w = np.ones_like(r)
    it_total = 0
    converged = False
    rounds = 1 + (robust_rounds if robust_k is not None else 0)
    for rnd in range(rounds):
        if rnd > 0:
            s = mad_sigma(r)
            if s <= 0:
                break
            w = huber_weights(r, robust_k * s)  # type: ignore[operator]
        cost = float(np.sum(w * r * r))
        lam = lam0
        converged = False
        for _ in range(max_iter):
            it_total += 1
            Jw = J * w[:, None]
            A = J.T @ Jw
            g = Jw.T @ r
            diag = np.clip(np.diag(A), 1e-12, None)
            improved = False
            for _try in range(8):
                try:
                    dx = -np.linalg.solve(A + lam * np.diag(diag), g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                xn = x + dx
                rn, Jn = fn(xn)
                cn = float(np.sum(w * rn * rn))
                if np.isfinite(cn) and cn < cost:
                    step_small = float(np.linalg.norm(dx)) <= xtol * (xtol + float(np.linalg.norm(x)))
                    gain_small = (cost - cn) <= ftol * max(cost, 1e-300)
                    x, r, J, cost = xn, rn, Jn, cn
                    lam = max(lam * 0.3, 1e-12)
                    improved = True
                    if step_small or gain_small:
                        converged = True
                    break
                lam *= 10.0
            if not improved:
                converged = True  # no descent direction left at this damping
                break
            if converged:
                break
    return LMResult(x=x, residuals=r, jac=J, weights=w, cost=float(np.sum(w * r * r)), iterations=it_total, converged=converged)


def numeric_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float | np.ndarray = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)
    J = np.empty((f0.size, x.size))
    e = np.broadcast_to(np.asarray(eps, dtype=float), x.shape)
    for i in range(x.size):
        h = e[i] * max(1.0, abs(x[i]))
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        J[:, i] = (np.asarray(f(xp), dtype=float).ravel() - np.asarray(f(xm), dtype=float).ravel()) / (2.0 * h)
    return J
