"""Independent scale check with one length measured by the operator.

Every length LUMEN reports is scaled by the declared tag sizes.  The tags
check each other, so a single wrong tag is found — but if *all* declared sizes
share an error (a wrong nominal value, a mis-zeroed caliper, a print that is
uniformly scaled) nothing inside the picture can reveal it.  One distance
between two tag corners, measured with a steel rule or caliper, does.

The check compares the measured distance with the same distance computed from
the registered tag layout.  It never adjusts the scale: an inconsistency
blocks guidance until the operator finds the cause.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ConflictError, LumenValidationError, NotFoundError
from ..schemas import Session
from ..vision.planar import PlanarSolution
from .context import utcnow
from .frames import sizes_from_state

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")
K_COVERAGE = 2.0


def suggest_pair(sol: PlanarSolution) -> dict[str, Any] | None:
    """The two tag corners farthest apart: the longest baseline the tags offer."""
    ids = sorted(sol.layout.poses)
    if len(ids) < 2:
        return None
    best = None
    for i, a in enumerate(ids):
        Ca = sol.layout.corners(a)
        for b in ids[i + 1:]:
            Cb = sol.layout.corners(b)
            D = np.linalg.norm(Ca[:, None, :] - Cb[None, :, :], axis=2)
            ka, kb = np.unravel_index(int(np.argmax(D)), D.shape)
            if best is None or D[ka, kb] > best[0]:
                best = (float(D[ka, kb]), a, int(ka), b, int(kb))
    assert best is not None
    d, a, ka, b, kb = best
    return {
        "a": {"tag_id": a, "corner": ka, "corner_name": CORNER_NAMES[ka]},
        "b": {"tag_id": b, "corner": kb, "corner_name": CORNER_NAMES[kb]},
        "points_plane": [sol.layout.corners(a)[ka].tolist(), sol.layout.corners(b)[kb].tolist()],
        "note": "Corners of the black squares (not of the white sticker). The corner names follow each tag's own upright orientation; use the marks in the picture.",
    }


def _computed(sol: PlanarSolution, s: Session, ref: dict[str, Any]) -> tuple[float, float] | None:
    """Distance from the layout and its standard uncertainty (mm), or ``None`` if a tag is not registered."""
    a, b = ref["a"], ref["b"]
    if a["tag_id"] not in sol.layout.poses or b["tag_id"] not in sol.layout.poses:
        return None
    pa = sol.layout.corners(a["tag_id"])[a["corner"]]
    pb = sol.layout.corners(b["tag_id"])[b["corner"]]
    d = float(np.linalg.norm(pb - pa))
    # position uncertainty of the two tags in the layout, along the connecting line
    u = (pb - pa) / max(d, 1e-9)
    var = 0.0
    for tid in (a["tag_id"], b["tag_id"]):
        if tid in sol.tag_slices:
            i0, i1 = sol.tag_slices[tid]
            C = sol.param_cov[i0:i1, i0:i1][:2, :2]
            var += float(u @ C @ u)
    # the declared sizes scale every length: stated caliper uncertainty of the tags, relative
    sizes = sizes_from_state(s.tag_sizes)
    rel = []
    for tid in sol.layout.poses:
        size, prov = sizes.size(tid)
        rel.append((sizes.measured_u_mm if prov == "measured" else sizes.nominal_tolerance_mm / np.sqrt(3.0)) / size)
    var += (d * float(np.mean(rel)) / np.sqrt(max(len(rel), 1))) ** 2 + (d * 0.03 / 50.0 / np.sqrt(3.0)) ** 2
    return d, float(np.sqrt(var))


def evaluate(sol: PlanarSolution | None, s: Session, ref: dict[str, Any]) -> dict[str, Any]:
    out = dict(ref)
    out.update({"computed_mm": None, "sigma_computed_mm": None, "difference_mm": None, "limit_mm": None, "consistent": None, "evaluable": False})
    if sol is None:
        return out
    c = _computed(sol, s, ref)
    if c is None:
        out["note"] = "one of the two tags is not part of the current registration"
        return out
    d, u_c = c
    lim = K_COVERAGE * float(np.hypot(u_c, ref["u_mm"]))
    diff = d - float(ref["measured_mm"])
    out.update({
        "computed_mm": d, "sigma_computed_mm": u_c, "difference_mm": diff, "limit_mm": lim,
        "relative_difference": diff / max(float(ref["measured_mm"]), 1e-9), "consistent": bool(abs(diff) <= lim), "evaluable": True,
    })
    return out


def reevaluate_all(s: Session, sol: PlanarSolution | None) -> None:
    """Refresh every stored check against the current registration (the measured values are kept)."""
    s.reference_distances = [evaluate(sol, s, {k: r[k] for k in ("id", "a", "b", "measured_mm", "u_mm", "instrument", "created")}) for r in s.reference_distances]


def add_reference(svc: "SessionService", sid: str, a: dict[str, Any], b: dict[str, Any], measured_mm: float, u_mm: float = 0.5, instrument: str = "") -> dict[str, Any]:
    s = svc.get(sid)
    sol, _ = svc._require_registration(s)
    for p in (a, b):
        if int(p.get("corner", -1)) not in (0, 1, 2, 3):
            raise LumenValidationError("invalid_corner", "corner must be 0 (top-left), 1, 2 or 3.")
        if int(p.get("tag_id", -1)) not in sol.layout.poses:
            raise NotFoundError("tag_not_registered", f"Tag {p.get('tag_id')} is not part of the registration.")
    if (int(a["tag_id"]), int(a["corner"])) == (int(b["tag_id"]), int(b["corner"])):
        raise LumenValidationError("same_point", "The two points are the same corner.")
    if not (5.0 <= float(measured_mm) <= 5000.0):
        raise LumenValidationError("distance_out_of_range", "The measured distance must be between 5 and 5000 mm.")
    if not (0.01 <= float(u_mm) <= 10.0):
        raise LumenValidationError("uncertainty_out_of_range", "The measurement uncertainty must be between 0.01 and 10 mm.")
    ref = {
        "id": "r" + secrets.token_hex(3),
        "a": {"tag_id": int(a["tag_id"]), "corner": int(a["corner"])}, "b": {"tag_id": int(b["tag_id"]), "corner": int(b["corner"])},
        "measured_mm": float(measured_mm), "u_mm": float(u_mm), "instrument": str(instrument)[:80], "created": utcnow(),
    }
    rec = evaluate(sol, s, ref)
    if rec["evaluable"] and rec["computed_mm"] < 60.0:
        raise LumenValidationError("baseline_too_short", "The two corners are less than 60 mm apart: use corners of two different, well separated tags.")

    def fn(x: Session) -> None:
        x.reference_distances = [r for r in x.reference_distances if not (r["a"] == ref["a"] and r["b"] == ref["b"])] + [rec]

    svc._mutate(sid, fn, "reference_distance_added", {"id": ref["id"], "consistent": rec["consistent"], "difference_mm": rec["difference_mm"]})
    return rec


def remove_reference(svc: "SessionService", sid: str, rid: str) -> Session:
    def fn(x: Session) -> Session:
        if not any(r["id"] == rid for r in x.reference_distances):
            raise NotFoundError("reference_not_found", "Unknown reference measurement.")
        x.reference_distances = [r for r in x.reference_distances if r["id"] != rid]
        return x

    return svc._mutate(sid, fn, "reference_distance_removed", {"id": rid})


def summary(s: Session) -> tuple[bool | None, str]:
    """(consistent?, message) over all evaluable checks; ``None`` when there is none."""
    ev = [r for r in s.reference_distances if r.get("evaluable")]
    if not ev:
        return None, ""
    bad = [r for r in ev if not r["consistent"]]
    if bad:
        r = max(bad, key=lambda r: abs(r["difference_mm"]))
        return False, (
            f"An independent length measurement contradicts the scale: {r['measured_mm']:.1f} mm measured, "
            f"{r['computed_mm']:.1f} mm from the tags ({r['relative_difference'] * 100:+.1f} %, limit ±{r['limit_mm']:.1f} mm)."
        )
    r = max(ev, key=lambda r: r["measured_mm"])
    return True, f"Scale agrees with an independent length measurement over {r['measured_mm']:.0f} mm (difference {r['difference_mm']:+.2f} mm, limit ±{r['limit_mm']:.1f} mm)."


def require_no_conflict(s: Session) -> None:
    ok, msg = summary(s)
    if ok is False:
        raise ConflictError("scale_contradicted", msg)
