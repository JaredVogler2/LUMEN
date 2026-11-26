"""Post-installation verification (prompt 11).

A new photo, taken after the placard has been installed and with the tags
still in place, is registered against the session's tag layout.  The placard's
perimeter is measured in that photo and compared with the target from the
operation record, in the local (datum) frame.

What the result is, and is not
------------------------------
The result is *measurement evidence*: observed signed deviations with an
uncertainty.  It is not an acceptance decision.  Where the record gives a
tolerance, the observation is classified with a guard band of the expanded
uncertainty:

* ``within_tolerance_observed``   — |deviation| + U <= tolerance
* ``outside_tolerance_observed``  — |deviation| - U >  tolerance
* ``indeterminate``               — anything in between, or no tolerance given

Evidence is refused (no numbers at all) when the photo cannot be registered,
when a registered tag has moved, or when fewer than three sides of the placard
can be measured.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import RESEARCH_NOTICE
from ..errors import ConflictError, GeometryError, LumenValidationError, NotFoundError
from ..geometry.homography import apply_h, local_scale_px_per_mm
from ..geometry.localframe import rectangle_pose_in_frame
from ..schemas import GuidanceStatus, PlacementSpec, Provenance, Session, SessionMode, Uncertainty, UncertaintyComponent, Verification
from ..units import wrap_angle
from ..vision.placard import SIDE_NAMES, PlacardConfig, PlacardMeasurement, measure_placard
from .context import utcnow

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

CORNER_NAMES = ("TL", "TR", "BR", "BL")
#: outline flags that keep a measured deviation from being reported as "within tolerance"
DOUBTFUL_OUTLINE = ("corner_lifted_or_hidden", "edge_not_straight", "glare_on_edge", "edge_partly_hidden", "ambiguous_proposal", "at_search_limit", "size_differs_from_record")


def _edge_deviations(m: PlacardMeasurement, Hinv: np.ndarray, rect: Any) -> dict[str, float]:
    """Signed offset of each measured side from the target side, along the placard's own axes.

    Left/right sides: positive toward the placard's +width direction; top/
    bottom: positive toward its +height direction.  Evaluated at the middle of
    the target side.
    """
    out: dict[str, float] = {}
    C = rect.corners
    for sd in m.sides:
        if not sd.ok or sd.n is None or sd.mid is None:
            continue
        k = sd.index
        t = np.array([sd.n[1], -sd.n[0]])
        q = apply_h(Hinv, np.stack([sd.mid - 0.5 * sd.length_px * t, sd.mid + 0.5 * sd.length_px * t]))
        d = q[1] - q[0]
        n_meas = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-12)
        mid_t = 0.5 * (C[k] + C[(k + 1) % 4])
        axis = rect.h_dir if k in (0, 2) else rect.w_dir
        den = float(axis @ n_meas)
        if abs(den) < 0.2:
            continue
        out[SIDE_NAMES[k]] = float(((q[0] - mid_t) @ n_meas) / den)
    return out


def _combine_uncertainty(base: Uncertainty | None, m: PlacardMeasurement, ppm: float, cfg: PlacardConfig, manual: bool) -> Uncertainty | None:
    """Uncertainty of the measured centre deviation.

    ``base`` is the uncertainty of the image-to-local mapping at the target in
    this photo (registration, scale, datums, lens, flatness).  The placard's
    own edges add a random part (edge fits) and a bound for how well an image
    edge represents the physical edge.
    """
    if base is None:
        return None
    var_sides = [sd.var_d for sd in m.sides if sd.ok]
    if not var_sides:
        return None
    # the centre is the mean of two opposite sides in each direction
    u_edge = float(np.sqrt(np.mean(var_sides) / 2.0)) / ppm
    # opposite sides shift the same way (toward the darker side), which mostly
    # cancels in the centre: half of the per-side bound is kept
    b_edge = 0.5 * cfg.edge_definition_bound_px / ppm
    comps = list(base.components)
    comps.append(UncertaintyComponent(name="placard edges (pixel noise)" if not manual else "placard edges (operator-marked)", kind="random", distribution="normal", value_mm=u_edge, variance_mm2=u_edge**2, note="edge fits of the placard sides, propagated to its centre"))
    comps.append(UncertaintyComponent(name="placard edge definition", kind="systematic", distribution="rectangular", value_mm=b_edge, variance_mm2=b_edge**2 / 3.0, note=f"image edge versus physical edge: {cfg.edge_definition_bound_px} px per side, half kept for the centre"))
    u_rand = float(np.hypot(base.u_random_mm, u_edge))
    bias_var = base.bias_bound_mm**2 + b_edge**2 / 3.0
    u_c = float(np.sqrt(u_rand**2 + bias_var))
    return Uncertainty(
        method=base.method, seed=base.seed, n_samples=base.n_samples, k=base.k,
        u_random_mm=u_rand, u_random_xz_mm=list(base.u_random_xz_mm),
        bias_bound_mm=float(np.sqrt(bias_var)), bias_linear_mm=float(base.bias_linear_mm + b_edge),
        combined_standard_mm=u_c, expanded_mm=float(base.k * u_c),
        u_angle_rad=base.u_angle_rad, per_corner_expanded_mm=list(base.per_corner_expanded_mm),
        components=comps, notes=list(base.notes) + ["uncertainty of the measured placard centre relative to the target"],
    )


def _rotation_sigma(m: PlacardMeasurement) -> float | None:
    v = [sd.var_theta for sd in m.sides if sd.ok]
    return float(np.sqrt(np.mean(v) / len(v))) if v else None


def verify_frame(
    svc: "SessionService",
    sid: str,
    frame_id: str,
    *,
    placard_id: str | None = None,
    manual_corners_px: list[list[float]] | None = None,
    snap: bool = True,
) -> dict[str, Any]:
    """Measure the installed placard in ``frame_id`` and store the result (unconfirmed)."""
    s = svc.get(sid)
    sol, views = svc._require_registration(s)
    f = svc._frame(s, frame_id)
    if f.id in sol.views:
        raise ConflictError(
            "verification_needs_new_photo", "This photo was used to register the session, before the placard was installed.",
            suggestions=["Take a new photo of the installed placard with the tags still in place."],
        )
    pid = placard_id or s.active_placard_id
    if not pid or not s.operation:
        raise ConflictError("no_placard", "No placard is selected.")
    op = svc._operation(s)
    spec = PlacementSpec(**op.placard(pid).resolved())
    lf = svc._local_frame(s)
    if lf is None or not lf.valid or not svc._spec_complete(spec):
        raise ConflictError("no_target", "There is no valid target to compare with: the datums or the placard definition are incomplete.", suggestions=["Complete and confirm the datums first."])
    try:
        H, loc, ext_view = svc._frame_mapping(s, sol, f)
    except GeometryError as exc:
        raise GeometryError(
            "registration_lost", "This photo cannot be registered: " + exc.message,
            suggestions=["The tags used during capture must still be on the panel and visible in the verification photo."],
        ) from exc
    rect = svc._target(lf, spec)
    Hinv = np.linalg.inv(H)
    cam = svc._camera(s, f)
    ppm = float(np.median(local_scale_px_per_mm(H, rect.corners)))
    flags: list[str] = []
    reg_flags = list(loc.flags) if loc is not None else []
    manual = manual_corners_px is not None
    vid = "v" + secrets.token_hex(4)
    base = Verification(id=vid, frame_id=f.id, placard_id=pid, created=utcnow(), notice=RESEARCH_NOTICE, registration_flags=reg_flags, observation="no_evidence")
    detail: dict[str, Any] = {"target_local": lf.to_local(rect.corners).tolist(), "target_px": None, "px_per_mm": ppm}
    tgt_ideal = apply_h(H, rect.corners)
    detail["target_px"] = (cam.distort_pixels(tgt_ideal) if cam is not None else tgt_ideal).tolist()

    if "high_residual" in reg_flags:
        base.flags = ["marker_moved"]
        return _store(svc, sid, base, detail, None, "A registered tag has moved since capture (or is misread): this photo is not evidence of the placard position.")

    rep = svc._gate_report(s, f.id, pid, True)
    if rep.status == GuidanceStatus.BLOCKED.value:
        failed = [g.code for g in rep.gates if not g.passed and g.severity == "blocking"]
        base.flags = ["registration_blocked"] + failed
        return _store(svc, sid, base, detail, None, "The target cannot be established in this photo: " + "; ".join(rep.reasons[:2]))

    man_ideal = None
    if manual:
        pts = np.asarray(manual_corners_px, dtype=float)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            raise LumenValidationError("invalid_corners", "Four corner points (u, v) are required.")
        if np.min([np.linalg.norm(pts[i] - pts[j]) for i in range(4) for j in range(i)]) < 8.0:
            raise LumenValidationError("invalid_corners", "The marked corners are too close together.")
        pts = _order_like_target(pts, tgt_ideal if cam is None else cam.distort_pixels(tgt_ideal))
        man_ideal = cam.undistort_pixels(pts) if cam is not None else pts
    with svc.cpu.slot(svc.settings.limits.request_timeout_seconds):
        m = measure_placard(svc._image(sid, f).bgr, cam, H, rect.corners, manual_corners_ideal=man_ideal, snap=snap, cfg=svc._placard_cfg)
    detail["measurement"] = m.to_dict()
    flags.extend(m.flags)
    if rep.status == GuidanceStatus.RESEARCH_PREVIEW.value:
        flags.append("registration_preview_only")
    base.flags = flags
    if not m.ok:
        msg = {
            "no_rectangle_stands_out": "No placard was found near the target position.",
            "expected_area_outside_image": "The target position is not inside this photo.",
            "too_few_sides_measured": "Too little of the placard's outline could be measured.",
        }.get(m.reason, "The placard outline could not be measured.")
        return _store(svc, sid, base, detail, None, msg + " Mark the four corners by hand, or retake the photo.")

    message = _evaluate(svc, base, detail, m, H=H, cam=cam, lf=lf, rect=rect, spec=spec, base_unc=rep.uncertainty, ppm=ppm, manual=manual, flags=flags)
    return _store(svc, sid, base, detail, m, message)


def _with_bound(unc: Uncertainty, name: str, bound_mm: float, note: str) -> Uncertainty:
    """``unc`` with one more systematic term given as a bound (rectangular)."""
    comps = list(unc.components) + [UncertaintyComponent(name=name, kind="systematic", distribution="rectangular", value_mm=bound_mm, variance_mm2=bound_mm**2 / 3.0, note=note)]
    bias_var = unc.bias_bound_mm**2 + bound_mm**2 / 3.0
    u_c = float(np.sqrt(unc.u_random_mm**2 + bias_var))
    return unc.model_copy(update={
        "bias_bound_mm": float(np.sqrt(bias_var)), "bias_linear_mm": float(unc.bias_linear_mm + bound_mm),
        "combined_standard_mm": u_c, "expanded_mm": float(unc.k * u_c), "components": comps,
    })


def _evaluate(
    svc: "SessionService", base: Verification, detail: dict[str, Any], m: PlacardMeasurement, *,
    H: np.ndarray, cam: Any, lf: Any, rect: Any, spec: PlacementSpec, base_unc: Uncertainty | None, ppm: float, manual: bool, flags: list[str],
    extra_bounds: list[tuple[str, float, str]] | None = None, doubtful: tuple[str, ...] | None = None,
) -> str:
    """Fill ``base`` with the measured outline, its deviation from the target and what that says against the
    tolerance.  ``extra_bounds`` are further systematic terms ``(name, bound in mm, note)`` of this measurement;
    ``doubtful`` are the flag prefixes that keep a result from being called "within" (default: every quality flag
    of the outline).  Returns a message for the operator (empty when there is nothing to add)."""
    Hinv = np.linalg.inv(H)
    C = m.corners_ideal
    have = np.isfinite(C).all(axis=1)
    plane = np.full((4, 2), np.nan)
    plane[have] = apply_h(Hinv, C[have])
    local = np.full((4, 2), np.nan)
    local[have] = lf.to_local(plane[have])
    raw = np.full((4, 2), np.nan)
    raw[have] = cam.distort_pixels(C[have]) if cam is not None else C[have]
    base.corners_plane = [[float(v) for v in r] if np.isfinite(r).all() else [] for r in plane]
    base.corners_local = [[float(v) for v in r] if np.isfinite(r).all() else [] for r in local]
    base.corners_px = [[float(v) for v in r] if np.isfinite(r).all() else [] for r in raw]
    prov = []
    for k in range(4):
        a, b = m.sides[(k - 1) % 4], m.sides[k]
        prov.append(Provenance.MANUAL if ("manual" in (a.method, b.method)) else Provenance.MEASURED)
    base.corner_provenance = prov
    base.edge_deviation_mm = {k: round(v, 4) for k, v in _edge_deviations(m, Hinv, rect).items()}
    unc = _combine_uncertainty(base_unc, m, ppm, svc._placard_cfg, manual)
    for name, bound, note in extra_bounds or []:
        if unc is not None and bound > 0:
            unc = _with_bound(unc, name, float(bound), note)
    base.uncertainty = unc
    if not m.complete:
        base.observation = "partial"
        return "Only part of the outline was measured: side offsets are reported, the centre is not."
    pose = rectangle_pose_in_frame(lf, plane)
    tgt_c = lf.to_local(rect.center)
    base.dx_mm = float(pose["x"] - tgt_c[0])
    base.dz_mm = float(pose["z"] - tgt_c[1])
    base.radial_mm = float(np.hypot(base.dx_mm, base.dz_mm))
    base.rotation_rad = float(wrap_angle(pose["angle"] - rect.angle))
    base.width_error_mm = float(pose["width"] - rect.width)
    base.height_error_mm = float(pose["height"] - rect.height)
    detail["rotation_sigma_rad"] = _rotation_sigma(m)
    detail["size_bound_mm"] = 2.0 * svc._placard_cfg.edge_definition_bound_px / ppm
    base.observation = _classify(base, spec, unc, detail)
    bad = doubtful if doubtful is not None else DOUBTFUL_OUTLINE
    if any(fl.startswith(bad) for fl in flags) and base.observation == "within_tolerance_observed":
        # a clean-looking number from a doubtful outline is not reported as "within"
        base.observation = "indeterminate"
        detail["observation_note"] = "downgraded: the outline has quality flags"
    return ""


def _order_like_target(pts: np.ndarray, target_px: np.ndarray) -> np.ndarray:
    """Order operator-marked corners as TL, TR, BR, BL by matching them to the target's corners."""
    best, best_cost = pts, float("inf")
    import itertools

    for perm in itertools.permutations(range(4)):
        cost = float(sum(np.linalg.norm(pts[perm[k]] - target_px[k]) for k in range(4)))
        if cost < best_cost:
            best, best_cost = pts[list(perm)], cost
    return best


def _classify(v: Verification, spec: PlacementSpec, unc: Uncertainty | None, detail: dict[str, Any]) -> str:
    tol = spec.tol_position_mm
    if tol is None or unc is None or v.radial_mm is None:
        detail["observation_note"] = "no tolerance in the record" if tol is None else "no uncertainty available"
        return "indeterminate"
    U = unc.expanded_mm
    pos = "within" if v.radial_mm + U <= tol else ("outside" if v.radial_mm - U > tol else "indeterminate")
    rot = "within"
    if spec.tol_rotation_rad is not None and v.rotation_rad is not None:
        sr = detail.get("rotation_sigma_rad") or 0.0
        ur = (unc.k * float(np.hypot(sr, unc.u_angle_rad or 0.0)))
        detail["rotation_expanded_rad"] = ur
        a = abs(v.rotation_rad)
        rot = "within" if a + ur <= spec.tol_rotation_rad else ("outside" if a - ur > spec.tol_rotation_rad else "indeterminate")
    detail["position_observation"], detail["rotation_observation"] = pos, rot
    if "outside" in (pos, rot):
        return "outside_tolerance_observed"
    if pos == "within" and rot == "within":
        return "within_tolerance_observed"
    return "indeterminate"


def _store(svc: "SessionService", sid: str, v: Verification, detail: dict[str, Any], m: PlacardMeasurement | None, message: str) -> dict[str, Any]:
    def fn(s: Session) -> None:
        s.verifications = [x for x in s.verifications if not (x.frame_id == v.frame_id and x.placard_id == v.placard_id)]
        s.verifications.append(v)
        s.research.setdefault("verification_detail", {})[v.id] = detail
        # keep the detail store bounded to the verifications that exist
        keep = {x.id for x in s.verifications}
        s.research["verification_detail"] = {k: d for k, d in s.research["verification_detail"].items() if k in keep}

    svc._mutate(sid, fn, "verification_measured", {"id": v.id, "frame_id": v.frame_id, "placard_id": v.placard_id, "observation": v.observation, "flags": v.flags})
    return {"verification": v.model_dump(mode="json"), "detail": detail, "message": message}


def confirm_verification(svc: "SessionService", sid: str, vid: str, confirmed: bool) -> Session:
    def fn(s: Session) -> Session:
        v = next((x for x in s.verifications if x.id == vid), None)
        if v is None:
            raise NotFoundError("verification_not_found", "Unknown verification result.")
        if confirmed and v.observation in ("no_evidence",):
            raise ConflictError("nothing_to_confirm", "This result contains no measurement to confirm.")
        v.confirmed = bool(confirmed)
        return s

    return svc._mutate(sid, fn, "verification_confirmed" if confirmed else "verification_unconfirmed", {"id": vid})


def delete_verification(svc: "SessionService", sid: str, vid: str) -> Session:
    def fn(s: Session) -> Session:
        if not any(x.id == vid for x in s.verifications):
            raise NotFoundError("verification_not_found", "Unknown verification result.")
        s.verifications = [x for x in s.verifications if x.id != vid]
        (s.research.get("verification_detail") or {}).pop(vid, None)
        return s

    return svc._mutate(sid, fn, "verification_deleted", {"id": vid})


def verification_primitives(s: Session, frame_id: str, mapper: Any, view_w: int, view_h: int) -> list[dict[str, Any]]:
    """Overlay primitives for the measured outlines on a frame (drawn with the target)."""
    out = []
    for v in s.verifications:
        if v.frame_id != frame_id:
            continue
        P = [p for p in v.corners_plane if p]
        if len(P) < 4:
            # partial: draw the sides that exist between available corners
            pts = [(k, p) for k, p in enumerate(v.corners_plane) if p]
            for (k0, p0), (k1, p1) in zip(pts, pts[1:]):
                if k1 == k0 + 1:
                    uv, ok = mapper(np.array([p0, p1], dtype=float))
                    if ok.all():
                        out.append({"kind": "segment", "points": uv.tolist(), "color": "#3ddc97", "width": 3})
            continue
        uv, ok = mapper(np.asarray(v.corners_plane, dtype=float))
        if not ok.all():
            continue
        out.append({
            "kind": "measured", "points": uv.tolist(), "corners": uv.tolist(), "closed": True, "label": f"{v.placard_id} · measured",
            "verification_id": v.id, "confirmed": v.confirmed,
        })
    return out


def simulated(s: Session) -> bool:
    return s.mode == SessionMode.DEMO.value
