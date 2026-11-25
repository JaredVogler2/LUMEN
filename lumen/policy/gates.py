"""Gating: what may be shown to the operator, and why (prompt 10).

The outcome for one placard on one displayed frame is a
:class:`~lumen.schemas.GuidanceStatus`:

``blocked``
    At least one *blocking* gate failed.  No placement overlay is offered; only
    diagnostics.  Missing or ambiguous geometry always ends here.
``research_preview``
    The geometry is computable but at least one *qualifying* gate failed
    (unmeasured tags, no independent check, uncharacterised lens, unconfirmed
    datum, uncertainty too large for the tolerance, incomplete record ...).
    The overlay is shown, watermarked as a preview.
``prototype_guidance``
    Every computed gate passed.  This is still research output: the gates are
    backed by synthetic tests only.
``validated``
    Additionally inside a *physically validated domain* supplied by an
    administrator (``LUMEN_VALIDATED_DOMAIN_FILE``).  None ships with LUMEN.

A low reprojection error never passes a gate on its own, a tolerance is never
invented, and a manual edit can only ever *add* a reason to stay below
``prototype_guidance`` — it cannot remove one.

Thresholds live in :class:`GatePolicy`.  The defaults are generic placeholders
chosen so the synthetic test-suite exercises every branch; they are **not**
engineering limits and must be set per programme (``LUMEN_POLICY_FILE``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ..schemas import GateReport, GateResult, GuidanceStatus, Severity, Uncertainty


@dataclass(frozen=True)
class GatePolicy:
    #: expanded uncertainty (k from the uncertainty config) may use at most this
    #: fraction of the position tolerance
    max_uncertainty_fraction_of_tolerance: float = 0.5
    #: same for rotation
    max_angle_uncertainty_fraction_of_tolerance: float = 0.5
    #: tags (with accepted observations) required for an independent check
    min_tags_for_guidance: int = 2
    #: the cross-validation must be able to see a size error of this relative magnitude (1 sigma)
    max_crossval_sigma_frac: float = 0.01
    max_reduced_chi2: float = 6.0
    max_tag_rms_px: float = 0.8
    #: farthest target corner from the tag centroid, in units of the tag constellation's RMS radius
    max_extrapolation_ratio: float = 4.0
    #: a snapped datum must beat the next candidate in its search band by this factor
    min_datum_distinctness: float = 1.5
    #: require a measured lens calibration for guidance
    require_calibration: bool = True
    #: require measured (not nominal) tag sizes for guidance
    require_measured_tags: bool = True
    #: smallest accepted tag edge in the displayed frame (px)
    min_tag_edge_px: float = 40.0
    provenance: str = "LUMEN generic placeholder policy — not an engineering limit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_policy(path: str | None) -> GatePolicy:
    """Policy from a JSON file (unknown keys are rejected so typos cannot
    silently loosen a gate)."""
    if not path:
        return GatePolicy()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(GatePolicy)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    return GatePolicy(**data)


@dataclass
class GateInputs:
    """Facts about the session collected by the service layer."""

    operation_selected: bool = False
    placard_selected: bool = False
    spec_geometry_complete: bool = False
    spec_issues: list[str] | None = None  # issues of the operation record relevant here
    #: "released", "draft" or "superseded"
    record_status: str = "released"
    surface_target: str = "plane"
    tolerance_position_mm: float | None = None
    tolerance_rotation_rad: float | None = None

    registered: bool = False
    registration_flags: list[str] | None = None
    n_tags: int = 0
    tags_consistent: bool = True
    inconsistent_reasons: list[str] | None = None
    excluded_tags: dict[int, list[str]] | None = None
    independently_checked: bool = False
    #: operator's ruler measurement between two tag corners: None = none made
    reference_consistent: bool | None = None
    reference_message: str = ""
    crossval_sigma_frac: float | None = None
    reduced_chi2: float | None = None
    rms_px: float | None = None
    all_tags_measured: bool = False

    frame_localized: bool = False
    frame_flags: list[str] | None = None
    frame_min_tag_edge_px: float | None = None
    frame_n_tags: int = 0
    frame_calibrated: bool = False
    all_frames_calibrated: bool = False
    pose_ambiguous: bool = False

    datums_present: tuple[bool, bool] = (False, False)  # (floor, side)
    datums_confirmed: tuple[bool, bool] = (False, False)
    datum_distinctness: tuple[float | None, float | None] = (None, None)
    datum_manual: tuple[bool, bool] = (False, False)
    datum_plane_evidence: tuple[str, str] = ("unverified", "unverified")
    datum_feature: tuple[str | None, str | None] = (None, None)
    datum_declared_on_plane: tuple[bool | None, bool | None] = (None, None)
    local_frame_valid: bool = False
    local_frame_reasons: list[str] | None = None
    axis_disagreement_deg: float | None = None
    axis_limit_deg: float | None = None

    multiview: bool = False
    multiview_flags: list[str] | None = None
    #: single tag observations left out of the multi-photo solution ("tag 41 in photo f005")
    excluded_observations: list[str] | None = None
    coplanarity_verdict: str | None = None
    coplanarity_power_mm: float | None = None
    calibration_consistent: bool | None = None
    held_out_consistent: bool | None = None
    plane_violated: bool = False

    target_in_view: bool = False
    extrapolation_ratio: float | None = None
    uncertainty: Uncertainty | None = None
    reference_distance_checks: list[dict[str, Any]] | None = None
    validated_domain: dict[str, Any] | None = None
    in_validated_domain: bool = False
    demo: bool = False


_SUGGEST = {
    "more_tags": "Add a second measured AprilTag on the flat panel, on the far side of the target from the first, and retake the photo.",
    "measure_tags": "Measure the black-square edge of each tag with a caliper and enter it (Markers step).",
    "calibrate": "Attach a lens calibration made at this resolution and orientation (Settings → Camera calibration).",
    "closer": "Move closer or use the photo mode so tags are larger in the image.",
    "retake_sharp": "Retake the photo: hold steady, avoid glare on the tags, keep all tags fully in view.",
    "confirm_datums": "Review the two datum lines on the rectified view and confirm them.",
    "tags_near_target": "Place the tags so that they surround the target location rather than sit to one side.",
}


def evaluate_gates(inp: GateInputs, policy: GatePolicy | None = None, placard_id: str | None = None) -> GateReport:
    policy = policy or GatePolicy()
    gates: list[GateResult] = []

    def gate(code: str, passed: bool, severity: Severity, message: str, *, value: float | None = None, limit: float | None = None, suggestion: str | None = None) -> None:
        gates.append(GateResult(code=code, passed=bool(passed), severity=severity, message=message, value=value, limit=limit, suggestion=None if passed else suggestion))

    B, W, INFO = Severity.BLOCKING, Severity.WARNING, Severity.INFO

    # ------------------------------------------------------------ blocking
    gate("operation_selected", inp.operation_selected and inp.placard_selected, B,
         "An operation and a placard are selected." if inp.operation_selected and inp.placard_selected else "No operation / placard is selected.",
         suggestion="Choose the operation and placard on the Identify step.")
    gate("placard_geometry_defined", inp.spec_geometry_complete, B,
         "Placard offsets, size, rotation and reference point are defined." if inp.spec_geometry_complete
         else "The operation record does not define the placard geometry completely; LUMEN does not invent placement requirements.",
         suggestion="Complete the record (offsets, size, rotation, reference point) in the operation library.")
    gate("surface_supported", inp.surface_target == "plane", B,
         "The target surface is declared planar." if inp.surface_target == "plane"
         else "The target itself is declared curved: experimental only, no placement guidance.",
         suggestion="Curved targets are outside the supported domain; use the diagnostic views only.")
    gate("scale_available", inp.registered and inp.n_tags >= 1, B,
         f"Scale established from {inp.n_tags} tag(s)." if inp.registered and inp.n_tags >= 1 else "No accepted AprilTag: there is no metric scale.",
         suggestion="Place a measured AprilTag on the flat panel, fully in view, and capture again.")
    if inp.registered:
        gate("tags_mutually_consistent", inp.tags_consistent, B,
             "Tags agree with each other (size and coplanarity)." if inp.tags_consistent
             else "Tags contradict each other (" + ", ".join(inp.inconsistent_reasons or ["inconsistent"]) + "); with two tags it cannot be told which one is wrong.",
             suggestion="Check that every tag lies flat on the same plane and that its measured size is entered for the right ID; then retake.")
        gate("pose_unambiguous", not inp.pose_ambiguous, B,
             "The camera pose has a single consistent solution." if not inp.pose_ambiguous
             else "Two different camera poses explain the tags almost equally well.",
             suggestion="Add a second tag or view the panel more obliquely / from closer, then retake.")
        gate("frame_registered", inp.frame_localized and "high_residual" not in (inp.frame_flags or []), B,
             "This frame is registered to the session's tags." if inp.frame_localized and "high_residual" not in (inp.frame_flags or [])
             else ("A registered tag appears to have moved or is misread in this frame." if inp.frame_localized else "This frame shows no registered tag."),
             suggestion="Keep the tags fixed; bring at least one registered tag fully into view. If a tag was moved, restart the capture.")
    if inp.registered and inp.reference_consistent is not None:
        gate("independent_length_check", bool(inp.reference_consistent), B, inp.reference_message,
             suggestion="Check the declared tag sizes (measure the black square, not the sticker) and the ruler reading; then enter the corrected values.")
    if inp.registered and inp.multiview:
        gate("plane_assumption", not inp.plane_violated, B,
             "The tags define one plane." if not inp.plane_violated
             else "The tags are not on one plane and no consistent majority could be identified.",
             suggestion="Put every tag flat on the same panel face (not on trim, floor or the curved edge) and capture again.")
    gate("datums_defined", all(inp.datums_present), B,
         "Floor and side datum lines are defined." if all(inp.datums_present)
         else "Missing datum: " + ", ".join(n for n, p in zip(("floor", "side"), inp.datums_present) if not p) + ".",
         suggestion="Define both datum lines on the Datums step.")
    if all(inp.datums_present):
        reasons = ", ".join(inp.local_frame_reasons or [])
        gate("local_frame_valid", inp.local_frame_valid, B,
             "The local frame is well defined." if inp.local_frame_valid else f"The local frame cannot be established ({reasons}).",
             value=inp.axis_disagreement_deg, limit=inp.axis_limit_deg,
             suggestion="Check that the two datum lines are the intended features and are not parallel; the axis disagreement must be within the configured limit.")
    if inp.registered and all(inp.datums_present) and inp.local_frame_valid and inp.spec_geometry_complete:
        gate("target_in_view", inp.target_in_view, B,
             "The target location is inside this frame." if inp.target_in_view else "The target location is outside this frame or behind the camera.",
             suggestion="Reframe so the target area and the tags are both visible.")

    # ---------------------------------------------------------- qualifying
    spec_ok = not (inp.spec_issues or [])
    gate("operation_record_complete", spec_ok, W,
         "The operation record is complete (datum signs, axis policy, tolerances)." if spec_ok
         else "The operation record is incomplete: " + ", ".join(inp.spec_issues or []) + ".",
         suggestion="Complete the record; a missing tolerance or datum definition prevents guidance status.")
    if inp.operation_selected:
        released = inp.record_status == "released"
        gate("operation_record_released", released, W,
             "The operation record is a released revision." if released
             else f"The operation record is a {inp.record_status} revision: its values have not been released for use.",
             suggestion="Have the record reviewed and saved as a released revision, or select the released revision.")
    if inp.registered:
        if policy.require_measured_tags:
            gate("tag_sizes_measured", inp.all_tags_measured, W,
                 "All tags used have measured sizes." if inp.all_tags_measured else "At least one tag uses the nominal size from the packaging.",
                 suggestion=_SUGGEST["measure_tags"])
        gate("independent_scale_check", inp.independently_checked and inp.n_tags >= policy.min_tags_for_guidance, W,
             "Scale and coplanarity were cross-checked between tags." if inp.independently_checked and inp.n_tags >= policy.min_tags_for_guidance
             else "Only one tag: nothing checks its size, flatness or the plane assumption.",
             value=float(inp.n_tags), limit=float(policy.min_tags_for_guidance), suggestion=_SUGGEST["more_tags"])
        if inp.independently_checked and inp.crossval_sigma_frac is not None:
            ok = inp.crossval_sigma_frac <= policy.max_crossval_sigma_frac
            gate("independent_check_sensitive", ok, W,
                 f"The cross-check resolves a size error of {inp.crossval_sigma_frac * 100:.2f} % (1σ)." if ok
                 else f"The cross-check is weak: it could not see a size error below {inp.crossval_sigma_frac * 100:.1f} % (1σ).",
                 value=inp.crossval_sigma_frac, limit=policy.max_crossval_sigma_frac, suggestion=_SUGGEST["more_tags"])
        if inp.reduced_chi2 is not None and inp.n_tags >= 2:
            ok = inp.reduced_chi2 <= policy.max_reduced_chi2
            gate("fit_consistent_with_noise", ok, W,
                 f"Residuals are consistent with the measured corner noise (reduced χ² {inp.reduced_chi2:.2f})." if ok
                 else f"Residuals are larger than the corner noise explains (reduced χ² {inp.reduced_chi2:.1f}): uncorrected lens distortion, a bent tag or a non-flat panel.",
                 value=inp.reduced_chi2, limit=policy.max_reduced_chi2, suggestion=_SUGGEST["calibrate"])
        if policy.require_calibration:
            gate("lens_characterised", inp.all_frames_calibrated and inp.frame_calibrated, W,
                 "A measured lens calibration is applied." if inp.all_frames_calibrated and inp.frame_calibrated
                 else "No matching lens calibration is applied; distortion is only bounded by assumption.",
                 suggestion=_SUGGEST["calibrate"])
        if inp.multiview:
            if inp.coplanarity_verdict == "coplanar":
                gate("plane_verified_3d", True, INFO,
                     f"Coplanarity of the tags was verified in 3-D (could have seen {inp.coplanarity_power_mm:.1f} mm)." if inp.coplanarity_power_mm is not None else "Coplanarity of the tags was verified in 3-D.")
            elif inp.coplanarity_verdict == "not_established":
                gate("plane_verified_3d", True, INFO, "The views are too similar to verify coplanarity in 3-D; the single-view cross-check applies.")
            if inp.excluded_observations:
                gate("observations_left_out", True, INFO,
                     "Left out because it did not fit the other photos of the same tag: " + "; ".join(inp.excluded_observations) + ". The tag itself is kept.")
            if inp.calibration_consistent is not None:
                gate("calibration_consistent", inp.calibration_consistent, W,
                     "The lens profile agrees with a focal length re-estimated from the tags." if inp.calibration_consistent
                     else "The lens profile disagrees with the focal length the tags imply: wrong profile, different zoom or capture mode?",
                     suggestion="Recalibrate at this zoom, resolution and orientation, or detach the profile.")
            if inp.held_out_consistent is not None:
                gate("held_out_views", inp.held_out_consistent, W,
                     "Each photo is predicted by the others (held-out check)." if inp.held_out_consistent
                     else "A photo is not predicted by the others: something changed between photos.",
                     suggestion="Remove the odd photo or recapture without touching the tags.")
        if inp.frame_min_tag_edge_px is not None:
            ok = inp.frame_min_tag_edge_px >= policy.min_tag_edge_px
            gate("tag_resolution", ok, W,
                 f"Smallest tag is {inp.frame_min_tag_edge_px:.0f} px." if ok else f"Smallest tag is only {inp.frame_min_tag_edge_px:.0f} px in this frame.",
                 value=inp.frame_min_tag_edge_px, limit=policy.min_tag_edge_px, suggestion=_SUGGEST["closer"])
    if all(inp.datums_present):
        gate("datums_confirmed", all(inp.datums_confirmed), W,
             "Both datums were reviewed and confirmed by the operator." if all(inp.datums_confirmed) else "Datum lines are not confirmed yet.",
             suggestion=_SUGGEST["confirm_datums"])
        for i, name in enumerate(("floor", "side")):
            dist = inp.datum_distinctness[i]
            if inp.datum_manual[i]:
                gate(f"datum_{name}_measured", False, W,
                     f"The {name} datum is a hand-drawn line that was not fitted to an image edge.",
                     suggestion="Use 'snap to edge' so the line is measured rather than drawn.")
            elif dist is not None:
                ok = dist >= policy.min_datum_distinctness
                gate(f"datum_{name}_unambiguous", ok, W,
                     f"The {name} datum edge is distinct (×{dist:.1f} over the next candidate)." if ok
                     else f"The {name} datum has a competing edge of similar strength nearby (×{dist:.1f}).",
                     value=dist, limit=policy.min_datum_distinctness,
                     suggestion="Zoom in and place the line on the intended edge; check the alternatives shown.")
            ev = inp.datum_plane_evidence[i]
            feat = inp.datum_feature[i]
            declared = inp.datum_declared_on_plane[i]
            if ev == "off_plane":
                gate(f"datum_{name}_on_plane", False, B,
                     f"Multi-view evidence shows the {name} datum feature is NOT on the target plane; it cannot be used as a planar line.",
                     suggestion="Use a feature that lies on the panel face, or capture a sweep so the 3-D position can be modelled.")
            elif feat in ("curve_tangent", "curve_outer_extreme") and ev not in ("verified_multiview", "cylinder_model"):
                gate(f"datum_{name}_needs_3d", False, W,
                     f"The {name} datum is a rounded-edge feature ({feat}); a single planar view cannot locate it.",
                     suggestion="Capture a sweep of 5–9 views around the rounded edge so its silhouette can be modelled.")
            elif ev == "verified_multiview":
                gate(f"datum_{name}_on_plane", True, INFO, f"The {name} datum stays fixed on the plane across views.")
            elif declared is True:
                gate(f"datum_{name}_on_plane", True, INFO, f"The {name} datum is declared on the target plane by the operation record (not verified by a second view).")
            else:
                gate(f"datum_{name}_on_plane", False, W,
                     f"Nothing establishes that the {name} datum feature lies on the target plane.",
                     suggestion="Declare the plane relation in the operation record or capture several views.")
    if inp.extrapolation_ratio is not None:
        ok = inp.extrapolation_ratio <= policy.max_extrapolation_ratio
        gate("target_distance", ok, W,
             f"The target is {inp.extrapolation_ratio:.1f}× the tag spread from the tag centre." if ok
             else f"The target is far from the tags ({inp.extrapolation_ratio:.1f}× their spread): the mapping is extrapolated.",
             value=inp.extrapolation_ratio, limit=policy.max_extrapolation_ratio, suggestion=_SUGGEST["tags_near_target"])
    u = inp.uncertainty
    if u is not None:
        if inp.tolerance_position_mm is None:
            gate("uncertainty_vs_tolerance", False, W,
                 f"Expanded uncertainty is {u.expanded_mm:.2f} mm (k={u.k:g}) but the record supplies no position tolerance to compare with.",
                 value=u.expanded_mm, suggestion="Add the authorised tolerance to the operation record.")
        else:
            lim = policy.max_uncertainty_fraction_of_tolerance * inp.tolerance_position_mm
            ok = bool(u.expanded_mm <= lim)
            gate("uncertainty_vs_tolerance", ok, W,
                 f"Expanded uncertainty {u.expanded_mm:.2f} mm (k={u.k:g}) is within {policy.max_uncertainty_fraction_of_tolerance * 100:.0f} % of the ±{inp.tolerance_position_mm:.2f} mm tolerance." if ok
                 else f"Expanded uncertainty {u.expanded_mm:.2f} mm (k={u.k:g}) exceeds {policy.max_uncertainty_fraction_of_tolerance * 100:.0f} % of the ±{inp.tolerance_position_mm:.2f} mm tolerance.",
                 value=u.expanded_mm, limit=lim, suggestion=_largest_component_hint(u))
        if inp.tolerance_rotation_rad is not None and u.u_angle_rad is not None:
            lim = policy.max_angle_uncertainty_fraction_of_tolerance * inp.tolerance_rotation_rad
            val = u.k * u.u_angle_rad
            ok = bool(val <= lim)
            gate("angle_uncertainty_vs_tolerance", ok, W,
                 f"Rotation uncertainty {_deg(val):.2f}° (k={u.k:g}, repeatability only) is within the limit." if ok
                 else f"Rotation uncertainty {_deg(val):.2f}° (k={u.k:g}) exceeds {policy.max_angle_uncertainty_fraction_of_tolerance * 100:.0f} % of the ±{_deg(inp.tolerance_rotation_rad):.2f}° tolerance.",
                 value=val, limit=lim, suggestion="Use longer, well-defined datum edges and keep them fully in view.")
    for chk in inp.reference_distance_checks or []:
        gate("reference_distance", bool(chk.get("passed")), W, str(chk.get("message", "")), value=chk.get("value"), limit=chk.get("limit"),
             suggestion="Re-measure the reference distance and the tag sizes; they disagree.")

    # -------------------------------------------------------------- outcome
    blocking = [g for g in gates if not g.passed and g.severity == B]
    qualifying = [g for g in gates if not g.passed and g.severity == W]
    if blocking:
        status = GuidanceStatus.BLOCKED
    elif qualifying:
        status = GuidanceStatus.RESEARCH_PREVIEW
    elif inp.in_validated_domain and inp.validated_domain:
        status = GuidanceStatus.VALIDATED
    else:
        status = GuidanceStatus.PROTOTYPE_GUIDANCE
    failing = blocking + qualifying
    seen: set[str] = set()
    suggestions: list[str] = []
    for g in failing:
        if g.suggestion and g.suggestion not in seen:
            seen.add(g.suggestion)
            suggestions.append(g.suggestion)
    notice = {
        GuidanceStatus.BLOCKED: "No placement guidance: required geometry is missing, ambiguous or unsupported.",
        GuidanceStatus.RESEARCH_PREVIEW: "Research preview only — one or more qualification gates are not met.",
        GuidanceStatus.PROTOTYPE_GUIDANCE: "Prototype guidance — all computed gates pass. Physical accuracy has NOT been validated; not product acceptance.",
        GuidanceStatus.VALIDATED: "Inside the administrator-supplied validated domain. Still measurement evidence, not product acceptance.",
    }[status]
    if inp.demo:
        notice = "SIMULATED demo data. " + notice
    return GateReport(
        status=status,
        placard_id=placard_id,
        gates=gates,
        reasons=[g.message for g in failing],
        suggestions=suggestions,
        uncertainty=u,
        physically_validated=bool(status == GuidanceStatus.VALIDATED),
        notice=notice,
    )


def _deg(rad: float) -> float:
    return float(rad) * 57.29577951308232


def _largest_component_hint(u: Uncertainty) -> str:
    if not u.components:
        return "Reduce the largest uncertainty contribution."
    top = max(u.components, key=lambda c: c.variance_mm2)
    hints = {
        "pixel noise": "Largest contribution: pixel noise. " + _SUGGEST["closer"],
        "tag size: nominal": "Largest contribution: unmeasured tag size. " + _SUGGEST["measure_tags"],
        "tag size": "Largest contribution: tag size uncertainty. Use tags closer to the target or a longer measured reference.",
        "edge localisation": "Largest contribution: edge bias on small tags. " + _SUGGEST["closer"],
        "lens distortion: not": "Largest contribution: uncharacterised lens distortion. " + _SUGGEST["calibrate"],
        "lens distortion": "Largest contribution: calibration uncertainty. Recalibrate with more, better-spread views.",
        "panel flatness": "Largest contribution: possible non-flatness seen at an angle. Photograph the target more squarely.",
    }
    for key, text in hints.items():
        if top.name.startswith(key):
            return text
    return f"Largest contribution: {top.name}."


def check_validated_domain(domain: dict[str, Any] | None, facts: dict[str, Any]) -> bool:
    """True only when a domain record exists and every stated limit is met.

    A domain record is a JSON object ``{"id", "evidence", "limits": {...}}``
    written by the people who ran the physical validation.  Supported limits:
    ``min_tag_edge_px``, ``max_view_angle_deg``, ``max_target_offset_mm``,
    ``requires_calibration``, ``min_tags``, ``max_expanded_uncertainty_mm``.
    Unknown limit names make the check fail (fail closed).
    """
    if not domain or not isinstance(domain.get("limits"), dict) or not domain.get("evidence"):
        return False
    for key, lim in domain["limits"].items():
        if key == "min_tag_edge_px":
            ok = facts.get("min_tag_edge_px") is not None and facts["min_tag_edge_px"] >= lim
        elif key == "max_view_angle_deg":
            ok = facts.get("view_angle_deg") is not None and facts["view_angle_deg"] <= lim
        elif key == "max_target_offset_mm":
            ok = facts.get("target_offset_mm") is not None and facts["target_offset_mm"] <= lim
        elif key == "requires_calibration":
            ok = (not lim) or bool(facts.get("calibrated"))
        elif key == "min_tags":
            ok = facts.get("n_tags", 0) >= lim
        elif key == "max_expanded_uncertainty_mm":
            ok = facts.get("expanded_mm") is not None and facts["expanded_mm"] <= lim
        else:
            ok = False
        if not ok:
            return False
    return True
