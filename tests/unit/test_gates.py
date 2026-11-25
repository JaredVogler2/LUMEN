"""Gate logic (prompt 10): every way of failing closed, and what never passes."""

import json

import pytest

from lumen.policy.gates import GateInputs, GatePolicy, check_validated_domain, evaluate_gates, load_policy
from lumen.schemas import Uncertainty, UncertaintyComponent


def unc(expanded=0.4, angle=0.001):
    c = UncertaintyComponent(name="pixel noise (tag corners, datum edges)", kind="random", value_mm=expanded / 2, variance_mm2=(expanded / 2) ** 2)
    return Uncertainty(u_random_mm=expanded / 2, bias_bound_mm=0.0, bias_linear_mm=0.0, combined_standard_mm=expanded / 2, expanded_mm=expanded, u_angle_rad=angle, components=[c])


def good(**over):
    base = dict(
        operation_selected=True, placard_selected=True, spec_geometry_complete=True, spec_issues=[], surface_target="plane",
        tolerance_position_mm=2.0, tolerance_rotation_rad=0.0175,
        registered=True, n_tags=4, tags_consistent=True, independently_checked=True, crossval_sigma_frac=0.002, reduced_chi2=1.1, rms_px=0.03,
        all_tags_measured=True, frame_localized=True, frame_flags=[], frame_min_tag_edge_px=90.0, frame_n_tags=4, frame_calibrated=True, all_frames_calibrated=True,
        datums_present=(True, True), datums_confirmed=(True, True), datum_distinctness=(20.0, 20.0), datum_manual=(False, False),
        datum_plane_evidence=("configured", "configured"), datum_feature=("planar_edge", "planar_edge"), datum_declared_on_plane=(True, True),
        local_frame_valid=True, local_frame_reasons=[], axis_disagreement_deg=0.1, axis_limit_deg=1.0, target_in_view=True, extrapolation_ratio=0.8,
        uncertainty=unc(),
    )
    base.update(over)
    return GateInputs(**base)


def failed(rep):
    return {g.code for g in rep.gates if not g.passed}


def test_all_gates_pass_is_prototype_guidance_never_validated_by_default():
    rep = evaluate_gates(good())
    assert rep.status == "prototype_guidance" and not failed(rep) and not rep.physically_validated
    assert "NOT been validated" in rep.notice


@pytest.mark.parametrize("over,code", [
    (dict(operation_selected=False), "operation_selected"),
    (dict(spec_geometry_complete=False), "placard_geometry_defined"),
    (dict(surface_target="curved"), "surface_supported"),
    (dict(registered=False, n_tags=0), "scale_available"),
    (dict(tags_consistent=False, inconsistent_reasons=["size_mismatch"]), "tags_mutually_consistent"),
    (dict(pose_ambiguous=True), "pose_unambiguous"),
    (dict(frame_localized=False), "frame_registered"),
    (dict(frame_flags=["high_residual"]), "frame_registered"),
    (dict(datums_present=(True, False)), "datums_defined"),
    (dict(local_frame_valid=False, local_frame_reasons=["datums_nearly_parallel"]), "local_frame_valid"),
    (dict(target_in_view=False), "target_in_view"),
    (dict(datum_plane_evidence=("off_plane", "configured")), "datum_floor_on_plane"),
])
def test_blocking_gates(over, code):
    rep = evaluate_gates(good(**over))
    assert rep.status == "blocked" and code in failed(rep)
    assert rep.reasons and rep.suggestions


@pytest.mark.parametrize("over,code", [
    (dict(spec_issues=["missing_position_tolerance"]), "operation_record_complete"),
    (dict(all_tags_measured=False), "tag_sizes_measured"),
    (dict(n_tags=1, independently_checked=False), "independent_scale_check"),
    (dict(crossval_sigma_frac=0.03), "independent_check_sensitive"),
    (dict(reduced_chi2=40.0), "fit_consistent_with_noise"),
    (dict(frame_calibrated=False), "lens_characterised"),
    (dict(all_frames_calibrated=False), "lens_characterised"),
    (dict(frame_min_tag_edge_px=22.0), "tag_resolution"),
    (dict(datums_confirmed=(True, False)), "datums_confirmed"),
    (dict(datum_distinctness=(20.0, 1.1)), "datum_side_unambiguous"),
    (dict(datum_manual=(True, False)), "datum_floor_measured"),
    (dict(datum_feature=("planar_edge", "curve_tangent")), "datum_side_needs_3d"),
    (dict(datum_declared_on_plane=(None, True), datum_plane_evidence=("unverified", "configured")), "datum_floor_on_plane"),
    (dict(extrapolation_ratio=9.0), "target_distance"),
    (dict(uncertainty=unc(1.6)), "uncertainty_vs_tolerance"),
    (dict(uncertainty=unc(0.4, angle=0.02)), "angle_uncertainty_vs_tolerance"),
    (dict(tolerance_position_mm=None), "uncertainty_vs_tolerance"),
    (dict(reference_distance_checks=[{"passed": False, "message": "reference distance differs by 0.8 %"}]), "reference_distance"),
])
def test_qualifying_gates_give_research_preview(over, code):
    rep = evaluate_gates(good(**over))
    assert rep.status == "research_preview" and code in failed(rep)


def test_low_reprojection_error_alone_never_establishes_accuracy():
    # perfect fit, but nothing else: one nominal tag, no calibration, unconfirmed datums
    rep = evaluate_gates(good(rms_px=0.0, reduced_chi2=None, n_tags=1, independently_checked=False, all_tags_measured=False,
                              frame_calibrated=False, all_frames_calibrated=False, datums_confirmed=(False, False)))
    assert rep.status == "research_preview"
    assert {"independent_scale_check", "tag_sizes_measured", "lens_characterised", "datums_confirmed"} <= failed(rep)


def test_no_tolerance_is_never_invented():
    rep = evaluate_gates(good(tolerance_position_mm=None, tolerance_rotation_rad=None, spec_issues=["missing_position_tolerance"]))
    assert rep.status == "research_preview"
    g = next(g for g in rep.gates if g.code == "uncertainty_vs_tolerance")
    assert not g.passed and g.limit is None and "no position tolerance" in g.message


def test_blocking_wins_over_everything():
    rep = evaluate_gates(good(tags_consistent=False, uncertainty=unc(0.01)))
    assert rep.status == "blocked"


def test_demo_sessions_are_labelled_simulated():
    assert evaluate_gates(good(demo=True)).notice.startswith("SIMULATED")


def test_validated_status_needs_an_explicit_domain_record():
    domain = {"id": "lab-2026-01", "evidence": "report 123", "limits": {"min_tag_edge_px": 80, "requires_calibration": True, "min_tags": 2, "max_expanded_uncertainty_mm": 0.6}}
    facts = {"min_tag_edge_px": 90.0, "calibrated": True, "n_tags": 4, "expanded_mm": 0.4}
    assert check_validated_domain(domain, facts)
    assert evaluate_gates(good(validated_domain=domain, in_validated_domain=True)).status == "validated"
    # outside any one limit -> not validated
    assert not check_validated_domain(domain, {**facts, "min_tag_edge_px": 60.0})
    assert not check_validated_domain(domain, {**facts, "calibrated": False})
    # fail closed on malformed or unknown content
    assert not check_validated_domain(None, facts)
    assert not check_validated_domain({"limits": {}}, facts)  # no evidence reference
    assert not check_validated_domain({"evidence": "x", "limits": {"phase_of_moon": 1}}, facts)
    # a qualifying failure still wins over the domain
    assert evaluate_gates(good(validated_domain=domain, in_validated_domain=True, datums_confirmed=(False, True))).status == "research_preview"


def test_policy_file_rejects_unknown_keys(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"max_uncertainty_fraction_of_tolerance": 0.25}))
    assert load_policy(str(p)).max_uncertainty_fraction_of_tolerance == 0.25
    p.write_text(json.dumps({"max_uncertainty_fraction_of_tolerence": 5.0}))  # typo must not loosen a gate
    with pytest.raises(ValueError):
        load_policy(str(p))
    assert load_policy(None) == GatePolicy()


def test_stricter_policy_changes_the_outcome():
    rep = evaluate_gates(good(uncertainty=unc(0.7)), GatePolicy(max_uncertainty_fraction_of_tolerance=0.25))
    assert rep.status == "research_preview"
    assert evaluate_gates(good(uncertainty=unc(0.7))).status == "prototype_guidance"


def test_suggestion_points_at_the_largest_contribution():
    u = unc(1.8)
    u.components.append(UncertaintyComponent(name="lens distortion: not characterised", kind="systematic", distribution="rectangular", value_mm=3.0, variance_mm2=3.0))
    rep = evaluate_gates(good(uncertainty=u))
    g = next(g for g in rep.gates if g.code == "uncertainty_vs_tolerance")
    assert "lens distortion" in g.suggestion
