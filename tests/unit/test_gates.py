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


def test_a_draft_or_superseded_record_stays_a_preview():
    assert "operation_record_released" in {g.code for g in evaluate_gates(good()).gates if g.passed}
    for status in ("draft", "superseded"):
        rep = evaluate_gates(good(record_status=status))
        assert failed(rep) == {"operation_record_released"} and rep.status == "research_preview"
        g = next(g for g in rep.gates if g.code == "operation_record_released")
        assert status in g.message and g.suggestion
    # nothing is said about the record's status before a record is chosen
    assert "operation_record_released" not in {g.code for g in evaluate_gates(good(operation_selected=False)).gates}


def test_a_step_beside_an_automatic_datum_withholds_guidance_until_the_operator_chooses():
    # the trim's front corner: 8 mm proud, right at the line that was applied as the floor datum
    nb = {"id": "c2", "height_mm": 7.9, "gap_mm": -0.4, "distance_mm": 7.9, "length_mm": 500.0, "relation": "beside"}
    rep = evaluate_gates(good(datum_depth_neighbour=(nb, None)))
    assert failed(rep) == {"datum_floor_no_step_beside"} and rep.status == "research_preview"
    g = next(g for g in rep.gates if g.code == "datum_floor_no_step_beside")
    assert "8 mm in front of the panel face" in g.message and "right beside the floor datum" in g.message and "Choose detected line" in g.suggestion
    # the trim's front face ending at the floor, 60 mm further out; and a recess instead of a raised part
    far = {**nb, "gap_mm": -60.2, "height_mm": -8.1, "relation": "beyond"}
    g = next(g for g in evaluate_gates(good(datum_depth_neighbour=(None, far))).gates if g.code == "datum_side_no_step_beside")
    assert "8 mm behind the panel face" in g.message and "60 mm beyond the side datum" in g.message
    # no neighbour (the operator chose the line, or there is none): the check does not appear at all
    assert not any(g.code.endswith("no_step_beside") for g in evaluate_gates(good()).gates)


def test_scale_that_rests_on_declared_sizes_alone_is_said_not_hidden():
    rep = evaluate_gates(good())
    g = next(g for g in rep.gates if g.code == "scale_source")
    # a caveat: it does not lower the status, but it is not a tick either, and it says how to close the gap
    assert g.passed and g.caveat and g.severity == "info" and rep.status == "prototype_guidance"
    assert "declared tag sizes alone" in g.message and "rule" in g.suggestion
    assert g.suggestion not in rep.suggestions, "what to do next lists only what stands in the way"
    # with a rule measurement the caveat is replaced by the check itself
    ok = evaluate_gates(good(reference_consistent=True, reference_message="Scale agrees with an independent length measurement over 300 mm."))
    codes = {g.code: g for g in ok.gates}
    assert "scale_source" not in codes and codes["independent_length_check"].passed and not codes["independent_length_check"].caveat
    bad = evaluate_gates(good(reference_consistent=False, reference_message="An independent length measurement contradicts the scale."))
    assert bad.status == "blocked" and "independent_length_check" in failed(bad)
    # ticks stay ticks
    assert not next(g for g in rep.gates if g.code == "operation_selected").caveat


def test_tags_that_were_set_aside_are_named_in_the_report():
    rep = evaluate_gates(good(n_tags=3, excluded_tags={41: ["moved_or_misread"]}))
    g = next(g for g in rep.gates if g.code == "tags_left_out")
    assert g.passed and g.caveat and rep.status == "prototype_guidance"
    assert "ID 41 (it moved between photos or was misread in one)" in g.message and "remaining 3" in g.message and "capture again" in g.suggestion
    two = next(g for g in evaluate_gates(good(n_tags=2, excluded_tags={41: ["moved_or_misread"], 317: ["off_the_common_plane", "unknown_code"]})).gates if g.code == "tags_left_out")
    assert two.message.startswith("Tags were left out") and "ID 317 (it is not on the plane of the other tags; unknown_code)" in two.message
    assert not any(g.code == "tags_left_out" for g in evaluate_gates(good()).gates)


def test_contradictions_that_setting_tags_aside_did_not_resolve_keep_it_a_preview():
    base = dict(multiview=True, coplanarity_verdict="coplanar", coplanarity_power_mm=0.6)
    assert not any(g.code == "photos_agree" for g in evaluate_gates(good(**base, multiview_flags=["off_plane_tags_excluded"])).gates)
    rep = evaluate_gates(good(**base, multiview_flags=["inconsistent_observations"], n_tags=2, excluded_tags={77: ["moved_or_misread"], 233: ["moved_or_misread"]}))
    assert failed(rep) == {"photos_agree"} and rep.status == "research_preview"
    g = next(g for g in rep.gates if g.code == "photos_agree")
    assert "still contradict" in g.message and "glare" in g.suggestion
