"""Placard perimeter measurement against synthetic truth (prompt 11).

The plane-to-image mapping used here is the true one, so the errors below are
those of the perimeter measurement alone.
"""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from lumen.geometry.homography import apply_h, dlt_homography
from lumen.synth import Box
from lumen.synth.fixtures import demo_placard, flat_panel_scene, phone_camera
from lumen.vision.placard import measure_placard, propose_rectangle
from tests.conftest import px, render_cached

CAM = phone_camera((330.0, -700.0, 440.0), width=1600, height=1200)


def truth_H(cam):
    P = np.array([[x, z] for x in (0, 200, 400, 600) for z in (50, 250, 450, 650)], float)
    return dlt_homography(P, np.array([px(cam, x, z) for x, z in P]))


def rect(x, z, w, h, ang=0.0):
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    wd, hd = np.array([c, s]), np.array([-s, c])
    ctr = np.array([x, z])
    return np.stack([ctr - w / 2 * wd + h / 2 * hd, ctr + w / 2 * wd + h / 2 * hd, ctr + w / 2 * wd - h / 2 * hd, ctr - w / 2 * wd - h / 2 * hd])


H = truth_H(CAM)
EXPECTED = rect(240, 300, 120, 60)


def shoot(placard=None, seed=7, **scene_kw):
    scene = flat_panel_scene()
    if placard is not None:
        scene = replace(scene, placards=(placard,))
    if scene_kw:
        scene = replace(scene, **scene_kw)
    return render_cached(scene, CAM, seed=seed).image


def errors(m, pl):
    got = apply_h(np.linalg.inv(H), m.corners_ideal)
    tc = rect(pl.center[0], pl.center[1], pl.width, pl.height, pl.rotation_deg)
    return float(np.max(np.linalg.norm(got - tc, axis=1))), float(np.linalg.norm(got.mean(0) - tc.mean(0))), got


@pytest.mark.parametrize(
    "pl",
    [
        demo_placard(),
        demo_placard(243.0, 298.5),
        demo_placard(228.0, 308.0, rotation_deg=2.0),
        demo_placard(rotation_deg=-4.0),
        demo_placard(275.0, 280.0),
        demo_placard(239.0, 299.0, corner_radius=6.0),
        demo_placard(241.0, 301.0, border=0.0, text_lines=0, albedo=(0.97, 0.97, 0.96)),
    ],
    ids=["on_target", "shifted", "shifted_rotated", "rotated", "far_off", "rounded_corners", "plain_white"],
)
def test_perimeter_is_measured_wherever_the_placard_actually_is(pl):
    m = measure_placard(shoot(pl), None, H, EXPECTED)
    assert m.complete and not m.flags, m.flags
    corner_err, centre_err, got = errors(m, pl)
    # printed border, text and rounded corners do not pull the outline inward
    assert centre_err < 0.03 and corner_err < 0.25
    w = np.linalg.norm(got[1] - got[0])
    assert abs(w - pl.width) < 0.35
    ang = np.degrees(np.arctan2(*(got[1] - got[0])[::-1]))
    assert abs(ang - pl.rotation_deg) < 0.06
    assert all(s.method == "esf" and s.support > 0.9 for s in m.sides)
    # reported precision is of the order of the actual scatter (1.66 px/mm here)
    assert 0.01 < np.sqrt(np.diag(m.corners_cov).max()) / m.proposal["px_per_mm"] < 0.12


def test_both_edge_signs_are_tried_and_the_recorded_size_decides():
    pl = demo_placard(243.0, 298.5)  # red border: dark against the panel, with a bright interior 3 mm further in
    m = measure_placard(shoot(pl), None, H, EXPECTED)
    tried = {t["polarity"]: t for t in m.proposal["polarities_tried"]}
    assert set(tried) == {1, -1}
    assert tried[m.proposal["edge_polarity"]]["size_mismatch"] < 0.01
    other = tried[-m.proposal["edge_polarity"]]
    assert other["sides_ok"] < 4 or other["size_mismatch"] > 0.04  # the inner border line is not the placard


def test_no_placard_is_reported_as_not_found():
    m = measure_placard(shoot(None, seed=1), None, H, EXPECTED)
    assert not m.ok and m.reason == "no_rectangle_stands_out" and "placard_not_found" in m.flags
    assert np.isnan(m.corners_ideal).all()


def test_wrong_size_part_is_measured_and_flagged():
    pl = demo_placard(240.0, 300.0, width=104.0, height=60.0)
    m = measure_placard(shoot(pl), None, H, EXPECTED)
    assert m.complete and "size_differs_from_record" in m.flags
    _, _, got = errors(m, pl)
    assert abs(np.linalg.norm(got[1] - got[0]) - 104.0) < 0.4


def test_folded_corner_is_flagged_but_rounded_corners_are_not():
    curled = demo_placard(242.0, 300.0, curl=(1, 12.0))
    m = measure_placard(shoot(curled), None, H, EXPECTED)
    assert m.complete and "corner_lifted_or_hidden:TR" in m.flags
    assert m.corner_gap_mm[1] == pytest.approx(12.0, abs=1.5) and max(g for i, g in enumerate(m.corner_gap_mm) if i != 1) < 2.0
    # the outline is still that of the flat placard: the fold does not move it
    assert errors(m, curled)[1] < 0.05
    rounded = demo_placard(239.0, 299.0, corner_radius=6.0)
    m2 = measure_placard(shoot(rounded), None, H, EXPECTED)
    assert not any(f.startswith("corner_lifted") for f in m2.flags)
    assert all(2.5 < g < 5.0 for g in m2.corner_gap_mm)


def test_covered_corner_is_still_located_from_the_sides_and_reported_as_not_seen():
    pl = demo_placard(241.0, 300.5)
    # a block resting on the panel hides the top-right corner (finger, tape, clamp)
    img = shoot(pl, boxes=(Box((286.0, -14.0, 318.0), (318.0, 0.0, 346.0)),))
    m = measure_placard(img, None, H, EXPECTED)
    assert m.complete
    assert any(f in m.flags for f in ("corner_lifted_or_hidden:TR", "corner_not_directly_seen:TR")), m.flags
    assert errors(m, pl)[0] < 0.4  # the hidden corner is where the two sides meet


def test_glare_or_overpainting_on_an_edge_lowers_its_support():
    pl = demo_placard(240.5, 300.0)
    img = shoot(pl).copy()
    # wash out most of the bottom side (a strong reflection)
    a, b = px(CAM, 190, 268), px(CAM, 285, 268)
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), 17)
    m = measure_placard(img, None, H, EXPECTED)
    bottom = m.sides[2]
    assert (not bottom.ok) or bottom.support < 0.85 or "glare_on_edge" in bottom.flags
    assert any(f.endswith(":bottom") for f in m.flags)
    if not bottom.ok:
        assert "partial_perimeter" in m.flags and not m.complete and m.ok


def test_placard_partly_outside_the_photo_gives_a_partial_result():
    cam = phone_camera((393.0, -560.0, 320.0), target=(393.0, 0.0, 300.0), width=1600, height=1200, hfov_deg=38.0)
    Hc = truth_H(cam)
    assert px(cam, 180, 300)[0] < 0 < px(cam, 230, 300)[0]  # the left end of the placard is cut off
    pl = demo_placard()
    img = render_cached(replace(flat_panel_scene(), placards=(pl,)), cam, seed=9).image
    m = measure_placard(img, None, Hc, EXPECTED)
    left = m.sides[3]
    assert not left.ok and left.reason in ("side_outside_image", "edge_not_found")
    assert m.ok and not m.complete and "partial_perimeter" in m.flags
    got = apply_h(np.linalg.inv(Hc), m.corners_ideal[[1, 2]])
    assert np.max(np.linalg.norm(got - rect(240, 300, 120, 60)[[1, 2]], axis=1)) < 0.4


def test_operator_marked_corners_are_snapped_to_the_edges_or_kept_as_marked():
    pl = demo_placard(243.0, 298.5)
    img = shoot(pl)
    truth = apply_h(H, rect(243.0, 298.5, 120, 60))
    rough = truth + np.array([[3.0, -2.5], [-2.0, 3.0], [2.5, 2.0], [-3.0, -1.5]])
    m = measure_placard(img, None, H, EXPECTED, manual_corners_ideal=rough, snap=True)
    assert m.complete and m.provenance == "manual" and all(s.method == "esf" for s in m.sides)
    assert errors(m, pl)[0] < 0.25
    kept = measure_placard(img, None, H, EXPECTED, manual_corners_ideal=rough, snap=False)
    assert np.allclose(kept.corners_ideal, rough, atol=1e-6) and all(s.method == "manual" for s in kept.sides)
    assert np.sqrt(kept.corners_cov[0, 0]) > 10 * np.sqrt(m.corners_cov[0, 0])  # hand-marked corners are much less certain


def test_proposal_reports_rivals_and_the_search_limit():
    # two identical placards 30 mm apart vertically... the nearer pose is taken, the rival is reported
    a, b = demo_placard(240.0, 262.0), demo_placard(240.0, 338.0, name="OTHER")
    img = render_cached(replace(flat_panel_scene(), placards=(a, b)), CAM, seed=11).image
    from lumen.vision.imaging import to_gray

    prop = propose_rectangle(to_gray(img), None, H, EXPECTED)
    assert prop["found"] and prop["ambiguous"] and prop["second_score_ratio"] > 0.8
    assert abs(abs(prop["offset_mm"][1]) - 38.0) < 1.0
    m = measure_placard(img, None, H, EXPECTED)
    assert "ambiguous_proposal" in m.flags
