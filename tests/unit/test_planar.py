"""Planar metrology against ground truth (prompt 04)."""

import numpy as np
import pytest

from lumen.errors import GeometryError
from lumen.geometry.camera import CameraModel
from lumen.synth import TagSticker
from lumen.synth.fixtures import DEMO_TAGS, demo_tags, flat_panel_scene, phone_camera
from lumen.synth.render import project_points
from lumen.vision.markers import detect_tags
from lumen.vision.planar import PlanarView, TagObs, backprojected_size, choose_gauge, localize_view, solve_planar
from lumen.vision.rectify import plan_rectification, rectify_image
from tests.conftest import render_cached

W, H = 1600, 1200
DIST = (0.09, -0.18, 0.0005, -0.0003, 0.0)


def _model(cam, source="charuco"):
    return CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist, source=source)


def _view(res, cam, name="f0", calibrated=False, ids=None, mode="auto"):
    cm = _model(cam) if calibrated else None
    dets = [d for d in detect_tags(res.image, camera=cm) if d.accepted and (ids is None or d.tag_id in ids)]
    return PlanarView(name, [TagObs.from_detection(d) for d in dets], cm, mode)


def _truth_plane(scene, gauge, xz):
    st = next(s for s in scene.stickers if s.tag_id == gauge)
    a = np.deg2rad(st.rotation_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(xz, dtype=float) - np.asarray(st.center)) @ R


def _map_error(scene, cam, sol, fid, calibrated):
    xs, zs = np.meshgrid(np.linspace(40, 700, 8), np.linspace(40, 620, 7))
    xz = np.stack([xs.ravel(), zs.ravel()], axis=1)
    uv, _ = project_points(cam, np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1))
    inside = (uv[:, 0] > 2) & (uv[:, 0] < cam.width - 3) & (uv[:, 1] > 2) & (uv[:, 1] < cam.height - 3)
    if calibrated:
        uv = _model(cam).undistort_pixels(uv)
    est = sol.image_to_plane(fid, uv)
    return np.linalg.norm(est - _truth_plane(scene, sol.layout.gauge_id, xz), axis=1)[inside]


@pytest.fixture(scope="module")
def shot():
    scene = flat_panel_scene()
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3)
    return scene, cam, render_cached(scene, cam, seed=1)


def test_four_tags_homography_is_metric_over_the_whole_panel(shot):
    scene, cam, res = shot
    sol = solve_planar([_view(res, cam)])
    err = _map_error(scene, cam, sol, "f0", False)
    assert sol.views["f0"].mode == "homography"
    assert err.mean() < 0.08 and err.max() < 0.2
    assert sol.consistent and sol.independently_checked and not sol.flags
    # the layout is recovered without knowing any inter-tag spacing
    for tid, c, rot in DEMO_TAGS:
        cx, cy, th = sol.layout.poses[tid]
        assert np.linalg.norm(np.array([cx, cy]) - _truth_plane(scene, sol.layout.gauge_id, [c])[0]) < 0.15
    for chk in sol.checks.values():
        assert chk.checked and abs(chk.size_ratio - 1.0) < 3 * chk.size_ratio_sigma + 1e-3


def test_one_tag_fixes_the_mapping_exactly_but_extrapolates_poorly(shot):
    scene, cam, res = shot
    one = solve_planar([_view(res, cam, ids=[3])])
    four = solve_planar([_view(res, cam)])
    assert one.rms_px < 1e-6 and one.dof == 0
    assert "single_tag_no_independent_check" in one.flags and not one.independently_checked
    e1 = _map_error(scene, cam, one, "f0", False)
    e4 = _map_error(scene, cam, four, "f0", False)
    assert e1.max() > 5 * e4.max(), "redundant tags must visibly improve the mapping away from the tags"


def test_pose_mode_with_calibration_extrapolates_better_from_one_tag(shot):
    scene, cam, res = shot
    free = solve_planar([_view(res, cam, ids=[3])])
    posed = solve_planar([_view(res, cam, calibrated=True, ids=[3])])
    v = posed.views["f0"]
    assert v.mode == "pose" and v.pose is not None
    assert _map_error(scene, cam, posed, "f0", True).max() < 0.5 * _map_error(scene, cam, free, "f0", False).max()
    assert "pose_ambiguous" not in v.flags and (v.ambiguity_delta_chi2 is None or v.ambiguity_delta_chi2 > 16)
    # recovered camera centre agrees with the truth (expressed in the gauge-tag frame: z out of the panel)
    C = v.pose.center
    assert abs(abs(C[2]) - 720.0) < 6.0


def test_lens_distortion_calibrated_vs_uncalibrated():
    scene = flat_panel_scene()
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3, dist=DIST)
    res = render_cached(scene, cam, seed=1)
    cal = solve_planar([_view(res, cam, calibrated=True, mode="homography")])
    unc = solve_planar([_view(res, cam)])
    assert _map_error(scene, cam, cal, "f0", True).max() < 0.25
    assert _map_error(scene, cam, unc, "f0", False).max() > 1.0
    assert "high_chi2" in unc.flags and "high_chi2" not in cal.flags


def test_non_coplanar_tag_is_detected_and_excluded():
    tags = tuple(TagSticker(t, c, r, lift=(6.0 if t == 11 else 0.0)) for t, c, r in DEMO_TAGS)
    scene = flat_panel_scene(tags=tags)
    cam = phone_camera((950, -520, 650), target=(330, 0, 300), width=W, height=H, roll_deg=-8)  # oblique: parallax shows
    res = render_cached(scene, cam, seed=1)
    sol = solve_planar([_view(res, cam)])
    assert 11 in sol.excluded_tags and 11 not in sol.layout.poses
    assert sol.consistent and _map_error(scene, cam, sol, "f0", False).max() < 0.6


def test_wrong_declared_size_is_detected_and_excluded():
    tags = tuple(TagSticker(t, c, r, size=(48.0 if t == 7 else 50.0)) for t, c, r in DEMO_TAGS)  # 7 is really 48 mm
    scene = flat_panel_scene(tags=tags)
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3)
    res = render_cached(scene, cam, seed=1)
    sol = solve_planar([_view(res, cam)])  # all declared 50 mm
    assert sol.excluded_tags.get(7) and "size_mismatch" in sol.excluded_tags[7]
    assert _map_error(scene, cam, sol, "f0", False).max() < 0.2


def test_two_inconsistent_tags_fail_closed():
    tags = (TagSticker(3, (95.0, 125.0), 12.0, size=50.0), TagSticker(20, (450.0, 505.0), 190.0, size=46.0))
    scene = flat_panel_scene(tags=tags)
    cam = phone_camera((430, -720, 430), width=W, height=H)
    res = render_cached(scene, cam, seed=1)
    sol = solve_planar([_view(res, cam)])
    # with two tags nobody can tell which one is wrong: nothing is excluded, the solution is flagged
    assert not sol.consistent and "tags_inconsistent" in sol.flags and not sol.excluded_tags


def test_no_tags_is_an_explicit_error():
    with pytest.raises(GeometryError) as e:
        solve_planar([PlanarView("f0", [])])
    assert e.value.code == "no_tags" and e.value.suggestions


def test_multi_view_layout_and_partial_visibility():
    scene = flat_panel_scene()
    cams = [
        phone_camera((430, -720, 430), width=W, height=H, roll_deg=3),
        phone_camera((120, -650, 520), target=(240, 0, 300), width=W, height=H, roll_deg=-6),
        phone_camera((700, -640, 330), target=(300, 0, 300), width=W, height=H, roll_deg=9),
    ]
    views = [_view(render_cached(scene, c, seed=1), c, name=f"f{i}") for i, c in enumerate(cams)]
    views[1] = PlanarView("f1", [t for t in views[1].tags if t.tag_id in (3, 11)], None)  # two tags hidden in view 1
    views[2] = PlanarView("f2", [t for t in views[2].tags if t.tag_id == 20], None)  # a single tag in view 2
    sol = solve_planar(views)
    assert set(sol.views) == {"f0", "f1", "f2"} and sol.n_tags == 4
    errs = [_map_error(scene, c, sol, f"f{i}", False).max() for i, c in enumerate(cams)]
    assert errs[0] < 0.25 and errs[1] < 0.6
    # a view with a single tag is still registered (the layout is known) but it
    # extrapolates from one 50 mm square, so it is much weaker than the others
    assert sol.views["f2"].n_tags == 1 and errs[0] < errs[2] < 8.0


def test_disconnected_tag_is_reported():
    a = TagObs(1, np.array([[100.0, 100], [200, 100], [200, 200], [100, 200]]), np.eye(8) * 1e-3, 50.0)
    b = TagObs(2, np.array([[400.0, 100], [500, 100], [500, 200], [400, 200]]), np.eye(8) * 1e-3, 50.0)
    c = TagObs(3, np.array([[300.0, 300], [400, 300], [400, 400], [300, 400]]), np.eye(8) * 1e-3, 50.0)
    # views share no tag between {1, 2} and {3}
    sol = solve_planar([PlanarView("v0", [a, b]), PlanarView("v1", [c])], gauge=1)
    assert sol.excluded_tags.get(3) == ["not_connected_to_gauge"] and "v1" not in sol.views


def test_localize_view_and_moved_marker_detection(shot):
    scene, cam, res = shot
    sol = solve_planar([_view(res, cam)])
    cam2 = phone_camera((250, -650, 350), target=(300, 0, 280), width=W, height=H, roll_deg=-5)
    res2 = render_cached(scene, cam2, seed=2)
    loc = localize_view(sol.layout, _view(res2, cam2, name="v"))
    xz = np.array([[240.0, 300.0], [600.0, 100.0]])
    uv, _ = project_points(cam2, np.stack([xz[:, 0], np.zeros(2), xz[:, 1]], axis=1))
    est = (np.linalg.inv(loc.H) @ np.concatenate([uv, np.ones((2, 1))], axis=1).T).T
    est = est[:, :2] / est[:, 2:]
    assert np.max(np.linalg.norm(est - _truth_plane(scene, sol.layout.gauge_id, xz), axis=1)) < 0.25
    assert not loc.flags and set(loc.layout_sens) == {t for t in sol.layout.poses if t != sol.layout.gauge_id}
    # now tag 7 has been moved by 4 mm between registration and this photo
    moved = tuple(TagSticker(t, ((c[0] + 4.0, c[1]) if t == 7 else c), r) for t, c, r in DEMO_TAGS)
    res3 = render_cached(flat_panel_scene(tags=moved), cam2, seed=2)
    loc3 = localize_view(sol.layout, _view(res3, cam2, name="v"))
    assert "high_residual" in loc3.flags or "high_chi2" in loc3.flags
    with pytest.raises(GeometryError):
        localize_view(sol.layout, PlanarView("v", []))


def test_gauge_choice_is_deterministic(shot):
    _, cam, res = shot
    v = _view(res, cam)
    assert choose_gauge([v]) == choose_gauge([PlanarView("x", list(reversed(v.tags)))])


def test_backprojected_size_of_a_true_tag_is_one(shot):
    _, cam, res = shot
    v = _view(res, cam)
    sol = solve_planar([v])
    for t in v.tags:
        ratio, shape = backprojected_size(np.linalg.inv(sol.views["f0"].H), t.corners, t.size_mm)
        assert ratio == pytest.approx(1.0, abs=2e-3) and shape < 2e-3


def test_rectified_view_is_metric_and_upright(shot):
    scene, cam, res = shot
    sol = solve_planar([_view(res, cam)])
    H = sol.views["f0"].H
    spec = plan_rectification(H, cam.width, cam.height, sol.layout.all_corners(), max_long_edge_px=1200)
    rect, valid = rectify_image(res.image, None, H, spec)
    assert rect.shape[:2] == (spec.height, spec.width) and valid.mean() > 0.3 and not spec.mirrored
    # tags are squares of 50 mm in the rectified picture
    dets = {d.tag_id: d for d in detect_tags(rect) if d.accepted}
    assert len(dets) == 4
    for d in dets.values():
        side = np.mean(d.edge_px) * spec.mm_per_px
        assert side == pytest.approx(50.0, abs=0.25)
        assert np.ptp(d.edge_px) < 0.6
    # image "up" stays up: tag 11 (higher on the panel) is above tag 3
    assert dets[11].corners_px[:, 1].mean() < dets[3].corners_px[:, 1].mean()
    # round trip rect px -> plane -> rect px
    p = np.array([[10.0, 20.0], [333.3, 444.4]])
    assert np.allclose(spec.plane_to_rect(spec.rect_to_plane(p)), p, atol=1e-9)
