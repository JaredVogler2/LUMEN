"""Multi-view localisation against ground truth (prompt 05)."""

import numpy as np
import pytest

from lumen.errors import GeometryError
from lumen.geometry.camera import CameraModel
from lumen.geometry.se3 import Pose
from lumen.synth import TagSticker
from lumen.synth.fixtures import DEMO_TAGS, flat_panel_scene, phone_camera, sweep_stations
from lumen.synth.render import project_points
from lumen.vision.markers import detect_tags
from lumen.vision.multiview import MVView, focal_from_tags, solve_multiview, triangulate_line, triangulate_point, view_plane_of_image_line
from lumen.vision.planar import PlanarView, TagObs, solve_planar
from tests.conftest import render_cached

W, H = 1280, 960


def model(cam, source="charuco"):
    return CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist, source=source)


def scene_T_gauge(scene, gid):
    """Pose of the gauge tag frame in scene coordinates (x right, y up in the tag, z toward the viewer)."""
    st = next(s for s in scene.stickers if s.tag_id == gid)
    a = np.deg2rad(st.rotation_deg)
    R = np.stack([[np.cos(a), 0, np.sin(a)], [-np.sin(a), 0, np.cos(a)], [0, -1, 0]], axis=1)
    return Pose(R, np.array([st.center[0], -st.lift, st.center[1]]))


def make_views(scene, cams, calibrated=True, sizes=None, drop=(), swap=None):
    views = []
    for i, cam in enumerate(cams):
        sc = swap[i] if (swap and i in swap) else scene
        res = render_cached(sc, cam, seed=i)
        cm = model(cam) if calibrated else None
        dets = [d for d in detect_tags(res.image, camera=cm) if d.accepted]
        tags = [TagObs(d.tag_id, d.corners_ideal, d.cov, (sizes or {}).get(d.tag_id, d.size_mm)) for d in dets if (i, d.tag_id) not in drop]
        views.append(MVView(f"f{i}", tags, cm, {d.tag_id: d.corners_px for d in dets}, (cam.width, cam.height)))
    return views


def errors(scene, cams, sol):
    T = scene_T_gauge(scene, sol.gauge_id)
    cam_err = [np.linalg.norm(T.apply(sol.camera_center(f"f{i}")) - np.array(c.C)) for i, c in enumerate(cams) if f"f{i}" in sol.cameras]
    tag_err = []
    for tid in sol.tags:
        st = next(s for s in scene.stickers if s.tag_id == tid)
        tag_err.append(np.linalg.norm(T.apply(sol.tags[tid].t) - np.array([st.center[0], -st.lift, st.center[1]])))
    return max(cam_err), max(tag_err)


@pytest.fixture(scope="module")
def sweep():
    scene = flat_panel_scene()
    cams = [phone_camera(e, width=W, height=H, roll_deg=r) for e, r in sweep_stations(6, distance=740, height=430, span_deg=55)]
    return scene, cams


def test_unknown_tag_transforms_and_camera_rolls_are_recovered(sweep):
    scene, cams = sweep
    sol = solve_multiview(make_views(scene, cams))
    cam_err, tag_err = errors(scene, cams, sol)
    assert len(sol.cameras) == 6 and sorted(sol.tags) == [3, 7, 11, 20] and not sol.flags
    assert cam_err < 0.6 and tag_err < 0.3
    assert sol.rms_px < 0.06 and sol.checks["positive_depth"]
    c = sol.checks["coplanarity"]
    assert c["verdict"] == "coplanar" and c["worst_center_offset_mm"] < 0.2 and c["resolving_power_mm"] < 1.5
    assert sol.checks["diversity"]["diverse"] and sol.checks["diversity"]["max_view_angle_deg"] > 40
    assert sol.checks["held_out"]["consistent"] and sol.checks["held_out"]["max_ratio"] < 2.0
    assert all(v["consistent"] for k, v in sol.checks["sizes"].items() if k != "suspect")
    # camera rolls differ between views; all recovered with the camera in front of the panel
    assert all(sol.tags[sol.gauge_id].R[:, 2] @ (sol.camera_center(f) - sol.tags[sol.gauge_id].t) > 0 for f in sol.cameras)


def test_multi_view_beats_single_view_for_the_tag_layout(sweep):
    scene, cams = sweep
    multi = solve_multiview(make_views(scene, cams))
    single = solve_multiview(make_views(scene, cams[:1]), checks=False)
    assert errors(scene, cams, multi)[1] < errors(scene, cams[:1], single)[1]
    # posterior uncertainty of a tag pose shrinks with more views
    a, b = multi.param_index["tag:7"]
    a1, b1 = single.param_index["tag:7"]
    assert np.trace(multi.cov[a:b, a:b]) < 0.5 * np.trace(single.cov[a1:b1, a1:b1])


def test_occlusion_and_partial_visibility(sweep):
    scene, cams = sweep
    drop = {(0, 3), (0, 11), (2, 7), (4, 20), (4, 7), (4, 11)}  # view 4 sees a single tag
    sol = solve_multiview(make_views(scene, cams, drop=drop))
    cam_err, tag_err = errors(scene, cams, sol)
    assert len(sol.cameras) == 6 and cam_err < 2.5 and tag_err < 0.4


def test_graph_disconnect_is_reported(sweep):
    scene, cams = sweep
    # views 0-2 see only {3, 11}; views 3-5 see only {7, 20}: no view links the two groups
    drop = {(i, t) for i in (0, 1, 2) for t in (7, 20)} | {(i, t) for i in (3, 4, 5) for t in (3, 11)}
    sol = solve_multiview(make_views(scene, cams, drop=drop))
    assert "graph_disconnected" in sol.flags
    assert set(sol.excluded_tags) in ({7, 20}, {3, 11}) and len(sol.excluded_views) == 3
    assert all(r == ["not_connected_to_gauge"] for r in sol.excluded_tags.values())


def test_moving_tag_is_detected_and_excluded(sweep):
    scene, cams = sweep
    moved = flat_panel_scene(tags=tuple(TagSticker(t, ((c[0] + 6.0, c[1] + 3.0) if t == 7 else c), r) for t, c, r in DEMO_TAGS))
    sol = solve_multiview(make_views(scene, cams, swap={3: moved, 4: moved, 5: moved}))  # tag 7 moved after view 2
    assert sol.excluded_tags.get(7) == ["moved_or_misread"]
    cam_err, tag_err = errors(scene, cams, sol)
    assert cam_err < 1.0 and tag_err < 0.3


def test_wrong_tag_size_is_detected(sweep):
    scene, cams = sweep
    sol = solve_multiview(make_views(scene, cams, sizes={7: 52.0}))
    assert sol.excluded_tags.get(7) == ["declared_size_inconsistent"]
    assert errors(scene, cams, sol)[1] < 0.3


def test_inconsistent_plane_is_reported(sweep):
    scene, cams = sweep
    lifted = flat_panel_scene(tags=tuple(TagSticker(t, c, r, lift=(8.0 if t == 11 else 0.0)) for t, c, r in DEMO_TAGS))
    sol = solve_multiview(make_views(lifted, cams))
    c = sol.checks["coplanarity"]
    assert c["verdict"] == "not_coplanar" and "tags_not_coplanar" in sol.flags
    assert c["off_plane_tags"] == [11] and c["majority_plane_tags"] == [3, 7, 20]
    # the solver does not force coplanarity: the raised tag is where it really is
    T = scene_T_gauge(lifted, sol.gauge_id)
    assert T.apply(sol.tags[11].t)[1] == pytest.approx(-8.0, abs=0.5)


def test_narrow_baseline_cannot_establish_coplanarity():
    scene = flat_panel_scene()
    cams = [phone_camera((300 + 10 * k, -700, 400), width=W, height=H) for k in range(3)]
    sol = solve_multiview(make_views(scene, cams))
    assert "narrow_baseline" in sol.flags and not sol.checks["diversity"]["diverse"]
    assert sol.checks["coplanarity"]["verdict"] == "not_established"


def test_self_calibration_recovers_focal_length_and_reports_its_uncertainty(sweep):
    scene, cams = sweep
    views = make_views(scene, cams, calibrated=False)
    assert focal_from_tags(views, W, H) == pytest.approx(cams[0].fx, rel=0.03)
    sol = solve_multiview(views, refine_intrinsics=("f",))
    sc = sol.checks["self_calibration"]
    assert sc["usable"] and abs(sc["f"] - cams[0].fx) < max(5 * sc["sigma_f"], 0.002 * cams[0].fx)
    assert errors(scene, cams, sol)[0] < 1.0


def test_self_calibration_with_frontal_views_is_not_usable():
    scene = flat_panel_scene()
    cams = [phone_camera((275 + 8 * k, -900, 300), target=(275 + 8 * k, 0, 300), width=W, height=H) for k in range(3)]
    sol = solve_multiview(make_views(scene, cams, calibrated=False), refine_intrinsics=("f",))
    sc = sol.checks["self_calibration"]
    assert not sc["usable"]


def test_wrong_scale_shows_up_as_distance_error_not_as_residual(sweep):
    """Low reprojection error does not establish accuracy: with every size
    declared 4 % too large the fit is just as good and everything is 4 % off."""
    scene, cams = sweep
    good = solve_multiview(make_views(scene, cams), checks=False)
    bad = solve_multiview(make_views(scene, cams, sizes={3: 52.0, 7: 52.0, 11: 52.0, 20: 52.0}), checks=False)
    assert bad.rms_px == pytest.approx(good.rms_px, abs=0.01)
    d_good = np.linalg.norm(good.tags[20].t - good.tags[3].t)
    d_bad = np.linalg.norm(bad.tags[20].t - bad.tags[3].t)
    assert d_bad / d_good == pytest.approx(1.04, abs=0.002)


def test_requires_intrinsics_or_image_size():
    t = TagObs(1, np.array([[100.0, 100], [200, 100], [200, 200], [100, 200]]), np.eye(8) * 1e-3, 50.0)
    with pytest.raises(GeometryError):
        solve_multiview([MVView("a", [t])])
    with pytest.raises(GeometryError):
        solve_multiview([MVView("a", [])])


def test_agrees_with_the_planar_solver(sweep):
    scene, cams = sweep
    views = make_views(scene, cams)
    mv = solve_multiview(views, checks=False)
    pl = solve_planar([PlanarView(v.frame_id, v.tags, v.camera, "homography") for v in views])
    # distances between tag centres agree between the 3-D and the planar solution
    for i, j in ((3, 20), (7, 11), (3, 7)):
        d3 = np.linalg.norm(mv.tags[i].t - mv.tags[j].t)
        d2 = np.linalg.norm(np.array(pl.layout.poses[i][:2]) - np.array(pl.layout.poses[j][:2]))
        assert d3 == pytest.approx(d2, abs=0.25)


def test_line_triangulation_separates_stationary_lines_from_view_dependent_ones(sweep):
    scene, cams = sweep
    sol = solve_multiview(make_views(scene, cams), checks=False)
    K = sol.intrinsics.without_distortion().K
    T = scene_T_gauge(scene, sol.gauge_id)

    def planes(points_for_view):
        out = []
        for i, cam in enumerate(cams):
            a, b = points_for_view(i, cam)
            uv, _ = project_points(cam, np.array([a, b]))
            out.append(view_plane_of_image_line(sol.cameras[f"f{i}"], K, uv[0], uv[1]))
        return out

    # the floor seam: one physical line, same in all views
    seam = triangulate_line(planes(lambda i, c: ((50.0, 0.0, 0.0), (600.0, 0.0, 0.0))))
    assert seam["rms_offset_mm"] < 0.3 and seam["max_angle_deg"] < 0.1
    p = T.apply(seam["point"])
    d = T.rotate(seam["direction"])
    assert abs(p[1]) < 0.5 and abs(p[2]) < 0.5 and abs(abs(d[0]) - 1.0) < 1e-3
    # a "line" that is a different 3-D line in every view (like a silhouette or a shadow edge)
    moving = triangulate_line(planes(lambda i, c: ((50.0, -5.0 * i, 40.0 + 12.0 * i), (600.0, -5.0 * i, 40.0 + 12.0 * i))))
    assert moving["rms_offset_mm"] > 10 * seam["rms_offset_mm"] + 1.0
    with pytest.raises(ValueError):
        triangulate_line(planes(lambda i, c: ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))[:1])


def test_point_triangulation(sweep):
    scene, cams = sweep
    sol = solve_multiview(make_views(scene, cams), checks=False)
    K = sol.intrinsics.without_distortion().K
    X_scene = np.array([240.0, 0.0, 300.0])
    obs = []
    for i, cam in enumerate(cams):
        uv, _ = project_points(cam, X_scene[None])
        obs.append((sol.cameras[f"f{i}"], K, uv[0]))
    X, rms = triangulate_point(obs)
    assert np.linalg.norm(scene_T_gauge(scene, sol.gauge_id).apply(X) - X_scene) < 0.3 and rms < 0.2
