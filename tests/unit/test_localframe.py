import numpy as np
import pytest

from lumen.geometry.lines import Line2D
from lumen.geometry.localframe import build_local_frame, rectangle_pose_in_frame, target_rectangle

FLOOR = Line2D.from_points([-100, 0], [500, 0])        # y = 0
SIDE = Line2D.from_points([0, -50], [0, 600])           # x = 0
IN = np.array([250.0, 300.0])                           # tags are up and to the right


def frame(policy="floor_primary", floor=FLOOR, side=SIDE, interior=IN, **kw):
    kw.setdefault("max_axis_disagreement_deg", 1.0)
    return build_local_frame(floor, side, interior, policy=policy, **kw)


def test_orthogonal_datums_all_policies_agree():
    for policy in ("floor_primary", "side_primary", "perpendicular_offsets"):
        f = frame(policy)
        assert f.valid and f.raw_angle_deg == pytest.approx(90.0) and f.side_of_panel == "left"
        assert np.allclose(f.to_plane(np.array([120.0, 300.0])), [120.0, 300.0])
        assert np.allclose(f.to_local(np.array([[120.0, 300.0]])), [[120.0, 300.0]])


def test_signs_follow_the_interior_point_not_the_line_direction():
    # same lines given with reversed direction: nothing changes
    f = frame(floor=Line2D.from_points([500, 0], [-100, 0]), side=Line2D.from_points([0, 600], [0, -50]))
    assert np.allclose(f.to_plane(np.array([10.0, 20.0])), [10.0, 20.0])
    # side datum on the right of the panel: +x points left in the plane frame
    right = Line2D.from_points([600, 0], [600, 500])
    f = frame(side=right)
    assert f.side_of_panel == "right"
    assert np.allclose(f.to_plane(np.array([100.0, 50.0])), [500.0, 50.0])


def test_configured_negative_direction():
    f = frame(x_positive="away_from_panel")
    assert np.allclose(f.to_plane(np.array([-100.0, 50.0])), [100.0, 50.0])
    f = frame(z_positive="away_from_panel")
    assert np.allclose(f.to_plane(np.array([100.0, -50.0])), [100.0, 50.0])


def test_unconfigured_sign_invalidates_the_frame():
    f = frame(x_positive="unset")
    assert not f.valid and "x_sign_not_configured" in f.reasons


def test_axis_disagreement_limit_and_preserved_raw_angle():
    tilted = Line2D.from_points([0, 0], [np.sin(np.deg2rad(3.0)) * 500, np.cos(np.deg2rad(3.0)) * 500])  # 3 deg off vertical
    f = frame(side=tilted)
    assert not f.valid and "axis_disagreement_exceeds_limit" in f.reasons
    assert f.raw_angle_deg == pytest.approx(87.0, abs=1e-6) and f.axis_disagreement_deg == pytest.approx(3.0, abs=1e-6)
    assert frame(side=tilted, max_axis_disagreement_deg=5.0).valid
    f = frame(side=tilted, max_axis_disagreement_deg=None)
    assert not f.valid and "axis_limit_not_configured" in f.reasons
    # perpendicular_offsets makes no orthogonality assumption and needs no limit
    assert frame("perpendicular_offsets", side=tilted, max_axis_disagreement_deg=None).valid


def test_policies_differ_exactly_as_documented_for_oblique_datums():
    ang = np.deg2rad(10.0)
    side = Line2D.from_points([0, 0], [np.sin(ang) * 500, np.cos(ang) * 500])
    p = np.array([200.0, 300.0])  # local coordinates
    fp = frame("floor_primary", side=side, max_axis_disagreement_deg=20.0).to_plane(p)
    sp = frame("side_primary", side=side, max_axis_disagreement_deg=20.0).to_plane(p)
    po = frame("perpendicular_offsets", side=side).to_plane(p)
    # floor_primary: x along the floor, z perpendicular to it
    assert np.allclose(fp, [200.0, 300.0])
    # side_primary: z along the side line, x perpendicular to it
    assert np.allclose(sp, 300.0 * np.array([np.sin(ang), np.cos(ang)]) + 200.0 * np.array([np.cos(ang), -np.sin(ang)]))
    # perpendicular_offsets: perpendicular distances to both lines
    assert float(FLOOR.oriented_towards(IN).signed_distance(po)) == pytest.approx(300.0)
    assert float(side.oriented_towards(IN).signed_distance(po)) == pytest.approx(200.0)
    assert not np.allclose(fp, sp) and not np.allclose(fp, po)


def test_parallel_datums_give_no_origin():
    f = frame(side=Line2D.from_points([-100, 50], [500, 51]))
    assert not f.valid and "datums_nearly_parallel" in f.reasons


def test_unexpected_side_is_flagged():
    assert frame(expected_side="left").valid
    f = frame(expected_side="right")
    assert not f.valid and "side_datum_on_unexpected_side" in f.reasons


def test_round_trip_dict():
    f = frame("perpendicular_offsets", side=Line2D.from_points([0, 0], [40, 500]))
    g = type(f).from_dict(f.to_dict())
    assert np.allclose(g.M, f.M) and g.policy == f.policy and g.valid == f.valid


@pytest.mark.parametrize("ref,center", [("center", (240, 300)), ("bottom_left", (300, 330)), ("bottom_right", (180, 330)), ("top_left", (300, 270)), ("top_right", (180, 270)), ("bottom_center", (240, 330)), ("left_center", (300, 300))])
def test_reference_points(ref, center):
    r = target_rectangle(frame(), 240.0, 300.0, 120.0, 60.0, 0.0, ref)
    assert np.allclose(r.center, center) and np.allclose(r.reference, [240.0, 300.0])
    assert np.allclose(np.linalg.norm(r.corners[1] - r.corners[0]), 120.0) and np.allclose(np.linalg.norm(r.corners[0] - r.corners[3]), 60.0)


def test_rectangle_is_a_true_rectangle_even_in_an_oblique_frame():
    side = Line2D.from_points([0, 0], [80, 500])
    r = target_rectangle(frame("perpendicular_offsets", side=side), 150.0, 200.0, 100.0, 50.0, np.deg2rad(7.0), "center")
    c = r.corners
    for k in range(4):
        a, b = c[(k + 1) % 4] - c[k], c[(k + 2) % 4] - c[(k + 1) % 4]
        assert abs(a @ b) < 1e-9
    assert np.linalg.norm(c[1] - c[0]) == pytest.approx(100.0) and np.linalg.norm(c[2] - c[1]) == pytest.approx(50.0)


def test_rotation_is_from_plus_x_toward_plus_z_for_both_handednesses():
    for side in (SIDE, Line2D.from_points([600, 0], [600, 500])):
        f = frame(side=side)
        r = target_rectangle(f, 200.0, 200.0, 100.0, 40.0, np.deg2rad(10.0), "center")
        top_right_local = f.to_local(r.corners[1])
        bottom_left_local = f.to_local(r.corners[3])
        assert top_right_local[0] > bottom_left_local[0] and top_right_local[1] > bottom_left_local[1]
        back = rectangle_pose_in_frame(f, r.corners)
        assert back["angle"] == pytest.approx(np.deg2rad(10.0)) and back["x"] == pytest.approx(200.0) and back["z"] == pytest.approx(200.0)
        assert back["width"] == pytest.approx(100.0) and back["height"] == pytest.approx(40.0)


def test_invalid_rectangle_arguments():
    with pytest.raises(ValueError):
        target_rectangle(frame(), 0, 0, -1, 10)
    with pytest.raises(ValueError):
        target_rectangle(frame(), 0, 0, 10, 10, 0.0, "middle")
    with pytest.raises(ValueError):
        frame("diagonal")
