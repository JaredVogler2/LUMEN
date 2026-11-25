import cv2
import numpy as np
import pytest

from lumen.geometry.camera import CameraModel
from lumen.geometry.homography import apply_h, dlt_homography, fit_rigid_2d, h_local_jacobian, homography_from_pose, local_scale_px_per_mm, pose_from_homography
from lumen.geometry.lines import Line2D, fit_line_tls, point_in_polygon, polygon_area
from lumen.geometry.optimize import lm_solve, numeric_jacobian
from lumen.geometry.se3 import Pose, exp_so3, log_so3, look_at, orthonormalize, rotation_angle

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ SE(3)
@pytest.mark.parametrize("w", [np.zeros(3), np.array([1e-10, 0, 0]), np.array([0.3, -0.2, 0.9]), np.array([np.pi - 1e-7, 0, 0]), np.array([0, np.pi, 0]), np.array([2.0, 2.0, -1.0])])
def test_exp_log_round_trip(w):
    R = exp_so3(w)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-9)


def test_pose_algebra_and_naming():
    T_a_b = Pose.from_rvec([0.2, -0.1, 0.4], [10, 20, 30])
    T_b_c = Pose.from_rvec([-0.3, 0.5, 0.1], [-5, 2, 8])
    T_a_c = T_a_b @ T_b_c
    x_c = RNG.normal(size=(5, 3))
    assert np.allclose(T_a_c.apply(x_c), T_a_b.apply(T_b_c.apply(x_c)))
    assert np.allclose((T_a_b @ T_a_b.inv()).matrix(), np.eye(4), atol=1e-12)
    # centre of frame a expressed in frame b
    assert np.allclose(T_a_b.apply(T_a_b.center), 0.0, atol=1e-12)


def test_look_at_points_the_optical_axis_at_the_target():
    T_wc = look_at([400, -700, 400], [300, 0, 300], [0, 0, 1])
    T_cw = T_wc.inv()
    p = T_cw.apply(np.array([300.0, 0.0, 300.0]))
    assert p[2] > 0 and abs(p[0]) < 1e-9 and abs(p[1]) < 1e-9
    # world "up" appears up in the image (negative camera y)
    up = T_cw.rotate(np.array([0.0, 0.0, 1.0]))
    assert up[1] < 0
    assert np.linalg.det(T_wc.R) == pytest.approx(1.0)


def test_retract_is_consistent_with_exp():
    T = Pose.from_rvec([0.1, 0.2, 0.3], [1, 2, 3])
    xi = np.array([0.01, -0.02, 0.005, 0.3, -0.1, 0.2])
    T2 = T.retract(xi)
    assert np.allclose(T2.R, exp_so3(xi[:3]) @ T.R)
    assert rotation_angle(T.R.T @ T2.R) == pytest.approx(np.linalg.norm(xi[:3]), rel=1e-6)
    assert np.allclose(orthonormalize(T2.R * 1.0001), T2.R, atol=1e-3)


# ----------------------------------------------------------------- camera
def _cam(dist=(0.09, -0.18, 0.0007, -0.0004, 0.05)):
    return CameraModel(1600, 1200, 1210.0, 1205.0, 801.3, 597.2, dist)


def test_projection_matches_opencv():
    cam = _cam()
    X = np.stack([RNG.uniform(-400, 400, 200), RNG.uniform(-300, 300, 200), RNG.uniform(500, 1200, 200)], axis=1)
    uv, front = cam.project(X)
    ref, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), cam.K, cam.dist_array)
    assert front.all()
    assert np.max(np.abs(uv - ref.reshape(-1, 2))) < 1e-8


def test_undistort_inverts_distort_and_matches_opencv():
    cam = _cam()
    g = np.stack(np.meshgrid(np.linspace(0, 1599, 15), np.linspace(0, 1199, 11)), axis=-1).reshape(-1, 2)
    ideal = cam.undistort_pixels(g)
    assert np.max(np.abs(cam.distort_pixels(ideal) - g)) < 1e-8
    ref = cv2.undistortPointsIter(
        g.reshape(-1, 1, 2), cam.K, cam.dist_array, None, cam.K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 1e-13)
    ).reshape(-1, 2)
    assert np.max(np.abs(ideal - ref)) < 1e-5


def test_scaled_model_follows_pixel_centre_convention():
    cam = _cam()
    half = cam.scaled(0.5)
    X = np.array([[100.0, -50.0, 800.0], [-300.0, 200.0, 900.0]])
    uv = cam.project(X)[0]
    uv_half = half.project(X)[0]
    # continuous coordinate (u + 0.5) scales; pixel centres do not simply halve
    assert np.allclose(uv_half + 0.5, 0.5 * (uv + 0.5), atol=1e-9)
    assert (half.width, half.height) == (800, 600)


def test_no_distortion_is_identity():
    cam = CameraModel.from_fov(1280, 960, 67.0)
    pts = RNG.uniform(0, 900, size=(10, 2))
    assert np.array_equal(cam.undistort_pixels(pts), pts)
    assert cam.max_distortion_px() == 0.0
    assert cam.hfov == pytest.approx(np.deg2rad(67.0))


# ------------------------------------------------------------- homography
def test_dlt_recovers_exact_homography():
    H = np.array([[1.9, 0.12, 400.0], [-0.07, 2.1, 300.0], [1.1e-4, -0.8e-4, 1.0]])
    src = RNG.uniform(-200, 200, size=(12, 2))
    dst = apply_h(H, src)
    He = dlt_homography(src, dst)
    assert np.max(np.abs(apply_h(He, src) - dst)) < 1e-8
    assert np.allclose(He / He[2, 2], H, rtol=1e-8, atol=1e-8)


def test_homography_from_four_points_is_exact():
    src = np.array([[-25.0, 25.0], [25.0, 25.0], [25.0, -25.0], [-25.0, -25.0]])
    dst = np.array([[410.2, 300.5], [495.9, 310.1], [490.3, 398.8], [405.7, 391.0]])
    H = dlt_homography(src, dst)
    assert np.max(np.abs(apply_h(H, src) - dst)) < 1e-9


def test_local_jacobian_matches_finite_differences():
    H = np.array([[1.9, 0.12, 400.0], [-0.07, 2.1, 300.0], [1.1e-4, -0.8e-4, 1.0]])
    p = np.array([[30.0, -80.0]])
    J = h_local_jacobian(H, p)[0]
    Jn = numeric_jacobian(lambda q: apply_h(H, q[None])[0], p[0])
    assert np.allclose(J, Jn, atol=1e-6)
    assert local_scale_px_per_mm(H, p)[0] == pytest.approx(np.sqrt(abs(np.linalg.det(Jn))), rel=1e-6)


def test_pose_homography_round_trip():
    K = _cam((0, 0, 0, 0, 0)).K
    T = look_at([150, -60, -700], [0, 0, 0], [0, 1, 0]).inv()  # T_cam_plane, plane = z=0
    H = homography_from_pose(K, T)
    T2 = pose_from_homography(K, H)
    ang, dist = T.distance_to(T2)
    assert ang < 1e-9 and dist < 1e-6


def test_fit_rigid_2d_with_and_without_scale():
    src = np.array([[-25.0, 25.0], [25.0, 25.0], [25.0, -25.0], [-25.0, -25.0]])
    th, t = 0.7, np.array([120.0, -45.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = 1.02 * (src @ R.T) + t
    a, tt, s, rms = fit_rigid_2d(src, dst, allow_scale=True)
    assert a == pytest.approx(th) and s == pytest.approx(1.02) and rms < 1e-9 and np.allclose(tt, t)
    a, tt, s, rms = fit_rigid_2d(src, dst, allow_scale=False)
    assert s == 1.0 and rms == pytest.approx(0.02 * 25 * np.sqrt(2), rel=1e-6)


# ------------------------------------------------------------------ lines
def test_line_basics():
    l = Line2D.from_points([0, 0], [10, 0])
    assert abs(float(l.signed_distance(np.array([3.0, 5.0])))) == pytest.approx(5.0)
    up = l.oriented_towards(np.array([0.0, 4.0]))
    assert float(up.signed_distance(np.array([0.0, 4.0]))) == pytest.approx(4.0)
    assert np.allclose(up.offset(2.0).project(np.array([7.0, 9.0])), [7.0, 2.0])
    v = Line2D.from_points([3, -5], [3, 5])
    assert np.allclose(l.intersect(v), [3.0, 0.0])
    assert l.intersect(Line2D.from_points([0, 1], [10, 1])) is None
    assert l.angle_to(v) == pytest.approx(np.pi / 2)


def test_line_transform_by_homography():
    H = np.array([[1.9, 0.12, 400.0], [-0.07, 2.1, 300.0], [1.1e-4, -0.8e-4, 1.0]])
    l = Line2D.from_points([-100, 20], [150, -60])
    lt = l.transformed(H)
    pts = apply_h(H, np.array([[-100.0, 20.0], [150.0, -60.0], [25.0, -20.0]]))
    assert np.max(np.abs(lt.signed_distance(pts))) < 1e-8


def test_line_clip_to_box():
    seg = Line2D.from_points([0, 0], [1, 1]).clip_to_box(-2, -1, 3, 5)
    assert seg is not None and np.allclose(sorted(map(tuple, seg)), [(-1, -1), (3, 3)])
    assert Line2D.from_points([10, 0], [10, 1]).clip_to_box(0, 0, 5, 5) is None


def test_line_fit_uncertainty_matches_monte_carlo():
    s = np.linspace(-80, 80, 60)
    truth = Line2D.from_points([0, 0], [np.cos(0.3), np.sin(0.3)])
    base = np.outer(s, truth.direction)
    rng = np.random.default_rng(3)
    offs, angs = [], []
    for _ in range(600):
        pts = base + rng.normal(scale=0.2, size=base.shape)
        f = fit_line_tls(pts)
        offs.append(float(f.line.signed_distance(np.zeros(2))))
        d = f.line.direction
        d = d if d @ truth.direction > 0 else -d
        angs.append(np.arctan2(d[1], d[0]) - 0.3)
    f = fit_line_tls(base + rng.normal(scale=0.2, size=base.shape))
    assert np.std(offs) == pytest.approx(f.sigma_perp, rel=0.25)
    assert np.std(angs) == pytest.approx(f.sigma_angle, rel=0.25)
    assert f.sigma_at(np.array([[0.0, 0.0]]))[0] < f.sigma_at(base[-1:] * 3)[0]


def test_polygon_helpers():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert polygon_area(sq) == pytest.approx(4.0)
    assert polygon_area(sq[::-1]) == pytest.approx(-4.0)
    assert point_in_polygon(np.array([1.0, 1.0]), sq) and not point_in_polygon(np.array([3.0, 1.0]), sq)


# -------------------------------------------------------------- optimiser
def test_lm_solves_a_curve_fit_and_reports_covariance():
    x = np.linspace(0, 4, 80)
    rng = np.random.default_rng(1)
    y = 2.5 * np.exp(-0.7 * x) + rng.normal(scale=0.01, size=x.size)

    def fn(p):
        m = p[0] * np.exp(-p[1] * x)
        J = np.stack([np.exp(-p[1] * x), -p[0] * x * np.exp(-p[1] * x)], axis=1)
        return m - y, J

    sol = lm_solve(fn, np.array([1.0, 0.2]))
    assert sol.converged and sol.x == pytest.approx([2.5, 0.7], abs=0.02)
    assert sol.sigma == pytest.approx(0.01, rel=0.3)
    assert np.all(np.sqrt(np.diag(sol.covariance())) < 0.01)


def test_lm_robust_rounds_reject_outliers():
    x = np.linspace(0, 1, 50)
    y = 3.0 * x + 1.0
    y[5] += 40.0
    y[30] -= 25.0

    def fn(p):
        return p[0] * x + p[1] - y, np.stack([x, np.ones_like(x)], axis=1)

    plain = lm_solve(fn, np.zeros(2))
    huber = lm_solve(fn, np.zeros(2), robust_k=2.5)
    tukey = lm_solve(fn, np.zeros(2), robust_k=2.5, robust_rounds=4, loss="tukey")
    # Huber bounds the damage, Tukey removes it
    assert abs(tukey.x[0] - 3.0) < 1e-6 < abs(huber.x[0] - 3.0) < abs(plain.x[0] - 3.0)
