import numpy as np
import pytest

from lumen.geometry.lines import Line2D
from lumen.vision.datums import fit_cylinder_from_silhouettes, side_of_panel, silhouette_offset


@pytest.mark.parametrize("c_e,c_h,R", [(0, 500, 30), (300, 500, 30), (-80, 600, 30), (450, 700, 60), (200, 300, 12)])
def test_silhouette_formula_matches_brute_force(c_e, c_h, R):
    phi = np.linspace(0, np.pi * 0.95, 200001)
    pe, ph = -R * np.sin(phi), -R * (1 - np.cos(phi))
    e_obs = c_e + (pe - c_e) * c_h / (c_h - ph)
    k = int(np.argmin(e_obs))
    e, p = silhouette_offset(c_e, c_h, R)
    assert e == pytest.approx(e_obs[k], abs=1e-6) and p == pytest.approx(phi[k], abs=1e-4)


def test_silhouette_tends_to_minus_radius_from_far_away():
    e, phi = silhouette_offset(0.0, 1e7, 25.0)
    assert e == pytest.approx(-25.0, abs=1e-3) and np.degrees(phi) == pytest.approx(90.0, abs=0.2)
    assert np.isnan(silhouette_offset(0.0, -5.0, 25.0)[0])


def _observations(e_t, psi_deg, R, cams, sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    psi = np.deg2rad(psi_deg)
    n = np.array([np.cos(psi), np.sin(psi)])  # into the flat region
    t = np.array([-np.sin(psi), np.cos(psi)])
    out = []
    for C in cams:
        c_e = float(n @ C[:2] - e_t)
        e_obs, _ = silhouette_offset(c_e, float(C[2]), R)
        base = (e_t + e_obs) * n
        ends = np.stack([base + 80 * t, base + 520 * t]) + rng.normal(scale=sigma, size=(2, 2))
        out.append((np.asarray(C, dtype=float), ends, max(sigma, 0.05)))
    return out


CAMS = [np.array([x, 300.0, 650.0 + 15 * k]) for k, x in enumerate((-150.0, 0.0, 150.0, 300.0, 450.0, 600.0))]
INTERIOR = np.array([250.0, 300.0])


@pytest.mark.parametrize("R", [8.0, 35.0, 90.0])
def test_cylinder_fit_recovers_tangent_axis_and_radius(R):
    fit = fit_cylinder_from_silhouettes(_observations(12.0, 3.0, R, CAMS), INTERIOR)
    assert fit.ok and fit.radius == pytest.approx(R, rel=1e-3)
    assert float(fit.tangent.n @ np.array([np.cos(np.deg2rad(3)), np.sin(np.deg2rad(3))])) == pytest.approx(1.0, abs=1e-6)
    assert fit.tangent.d == pytest.approx(12.0, abs=1e-2) and fit.rms_mm < 1e-3
    assert float(fit.tangent.signed_distance(INTERIOR)) > 0, "normal points into the flat region"


def test_cylinder_fit_uncertainty_is_honest_under_noise():
    errs_t, errs_r, sig_t, sig_r = [], [], [], []
    for seed in range(40):
        fit = fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, CAMS, sigma=0.15, seed=seed), INTERIOR)
        assert fit.ok
        errs_t.append(fit.tangent.d)
        errs_r.append(fit.radius - 35.0)
        sig_t.append(fit.sigma_tangent_mm)
        sig_r.append(fit.sigma_radius_mm)
    assert np.std(errs_t) == pytest.approx(np.mean(sig_t), rel=0.6)
    assert np.std(errs_r) == pytest.approx(np.mean(sig_r), rel=0.6)


def test_radius_is_not_identifiable_from_similar_views_but_a_configured_radius_helps():
    near = [np.array([300.0 + 4 * k, 300.0, 650.0]) for k in range(4)]
    fit = fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, near, sigma=0.1, seed=1), INTERIOR)
    assert (not fit.ok and fit.reason in ("radius_not_identifiable", "radius_out_of_range", "degenerate")) or fit.sigma_radius_mm > 5.0
    fixed = fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, near, sigma=0.1, seed=1), INTERIOR, radius=35.0)
    assert fixed.ok and fixed.radius_fixed and abs(fixed.tangent.d) < 0.5
    one = fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, CAMS[2:3]), INTERIOR, radius=35.0)
    assert one.ok and abs(one.tangent.d) < 1e-2  # single view suffices with a configured radius
    assert not fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, CAMS[2:3]), INTERIOR).ok


def test_a_uniform_cylinder_gives_direction_but_no_position_along_its_axis():
    a = fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, CAMS), INTERIOR)
    shifted = [np.array([c[0], c[1] + 400.0, c[2]]) for c in CAMS]  # everything moved along the axis
    b = fit_cylinder_from_silhouettes(_observations(0.0, 0.0, 35.0, shifted), INTERIOR)
    assert a.tangent.d == pytest.approx(b.tangent.d, abs=1e-6) and a.radius == pytest.approx(b.radius, rel=1e-6)


def test_side_of_panel_is_a_handedness_not_an_image_direction():
    floor = Line2D.from_points([0, 0], [500, 0])
    interior = np.array([250.0, 300.0])
    assert side_of_panel(floor, Line2D.from_points([0, 0], [0, 500]), interior) == "left"
    assert side_of_panel(floor, Line2D.from_points([600, 0], [600, 500]), interior) == "right"
    # rotate everything by an arbitrary angle: the answer is unchanged
    a = np.deg2rad(137.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = lambda l: Line2D.from_points(R @ l.point, R @ (l.point + 100 * l.direction))  # noqa: E731
    assert side_of_panel(rot(floor), rot(Line2D.from_points([0, 0], [0, 500])), R @ interior) == "left"


# ---------------------------------------------------------------- is a line on the panel face? (height with its margin)
def _look_at(C, target):
    """T_cam_plane of a camera at ``C`` (plane frame, z out of the panel) looking at ``target``."""
    from lumen.geometry.se3 import Pose

    C, target = np.asarray(C, float), np.asarray(target, float)
    zc = (target - C) / np.linalg.norm(target - C)
    xc = np.cross(zc, np.array([0.0, 1.0, 0.0]))
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    return Pose(R, -R @ C)


def _classified(line_pts3d, cams, shift_px=0.0, noise_px=0.03, plane_cov=None, seed=0):
    """Run the height estimate on exact projections of a 3-D line (optionally with every image line shifted)."""
    from lumen.vision.datums import DatumConfig, FrameContext, LineCandidate, _line_heights
    from lumen.vision.edgefit import EdgeFit
    from lumen.vision.multiview import view_plane_of_image_line

    rng = np.random.default_rng(seed)
    K = np.array([[1200.0, 0, 800.0], [0, 1200.0, 600.0], [0, 0, 1]])
    P0, P1 = (np.asarray(p, float) for p in line_pts3d)
    ends2d = np.stack([P0[:2], P1[:2]])
    d2 = (ends2d[1] - ends2d[0]) / np.linalg.norm(ends2d[1] - ends2d[0])
    line = Line2D(np.array([-d2[1], d2[0]]), float(np.array([-d2[1], d2[0]]) @ ends2d[0]))
    planes, shifted, fits = [], [], []
    for C in cams:
        T = _look_at(C, 0.5 * (P0 + P1) * np.array([1, 1, 0]))
        uv = []
        for P in (P0, P1):
            x = K @ T.apply(P)
            uv.append(x[:2] / x[2])
        a, b = np.array(uv[0]), np.array(uv[1])
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        mid = 0.5 * (a + b) + n * (shift_px + rng.normal(0, noise_px if shift_px == 0 else 0.0))
        e = EdgeFit(ok=True, n=n, mid=mid, length=float(np.linalg.norm(b - a)), var_d=noise_px**2, var_theta=(2 * noise_px / np.linalg.norm(b - a)) ** 2)
        h = 0.5 * e.length
        fc = FrameContext(frame_id=f"f{len(fits)}", gray=np.zeros((2, 2), np.uint8), camera=None, H=np.eye(3), pose=T, K=K)
        planes.append(view_plane_of_image_line(T, K, e.mid - h * e.t, e.mid + h * e.t))
        shifted.append(view_plane_of_image_line(T, K, e.mid - h * e.t + e.n, e.mid + h * e.t + e.n))
        fits.append((fc, e))
    cand = LineCandidate(id="c", frame_id="f0", kind="step", fit=fits[0][1], line=line, endpoints=ends2d, length_mm=float(np.linalg.norm(ends2d[1] - ends2d[0])), contrast=0.3, sigma_blur_px=1.0, distinctness=5.0)
    return _line_heights(cand, planes, shifted, fits, DatumConfig())


def test_height_of_a_line_is_found_when_the_photos_see_it_from_different_sides():
    # a vertical seam (along y) seen from the left and from the right: strong parallax across it
    cams = [(x, 300.0, 750.0) for x in (-300.0, -100.0, 150.0, 400.0, 650.0)]
    for true_h in (0.0, 8.0, -5.0):
        h, shift, sig, _ = _classified([(0.0, 80.0, true_h), (0.0, 600.0, true_h)], cams, noise_px=0.0)
        assert h == pytest.approx([true_h, true_h], abs=0.02)
        assert max(shift) < 0.25 and max(sig) < 0.05, "well resolved: a quarter-pixel common shift can fake only a fraction of a millimetre"


def test_a_common_edge_shift_fakes_a_height_when_all_photos_look_from_similar_heights():
    # a horizontal floor line (along x) at the bottom of the panel, seen from five places that differ mostly sideways
    # and only a little in height above the line: this is the geometry of an ordinary sweep
    cams = [(x, 430.0 + dy, 760.0) for x, dy in ((-200.0, 40.0), (50.0, -50.0), (300.0, 10.0), (500.0, 60.0), (750.0, -40.0))]
    line = [(-100.0, 0.0, 0.0), (700.0, 0.0, 0.0)]
    h0, shift, sig, _ = _classified(line, cams, noise_px=0.0)
    assert h0 == pytest.approx([0.0, 0.0], abs=0.02), "exact image lines give the true height"
    # every image line a quarter of a pixel to one side: the same in all photos, as a tone curve or blur does it
    h1, _, _, _ = _classified(line, cams, shift_px=0.25)
    faked = max(abs(a) for a in h1)
    assert faked > 0.15, "the shift looks like a height (otherwise this geometry would not be a problem)"
    assert np.abs(h1) == pytest.approx(shift, abs=0.01), "the stated margin is exactly what such a shift fakes"
    seam = _classified([(0.0, 80.0, 0.0), (0.0, 600.0, 0.0)], [(x, 300.0, 750.0) for x in (-300.0, -100.0, 150.0, 400.0, 650.0)], noise_px=0.0)
    assert max(shift) > 3 * max(seam[1]), "a seam seen from left and right is far less exposed to it"
    # the less the photos differ in height, the less the line fits pin the height down
    flat = [(x, 430.0 + dy, 760.0) for x, dy in ((-200.0, 12.0), (50.0, -15.0), (300.0, 3.0), (500.0, 15.0), (750.0, -10.0))]
    assert max(_classified(line, flat, noise_px=0.0)[2]) > 3 * max(sig)
    # the same photos resolve an 8 mm proud edge without doubt
    h8, shift8, sig8, _ = _classified([(-100.0, 0.0, 8.0), (700.0, 0.0, 8.0)], cams, noise_px=0.0)
    assert h8 == pytest.approx([8.0, 8.0], abs=0.05) and min(abs(a) - (b + 2 * c) for a, b, c in zip(h8, shift8, sig8)) > 1.0


def test_no_height_is_claimed_when_the_photos_cannot_separate_it_from_a_sideways_offset():
    # all cameras on one line through the feature's direction of sight: the view planes are (almost) the same plane
    cams = [(300.0, 430.0, 760.0 + 0.01 * k) for k in range(4)]
    h, shift, sig, _ = _classified([(-100.0, 0.0, 0.0), (700.0, 0.0, 0.0)], cams, noise_px=0.0)
    assert h is None or max(b + 2 * c for b, c in zip(shift, sig)) > 50.0
