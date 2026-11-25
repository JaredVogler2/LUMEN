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
