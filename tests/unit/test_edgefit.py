"""Edge and line fits on pictures whose geometry is known exactly (prompts 03, 06).

One feature, three appearances: a seam is an unresolved thin line in a small photo, two edges a few pixels
apart in a full-resolution one, and a datum either way.  Its centre line is what has to come out.
"""

import numpy as np
import pytest
from scipy.special import ndtr

from lumen.vision.edgefit import fit_trench, snap_line


def picture(profile, angle_deg=7.0, size=(260, 420), noise=0.0, seed=0):
    """8-bit sRGB picture of a straight feature through the middle; ``profile(s)`` is the linear level at distance s (px)."""
    H, W = size
    a = np.deg2rad(angle_deg)
    t = np.array([np.cos(a), np.sin(a)])
    n = np.array([-t[1], t[0]])
    c = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    lin = profile((us - c[0]) * n[0] + (vs - c[1]) * n[1])
    if noise:
        lin = lin + np.random.default_rng(seed).normal(0.0, noise, lin.shape)
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8), c, t, n


def trench(h, sigma=0.9, base=0.6, inside=0.2):
    return lambda s: base + (inside - base) * (ndtr((s + h) / sigma) - ndtr((s - h) / sigma))


def seed_line(c, t, n, offset, tilt_deg=0.0, half=170.0):
    a = np.deg2rad(tilt_deg)
    t2 = np.array([np.cos(a) * t[0] - np.sin(a) * t[1], np.sin(a) * t[0] + np.cos(a) * t[1]])
    return c + offset * n - half * t2, c + offset * n + half * t2


def off_centre(fit, c, n):
    return float((fit.mid - c) @ n)


def test_a_resolved_line_is_fitted_as_one_feature_with_its_centre_and_width():
    gray, c, t, n = picture(trench(3.5), noise=0.004)
    p0, p1 = seed_line(c, t, n, offset=0.8, tilt_deg=0.3)
    f = fit_trench(gray, p0, p1, half_width=9.0, h0=3.0)
    assert f.ok, f.reason
    assert abs(off_centre(f, c, n)) < 0.03 and f.bar_half_px == pytest.approx(3.5, abs=0.08)
    assert abs(float(f.t @ n)) < np.sin(np.deg2rad(0.03)), "the tilt of the seed is taken out"
    assert f.sigma_blur == pytest.approx(np.sqrt(0.9**2 + 3.5**2 / 3), abs=0.15) and f.contrast == pytest.approx(0.4, abs=0.02)
    assert 0 < f.var_d < 0.02**2
    # a line the photo does not resolve is not this model's business
    thin, c2, t2, n2 = picture(trench(0.6))
    assert not fit_trench(thin, *seed_line(c2, t2, n2, 0.0), half_width=6.0, h0=1.0).ok


@pytest.mark.parametrize("flank", [+1, -1])
def test_a_snap_that_lands_on_one_edge_of_a_seam_returns_the_centre_of_the_seam(flank):
    """The fault this was written for: at 4032x3024 a 2 mm seam is seven pixels wide, the snap took one of its
    two edges, and every target was 1.0 mm off, outside its stated uncertainty."""
    gray, c, t, n = picture(trench(3.5), noise=0.004)
    p0, p1 = seed_line(c, t, n, offset=flank * 3.5 + 0.7, tilt_deg=-0.4)
    r = snap_line(gray, p0, p1, search_px=6.0, max_tilt_deg=2.0, max_bar_px=20.0)
    assert r.ok and r.kind == "bar" and r.fit.bar_half_px == pytest.approx(3.5, abs=0.1)
    assert abs(off_centre(r.fit, c, n)) < 0.04
    # the single edge is still there for whoever asks for it
    assert r.other_kind == "step" and abs(abs(off_centre(r.other, c, n)) - 3.5) < 0.15
    # without the width limit the snap behaves as it used to: one edge, half the width away from the centre
    old = snap_line(gray, p0, p1, search_px=6.0, max_tilt_deg=2.0)
    assert old.ok and old.kind == "step" and abs(abs(off_centre(old.fit, c, n)) - 3.5) < 0.15


def test_a_thin_line_keeps_its_own_model_and_is_found_at_its_centre():
    gray, c, t, n = picture(trench(1.0, sigma=1.0), noise=0.004)
    r = snap_line(gray, *seed_line(c, t, n, offset=1.5), search_px=6.0, max_tilt_deg=2.0, max_bar_px=20.0)
    assert r.ok and r.kind == "bar" and r.fit.bar_half_px == 0.0 and abs(off_centre(r.fit, c, n)) < 0.05


def test_what_is_not_a_seam_stays_an_edge():
    # an edge between two surfaces
    step, c, t, n = picture(lambda s: 0.2 + 0.4 * ndtr(s / 0.9), noise=0.004)
    r = snap_line(step, *seed_line(c, t, n, offset=1.0), search_px=6.0, max_tilt_deg=2.0, max_bar_px=20.0)
    assert r.ok and r.kind == "step" and abs(off_centre(r.fit, c, n)) < 0.05
    # a dark strip with different levels on its two sides is two edges of two things, not one line
    uneven = lambda s: 0.6 + (0.2 - 0.6) * ndtr((s + 3.5) / 0.9) + (0.4 - 0.2) * ndtr((s - 3.5) / 0.9)  # noqa: E731
    gray, c, t, n = picture(uneven, noise=0.004)
    r = snap_line(gray, *seed_line(c, t, n, offset=3.5 + 0.5), search_px=4.0, max_tilt_deg=2.0, max_bar_px=20.0)
    assert r.ok and r.kind == "step" and abs(off_centre(r.fit, c, n) - 3.5) < 0.2
    # a strip wider than the limit is two features
    wide, c, t, n = picture(trench(9.0), noise=0.004)
    r = snap_line(wide, *seed_line(c, t, n, offset=9.0 + 0.5), search_px=4.0, max_tilt_deg=2.0, max_bar_px=12.0)
    assert r.ok and r.kind == "step" and abs(off_centre(r.fit, c, n) - 9.0) < 0.2


@pytest.mark.parametrize("sigma", [0.7, 1.0, 1.4])
def test_the_centre_is_found_for_every_width_from_a_hairline_to_the_limit(sigma):
    """Lines 1 to 19 px wide, snapped from either edge: one answer, the centre, whichever model does the work.

    Narrow lines go to the thin-line model, lines of about 5.5 px and more to the model with two edges; in
    between (4 to 5 px wide, sharp photos) the thin-line model is up to 0.08 px off.
    """
    used = set()
    for h in (0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 8.0, 9.5):
        for flank in (+1, -1):
            for extra in (0.6, -0.8):
                gray, c, t, n = picture(trench(h, sigma=sigma), noise=0.004, seed=int(h * 10) + (flank > 0))
                r = snap_line(gray, *seed_line(c, t, n, offset=flank * h + extra, tilt_deg=0.3 * flank), search_px=6.0, max_tilt_deg=2.0, max_bar_px=20.0)
                assert r.ok and r.kind == "bar", (h, flank, extra, r.reason, r.kind)
                assert abs(off_centre(r.fit, c, n)) < 0.1, (h, flank, extra, off_centre(r.fit, c, n))
                if r.fit.bar_half_px > 0:
                    assert r.fit.bar_half_px == pytest.approx(h, abs=0.25)
                used.add((h, r.fit.bar_half_px > 0))
    assert (0.5, False) in used and (1.2, False) in used, "a hairline stays with the thin-line model"
    assert all((h, True) in used for h in (3.5, 5.0, 8.0, 9.5)), "a line the photo resolves is fitted with its two edges"
