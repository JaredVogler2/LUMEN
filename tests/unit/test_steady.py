"""Has the picture around a tag moved?  (The check behind a held registration of a phone that stands still.)"""

import cv2
import numpy as np
import pytest

from lumen.vision.steady import make_patch, patch_shift

CORNERS = np.array([[120.0, 90.0], [200.0, 92.0], [198.0, 172.0], [118.0, 170.0]])


def tag_picture(seed=3, size=(260, 320)):
    """A dark block pattern on a grey panel: what the neighbourhood of a tag looks like, with some noise."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 150.0, np.float32)
    cells = rng.integers(0, 2, size=(8, 8))
    for r in range(8):
        for c in range(8):
            if r in (0, 7) or c in (0, 7) or cells[r, c]:
                img[90 + 10 * r : 100 + 10 * r, 120 + 10 * c : 130 + 10 * c] = 35.0
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    return img


def noisy(img, seed, sigma=1.5):
    return np.clip(img + np.random.default_rng(seed).normal(0.0, sigma, img.shape), 0, 255).astype(np.uint8)


def shifted(img, dx, dy):
    # spline interpolation in floating point (OpenCV's warps place a picture only to 1/32 px)
    from scipy import ndimage

    return ndimage.shift(img, (dy, dx), order=3, mode="nearest")


@pytest.fixture(scope="module")
def ref():
    base = tag_picture()
    p = make_patch(noisy(base, 1), 7, CORNERS)
    assert p is not None and p.tag_id == 7
    return base, p


@pytest.mark.parametrize("dx,dy", [(0.0, 0.0), (0.05, 0.0), (-0.08, 0.06), (0.2, -0.15), (0.45, 0.3)])
def test_small_shifts_are_measured_to_a_few_hundredths_of_a_pixel(ref, dx, dy):
    base, p = ref
    gx, gy, r2 = patch_shift(p, noisy(shifted(base, dx, dy), 2))
    assert gx == pytest.approx(dx, abs=0.03) and gy == pytest.approx(dy, abs=0.03), (gx, gy)
    assert r2 > 0.99


def test_light_is_not_movement(ref):
    base, p = ref
    for gain, offset in ((1.25, 0.0), (0.7, 0.0), (0.85, 20.0)):
        gx, gy, r2 = patch_shift(p, noisy(np.clip(base * gain + offset, 0, 255), 4))
        assert abs(gx) < 0.02 and abs(gy) < 0.02 and r2 > 0.99, (gain, offset, gx, gy, r2)


def test_something_over_the_tag_or_another_picture_is_not_the_same_picture(ref):
    base, p = ref
    # a hand over two thirds of it
    hand = base.copy()
    hand[80:200, 150:240] = 110.0
    assert patch_shift(p, noisy(hand, 5))[2] < 0.6
    # another tag in its place
    assert patch_shift(p, noisy(tag_picture(seed=11), 6))[2] < 0.8
    # nothing there at all
    assert patch_shift(p, noisy(np.full_like(base, 150.0), 7))[2] < 0.2
    # a tag that was moved by half its cell: either unexplained, or far beyond any tolerance
    gx, gy, r2 = patch_shift(p, noisy(shifted(base, 5.0, 0.0), 8))
    assert r2 < 0.9 or abs(gx) > 1.0
    # a frame of another size
    assert patch_shift(p, np.zeros((100, 100), np.uint8))[2] == 0.0


def test_where_no_patch_can_be_kept():
    base = noisy(tag_picture(), 1)
    # at the border of the picture there is no room around the tag
    assert make_patch(base, 1, CORNERS - [118.0, 0.0]) is None
    assert make_patch(base, 1, CORNERS + [0.0, 85.0]) is None
    # a plain surface, or stripes along one direction, cannot show a shift (in every direction)
    plain = np.full((260, 320), 150, np.uint8)
    assert make_patch(plain, 1, CORNERS) is None
    stripes = np.tile((np.arange(320) // 10 % 2 * 120 + 60).astype(np.uint8), (260, 1))
    assert make_patch(stripes, 1, CORNERS) is None


def test_no_larger_shift_passes_for_standing_still(ref):
    """The estimate is only right for small shifts.  What matters for a large one is that it never looks like
    a small one that is well explained: that is what would keep a registration that no longer holds."""
    base, p = ref
    for mag in (0.6, 0.8, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0):
        for ang in np.deg2rad(np.arange(0, 360, 30)):
            dx, dy = mag * np.cos(ang), mag * np.sin(ang)
            gx, gy, r2 = patch_shift(p, noisy(shifted(base, dx, dy), 9))
            still = r2 >= 0.97 and np.isfinite(gx) and np.hypot(gx, gy) <= 0.5
            assert not still, (mag, float(np.degrees(ang)), gx, gy, r2)
