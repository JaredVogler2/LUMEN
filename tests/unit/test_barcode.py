"""Code reading for identification (prompt 08): zxing-cpp with an OpenCV QR fallback."""

import cv2
import numpy as np
import pytest

from lumen.vision.barcode import decoders_available, read_codes

zxingcpp = pytest.importorskip("zxingcpp")


def symbol(text: str, fmt: str, scale: int = 6) -> np.ndarray:
    bc = zxingcpp.create_barcode(text, getattr(zxingcpp.BarcodeFormat, fmt))
    img = np.array(zxingcpp.write_barcode_to_image(bc, scale=scale))
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def photo_of(sym: np.ndarray, size=(900, 1200), angle=14.0, tilt=0.22, blur=1.2, noise=4.0, seed=0, at=(0.5, 0.5)) -> np.ndarray:
    """A label photographed at an angle on a mid-grey surface, with blur and noise."""
    rng = np.random.default_rng(seed)
    H, W = size
    canvas = np.full((H, W), 150, np.uint8)
    h, w = sym.shape
    pad = 24
    lab = np.full((h + 2 * pad, w + 2 * pad), 245, np.uint8)
    lab[pad : pad + h, pad : pad + w] = sym
    hh, ww = lab.shape
    src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
    cx, cy = at[0] * W, at[1] * H
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    base = (src - [ww / 2, hh / 2]) @ R.T
    base[:, 0] *= 1.0 + tilt * (base[:, 1] / hh)  # keystone
    dst = np.float32(base + [cx, cy])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(lab, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=150)
    mask = cv2.warpPerspective(np.full_like(lab, 255), M, (W, H))
    canvas = np.where(mask > 0, warped, canvas).astype(np.float32)
    canvas = cv2.GaussianBlur(canvas, (0, 0), blur) + rng.normal(0, noise, canvas.shape)
    return cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("fmt,text", [("QRCode", "DEMO-FLAT-01"), ("Code128", "LUMEN-DEMO-PNL-FLAT"), ("DataMatrix", "PLC-DEMO-101"), ("Code39", "PNL-DEMO-CURVE")])
def test_codes_are_read_from_an_angled_noisy_photo(fmt, text):
    # label filling a good part of the frame, phone tilted 14 degrees and not square to it
    two_d = fmt in ("QRCode", "DataMatrix")
    img = photo_of(symbol(text, fmt, scale=8 if two_d else 3), size=(1000, 1400), angle=14.0 if fmt != "Code39" else 4.0)
    got = read_codes(img)
    assert [c.text for c in got] == [text], [c.to_dict() for c in got]
    c = got[0]
    assert c.decoder == "zxing" and len(c.corners) == 4
    # the reported outline is where the label is (centre of the picture)
    cx, cy = np.mean(c.corners, axis=0)
    assert abs(cx - 700) < 70 and abs(cy - 500) < 70


def test_two_labels_in_one_picture_are_both_reported_once():
    a = photo_of(symbol("LUMEN-DEMO-PLC-101", "QRCode", 7), size=(900, 1500), at=(0.27, 0.5), angle=-8, seed=1)
    b = photo_of(symbol("LUMEN-DEMO-PLC-102", "QRCode", 7), size=(900, 1500), at=(0.73, 0.5), angle=10, seed=2)
    img = np.where(np.abs(a.astype(int) - 150) > np.abs(b.astype(int) - 150), a, b).astype(np.uint8)
    assert [c.text for c in read_codes(img)] == ["LUMEN-DEMO-PLC-101", "LUMEN-DEMO-PLC-102"]


def test_nothing_readable_gives_an_empty_list_and_apriltags_are_not_codes():
    rng = np.random.default_rng(5)
    noise = np.clip(rng.normal(140, 25, (600, 800, 3)), 0, 255).astype(np.uint8)
    assert read_codes(noise) == []
    from lumen.synth.tags import tag_image

    tag = tag_image(7, cell_px=14, quiet_cells=2.0)
    canvas = np.full((480, 640), 160, np.uint8)
    canvas[100 : 100 + tag.shape[0], 200 : 200 + tag.shape[1]] = tag
    assert read_codes(canvas) == []


def test_control_characters_and_overlong_content_are_dropped():
    assert read_codes(photo_of(symbol("line1\nline2", "QRCode", 8)))[0].text == "line1 line2"  # whitespace is normalised
    long = symbol("X" * 400, "QRCode", 3)
    assert read_codes(photo_of(long, size=(1400, 1600), angle=0, tilt=0, blur=0.6, noise=1.0)) == []


def test_opencv_fallback_reads_qr_without_zxing():
    assert decoders_available()["opencv_qr"]
    img = photo_of(symbol("DEMO-CURVE-01", "QRCode", 8), angle=5, tilt=0.05, blur=0.8, noise=2.0)
    got = read_codes(img, use_zxing=False)
    assert [c.text for c in got] == ["DEMO-CURVE-01"] and got[0].decoder == "opencv"


def test_a_long_linear_code_is_read_at_any_tilt_of_the_phone():
    """A row scan cannot cross every bar of a tilted 1-D code; rotation attempts make it readable."""
    sym = symbol("LUMEN-DEMO-PNL-CURVE", "Code128", 3)  # 3 px per narrowest bar
    for ang in (0, 14, -31, 47, 90, 118):
        img = photo_of(sym, size=(1000, 1400), angle=float(ang), tilt=0.1, seed=ang % 7)
        got = read_codes(img)
        assert [c.text for c in got] == ["LUMEN-DEMO-PNL-CURVE"], ang
        cx, cy = np.mean(got[0].corners, axis=0)
        assert abs(cx - 700) < 90 and abs(cy - 500) < 90, (ang, cx, cy)  # outline mapped back to the original picture
    # the quick mode used by a live scanning loop does not search rotations
    tilted = photo_of(sym, size=(1000, 1400), angle=31.0, tilt=0.1)
    assert read_codes(tilted, thorough=False) == [] and len(read_codes(tilted, thorough=True)) == 1
