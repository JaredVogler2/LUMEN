"""Lens profile from board photos through the Settings screen (prompt 03), simulated board photos."""

import cv2
import pytest

from lumen.synth.boards import PrintedBoard, phone_base, render_board, view_set
from tests.browser.conftest import assert_clean_text, shot, to_capture_step

pytestmark = pytest.mark.browser

W, H = 1600, 1200
BASE = phone_base(W, H, pp_offset=(7.3, -5.1), aspect=1.001)
CHU = PrintedBoard(kind="charuco", squares_x=9, squares_y=6, square_mm=30.0, marker_mm=21.0, dictionary="DICT_5X5_100")


@pytest.fixture(scope="module")
def board_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("boards")
    out = []
    for k, cam in enumerate(view_set(CHU, BASE, 11, seed=3)):
        p = d / f"board_{k:02d}.jpg"
        cv2.imwrite(str(p), render_board(CHU, cam, seed=40 + k).image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        out.append(str(p))
    return out


def open_settings(page):
    page.click('[data-action="menu"]')
    page.click("text=Settings & camera calibration")
    page.wait_for_selector('[data-region="settings"]')


def test_profile_is_computed_saved_on_the_phone_and_attached_to_a_session(page, board_files):
    to_capture_step(page)
    open_settings(page)
    assert "None saved" in page.inner_text('[data-region="settings"]')
    assert "not told the lens parameters" in page.inner_text('[data-region="cal-board"]')
    href = page.get_attribute('[data-action="cal-download"]', "href")
    assert "api/calibration/board.png" in href and "kind=charuco" in href
    # the measured sizes are required: nothing is assumed from the print request
    page.click('[data-action="cal-start"]')
    page.wait_for_selector(".toast")
    assert "measured square size" in page.inner_text(".toasts").lower()
    page.click('.toast [aria-label="Dismiss"]')
    page.wait_for_selector(".toast", state="detached")
    page.fill('[data-field="square_mm"]', "30")
    page.fill('[data-field="marker_mm"]', "21")
    page.click('[data-action="cal-start"]')
    page.wait_for_selector('[data-region="cal-photos"]')
    # three photos are not enough: refused with advice
    page.set_input_files('[data-input="cal-photos"]', board_files[:3])
    page.wait_for_function("() => document.querySelectorAll('[data-view]').length === 3 && !document.querySelector('.busy')", timeout=120_000)
    page.click('[data-action="cal-solve"]')
    page.wait_for_selector('[data-region="cal-photos"] .banner.danger', timeout=120_000)
    assert "too few usable photos" in page.inner_text('[data-region="cal-photos"] .banner.danger').lower()
    page.set_input_files('[data-input="cal-photos"]', board_files[3:])
    page.wait_for_function("() => document.querySelectorAll('[data-view]').length === 11 && !document.querySelector('.busy')", timeout=240_000)
    # the coverage map is really drawn (the board has been somewhere in the picture)
    assert page.evaluate("""() => { const c = document.querySelector('[data-region="cal-coverage"]'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n += 1; return n / (d.length / 4); }""") > 0.3
    shot(page, "settings_calibration_photos")
    page.click('[data-action="cal-solve"]')
    page.wait_for_selector('[data-region="cal-photos"] .banner.ok', timeout=180_000)
    text = page.inner_text('[data-region="cal-photos"] .banner.ok')
    assert "11 photos" in text and "RMS" in text and "focal length" in text
    shot(page, "settings_calibration_result")
    page.click('[data-action="cal-save"]')
    page.wait_for_selector("[data-profile]")
    assert_clean_text(page)
    prof = page.evaluate("() => JSON.parse(localStorage.getItem('lumen.calibrations'))")
    assert len(prof) == 1 and prof[0]["method"] == "charuco" and prof[0]["provenance"] == "measured"
    assert abs(prof[0]["fx"] - BASE.fx) < 1.0 and abs(prof[0]["cx"] - BASE.cx) < 1.0 and (prof[0]["width"], prof[0]["height"]) == (W, H)
    # attach to the open session, then detach
    page.click('[data-action="profile-use"]')
    page.wait_for_selector('[data-action="profile-detach"]', timeout=60_000)
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    assert sess["calibration"]["id"] == prof[0]["id"]
    page.click('[data-action="profile-detach"]')
    page.wait_for_selector('[data-action="profile-use"]', timeout=60_000)
    assert page.evaluate("async () => (await import('./static/js/store.js')).state.session.calibration") is None
    # the profile survives a reload of the page (it lives in this browser)
    page.reload()
    page.wait_for_selector('[data-step="identify"]')
    assert len(page.evaluate("() => JSON.parse(localStorage.getItem('lumen.calibrations'))")) == 1
    assert not page._lumen_errors
