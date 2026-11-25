"""Scanning a code with the (fake) camera to choose the operation (prompt 08)."""

import numpy as np
import pytest

from tests.browser.conftest import launch_with_camera, open_app, shot, write_y4m

pytestmark = pytest.mark.browser
zxingcpp = pytest.importorskip("zxingcpp")


def label_frame(texts, size=(540, 960)):
    from tests.unit.test_barcode import photo_of, symbol

    H, W = size
    if len(texts) == 1:
        return photo_of(symbol(texts[0], "QRCode", 7), size=size, angle=6.0, tilt=0.1, blur=0.8, noise=2.0)
    a = photo_of(symbol(texts[0], "QRCode", 6), size=size, at=(0.27, 0.5), angle=-5.0, tilt=0.05, blur=0.8, noise=2.0, seed=1)
    b = photo_of(symbol(texts[1], "QRCode", 6), size=size, at=(0.73, 0.5), angle=5.0, tilt=0.05, blur=0.8, noise=2.0, seed=2)
    out = a.copy()
    out[:, W // 2 :] = b[:, W // 2 :]
    return out


def cam_browser(playwright_instance, tmp_path_factory, texts):
    y4m = write_y4m(str(tmp_path_factory.mktemp("scan") / "label.y4m"), [label_frame(texts)] * 4, fps=10)
    return launch_with_camera(playwright_instance, y4m)


def test_scanning_a_placard_code_selects_exactly_that_record(playwright_instance, tmp_path_factory, live_server):
    b = cam_browser(playwright_instance, tmp_path_factory, ["LUMEN-DEMO-PLC-102"])
    try:
        page = open_app(b, live_server.url)
        page.click('[data-action="scan"]')
        page.wait_for_selector(".cam")
        # the scanner closes by itself once one code is read, and the exact match is selected
        page.wait_for_selector(".cam", state="detached", timeout=60_000)
        page.wait_for_selector('[data-action="continue"]:not([disabled])', timeout=30_000)
        assert page.evaluate("() => window.__lumenScanner") is None  # camera released
        # the record is shown for the operator to check before it is used
        assert page.get_attribute('[data-placard="PLC-DEMO-102"]', "aria-checked") == "true"
        assert "419.1" in page.inner_text("#view")
        page.click('[data-action="continue"]')
        page.wait_for_selector('[data-action="add-tag"]')
        sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
        assert sess["operation"]["summary"]["operation_id"] == "DEMO-FLAT-01" and sess["active_placard_id"] == "PLC-DEMO-102"
        assert not page._lumen_errors
        page.context.close()
    finally:
        b.close()


def test_two_codes_in_view_are_offered_for_choice_and_unknown_codes_select_nothing(playwright_instance, tmp_path_factory, live_server):
    b = cam_browser(playwright_instance, tmp_path_factory, ["LUMEN-DEMO-PNL-CURVE", "NOT-IN-THE-LIBRARY-7"])
    try:
        page = open_app(b, live_server.url)
        page.click('[data-action="scan"]')
        page.wait_for_selector('[data-region="scan-choices"]', timeout=60_000)
        text = page.inner_text('[data-region="scan-choices"]')
        assert "LUMEN-DEMO-PNL-CURVE" in text and "NOT-IN-THE-LIBRARY-7" in text
        assert "in the library" in text and "not in the library" in text
        shot(page, "scan_choices")
        # choosing the unknown code selects no record and says so
        page.click('[data-code="NOT-IN-THE-LIBRARY-7"]')
        page.wait_for_selector(".cam", state="detached")
        page.wait_for_function("() => [...document.querySelectorAll('.toast')].some((t) => t.textContent.includes('No exact match'))", timeout=20_000)
        assert page.evaluate("async () => (await import('./static/js/store.js')).state.session") is None
        # scanning again and choosing the known panel code selects its operation
        page.click('[data-action="scan"]')
        page.wait_for_selector('[data-code="LUMEN-DEMO-PNL-CURVE"]', timeout=60_000)
        page.click('[data-code="LUMEN-DEMO-PNL-CURVE"]')
        page.wait_for_selector('[data-action="continue"]:not([disabled])', timeout=30_000)
        page.click('[data-action="continue"]')
        page.wait_for_selector('[data-action="add-tag"]')
        sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
        assert sess["operation"]["summary"]["operation_id"] == "DEMO-CURVE-01"
        assert not page._lumen_errors
        page.context.close()
    finally:
        b.close()
