"""Verification step in the browser, on the simulated demo session (prompt 11).

The demo's verification photo shows the placard installed 1.4 mm to the right,
0.9 mm low and rotated 0.7 degrees (ground truth in the demo manifest).
"""

import json
import math

import pytest

from tests.browser.conftest import assert_clean_text, load_demo_to_place, shot

pytestmark = pytest.mark.browser


def session(page):
    return page.evaluate("async () => (await import('./static/js/store.js')).state.session")


def to_verify(page):
    load_demo_to_place(page)
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="verify-demo"]')


def test_verify_reports_signed_deviations_as_evidence_not_acceptance(page):
    to_verify(page)
    text = page.inner_text("#view").lower()
    assert "tags must not have moved" in text and "not an acceptance decision" in text
    assert page.is_disabled('[data-action="confirm"]') if page.locator('[data-action="confirm"]').count() else True
    shot(page, "verify_empty")
    page.click('[data-action="verify-demo"]')
    page.wait_for_selector("[data-observation]", timeout=120_000)
    v = session(page)["verifications"][0]
    assert v["dx_mm"] == pytest.approx(1.4, abs=0.25) and v["dz_mm"] == pytest.approx(-0.9, abs=0.25)
    assert math.degrees(v["rotation_rad"]) == pytest.approx(0.7, abs=0.15)
    assert v["uncertainty"]["expanded_mm"] > 0.3 and not v["confirmed"]
    # 1.66 mm from the target with a tolerance of 2.0 mm: closer to the limit than the uncertainty
    assert page.get_attribute("[data-observation]", "data-observation") == "indeterminate"
    hero = page.inner_text("[data-observation]").lower()
    assert "not an acceptance decision" in hero and "simulated" in hero
    dev = page.inner_text('[data-region="deviations"]')
    assert "+1." in dev and "-0." in dev  # signs are shown
    kinds = page.evaluate("() => window.__lumenViewer.overlay.primitives.map((p) => p.kind)")
    assert "measured" in kinds and any(k.startswith("target") for k in kinds)
    shot(page, "verify_result")
    assert_clean_text(page)
    # confirming records the outline as evidence and completes the step
    page.click('[data-action="confirm"]')
    page.wait_for_selector('[data-action="unconfirm"]')
    assert session(page)["verifications"][0]["confirmed"]
    assert "done" in page.get_attribute('[data-step="verify"]', "class")
    # the export has numbers and meanings, and no pictures
    with page.expect_download() as dl:
        page.click('[data-action="verify-export"]')
    data = json.load(open(dl.value.path()))
    assert data["images_included"] is False and data["simulated"] is True
    assert data["verifications"][0]["confirmed"] and "local" in data["coordinates"]
    assert "acceptance" in data["status_semantics"].lower()
    assert not page._lumen_errors


def test_corners_marked_by_hand_are_measured_too(page):
    to_verify(page)
    page.click('[data-action="verify-demo"]')
    page.wait_for_selector("[data-observation]", timeout=120_000)
    auto = session(page)["verifications"][0]
    page.click('[data-action="verify-manual"]')
    assert page.is_disabled('[data-action="manual-snap"]')
    box = page.locator(".viewer-wrap canvas").bounding_box()
    # tap near each measured corner (a few screen pixels off, in a different order)
    order = [2, 0, 3, 1]
    offs = [(3, -2), (-2, 3), (2, 2), (-3, -2)]
    for k, (ox, oy) in zip(order, offs):
        u, v = auto["corners_px"][k]
        x, y = page.evaluate("([u, v]) => window.__lumenViewer.toView(u, v)", [u, v])
        page.mouse.click(box["x"] + x + ox, box["y"] + y + oy)
    page.wait_for_selector('[data-action="manual-snap"]:not([disabled])')
    assert "4/4" in page.inner_text("[data-mode]")
    shot(page, "verify_manual")
    page.click('[data-action="manual-snap"]')
    page.wait_for_selector("[data-observation]", timeout=120_000)
    man = session(page)["verifications"][0]
    assert man["id"] != auto["id"]
    assert man["dx_mm"] == pytest.approx(auto["dx_mm"], abs=0.1) and man["dz_mm"] == pytest.approx(auto["dz_mm"], abs=0.1)
    # deleting the photo removes its result
    page.click('[data-action="verify-delete-photo"]')
    page.wait_for_selector('[data-action="verify-demo"]')
    assert session(page)["verifications"] == []
    assert not page._lumen_errors
