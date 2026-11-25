"""Rounded-edge demo: the edge model on the Datums screen and the research-only flashlight card (prompts 06, 07)."""

import pytest

from tests.browser.conftest import assert_clean_text, shot

pytestmark = pytest.mark.browser


def test_rounded_edge_demo_shows_the_edge_model_and_the_flashlight_experiment(page):
    page.click('[data-action="demo-curve"]')
    page.wait_for_selector('[data-action="analyze"]', timeout=180_000)
    page.click('[data-action="analyze"]')
    page.wait_for_selector('[data-datum="side"] .chip.ok', timeout=240_000)
    page.wait_for_selector(".busy", state="detached", timeout=180_000)
    panel = page.inner_text(".panel").lower()  # headings are shown in capitals
    assert "rounded edge" in panel and "radius" in panel and "silhouettes" in panel
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    assert sess["mode"] == "demo" and sess["datums"]["side"]["feature_kind"] == "cylinder_model"
    assert sess["surface"]["cylinder"]["ok"] and abs(sess["surface"]["cylinder"]["radius_mm"] - 35.0) < 1.5
    # the experiment is offered as research, with what it is and is not
    card = page.wait_for_selector('[data-region="photometric"]')
    text = card.inner_text()
    assert "research only" in text.lower() and "never used for a datum" in text and "No photo with the light on yet" in text
    before = page.evaluate("async () => { const s = (await import('./static/js/store.js')).state.session; return JSON.stringify([s.datums, s.local_frame]); }")
    page.click('[data-action="photometric-run"]')
    page.wait_for_function("() => /needs other photos|no result|consistent shading model/.test(document.querySelector('[data-region=\"photometric\"]').innerText)", timeout=180_000)
    text = page.inner_text('[data-region="photometric"]')
    # the demo photos were taken without the phone light: it says what it needs instead of estimating
    assert "needs other photos" in text and "light" in text.lower()
    after = page.evaluate("async () => { const s = (await import('./static/js/store.js')).state.session; return [JSON.stringify([s.datums, s.local_frame]), s.research.photometric.guidance_enabled, s.research.photometric.label]; }")
    assert after[0] == before and after[1] is False and after[2] == "research_evidence_only"
    shot(page, "datums_curve_photometric")
    assert_clean_text(page)
    assert not page._lumen_errors, page._lumen_errors
