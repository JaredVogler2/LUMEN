"""End-to-end browser walkthrough of the demo session on a phone-sized screen."""

import json
import re

import pytest

from tests.browser.conftest import assert_clean_text, PHONE_LANDSCAPE, load_demo_to_place, shot

pytestmark = pytest.mark.browser


def test_first_screen_states(page):
    assert page.inner_text(".research-strip").lower().startswith("research prototype")
    assert page.get_attribute('[data-step="identify"]', "aria-current") == "step"
    # nothing selected yet: the primary action is disabled and later steps refuse to open
    assert page.is_disabled('[data-action="continue"]')
    page.click('[data-step="capture"]')
    page.wait_for_selector(".toast")
    assert "operation" in page.inner_text(".toast").lower()
    assert page.get_attribute('[data-step="identify"]', "aria-current") == "step"
    # the three sample records are listed, the incomplete one is marked
    assert page.locator("[data-operation]").count() == 3
    assert "incomplete" in page.inner_text('[data-operation="DEMO-INCOMPLETE-01"]').lower()
    page.fill('input[type="search"]', "curve")
    assert page.locator("[data-operation]").count() == 1
    shot(page, "identify")
    assert not page._lumen_errors


def test_choosing_an_operation_shows_its_definition_and_units(page):
    page.click('[data-operation="DEMO-FLAT-01"]')
    page.wait_for_selector('[data-placard="PLC-DEMO-102"]')
    page.click('[data-placard="PLC-DEMO-102"]')
    text = page.inner_text("#view")
    assert "419.1" in text and "76×76" in text  # 16.5 in and 3 in, converted once, shown in mm
    assert "Floor / panel interface" in text and "Left joint seam" in text
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="add-tag"]')
    assert "DEMO-FLAT-01" in page.inner_text(".app-header")
    shot(page, "markers")
    # enter two measured tags and continue
    page.fill('input[aria-label="Tag ID to add"]', "3")
    page.fill('input[aria-label="Measured edge in millimetres"]', "49.9")
    page.click('[data-action="add-tag"]')
    assert page.locator('tr[data-tag="3"]').count() == 1
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="choose-photos"]')
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    assert sess["tag_sizes"]["measured_mm"] == {"3": 49.9} and sess["mode"] == "real"
    shot(page, "capture_empty")
    assert not page._lumen_errors


def test_demo_walkthrough_reaches_guidance_and_labels_everything(page):
    load_demo_to_place(page)
    assert_clean_text(page)
    assert page.get_attribute(".status-hero", "data-status") == "prototype_guidance"
    hero = page.inner_text(".status-hero").lower()
    assert "simulated" in hero and "not been validated" in hero
    assert "simulated demo data" in page.inner_text(".research-strip").lower()
    assert page.get_attribute("[data-mode]", "data-mode") == "frozen"
    assert "frozen" in page.inner_text("[data-mode]").lower()
    assert page.inner_text(".viewer-watermark").lower().startswith("research only")
    shot(page, "place_portrait")
    # switching placard keeps the registration and updates the readout
    page.click('[data-placard="PLC-DEMO-102"]')
    # (function form: the app's CSP forbids eval, which Playwright needs for bare expressions)
    page.wait_for_function("() => document.querySelector('.app-header .sub').textContent.includes('PLC-DEMO-102')")
    page.wait_for_function("() => document.querySelector('.panel').innerText.includes('419.1')", timeout=60_000)
    # the uncertainty budget is available as text, not only as a picture
    page.click("details.fold:has-text('Uncertainty budget') summary")
    budget = page.inner_text("details.fold:has-text('Uncertainty budget')").lower()
    assert "repeatability" in budget and "systematic" in budget and "k=2" in budget
    assert not page._lumen_errors


def test_landscape_layout_and_theme_toggle(browser, live_server):
    ctx = browser.new_context(**PHONE_LANDSCAPE)
    page = ctx.new_page()
    page.goto(live_server.url)
    page.wait_for_selector('[data-step="identify"]')
    load_demo_to_place(page)
    # steps move to a side rail; viewer and panel sit side by side
    steps = page.locator(".steps").bounding_box()
    viewer = page.locator(".viewer-wrap").bounding_box()
    panel = page.locator(".panel").bounding_box()
    assert steps["height"] > steps["width"] and viewer["x"] >= steps["width"] - 1
    assert panel["x"] >= viewer["x"] + viewer["width"] - 2 and viewer["width"] > 300
    shot(page, "place_landscape")
    before = page.evaluate("document.documentElement.dataset.theme")
    page.click('[data-action="menu"]')
    page.click(f"text={'Light mode' if before == 'dark' else 'Dark mode'}")
    after = page.evaluate("document.documentElement.dataset.theme")
    assert after != before
    shot(page, f"place_landscape_{after}")
    ctx.close()


def test_session_survives_reload_and_can_be_deleted(page):
    load_demo_to_place(page)
    sid = page.evaluate("async () => (await import('./static/js/store.js')).state.session.id")
    page.reload()
    page.wait_for_selector('[data-step="identify"]')
    page.wait_for_function("async () => { const s = (await import('./static/js/store.js')).state.session; return !!s; }")
    assert page.evaluate("async () => (await import('./static/js/store.js')).state.session.id") == sid
    # export contains no image data
    with page.expect_download() as dl:
        page.click('[data-action="menu"]')
        page.click("text=Export session summary")
    data = json.loads(open(dl.value.path()).read())
    assert data["images_included"] is False and data["simulated"] is True and data["session"]["id"] == sid
    assert not re.search(r"/9j/[A-Za-z0-9+/]{100}", json.dumps(data)), "no embedded JPEG"
    page.click('[data-action="menu"]')
    page.click("text=End session and delete photos")
    page.click('[data-action="confirm-delete"]')
    page.wait_for_function("async () => (await import('./static/js/store.js')).state.session === null")
    status = page.evaluate(f"async () => (await fetch('api/sessions/{sid}')).status")
    assert status == 404


def test_accessibility_basics(page):
    load_demo_to_place(page)
    assert page.get_attribute("html", "lang") == "en"
    # every button has an accessible name
    unnamed = page.evaluate(
        """() => [...document.querySelectorAll('button')].filter(b => !(b.textContent || '').trim() && !b.getAttribute('aria-label')).length"""
    )
    assert unnamed == 0
    # primary touch targets are glove sized
    sizes = page.evaluate("""() => [...document.querySelectorAll('.actions .btn')].map(b => b.getBoundingClientRect().height)""")
    assert sizes and min(sizes) >= 56
    small = page.evaluate("""() => [...document.querySelectorAll('.btn, .icon-btn, .step')].filter(b => b.offsetParent && b.getBoundingClientRect().height < 40).length""")
    assert small == 0
    # the canvas is described and keyboard reachable; the status is announced
    assert page.get_attribute(".viewer-wrap canvas", "role") == "img" and page.get_attribute(".viewer-wrap canvas", "aria-label")
    assert page.get_attribute(".viewer-wrap canvas", "tabindex") == "0"
    assert page.get_attribute(".status-hero", "role") == "status"
    # text contrast of body text against its background (WCAG AA needs 4.5)
    ratio = page.evaluate(
        """() => { const lum = c => { const v = c.match(/\\d+(\\.\\d+)?/g).slice(0,3).map(Number).map(x => { x /= 255; return x <= 0.03928 ? x/12.92 : Math.pow((x+0.055)/1.055, 2.4); }); return 0.2126*v[0]+0.7152*v[1]+0.0722*v[2]; };
            const st = getComputedStyle(document.body); const a = lum(st.color), b = lum(st.backgroundColor); return (Math.max(a,b)+0.05)/(Math.min(a,b)+0.05); }"""
    )
    assert ratio >= 7.0


def test_ruler_measurement_on_the_rectified_screen_blocks_a_wrong_scale(page):
    """Independent scale check from the interface: a contradicting ruler reading removes the target."""
    load_demo_to_place(page)
    page.click('[data-step="rectified"]')
    page.wait_for_selector('[data-region="scale-check"] [data-action="ref-check"]', timeout=60_000)
    page.click('[data-action="ref-show"]')
    page.wait_for_function("() => window.__lumenViewer.overlay.primitives.filter((p) => p.kind === 'handle').length === 2", timeout=30_000)
    shot(page, "rectified_scale_check")
    sug = page.evaluate("async () => { const s = (await import('./static/js/store.js')).state.session; return (await (await fetch(`api/sessions/${s.id}/reference`)).json()).suggestion; }")
    a, b = sug["points_plane"]
    d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    # a ruler reading 3 % shorter than what the tags imply
    page.fill('input[aria-label="Measured distance in millimetres"]', f"{d * 0.97:.1f}")
    page.click('[data-action="ref-check"]')
    page.wait_for_selector('[data-region="scale-check"] .banner.danger', timeout=30_000)
    assert "contradiction" in page.inner_text('[data-region="scale-check"]').lower()
    page.wait_for_selector('.status-hero[data-status="blocked"]', timeout=30_000)
    assert not page.evaluate("() => window.__lumenViewer.overlay.primitives.some((p) => p.kind.startsWith('target'))")
    # removing the wrong reading and entering the right one restores guidance
    page.click('[data-action="ref-remove"]')
    page.wait_for_selector('[data-action="ref-check"]', timeout=30_000)
    page.fill('input[aria-label="Measured distance in millimetres"]', f"{d + 0.3:.1f}")
    page.click('[data-action="ref-check"]')
    page.wait_for_selector('[data-region="scale-check"] .banner.ok', timeout=30_000)
    page.wait_for_selector('.status-hero[data-status="prototype_guidance"]', timeout=30_000)
    assert not page._lumen_errors
