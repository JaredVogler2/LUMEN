"""Live guidance in headless Chromium with a fake camera (prompt 09).

The fake camera shows the same synthetic panel as the demo session, from a
viewpoint that is not among the demo photos.  The tests check that the
overlay drawn for a live frame belongs to that frame, that it is in the right
place on screen, and that it disappears when tracking is lost.
"""

import numpy as np
import pytest

from tests.browser.conftest import camera_frames, launch_with_camera, load_demo_to_place, open_app, shot, write_y4m
from tests.conftest import px

pytestmark = pytest.mark.browser

TARGET = [(180, 330), (300, 330), (300, 270), (180, 270)]  # PLC-DEMO-101 corners in the scene (x, z)


@pytest.fixture(scope="module")
def steady(playwright_instance, tmp_path_factory):
    frames, cams = camera_frames(3)
    y4m = write_y4m(str(tmp_path_factory.mktemp("live") / "steady.y4m"), [frames[1]] * 4, fps=10)
    b = launch_with_camera(playwright_instance, y4m)
    yield b, cams[1]
    b.close()


@pytest.fixture(scope="module")
def blinking(playwright_instance, tmp_path_factory):
    frames, cams = camera_frames(3)
    blank = np.full_like(frames[1], 135)
    y4m = write_y4m(str(tmp_path_factory.mktemp("live") / "blink.y4m"), [frames[1]] * 40 + [blank] * 40, fps=10)  # 4 s panel, 4 s nothing
    b = launch_with_camera(playwright_instance, y4m)
    yield b, cams[1]
    b.close()


def start_live(page):
    load_demo_to_place(page)
    assert page.get_attribute("[data-mode]", "data-mode") == "frozen"
    page.click('[data-action="live"]')
    page.wait_for_selector('[data-mode="live"]', timeout=60_000)


def test_live_overlay_belongs_to_the_frame_and_lands_on_the_target(steady, live_server):
    browser, cam = steady
    page = open_app(browser, live_server.url)
    start_live(page)
    page.wait_for_function("() => window.__lumenLive && window.__lumenLive.updates >= 3", timeout=60_000)
    res = page.evaluate("() => window.__lumenLive.lastResult")
    t = res["tracking"]
    assert t["state"] == "tracking" and sorted(t["used_tags"]) == [3, 7, 11, 20]
    assert (res["frame"]["width"], res["frame"]["height"]) == (960, 540) and res["frozen"] is False and res["simulated"] is True
    tgt = next(p for p in res["primitives"] if p["kind"].startswith("target") and p.get("active") is not False)
    truth = np.array([px(cam, x, z) for x, z in TARGET])
    err_px = float(np.max(np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)))
    assert err_px / res["px_per_mm"] < 0.8, (err_px, res["px_per_mm"])  # millimetres on the panel, 960x540 stream through the browser
    # what the viewer shows is that frame, with that overlay, mapped by the viewer transform
    v = page.evaluate("() => { const v = window.__lumenViewer; return { W: v.W, H: v.H, n: v.overlay.primitives.length, c: v.toView(%f, %f), t: v.t }; }" % tuple(tgt["corners"][0]))
    assert (v["W"], v["H"]) == (960, 540) and v["n"] == len(res["primitives"])
    assert abs(v["c"][0] - (v["t"]["tx"] + (tgt["corners"][0][0] + 0.5) * v["t"]["s"])) < 1e-6
    # the panel says what this view is worth
    box = page.inner_text('[data-region="live-status"]').lower()
    assert "simulated" in box and "uncertainty" in box
    shot(page, "live_tracking")
    # freeze: the picture stops updating and is labelled as not live
    page.click('[data-action="live-freeze"]')
    page.wait_for_selector('[data-mode="frozen"]')
    assert "not live" in page.inner_text("[data-mode]").lower()
    n = page.evaluate("() => window.__lumenLive.updates")
    page.wait_for_timeout(1500)
    assert page.evaluate("() => window.__lumenLive.updates") == n
    page.click('[data-action="live-freeze"]')
    page.wait_for_selector('[data-mode="live"]', timeout=30_000)
    # stopping returns to the frozen photo and releases the camera
    page.evaluate("() => { window.__t = window.__lumenLive.camera.track; }")
    page.click('[data-action="live"]')
    page.wait_for_function("() => window.__lumenLive === null")
    page.wait_for_selector('[data-mode="frozen"]')
    assert page.evaluate("() => window.__t.readyState") == "ended"
    assert "frozen photo" in page.inner_text("[data-mode]").lower()
    assert not page._lumen_errors
    page.context.close()


def test_overlay_disappears_when_tracking_is_lost_and_returns_with_it(blinking, live_server):
    browser, cam = blinking
    page = open_app(browser, live_server.url)
    start_live(page)
    page.wait_for_selector('[data-mode="lost"]', timeout=30_000)
    lost = page.evaluate("() => ({ n: window.__lumenViewer.overlay.primitives.length, code: window.__lumenLive.lastResult.tracking.code, prims: window.__lumenLive.lastResult.primitives.length })")
    assert lost == {"n": 0, "code": "no_tags_in_view", "prims": 0}
    assert "no tag in view" in page.inner_text(".live-coach").lower()
    assert "nothing is drawn" in page.inner_text('[data-region="live-status"]').lower()
    shot(page, "live_lost")
    page.wait_for_selector('[data-mode="live"]', timeout=30_000)
    assert page.evaluate("() => window.__lumenViewer.overlay.primitives.some((p) => p.kind.startsWith('target'))")
    page.click('[data-action="live"]')
    page.wait_for_function("() => window.__lumenLive === null")
    assert not page._lumen_errors
    page.context.close()


def test_stale_answers_hide_the_overlay(steady, live_server):
    browser, cam = steady
    page = open_app(browser, live_server.url)
    start_live(page)
    page.wait_for_function("() => window.__lumenLive && window.__lumenLive.updates >= 2", timeout=60_000)
    # tracking requests stop being answered (the request leaves, nothing comes back)
    page.evaluate("""() => {
      const open = XMLHttpRequest.prototype.open, send = XMLHttpRequest.prototype.send;
      XMLHttpRequest.prototype.open = function (m, u, ...r) { this.__url = String(u); return open.call(this, m, u, ...r); };
      XMLHttpRequest.prototype.send = function (b) { if (this.__url && this.__url.includes('/track')) return undefined; return send.call(this, b); };
    }""")
    page.wait_for_selector('[data-mode="stale"]', timeout=15_000)
    assert page.evaluate("() => window.__lumenViewer.overlay.primitives.length") == 0
    assert "overlay hidden" in page.inner_text("[data-mode]").lower()
    page.context.close()
