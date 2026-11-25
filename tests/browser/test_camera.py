"""In-app camera in headless Chromium with a fake capture device (prompt 02).

The fake device plays a synthetic video of the panel.  These tests prove the
browser code path (permission, capture, torch probing, recording, recovery,
stopping the camera); they say nothing about iPhone Safari.
"""

import pytest

from tests.browser.conftest import assert_clean_text, camera_frames, launch_with_camera, open_app, shot, to_capture_step, write_y4m

pytestmark = pytest.mark.browser

# A torch-capable camera, as some mobile browsers expose it: the capability is
# listed and the setting reports the state that was asked for.
TORCH_REPORTED = """
(() => {
  const P = MediaStreamTrack.prototype;
  const caps = P.getCapabilities, sets = P.getSettings, apply = P.applyConstraints;
  window.__torch = false; window.__torchCalls = [];
  P.getCapabilities = function () { return { ...(caps ? caps.call(this) : {}), torch: true }; };
  P.getSettings = function () { return { ...sets.call(this), torch: window.__torch }; };
  P.applyConstraints = function (c) {
    const adv = c && c.advanced && c.advanced.find((a) => 'torch' in a);
    if (adv) { window.__torch = !!adv.torch; window.__torchCalls.push(!!adv.torch); return Promise.resolve(); }
    return apply.call(this, c);
  };
})();
"""

# A browser that lists the capability, accepts the request, but neither
# reports the state nor lights anything: LUMEN must not claim the light is on.
TORCH_SILENT = """
(() => {
  const P = MediaStreamTrack.prototype;
  const caps = P.getCapabilities, apply = P.applyConstraints;
  P.getCapabilities = function () { return { ...(caps ? caps.call(this) : {}), torch: true }; };
  P.applyConstraints = function (c) {
    if (c && c.advanced && c.advanced.find((a) => 'torch' in a)) return Promise.resolve();
    return apply.call(this, c);
  };
})();
"""

DENIED = """
navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError'));
"""


@pytest.fixture(scope="module")
def static_cam(playwright_instance, tmp_path_factory):
    frames, cams = camera_frames(1)
    y4m = write_y4m(str(tmp_path_factory.mktemp("cam") / "static.y4m"), frames * 4, fps=10)
    b = launch_with_camera(playwright_instance, y4m)
    yield b
    b.close()


@pytest.fixture(scope="module")
def sweep_cam(playwright_instance, tmp_path_factory):
    frames, cams = camera_frames(5)
    seq = [f for f in frames for _ in range(15)]  # each view held 1.5 s at 10 fps, looping
    y4m = write_y4m(str(tmp_path_factory.mktemp("cam") / "sweep.y4m"), seq, fps=10)
    b = launch_with_camera(playwright_instance, y4m)
    yield b
    b.close()


def open_camera(page):
    page.click('[data-action="live-camera"]')
    page.wait_for_selector('[data-cam-state="live"]', timeout=30_000)
    page.wait_for_selector('[data-action="cam-shutter"]:not([disabled])')


def photos(page) -> int:
    return int(page.get_attribute("[data-cam-count]", "data-cam-count"))


def test_camera_opens_only_on_request_and_stops_when_closed(static_cam, live_server):
    page = open_app(static_cam, live_server.url)
    to_capture_step(page)
    # nothing touches the camera until the operator asks for it
    assert page.evaluate("() => window.__lumenCamera === undefined || window.__lumenCamera === null")
    open_camera(page)
    v = page.evaluate("() => { const c = window.__lumenCamera; const v = c.video; return { w: v.videoWidth, h: v.videoHeight, inline: v.hasAttribute('playsinline'), muted: v.muted, audio: c.stream.getAudioTracks().length, state: c.track.readyState }; }")
    assert (v["w"], v["h"]) == (960, 540) and v["inline"] and v["muted"] and v["audio"] == 0 and v["state"] == "live"
    # no torch on this device: the control is not offered, and no flash-pair mode either
    assert page.is_hidden('[data-action="cam-torch"]') and page.locator('[data-action="cam-mode-pair"]').count() == 0
    shot(page, "camera_live")
    page.evaluate("() => { window.__t = window.__lumenCamera.track; }")
    page.click('[data-action="cam-close"]')
    page.wait_for_selector(".cam", state="detached")
    assert page.evaluate("() => window.__t.readyState") == "ended"
    assert not page._lumen_errors
    page.context.close()


def test_photo_capture_uploads_full_frame_and_coaches(static_cam, live_server):
    page = open_app(static_cam, live_server.url)
    to_capture_step(page)
    open_camera(page)
    assert "first photo" in page.inner_text(".cam-coach").lower()
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => document.querySelector('[data-cam-count]').dataset.camCount === '1' && !document.querySelector('[data-cam-count]').textContent.includes('uploading')", timeout=60_000)
    page.wait_for_selector('.cam-coach [data-coach="more_views"]', timeout=20_000)
    assert page.get_attribute(".cam-coach .coach", "data-arrow") in ("left", "right")
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    f = sess["frames"][0]
    assert (f["width"], f["height"]) == (960, 540) and f["source"] == "camera_still" and f["torch"] == "unknown"
    assert sorted(m["tag_id"] for m in f["markers"] if m["accepted"]) == [3, 7, 11, 20]
    assert f["camera_label"]  # the stream's device label is recorded with the photo
    page.click('[data-action="cam-done"]')
    page.wait_for_selector(".cam", state="detached")
    page.wait_for_selector('[data-frame="f001"]')
    page.wait_for_selector('[data-region="coach"] .coach')
    shot(page, "capture_one_photo")
    assert not page._lumen_errors
    page.context.close()


def test_sweep_of_photos_registers_end_to_end(sweep_cam, live_server):
    page = open_app(sweep_cam, live_server.url)
    to_capture_step(page)
    open_camera(page)
    for k in range(5):
        page.click('[data-action="cam-shutter"]')
        page.wait_for_function("(n) => document.querySelector('[data-cam-count]').dataset.camCount === String(n) && !document.querySelector('[data-cam-count]').textContent.includes('uploading')", arg=k + 1, timeout=60_000)
        page.wait_for_timeout(1300)  # the fake camera moves on to the next viewpoint
    page.click('[data-action="cam-done"]')
    page.wait_for_selector('[data-action="analyze"]:not([disabled])')
    page.click('[data-action="analyze"]')
    page.wait_for_selector('[data-datum="floor"]', timeout=180_000)
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    reg = sess["registration"]
    assert reg["consistent"] and reg["n_tags"] == 4 and len(reg["frames_used"]) == 5
    # the photos really came from different places (the fake camera cycles through 5 viewpoints)
    assert reg["multiview"]["checks"]["diversity"]["max_view_angle_deg"] > 10.0
    # a real (not demo) session all the way to the placement and verification screens
    assert_clean_text(page)
    page.wait_for_selector('.busy', state="detached", timeout=120_000)
    page.wait_for_selector('[data-datum="side"] .chip', timeout=120_000)
    if page.is_enabled('[data-action="confirm-datums"]'):
        page.click('[data-action="confirm-datums"]')
        page.wait_for_selector(".status-hero", timeout=120_000)
    for step in ("rectified", "place"):
        page.click(f'[data-step="{step}"]')
        page.wait_for_selector(".status-hero", timeout=120_000)
        page.wait_for_selector('.busy', state="detached", timeout=60_000)
        assert_clean_text(page)
    assert "research prototype" in page.inner_text(".research-strip").lower() and "simulated" not in page.inner_text(".research-strip").lower()
    # Five small video-size frames, no lens profile: an outline is shown as a preview with reasons.  It is not
    # blocked: this is where a wrong "the floor datum is not on the panel face" used to stop a correct capture.
    assert page.get_attribute(".status-hero", "data-status") == "research_preview", page.inner_text(".panel")[:600]
    failed = {el.get_attribute("data-gate") for el in page.query_selector_all(".gate.fail")}
    assert "lens_characterised" in failed and not {g for g in failed if g.endswith("_on_plane")}, failed
    shot(page, "place_real_mode")
    assert not page._lumen_errors
    page.context.close()


def test_recorded_sweep_is_uploaded_and_frames_are_chosen(sweep_cam, live_server):
    page = open_app(sweep_cam, live_server.url)
    to_capture_step(page)
    open_camera(page)
    if page.locator('[data-action="cam-mode-sweep"]').count() == 0:
        pytest.skip("this Chromium build cannot record video")
    page.click('[data-action="cam-mode-sweep"]')
    page.click('[data-action="cam-shutter"]')
    page.wait_for_selector('[data-cam-state="recording"]')
    page.wait_for_timeout(8000)  # one loop of the fake sweep is 7.5 s
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => Number(document.querySelector('[data-cam-count]').dataset.camCount) >= 3", timeout=120_000)
    page.wait_for_selector('[data-cam-state="live"]', timeout=60_000)
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    assert all(f["source"] == "live_sweep" and f["video_time_s"] is not None for f in sess["frames"])
    assert all(len([m for m in f["markers"] if m["accepted"]]) >= 3 for f in sess["frames"])
    job = [j for j in sess["jobs"].values()][-1]
    assert job["state"] == "done" and job["result"]["video"]["container"] in ("webm", "mp4")
    assert job["result"]["spread_deg"] is not None and job["result"]["spread_deg"] > 15.0
    assert not page._lumen_errors
    page.context.close()


def test_torch_control_and_flash_pair_when_the_browser_reports_the_light(static_cam, live_server):
    page = open_app(static_cam, live_server.url, init_script=TORCH_REPORTED)
    to_capture_step(page)
    open_camera(page)
    assert page.is_visible('[data-action="cam-torch"]')
    page.click('[data-action="cam-torch"]')
    page.wait_for_selector('[data-action="cam-torch"][aria-pressed="true"]')
    assert page.evaluate("() => window.__lumenCamera.torchState()") == "on"
    assert "reported by the browser" in page.evaluate("() => window.__lumenCamera.torchEvidence()")
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => document.querySelector('[data-cam-count]').dataset.camCount === '1'", timeout=60_000)
    page.click('[data-action="cam-torch"]')
    page.wait_for_selector('[data-action="cam-torch"][aria-pressed="false"]')
    # flash pair: light off, photo, light on, photo, light off again
    page.click('[data-action="cam-mode-pair"]')
    page.evaluate("() => { window.__torchCalls.length = 0; }")
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => document.querySelector('[data-cam-count]').dataset.camCount === '3' && !document.querySelector('[data-cam-count]').textContent.includes('uploading')", timeout=90_000)
    page.wait_for_selector('.toast.ok', timeout=30_000)
    assert page.evaluate("() => window.__torchCalls") == [False, True, False]
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    fr = sess["frames"]
    assert fr[0]["torch"] == "on" and fr[0]["pair_id"] is None
    assert [fr[1]["torch"], fr[2]["torch"]] == ["off", "on"] and fr[1]["pair_id"] == fr[2]["pair_id"] != None  # noqa: E711
    adv = page.evaluate("async (sid) => (await (await fetch(`api/sessions/${sid}/capture/advice`)).json())", sess["id"])
    assert adv["pairs"][0]["usable"] and adv["pairs"][0]["status"] == "aligned"
    page.click('[data-action="cam-close"]')
    assert page.evaluate("() => window.__torch") is False  # never left on
    assert not page._lumen_errors
    page.context.close()


def test_unconfirmed_light_is_not_recorded_as_on(static_cam, live_server):
    page = open_app(static_cam, live_server.url, init_script=TORCH_SILENT)
    to_capture_step(page)
    open_camera(page)
    page.click('[data-action="cam-torch"]')
    page.wait_for_selector(".toast")
    assert "could not be confirmed" in page.inner_text(".toasts").lower()
    assert page.evaluate("() => window.__lumenCamera.torchState()") == "unknown"
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => document.querySelector('[data-cam-count]').dataset.camCount === '1'", timeout=60_000)
    sess = page.evaluate("async () => (await import('./static/js/store.js')).state.session")
    assert sess["frames"][0]["torch"] == "unknown"
    # a flash pair without a confirmed light is refused, and nothing is kept from it
    page.click('[data-action="cam-torch"]')
    page.click('[data-action="cam-mode-pair"]')
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => [...document.querySelectorAll('.toast')].some((t) => t.textContent.includes('did not come on'))", timeout=30_000)
    assert photos(page) == 1
    page.context.close()


def test_permission_refusal_explains_and_offers_the_fallback(static_cam, live_server):
    page = open_app(static_cam, live_server.url, init_script=DENIED)
    to_capture_step(page)
    page.click('[data-action="live-camera"]')
    page.wait_for_selector('[data-cam-error="denied"]')
    text = page.inner_text(".cam-card").lower()
    assert "refused" in text and "allow" in text and "upload" in text
    assert page.is_disabled('[data-action="cam-shutter"]')
    assert page.is_visible('[data-action="cam-retry"]') and page.is_visible('[data-action="cam-fallback"]')
    shot(page, "camera_denied")
    page.click('[data-action="cam-close"]')
    page.wait_for_selector(".cam", state="detached")
    page.context.close()


def test_interruption_and_background_stop_the_camera_and_can_be_resumed(static_cam, live_server):
    page = open_app(static_cam, live_server.url)
    to_capture_step(page)
    open_camera(page)
    # the stream ends underneath the page (phone call, another app takes the camera)
    page.evaluate("() => window.__lumenCamera.track.dispatchEvent(new Event('ended'))")
    page.wait_for_selector('[data-cam-state="paused"]')
    assert page.is_disabled('[data-action="cam-shutter"]')
    assert page.evaluate("() => window.__lumenCamera.stream === null")
    page.click('[data-action="cam-resume"]')
    page.wait_for_selector('[data-cam-state="live"]', timeout=30_000)
    # the page goes to the background: the camera must not keep running
    page.evaluate("() => { window.__t = window.__lumenCamera.track; Object.defineProperty(document, 'hidden', { configurable: true, get: () => true }); document.dispatchEvent(new Event('visibilitychange')); }")
    page.wait_for_selector('[data-cam-state="paused"]')
    assert page.evaluate("() => window.__t.readyState") == "ended"
    page.evaluate("() => { Object.defineProperty(document, 'hidden', { configurable: true, get: () => false }); }")
    page.click('[data-action="cam-resume"]')
    page.wait_for_selector('[data-cam-state="live"]', timeout=30_000)
    page.click('[data-action="cam-shutter"]')
    page.wait_for_function("() => document.querySelector('[data-cam-count]').dataset.camCount === '1'", timeout=60_000)
    assert not page._lumen_errors
    page.context.close()
