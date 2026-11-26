"""The application in a browser, on a host that forgets everything between two requests: the browser carries the job.

``forgetful`` is the worst such host: every request, the pages and scripts included, is served by an application
made for that request, and nothing above 4.5 MB passes in either direction.  ``warm`` is one application that
stays, as an instance of a function does while requests keep coming; the live loop is tried there, where what an
instance remembers from frame to frame (the held registration) can show.

What this is not: Vercel, or a phone.  It is Chromium against the same application behind a stand-in for such a
host (``tests/carry_client.py``).
"""

import time
from dataclasses import replace

import cv2
import pytest

from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
from tests.browser.conftest import LiveServer, assert_clean_text, open_app, shot
from tests.browser.test_placement import PLACEMENT, approach, session, start  # noqa: F401  (approach is a fixture)
from tests.carry_client import StatelessHost, carried_app
from tests.conftest import render_cached

pytestmark = pytest.mark.browser

HELD = "async () => (await (await import('./static/js/carry.js')).held()).map((j) => ({ state: j.state_bytes, pictures: j.pictures }))"
STATS = "async () => { const c = await import('./static/js/carry.js'); return { ...c.stats, recent: c.recent.map((r) => [r.method, r.path, r.status, r.pictures, r.bytes]) }; }"
IN_DATABASE = """() => new Promise((ok) => { const r = indexedDB.open('lumen-carry'); r.onsuccess = () => { const t = r.result.transaction(['jobs', 'pictures']);
  const a = t.objectStore('jobs').count(), b = t.objectStore('pictures').count(); t.oncomplete = () => ok([a.result, b.result]); }; r.onerror = () => ok(null); })"""
SESSION = "async () => (await import('./static/js/store.js')).state.session"


def until(page, js, test, timeout=30.0, arg=None):
    """Poll an expression (which may be async) until ``test`` holds for its value."""
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = page.evaluate(js, arg) if arg is not None else page.evaluate(js)
        if test(last):
            return last
        page.wait_for_timeout(150)
    raise AssertionError(f"not reached within {timeout} s; last value {str(last)[:300]}")


@pytest.fixture(scope="module")
def forgetful():
    host = StatelessHost(carried_app)
    srv = LiveServer(host).start()
    srv.host = host
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def warm():
    srv = LiveServer(carried_app()).start()
    yield srv
    srv.stop()


def test_the_demo_job_runs_where_the_server_forgets_everything(browser, forgetful):
    page = open_app(browser, forgetful.url)
    served_before = forgetful.host.requests
    # nothing on the first screen says that this cannot work: it can
    cfg = until(page, "async () => (await import('./static/js/store.js')).state.config", lambda v: bool(v))
    assert cfg["sessions"]["kept_by"] == "client" and cfg["hosting"]["message"] == ""
    assert page.locator(".banner.danger").count() == 0
    page.click('[data-action="demo"]')
    page.wait_for_selector('[data-action="analyze"]', timeout=180_000)
    # the photos are shown from the browser's own copies, and no request asks a server for them
    page.wait_for_function("() => { const i = [...document.querySelectorAll('.frame-tile img')]; return i.length === 5 && i.every((x) => x.complete && x.naturalWidth === 1600); }", timeout=30_000)
    assert all(src.startswith("blob:") for src in page.evaluate("() => [...document.querySelectorAll('.frame-tile img')].map((i) => i.src)"))
    # said on the Photos step: photos are reduced here; and there is no video where nobody keeps the job while it is worked on
    assert "1600 px" in page.inner_text('[data-region="photos-reduced"]') and page.locator('[data-action="choose-video"]').count() == 0
    shot(page, "carried_photos")
    page.click('[data-action="analyze"]')
    page.wait_for_selector('[data-datum="floor"] .chip.ok', timeout=180_000)
    page.wait_for_selector('[data-datum="side"] .chip.ok', timeout=60_000)
    page.wait_for_selector(".busy", state="detached", timeout=60_000)
    page.click('[data-action="confirm-datums"]')
    page.wait_for_selector('[data-step="place"][aria-current="step"]', timeout=120_000)
    page.wait_for_selector('.status-hero[data-status="prototype_guidance"]', timeout=120_000)
    assert page.evaluate("() => window.__lumenViewer.overlay.primitives.some((p) => p.kind === 'target')")
    assert_clean_text(page)
    # ---- the job is in this browser: one document and the five photos, in memory and in its database
    held = page.evaluate(HELD)
    assert len(held) == 1 and 40_000 < held[0]["state"] < 200_000 and [p["name"] for p in held[0]["pictures"]] == [f"f00{i}.jpg" for i in range(1, 6)]
    assert page.evaluate(IN_DATABASE) == [1, 5]
    st = page.evaluate(STATS)
    # the five photos went up once, for the step that searches the lines in them; no step had to ask for a photo
    assert st["picturesSent"] == 5 and st["askedAgain"] == 0, st
    assert [r for r in st["recent"] if r[3]] == [["POST", "api/sessions/…/datums/propose", 200, 5, r[4]] for r in st["recent"] if r[3]]
    # ---- a reload in the middle (a phone does that by itself when memory is short): the job is still there
    sid = page.evaluate(SESSION)["id"]
    page.reload()
    page.wait_for_selector('[data-step="identify"]')
    s = until(page, SESSION, lambda v: bool(v))
    assert s["id"] == sid and len(s["frames"]) == 5 and s["registration"] and s["datums"]["floor"]["confirmed"]
    page.click('[data-step="place"]')
    page.wait_for_selector('.status-hero[data-status="prototype_guidance"]', timeout=120_000)
    # ---- a record from a photo, and the straight-on view (each takes the one photo it looks at)
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="verify-demo"]')
    page.click('[data-action="verify-demo"]')
    page.wait_for_selector('[data-observation="indeterminate"]', timeout=120_000)
    v = page.evaluate(SESSION)["verifications"][0]
    assert v["dx_mm"] == pytest.approx(1.4, abs=0.25) and v["dz_mm"] == pytest.approx(-0.9, abs=0.25)
    page.click('[data-step="datums"]')
    page.click('[data-action="scale-check"]')
    page.wait_for_selector('[data-region="scale-check"]', timeout=60_000)
    page.wait_for_function("() => window.__lumenViewer && window.__lumenViewer.W > 0 && window.__lumenViewer.overlay.primitives.length > 0", timeout=60_000)
    st = page.evaluate(STATS)
    sent = [(r[1], r[3]) for r in st["recent"] if r[3]]
    assert ("api/sessions/…/verify", 1) in sent and any(p.endswith("/rectified") and n == 1 for p, n in sent), sent
    assert all(n == 1 for _, n in sent), sent
    assert st["askedAgain"] == 0
    shot(page, "carried_scale_check")
    # the diagnostics say where the job is and how the last requests went
    page.click('[data-action="menu"]')
    page.click("text=Diagnostics")
    card = page.wait_for_selector('[data-region="carrier"]')
    page.wait_for_selector('[data-region="carrier-recent"] tbody tr')
    text = card.inner_text()
    assert "is on this phone" in text and "6 photos" in text and "verify" in text and "rectified" in text
    shot(page, "carried_diagnostics")
    page.keyboard.press("Escape")
    page.wait_for_selector('[data-region="carrier"]', state="detached")
    # ---- ended: nothing of the job is left in this browser
    page.click('[data-action="menu"]')
    page.click("text=End session and delete photos")
    assert "this phone" in page.inner_text(".sheet").lower()
    page.click('[data-action="confirm-delete"]')
    until(page, SESSION, lambda v: v is None)
    assert until(page, HELD, lambda v: v == []) == [] and page.evaluate(IN_DATABASE) == [0, 0]
    # ---- and the host: every request met an application that knew nothing, and none was too large for a function
    assert forgetful.host.requests - served_before >= 20
    assert forgetful.host.largest_request < 1_200_000 and forgetful.host.largest_answer < 1_200_000
    assert not page._lumen_errors, page._lumen_errors
    page.context.close()


def test_photos_from_the_phone_are_reduced_so_that_the_job_fits_one_request(browser, forgetful, tmp_path):
    # five photos as a phone's camera app makes them: larger than this host would take together
    scene = flat_panel_scene()
    cams = [phone_camera(e, width=2880, height=2160, roll_deg=r) for e, r in sweep_stations(5, distance=760, height=430, span_deg=50)]
    files = []
    for i, cam in enumerate(cams):
        p = tmp_path / f"IMG_{i + 1}.jpg"
        cv2.imwrite(str(p), render_cached(scene, cam, seed=40 + i).image, [cv2.IMWRITE_JPEG_QUALITY, 96])
        files.append(str(p))
    total = sum((tmp_path / f"IMG_{i + 1}.jpg").stat().st_size for i in range(5))
    assert total > 4_500_000, "as they come from the camera the five photos do not fit one request"
    page = open_app(browser, forgetful.url)
    page.click('[data-operation="DEMO-FLAT-01"]')
    page.wait_for_selector('[data-action="continue"]:not([disabled])')
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="add-tag"]')
    for tid in (3, 7, 11, 20):
        page.fill('input[aria-label="Tag ID to add"]', str(tid))
        page.fill('input[aria-label="Measured edge in millimetres"]', "50")
        page.click('[data-action="add-tag"]')
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="choose-photos"]')
    page.set_input_files('[data-input="photos"]', files)
    s = until(page, SESSION, lambda v: bool(v) and len(v["frames"]) == 5, timeout=120.0)
    lim = page.evaluate("async () => (await import('./static/js/store.js')).state.config.sessions")
    for f in s["frames"]:
        # each photo arrived upright, at 1600 px, small enough that nine of them and the job fit one request
        assert (f["width"], f["height"]) == (1600, 1200) and f["processing_scale"] == 1.0 and not f["orientation_applied"]
        assert f["n_bytes"] <= lim["photo_max_bytes"] and len([m for m in f["markers"] if m["accepted"]]) == 4, (f["id"], f["n_bytes"])
    page.wait_for_function("() => { const i = [...document.querySelectorAll('.frame-tile img')]; return i.length === 5 && i.every((x) => x.complete && x.naturalWidth === 1600); }", timeout=30_000)
    page.click('[data-action="analyze"]')
    page.wait_for_selector('[data-datum="floor"] .chip', timeout=180_000)
    page.wait_for_selector(".busy", state="detached", timeout=120_000)
    s = page.evaluate(SESSION)
    reg = s["registration"]
    assert reg["consistent"] and reg["n_tags"] == 4 and len(reg["frames_used"]) == 5
    assert set(s["datums"]) == {"floor", "side"}, "the lines were found in the reduced photos"
    # the lines lie where the panel has them (floor line z = 0, seam x = 0 in the panel's own frame is not known here:
    # what is checked is what the placement needs, the target's place between them)
    page.click('[data-action="confirm-datums"]')
    page.wait_for_selector('[data-step="place"][aria-current="step"]', timeout=120_000)
    page.wait_for_selector(".status-hero", timeout=120_000)
    # real photos without a lens profile: a target is shown, with what is open
    assert page.get_attribute(".status-hero", "data-status") == "research_preview"
    assert page.evaluate("() => window.__lumenViewer.overlay.primitives.some((p) => p.kind.startsWith('target'))")
    st = page.evaluate(STATS)
    assert st["picturesSent"] == 5 and st["askedAgain"] == 0 and max(r[4] for r in st["recent"]) < lim["max_request_bytes"], st
    assert_clean_text(page)
    shot(page, "carried_place_real")
    assert not page._lumen_errors, page._lumen_errors
    page.context.close()


def test_live_placement_with_a_job_the_browser_carries(approach, warm):
    """The film of ``test_placement.py``: the placard is brought to its target and rests there.  The job is carried
    by the browser; the application is one that stays, as a warm instance does."""
    browser, steps = approach
    page = open_app(browser, warm.url)
    start(page)
    page.wait_for_selector('[data-region="placement"][data-phase="guiding"]', timeout=60_000)
    assert page.evaluate(PLACEMENT)["last"]["found"]
    page.wait_for_selector('[data-region="placement"][data-phase="in_position"]', timeout=60_000)
    assert page.get_attribute(".split", "data-placing") == "ok"
    page.wait_for_selector('[data-region="placement-record"][data-observation="within_tolerance_observed"]', timeout=90_000)
    s = session(page)
    recs = [v for v in s["verifications"] if v["placard_id"] == "PLC-DEMO-101"]
    dx, dz, _ = steps[-1]
    assert len(recs) == 1 and recs[0]["dx_mm"] == pytest.approx(dx, abs=0.3) and recs[0]["dz_mm"] == pytest.approx(dz, abs=0.3)
    done = page.evaluate("() => window.__lumenPlacementLast")
    log = done["log"]
    assert log.index("guide") < log.index("chime:in") < log.index("hold") < log.index("chime:recorded"), log
    # the record's photo is one more picture of the job, kept by this browser; a live frame is not
    held = page.evaluate(HELD)
    assert [p["name"] for p in held[0]["pictures"]] == [f"f00{i}.jpg" for i in range(1, 7)]
    st = page.evaluate(STATS)
    live = [r for r in st["recent"] if r[1].endswith("/track")]
    assert live and all(r[2] == 200 and r[3] == 0 for r in live), "a live frame brings the job's state and itself, never a photo"
    # with the instance staying, the phone that stands still is recognised as such: the registration was held
    assert done["stats"]["results"] >= 6
    assert not page._lumen_errors, page._lumen_errors
    page.context.close()
