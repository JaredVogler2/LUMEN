"""Assisted placement in headless Chromium with a fake camera.

The fake camera plays a short film of the demo panel (the same invented panel as the demo session) in which the
placard PLC-DEMO-101 is brought to its target and then rests there.  The app has to follow it, say which way to
move it, turn green when it is in position, take the record by itself and announce it — and it must do none of
that for a placard that is not there, is covered, or is no longer being watched.

What this proves and what it does not: the browser code, the sound *commands* (headless Chromium has no
loudspeaker) and the whole path through the server.  It is not an iPhone, and nothing here was heard by anyone.
"""

from dataclasses import replace

import numpy as np
import pytest

from lumen.synth.fixtures import demo_placard, flat_panel_scene, phone_camera
from lumen.synth.scene import Box
from tests.browser.conftest import assert_clean_text, launch_with_camera, load_demo_to_place, open_app, shot, write_y4m
from tests.conftest import px, render_cached

pytestmark = pytest.mark.browser

VW, VH = 1280, 720
CAM = phone_camera((250.0, -640.0, 330.0), target=(245.0, 0.0, 300.0), width=VW, height=VH, roll_deg=2.0)
FPS = 4


def view(dx=None, dz=0.0, rot=0.0, seed=0, boxes=()):
    """The panel as the phone on its stand sees it; with the placard ``(dx, dz)`` mm from its target, or without it."""
    scene = flat_panel_scene()
    if dx is not None:
        scene = replace(scene, placards=(demo_placard(x=240.0 + dx, z=300.0 + dz, rotation_deg=rot),))
    if boxes:
        scene = replace(scene, boxes=tuple(boxes))
    return render_cached(scene, CAM, ss=2, seed=seed).image


def film(tmp_path_factory, name, shots):
    """``shots``: (image, seconds) in order; the camera plays them in a loop."""
    frames = [img for img, sec in shots for _ in range(max(1, round(sec * FPS)))]
    return write_y4m(str(tmp_path_factory.mktemp("place") / f"{name}.y4m"), frames, fps=FPS)


@pytest.fixture(scope="module")
def approach(playwright_instance, tmp_path_factory):
    steps = [(34.0, -20.0, 5.0), (15.0, -9.0, 2.5), (5.0, -3.0, 1.2), (0.3, 0.2, 0.1)]
    shots = [(view(dx, dz, rot, seed=300 + k), 1.5 if k < 3 else 40.0) for k, (dx, dz, rot) in enumerate(steps)]
    b = launch_with_camera(playwright_instance, film(tmp_path_factory, "approach", shots))
    yield b, steps
    b.close()


@pytest.fixture(scope="module")
def far_away(playwright_instance, tmp_path_factory):
    b = launch_with_camera(playwright_instance, film(tmp_path_factory, "far", [(view(22.0, 12.0, -3.0, seed=310), 1.0)]))
    yield b
    b.close()


@pytest.fixture(scope="module")
def covered(playwright_instance, tmp_path_factory):
    # in position, with something 40 mm thick over its whole right side: a hand that has not let go yet.  The
    # hand's straight edge runs parallel to the side, 12 mm inside the placard.
    hand = Box(lo=(288.4, -40.0, 254.7), hi=(350.4, -0.5, 344.7), albedo=(0.55, 0.42, 0.36))
    b = launch_with_camera(playwright_instance, film(tmp_path_factory, "covered", [(view(0.4, -0.3, 0.1, seed=320, boxes=(hand,)), 1.0)]))
    yield b
    b.close()


def start(page):
    load_demo_to_place(page)
    assert "start live placement" in page.inner_text('[data-action="live"]').lower()
    page.click('[data-action="live"]')
    page.wait_for_selector('[data-mode="live"]', timeout=60_000)
    page.wait_for_selector('[data-region="placement"]')


def session(page):
    return page.evaluate("async () => (await import('./static/js/store.js')).state.session")


PLACEMENT = "() => { const p = window.__lumenPlacement; return p ? { phase: p.phase, stats: p.stats, log: p.sound.log.map((e) => e.kind), last: p.last && p.last.placement } : null; }"


def test_a_placard_is_followed_to_its_target_turns_green_and_is_recorded(approach, live_server):
    browser, steps = approach
    page = open_app(browser, live_server.url)
    start(page)
    # ---- on the way: told to move, with the distance, and beeps
    page.wait_for_selector('[data-region="placement"][data-phase="guiding"]', timeout=60_000)
    st = page.evaluate(PLACEMENT)
    p = st["last"]
    assert p["found"] and p["observation"] == "outside_tolerance_observed" and p["radial_mm"] > 4.0
    assert "guide" in st["log"] and "hold" not in st["log"]
    text = page.inner_text('[data-region="placement"]').lower()
    assert "move it" in text and "mm" in text and "simulated" in text and "uncertainty" in text
    # the live outline and the arrow to the target are drawn on that frame
    kinds = page.evaluate("() => window.__lumenViewer.extra.map((x) => x.kind)")
    assert "placard_live" in kinds and "move_arrow" in kinds
    shot(page, "placement_guiding")
    # ---- at the target: green, the steady tone, then the record by itself, announced
    page.wait_for_selector('[data-region="placement"][data-phase="in_position"]', timeout=60_000)
    assert page.get_attribute(".split", "data-placing") == "ok"
    shot(page, "placement_in_position")
    page.wait_for_selector('[data-region="placement-record"][data-observation="within_tolerance_observed"]', timeout=90_000)
    s = session(page)
    recs = [v for v in s["verifications"] if v["placard_id"] == "PLC-DEMO-101"]
    assert len(recs) == 1 and recs[0]["observation"] == "within_tolerance_observed"
    dx, dz, rot = steps[-1]
    assert recs[0]["dx_mm"] == pytest.approx(dx, abs=0.3) and recs[0]["dz_mm"] == pytest.approx(dz, abs=0.3)
    # the record is a stored photo of its own, at the camera's full size, and the registration is untouched
    rec_frame = next(f for f in s["frames"] if f["id"] == recs[0]["frame_id"])
    assert rec_frame["role"] == "verification" and (rec_frame["width"], rec_frame["height"]) == (VW, VH)
    assert s["registration"] is not None
    # the camera was released and the Place step says what was recorded
    page.wait_for_function("() => window.__lumenLive === null", timeout=15_000)
    page.wait_for_selector('[data-mode="frozen"]')
    done = page.evaluate("() => window.__lumenPlacementLast")
    log = done["log"]
    # heard in this order: beeps on the way, the chime of reaching it, the steady tone, the chime of the record
    assert log.index("guide") < log.index("chime:in") < log.index("hold") < log.index("chime:recorded"), log
    assert "chime:fail" not in log and done["stats"]["attempts"] == 1 and done["stats"]["records"] == 1
    assert done["stats"]["phases"][-2:] == ["recording", "recorded"] and "in_position" in done["stats"]["phases"]
    card = page.inner_text('[data-region="placement-record"]').lower()
    assert "recorded" in card and "within the tolerance" in card and "not an acceptance decision" in card
    assert "verify placement" in page.inner_text('[data-action="continue"]').lower()
    assert_clean_text(page)
    shot(page, "placement_recorded")
    assert not page._lumen_errors
    page.context.close()


def test_far_from_the_target_it_says_which_way_and_records_only_when_asked(far_away, live_server):
    page = open_app(far_away, live_server.url)
    start(page)
    page.wait_for_selector('[data-region="placement"][data-phase="guiding"]', timeout=60_000)
    page.wait_for_function("() => window.__lumenPlacement.stats.results >= 4", timeout=30_000)
    st = page.evaluate(PLACEMENT)
    p = st["last"]
    # 22 mm right of the target and 12 mm above it, turned 3 degrees clockwise as seen from the front
    assert p["dx_mm"] == pytest.approx(22.0, abs=0.3) and p["dz_mm"] == pytest.approx(12.0, abs=0.3) and p["rotation_deg"] == pytest.approx(-3.0, abs=0.15)
    assert p["method"] == "followed" and not p["in_position"]
    # the arrow beside the dial points, on the screen, from the placard to its target
    want = px(CAM, 240.0, 300.0) - px(CAM, 262.0, 312.0)
    want_deg = float(np.degrees(np.arctan2(want[1], want[0])))
    got = page.evaluate("() => document.querySelector('.placing-arrow').style.transform")
    got_deg = float(got.replace("rotate(", "").replace("deg)", ""))
    assert abs((got_deg - want_deg + 180) % 360 - 180) < 2.0, (got, want_deg)
    assert page.evaluate("() => getComputedStyle(document.querySelector('.placing-arrow')).visibility") == "visible"
    # it is turned clockwise on the screen, so the sign says: turn it back, counter-clockwise
    assert page.evaluate("() => document.querySelector('.placing-turn').classList.contains('ccw')") and p["move"]["turn_screen_deg"] < 0
    # on the dial the disc sits where the placard is relative to the target (here: right and above), in the state's colour
    px_col = page.evaluate("""async () => {
      const { gaugeMap } = await import('./static/js/views/placement.js');
      const p = window.__lumenPlacement.last.placement, cv = document.querySelector('canvas.gauge');
      const W = cv.clientWidth, R = W / 2 - 4, k = cv.width / W, len = Math.hypot(p.move.px[0], p.move.px[1]);
      const rr = gaugeMap(p.radial_mm / p.tolerance_mm) * R;
      const x = W / 2 - p.move.px[0] / len * rr, y = W / 2 - p.move.px[1] / len * rr;
      const d = cv.getContext('2d').getImageData(Math.round(x * k), Math.round(y * k), 1, 1).data;
      return { rgb: [d[0], d[1], d[2]], right: x > W / 2, above: y < W / 2 };
    }""")
    assert px_col["rgb"] == [92, 200, 255] and px_col["right"] and px_col["above"], px_col
    # nothing was recorded by itself: it never was in position
    assert st["stats"]["attempts"] == 0 and "guide" in st["log"] and "hold" not in st["log"] and not any(k.startswith("chime") for k in st["log"])
    assert not [v for v in session(page)["verifications"]]
    # ---- a frozen picture is not a reading: the sound stops and the panel says so
    page.click('[data-action="live-freeze"]')
    page.wait_for_selector('[data-region="placement"][data-phase="lost"]')
    assert page.evaluate("() => window.__lumenPlacement.sound.mode") == "silent"
    assert "frozen" in page.inner_text('[data-region="placement-message"]').lower()
    assert page.evaluate("() => window.__lumenViewer.extra.length") == 0
    page.click('[data-action="live-freeze"]')
    page.wait_for_selector('[data-region="placement"][data-phase="guiding"]', timeout=30_000)
    # ---- sound can be switched off; the choice is kept
    page.click('[data-action="place-sound"]')
    assert page.get_attribute('[data-action="place-sound"]', "aria-pressed") == "false" and "sound off" in page.inner_text('[data-action="place-sound"]').lower()
    assert page.evaluate("() => window.__lumenPlacement.sound.muted") is True
    assert page.evaluate("() => JSON.parse(localStorage.getItem('lumen.placeSound'))") is False
    shot(page, "placement_far")
    # ---- recorded by hand, it is stored as what it is: outside the tolerance, and announced as that
    page.click('[data-action="place-record"]')
    page.wait_for_selector('[data-region="placement-record"][data-observation="outside_tolerance_observed"]', timeout=90_000)
    page.wait_for_function("() => window.__lumenLive === null", timeout=15_000)
    done = page.evaluate("() => window.__lumenPlacementLast")
    assert "chime:recorded" not in done["log"] and done["stats"]["records"] == 0 and done["stats"]["attempts"] == 1
    card = page.inner_text('[data-region="placement-record"]').lower()
    assert "outside the tolerance" in card and "not an acceptance decision" in card
    recs = session(page)["verifications"]
    assert len(recs) == 1 and recs[0]["dx_mm"] == pytest.approx(22.0, abs=0.3)
    assert "place again" in page.inner_text('[data-action="live"]').lower()
    assert_clean_text(page)
    assert not page._lumen_errors
    page.context.close()


def test_in_position_under_a_hand_it_waits_and_a_lost_connection_ends_the_tone(covered, live_server):
    page = open_app(covered, live_server.url)
    start(page)
    page.wait_for_selector('[data-region="placement"][data-phase="in_position"]', timeout=60_000)
    assert "let go to record" in page.inner_text('[data-region="placement-message"]').lower()
    # it stays green and sounding, and takes no record while a corner is covered
    page.wait_for_function("() => window.__lumenPlacement.stats.in_position >= 12", timeout=60_000)
    st = page.evaluate(PLACEMENT)
    assert st["phase"] == "in_position" and st["stats"]["attempts"] == 0 and st["last"]["hands"] and not st["last"]["clean"]
    assert st["log"].count("chime:in") == 1 and "hold" in st["log"] and "chime:recorded" not in st["log"]
    assert page.evaluate("() => window.__lumenPlacement.sound.mode") == "hold"
    assert page.get_attribute(".split", "data-placing") == "ok"
    # the covered side was not measured: it is placed by the placard's size, drawn dashed, and said so
    assert st["last"]["sides"] == ["measured", "record_size", "measured", "measured"]
    assert st["last"]["dx_mm"] == pytest.approx(0.4, abs=0.3) and st["last"]["dz_mm"] == pytest.approx(-0.3, abs=0.3)
    outline = page.evaluate("() => window.__lumenViewer.extra.find((x) => x.kind === 'placard_live')")
    assert outline["placed"] == [False, True, False, False]
    assert any("placed by the placard" in t for t in page.locator(".placing-flags li").all_inner_texts())
    shot(page, "placement_hand")
    # ---- the answers stop coming: within a moment the tone ends, and then the reading is withdrawn
    page.evaluate("""() => {
      const open = XMLHttpRequest.prototype.open, send = XMLHttpRequest.prototype.send;
      XMLHttpRequest.prototype.open = function (m, u, ...r) { this.__url = String(u); return open.call(this, m, u, ...r); };
      XMLHttpRequest.prototype.send = function (b) { if (this.__url && this.__url.includes('/track')) return undefined; return send.call(this, b); };
    }""")
    page.wait_for_function("() => window.__lumenPlacement.sound.mode === 'silent'", timeout=4_000)
    page.wait_for_selector('[data-region="placement"][data-phase="lost"]', timeout=10_000)
    assert page.get_attribute(".split", "data-placing") == "idle"
    assert page.evaluate("() => window.__lumenViewer.extra.length") == 0
    assert not session(page)["verifications"]
    page.context.close()


def test_without_a_placard_nothing_is_judged_or_sounded(playwright_instance, tmp_path_factory, live_server):
    b = launch_with_camera(playwright_instance, film(tmp_path_factory, "bare", [(view(None, seed=330), 1.0)]))
    try:
        page = open_app(b, live_server.url)
        start(page)
        page.wait_for_selector('[data-region="placement"][data-phase="searching"]', timeout=60_000)
        page.wait_for_function("() => window.__lumenPlacement.stats.results >= 3", timeout=30_000)
        st = page.evaluate(PLACEMENT)
        assert not st["last"]["found"] and st["log"] == [] and st["stats"]["attempts"] == 0
        assert page.is_disabled('[data-action="place-record"]')
        assert "dashed outline" in page.inner_text('[data-region="placement-message"]').lower()
        # the target is still drawn for the frame, as before
        assert page.evaluate("() => window.__lumenViewer.overlay.primitives.some((p) => p.kind.startsWith('target'))")
        assert page.evaluate("() => window.__lumenViewer.extra.length") == 0
        shot(page, "placement_searching")
        page.click('[data-action="live"]')
        page.wait_for_function("() => window.__lumenLive === null")
        page.wait_for_selector('[data-mode="frozen"]')
        assert "start live placement" in page.inner_text('[data-action="live"]').lower()
        assert not page._lumen_errors
        page.context.close()
    finally:
        b.close()
