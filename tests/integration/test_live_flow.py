"""Live guidance: one camera frame in, the overlay for that frame out (prompt 09)."""

import json
from dataclasses import replace
from urllib.parse import quote

import cv2
import numpy as np
import pytest

from lumen.synth import TagSticker
from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import calibration_for, jpeg_bytes, px, render_cached

W, H = 1600, 1200
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}
TARGET = [(180, 330), (300, 330), (300, 270), (180, 270)]  # PLC-DEMO-101 corners in the scene (x, z)


def meta(d):
    return {"x-lumen-meta": quote(json.dumps(d))}


@pytest.fixture()
def ready(client):
    """A registered session with confirmed datums (five calibrated photos)."""
    scene = flat_panel_scene()
    cams = [phone_camera(e, width=W, height=H, roll_deg=r) for e, r in sweep_stations(5, distance=760, height=430, span_deg=50)]
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cams[0])})
    for i, c in enumerate(cams):
        assert client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render_cached(scene, c, seed=i).image), headers={"content-type": "image/jpeg"}).status_code == 200
    client.post(f"/api/sessions/{sid}/register")
    client.post(f"/api/sessions/{sid}/datums/propose")
    client.post(f"/api/sessions/{sid}/datums/confirm")
    return sid, scene


def track(client, sid, img, m=None, **params):
    r = client.post(f"/api/sessions/{sid}/track", params=params, content=jpeg_bytes(img, 88), headers={"content-type": "image/jpeg", **(meta(m) if m else {})})
    assert r.status_code == 200, r.text
    return r.json()


def target_error_mm(ov, cam):
    tgt = next(p for p in ov["primitives"] if p["kind"].startswith("target") and "corners" in p)
    truth = np.array([px(cam, x, z) for x, z in TARGET])
    return float(np.max(np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)) / ov["px_per_mm"])


def test_live_frame_from_a_new_viewpoint_gets_its_own_overlay(client, ready):
    sid, scene = ready
    # a viewpoint that was never photographed, at the reduced size a phone would send
    cam = phone_camera((120.0, -640.0, 520.0), target=(250.0, 0.0, 310.0), width=1280, height=960, roll_deg=-6.0)
    img = render_cached(scene, cam, ss=2, seed=40).image
    n_before = len(client.get(f"/api/sessions/{sid}").json()["frames"])
    ov = track(client, sid, img)
    t = ov["tracking"]
    assert t["state"] == "tracking" and t["code"] == "tracking" and sorted(t["used_tags"]) == [3, 7, 11, 20]
    assert ov["frame_id"] == "live" and ov["frozen"] is False and ov["frame"]["width"] == 1280
    assert target_error_mm(ov, cam) < 0.4
    rep = ov["reports"]["PLC-DEMO-101"]
    assert rep["uncertainty"] is not None and not t["uncertainty_reused"]
    assert rep["status"] in ("prototype_guidance", "research_preview")
    # the frame was used and thrown away: nothing was added to the session
    s = client.get(f"/api/sessions/{sid}").json()
    assert len(s["frames"]) == n_before and all(f["id"] != "live" for f in s["frames"])
    assert set(ov["timing_ms"]) == {"decode", "tags", "localise", "overlay", "placard", "total"} and ov["timing_ms"]["placard"] < 1.0, "no placard is measured unless asked for"


def test_uncertainty_is_reused_only_for_a_nearly_identical_view(client, ready):
    sid, scene = ready
    cam = phone_camera((330.0, -700.0, 440.0), width=1280, height=960)
    a = track(client, sid, render_cached(scene, cam, ss=2, seed=41).image)
    b = track(client, sid, render_cached(scene, cam, ss=2, seed=42).image)  # same pose, new noise
    assert not a["tracking"]["uncertainty_reused"] and b["tracking"]["uncertainty_reused"]
    ua, ub = a["reports"]["PLC-DEMO-101"]["uncertainty"], b["reports"]["PLC-DEMO-101"]["uncertainty"]
    assert ua["expanded_mm"] == ub["expanded_mm"]
    # the caller can insist on a fresh evaluation
    c = track(client, sid, render_cached(scene, cam, ss=2, seed=42).image, full="true")
    assert not c["tracking"]["uncertainty_reused"]
    # 150 mm to the side is a different view: evaluated afresh
    moved = phone_camera((480.0, -700.0, 440.0), width=1280, height=960)
    d = track(client, sid, render_cached(scene, moved, ss=2, seed=43).image)
    assert not d["tracking"]["uncertainty_reused"]
    assert abs(target_error_mm(d, moved)) < 0.4


def test_lost_tracking_returns_no_geometry_and_says_why(client, ready):
    sid, scene = ready
    blank = np.full((960, 1280, 3), 140, np.uint8)
    cv2.randn(blank, 140, 5)
    ov = track(client, sid, blank)
    assert ov["tracking"]["state"] == "lost" and ov["tracking"]["code"] == "no_tags_in_view"
    assert ov["primitives"] == [] and ov["reports"] == {}
    far = phone_camera((270.0, -2050.0, 420.0), width=1280, height=960)  # tags about 23 px across: found, but too small to measure
    ov = track(client, sid, render_cached(scene, far, ss=2, seed=44).image)
    assert ov["tracking"]["state"] == "lost" and ov["tracking"]["code"] == "tags_too_small" and ov["tracking"]["arrow"] == "closer"
    assert ov["primitives"] == []
    # tags that were never registered are not a basis for guidance
    other = replace(scene, stickers=tuple(replace(s, tag_id=s.tag_id + 100) for s in scene.stickers))
    cam = phone_camera((330.0, -700.0, 440.0), width=1280, height=960)
    ov = track(client, sid, render_cached(other, cam, ss=2, seed=45).image)
    assert ov["tracking"]["state"] == "lost" and ov["tracking"]["code"] == "unknown_tags" and ov["primitives"] == []


def test_target_outside_the_picture_gives_a_direction_and_no_outline(client, ready):
    sid, scene = ready
    # close to the panel and aimed to the right of the target: the two right-hand tags are in view, the target is not
    cam = phone_camera((670.0, -520.0, 305.0), target=(670.0, 0.0, 305.0), width=1280, height=960)
    assert px(cam, 300, 300)[0] < 0  # even the target's right edge is off-screen to the left
    ov = track(client, sid, render_cached(scene, cam, ss=2, seed=46).image)
    t = ov["tracking"]
    assert t["state"] == "tracking" and t["code"] == "target_out_of_view" and t["arrow"] == "left", t
    assert sorted(t["used_tags"]) == [7, 20]
    assert not any(p["kind"].startswith("target") for p in ov["primitives"])
    assert ov["reports"]["PLC-DEMO-101"]["status"] == "blocked"


def test_a_moved_tag_blocks_live_guidance(client, ready):
    sid, scene = ready
    moved = replace(scene, stickers=tuple(replace(s, center=(s.center[0] + 9.0, s.center[1])) if s.tag_id == 7 else s for s in scene.stickers))
    cam = phone_camera((330.0, -700.0, 440.0), width=1280, height=960)
    ov = track(client, sid, render_cached(moved, cam, ss=2, seed=47).image)
    t = ov["tracking"]
    assert t["state"] == "tracking" and t["code"] == "tag_moved" and "high_residual" in t["flags"]
    assert not any(p["kind"].startswith("target") for p in ov["primitives"])
    assert ov["reports"]["PLC-DEMO-101"]["status"] == "blocked"


def test_reduced_live_frame_uses_the_calibration_of_its_stream(client, ready):
    sid, scene = ready
    cam = phone_camera((330.0, -700.0, 440.0), width=W, height=H)
    full = render_cached(scene, cam, seed=48).image
    small = cv2.resize(full, (800, 600), interpolation=cv2.INTER_AREA)
    # without the stream size the profile (made for 1600x1200) cannot be matched
    a = track(client, sid, small)
    assert a["frame"]["calibrated"] is False
    b = track(client, sid, small, {"native_width": W, "native_height": H})
    assert b["frame"]["calibrated"] is True and b["tracking"]["state"] == "tracking"
    # a declared size with a different aspect ratio is not a uniform rescale: ignored
    c = track(client, sid, small, {"native_width": 1920, "native_height": 1080})
    assert c["frame"]["calibrated"] is False


def test_tracking_needs_a_registration(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/track", content=jpeg_bytes(np.zeros((120, 160, 3), np.uint8)), headers={"content-type": "image/jpeg"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_registered"
