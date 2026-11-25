"""Multi-photo registration through the API: 3-D checks and what they catch."""

import numpy as np
import pytest

from lumen.synth import TagSticker
from lumen.synth.fixtures import DEMO_TAGS, flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import calibration_for, jpeg_bytes, px, render_cached

W, H = 1280, 960
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}


def cams(n=5, **kw):
    return [phone_camera(e, width=W, height=H, roll_deg=r, **kw) for e, r in sweep_stations(n, distance=740, height=430, span_deg=50)]


def start(client, camera=None, sizes=MEASURED, cal=None):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    if sizes:
        client.put(f"/api/sessions/{sid}/tags", json=sizes)
    if cal is not None or camera is not None:
        client.put(f"/api/sessions/{sid}/calibration", json={"profile": cal or calibration_for(camera)})
    return sid


def upload(client, sid, img):
    r = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(img), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200, r.text
    return r.json()["frame"]


def register(client, sid):
    r = client.post(f"/api/sessions/{sid}/register")
    assert r.status_code == 200, r.text
    return r.json()


def test_calibrated_sweep_verifies_the_plane_and_the_lens_profile(client):
    scene = flat_panel_scene()
    cs = cams()
    sid = start(client, cs[0])
    for i, c in enumerate(cs):
        upload(client, sid, render_cached(scene, c, seed=i).image)
    out = register(client, sid)
    reg, sess = out["registration"], out["session"]
    mv = reg["multiview"]
    assert reg["method"] == "planar_joint+multiview_ba" and mv["calibrated"] and mv["intrinsics_provenance"] == "measured"
    assert mv["checks"]["coplanarity"]["verdict"] == "coplanar"
    assert mv["checks"]["diversity"]["diverse"] and mv["checks"]["held_out"]["consistent"]
    assert mv["calibration_check"]["consistent"] and abs(mv["calibration_check"]["relative_difference"]) < 0.003
    assert sess["surface"]["planarity"] == "verified_multiview" and sess["surface"]["out_of_plane_max_mm"] < 1.0
    # camera positions are available for later 3-D reasoning
    assert len(mv["camera_centers"]) == 5
    rep = client.get(f"/api/sessions/{sid}/report").json()
    codes = {g["code"]: g for g in rep["gates"]}
    assert codes["plane_assumption"]["passed"] and codes["plane_verified_3d"]["passed"] and codes["calibration_consistent"]["passed"] and codes["held_out_views"]["passed"]


def test_tag_moved_between_photos_is_excluded_everywhere(client):
    scene = flat_panel_scene()
    moved = flat_panel_scene(tags=tuple(TagSticker(t, ((c[0] + 6.0, c[1] + 3.0) if t == 7 else c), r) for t, c, r in DEMO_TAGS))
    cs = cams()
    sid = start(client, cs[0])
    for i, c in enumerate(cs):
        upload(client, sid, render_cached(moved if i >= 3 else scene, c, seed=i).image)
    reg = register(client, sid)["registration"]
    assert reg["excluded_tags"].get("7") == ["moved_or_misread"]
    assert "7" not in reg["layout"]["poses"] and reg["n_tags"] == 3 and reg["consistent"]
    # the target is still right in the photos taken after the tag moved
    ref = reg["reference_frame_id"]
    idx = [f["id"] for f in client.get(f"/api/sessions/{sid}").json()["frames"]].index(ref)
    cam = cs[idx]
    for kind, a, b in (("floor", (60, 0), (560, 0)), ("side", (0, 80), (0, 620))):
        r = client.post(f"/api/sessions/{sid}/datums/{kind}", json={"frame_id": ref, "p0": px(cam, *a).tolist(), "p1": px(cam, *b).tolist()})
        assert r.status_code == 200
    client.post(f"/api/sessions/{sid}/datums/confirm")
    frames = client.get(f"/api/sessions/{sid}").json()["frames"]
    last = frames[-1]["id"]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": last, "uncertainty": "false"}).json()
    tgt = next(p for p in ov["primitives"] if p["kind"].startswith("target") and "corners" in p)
    truth = np.array([px(cs[-1], 180, 330), px(cs[-1], 300, 330), px(cs[-1], 300, 270), px(cs[-1], 180, 270)])
    assert np.max(np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)) / ov["px_per_mm"] < 0.3


def test_raised_tag_is_excluded_as_off_plane(client):
    lifted = flat_panel_scene(tags=tuple(TagSticker(t, c, r, lift=(8.0 if t == 11 else 0.0)) for t, c, r in DEMO_TAGS))
    cs = cams()
    sid = start(client, cs[0])
    for i, c in enumerate(cs):
        upload(client, sid, render_cached(lifted, c, seed=i).image)
    out = register(client, sid)
    reg = out["registration"]
    assert "11" in reg["excluded_tags"] and "11" not in reg["layout"]["poses"]
    assert reg["multiview"]["checks"]["coplanarity"]["off_plane_tags"] == [11]
    assert out["session"]["surface"]["planarity"] == "verified_multiview_after_exclusion"
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert next(g for g in rep["gates"] if g["code"] == "plane_assumption")["passed"]
    # What is used further on describes the three tags that carry the result, not the four that were seen:
    # the raised tag is out of the 3-D solution, and the plane and its flatness figures are those of the rest.
    mv = reg["multiview"]
    cop = mv["checks"]["coplanarity"]
    assert "off_plane_tags_excluded" in mv["flags"] and "tags_not_coplanar" not in mv["flags"]
    assert sorted(mv["tags"]) == ["20", "3", "7"] and cop["set_aside_before_this_solution"] == [11]
    assert cop["verdict"] == "coplanar" and cop["rms_mm"] < 0.15 and abs(cop["d"]) < 0.3 and abs(cop["normal"][2]) > 0.9999
    assert out["session"]["surface"]["out_of_plane_rms_mm"] < 0.15
    left_out = next(g for g in rep["gates"] if g["code"] == "tags_left_out")
    assert left_out["passed"] and left_out["caveat"] and "ID 11 (it is not on the plane of the other tags)" in left_out["message"]
    # ... so the datum lines are measured from the panel face and not from a plane tilted by the shim
    p = client.post(f"/api/sessions/{sid}/datums/propose").json()["proposals"]
    assert set(p["applied"]) == {"floor", "side"}
    by_id = {c["id"]: c for c in p["candidates"]}
    for kind in ("floor", "side"):
        c = by_id[p["applied"][kind]]
        assert c["classification"] in ("on_plane", "unverified") and abs(c["height_mm"]) <= max(c["height_margin_mm"], 0.5), (kind, c["height_mm"], c["height_margin_mm"])
    assert by_id[p["applied"]["side"]]["classification"] == "on_plane"
    client.post(f"/api/sessions/{sid}/datums/confirm")
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] != "blocked" and not [g["code"] for g in rep["gates"] if not g["passed"] and g["code"].endswith("_on_plane")]


def test_wrong_lens_profile_is_detected_from_the_tags(client):
    scene = flat_panel_scene()
    cs = cams()
    wrong = calibration_for(cs[0])
    wrong["fx"] *= 1.06
    wrong["fy"] *= 1.06  # e.g. a profile made at another zoom step
    sid = start(client, cal=wrong)
    for i, c in enumerate(cs):
        upload(client, sid, render_cached(scene, c, seed=i).image)
    reg = register(client, sid)["registration"]
    cc = reg["multiview"]["calibration_check"]
    assert not cc["consistent"] and cc["relative_difference"] == pytest.approx(-0.0566, abs=0.004)
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert not next(g for g in rep["gates"] if g["code"] == "calibration_consistent")["passed"]


def test_uncalibrated_sweep_self_calibrates_for_diagnostics_only(client):
    scene = flat_panel_scene()
    cs = cams()
    sid = start(client)
    for i, c in enumerate(cs):
        upload(client, sid, render_cached(scene, c, seed=i).image)
    reg = register(client, sid)["registration"]
    mv = reg["multiview"]
    assert not mv["calibrated"] and mv["intrinsics_provenance"] == "inferred"
    sc = mv["checks"]["self_calibration"]
    assert sc["usable"] and sc["f"] == pytest.approx(cs[0].fx, rel=0.004)
    assert mv["checks"]["coplanarity"]["verdict"] == "coplanar"
    # an inferred focal length does not satisfy the calibration gate
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert not next(g for g in rep["gates"] if g["code"] == "lens_characterised")["passed"]


def test_single_photo_has_no_3d_check(client):
    scene = flat_panel_scene()
    c = cams(1)[0]
    sid = start(client, c)
    upload(client, sid, render_cached(scene, c, seed=0).image)
    out = register(client, sid)
    assert out["registration"]["multiview"] is None and out["registration"]["method"] == "planar_joint"
    assert out["session"]["surface"]["planarity"] == "crossvalidated_single_view"
