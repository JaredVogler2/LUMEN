"""Post-installation verification through the API (prompt 11).

The session is registered on photos of the bare panel; verification photos
show the placard installed at a known position, from a viewpoint that was not
used before.  Target: PLC-DEMO-101 at x 240, z 300, 120 x 60 mm, +-2.0 mm, +-1 deg.
"""

from dataclasses import replace

import numpy as np
import pytest

from lumen.synth.fixtures import demo_placard, flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import calibration_for, jpeg_bytes, px, render_cached

W, H = 1600, 1200
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}
VCAM = phone_camera((300.0, -640.0, 470.0), target=(255.0, 0.0, 300.0), width=W, height=H, roll_deg=3.0)


@pytest.fixture()
def ready(client):
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


def add_verification_photo(client, sid, scene, cam=VCAM, seed=60):
    img = render_cached(scene, cam, seed=seed).image
    r = client.post(f"/api/sessions/{sid}/frames", params={"role": "verification"}, content=jpeg_bytes(img, 95), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200, r.text
    return r.json()["frame"]["id"]


def verify(client, sid, fid, **body):
    r = client.post(f"/api/sessions/{sid}/verify", json={"frame_id": fid, **body})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize(
    "dx,dz,rot,expect",
    [
        (0.3, -0.2, 0.0, "within_tolerance_observed"),
        (1.4, -0.9, 0.0, "indeterminate"),  # 1.66 mm off: inside the tolerance, but not by more than the uncertainty
        (4.0, 2.5, 0.0, "outside_tolerance_observed"),
        (0.2, 0.1, 2.5, "outside_tolerance_observed"),  # rotation tolerance is 1 degree
        (-0.4, 0.3, -0.5, "indeterminate"),  # rotation inside the tolerance by less than its uncertainty? see assertions
    ],
)
def test_known_offsets_and_rotations_are_recovered_with_signs(client, ready, dx, dz, rot, expect):
    sid, scene = ready
    installed = replace(scene, placards=(demo_placard(240.0 + dx, 300.0 + dz, rotation_deg=rot),))
    n_reg = len(client.get(f"/api/sessions/{sid}").json()["registration"]["frames_used"])
    fid = add_verification_photo(client, sid, installed, seed=60 + int(abs(dx * 10)))
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["registration"] is not None and len(s["registration"]["frames_used"]) == n_reg, "a verification photo must not disturb the registration"
    out = verify(client, sid, fid)
    v = out["verification"]
    assert v["frame_id"] == fid and v["placard_id"] == "PLC-DEMO-101" and not v["confirmed"]
    assert v["dx_mm"] == pytest.approx(dx, abs=0.2) and v["dz_mm"] == pytest.approx(dz, abs=0.2)
    assert np.degrees(v["rotation_rad"]) == pytest.approx(rot, abs=0.12)
    assert abs(v["width_error_mm"]) < 0.4 and abs(v["height_error_mm"]) < 0.4
    assert v["radial_mm"] == pytest.approx(np.hypot(dx, dz), abs=0.2)
    U = v["uncertainty"]["expanded_mm"]
    assert 0.3 < U < 1.6
    # the true error of the measured deviation is inside the stated uncertainty
    assert np.hypot(v["dx_mm"] - dx, v["dz_mm"] - dz) < U
    names = [c["name"] for c in v["uncertainty"]["components"]]
    assert any("placard edges" in n for n in names) and any("edge definition" in n for n in names)
    # side offsets carry the same signs as the shift
    e = v["edge_deviation_mm"]
    assert set(e) == {"top", "right", "bottom", "left"}
    if rot == 0.0:
        assert e["left"] == pytest.approx(dx, abs=0.3) and e["right"] == pytest.approx(dx, abs=0.3)
        assert e["bottom"] == pytest.approx(dz, abs=0.3) and e["top"] == pytest.approx(dz, abs=0.3)
    if expect == "indeterminate" and rot != 0.0:
        assert v["observation"] in ("indeterminate", "within_tolerance_observed")
    else:
        assert v["observation"] == expect, (v["observation"], out["detail"].get("position_observation"), out["detail"].get("rotation_observation"), U)
    assert "not" in v["notice"].lower() and v["corner_provenance"] == ["measured"] * 4
    # corners in the picture are where the placard really is
    truth = np.array([px(VCAM, x, z) for x, z in _corners(240.0 + dx, 300.0 + dz, rot)])
    assert np.max(np.linalg.norm(np.array(v["corners_px"]) - truth, axis=1)) < 0.6


def _corners(x, z, rot):
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    wd, hd = np.array([c, s]), np.array([-s, c])
    ctr = np.array([x, z])
    return [ctr - 60 * wd + 30 * hd, ctr + 60 * wd + 30 * hd, ctr + 60 * wd - 30 * hd, ctr - 60 * wd - 30 * hd]


def test_overlay_shows_target_and_measured_outline_and_result_can_be_confirmed(client, ready):
    sid, scene = ready
    fid = add_verification_photo(client, sid, replace(scene, placards=(demo_placard(241.0, 299.5),)), seed=71)
    v = verify(client, sid, fid)["verification"]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": fid}).json()
    kinds = [p["kind"] for p in ov["primitives"]]
    assert "measured" in kinds and any(k.startswith("target") for k in kinds)
    meas = next(p for p in ov["primitives"] if p["kind"] == "measured")
    assert np.allclose(meas["corners"], v["corners_px"], atol=0.05) and meas["verification_id"] == v["id"]
    s = client.post(f"/api/sessions/{sid}/verifications/{v['id']}/confirm", json={"confirmed": True}).json()
    assert s["verifications"][0]["confirmed"]
    # the export carries the evidence, its meaning and no image
    ex = client.get(f"/api/sessions/{sid}/export").json()
    assert ex["images_included"] is False and ex["verifications"][0]["id"] == v["id"]
    assert "local" in ex["coordinates"] and "acceptance" in ex["status_semantics"].lower()
    assert ex["verifications"][0]["uncertainty"]["expanded_mm"] > 0 and ex["algorithm_version"]
    assert not any("data:image" in str(x) for x in ex.values())
    assert client.delete(f"/api/sessions/{sid}/verifications/{v['id']}").json()["verifications"] == []
    assert client.post(f"/api/sessions/{sid}/verifications/{v['id']}/confirm", json={}).status_code == 404


def test_registration_photo_is_not_accepted_as_verification(client, ready):
    sid, scene = ready
    fid = client.get(f"/api/sessions/{sid}").json()["registration"]["frames_used"][0]
    r = client.post(f"/api/sessions/{sid}/verify", json={"frame_id": fid})
    assert r.status_code == 409 and r.json()["error"]["code"] == "verification_needs_new_photo"


def test_lost_registration_and_moved_tags_give_no_numbers(client, ready):
    sid, scene = ready
    installed = replace(scene, placards=(demo_placard(240.5, 300.0),))
    # the tags were removed before the verification photo
    no_tags = replace(installed, stickers=())
    fid = add_verification_photo(client, sid, no_tags, seed=80)
    r = client.post(f"/api/sessions/{sid}/verify", json={"frame_id": fid})
    assert r.status_code == 422 and r.json()["error"]["code"] == "registration_lost" and r.json()["error"]["suggestions"]
    # one tag was moved 9 mm after capture
    moved = replace(installed, stickers=tuple(replace(s, center=(s.center[0] + 9.0, s.center[1])) if s.tag_id == 7 else s for s in installed.stickers))
    fid2 = add_verification_photo(client, sid, moved, seed=81)
    out = verify(client, sid, fid2)
    v = out["verification"]
    assert v["flags"] == ["marker_moved"] and v["observation"] == "no_evidence"
    assert v["dx_mm"] is None and v["corners_px"] == [] and "moved" in out["message"].lower()
    assert client.post(f"/api/sessions/{sid}/verifications/{v['id']}/confirm", json={}).status_code == 409


def test_missing_placard_and_manual_corners(client, ready):
    sid, scene = ready
    fid = add_verification_photo(client, sid, scene, seed=82)  # nothing installed
    out = verify(client, sid, fid)
    assert out["verification"]["observation"] == "no_evidence" and "placard_not_found" in out["verification"]["flags"]
    assert "by hand" in out["message"]
    # installed 3 mm right, 1.5 mm low; the operator taps the corners roughly (a few pixels off, any order)
    fid2 = add_verification_photo(client, sid, replace(scene, placards=(demo_placard(243.0, 298.5),)), seed=83)
    truth = np.array([px(VCAM, x, z) for x, z in _corners(243.0, 298.5, 0.0)])
    taps = (truth + np.array([[3.0, -2.0], [-2.5, 2.5], [2.0, 3.0], [-3.0, -2.0]]))[[2, 0, 3, 1]]
    snapped = verify(client, sid, fid2, corners_px=taps.tolist(), snap=True)["verification"]
    assert snapped["dx_mm"] == pytest.approx(3.0, abs=0.2) and snapped["dz_mm"] == pytest.approx(-1.5, abs=0.2)
    assert snapped["corner_provenance"] == ["measured"] * 4
    kept = verify(client, sid, fid2, corners_px=taps.tolist(), snap=False)["verification"]
    assert kept["corner_provenance"] == ["manual"] * 4
    assert np.max(np.linalg.norm(np.array(kept["corners_px"]) - (truth + np.array([[3.0, -2.0], [-2.5, 2.5], [2.0, 3.0], [-3.0, -2.0]])), axis=1)) < 0.05
    assert kept["uncertainty"]["expanded_mm"] > snapped["uncertainty"]["expanded_mm"]
    # one result per photo and placard: the newer measurement replaced the older
    assert len(client.get(f"/api/sessions/{sid}").json()["verifications"]) == 2
    r = client.post(f"/api/sessions/{sid}/verify", json={"frame_id": fid2, "corners_px": [[1, 1], [2, 2], [3, 3], [4, 4]]})
    assert r.status_code == 422


def test_quality_flags_prevent_a_within_statement(client, ready):
    sid, scene = ready
    # perfectly placed, but one corner is folded over
    fid = add_verification_photo(client, sid, replace(scene, placards=(demo_placard(240.2, 300.1, curl=(1, 12.0)),)), seed=84)
    out = verify(client, sid, fid)
    v = out["verification"]
    assert any(f.startswith("corner_lifted_or_hidden") for f in v["flags"])
    assert v["radial_mm"] < 0.5 and v["observation"] == "indeterminate"
    assert out["detail"]["observation_note"].startswith("downgraded")


def test_new_registration_discards_earlier_verification_results(client, ready):
    sid, scene = ready
    fid = add_verification_photo(client, sid, replace(scene, placards=(demo_placard(),)), seed=85)
    verify(client, sid, fid)
    assert len(client.get(f"/api/sessions/{sid}").json()["verifications"]) == 1
    first = client.get(f"/api/sessions/{sid}").json()["frames"][0]["id"]
    s = client.patch(f"/api/sessions/{sid}/frames/{first}", json={"selected": False}).json()
    assert s["registration"] is None and s["verifications"] == []


def test_deleting_a_verification_photo_takes_only_its_own_result(client, ready):
    """Found while building live placement: a record photo that is discarded must leave the registration,
    the datums and the other records alone (it used to discard them all)."""
    sid, scene = ready
    with_placard = replace(scene, placards=(demo_placard(x=240.3, z=299.8),))
    first = add_verification_photo(client, sid, with_placard, seed=86)
    keep = verify(client, sid, first)["verification"]["id"]
    second = add_verification_photo(client, sid, with_placard, seed=87)
    verify(client, sid, second)
    before = client.get(f"/api/sessions/{sid}").json()
    assert len(before["verifications"]) == 2 and before["registration"] is not None
    s = client.delete(f"/api/sessions/{sid}/frames/{second}").json()
    assert [v["id"] for v in s["verifications"]] == [keep]
    assert s["registration"] is not None and s["registration"]["computed_at"] == before["registration"]["computed_at"]
    assert s["local_frame"] is not None and s["local_frame"]["valid"] and set(s["datums"]) == {"floor", "side"}
    # the session still measures: a live frame is registered, another photo can be verified
    third = add_verification_photo(client, sid, with_placard, seed=88)
    assert verify(client, sid, third)["verification"]["observation"] == "within_tolerance_observed"
    # a photo the registration was made from is another matter: without it the registration has to be made again
    reg = next(f["id"] for f in s["frames"] if f["role"] == "registration")
    s = client.delete(f"/api/sessions/{sid}/frames/{reg}").json()
    assert s["registration"] is None and s["verifications"] == []
