"""Assisted placement through the API: the placard is measured in live frames while it is being positioned.

The session is registered on photos of the bare panel.  Live frames are video-sized (1280x720), come from a
viewpoint that was not used before, and show the placard at known places on its way to the target
(PLC-DEMO-101: centre x 240, z 300, 120 x 60 mm, +-2.0 mm, +-1 degree).
"""

import json
from dataclasses import replace
from urllib.parse import quote

import numpy as np
import pytest

from lumen.synth.fixtures import demo_placard, flat_panel_scene, phone_camera, sweep_stations
from lumen.synth.scene import Box
from tests.conftest import calibration_for, jpeg_bytes, px, render_cached

W, H = 1600, 1200
VW, VH = 1280, 720
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}
SQUARE = phone_camera((250.0, -640.0, 330.0), target=(245.0, 0.0, 300.0), width=VW, height=VH, roll_deg=2.0)
OBLIQUE = phone_camera((600.0, -560.0, 330.0), target=(245.0, 0.0, 300.0), width=VW, height=VH, roll_deg=-1.0)


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


def live(client, sid, img, meta=None, **params):
    headers = {"content-type": "image/jpeg"}
    if meta:
        headers["x-lumen-meta"] = quote(json.dumps(meta))
    r = client.post(f"/api/sessions/{sid}/track", params={"placement": "true", **params}, content=jpeg_bytes(img, 85), headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def placed(scene, dx=0.0, dz=0.0, rot=0.0, **kw):
    return replace(scene, placards=(demo_placard(x=240.0 + dx, z=300.0 + dz, rotation_deg=rot, **kw),))


def test_a_placard_brought_to_its_target_is_followed_and_called_in_position_when_it_is(client, ready):
    sid, scene = ready
    steps = [(30.0, -18.0, 6.0), (12.0, -7.0, 3.0), (5.0, -3.0, 1.5), (2.2, -1.0, 0.6), (0.6, -0.3, 0.2), (0.2, 0.1, 0.0)]
    meta: dict = {}
    seen = []
    for k, (dx, dz, rot) in enumerate(steps):
        out = live(client, sid, render_cached(placed(scene, dx, dz, rot), SQUARE, seed=200 + k).image, meta)
        p, tr = out["placement"], out["tracking"]
        assert tr["state"] == "tracking" and p["found"] and p["complete"], (k, p.get("message"))
        # the numbers are those of the scene, with their signs
        assert p["dx_mm"] == pytest.approx(dx, abs=0.12) and p["dz_mm"] == pytest.approx(dz, abs=0.12) and p["rotation_deg"] == pytest.approx(rot, abs=0.08), (k, p["dx_mm"], p["dz_mm"], p["rotation_deg"])
        assert p["move"]["x_mm"] == pytest.approx(-p["dx_mm"]) and p["move"]["z_mm"] == pytest.approx(-p["dz_mm"])
        # the arrow on the screen points from the placard to the target
        want = px(SQUARE, 240.0, 300.0) - px(SQUARE, 240.0 + dx, 300.0 + dz)
        got = np.array(p["move"]["px"])
        assert np.linalg.norm(got - want) < 1.0, (k, got, want)
        assert p["tolerance_mm"] == 2.0 and p["tolerance_rotation_deg"] == pytest.approx(1.0) and 0.6 < p["expanded_mm"] < 1.5 and p["reachable"] is True
        assert p["clean"] and not p["hands"] and not [f for f in p["flags"] if f != "registration_preview_only"]
        seen.append((p["observation"], p["in_position"], p["method"], tr["edge_shift"]["source"], out["timing_ms"]["total"]))
        # what the client hands back with the next frame
        meta = {"placement_hint": {"corners_plane": p["corners_plane"], "polarity": p["polarity"]}, "edge_shift": tr["edge_shift"]}
    obs = [s[0] for s in seen]
    assert obs[:3] == ["outside_tolerance_observed"] * 3
    assert obs[3] == "indeterminate", "2.4 mm off with +-1 mm of uncertainty: neither inside nor outside +-2 mm"
    assert obs[4:] == ["within_tolerance_observed"] * 2 and [s[1] for s in seen] == [False] * 4 + [True] * 2
    # the first frame is searched and measures the stream's edge shift; later frames follow and reuse it
    assert seen[0][2] == "searched" and seen[0][3] == "this_frame"
    assert [s[2] for s in seen[2:]] == ["followed"] * 4 and all(s[3] == "stream" for s in seen[1:])
    assert np.median([s[4] for s in seen[2:]]) < 0.6 * seen[0][4], "a followed frame costs much less than the first"
    # guidance only: nothing was stored
    s = client.get(f"/api/sessions/{sid}").json()
    assert len(s["frames"]) == 5 and not s.get("verifications")


def test_followed_and_searched_give_the_same_reading(client, ready):
    sid, scene = ready
    img = render_cached(placed(scene, 3.0, 2.0, -1.2), SQUARE, seed=230).image
    a = live(client, sid, img, full="true")["placement"]
    b = live(client, sid, img, {"placement_hint": {"corners_plane": a["corners_plane"], "polarity": a["polarity"]}}, full="true")["placement"]
    assert a["method"] == "searched" and b["method"] == "followed"
    for key in ("dx_mm", "dz_mm"):
        assert b[key] == pytest.approx(a[key], abs=0.03)
    assert b["rotation_deg"] == pytest.approx(a["rotation_deg"], abs=0.02) and b["observation"] == a["observation"]
    # a hint that points at nothing is not believed: the placard is searched for and found where it is
    stale = [[x + 60.0, y] for x, y in a["corners_plane"]]
    c = live(client, sid, img, {"placement_hint": {"corners_plane": stale, "polarity": a["polarity"]}}, full="true")["placement"]
    assert c["method"] == "searched" and c["dx_mm"] == pytest.approx(a["dx_mm"], abs=0.03)
    # and a malformed one is ignored
    d = live(client, sid, img, {"placement_hint": {"corners_plane": [[1, 2], [3, None]], "polarity": "x"}, "edge_shift": {"delta_px": 9, "sigma_px": -1}}, full="true")
    assert d["placement"]["method"] == "searched" and d["tracking"]["edge_shift"]["source"] == "this_frame"


def test_a_hand_over_a_corner_does_not_stop_the_reading_but_the_result_is_not_clean(client, ready):
    sid, scene = ready
    # something 40 mm thick over the top-right corner: 10 mm of the top side and 15 mm of the right side are covered
    hand = Box(lo=(290.0, -40.0, 315.0), hi=(335.0, -0.5, 348.0), albedo=(0.55, 0.42, 0.36))
    sc = replace(placed(scene, 0.4, -0.3, 0.1), boxes=(hand,))
    p = live(client, sid, render_cached(sc, SQUARE, seed=240).image)["placement"]
    assert p["found"] and p["complete"], p.get("message")
    assert p["dx_mm"] == pytest.approx(0.4, abs=0.25) and p["dz_mm"] == pytest.approx(-0.3, abs=0.25)
    assert p["hands"] and not p["clean"]
    assert any(f.startswith(("corner_not_directly_seen:TR", "edge_partly_hidden", "corner_lifted_or_hidden:TR")) for f in p["flags"]), p["flags"]
    assert p["in_position"] and p["message"] == "In position. Let go to record."


def test_a_placard_held_off_the_surface_is_not_judged(client, ready):
    sid, scene = ready
    # a thin sheet the size of the placard, 8 mm in front of the panel, exactly over the target
    sheet = Box(lo=(180.0, -8.2, 270.0), hi=(300.0, -8.0, 330.0), albedo=(0.93, 0.93, 0.90))
    p = live(client, sid, render_cached(replace(scene, boxes=(sheet,)), SQUARE, seed=250).image)["placement"]
    assert p["found"] and p["complete"], p.get("message")
    # 8 mm nearer to a camera 640 mm away: it measures about 1.3 % too large
    assert min(p["size_error_mm"]) > 0.5 and max(p["size_error_mm"]) == pytest.approx(1.5, abs=0.6), p["size_error_mm"]
    assert "not_flat_on_surface" in p["flags"] and not p["in_position"] and p["observation"] != "within_tolerance_observed"
    assert "not lying on the surface" in p["message"]


def test_a_small_height_that_does_not_show_in_the_size_is_covered_by_the_uncertainty(client, ready):
    sid, scene = ready
    on = live(client, sid, render_cached(placed(scene, 0.0, 0.0, 0.0), OBLIQUE, seed=260).image)["placement"]
    sq = live(client, sid, render_cached(placed(scene, 0.0, 0.0, 0.0), SQUARE, seed=261).image)["placement"]
    assert 25.0 < on["view_angle_deg"] < 40.0 and sq["view_angle_deg"] < 8.0
    # the same placard on the surface: the oblique view states more uncertainty, for the height it could not exclude
    assert on["expanded_mm"] > sq["expanded_mm"] + 0.5 and on["lift_bound_mm"] > 1.5
    # 2 mm off the surface, seen about 30 degrees off the normal: displaced by about a millimetre in the picture
    sheet = Box(lo=(180.0, -2.2, 270.0), hi=(300.0, -2.0, 330.0), albedo=(0.93, 0.93, 0.90))
    p = live(client, sid, render_cached(replace(scene, boxes=(sheet,)), OBLIQUE, seed=262).image)["placement"]
    assert p["found"] and p["complete"], p.get("message")
    assert 0.5 < p["radial_mm"] < 2.0, "the apparent displacement of a sheet that is truly on target"
    assert p["radial_mm"] <= p["expanded_mm"], (p["radial_mm"], p["expanded_mm"])
    assert not p["in_position"]
    # ... and it says why nothing is judged: the sheet measures too large here (seen from the side its shadow joins
    # its outline), or, where that does not show, the view is too oblique for the tolerance
    assert "not_flat_on_surface" in p["flags"] or p["reachable"] is False
    assert ("not lying on the surface" in p["message"]) or ("square to the placard" in p["message"]), p["message"]


def test_no_placard_in_view_and_no_target(client, ready):
    sid, scene = ready
    p = live(client, sid, render_cached(scene, SQUARE, seed=270).image)["placement"]
    assert not p["found"] and p["state"] == "searching" and "dashed outline" in p["message"] and len(p["target_px"]) == 4
    # without the placement flag the answer is the plain live overlay
    r = client.post(f"/api/sessions/{sid}/track", content=jpeg_bytes(render_cached(scene, SQUARE, seed=270).image, 85), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200 and "placement" not in r.json()
    # no datums, no target, no reading
    client.delete(f"/api/sessions/{sid}/datums/floor")
    q = live(client, sid, render_cached(placed(scene, 1.0, 1.0), SQUARE, seed=271).image)["placement"]
    assert q["state"] in ("no_target", "blocked") and not q["found"] and not q["in_position"]


def test_what_is_said_when_nothing_can_be_judged():
    from lumen.services.placement import _message

    base = {"complete": True, "flags": [], "radial_mm": 0.4, "rotation_deg": 0.1, "position_observation": "within", "rotation_observation": "within"}
    assert "no tolerance" in _message({**base, "tolerance_mm": None, "expanded_mm": 1.0}, "")
    coarse = {**base, "tolerance_mm": 2.0, "expanded_mm": 2.4, "reachable": False, "view_angle_deg": 3.0}
    assert "too coarse for ±2.0 mm" in _message(coarse, "") and "Move closer" in _message(coarse, "")
    assert "square to the placard" in _message({**coarse, "view_angle_deg": 31.0}, "")
    # where the budget says what limits the reading, that is the advice given
    assert _message({**coarse, "limit_hint": "Largest contribution: uncharacterised lens distortion. Calibrate."}, "").endswith("lens distortion. Calibrate.")
    # numbers inside the tolerance from an outline that is not trusted are not called "close"
    assert "not clear enough" in _message({**base, "tolerance_mm": 2.0, "expanded_mm": 1.0, "reachable": True, "doubtful": ["glare_on_edge:top"]}, "")
    assert _message({**base, "tolerance_mm": 2.0, "expanded_mm": 1.0, "reachable": True, "in_position": True, "hands": False}, "") == "In position."
    assert _message({**base, "tolerance_mm": 2.0, "expanded_mm": 1.0, "reachable": True, "position_observation": "outside", "radial_mm": 5.83, "rotation_observation": "outside", "rotation_deg": -1.5}, "") == "Move 5.8 mm, turn 1.5°."
    assert "uncover more of it" in _message({"complete": False, "flags": []}, "")
    assert _message({"complete": False, "flags": [], "sides": ["measured", "hidden", "measured", "hidden"]}, "") == "The right and left sides are both covered: uncover one of them."


def test_without_a_lens_calibration_the_reading_says_what_takes_the_tolerance(client):
    """An ordinary phone session: no calibration profile.  The allowance for an uncharacterised lens (it acts on
    the datum lines of the registration photos) takes most of a +-2 mm tolerance.  The reading must say so
    instead of leaving the operator at "almost" for no visible reason."""
    scene = flat_panel_scene()
    cams = [phone_camera(e, width=W, height=H, roll_deg=r) for e, r in sweep_stations(5, distance=760, height=430, span_deg=50)]
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    for i, c in enumerate(cams):
        assert client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render_cached(scene, c, seed=i).image), headers={"content-type": "image/jpeg"}).status_code == 200
    client.post(f"/api/sessions/{sid}/register")
    client.post(f"/api/sessions/{sid}/datums/propose")
    client.post(f"/api/sessions/{sid}/datums/confirm")
    p = live(client, sid, render_cached(placed(scene, 0.4, -0.3, 0.1), SQUARE, seed=280).image)["placement"]
    assert p["found"] and p["complete"] and p["radial_mm"] == pytest.approx(0.5, abs=0.15)
    # half a millimetre off, and still not "in position": the uncertainty leaves a fraction of a millimetre
    assert 1.5 < p["expanded_mm"] < 2.0 and p["reachable"] is True and not p["in_position"] and p["observation"] == "indeterminate"
    assert p["tight"] is True and p["room_mm"] == pytest.approx(2.0 - p["expanded_mm"], abs=1e-3) and 0.0 < p["room_mm"] < 0.5
    assert "lens distortion" in p["limit_hint"] and "calibration" in p["limit_hint"]
    # with the calibration (the fixture of the other tests) the same placard is in position, with room to spare
    assert p["status"] == "research_preview"


# --------------------------------------------------------------------------- hands on the placard
SKIN = (0.55, 0.42, 0.36)


def hand(x0, x1, z0, z1, dx=0.0, dz=0.0):
    """Something 40 mm thick lying over the panel between the given plane coordinates (it moves with the placard)."""
    return Box(lo=(x0 + dx, -40.0, z0 + dz), hi=(x1 + dx, -0.5, z1 + dz), albedo=SKIN)


def hint_of(out):
    p = out["placement"]
    meta = {"edge_shift": out["tracking"]["edge_shift"]}
    if p.get("complete"):
        meta["placement_hint"] = {"corners_plane": p["corners_plane"], "polarity": p["polarity"]}
    return meta


def test_a_side_under_a_hand_is_placed_by_the_placards_size_and_the_reading_goes_on(client, ready):
    sid, scene = ready
    clean = live(client, sid, render_cached(placed(scene, 0.4, -0.3, 0.1), SQUARE, seed=700).image)["placement"]
    # a hand over the whole right side (its straight edge 12 mm inside the placard, parallel to the side)
    sc = replace(placed(scene, 0.4, -0.3, 0.1), boxes=(hand(288, 350, 255, 345),))
    p = live(client, sid, render_cached(sc, SQUARE, seed=701).image)["placement"]
    assert p["found"] and p["complete"], p.get("message")
    assert p["sides"] == ["measured", "record_size", "measured", "measured"]
    assert "side_from_record_size:right" in p["flags"] and "corner_not_directly_seen:TR" in p["flags"]
    # the hand's edge was not taken for the side: the placard is where it is
    assert p["dx_mm"] == pytest.approx(0.4, abs=0.25) and p["dz_mm"] == pytest.approx(-0.3, abs=0.15) and p["rotation_deg"] == pytest.approx(0.1, abs=0.1)
    # guidance goes on, the record waits: in position, not clean
    assert p["hands"] and not p["clean"] and p["in_position"] and p["message"] == "In position. Let go to record."
    # the assumption costs uncertainty: the part may not have exactly the record's width, and its size was checked in one direction only
    assert p["expanded_mm"] > clean["expanded_mm"] + 0.2 and clean["sides"] == ["measured"] * 4 and clean["clean"]
    assert len(p["size_error_mm"]) == 2 and len(clean["size_error_mm"]) == 4


def test_a_placard_is_followed_while_hands_come_and_go(client, ready):
    sid, scene = ready
    steps = [
        ("whole", 14.0, 9.0, 3.0, ()),
        ("right side covered", 9.0, 6.0, 2.0, (hand(288, 350, 255, 345, 9.0, 6.0),)),
        ("right side and part of the bottom covered", 5.0, 3.0, 1.0, (hand(288, 350, 255, 345, 5.0, 3.0), hand(240, 310, 240, 285, 5.0, 3.0))),
        ("right side covered, at the target", 0.3, -0.2, 0.1, (hand(288, 350, 255, 345, 0.3, -0.2),)),
        ("right side and bottom covered, at the target", 0.3, -0.2, 0.1, (hand(288, 350, 255, 345, 0.3, -0.2), hand(170, 310, 240, 282, 0.3, -0.2))),
        ("hands off", 0.3, -0.2, 0.1, ()),
    ]
    meta: dict = {}
    seen = {}
    for k, (name, dx, dz, rot, boxes) in enumerate(steps):
        out = live(client, sid, render_cached(replace(placed(scene, dx, dz, rot), boxes=tuple(boxes)), SQUARE, seed=710 + k).image, meta)
        p = out["placement"]
        assert p["found"] and p["complete"], (name, p.get("message"))
        # wherever the hands are, the numbers are those of the placard
        assert p["dx_mm"] == pytest.approx(dx, abs=0.3) and p["dz_mm"] == pytest.approx(dz, abs=0.3) and p["rotation_deg"] == pytest.approx(rot, abs=0.15), (name, p["dx_mm"], p["dz_mm"], p["rotation_deg"])
        seen[name] = p
        meta = hint_of(out)
    assert seen["whole"]["clean"] and not seen["whole"]["hands"]
    for name in ("right side covered", "right side and part of the bottom covered", "right side covered, at the target"):
        assert seen[name]["sides"][1] == "record_size" and seen[name]["hands"] and not seen[name]["clean"], name
    assert seen["right side covered, at the target"]["in_position"]
    # two neighbouring sides covered: the placard is still located (it was whole a moment ago), but nothing says
    # whether it lies on the surface, so it is not called in position
    two = seen["right side and bottom covered, at the target"]
    assert two["method"] == "followed" and two["sides"] == ["measured", "record_size", "record_size", "measured"]
    assert "size_not_seen" in two["flags"] and not two["in_position"] and two["observation"] == "indeterminate"
    assert two["message"] == "Too much of the placard is covered to judge its position: uncover one more side."
    off = seen["hands off"]
    assert off["method"] == "followed" and off["clean"] and off["in_position"] and off["sides"] == ["measured"] * 4


def test_what_cannot_be_located_under_hands_is_not_guessed(client, ready):
    sid, scene = ready
    # both short sides gripped, ends and all: nothing says where the placard is along its length
    grip = (hand(130, 192, 255, 345, 6.0, 4.0), hand(288, 350, 255, 345, 6.0, 4.0))
    p = live(client, sid, render_cached(replace(placed(scene, 6.0, 4.0, 1.5), boxes=grip), SQUARE, seed=720).image)["placement"]
    assert not (p["found"] and p.get("complete")) and p.get("dx_mm") is None and not p["in_position"]
    assert "both covered: uncover one of them" in p["message"] or "dashed outline" in p["message"]
    # the same grip with the corners left free (the hands on the middle third of each short side): located
    mid = (hand(140, 190, 290, 310, 6.0, 4.0), hand(290, 345, 290, 310, 6.0, 4.0))
    q = live(client, sid, render_cached(replace(placed(scene, 6.0, 4.0, 1.5), boxes=mid), SQUARE, seed=721).image)["placement"]
    assert q["found"] and q["complete"] and q["dx_mm"] == pytest.approx(6.0, abs=0.3) and q["dz_mm"] == pytest.approx(4.0, abs=0.3) and q["hands"]
    # a palm over the lower half: the top and both ends are there, the bottom is placed by the size
    palm = (hand(170, 310, 230, 298),)
    r = live(client, sid, render_cached(replace(placed(scene, 0.4, -0.3, 0.1), boxes=palm), SQUARE, seed=722).image)["placement"]
    assert r["found"] and r["complete"] and r["sides"][2] == "record_size", r.get("message")
    assert r["dx_mm"] == pytest.approx(0.4, abs=0.3) and r["dz_mm"] == pytest.approx(-0.3, abs=0.3)


def test_a_part_of_another_size_is_not_passed_off_as_a_covered_placard(client, ready):
    sid, scene = ready
    # 100 mm wide instead of 120, its left side where the target's is: the right side is 20 mm short, with nothing over it
    narrow = replace(scene, placards=(demo_placard(x=230.0, z=300.0, width=100.0),))
    p = live(client, sid, render_cached(narrow, SQUARE, seed=730).image)["placement"]
    assert not p["in_position"] and not p.get("clean")
    assert not (p["found"] and p.get("complete")) or "not_flat_on_surface" in p["flags"] or "size_differs_from_record" in " ".join(p["flags"]), p
    assert "right part" in p["message"] or "another part" in p["message"] or "uncover" in p["message"], p["message"]
    assert "side_from_record_size" not in " ".join(p.get("flags", []))



def test_a_side_missed_after_a_jump_is_measured_not_called_covered(client, ready):
    """Between two frames the placard moved further than a side is followed (a film cut in the browser test, a
    quick hand in a workshop).  The side that was not found is plainly visible: it must come out measured."""
    sid, scene = ready
    first = live(client, sid, render_cached(placed(scene, 5.0, -3.0, 1.2), SQUARE, seed=740).image)
    assert first["placement"]["clean"]
    out = live(client, sid, render_cached(placed(scene, 0.3, 0.2, 0.1), SQUARE, seed=741).image, hint_of(first))
    p = out["placement"]
    assert p["found"] and p["complete"] and p["sides"] == ["measured"] * 4, (p.get("sides"), p.get("flags"))
    assert p["clean"] and not p["hands"] and p["in_position"]
    assert p["dx_mm"] == pytest.approx(0.3, abs=0.12) and p["dz_mm"] == pytest.approx(0.2, abs=0.12)
