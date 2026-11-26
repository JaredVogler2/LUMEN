"""A phone that stands still: the registration of one live frame is kept and only checked in the next ones.

Same panel, session and live camera as ``test_placement_flow`` (three of the four tags are in that view: 3, 7, 11).
"""

from dataclasses import replace

import pytest

import lumen.services.live as live_mod
from lumen.synth.fixtures import phone_camera
from lumen.synth.scene import Box
from tests.conftest import render_cached
from tests.integration.test_placement_flow import SKIN, SQUARE, VH, VW, hint_of, live, placed, ready  # noqa: F401  (ready is a fixture)


def shifted_camera(mm: float):
    """The live camera moved sideways by ``mm`` without turning (the picture of the panel shifts by 1.5 px per mm)."""
    return phone_camera((250.0 + mm, -640.0, 330.0), target=(245.0 + mm, 0.0, 300.0), width=VW, height=VH, roll_deg=2.0)


def arm_over(*tags):
    """Something 50 mm thick over the named tags (an arm reaching across the panel)."""
    where = {3: (95.0, 125.0), 7: (455.0, 105.0), 11: (110.0, 495.0)}
    return tuple(Box(lo=(where[t][0] - 45.0, -50.0, where[t][1] - 45.0), hi=(where[t][0] + 45.0, -3.0, where[t][1] + 45.0), albedo=SKIN) for t in tags)


def test_a_standing_phone_keeps_its_registration_and_the_reading_is_the_same(client, ready):
    sid, scene = ready
    first = live(client, sid, render_cached(placed(scene, 14.0, 9.0, 3.0), SQUARE, seed=1000).image)
    assert "held" not in first["tracking"] and sorted(first["tracking"]["used_tags"]) == [3, 7, 11]
    assert first["timing_ms"]["tags"] > 0 and first["timing_ms"]["hold"] == 0
    meta = hint_of(first)
    img = render_cached(placed(scene, 5.0, 3.0, 1.0), SQUARE, seed=1001).image
    out = live(client, sid, img, meta)
    h = out["tracking"]["held"]
    assert sorted(h["tags_seen"]) == [3, 7, 11] and h["tags_covered"] == [] and h["shift_px"] < 0.05 and h["moved_along"] is False
    # no tag was measured for this frame, and nothing was drawn afresh
    assert out["timing_ms"]["tags"] == 0 and out["timing_ms"]["overlay"] == 0 and out["timing_ms"]["hold"] > 0
    assert out["tracking"]["state"] == "tracking" and sorted(out["tracking"]["used_tags"]) == [3, 7, 11]
    assert any(p["kind"].startswith("target") for p in out["primitives"]) and "PLC-DEMO-101" in out["reports"]
    p = out["placement"]
    assert p["found"] and p["dx_mm"] == pytest.approx(5.0, abs=0.12) and p["dz_mm"] == pytest.approx(3.0, abs=0.12) and p["rotation_deg"] == pytest.approx(1.0, abs=0.08)
    # the same picture with its tags measured afresh: the same reading
    fresh = live(client, sid, img, meta, full="true")
    assert "held" not in fresh["tracking"] and fresh["timing_ms"]["tags"] > 0
    q = fresh["placement"]
    assert p["dx_mm"] == pytest.approx(q["dx_mm"], abs=0.05) and p["dz_mm"] == pytest.approx(q["dz_mm"], abs=0.05) and p["rotation_deg"] == pytest.approx(q["rotation_deg"], abs=0.03)
    assert p["observation"] == q["observation"]
    # what the check cannot see (a tenth of a pixel) is a term of the held reading's budget, and it is small
    held_term = [t for t in p["live_terms"] if t["name"].startswith("registration held")]
    assert len(held_term) == 1 and held_term[0]["bound_mm"] == pytest.approx(0.1 / p["px_per_mm"], rel=0.01)
    assert not [t for t in q["live_terms"] if t["name"].startswith("registration held")]
    assert abs(p["expanded_mm"] - q["expanded_mm"]) < 0.03
    # held frames cost a fraction of measured ones
    again = live(client, sid, render_cached(placed(scene, 0.3, -0.2, 0.1), SQUARE, seed=1002).image, hint_of(fresh))
    assert "held" in again["tracking"] and again["placement"]["in_position"]
    assert again["timing_ms"]["total"] < 0.6 * fresh["timing_ms"]["total"]


def test_an_arm_over_the_tags_does_not_change_a_held_registration(client, ready, monkeypatch):
    sid, scene = ready
    first = live(client, sid, render_cached(placed(scene, 0.4, -0.3, 0.1), SQUARE, seed=1010).image)
    meta = hint_of(first)
    # one tag covered: the two others say that nothing moved
    one = live(client, sid, render_cached(replace(placed(scene, 0.4, -0.3, 0.1), boxes=arm_over(11)), SQUARE, seed=1011).image, meta)
    h = one["tracking"]["held"]
    assert sorted(h["tags_seen"]) == [3, 7] and h["tags_covered"] == [11]
    p = one["placement"]
    assert p["dx_mm"] == pytest.approx(0.4, abs=0.12) and p["dz_mm"] == pytest.approx(-0.3, abs=0.12) and p["in_position"] and p["clean"]
    assert sorted(one["tracking"]["used_tags"]) == [3, 7, 11], "the registration is the reference's, made with all three"
    # two covered, a moment later: the one that is left has not moved, and two agreed just before
    two_img = render_cached(replace(placed(scene, 0.4, -0.3, 0.1), boxes=arm_over(3, 11)), SQUARE, seed=1012).image
    two = live(client, sid, two_img, hint_of(one))
    assert two["tracking"]["held"]["tags_seen"] == [7] and sorted(two["tracking"]["held"]["tags_covered"]) == [3, 11]
    assert two["placement"]["dx_mm"] == pytest.approx(0.4, abs=0.12) and two["placement"]["in_position"]
    # the same picture without a registration to hold: one tag, and a reading that cannot say "in position"
    fresh = live(client, sid, two_img, hint_of(one), full="true")
    assert fresh["tracking"]["used_tags"] == [7] and not fresh["placement"]["in_position"]
    assert fresh["placement"].get("expanded_mm") is None or fresh["placement"]["expanded_mm"] > 1.5 * two["placement"]["expanded_mm"]
    # one tag alone is believed only for a moment after two agreed
    first = live(client, sid, render_cached(placed(scene, 0.4, -0.3, 0.1), SQUARE, seed=1010).image, full="true")
    monkeypatch.setattr(live_mod, "HOLD_ONE_TAG_S", 0.0)
    late = live(client, sid, two_img, hint_of(first))
    assert "held" not in late["tracking"] and late["tracking"]["used_tags"] == [7]


def test_a_phone_that_was_moved_is_measured_afresh(client, ready):
    sid, scene = ready
    first = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), SQUARE, seed=1020).image)
    meta = hint_of(first)
    # 2 mm to the side: the tags are 3 px from where they were
    moved = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), shifted_camera(2.0), seed=1021).image, meta)
    assert "held" not in moved["tracking"] and moved["timing_ms"]["tags"] > 0
    p = moved["placement"]
    assert p["dx_mm"] == pytest.approx(2.0, abs=0.12) and p["dz_mm"] == pytest.approx(1.0, abs=0.12)
    # and from there on it stands again
    after = live(client, sid, render_cached(placed(scene, 1.0, 0.5, 0.2), shifted_camera(2.0), seed=1022).image, hint_of(moved))
    assert "held" in after["tracking"] and after["placement"]["dx_mm"] == pytest.approx(1.0, abs=0.12)


def test_a_stand_that_gives_a_little_moves_the_registration_along(client, ready):
    sid, scene = ready
    first = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), SQUARE, seed=1030).image)
    # 0.2 mm to the side: the whole picture shifts by 0.3 px, and the three tags agree on it
    out = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), shifted_camera(0.2), seed=1031).image, hint_of(first))
    h = out["tracking"]["held"]
    assert h["moved_along"] is True and 0.2 < h["shift_px"] < 0.4 and sorted(h["tags_seen"]) == [3, 7, 11]
    p = out["placement"]
    # the placard has not moved on the panel, and the reading says so (unmoved, the registration would put it 0.2 mm off)
    assert p["dx_mm"] == pytest.approx(2.0, abs=0.08) and p["dz_mm"] == pytest.approx(1.0, abs=0.08)
    # 0.05 mm: within what counts as standing still; used as it is
    still = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), shifted_camera(0.05), seed=1032).image, hint_of(first))
    assert still["tracking"]["held"]["moved_along"] is False and still["tracking"]["held"]["shift_px"] < 0.1
    assert still["placement"]["dx_mm"] == pytest.approx(2.0, abs=0.12)


def test_a_tag_that_was_moved_is_not_asked_and_other_changes_end_the_hold(client, ready):
    sid, scene = ready
    first = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), SQUARE, seed=1040).image)
    meta = hint_of(first)
    # tag 11 was knocked 4 mm to the side while the phone stood: its picture is no longer the reference's
    from lumen.synth.fixtures import DEMO_TAGS, demo_tags

    moved_tags = tuple(replace(t, center=(t.center[0] + 4.0, t.center[1])) if t.tag_id == 11 else t for t in demo_tags())
    assert len(moved_tags) == len(DEMO_TAGS)
    out = live(client, sid, render_cached(replace(placed(scene, 2.0, 1.0, 0.5), stickers=moved_tags), SQUARE, seed=1041).image, meta)
    h = out["tracking"]["held"]
    assert sorted(h["tags_seen"]) == [3, 7] and h["tags_covered"] == [11]
    assert out["placement"]["dx_mm"] == pytest.approx(2.0, abs=0.12), "the registration of the reference frame, in which the tag was in its place"
    # another placard is chosen: what was kept belongs to the other target
    client.put(f"/api/sessions/{sid}/placard", json={"placard_id": "PLC-DEMO-102"})
    other = live(client, sid, render_cached(placed(scene, 2.0, 1.0, 0.5), SQUARE, seed=1042).image)
    assert "held" not in other["tracking"] and other["placement"]["placard_id"] == "PLC-DEMO-102"
    # without the placement question nothing is held at all
    plain = client.post(f"/api/sessions/{sid}/track", content=__import__("tests.conftest", fromlist=["jpeg_bytes"]).jpeg_bytes(render_cached(scene, SQUARE, seed=1043).image, 85), headers={"content-type": "image/jpeg"})
    assert plain.status_code == 200 and "held" not in plain.json()["tracking"]
