"""Placard perimeter measurement against synthetic truth (prompt 11).

The plane-to-image mapping used here is the true one, so the errors below are
those of the perimeter measurement alone.
"""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from lumen.geometry.homography import apply_h, dlt_homography
from lumen.synth import Box
from lumen.synth.fixtures import demo_placard, flat_panel_scene, phone_camera
from lumen.vision.placard import PlacardConfig, SideFit, _linear_planes, complete_from_size, edge_fraction, foreign_sides, measure_placard, plain_fraction, propose_rectangle
from tests.conftest import px, render_cached

CAM = phone_camera((330.0, -700.0, 440.0), width=1600, height=1200)


def truth_H(cam):
    P = np.array([[x, z] for x in (0, 200, 400, 600) for z in (50, 250, 450, 650)], float)
    return dlt_homography(P, np.array([px(cam, x, z) for x, z in P]))


def rect(x, z, w, h, ang=0.0):
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    wd, hd = np.array([c, s]), np.array([-s, c])
    ctr = np.array([x, z])
    return np.stack([ctr - w / 2 * wd + h / 2 * hd, ctr + w / 2 * wd + h / 2 * hd, ctr + w / 2 * wd - h / 2 * hd, ctr - w / 2 * wd - h / 2 * hd])


H = truth_H(CAM)
EXPECTED = rect(240, 300, 120, 60)


def shoot(placard=None, seed=7, **scene_kw):
    scene = flat_panel_scene()
    if placard is not None:
        scene = replace(scene, placards=(placard,))
    if scene_kw:
        scene = replace(scene, **scene_kw)
    return render_cached(scene, CAM, seed=seed).image


def errors(m, pl):
    got = apply_h(np.linalg.inv(H), m.corners_ideal)
    tc = rect(pl.center[0], pl.center[1], pl.width, pl.height, pl.rotation_deg)
    return float(np.max(np.linalg.norm(got - tc, axis=1))), float(np.linalg.norm(got.mean(0) - tc.mean(0))), got


@pytest.mark.parametrize(
    "pl",
    [
        demo_placard(),
        demo_placard(243.0, 298.5),
        demo_placard(228.0, 308.0, rotation_deg=2.0),
        demo_placard(rotation_deg=-4.0),
        demo_placard(275.0, 280.0),
        demo_placard(239.0, 299.0, corner_radius=6.0),
        demo_placard(241.0, 301.0, border=0.0, text_lines=0, albedo=(0.97, 0.97, 0.96)),
    ],
    ids=["on_target", "shifted", "shifted_rotated", "rotated", "far_off", "rounded_corners", "plain_white"],
)
def test_perimeter_is_measured_wherever_the_placard_actually_is(pl):
    m = measure_placard(shoot(pl), None, H, EXPECTED)
    assert m.complete and not m.flags, m.flags
    corner_err, centre_err, got = errors(m, pl)
    # printed border, text and rounded corners do not pull the outline inward
    assert centre_err < 0.03 and corner_err < 0.25
    w = np.linalg.norm(got[1] - got[0])
    assert abs(w - pl.width) < 0.35
    ang = np.degrees(np.arctan2(*(got[1] - got[0])[::-1]))
    assert abs(ang - pl.rotation_deg) < 0.06
    assert all(s.method == "esf" and s.support > 0.9 for s in m.sides)
    # reported precision is of the order of the actual scatter (1.66 px/mm here)
    assert 0.01 < np.sqrt(np.diag(m.corners_cov).max()) / m.proposal["px_per_mm"] < 0.12


def test_both_edge_signs_are_tried_and_the_recorded_size_decides():
    pl = demo_placard(243.0, 298.5)  # red border: dark against the panel, with a bright interior 3 mm further in
    m = measure_placard(shoot(pl), None, H, EXPECTED)
    tried = {t["polarity"]: t for t in m.proposal["polarities_tried"]}
    assert set(tried) == {1, -1}
    assert tried[m.proposal["edge_polarity"]]["size_mismatch"] < 0.01
    other = tried[-m.proposal["edge_polarity"]]
    assert other["sides_ok"] < 4 or other["size_mismatch"] > 0.04  # the inner border line is not the placard


def test_no_placard_is_reported_as_not_found():
    m = measure_placard(shoot(None, seed=1), None, H, EXPECTED)
    assert not m.ok and m.reason == "no_rectangle_stands_out" and "placard_not_found" in m.flags
    assert np.isnan(m.corners_ideal).all()


def test_wrong_size_part_is_measured_and_flagged():
    pl = demo_placard(240.0, 300.0, width=104.0, height=60.0)
    m = measure_placard(shoot(pl), None, H, EXPECTED)
    assert m.complete and "size_differs_from_record" in m.flags
    _, _, got = errors(m, pl)
    assert abs(np.linalg.norm(got[1] - got[0]) - 104.0) < 0.4


def test_folded_corner_is_flagged_but_rounded_corners_are_not():
    curled = demo_placard(242.0, 300.0, curl=(1, 12.0))
    m = measure_placard(shoot(curled), None, H, EXPECTED)
    assert m.complete and "corner_lifted_or_hidden:TR" in m.flags
    assert m.corner_gap_mm[1] == pytest.approx(12.0, abs=1.5) and max(g for i, g in enumerate(m.corner_gap_mm) if i != 1) < 2.0
    # the outline is still that of the flat placard: the fold does not move it
    assert errors(m, curled)[1] < 0.05
    rounded = demo_placard(239.0, 299.0, corner_radius=6.0)
    m2 = measure_placard(shoot(rounded), None, H, EXPECTED)
    assert not any(f.startswith("corner_lifted") for f in m2.flags)
    assert all(2.5 < g < 5.0 for g in m2.corner_gap_mm)


def test_covered_corner_is_still_located_from_the_sides_and_reported_as_not_seen():
    pl = demo_placard(241.0, 300.5)
    # a block resting on the panel hides the top-right corner (finger, tape, clamp)
    img = shoot(pl, boxes=(Box((286.0, -14.0, 318.0), (318.0, 0.0, 346.0)),))
    m = measure_placard(img, None, H, EXPECTED)
    assert m.complete
    assert any(f in m.flags for f in ("corner_lifted_or_hidden:TR", "corner_not_directly_seen:TR")), m.flags
    assert errors(m, pl)[0] < 0.4  # the hidden corner is where the two sides meet


def test_glare_or_overpainting_on_an_edge_lowers_its_support():
    pl = demo_placard(240.5, 300.0)
    img = shoot(pl).copy()
    # wash out most of the bottom side (a strong reflection)
    a, b = px(CAM, 190, 268), px(CAM, 285, 268)
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), 17)
    m = measure_placard(img, None, H, EXPECTED)
    bottom = m.sides[2]
    assert (not bottom.ok) or bottom.support < 0.85 or "glare_on_edge" in bottom.flags
    assert any(f.endswith(":bottom") for f in m.flags)
    if not bottom.ok:
        assert "partial_perimeter" in m.flags and not m.complete and m.ok


def test_placard_partly_outside_the_photo_gives_a_partial_result():
    cam = phone_camera((393.0, -560.0, 320.0), target=(393.0, 0.0, 300.0), width=1600, height=1200, hfov_deg=38.0)
    Hc = truth_H(cam)
    assert px(cam, 180, 300)[0] < 0 < px(cam, 230, 300)[0]  # the left end of the placard is cut off
    pl = demo_placard()
    img = render_cached(replace(flat_panel_scene(), placards=(pl,)), cam, seed=9).image
    m = measure_placard(img, None, Hc, EXPECTED)
    left = m.sides[3]
    assert not left.ok and left.reason in ("side_outside_image", "edge_not_found")
    assert m.ok and not m.complete and "partial_perimeter" in m.flags
    got = apply_h(np.linalg.inv(Hc), m.corners_ideal[[1, 2]])
    assert np.max(np.linalg.norm(got - rect(240, 300, 120, 60)[[1, 2]], axis=1)) < 0.4


def test_operator_marked_corners_are_snapped_to_the_edges_or_kept_as_marked():
    pl = demo_placard(243.0, 298.5)
    img = shoot(pl)
    truth = apply_h(H, rect(243.0, 298.5, 120, 60))
    rough = truth + np.array([[3.0, -2.5], [-2.0, 3.0], [2.5, 2.0], [-3.0, -1.5]])
    m = measure_placard(img, None, H, EXPECTED, manual_corners_ideal=rough, snap=True)
    assert m.complete and m.provenance == "manual" and all(s.method == "esf" for s in m.sides)
    assert errors(m, pl)[0] < 0.25
    kept = measure_placard(img, None, H, EXPECTED, manual_corners_ideal=rough, snap=False)
    assert np.allclose(kept.corners_ideal, rough, atol=1e-6) and all(s.method == "manual" for s in kept.sides)
    assert np.sqrt(kept.corners_cov[0, 0]) > 10 * np.sqrt(m.corners_cov[0, 0])  # hand-marked corners are much less certain


def test_proposal_reports_rivals_and_the_search_limit():
    # two identical placards 30 mm apart vertically... the nearer pose is taken, the rival is reported
    a, b = demo_placard(240.0, 262.0), demo_placard(240.0, 338.0, name="OTHER")
    img = render_cached(replace(flat_panel_scene(), placards=(a, b)), CAM, seed=11).image
    from lumen.vision.imaging import to_gray

    prop = propose_rectangle(to_gray(img), None, H, EXPECTED)
    assert prop["found"] and prop["ambiguous"] and prop["second_score_ratio"] > 0.8
    assert abs(abs(prop["offset_mm"][1]) - 38.0) < 1.0
    m = measure_placard(img, None, H, EXPECTED)
    assert "ambiguous_proposal" in m.flags


# ---------------------------------------------------------------------------------------------------------------
# a placard in a hand (live placement): sides that are covered, and edges that are not the placard's
# ---------------------------------------------------------------------------------------------------------------
LIVE = PlacardConfig(min_side_support=0.3, wide_side_retry=False)
SKIN = (0.55, 0.42, 0.36)


def centre_error(m, pl):
    got = apply_h(np.linalg.inv(H), m.corners_ideal)
    return float(np.linalg.norm(got.mean(0) - np.array(pl.center)))


def test_a_hidden_side_is_placed_by_the_known_size_where_something_covers_it():
    pl = demo_placard(243.0, 298.5)
    # something 40 mm thick over the whole right side; its straight edge runs parallel to the side, 12 mm inside
    img = shoot(pl, boxes=(Box((291.0, -40.0, 253.0), (353.0, -0.5, 344.0), albedo=SKIN),))
    m = measure_placard(img, None, H, EXPECTED, cfg=LIVE)
    assert m.ok and not m.complete and [s.ok for s in m.sides] == [True, False, True, True], [(s.name, s.ok, s.reason) for s in m.sides]
    # the sides that were measured are the placard's own, not the printed border 3 mm inside them
    P = apply_h(np.linalg.inv(H), np.stack([m.sides[0].mid, m.sides[2].mid]))
    assert abs(abs(P[0, 1] - P[1, 1]) - 60.0) < 0.3
    placed = complete_from_size(m, H, 120.0, 60.0, image=img)
    assert placed == [1] and m.complete and m.sides[1].method == "record_size"
    assert "side_from_record_size:right" in m.flags and "corner_not_directly_seen:TR" in m.flags and "partial_perimeter" not in m.flags
    assert m.proposal["placed_side_plain"]["right"] < 0.2
    assert centre_error(m, pl) < 0.3
    # the measured width is, by construction, the record's: it says nothing about the part
    got = apply_h(np.linalg.inv(H), m.corners_ideal)
    assert np.linalg.norm(got[1] - got[0]) == pytest.approx(120.0, abs=0.05)


def test_a_side_is_not_called_hidden_where_nothing_covers_it():
    # a part 20 mm narrower than the record, its left side where the record's is: the right side is not "hidden"
    narrow = demo_placard(233.0, 298.5, width=100.0)
    img = shoot(narrow)
    m = measure_placard(img, None, H, EXPECTED, cfg=LIVE)
    ok = [s.ok for s in m.sides]
    assert sum(ok) == 3 and not m.complete, [(s.name, s.ok, s.reason) for s in m.sides]
    missing = ok.index(False)
    before = [s.ok for s in m.sides]
    assert complete_from_size(m, H, 120.0, 60.0, image=img) == []
    assert not m.complete and [s.ok for s in m.sides] == before
    assert max(v for v in m.proposal["placed_side_plain"].values() if v is not None) > 0.8
    # without the look at the picture the assumption would have gone through, 10 mm off
    assert complete_from_size(m, H, 120.0, 60.0) == [missing] and centre_error(m, narrow) > 8.0


def test_which_hidden_sides_can_be_placed():
    pl = demo_placard(243.0, 298.5)
    whole = measure_placard(shoot(pl), None, H, EXPECTED, cfg=LIVE)
    assert whole.complete and complete_from_size(whole, H, 120.0, 60.0) == []

    def without(*idx):
        m = measure_placard(shoot(pl), None, H, EXPECTED, cfg=LIVE)
        for k in idx:
            m.sides[k].ok = False
        return m

    # one side: from the three others.  Two neighbours: only when asked (a placard seen whole a moment ago).
    assert complete_from_size(without(3), H, 120.0, 60.0) == [3]
    assert complete_from_size(without(1, 2), H, 120.0, 60.0) == []
    m = without(1, 2)
    assert complete_from_size(m, H, 120.0, 60.0, allow_two=True) == [1, 2] and centre_error(m, pl) < 0.3
    # two opposite sides, or three: nothing says where the placard is
    assert complete_from_size(without(1, 3), H, 120.0, 60.0, allow_two=True) == []
    assert complete_from_size(without(0, 1, 2), H, 120.0, 60.0, allow_two=True) == []


def test_a_placard_is_followed_from_where_it_was_with_the_sign_of_its_edges():
    pl = demo_placard(243.0, 298.5, rotation_deg=1.0)
    img = shoot(pl)
    first = measure_placard(img, None, H, EXPECTED, cfg=LIVE)
    sign = first.proposal["edge_polarity"]
    for off in ((6.0, 0.0, 0.0), (-6.0, 0.0, 0.0), (0.0, 5.0, 0.0), (4.0, -3.0, 1.0)):
        seed = apply_h(H, rect(243.0 + off[0], 298.5 + off[1], 120, 60, 1.0 + off[2]))
        m = measure_placard(img, None, H, EXPECTED, cfg=LIVE, seed_corners_ideal=seed, seed_polarity=sign)
        assert m.complete and centre_error(m, pl) < 0.05, (off, [(s.name, s.ok, s.reason) for s in m.sides])
        # only the placard's own sign is tried: print inside it has the other
        assert [t["polarity"] for t in m.proposal["polarities_tried"]] == [sign] and m.proposal.get("seeded")
    # a seed far from the placard finds nothing, and nothing is taken from the seed
    lost = measure_placard(img, None, H, EXPECTED, cfg=LIVE, seed_corners_ideal=apply_h(H, rect(283.0, 330.0, 120, 60)), seed_polarity=sign)
    assert not lost.complete


def canvas(block=None, shadow=None):
    """A white placard (x 100..300, y 80..200) on a grey panel, optionally with a skin-coloured block and a shadow."""
    img = np.full((300, 420, 3), 150, np.uint8)
    img[80:200, 100:300] = (232, 232, 232)
    if block is not None:
        x0, x1, y0, y1 = block
        img[y0:y1, x0:x1] = (95, 115, 175)
    if shadow is not None:
        x0, x1 = shadow
        img[:, x0:x1] = (img[:, x0:x1].astype(np.float32) * 0.74).astype(np.uint8)  # about half the light
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def side(index, mid, n, length):
    return SideFit(index=index, ok=True, n=np.array(n, float), mid=np.array(mid, float), length_px=float(length), method="esf", support=1.0)


def test_an_edge_of_what_covers_the_placard_is_not_one_of_its_sides():
    top, bottom, left = side(0, (170, 80), (0, -1), 110), side(2, (170, 200), (0, 1), 110), side(3, (100, 140), (-1, 0), 90)
    true_right = side(1, (300, 140), (1, 0), 90)
    hand_edge = side(1, (240, 140), (1, 0), 90)  # the block's edge: placard inside, block outside
    covered = canvas(block=(240, 380, 60, 220))
    assert foreign_sides(canvas(), None, [top, true_right, bottom, left]) == []
    assert foreign_sides(covered, None, [top, hand_edge, bottom, left]) == [1]
    # shade does not make an edge foreign: the hue outside it is still the panel's (here the light is halved)
    assert foreign_sides(canvas(shadow=(270, 420)), None, [top, true_right, bottom, left]) == []
    # the ends of a block of print inside the placard: the placard's own colour on both sides, darker print inside
    printed = canvas()
    printed[100:180, 130:200] = (60, 60, 60)
    print_edge = side(3, (130, 140), (-1, 0), 70)
    assert foreign_sides(printed, None, [top, true_right, bottom, print_edge]) == [3]
    # two sides that disagree, and no third to say which is the odd one: both
    assert foreign_sides(covered, None, [hand_edge, left]) == [1, 3]
    assert foreign_sides(canvas(), None, [true_right, left]) == []
    # a side placed by the size has no surroundings of its own and is left out of the comparison
    placed = SideFit(index=1, ok=True, n=np.array([1.0, 0.0]), mid=np.array([300.0, 140.0]), length_px=90.0, method="record_size")
    assert foreign_sides(covered, None, [top, placed, bottom, left]) == []


def test_how_much_of_a_placed_side_lies_in_plain_surroundings():
    top, bottom, left = side(0, (170, 80), (0, -1), 110), side(2, (170, 200), (0, 1), 110), side(3, (100, 140), (-1, 0), 90)
    measured = [top, bottom, left]

    def placed_at(x):
        return SideFit(index=1, ok=True, n=np.array([1.0, 0.0]), mid=np.array([float(x), 140.0]), length_px=100.0, method="record_size")

    covered = _linear_planes(canvas(block=(240, 380, 60, 220)))
    bare = _linear_planes(canvas())
    assert plain_fraction(covered, None, placed_at(300), measured) < 0.1  # under the block: something covers it
    assert plain_fraction(bare, None, placed_at(340), measured) > 0.9  # in bare panel: the placard does not reach that far
    assert plain_fraction(bare, None, placed_at(200), measured) > 0.9  # in bare placard: the placard goes on beyond it
    assert plain_fraction(bare, None, placed_at(300), measured) < 0.1  # on a visible edge: neither (it would have been measured)
    assert plain_fraction(bare, None, placed_at(300), []) is None  # nothing to compare with
    # and whether the placard's own edge is to be seen there: then the side is not hidden, only not found
    assert edge_fraction(bare, None, placed_at(300), measured) > 0.9
    assert edge_fraction(covered, None, placed_at(300), measured) < 0.1
    assert edge_fraction(bare, None, placed_at(340), measured) < 0.1
