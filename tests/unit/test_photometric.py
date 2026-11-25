"""Experimental photometric (phone torch) module against synthetic photos (prompt 07).

All photos come from the independent ray caster in ``lumen.synth`` (near-field
torch with inverse-square fall-off, incidence, beam profile, ambient light,
gloss, exposure) and are cached on disk.  Unless a test says otherwise the
camera poses are the renderer's; ``test_end_to_end_with_poses_from_the_tag_pipeline``
goes through tag detection and bundle adjustment instead.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from lumen.geometry.camera import CameraModel
from lumen.geometry.lines import Line2D
from lumen.geometry.se3 import Pose
from lumen.synth import Box, LightSpec, Placard
from lumen.synth.fixtures import curved_panel_scene, flat_panel_scene, phone_camera, sweep_stations
from lumen.vision import photometric as ph
from lumen.vision.markers import detect_tags
from lumen.vision.multiview import MVView, solve_multiview
from lumen.vision.photometric import (
    EVIDENCE_LABEL, EdgeHint, FlashView, PhotometricConfig, PhotometricResult, analyse_flash_sweep, combine_with_silhouette, tag_exclusion_polygons,
)
from lumen.vision.planar import TagObs
from tests.conftest import render_cached

W, H = 1600, 1200
#: scene face frame (x = scene X, y = scene Z, z out of the face) -> scene world frame
T_WORLD_FACE = Pose(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), np.zeros(3))
Z_MID = 375.0
LIGHT = dict(flash=True, flash_strength=0.9)


def stations(idx=(1, 3, 4, 5), **kw):
    """Phone positions on an arc in front of the rounded edge (auto exposure on)."""
    kw.setdefault("auto_exposure", True)
    st = sweep_stations(7, distance=760, height=430, span_deg=62.0, target=(230.0, 0.0, 300.0))
    return [phone_camera(st[i][0], target=(230.0, 0.0, 300.0), width=W, height=H, roll_deg=st[i][1], **kw) for i in idx]


def edge_scene(radius=35.0, ambient=0.25, light_kw=None, **kw):
    return curved_panel_scene(radius=radius, light=LightSpec(ambient=ambient, **{**LIGHT, **(light_kw or {})}), **kw)


def model(c):
    return CameraModel(c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.dist)


def truth_pose(c):
    return Pose(np.asarray(c.R_wc), np.asarray(c.C)).inv() @ T_WORLD_FACE


def views_of(scene, cams, torch="on", seed0=0):
    return [FlashView(render_cached(scene, c, seed=seed0 + i).image, model(c), truth_pose(c), torch, None, f"{torch}{i}", c.blur_sigma) for i, c in enumerate(cams)]


def with_partners(scene, cams, jitter_mm=4.0):
    """Torch-on photos followed by flash-off partners taken a few millimetres away."""
    on = views_of(scene, cams)
    off_scene = replace(scene, light=replace(scene.light, flash=False))
    off_cams = []
    for i, c in enumerate(cams):
        v = np.random.default_rng(100 + i).normal(size=3)
        off_cams.append(replace(c, C=tuple(float(x) for x in np.asarray(c.C) + v / np.linalg.norm(v) * jitter_mm)))
    off = views_of(off_scene, off_cams, torch="off", seed0=50)
    for i in range(len(on)):
        on[i].pair, off[i].pair = len(on) + i, i
    return on + off


def tag_polygons(scene):
    out = []
    for s in scene.stickers:
        h = 0.5 * s.size
        a = np.deg2rad(s.rotation_deg)
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(np.array([[-h, h], [h, h], [h, -h], [-h, -h]]) @ Rm.T + np.asarray(s.center))
    return tag_exclusion_polygons(out)


def hint(x=6.0, tilt_deg=2.0, radius=None):
    """A hint that is deliberately a little off (6 mm, 2 degrees)."""
    a = np.deg2rad(tilt_deg)
    return EdgeHint(point=(x, Z_MID), direction=(np.sin(a), np.cos(a)), inward=(1.0, 0.0), radius_mm=radius, extent_mm=(-225.0, 225.0))


def tangent_errors(res, source=None):
    """(offset error mm at mid height, direction error deg) of a tangent line whose truth is x = 0."""
    if source is None:
        n, d = res.tangent.n, res.tangent.d
    else:
        n, d = np.asarray(source["n"]), source["d"]
    if n[0] < 0:
        n, d = -n, -d
    return float(d - n @ np.array([0.0, Z_MID])), float(np.degrees(np.arctan2(n[1], n[0])))


@pytest.fixture(scope="module")
def plain():
    scene = edge_scene()
    cams = stations()
    return scene, cams, analyse_flash_sweep(views_of(scene, cams), hint(), exclude_polygons=tag_polygons(scene))


# --------------------------------------------------------------------------
# the forward model is the renderer's physics
# --------------------------------------------------------------------------
def test_forward_model_matches_the_renderer():
    light = LightSpec(ambient=0.25, flash=True, flash_strength=0.9, flash_offset_cam=(-12.0, 9.0, 0.0), flash_beam_exponent=1.0)
    scene = curved_panel_scene(radius=35.0, light=light)
    cam = stations((3,))[0]
    img = render_cached(scene, cam, seed=3).image
    cfg = PhotometricConfig()
    grid = ph._Grid(np.array([0.0, Z_MID]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), np.arange(-22.0, 50.0, 1.0), np.arange(-200.0, 201.0, 25.0), 1.0, 25.0)
    T = truth_pose(cam)
    cell = ph._sample(ph._prepare_image(img, cfg.saturation_code, model(cam), T, grid), model(cam), T, grid, cfg)
    G = ph._make_geom(grid, [T], [ph._ALONE], np.zeros((1,) + grid.shape))
    th = np.zeros(ph._N_SHARED)
    th[[ph._I_LOGR, ph._I_M, ph._I_RHO, ph._I_OX, ph._I_OY, ph._I_GAMMA, ph._I_LOGSH]] = [np.log(35.0), 1.0, 0.25 / 0.9, -12.0, 9.0, 1.0, np.log(40.0)]
    pred, aux = ph._predict(th, np.zeros(1), G, aux=True, blur=False)
    ok = cell["inside"] & aux["valid"][0] & (aux["ndv"][0] > 0.4)
    rel = cell["mean"][ok] / pred[0][ok]
    rel = rel / np.median(rel) - 1.0
    assert ok.sum() > 800 and (aux["e"][0][ok] < 0).sum() > 150, "flat face and bend are both covered"
    # one gain explains flat face and bend alike: geometry, inverse square, incidence, beam and ambient term agree
    assert np.sqrt(np.mean(rel**2)) < 0.006
    bend = aux["e"][0][ok] < -3.0
    assert abs(np.mean(rel[bend]) - np.mean(rel[~bend])) < 0.003
    # a torch on the wrong side of the lens, or a different radius, does not fit
    for i, v in ((ph._I_OX, 12.0), (ph._I_LOGR, np.log(42.0))):
        bad = th.copy()
        bad[i] = v
        r2 = cell["mean"][ok] / ph._predict(bad, np.zeros(1), G, blur=False)[0][ok]
        r2 = r2 / np.median(r2) - 1.0
        assert abs(np.mean(r2[bend]) - np.mean(r2[~bend])) > 0.004


# --------------------------------------------------------------------------
# recovery of the geometry
# --------------------------------------------------------------------------
def test_recovers_tangent_line_axis_and_radius(plain):
    scene, cams, res = plain
    assert res.status == "supported" and res.reasons == []
    off, ang = tangent_errors(res)
    assert abs(off) < 0.25 and abs(ang) < 0.05 and abs(res.radius_mm - 35.0) < 0.5
    assert abs(off) < 3 * res.sigma_tangent_mm and abs(res.radius_mm - 35.0) < 3 * res.sigma_radius_mm
    assert res.radius_source == "fitted" and res.n_views_used == 4
    assert float(res.tangent.signed_distance(np.array([200.0, Z_MID]))) > 0, "normal points into the flat region"
    assert abs(float(res.direction @ np.array([0.0, 1.0]))) > 0.99999
    # evidence the decision is based on
    c = res.conditioning
    assert c["converged"] and c["condition_number_scaled"] < 1000 and 0.3 < c["reduced_chi2"] < 3
    assert set(c["sigma_formal"]) == {"offset_mm", "angle_rad", "log_radius"} and c["sigma_view_jackknife"]["offset_mm"] is not None
    assert c["stationarity_mm"] < 0.3, "the transition is the same line of the panel in every view"
    assert res.heldout["available"] and res.heldout["max_ratio"] < 1.5 and len(res.heldout["per_view"]) == 4
    # nuisance: torch 12 mm left of and 9 mm below the lens, beam exponent 1, ambient / torch = 0.25 / 0.9
    n = res.nuisance
    assert n["ambient_to_torch_ratio"]["value"] == pytest.approx(0.25 / 0.9, abs=0.04)
    assert n["tone_gamma"]["value"] == pytest.approx(1.0, abs=0.06)
    assert n["light_offset_cam_mm"]["value"][0] == pytest.approx(-12.0, abs=6.0) and n["light_offset_cam_mm"]["value"][1] == pytest.approx(9.0, abs=6.0)


def test_normalised_brightness_gradients_give_the_axis_without_a_shape_model(plain):
    _scene, _cams, res = plain
    g = res.gradient_axis
    t = np.asarray(g["direction"])
    err = np.degrees(np.arcsin(abs(t[0])))  # truth: the edge runs along y
    assert err < 0.5 and g["n_views"] == 4
    assert g["angle_from_hint_deg"] == pytest.approx(2.0, abs=0.5), "the 2 degree tilt of the hint was found"
    assert g["spread_deg"] < 1.0, "iso-brightness lines are parallel to the axis in every view"


@pytest.mark.parametrize("radius", [20.0, 60.0])
def test_other_radii(radius):
    scene = edge_scene(radius=radius)
    res = analyse_flash_sweep(views_of(scene, stations((1, 3, 5))), hint(), exclude_polygons=tag_polygons(scene))
    off, ang = tangent_errors(res)
    assert res.status == "supported"
    assert abs(off) < 0.3 and abs(ang) < 0.05 and res.radius_mm == pytest.approx(radius, abs=0.6)


def test_end_to_end_with_poses_from_the_tag_pipeline():
    """Photos -> tag detection -> bundle adjustment -> photometric analysis: pose error is included."""
    scene = edge_scene()
    cams = stations()
    mv, images, blur = [], [], []
    for i, c in enumerate(cams):
        img = render_cached(scene, c, seed=i).image
        dets = [d for d in detect_tags(img, camera=model(c)) if d.accepted]
        images.append(img)
        blur.append(float(np.median([d.sigma_blur_px for d in dets])))
        mv.append(MVView(f"f{i}", [TagObs(d.tag_id, d.corners_ideal, d.cov, d.size_mm) for d in dets], model(c), {d.tag_id: d.corners_px for d in dets}, (W, H)))
    sol = solve_multiview(mv, checks=False)
    # plane frame = frame of the gauge tag: express the hint and the truth there
    st = next(s for s in scene.stickers if s.tag_id == sol.gauge_id)
    a = np.deg2rad(st.rotation_deg)
    Rg, cg = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), np.asarray(st.center)
    h = hint()
    h_tag = EdgeHint(tuple(Rg.T @ (np.asarray(h.point) - cg)), tuple(Rg.T @ np.asarray(h.direction)), tuple(Rg.T @ np.asarray(h.inward)), extent_mm=h.extent_mm)
    views = [FlashView(images[i], model(c), sol.cameras[f"f{i}"], "on", None, f"f{i}", blur[i]) for i, c in enumerate(cams)]
    res = analyse_flash_sweep(views, h_tag, exclude_polygons=tag_exclusion_polygons([sol.tag_corners_world(t)[:, :2] for t in sol.tags]))
    assert res.status == "supported"
    n_scene = Rg @ res.tangent.n
    off, ang = tangent_errors(res, {"n": n_scene, "d": res.tangent.d + n_scene @ cg})
    assert abs(off) < 0.35 and abs(ang) < 0.06 and res.radius_mm == pytest.approx(35.0, abs=0.5)
    assert abs(off) < 3 * res.sigma_tangent_mm
    # research evidence only, inspectable, JSON-serialisable
    d = json.loads(json.dumps(res.to_dict()))
    assert d["label"] == EVIDENCE_LABEL == "research_evidence_only" and d["guidance_enabled"] is False
    assert d["status"] == "supported" and d["tangent"]["d"] == pytest.approx(res.tangent.d)
    assert len(d["views"]) == 4 and all(v["used"] and v["role"] == "torch" and v["gain"] > 0 for v in d["views"])
    assert {"saturated", "dark", "glare", "shadow_or_occlusion", "excluded", "used"} <= set(d["views"][0]["mask_fractions"])
    p = d["profiles"][0]
    assert len(p["e_mm"]) == len(p["measured"]) == len(p["model"]) == len(p["used"]) > 50
    flat_part = [m for e, m, u in zip(p["e_mm"], p["measured"], p["used"]) if u and e > 10]
    bend_part = [m for e, m, u in zip(p["e_mm"], p["measured"], p["used"]) if u and e < -8]
    assert np.median(flat_part) == pytest.approx(1.0, abs=0.03) and np.median(bend_part) < 0.97, "measured / predicted-flat leaves 1 on the bend"


# --------------------------------------------------------------------------
# refusing instead of inventing
# --------------------------------------------------------------------------
def test_one_view_is_not_cross_checked_and_the_request_is_specific():
    scene = edge_scene()
    res = analyse_flash_sweep(views_of(scene, stations((3,)), seed0=3), hint(), exclude_polygons=tag_polygons(scene))
    assert res.status == "needs_view" and res.reasons[0] == "too_few_views"
    assert res.tangent is None and res.radius_mm is None and res.sigma_tangent_mm is None
    assert res.requests[0]["code"] == "add_view" and res.requests[0]["torch"] == "on" and res.requests[0]["distance_mm"] >= 100
    assert "torch-on photo" in res.suggestions[0]
    w = res.conditioning["withheld_estimate"]  # kept for a researcher, never as a result
    assert w["tangent"] is not None and abs(tangent_errors(res, w["tangent"])[0]) < 1.0
    assert res.heldout["available"] is False and res.to_dict()["tangent"] is None


def test_without_torch_photos_nothing_is_estimated_and_the_planar_workflow_is_not_blocked():
    scene = edge_scene()
    cams = stations((3, 5))
    off = views_of(replace(scene, light=replace(scene.light, flash=False)), cams, torch="off", seed0=50)
    res = analyse_flash_sweep(off, hint())
    assert res.status == "needs_view" and res.reasons == ["no_torch_on_views"] and res.tangent is None
    assert res.requests[0] == {"code": "torch_on_sweep", "torch": "on", "n_photos": 3} and "planar workflow does not need" in res.suggestions[0]
    unknown = [replace(v, torch="unknown") for v in off]
    res = analyse_flash_sweep(unknown, hint())
    assert res.status == "needs_view" and set(res.reasons) == {"no_torch_on_views", "torch_state_unknown"}
    assert analyse_flash_sweep([], hint()).status == "needs_view"
    with pytest.raises(ValueError):
        analyse_flash_sweep(off, EdgeHint((0.0, 0.0), (0.0, 1.0), (0.0, 1.0)))  # 'inward' along the edge


@pytest.fixture(scope="module")
def flat_views():
    scene = flat_panel_scene(light=LightSpec(ambient=0.25, **LIGHT))
    return scene, stations()


def test_flat_panel_has_no_transition(flat_views):
    scene, cams = flat_views
    res = analyse_flash_sweep(views_of(scene, cams), hint(x=200.0), exclude_polygons=tag_polygons(scene))
    assert res.status == "unsupported" and res.reasons == ["no_shading_transition"] and res.tangent is None
    # ... also when the hint sits on a dark joint line of the flat panel
    res = analyse_flash_sweep(views_of(scene, cams), hint(x=0.0), exclude_polygons=tag_polygons(scene))
    assert res.status == "unsupported" and res.tangent is None


def test_the_edge_of_a_torch_shadow_is_never_reported_as_a_datum(flat_views):
    """A block standing proud of a flat panel throws a torch shadow whose edge
    moves with the phone: there is a brightness transition, but no bend."""
    scene, cams = flat_views
    scene = replace(scene, boxes=(Box(lo=(150.0, -60.0, 100.0), hi=(190.0, 0.0, 650.0)),))
    for x in (205.0, 140.0):
        res = analyse_flash_sweep(views_of(scene, cams), hint(x=x), exclude_polygons=tag_polygons(scene))
        assert res.status != "supported" and res.tangent is None and res.radius_mm is None


def test_shadow_and_occluder_next_to_a_real_edge_are_masked():
    scene = replace(edge_scene(), boxes=(Box(lo=(40.0, -40.0, 300.0), hi=(70.0, 0.0, 380.0)),))
    res = analyse_flash_sweep(views_of(scene, stations()), hint(), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported"
    off, _ang = tangent_errors(res)
    assert abs(off) < 0.3 and res.radius_mm == pytest.approx(35.0, abs=0.6)
    masked = [v.mask_fractions["shadow_or_occlusion"] + v.mask_fractions["material_or_fixed_pattern"] for v in res.views]
    assert max(masked) > 0.01, "the block and its shadow were taken out, not fitted"


# --------------------------------------------------------------------------
# exposure, gloss, tone
# --------------------------------------------------------------------------
def test_exposure_differences_go_into_the_view_gains():
    scene = edge_scene()
    exposures = (1.5, 0.7, 1.2, 0.9)
    cams = [replace(c, auto_exposure=False, exposure=e) for c, e in zip(stations(), exposures)]
    res = analyse_flash_sweep(views_of(scene, cams), hint(), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported" and abs(tangent_errors(res)[0]) < 0.3
    ratio = np.array([v.gain for v in res.views]) / np.array(exposures)
    assert np.ptp(ratio) / np.mean(ratio) < 0.02, "gain follows the (unknown) exposure of each photo"
    # clipped and far too dark photos are not analysed, and the operator is told why
    hot = [replace(c, auto_exposure=False, exposure=4.0) for c in stations((3, 5))]
    res = analyse_flash_sweep(views_of(scene, hot, seed0=3), hint(), exclude_polygons=tag_polygons(scene))
    assert res.status != "supported" and (set(res.reasons) & {"overexposed_views", "edge_region_not_usable", "overexposed"})


def test_gloss_is_modelled_and_its_core_masked():
    scene = edge_scene(panel_kw={"spec_strength": 0.4, "shininess": 15.0})
    res = analyse_flash_sweep(views_of(scene, stations()), hint(), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported" and abs(tangent_errors(res)[0]) < 0.3 and res.radius_mm == pytest.approx(35.0, abs=0.6)
    assert res.nuisance["specular"]["fitted"] and res.nuisance["specular"]["strength"] > 0.05
    assert max(v.mask_fractions["glare"] + v.mask_fractions["saturated"] for v in res.views) > 0.0


def test_unknown_tone_exponent_is_estimated_not_assumed():
    scene = edge_scene()
    cams = stations(tone_gamma=1.2)  # the photo is not sRGB: brightness ratios are stretched
    views = views_of(scene, cams)
    res = analyse_flash_sweep(views, hint(), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported" and res.nuisance["tone_gamma"]["value"] == pytest.approx(1.2, abs=0.06)
    assert abs(tangent_errors(res)[0]) < 0.3 and res.radius_mm == pytest.approx(35.0, abs=0.6)
    fixed = analyse_flash_sweep(views, hint(), PhotometricConfig(fit_tone_gamma=False), exclude_polygons=tag_polygons(scene))
    w = fixed.conditioning.get("withheld_estimate") or {"radius_mm": fixed.radius_mm}
    assert abs(w["radius_mm"] - 35.0) > abs(res.radius_mm - 35.0), "assuming sRGB biases the radius"


# --------------------------------------------------------------------------
# flash-off partners
# --------------------------------------------------------------------------
def test_paint_boundary_is_refused_alone_and_cancelled_by_the_ratio_of_a_pair():
    strip = Placard("strip", center=(-15.0, Z_MID), width=60.0, height=700.0, albedo=(0.5, 0.5, 0.5), border=0.0, text_lines=0)
    scene = edge_scene(placards=(strip,))  # darker paint from the bend to 15 mm onto the face
    views = with_partners(scene, stations())
    alone = analyse_flash_sweep(views[:4], hint(), exclude_polygons=tag_polygons(scene))
    assert alone.status == "unsupported" and alone.tangent is None
    assert set(alone.reasons) & {"surface_fixed_pattern_at_edge", "surface_not_uniform"}
    res = analyse_flash_sweep(views, hint(), PhotometricConfig(pair_mode="ratio"), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported" and abs(tangent_errors(res)[0]) < 0.3 and res.radius_mm == pytest.approx(35.0, abs=0.6)
    on, partner = res.views[0], res.views[4]
    assert on.role == "torch_over_partner" and partner.role == "partner" and on.pair_frame_id == "off0"
    assert on.pair_center_distance_mm == pytest.approx(4.0, abs=0.01)
    assert 0.2 < on.pair_exposure_ratio < 1.0, "auto exposure made the torch photo shorter: estimated, not assumed"


def test_uneven_room_light_is_refused_alone_and_removed_by_the_difference_of_a_pair():
    scene = edge_scene(light_kw={"directional": ((0.6, 0.7, -0.4), 0.3)})  # room light from one side: it shades the bend too
    views = with_partners(scene, stations())
    alone = analyse_flash_sweep(views[:4], hint(), exclude_polygons=tag_polygons(scene))
    assert alone.status != "supported" and alone.tangent is None
    res = analyse_flash_sweep(views, hint(), PhotometricConfig(pair_mode="difference"), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported" and abs(tangent_errors(res)[0]) < 0.35 and res.radius_mm == pytest.approx(35.0, abs=0.8)
    assert res.views[0].role == "torch_minus_partner" and res.nuisance["pair_mode"] == "difference"


def test_a_partner_is_used_only_if_aligned_and_usable():
    scene = edge_scene()
    views = with_partners(scene, stations((3, 5)))
    # the module receives poses: a partner reported 40 mm away is not a pair
    moved = [replace(v) for v in views]
    for v in moved[2:]:
        v.T_cam_plane = Pose(v.T_cam_plane.R, v.T_cam_plane.t + np.array([40.0, 0.0, 0.0]))
    res = analyse_flash_sweep(moved, hint(), exclude_polygons=tag_polygons(scene))
    assert "pair_misaligned" in res.reasons and all(v.role == "torch" for v in res.views[:2]) and res.views[2].role == "unused"
    apart = float(np.linalg.norm(moved[0].T_cam_plane.center - moved[2].T_cam_plane.center))
    assert apart > 30.0 and res.views[0].pair_center_distance_mm == pytest.approx(apart, abs=1e-6)
    assert any("retake the pair from a rest" in s for s in res.suggestions) and {"code": "retake_pair_from_rest", "max_center_distance_mm": 15.0} in res.requests
    # a partner that is almost black tells nothing about the ambient light
    dark = [replace(v) for v in views]
    for v in dark[2:]:
        v.image_bgr = (v.image_bgr * 0.08).astype(np.uint8)
    res = analyse_flash_sweep(dark, hint(), exclude_polygons=tag_polygons(scene))
    assert "pair_off_too_dark" in res.reasons and all(v.role == "torch" for v in res.views[:2])
    # an aligned pair is used
    res = analyse_flash_sweep(views, hint(), exclude_polygons=tag_polygons(scene))
    assert [v.role for v in res.views] == ["torch_minus_partner", "torch_minus_partner", "partner", "partner"]
    assert res.status == "supported" and abs(tangent_errors(res)[0]) < 0.4


# --------------------------------------------------------------------------
# options, combination, helpers
# --------------------------------------------------------------------------
def test_configured_radius_is_a_fallback_and_is_labelled(plain):
    scene, cams, free = plain
    views = views_of(scene, cams)
    # pretend the photos cannot determine the radius: the configured value is used and said so
    res = analyse_flash_sweep(views, hint(radius=35.0), PhotometricConfig(max_rel_sigma_radius=1e-6), exclude_polygons=tag_polygons(scene))
    assert res.status == "supported" and res.radius_source == "hint_fixed" and res.radius_mm == 35.0 and res.sigma_radius_mm is None
    assert "radius_fixed_to_configured_value" in res.reasons and abs(tangent_errors(res)[0]) < 0.3
    # without a configured radius the same threshold makes the radius "not identifiable": tangent kept, radius withheld
    res = analyse_flash_sweep(views, hint(), PhotometricConfig(max_rel_sigma_radius=1e-6), exclude_polygons=tag_polygons(scene))
    assert res.status == "needs_view" and res.reasons[0] == "radius_not_identifiable" and res.radius_mm is None and res.tangent is not None
    assert res.requests[0]["torch"] == "on" and res.requests[0]["code"] in ("move_toward_edge", "add_view", "closer_view")
    with pytest.raises(ValueError):
        analyse_flash_sweep(views, hint(), PhotometricConfig(pair_mode="subtract"))


def _result(d=0.0, sigma=0.2, radius=35.0, status="supported"):
    n = np.array([1.0, 0.0])
    return PhotometricResult(status=status, tangent=Line2D(n, d), tangent_point=np.array([d, 300.0]), direction=np.array([0.0, -1.0]),
                             sigma_tangent_mm=sigma, sigma_angle_deg=0.02, radius_mm=radius, sigma_radius_mm=0.3)


def test_combination_is_inverse_variance_and_refuses_when_the_estimates_disagree():
    sil = Line2D(np.array([1.0, 0.0]), 0.3)
    k = combine_with_silhouette(_result(), sil, 0.1, 0.01, 35.4, 0.2)
    assert k["available"] and k["consistent"] and k["label"] == EVIDENCE_LABEL
    assert k["tangent"]["d"] == pytest.approx(0.3 * 0.2**2 / (0.2**2 + 0.1**2)) and k["sigma_tangent_mm"] == pytest.approx(1 / np.sqrt(1 / 0.04 + 1 / 0.01))
    assert k["radius_mm"] == pytest.approx((35.0 / 0.09 + 35.4 / 0.04) / (1 / 0.09 + 1 / 0.04)) and k["weight_photometric"] == pytest.approx(0.2)
    # the same line given with the opposite normal is the same line
    k2 = combine_with_silhouette(_result(), sil.flipped(), 0.1, 0.01, 35.4, 0.2)
    assert k2["tangent"]["d"] == pytest.approx(k["tangent"]["d"])
    # the silhouette uncertainty applies at its own reference point: direction uncertainty is carried along the line
    k3 = combine_with_silhouette(_result(), sil, 0.1, 0.5, reference_point=np.array([0.3, 0.0]))
    assert k3["sigma_tangent_mm"] > k["sigma_tangent_mm"]
    far = combine_with_silhouette(_result(), Line2D(np.array([1.0, 0.0]), 1.5), 0.1, 0.01)
    assert not far["available"] and far["reason"] == "estimates_disagree" and far["z_offset"] > 3 and "tangent" not in far
    assert combine_with_silhouette(_result(status="needs_view"), sil, 0.1, 0.01)["reason"] == "photometric_not_supported"
    assert combine_with_silhouette(_result(), None, float("nan"), 0.01)["reason"] == "silhouette_not_available"


def test_exclusion_polygons_and_hint_frame():
    sq = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    q = np.array([[5.0, 5.0], [12.0, 5.0], [14.0, 5.0], [-1.0, -1.0], [30.0, 30.0]])
    assert ph._in_polygons(q, [sq], margin=3.0).tolist() == [True, True, False, True, False]
    assert ph._in_polygons(q, [], margin=3.0).sum() == 0
    grown = tag_exclusion_polygons([sq], grow=0.5)[0]
    assert np.allclose(grown.min(axis=0), [-5.0, -5.0]) and np.allclose(grown.max(axis=0), [15.0, 15.0])
    p, t, n = ph._hint_frame(EdgeHint((1.0, 2.0), (0.0, 3.0), (2.0, 0.5)))
    assert np.allclose(n, [1.0, 0.0]) and abs(t @ n) < 1e-12 and np.allclose([-t[1], t[0]], n), "n is 90 degrees counter-clockwise from t"


def test_bounded_solver_respects_bounds_and_down_weights_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 200)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.01, size=x.size)
    y[::20] += 5.0  # gross outliers
    fun = lambda p: (p[0] + p[1] * x - y) / 0.01  # noqa: E731
    jac = lambda p: np.stack([np.ones_like(x), x], axis=1) / 0.01  # noqa: E731
    sol, ok = ph._bounded_lm(fun, jac, np.array([0.0, 0.0]), np.array([-10.0, -10.0]), np.array([10.0, 10.0]), x.size, 2.0, 50)
    assert ok and sol == pytest.approx([2.0, 3.0], abs=0.02)
    sol, _ = ph._bounded_lm(fun, jac, np.array([0.0, 0.0]), np.array([-10.0, -10.0]), np.array([10.0, 2.5]), x.size, 2.0, 50)
    assert sol[1] == pytest.approx(2.5) and sol[0] > 2.0


# --------------------------------------------------------------------------
# longer studies (deselect with -m "not slow")
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_stated_uncertainty_covers_the_error_over_noise_seeds():
    scene = edge_scene()
    cams = stations((1, 3, 5))
    z_t, z_r = [], []
    for k in range(6):
        res = analyse_flash_sweep(views_of(scene, cams, seed0=200 + 10 * k), hint(), exclude_polygons=tag_polygons(scene))
        assert res.status == "supported"
        z_t.append(tangent_errors(res)[0] / res.sigma_tangent_mm)
        z_r.append((res.radius_mm - 35.0) / res.sigma_radius_mm)
    assert np.max(np.abs(z_t)) < 3.0 and np.max(np.abs(z_r)) < 3.0
    assert np.sqrt(np.mean(np.square(z_t))) < 1.5, "not optimistic"


@pytest.mark.slow
def test_strong_tone_curve_is_refused_not_trusted():
    scene = edge_scene()
    cams = [phone_camera(e, target=(230.0, 0.0, 300.0), width=W, height=H, roll_deg=r, auto_exposure=True, tone_contrast=0.4)
            for e, r in sweep_stations(7, distance=760, height=430, span_deg=62.0, target=(230.0, 0.0, 300.0))]
    res = analyse_flash_sweep(views_of(scene, cams), hint(), exclude_polygons=tag_polygons(scene))
    assert res.status == "unsupported" and set(res.reasons) & {"heldout_inconsistent", "transition_not_stationary", "model_does_not_explain_brightness"}
