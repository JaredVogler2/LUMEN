"""AprilTag measurement against renderer ground truth (prompt 03)."""

import cv2
import numpy as np
import pytest

from lumen.geometry.camera import CameraModel
from lumen.synth import Box, LightSpec, TagSticker
from lumen.synth.fixtures import demo_tags, flat_panel_scene, curved_panel_scene, phone_camera
from lumen.vision.markers import CORNER_SIGNS, MarkerConfig, TagSizeTable, accepted_by_id, detect_tags, tag_object_points
from tests.conftest import render_cached

W, H = 1280, 960
DIST = (0.08, -0.15, 0.0006, -0.0004, 0.0)


def _errors(res, dets):
    e, radial = [], []
    for d in dets:
        gt = np.array(next(t for t in res.truth["tags"] if t["id"] == d.tag_id)["corners_px"])
        diff = d.corners_px - gt
        e.append(np.linalg.norm(diff, axis=1))
        out = (gt - gt.mean(0)) / np.linalg.norm(gt - gt.mean(0), axis=1, keepdims=True)
        radial.append(np.sum(diff * out, axis=1))
    return np.concatenate(e), np.concatenate(radial)


def _model(cam):
    return CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist)


def test_all_tags_found_with_subpixel_corners_for_any_rotation_and_id():
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3)
    res = render_cached(flat_panel_scene(), cam)
    dets = detect_tags(res.image)
    assert sorted(d.tag_id for d in dets) == [3, 7, 11, 20] and all(d.accepted and d.refined for d in dets)
    err, radial = _errors(res, dets)
    assert err.max() < 0.08 and np.sqrt(np.mean(err**2)) < 0.04
    assert abs(radial.mean()) < 0.02, "no net shrink/grow of the tags (scale bias)"


def test_edge_fit_removes_the_scale_bias_of_stock_corner_refinement():
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3)
    res = render_cached(flat_panel_scene(), cam)
    gray = cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p).detectMarkers(gray)
    stock = []
    for c, i in zip(corners, ids.ravel()):
        gt = np.array(next(t for t in res.truth["tags"] if t["id"] == i)["corners_px"])
        out = (gt - gt.mean(0)) / np.linalg.norm(gt - gt.mean(0), axis=1, keepdims=True)
        stock.append(np.sum((c.reshape(4, 2) - gt) * out, axis=1))
    stock_bias = float(np.mean(np.concatenate(stock)))
    _, radial = _errors(res, detect_tags(res.image))
    assert stock_bias < -0.3, "the stock detector is expected to make tags look smaller"
    assert abs(radial.mean()) < 0.1 * abs(stock_bias)


def test_calibrated_lens_distortion():
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3, dist=DIST)
    res = render_cached(flat_panel_scene(), cam)
    dets = detect_tags(res.image, camera=_model(cam))
    err, radial = _errors(res, dets)
    assert len(dets) == 4 and err.max() < 0.1 and abs(radial.mean()) < 0.03
    for d in dets:  # ideal corners really are the undistorted truth
        gt = np.array(next(t for t in res.truth["tags"] if t["id"] == d.tag_id)["corners_px"])
        assert np.max(np.linalg.norm(d.corners_ideal - _model(cam).undistort_pixels(gt), axis=1)) < 0.1


def test_camera_model_must_match_the_image():
    cam = phone_camera((430, -720, 430), width=W, height=H)
    res = render_cached(flat_panel_scene(), cam)
    with pytest.raises(ValueError):
        detect_tags(res.image, camera=CameraModel.from_fov(640, 480, 60.0))


@pytest.mark.parametrize("kw,limit", [({"noise_shot": 0.004, "noise_read": 0.012}, 0.15), ({"blur_sigma": 1.6}, 0.08), ({"motion_blur_px": 4.0, "motion_blur_deg": 20.0}, 0.08), ({"jpeg_quality": 75}, 0.12)])
def test_noise_blur_and_compression(kw, limit):
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3, **kw)
    res = render_cached(flat_panel_scene(), cam, seed=3)
    dets = [d for d in detect_tags(res.image) if d.accepted]
    assert len(dets) == 4
    err, radial = _errors(res, dets)
    assert err.max() < limit and abs(radial.mean()) < 0.04


@pytest.mark.parametrize("kw", [{"tone_gamma": 1.25}, {"tone_gamma": 0.8}, {"sharpen": 1.0}, {"tone_contrast": 0.6, "sharpen": 0.6}])
def test_polarity_balanced_correction_handles_unknown_tone_curves(kw):
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3, blur_sigma=0.9, noise_shot=0.0, noise_read=0.0, **kw)
    res = render_cached(flat_panel_scene(), cam)
    _, raw = _errors(res, detect_tags(res.image, cfg=MarkerConfig(bias_correction=False)))
    dets = detect_tags(res.image)
    _, corrected = _errors(res, dets)
    assert abs(raw.mean()) > 0.12, "the mismatch must create a visible bias, otherwise the test proves nothing"
    assert abs(corrected.mean()) < 0.03 and abs(corrected.mean()) < 0.2 * abs(raw.mean())
    assert all(d.bias_applied and abs(d.bias_px) > 0.05 for d in dets)
    # the correction is reported with an uncertainty that grows with its size
    assert all(d.bias_sigma_px >= 0.35 * abs(d.bias_px) for d in dets)


def test_scale_tags_far_away_are_rejected_as_too_small():
    cam = phone_camera((300, -2600, 400), target=(280, 0, 300), width=W, height=H)
    res = render_cached(flat_panel_scene(), cam)
    dets = detect_tags(res.image)
    assert dets, "tags are still decoded"
    assert all((not d.accepted) and "too_small" in d.reject_reasons for d in dets)


def test_predicted_corner_sigma_grows_with_distance():
    near = render_cached(flat_panel_scene(), phone_camera((300, -560, 320), target=(280, 0, 300), width=W, height=H))
    far = render_cached(flat_panel_scene(), phone_camera((300, -1200, 320), target=(280, 0, 300), width=W, height=H))
    dn = accepted_by_id(detect_tags(near.image))
    df = accepted_by_id(detect_tags(far.image))
    assert set(dn) & set(df)
    for i in set(dn) & set(df):
        assert df[i].mean_edge_px < dn[i].mean_edge_px
    # relative size precision is what matters for scale
    rel = lambda d: d.corner_sigma_px / d.mean_edge_px  # noqa: E731
    assert np.mean([rel(d) for d in df.values()]) > np.mean([rel(d) for d in dn.values()])


def test_duplicate_ids_in_one_scene_are_rejected():
    tags = demo_tags() + (TagSticker(7, (300.0, 320.0), 5.0),)
    res = render_cached(flat_panel_scene(tags=tags), phone_camera((430, -720, 430), width=W, height=H))
    dets = detect_tags(res.image)
    sevens = [d for d in dets if d.tag_id == 7]
    assert len(sevens) == 2 and all(not d.accepted and "duplicate_id" in d.reject_reasons for d in sevens)
    assert all(d.accepted for d in dets if d.tag_id != 7)


def test_clipped_tag_is_rejected():
    # tag 3 sits on the image border
    cam = phone_camera((560, -560, 330), target=(560, 0, 330), width=W, height=H)
    res = render_cached(flat_panel_scene(), cam)
    t3 = np.array(next(t for t in res.truth["tags"] if t["id"] == 3)["corners_px"])
    assert t3[:, 0].min() < 15, "scene must put the tag at the border for the test to be meaningful"
    d3 = [d for d in detect_tags(res.image) if d.tag_id == 3]
    assert not d3 or (not d3[0].accepted and "clipped" in d3[0].reject_reasons)


def test_occluded_corner_never_yields_an_accepted_tag():
    # an opaque block in front of the panel hides one corner of tag 7
    box = Box((455.0, -40.0, 60.0), (520.0, -1.0, 108.0))
    scene = flat_panel_scene()
    scene = type(scene)(panel=scene.panel, floor=scene.floor, stickers=scene.stickers, seams=scene.seams, boxes=(box,), light=scene.light)
    res = render_cached(scene, phone_camera((430, -720, 430), width=W, height=H))
    d7 = [d for d in detect_tags(res.image) if d.tag_id == 7]
    assert not d7 or not d7[0].accepted
    assert sum(d.accepted for d in detect_tags(res.image)) == 3


def test_glare_washing_out_a_tag_is_not_silently_accepted():
    # strong flash and a glossy sticker: tag 3 sits in the specular highlight
    tags = tuple(TagSticker(t.tag_id, t.center, t.rotation_deg, spec_strength=3.0, shininess=120.0) for t in demo_tags())
    scene = flat_panel_scene(tags=tags, light=LightSpec(ambient=0.5, flash=True, flash_strength=1.6))
    cam = phone_camera((95, -520, 125), target=(95, 0, 125), width=W, height=H)  # looking straight at tag 3
    res = render_cached(scene, cam)
    for d in detect_tags(res.image):
        gt = np.array(next(t for t in res.truth["tags"] if t["id"] == d.tag_id)["corners_px"])
        err = np.linalg.norm(d.corners_px - gt, axis=1).max()
        # every accepted tag must still be accurate; a damaged one must be rejected or flagged
        assert (not d.accepted) or err < 0.25 or "highlights_clipped" in d.flags, (d.tag_id, err, d.flags)


def test_bent_tag_on_the_curved_edge_is_rejected():
    tags = demo_tags() + (TagSticker(30, (-12.0, 300.0), 0.0),)  # straddles the tangent line, wraps onto the bend
    res = render_cached(curved_panel_scene(radius=30.0, tags=tags), phone_camera((330, -640, 380), target=(200, 0, 300), width=W, height=H))
    d30 = [d for d in detect_tags(res.image) if d.tag_id == 30]
    assert not d30 or not d30[0].accepted or d30[0].max_sagitta_px > 0.2


def test_measured_sizes_and_provenance():
    cam = phone_camera((430, -720, 430), width=W, height=H)
    res = render_cached(flat_panel_scene(), cam)
    sizes = TagSizeTable(nominal_mm=50.0, measured_mm={3: 49.82, 7: 50.11})
    by = {d.tag_id: d for d in detect_tags(res.image, sizes=sizes)}
    assert (by[3].size_mm, by[3].size_provenance) == (49.82, "measured")
    assert (by[11].size_mm, by[11].size_provenance) == (50.0, "nominal")
    assert TagSizeTable.from_dict(sizes.to_dict()).measured_mm == sizes.measured_mm


def test_tag_frame_convention():
    obj = tag_object_points(50.0)
    assert obj.shape == (4, 3) and np.allclose(obj[:, 2], 0)
    # TL, TR, BR, BL in a frame with x right and y up
    assert np.array_equal(np.sign(obj[:, :2]), CORNER_SIGNS)
    assert np.cross(obj[1] - obj[0], obj[3] - obj[0])[2] < 0  # clockwise seen from +z (the viewer)


def test_no_tags_in_a_blank_image():
    assert detect_tags(np.full((480, 640, 3), 128, np.uint8)) == []
