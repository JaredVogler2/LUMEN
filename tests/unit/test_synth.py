"""The renderer is the reference for every accuracy claim, so it is tested
against two independent implementations of the camera model."""

import cv2
import numpy as np
import pytest

from lumen.geometry.camera import CameraModel
from lumen.synth import CameraSpec, SceneGeometry, TagSticker, project_points, render, scene_truth
from lumen.synth.fixtures import curved_panel_scene, flat_panel_scene, phone_camera
from lumen.synth.render import pixel_rays, srgb_decode, srgb_encode
from lumen.synth.tags import tag_bits, tag_image
from tests.conftest import render_cached

DIST = (0.08, -0.15, 0.0006, -0.0004, 0.0)


def test_truth_projection_agrees_with_opencv_and_lumen_geometry():
    cam = phone_camera((430, -720, 430), roll_deg=3, dist=DIST)
    scene = flat_panel_scene()
    truth = scene_truth(scene, cam)
    for t in truth["tags"]:
        Xw = np.array(t["corners_world"])
        R_wc = np.array(truth["camera"]["R_wc"])
        C = np.array(truth["camera"]["C"])
        rvec, _ = cv2.Rodrigues(R_wc.T)
        tvec = -R_wc.T @ C
        K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])
        ref, _ = cv2.projectPoints(Xw, rvec, tvec, K, np.array(cam.dist))
        assert np.max(np.abs(ref.reshape(-1, 2) - np.array(t["corners_px"]))) < 1e-7
        model = CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist)
        uv, _ = model.project((Xw - C) @ R_wc)
        assert np.max(np.abs(uv - np.array(t["corners_px"]))) < 1e-9


def test_pixel_rays_invert_the_forward_projection():
    cam = phone_camera((300, -650, 380), dist=DIST)
    uv = np.array([[0.0, 0.0], [799.5, 599.5], [1599.0, 1199.0], [120.3, 1010.7]])
    d = pixel_rays(cam, uv)
    X = np.asarray(cam.C) + 700.0 * d
    back, depth = project_points(cam, X)
    assert np.all(depth > 0) and np.max(np.abs(back - uv)) < 1e-8


def test_rendering_is_deterministic():
    cam = phone_camera((380, -640, 400), width=480, height=360, noise_shot=0.001, noise_read=0.005)
    scene = flat_panel_scene()
    a = render(scene, cam, ss=2, seed=5).image
    b = render(scene, cam, ss=2, seed=5).image
    c = render(scene, cam, ss=2, seed=6).image
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_adaptive_supersampling_matches_full_supersampling():
    cam = phone_camera((380, -640, 400), width=400, height=300, blur_sigma=0.0, noise_shot=0.0, noise_read=0.0)
    scene = flat_panel_scene()
    a = render(scene, cam, ss=3, adaptive=True).image.astype(int)
    b = render(scene, cam, ss=3, adaptive=False).image.astype(int)
    diff = np.abs(a - b)
    assert np.percentile(diff, 99.5) <= 2 and diff.max() <= 40


def test_tag_orientation_matches_detector_corner_order():
    """The first detected corner must be the tag's own top-left, for any rotation."""
    cam = phone_camera((300, -600, 300), target=(300, 0, 300), width=960, height=720)
    for rot in (0.0, 37.0, 90.0, 201.0):
        scene = flat_panel_scene(tags=(TagSticker(42, (300.0, 300.0), rot),))
        res = render_cached(scene, cam, ss=3)
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY))
        assert ids is not None and ids.ravel().tolist() == [42]
        gt = np.array(res.truth["tags"][0]["corners_px"])
        assert np.max(np.linalg.norm(corners[0].reshape(4, 2) - gt, axis=1)) < 2.5, f"rotation {rot}"


def test_tag_bits_have_a_black_border_and_are_distinct():
    for tid in (0, 1, 17, 586):
        b = tag_bits(tid)
        assert b.shape == (8, 8)
        assert not b[0].any() and not b[-1].any() and not b[:, 0].any() and not b[:, -1].any()
    assert not np.array_equal(tag_bits(3), tag_bits(4))
    with pytest.raises(ValueError):
        tag_bits(587)
    img = tag_image(5, cell_px=10, quiet_cells=1.0)
    assert img.shape == (100, 100) and img[0, 0] == 255 and img[15, 15] == 0


def test_srgb_round_trip():
    x = np.linspace(0, 1, 101)
    assert np.allclose(srgb_decode(srgb_encode(x)), x, atol=1e-12)
    assert srgb_encode(np.array([0.2140]))[0] == pytest.approx(0.5, abs=0.002)  # mid-grey is ~21 % linear


def test_curved_edge_geometry_is_continuous_and_isometric():
    g = SceneGeometry(curved_panel_scene(radius=40.0, taper_deg=6.0))
    eg = g.left
    s = np.linspace(-0.8 * 40.0 * np.pi / 2, 30.0, 400)
    w = np.full_like(s, 250.0)
    P, N = eg.surface_point(s, w)
    # arc length along the surface equals the unrolled distance
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    assert np.allclose(seg, np.diff(s), rtol=2e-4)
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)
    # normals are perpendicular to the surface tangent and face the viewer on the flat part
    tang = np.diff(P, axis=0)
    assert np.max(np.abs(np.einsum("ij,ij->i", tang, N[:-1]) / seg)) < 2e-2
    assert N[-1, 1] == pytest.approx(-1.0)
    # the bend stays on the far side of the panel plane
    assert P[:, 1].min() >= -1e-9


def test_lifted_and_floor_tags_have_off_plane_truth():
    scene = flat_panel_scene(tags=(TagSticker(1, (200.0, 200.0), lift=6.0), TagSticker(2, (300.0, 150.0), host="floor")))
    g = SceneGeometry(scene)
    assert np.allclose(g.sticker_corners_world(scene.stickers[0])[:, 1], -6.0)
    floor = g.sticker_corners_world(scene.stickers[1])
    assert np.allclose(floor[:, 2], 0.0) and floor[:, 1].max() < 0  # on the floor, in front of the panel
