"""End-to-end API flows on synthetic photos, compared with ground truth."""

import json

import numpy as np
import pytest

from lumen.synth.fixtures import flat_panel_scene, phone_camera
from lumen.synth.render import project_points
from tests.conftest import calibration_for, camera_from_truth, jpeg_bytes, px, render_cached

W, H = 1600, 1200
DIST = (0.06, -0.12, 0.0004, -0.0003, 0.0)
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}


def _new(client, mode="real"):
    r = client.post("/api/sessions", json={"mode": mode})
    assert r.status_code == 200
    return r.json()["id"]


def _upload(client, sid, img, **params):
    r = client.post(f"/api/sessions/{sid}/frames", params=params, content=jpeg_bytes(img), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200, r.text
    return r.json()["frame"]


def _datums(client, sid, fid, cam, offset=(4.0, -3.0)):
    fl = [(px(cam, 60, 0) + [0, offset[0]]).tolist(), (px(cam, 560, 0) + [0, offset[1]]).tolist()]
    sd = [(px(cam, 0, 80) + [offset[0], 0]).tolist(), (px(cam, 0, 620) + [offset[1], 0]).tolist()]
    a = client.post(f"/api/sessions/{sid}/datums/floor", json={"frame_id": fid, "p0": fl[0], "p1": fl[1]})
    b = client.post(f"/api/sessions/{sid}/datums/side", json={"frame_id": fid, "p0": sd[0], "p1": sd[1]})
    assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
    return a.json()["datum"], b.json()["datum"]


def _target(ov, pid="PLC-DEMO-101"):
    return next(p for p in ov["primitives"] if p["kind"] in ("target", "target_preview") and p.get("label") == pid and "corners" in p)


@pytest.fixture(scope="module")
def shot():
    scene = flat_panel_scene()
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3, dist=DIST)
    return scene, cam, render_cached(scene, cam, seed=1)


def test_health_and_config(client):
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    assert h.json()["checks"]["apriltag_36h11"] is True
    assert h.headers["cache-control"] == "no-store" and h.headers["x-content-type-options"] == "nosniff"
    c = client.get("/api/config").json()
    assert c["features"]["assistant"] is False and c["validated_domain"] is None
    assert "not product acceptance" in c["notice"].lower()


def test_single_photo_calibrated_flow_matches_ground_truth(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    assert client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"}).status_code == 200
    assert client.put(f"/api/sessions/{sid}/tags", json=MEASURED).status_code == 200
    assert client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cam)}).status_code == 200
    f = _upload(client, sid, res.image)
    assert sorted(m["tag_id"] for m in f["markers"] if m["accepted"]) == [3, 7, 11, 20]
    assert f["calibration_id"] == "truth" and all(m["size_provenance"] == "measured" for m in f["markers"])
    reg = client.post(f"/api/sessions/{sid}/register").json()["registration"]
    assert reg["n_tags"] == 4 and reg["consistent"] and reg["independently_checked"] and reg["reference_frame_id"] == f["id"]
    d_floor, d_side = _datums(client, sid, f["id"], cam)
    assert d_floor["provenance"] == "measured" and d_floor["feature_kind"] == "step"
    assert d_side["provenance"] == "measured" and d_side["feature_kind"] == "bar"
    assert d_floor["distinctness"] > 3 and d_side["distinctness"] > 3

    # before confirmation: preview only
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"]}).json()
    rep = ov["reports"]["PLC-DEMO-101"]
    assert rep["status"] == "research_preview" and _target(ov)["kind"] == "target_preview"
    assert "datums_confirmed" in {g["code"] for g in rep["gates"] if not g["passed"]}

    s = client.post(f"/api/sessions/{sid}/datums/confirm").json()
    lf = s["local_frame"]
    assert lf["valid"] and lf["side_of_panel"] == "left" and abs(lf["raw_angle_deg"] - 90.0) < 0.05

    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"]}).json()
    rep = ov["reports"]["PLC-DEMO-101"]
    tgt = _target(ov)
    truth = np.array([px(cam, 180, 330), px(cam, 300, 330), px(cam, 300, 270), px(cam, 180, 270)])  # TL TR BR BL
    err_px = np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)
    err_mm = err_px / ov["px_per_mm"]
    assert err_mm.max() < 0.15, err_mm
    u = rep["uncertainty"]
    assert u is not None and err_mm.max() < u["expanded_mm"]
    assert {c["kind"] for c in u["components"]} == {"random", "systematic"}
    names = " ".join(c["name"] for c in u["components"])
    assert "calibration uncertainty" in names and "not characterised" not in names
    # grid, axes and dimensions are real geometry in image pixels
    kinds = {p["kind"] for p in ov["primitives"]}
    assert {"grid_minor", "grid_major", "grid_axis", "datum_floor", "datum_side", "axis_x", "axis_z", "origin", "dim", "tag", "tick"} <= kinds
    origin = next(p for p in ov["primitives"] if p["kind"] == "origin")["points"][0]
    assert np.linalg.norm(np.array(origin) - px(cam, 0, 0)) < 0.5
    assert ov["frozen"] is True and "not product acceptance" in ov["notice"].lower()
    # a grid line x = 250 mm really passes through the image of (250, z)
    gl = next(p for p in ov["primitives"] if p["kind"] == "grid_major" and p["axis"] == "x" and p["value"] == 250.0)
    d = np.min(np.linalg.norm(np.array(gl["points"]) - px(cam, 250, 200), axis=1))
    assert d < 12.0  # polyline sample spacing
    return


def test_rectified_view_and_drawing_datums_in_rectified_space(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cam)})
    f = _upload(client, sid, res.image)
    client.post(f"/api/sessions/{sid}/register")
    r = client.get(f"/api/sessions/{sid}/frames/{f['id']}/rectified", params={"max": 1200})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    spec = json.loads(r.headers["x-lumen-rect"])
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"], "view": "rect", "rect_max": 1200, "uncertainty": "false"}).json()
    assert ov["rect"] == spec, "image and overlay must describe the same rectification"
    # draw the datums on the rectified picture: plane point -> rect pixel via the spec
    A = np.array(spec["A"])
    Ainv = np.linalg.inv(A)
    reg = client.get(f"/api/sessions/{sid}").json()["registration"]
    Hm = np.array(reg["views"][f["id"]]["H"])
    from lumen.geometry.camera import CameraModel
    from lumen.geometry.homography import apply_h

    model = CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist)

    def rect_px(x, z):
        q = apply_h(np.linalg.inv(Hm), model.undistort_pixels(px(cam, x, z)[None]))[0]
        return (Ainv @ np.array([q[0], q[1], 1.0]))[:2].tolist()

    body = {"frame_id": f["id"], "space": "rect", "rect": spec}
    a = client.post(f"/api/sessions/{sid}/datums/floor", json={**body, "p0": rect_px(60, 0), "p1": rect_px(560, 0)})
    b = client.post(f"/api/sessions/{sid}/datums/side", json={**body, "p0": rect_px(0, 80), "p1": rect_px(0, 620)})
    assert a.status_code == 200 and b.status_code == 200
    client.post(f"/api/sessions/{sid}/datums/confirm")
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"]}).json()
    truth = np.array([px(cam, 180, 330), px(cam, 300, 330), px(cam, 300, 270), px(cam, 180, 270)])
    assert np.max(np.linalg.norm(np.array(_target(ov)["corners"]) - truth, axis=1)) / ov["px_per_mm"] < 0.15
    # once the frame exists the rectified view is oriented with +z up and is never mirrored
    spec2 = json.loads(client.get(f"/api/sessions/{sid}/frames/{f['id']}/rectified").headers["x-lumen-rect"])
    er, eu = np.array(spec2["e_right"]), np.array(spec2["e_up"])
    assert er[0] * eu[1] - er[1] * eu[0] > 0


def test_uncalibrated_nominal_single_tag_is_only_a_preview(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    f = _upload(client, sid, res.image)
    assert f["calibration_id"] == "" and all(m["size_provenance"] == "nominal" for m in f["markers"])
    client.post(f"/api/sessions/{sid}/register")
    _datums(client, sid, f["id"], cam)
    client.post(f"/api/sessions/{sid}/datums/confirm")
    rep = client.get(f"/api/sessions/{sid}/report", params={"frame": f["id"]}).json()
    failed = {g["code"] for g in rep["gates"] if not g["passed"]}
    assert rep["status"] == "research_preview"
    assert {"tag_sizes_measured", "lens_characterised"} <= failed
    names = [c["name"] for c in rep["uncertainty"]["components"]]
    assert any("nominal" in n for n in names) and any("not characterised" in n for n in names)
    assert rep["suggestions"]


def test_blocked_states_fail_closed(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    # no operation, no frames
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "blocked" and rep["uncertainty"] is None
    r = client.post(f"/api/sessions/{sid}/register")
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_tags" and r.json()["error"]["suggestions"]
    r = client.get(f"/api/sessions/{sid}/overlay", params={"frame": "f001"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_registered"
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    f = _upload(client, sid, res.image)
    client.post(f"/api/sessions/{sid}/register")
    # registered but no datums: grid and target must not appear
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"]}).json()
    kinds = {p["kind"] for p in ov["primitives"]}
    assert ov["reports"]["PLC-DEMO-101"]["status"] == "blocked"
    assert "tag" in kinds and not ({"target", "target_preview", "grid_major", "grid_minor"} & kinds)
    assert client.post(f"/api/sessions/{sid}/datums/confirm").status_code == 409
    # nearly parallel datums -> invalid frame, still no target
    a = px(cam, 60, 0).tolist()
    b = px(cam, 560, 0).tolist()
    client.post(f"/api/sessions/{sid}/datums/floor", json={"frame_id": f["id"], "p0": a, "p1": b})
    client.post(f"/api/sessions/{sid}/datums/side", json={"frame_id": f["id"], "p0": a, "p1": b})
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["local_frame"] is not None and not s["local_frame"]["valid"]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"]}).json()
    assert ov["reports"]["PLC-DEMO-101"]["status"] == "blocked" and not any(p["kind"].startswith("target") for p in ov["primitives"])


def test_incomplete_record_never_reaches_guidance(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    r = client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-INCOMPLETE-01", "revision": "A", "placard_id": "PLC-DEMO-901"})
    assert r.status_code == 200 and r.json()["operation"]["summary"]["complete"] is False
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cam)})
    f = _upload(client, sid, res.image)
    client.post(f"/api/sessions/{sid}/register")
    _datums(client, sid, f["id"], cam)
    rep = client.get(f"/api/sessions/{sid}/report", params={"frame": f["id"]}).json()
    # the side datum has no configured sign: the frame cannot be built at all
    assert rep["status"] == "blocked"
    assert "operation_record_complete" in {g["code"] for g in rep["gates"] if not g["passed"]}


def test_switching_placard_keeps_registration_and_datums(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cam)})
    f = _upload(client, sid, res.image)
    client.post(f"/api/sessions/{sid}/register")
    _datums(client, sid, f["id"], cam)
    client.post(f"/api/sessions/{sid}/datums/confirm")
    s = client.put(f"/api/sessions/{sid}/placard", json={"placard_id": "PLC-DEMO-102"}).json()
    assert s["active_placard_id"] == "PLC-DEMO-102" and s["registration"] is not None and s["local_frame"]["valid"]
    assert s["datums"]["floor"]["confirmed"] and s["datums"]["side"]["confirmed"]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"], "placards": "PLC-DEMO-101,PLC-DEMO-102"}).json()
    t2 = _target(ov, "PLC-DEMO-102")
    # 3 in square whose bottom-left corner is at (16.5 in, 5.0 in)
    x0, z0, sde = 16.5 * 25.4, 5.0 * 25.4, 76.2
    truth = np.array([px(cam, x0, z0 + sde), px(cam, x0 + sde, z0 + sde), px(cam, x0 + sde, z0), px(cam, x0, z0)])
    assert np.max(np.linalg.norm(np.array(t2["corners"]) - truth, axis=1)) / ov["px_per_mm"] < 0.2
    assert client.put(f"/api/sessions/{sid}/placard", json={"placard_id": "PLC-DEMO-201"}).status_code == 404


def test_changing_tag_sizes_or_frames_invalidates_results(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    f = _upload(client, sid, res.image)
    client.post(f"/api/sessions/{sid}/register")
    _datums(client, sid, f["id"], cam)
    s = client.put(f"/api/sessions/{sid}/tags", json=MEASURED).json()
    assert s["registration"] is None and s["datums"] == {} and s["local_frame"] is None
    assert all(m["size_provenance"] == "measured" for m in s["frames"][0]["markers"])
    client.post(f"/api/sessions/{sid}/register")
    s = client.delete(f"/api/sessions/{sid}/frames/{f['id']}").json()
    assert s["frames"] == [] and s["registration"] is None
    assert client.get(f"/api/sessions/{sid}/frames/{f['id']}/image").status_code == 404


def test_mismatched_calibration_is_refused_not_rescaled(client, shot):
    scene, cam, res = shot
    sid = _new(client)
    wrong = calibration_for(cam)
    wrong.update(width=4032, height=3024)
    client.put(f"/api/sessions/{sid}/calibration", json={"profile": wrong})
    f = _upload(client, sid, res.image)
    assert f["calibration_id"] == "" and "4032x3024" in f["calibration_note"]
    rotated = calibration_for(cam)
    rotated.update(width=cam.height, height=cam.width)
    s = client.put(f"/api/sessions/{sid}/calibration", json={"profile": rotated}).json()
    assert s["frames"][0]["calibration_id"] == "" and "orientation" in s["frames"][0]["calibration_note"]
    bad = calibration_for(cam)
    bad["fx"] = -5
    assert client.put(f"/api/sessions/{sid}/calibration", json={"profile": bad}).status_code == 422


def test_upload_validation_and_limits(client):
    sid = _new(client)
    url = f"/api/sessions/{sid}/frames"
    assert client.post(url, content=b"", headers={"content-type": "image/jpeg"}).json()["error"]["code"] == "empty_upload"
    assert client.post(url, content=b"GIF89a....", headers={"content-type": "image/jpeg"}).json()["error"]["code"] == "unsupported_image_type"
    assert client.post(url, content=b"\xff\xd8\xff\xe0 truncated", headers={"content-type": "image/jpeg"}).json()["error"]["code"] == "image_decode_failed"
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
    r = client.post(url, content=heic, headers={"content-type": "image/heic"})
    assert r.status_code == 415 and r.json()["error"]["code"] == "heic_not_supported" and r.json()["error"]["suggestions"]
    assert client.post(url, params={"role": "selfie"}, content=jpeg_bytes(np.zeros((64, 64, 3), np.uint8)), headers={"content-type": "image/jpeg"}).status_code == 422
    r = client.post(url, content=jpeg_bytes(np.zeros((64, 64, 3), np.uint8)), headers={"content-type": "image/jpeg", "x-lumen-meta": "%7Bnot json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_meta"


def test_frame_and_size_limits():
    from fastapi.testclient import TestClient

    from lumen.api.app import create_app
    from lumen.config import Limits
    from lumen.operations.library import OperationLibrary
    from lumen.services.sessions import SessionService
    from lumen.storage.base import MemorySessionStore
    from tests.conftest import BUNDLED_OPS, make_settings

    settings = make_settings(limits=Limits(max_frames_per_session=2, max_image_bytes=40_000, max_sessions=2, max_image_pixels=300_000))
    svc = SessionService(MemorySessionStore(), settings, OperationLibrary(BUNDLED_OPS))
    with TestClient(create_app(settings, svc)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        small = jpeg_bytes(np.full((120, 160, 3), 128, np.uint8))
        for _ in range(2):
            assert c.post(f"/api/sessions/{sid}/frames", content=small, headers={"content-type": "image/jpeg"}).status_code == 200
        r = c.post(f"/api/sessions/{sid}/frames", content=small, headers={"content-type": "image/jpeg"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_frames"
        big = jpeg_bytes(np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8), 95)
        assert len(big) > 40_000
        sid2 = c.post("/api/sessions", json={}).json()["id"]
        assert c.post(f"/api/sessions/{sid2}/frames", content=big, headers={"content-type": "image/jpeg"}).status_code == 413
        many_px = jpeg_bytes(np.full((600, 600, 3), 128, np.uint8), 30)
        r = c.post(f"/api/sessions/{sid2}/frames", content=many_px, headers={"content-type": "image/jpeg"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_many_pixels"
        r = c.post("/api/sessions", json={})
        assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_sessions"
        # deleting a session frees capacity and removes its photos
        assert c.delete(f"/api/sessions/{sid}").json() == {"deleted": True}
        assert c.get(f"/api/sessions/{sid}").status_code == 404
        assert c.post("/api/sessions", json={}).status_code == 200


def test_session_ids_are_capabilities_and_bound_to_the_connect_user(client):
    creds = lambda u: {"RStudio-Connect-Credentials": json.dumps({"user": u, "groups": []})}  # noqa: E731
    sid = client.post("/api/sessions", json={}, headers=creds("alice")).json()["id"]
    assert client.get(f"/api/sessions/{sid}", headers=creds("alice")).json()["operator"] == "alice"
    assert client.get(f"/api/sessions/{sid}", headers=creds("bob")).status_code == 404
    for bad in ("..%2F..%2Fetc%2Fpasswd", "%2e%2e", "short", "y" * 70, "aaaaaaaaaaaaaaaaaaaaaaaa%00"):
        r = client.get(f"/api/sessions/{bad}")
        # either no route matches the decoded path, or the id fails validation —
        # in both cases storage is never touched
        assert r.status_code == 404 and r.json()["error"]["code"] in ("session_not_found", "http_404"), bad
    assert client.get(f"/api/sessions/{sid}/frames/..%2F..%2Fstate.json/image", headers=creds("alice")).status_code == 404


def test_demo_session_reaches_prototype_guidance_and_is_labelled_simulated(client, demo_manifest):
    sid = _new(client, "demo")
    s = client.post(f"/api/sessions/{sid}/demo/load", json={"scenario": "flat"}).json()
    assert s["mode"] == "demo" and len(s["frames"]) == 5 and all(f["source"] == "demo" for f in s["frames"])
    assert s["calibration"]["provenance"] == "simulated"
    reg = client.post(f"/api/sessions/{sid}/register").json()["registration"]
    assert len(reg["frames_used"]) == 5 and reg["n_tags"] == 4 and reg["consistent"]
    ref = reg["reference_frame_id"]
    idx = [f["id"] for f in s["frames"]].index(ref)
    cam = camera_from_truth(demo_manifest["frames"][idx]["truth"]["camera"])
    _datums(client, sid, ref, cam)
    client.post(f"/api/sessions/{sid}/datums/confirm")
    # the target is correct in *every* registered photo, not only the one the datums were drawn on
    worst = 0.0
    for k, f in enumerate(s["frames"]):
        c = camera_from_truth(demo_manifest["frames"][k]["truth"]["camera"])
        ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": f["id"], "uncertainty": "false"}).json()
        truth = np.array([px(c, 180, 330), px(c, 300, 330), px(c, 300, 270), px(c, 180, 270)])
        worst = max(worst, float(np.max(np.linalg.norm(np.array(_target(ov)["corners"]) - truth, axis=1)) / ov["px_per_mm"]))
        assert ov["simulated"] is True
    assert worst < 0.2, worst
    rep = client.get(f"/api/sessions/{sid}/report", params={"frame": ref}).json()
    assert rep["status"] == "prototype_guidance" and rep["notice"].startswith("SIMULATED") and not rep["physically_validated"]
    assert worst < rep["uncertainty"]["expanded_mm"]
    exp = client.get(f"/api/sessions/{sid}/export").json()
    assert exp["simulated"] is True and exp["images_included"] is False and "local" in exp["coordinates"]
    assert exp["operation"]["content_hash"] and exp["reports"]["PLC-DEMO-101"]["status"] == "prototype_guidance"
    assert "jpg" not in json.dumps(exp["frames"][0].get("blob", ""))
    # demo content cannot be loaded into a real session
    real = _new(client)
    assert client.post(f"/api/sessions/{real}/demo/load", json={"scenario": "flat"}).status_code == 409
    assert client.post(f"/api/sessions/{sid}/demo/load", json={"scenario": "../etc"}).status_code == 404


def test_operations_api(client):
    ops = client.get("/api/operations").json()
    assert ops["editable"] is False and len(ops["operations"]) == 3
    d = client.get("/api/operations/DEMO-FLAT-01").json()
    assert d["placards_resolved"][1]["x_mm"] == pytest.approx(419.1)
    assert client.get("/api/operations/NOPE").status_code == 404
    assert client.post("/api/operations/resolve", json={"code": "LUMEN-DEMO-PLC-101"}).json()["status"] == "exact"
    assert client.post("/api/operations", json={"operation": d["operation"]}).status_code == 409  # read-only deployment
