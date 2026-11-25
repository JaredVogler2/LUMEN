"""Camera calibration through the API, and measuring through the resulting profile (prompt 03).

Board photos come from ``lumen.synth.boards`` (simulated).  The last test is
the end-to-end one: calibrate a lens from board photos, then measure a panel
photographed through the same simulated lens using only that profile.
"""

import json
from dataclasses import replace
from urllib.parse import quote

import cv2
import numpy as np
import pytest

from lumen.synth import render
from lumen.synth.boards import PrintedBoard, phone_base, render_board, view_set
from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
from lumen.vision.calibration import BoardSpec, detect_board
from tests.conftest import jpeg_bytes, px

W, H = 1600, 1200
BASE = phone_base(W, H, pp_offset=(7.3, -5.1), aspect=1.001)
CHU = PrintedBoard(kind="charuco", squares_x=9, squares_y=6, square_mm=30.0, marker_mm=21.0, dictionary="DICT_5X5_100")
BOARD = {"kind": "charuco", "squares_x": 9, "squares_y": 6, "square_mm": 30.0, "marker_mm": 21.0, "dictionary": "DICT_5X5_100"}
N = 12


@pytest.fixture(scope="module")
def shots():
    cams = view_set(CHU, BASE, N + 1, seed=3)
    return [render_board(CHU, c, seed=40 + k) for k, c in enumerate(cams)]


def meta(d):
    return {"x-lumen-meta": quote(json.dumps(d))}


def add(client, cid, img, **m):
    r = client.post(f"/api/calibration/{cid}/photos", content=jpeg_bytes(img, 95), headers={"content-type": "image/jpeg", **(meta(m) if m else {})})
    assert r.status_code == 200, r.text
    return r.json()


def calibrated(client, shots, n=N, **m):
    cid = client.post("/api/calibration", json={"board": BOARD}).json()["id"]
    for s in shots[:n]:
        add(client, cid, s.image, **m)
    return cid, client.post(f"/api/calibration/{cid}/solve", json={"label": "test phone"}).json()


def test_printable_board_is_the_board_the_detector_expects(client):
    r = client.get("/api/calibration/board.png", params={"dpi": 100})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and "attachment" in r.headers["content-disposition"]
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape[1] > 9 * 110 and int(img[5, 5]) == 255  # white margin
    obs = detect_board(img, BoardSpec.from_dict(BOARD))
    assert obs.ok and obs.n_points == 8 * 5, (obs.reasons, obs.n_points)
    chk = client.get("/api/calibration/board.png", params={"kind": "checkerboard", "squares_x": 10, "squares_y": 7, "square_mm": 25, "dpi": 100})
    g = cv2.imdecode(np.frombuffer(chk.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert detect_board(g, BoardSpec("checkerboard", 10, 7, 25.0)).n_points == 9 * 6
    assert client.get("/api/calibration/board.png", params={"squares_x": 1}).status_code == 422
    assert client.get("/api/calibration/board.png", params={"dpi": 5000}).status_code == 422


def test_calibration_from_photos_gives_a_profile_close_to_the_true_lens(client, shots):
    cid = client.post("/api/calibration", json={"board": BOARD}).json()["id"]
    first = add(client, cid, shots[0].image)
    v = first["view"]
    assert v["ok"] and v["n_points"] >= 30 and v["image_size"] == [W, H] and v["native_size"] == [W, H]
    assert first["calibration"]["result"] is None and "no calibrated intrinsics" in first["calibration"]["web_api_note"].lower() or "intrinsics" in first["calibration"]["web_api_note"].lower()
    # too few photos: refused with advice, and no profile
    few = client.post(f"/api/calibration/{cid}/solve", json={}).json()
    assert few["result"]["ok"] is False and "too_few_views" in few["result"]["reasons"] and few["profile"] is None and few["result"]["suggestions"]
    for s in shots[1:N]:
        add(client, cid, s.image)
    out = client.post(f"/api/calibration/{cid}/solve", json={"label": "test phone"}).json()
    res, prof = out["result"], out["profile"]
    assert res["ok"] and res["status"] == "calibrated" and out["n_usable"] == N
    assert prof["method"] == "charuco" and prof["provenance"] == "measured" and (prof["width"], prof["height"]) == (W, H) and prof["n_views"] == N
    truth = np.array([BASE.fx, BASE.fy, BASE.cx, BASE.cy, BASE.dist[0], BASE.dist[1]])
    got = np.array([prof["fx"], prof["fy"], prof["cx"], prof["cy"], prof["dist"][0], prof["dist"][1]])
    sig = np.array(prof["param_sigma"][:6])
    assert np.all(np.abs(got - truth)[:4] < 0.6), got - truth
    assert np.all(np.abs(got - truth) < 4.0 * sig + 1e-9)
    assert prof["rms_px"] < 0.08 and 0.4 < prof["coverage"] <= 1.0
    # corner lists stay on the server; the client gets the verdicts
    assert "image_points" not in json.dumps(out["views"])
    # a photo is kept with the calibration and can be looked at again
    img = client.get(f"/api/calibration/{cid}/photos/{out['views'][0]['view_id']}/image", params={"max": 320})
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    # check on a board photo that was not used
    val = client.post(f"/api/calibration/{cid}/validate", content=jpeg_bytes(shots[N].image, 95), headers={"content-type": "image/jpeg"}).json()["validation"]
    assert val["ok"] and val["independent"]
    assert val["with_model"]["rms_px"] < 0.08 < val["without_model"]["rms_px"]
    used_again = client.post(f"/api/calibration/{cid}/validate", content=jpeg_bytes(shots[0].image, 95), headers={"content-type": "image/jpeg"})
    assert used_again.status_code == 409 and used_again.json()["error"]["code"] == "photo_was_used"
    # removing a photo makes the result stale
    after = client.delete(f"/api/calibration/{cid}/photos/{out['views'][0]['view_id']}").json()
    assert after["result"] is None and after["profile"] is None and len(after["views"]) == N - 1
    assert client.delete(f"/api/calibration/{cid}").json()["deleted"] and client.get(f"/api/calibration/{cid}").status_code == 404


def test_refusals_and_separation_from_sessions(client, shots):
    cid = client.post("/api/calibration", json={"board": BOARD}).json()["id"]
    # a calibration workspace is not a session, and a session is not a workspace
    assert client.get(f"/api/sessions/{cid}").status_code == 404
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert client.get(f"/api/calibration/{sid}").status_code == 404
    # no board in the picture
    blank = np.full((H, W, 3), 120, np.uint8)
    v = add(client, cid, blank)["view"]
    assert not v["ok"] and v["reasons"] == ["board_not_found"] and v["suggestions"]
    # mixed pixel sizes are one configuration too many
    for s in shots[:6]:
        add(client, cid, s.image)
    for s in shots[6:9]:
        add(client, cid, cv2.resize(s.image, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    r = client.post(f"/api/calibration/{cid}/solve", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "mixed_camera_configurations"
    assert client.post("/api/calibration", json={"board": {"kind": "charuco", "squares_x": 9}}).status_code == 422
    assert client.post(f"/api/calibration/{cid}/validate", content=jpeg_bytes(shots[0].image), headers={"content-type": "image/jpeg"}).status_code == 409


def test_measuring_through_a_profile_calibrated_from_board_photos(client, shots):
    """The whole chain: board photos -> profile -> panel photos through the same lens -> target position."""
    cid, out = calibrated(client, shots, camera_label="Sim Back Camera", zoom=1.0)
    prof = out["profile"]
    assert out["result"]["ok"] and prof["device"] == "Sim Back Camera" and prof["zoom"] == 1.0
    scene = flat_panel_scene()
    cams = []
    for e, r in sweep_stations(5, distance=760, height=430, span_deg=50):
        c = phone_camera(e, width=W, height=H, roll_deg=r)
        cams.append(replace(c, fx=BASE.fx, fy=BASE.fy, cx=BASE.cx, cy=BASE.cy, dist=BASE.dist))  # the lens that was calibrated
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json={"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}})
    assert client.put(f"/api/sessions/{sid}/calibration", json={"profile": prof}).status_code == 200
    for k, c in enumerate(cams):
        f = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render(scene, c, ss=3, seed=k).image, 95), headers={"content-type": "image/jpeg", **meta({"camera_label": "Sim Back Camera", "zoom": 1.0})}).json()["frame"]
        assert f["calibration_id"] == prof["id"], f["calibration_note"]
    reg = client.post(f"/api/sessions/{sid}/register").json()["registration"]
    assert reg["consistent"]
    cc = reg["multiview"]["calibration_check"]
    assert cc["consistent"] and abs(cc["relative_difference"]) < 0.003  # the tags independently agree with the calibrated focal length
    client.post(f"/api/sessions/{sid}/datums/propose")
    client.post(f"/api/sessions/{sid}/datums/confirm")
    fid = reg["reference_frame_id"]
    frames = [f["id"] for f in client.get(f"/api/sessions/{sid}").json()["frames"]]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": fid}).json()
    tgt = next(p for p in ov["primitives"] if p["kind"].startswith("target") and "corners" in p)
    truth = np.array([px(cams[frames.index(fid)], x, z) for x, z in [(180, 330), (300, 330), (300, 270), (180, 270)]])
    err = float(np.max(np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)) / ov["px_per_mm"])
    rep = ov["reports"]["PLC-DEMO-101"]
    assert err < 0.3 and err < rep["uncertainty"]["expanded_mm"], (err, rep["uncertainty"]["expanded_mm"])
    assert not any(g["code"] == "frames_calibrated" and not g["passed"] for g in rep["gates"])
    # a picture that reports another camera or zoom does not get this profile
    other = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render(scene, cams[0], ss=2, seed=9).image), headers={"content-type": "image/jpeg", **meta({"camera_label": "Sim Ultra Wide Camera", "zoom": 1.0})}).json()["frame"]
    assert other["calibration_id"] == "" and "Sim Back Camera" in other["calibration_note"] and "Ultra Wide" in other["calibration_note"]
    zoomed = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render(scene, cams[0], ss=2, seed=10).image), headers={"content-type": "image/jpeg", **meta({"camera_label": "Sim Back Camera", "zoom": 2.0})}).json()["frame"]
    assert zoomed["calibration_id"] == "" and "zoom" in zoomed["calibration_note"]
