"""Independent scale check: one ruler measurement between two tag corners (prompts 03 and 10)."""

import numpy as np
import pytest

from lumen.synth import scene_truth
from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import calibration_for, jpeg_bytes, px, render_cached

W, H = 1600, 1200
TARGET = [(180, 330), (300, 330), (300, 270), (180, 270)]


def start(client, sizes):
    scene = flat_panel_scene()
    cams = [phone_camera(e, width=W, height=H, roll_deg=r) for e, r in sweep_stations(5, distance=760, height=430, span_deg=50)]
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json={"nominal_mm": 50.0, "measured_mm": sizes})
    client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cams[0])})
    for i, c in enumerate(cams):
        client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render_cached(scene, c, seed=i).image), headers={"content-type": "image/jpeg"})
    reg = client.post(f"/api/sessions/{sid}/register").json()["registration"]
    client.post(f"/api/sessions/{sid}/datums/propose")
    client.post(f"/api/sessions/{sid}/datums/confirm")
    return sid, scene, cams, reg


def true_distance(scene, cam, a, b):
    tags = {t["id"]: np.array(t["corners_world"])[:, [0, 2]] for t in scene_truth(scene, cam)["tags"]}
    return float(np.linalg.norm(tags[a["tag_id"]][a["corner"]] - tags[b["tag_id"]][b["corner"]]))


def target_error(client, sid, reg, cams):
    fid = reg["reference_frame_id"]
    frames = [f["id"] for f in client.get(f"/api/sessions/{sid}").json()["frames"]]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": fid}).json()
    tgt = next((p for p in ov["primitives"] if p["kind"].startswith("target") and "corners" in p), None)
    if tgt is None:
        return None, ov
    truth = np.array([px(cams[frames.index(fid)], x, z) for x, z in TARGET])
    return float(np.max(np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)) / ov["px_per_mm"]), ov


def test_true_sizes_agree_with_a_ruler_measurement(client):
    sid, scene, cams, reg = start(client, {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0})
    sug = client.get(f"/api/sessions/{sid}/reference").json()["suggestion"]
    a, b = sug["a"], sug["b"]
    d_true = true_distance(scene, cams[0], a, b)
    assert d_true > 450.0, "the suggestion uses the longest baseline the tags offer"
    assert np.linalg.norm(np.subtract(*sug["points_plane"])) == pytest.approx(d_true, abs=0.3)
    # the operator reads the ruler 0.4 mm off
    r = client.post(f"/api/sessions/{sid}/reference", json={"a": a, "b": b, "measured_mm": d_true + 0.4, "u_mm": 0.5, "instrument": "steel rule"}).json()
    ref = r["reference"]
    assert ref["consistent"] and abs(ref["difference_mm"] + 0.4) < 0.3 and 1.0 < ref["limit_mm"] < 2.5
    rep = client.get(f"/api/sessions/{sid}/report").json()
    g = next(g for g in rep["gates"] if g["code"] == "independent_length_check")
    assert g["passed"] and "agrees" in g["message"] and rep["status"] == "prototype_guidance"
    # the two corners are marked on the picture
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": reg["reference_frame_id"], "references": "true"}).json()
    handles = [p for p in ov["primitives"] if p["kind"] == "handle"]
    frames = [f["id"] for f in client.get(f"/api/sessions/{sid}").json()["frames"]]
    cam = cams[frames.index(reg["reference_frame_id"])]
    tags = {t["id"]: np.array(t["corners_px"]) for t in scene_truth(scene, cam)["tags"]}
    assert len(handles) == 2
    assert np.linalg.norm(np.array(handles[0]["points"][0]) - tags[a["tag_id"]][a["corner"]]) < 1.0
    assert np.linalg.norm(np.array(handles[1]["points"][0]) - tags[b["tag_id"]][b["corner"]]) < 1.0


def test_a_common_size_error_is_invisible_to_the_tags_and_caught_by_the_ruler(client):
    # every tag declared 52 mm; the stickers are 50 mm
    sid, scene, cams, reg = start(client, {"3": 52.0, "7": 52.0, "11": 52.0, "20": 52.0})
    assert reg["consistent"] and not reg["excluded_tags"], "tag-to-tag checks cannot see a common factor"
    err, ov = target_error(client, sid, reg, cams)
    rep0 = ov["reports"]["PLC-DEMO-101"]
    # the target is drawn 4 % too far from the datums (18 mm at its far corner), with a clean report: a false acceptance
    assert 12.0 < err < 22.0 and rep0["status"] == "prototype_guidance" and err > rep0["uncertainty"]["expanded_mm"]
    sug = client.get(f"/api/sessions/{sid}/reference").json()["suggestion"]
    d_true = true_distance(scene, cams[0], sug["a"], sug["b"])
    ref = client.post(f"/api/sessions/{sid}/reference", json={"a": sug["a"], "b": sug["b"], "measured_mm": d_true, "u_mm": 0.5}).json()["reference"]
    assert not ref["consistent"] and ref["relative_difference"] == pytest.approx(0.04, abs=0.004)
    err2, ov2 = target_error(client, sid, reg, cams)
    rep = ov2["reports"]["PLC-DEMO-101"]
    assert err2 is None and rep["status"] == "blocked", "no target is drawn once the scale is contradicted"
    g = next(g for g in rep["gates"] if g["code"] == "independent_length_check")
    assert not g["passed"] and g["severity"] == "blocking" and "contradicts" in g["message"] and "+4." in g["message"]
    # correcting the sizes keeps the ruler measurement and re-checks it against the new registration
    s = client.put(f"/api/sessions/{sid}/tags", json={"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}).json()
    assert len(s["reference_distances"]) == 1 and s["reference_distances"][0]["evaluable"] is False and s["reference_distances"][0]["measured_mm"] == pytest.approx(d_true)
    reg2 = client.post(f"/api/sessions/{sid}/register").json()["registration"]
    client.post(f"/api/sessions/{sid}/datums/propose")
    client.post(f"/api/sessions/{sid}/datums/confirm")
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["reference_distances"][0]["consistent"] is True
    err3, ov3 = target_error(client, sid, reg2, cams)
    assert err3 < 0.2 and ov3["reports"]["PLC-DEMO-101"]["status"] == "prototype_guidance"


def test_reference_input_is_validated_and_can_be_removed(client):
    sid, scene, cams, reg = start(client, {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0})
    post = lambda **b: client.post(f"/api/sessions/{sid}/reference", json=b)  # noqa: E731
    assert post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 3, "corner": 0}, measured_mm=100).status_code == 422
    assert post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 3, "corner": 1}, measured_mm=50).json()["error"]["code"] == "baseline_too_short"
    assert post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 99, "corner": 1}, measured_mm=300).status_code == 404
    assert post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 20, "corner": 7}, measured_mm=300).status_code == 422
    assert post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 20, "corner": 2}, measured_mm=1.0).status_code == 422
    d = true_distance(scene, cams[0], {"tag_id": 3, "corner": 0}, {"tag_id": 20, "corner": 2})
    rid = post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 20, "corner": 2}, measured_mm=d).json()["reference"]["id"]
    # the same pair measured again replaces the earlier value
    rid2 = post(a={"tag_id": 3, "corner": 0}, b={"tag_id": 20, "corner": 2}, measured_mm=d + 0.2).json()["reference"]["id"]
    s = client.get(f"/api/sessions/{sid}").json()
    assert [r["id"] for r in s["reference_distances"]] == [rid2] and rid != rid2
    assert client.delete(f"/api/sessions/{sid}/reference/{rid2}").json()["reference_distances"] == []
    assert client.delete(f"/api/sessions/{sid}/reference/{rid2}").status_code == 404
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert not any(g["code"] == "independent_length_check" for g in rep["gates"]), "without a measurement the check is simply absent"
    # needs a registration
    sid2 = client.post("/api/sessions", json={}).json()["id"]
    assert client.get(f"/api/sessions/{sid2}/reference").status_code == 409
