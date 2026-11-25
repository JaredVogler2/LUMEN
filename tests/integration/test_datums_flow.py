"""Automatic datum detection through the API (prompt 06 test cases)."""

from dataclasses import replace

import numpy as np
import pytest

from lumen.synth import Seam, TrimSpec
from lumen.synth.fixtures import curved_panel_scene, flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import calibration_for, camera_from_truth, jpeg_bytes, px, render_cached

W, H = 1600, 1200
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}


def sweep(n=5, span=50.0, target=(270.0, 0.0, 300.0), **kw):
    return [phone_camera(e, target=target, width=W, height=H, roll_deg=r, **kw) for e, r in sweep_stations(n, distance=760, height=430, span_deg=span, target=target)]


def session(client, scene, cams, operation="DEMO-FLAT-01", placard="PLC-DEMO-101", calibrated=True):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": operation, "placard_id": placard})
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    if calibrated:
        client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cams[0])})
    for i, c in enumerate(cams):
        r = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render_cached(scene, c, seed=i).image), headers={"content-type": "image/jpeg"})
        assert r.status_code == 200
    reg = client.post(f"/api/sessions/{sid}/register").json()
    return sid, reg


def propose(client, sid):
    r = client.post(f"/api/sessions/{sid}/datums/propose")
    assert r.status_code == 200, r.text
    return r.json()


def scene_line_of(datum, reg, scene, gauge_xz, gauge_rot_deg):
    """Datum line (plane frame) expressed in scene coordinates: returns (point, direction)."""
    a = np.deg2rad(gauge_rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ends = np.array(datum["endpoints"]) @ R.T + np.asarray(gauge_xz)
    return ends


def gauge_of(reg, scene):
    g = reg["registration"]["layout"]["gauge_id"]
    st = next(s for s in scene.stickers if s.tag_id == g)
    return st.center, st.rotation_deg


def target_error_mm(client, sid, reg, cams, corners_xz):
    fid = reg["registration"]["reference_frame_id"]
    frames = [f["id"] for f in client.get(f"/api/sessions/{sid}").json()["frames"]]
    cam = cams[frames.index(fid)]
    ov = client.get(f"/api/sessions/{sid}/overlay", params={"frame": fid}).json()
    tgt = next(p for p in ov["primitives"] if p["kind"].startswith("target") and "corners" in p)
    truth = np.array([px(cam, x, z) for x, z in corners_xz])
    return float(np.max(np.linalg.norm(np.array(tgt["corners"]) - truth, axis=1)) / ov["px_per_mm"]), ov


def test_flat_panel_datums_are_found_and_verified_without_any_drawing(client):
    scene = flat_panel_scene()
    cams = sweep()
    sid, reg = session(client, scene, cams)
    out = propose(client, sid)
    p, s = out["proposals"], out["session"]
    assert set(p["applied"]) == {"floor", "side"} and not p["ambiguous_floor"] and not p["ambiguous_side"] and not p["hints"]
    fl, sd = s["datums"]["floor"], s["datums"]["side"]
    assert fl["feature_kind"] == "step" and sd["feature_kind"] == "bar"
    assert fl["on_plane_evidence"] == "verified_multiview" and sd["on_plane_evidence"] == "verified_multiview"
    assert fl["provenance"] == "measured" and not fl["confirmed"], "automatic datums still need operator confirmation"
    gxz, grot = gauge_of(reg, scene)
    ef = scene_line_of(fl, reg, scene, gxz, grot)
    es = scene_line_of(sd, reg, scene, gxz, grot)
    assert np.max(np.abs(ef[:, 1])) < 0.15 and np.max(np.abs(es[:, 0])) < 0.15  # z = 0 and x = 0
    lf = s["local_frame"]
    assert lf["valid"] and lf["side_of_panel"] == "left" and abs(lf["raw_angle_deg"] - 90) < 0.05
    # unconfirmed -> preview; confirmed -> guidance, target within 0.2 mm of truth
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "research_preview"
    client.post(f"/api/sessions/{sid}/datums/confirm")
    err, ov = target_error_mm(client, sid, reg, cams, [(180, 330), (300, 330), (300, 270), (180, 270)])
    assert err < 0.2
    assert ov["reports"]["PLC-DEMO-101"]["status"] == "prototype_guidance"


def test_floor_trim_offers_on_plane_junction_and_flags_the_proud_edge(client):
    scene = replace(flat_panel_scene(), trim=TrimSpec(height=60.0, thickness=8.0))
    cams = sweep()
    sid, reg = session(client, scene, cams)
    p = propose(client, sid)["proposals"]
    gxz, grot = gauge_of(reg, scene)
    cls = {}
    for c in p["candidates"]:
        e = scene_line_of(c, reg, scene, gxz, grot)
        if abs(e[0, 1] - e[1, 1]) < 3:  # a horizontal feature
            cls[round(float(e[:, 1].mean()))] = c
    # the trim/panel junction (z = 60) lies on the face; the trim's lower front edge is 8 mm proud
    assert cls[60]["classification"] == "on_plane"
    proud = next(c for z, c in cls.items() if z < 20)
    assert proud["classification"] == "off_plane" and proud["height_mm"] == pytest.approx(8.0, abs=0.4)
    # the on-plane junction is preferred; the true floor line is hidden behind the trim and is NOT invented
    best_floor = next(c for c in p["candidates"] if c["id"] == p["applied"]["floor"])
    e = scene_line_of(best_floor, reg, scene, gxz, grot)
    assert e[:, 1].mean() == pytest.approx(60.0, abs=0.3)
    # choosing the proud edge instead is possible but blocks guidance
    r = client.post(f"/api/sessions/{sid}/datums/select", json={"kind": "floor", "candidate_id": proud["id"]})
    assert r.status_code == 200 and r.json()["datum"]["on_plane_evidence"] == "off_plane"
    client.post(f"/api/sessions/{sid}/datums/confirm")
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "blocked" and not next(g for g in rep["gates"] if g["code"] == "datum_floor_on_plane")["passed"]


def test_false_seam_makes_the_side_datum_ambiguous(client):
    scene = flat_panel_scene(extra_seams=(Seam((-45.0, 0.0), (-45.0, 1100.0), width=2.0, darkness=0.65),))
    cams = sweep()
    sid, reg = session(client, scene, cams)
    out = propose(client, sid)
    p = out["proposals"]
    assert p["ambiguous_side"] and not p["ambiguous_floor"] and len(p["side"]) == 2
    client.post(f"/api/sessions/{sid}/datums/confirm")
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "research_preview"
    assert not next(g for g in rep["gates"] if g["code"] == "datum_side_unambiguous")["passed"]
    # the operator picks the intended seam explicitly: ambiguity is resolved by a human, on the record
    gxz, grot = gauge_of(reg, scene)
    true_seam = min(p["side"], key=lambda c: abs(scene_line_of(c, reg, scene, gxz, grot)[:, 0].mean()))
    r = client.post(f"/api/sessions/{sid}/datums/select", json={"kind": "side", "candidate_id": true_seam["id"]})
    assert "chosen by the operator" in " ".join(r.json()["datum"]["notes"])
    client.post(f"/api/sessions/{sid}/datums/confirm")
    err, ov = target_error_mm(client, sid, reg, cams, [(180, 330), (300, 330), (300, 270), (180, 270)])
    assert err < 0.2 and ov["reports"]["PLC-DEMO-101"]["status"] == "prototype_guidance"


def test_missing_floor_is_reported_with_a_capture_hint(client):
    scene = flat_panel_scene()
    cams = [phone_camera(e, target=(270.0, 0.0, 520.0), width=W, height=H, roll_deg=r, hfov_deg=50.0) for e, r in sweep_stations(4, distance=700, height=560, span_deg=40, target=(270.0, 0.0, 520.0))]
    # the floor seam is below the bottom of every photo
    assert all(px(c, 300, 0)[1] > H + 20 for c in cams)
    sid, reg = session(client, scene, cams)
    out = propose(client, sid)
    p = out["proposals"]
    assert "floor" not in p["applied"] and any("floor" in h.lower() for h in p["hints"])
    assert out["session"]["local_frame"] is None
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "blocked" and not next(g for g in rep["gates"] if g["code"] == "datums_defined")["passed"]


@pytest.mark.parametrize("radius", [20.0, 35.0, 60.0])
def test_rounded_edge_tangent_from_silhouettes(client, radius):
    scene = curved_panel_scene(radius=radius)
    cams = sweep(7, span=62.0, target=(230.0, 0.0, 300.0))
    sid, reg = session(client, scene, cams, operation="DEMO-CURVE-01", placard="PLC-DEMO-201")
    out = propose(client, sid)
    p, s = out["proposals"], out["session"]
    cyl = p["cylinder"]
    assert cyl["ok"] and cyl["n_views"] >= 4
    assert cyl["radius_mm"] == pytest.approx(radius, abs=max(1.0, 3 * cyl["sigma_radius_mm"]))
    sd = s["datums"]["side"]
    assert sd["feature_kind"] == "cylinder_model" and sd["on_plane_evidence"] == "cylinder_model"
    gxz, grot = gauge_of(reg, scene)
    es = scene_line_of(sd, reg, scene, gxz, grot)
    assert np.max(np.abs(es[:, 0])) < 0.5, "tangent line is at scene x = 0"
    assert s["surface"]["kind"] == "plane_with_cylindrical_edge"
    # the silhouette itself is never offered as a planar datum
    assert any(c["classification"] == "view_dependent" for c in p["candidates"])
    client.post(f"/api/sessions/{sid}/datums/confirm")
    err, ov = target_error_mm(client, sid, reg, cams, [(100, 345), (200, 345), (200, 295), (100, 295)])
    assert err < 0.6
    rep = ov["reports"]["PLC-DEMO-201"]
    assert err < rep["uncertainty"]["expanded_mm"]
    assert not any(g["code"] == "datum_side_needs_3d" and not g["passed"] for g in rep["gates"])


def test_rounded_edge_from_a_single_view_stays_a_preview(client):
    scene = curved_panel_scene()
    cams = sweep(1, target=(230.0, 0.0, 300.0))
    sid, reg = session(client, scene, cams, operation="DEMO-CURVE-01", placard="PLC-DEMO-201")
    out = propose(client, sid)
    s = out["session"]
    # an operator can still draw a line on the silhouette, but that is not the tangent line
    fid = reg["registration"]["reference_frame_id"]
    cam = cams[0]
    r = client.post(f"/api/sessions/{sid}/datums/side", json={"frame_id": fid, "p0": px(cam, -25, 80).tolist(), "p1": px(cam, -25, 600).tolist()})
    assert r.status_code == 200
    if "floor" not in s["datums"]:
        client.post(f"/api/sessions/{sid}/datums/floor", json={"frame_id": fid, "p0": px(cam, 60, 0).tolist(), "p1": px(cam, 560, 0).tolist()})
    client.post(f"/api/sessions/{sid}/datums/confirm")
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "research_preview"
    assert not next(g for g in rep["gates"] if g["code"] == "datum_side_needs_3d")["passed"]


def test_tapered_edge_is_not_silently_orthogonalised(client):
    scene = curved_panel_scene(taper_deg=5.0)  # the edge leans 5 degrees: not perpendicular to the floor seam
    cams = sweep(7, span=62.0, target=(230.0, 0.0, 300.0))
    sid, reg = session(client, scene, cams, operation="DEMO-CURVE-01", placard="PLC-DEMO-201")
    out = propose(client, sid)
    s = out["session"]
    assert out["proposals"]["cylinder"]["ok"]
    lf = s["local_frame"]
    # raw angle is preserved and compared with the configured 1 degree limit
    assert lf["raw_angle_deg"] == pytest.approx(85.0, abs=0.4) and lf["axis_disagreement_deg"] == pytest.approx(5.0, abs=0.4)
    assert not lf["valid"] and "axis_disagreement_exceeds_limit" in lf["reasons"]
    client.post(f"/api/sessions/{sid}/datums/confirm")
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"] == "blocked" and not next(g for g in rep["gates"] if g["code"] == "local_frame_valid")["passed"]


def test_symmetric_edges_are_resolved_only_by_the_configured_side(client):
    # two identical joint seams, left and right of the tags: geometry alone cannot say which is the datum
    scene = flat_panel_scene(extra_seams=(Seam((560.0, 0.0), (560.0, 1100.0), width=2.0, darkness=0.7),))
    cams = sweep(5, target=(280.0, 0.0, 300.0))
    sid, reg = session(client, scene, cams)
    out = propose(client, sid)
    gxz, grot = gauge_of(reg, scene)
    sd = out["session"]["datums"]["side"]
    # DEMO-FLAT-01 says the side datum is on the LEFT when facing the panel with the floor down
    assert abs(scene_line_of(sd, reg, scene, gxz, grot)[:, 0].mean()) < 0.3
    assert out["session"]["local_frame"]["side_of_panel"] == "left"
    # every offered pair respects the configured handedness
    assert all(pr["side_of_panel"] == "left" for pr in out["proposals"]["pairs"])
    # the right-hand seam is offered as a floor/side pair only with the other handedness, i.e. not at all
    right = [c for c in out["proposals"]["candidates"] if abs(scene_line_of(c, reg, scene, gxz, grot)[:, 0].mean() - 560) < 2]
    assert right and all(c["id"] not in [pr["side"] for pr in out["proposals"]["pairs"]] for c in right)


def test_operator_drawn_line_is_verified_in_3d_too(client):
    scene = flat_panel_scene()
    cams = sweep()
    sid, reg = session(client, scene, cams)
    fid = reg["registration"]["reference_frame_id"]
    frames = [f["id"] for f in client.get(f"/api/sessions/{sid}").json()["frames"]]
    cam = cams[frames.index(fid)]
    d = client.post(f"/api/sessions/{sid}/datums/floor", json={"frame_id": fid, "p0": (px(cam, 60, 0) + [0, 4]).tolist(), "p1": (px(cam, 560, 0) + [0, -3]).tolist()}).json()["datum"]
    assert d["on_plane_evidence"] == "verified_multiview" and d["image_fit"]["n_views"] == 5


def test_proposals_need_a_registration(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    assert client.post(f"/api/sessions/{sid}/datums/propose").status_code == 409
    assert client.post(f"/api/sessions/{sid}/datums/select", json={"kind": "floor", "candidate_id": "c0"}).status_code == 409
