"""The flashlight experiment through the API (prompt 07): research evidence, never guidance."""

import json
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from lumen.api.app import create_app
from lumen.services.sessions import SessionService
from lumen.storage.base import MemorySessionStore
from tests.conftest import calibration_for, jpeg_bytes, make_settings, render_cached
from tests.unit.test_photometric import edge_scene, stations

MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}


def start(c, torch="on", radius=35.0):
    scene, cams = edge_scene(radius=radius), stations()
    sid = c.post("/api/sessions", json={}).json()["id"]
    assert c.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-CURVE-01", "placard_id": "PLC-DEMO-201"}).status_code == 200
    c.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    c.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cams[0])})
    for i, cam in enumerate(cams):
        r = c.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(render_cached(scene, cam, seed=i).image, 95),
                   headers={"content-type": "image/jpeg", "x-lumen-meta": quote(json.dumps({"torch": torch}))})
        assert r.status_code == 200 and r.json()["frame"]["torch"] == torch, r.text
    return sid


def test_the_experiment_reports_next_to_the_outline_method_and_changes_nothing(client):
    sid = start(client)
    # before the photos are analysed there is nothing to look at
    r = client.post(f"/api/sessions/{sid}/research/photometric")
    assert r.status_code == 409 and r.json()["error"]["code"] in ("not_registered", "needs_camera_positions")
    assert client.post(f"/api/sessions/{sid}/register").status_code == 200
    r = client.post(f"/api/sessions/{sid}/research/photometric")
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_side_datum"
    client.post(f"/api/sessions/{sid}/datums/propose")
    client.post(f"/api/sessions/{sid}/datums/confirm")
    s0 = client.get(f"/api/sessions/{sid}").json()
    side0 = s0["datums"]["side"]
    assert side0["feature_kind"] == "cylinder_model"
    # the outline of a lit rounded edge is not trusted silently
    assert any("phone light on" in n for n in side0["notes"])
    rep0 = client.get(f"/api/sessions/{sid}/report").json()

    out = client.post(f"/api/sessions/{sid}/research/photometric").json()
    ph = out["photometric"]
    assert ph["label"] == "research_evidence_only" and ph["guidance_enabled"] is False and "not used for any datum" in ph["notice"]
    assert ph["n_photos"] == 4 and ph["n_torch_on"] == 4 and ph["result"]["status"] in ("supported", "needs_view", "unsupported")
    res = ph["result"]
    assert res["label"] == "research_evidence_only" and len(res["profiles"]) <= 4
    if res["status"] == "supported":
        # true radius 35 mm; the estimate sits within a millimetre of the outline-method datum
        assert res["radius_mm"] == pytest.approx(35.0, abs=1.0) and abs(ph["offset_from_side_datum_mm"]) < 1.5
        assert ph["comparison_with_outline_method"] is not None
    else:
        assert res["reasons"]
    # kept with the session as research; the datum, the report and the target are exactly as before
    s1 = out["session"]
    assert s1["research"]["photometric"]["computed_at"] == ph["computed_at"]
    assert s1["datums"] == s0["datums"] and s1["local_frame"] == s0["local_frame"]
    rep1 = client.get(f"/api/sessions/{sid}/report").json()
    assert rep1["status"] == rep0["status"] and [(g["code"], g["passed"]) for g in rep1["gates"]] == [(g["code"], g["passed"]) for g in rep0["gates"]]
    assert not any("photometric" in g["code"] or "torch" in g["code"] for g in rep1["gates"])
    # a new analysis of the photos discards the research result with everything else computed from them
    client.put(f"/api/sessions/{sid}/tags", json={"nominal_mm": 50.0, "measured_mm": {"3": 50.1, "7": 50.0, "11": 50.0, "20": 50.0}})
    assert "photometric" not in client.get(f"/api/sessions/{sid}").json()["research"]


def test_without_torch_photos_it_asks_for_them_and_the_workflow_goes_on(client):
    sid = start(client, torch="off")
    client.post(f"/api/sessions/{sid}/register")
    client.post(f"/api/sessions/{sid}/datums/propose")
    side = client.get(f"/api/sessions/{sid}").json()["datums"]["side"]
    assert not any("phone light on" in n for n in side["notes"])
    ph = client.post(f"/api/sessions/{sid}/research/photometric").json()["photometric"]
    assert ph["n_torch_on"] == 0 and ph["result"]["status"] == "needs_view" and "no_torch_on_views" in ph["result"]["reasons"]
    assert ph["result"]["tangent"] is None and ph["offset_from_side_datum_mm"] is None
    assert any(q.get("code") == "torch_on_sweep" for q in ph["result"]["requests"])


def test_switched_off_by_configuration(library):
    svc = SessionService(MemorySessionStore(), make_settings(photometric_enabled=False), library)
    with TestClient(create_app(svc.settings, svc)) as c:
        assert c.get("/api/config").json()["features"]["photometric_research"] is False
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}/research/photometric")
        assert r.status_code == 403 and r.json()["error"]["code"] == "photometric_disabled"
