"""Entering and revising operation records through the API (prompt 08)."""

import copy

import pytest

from lumen.operations.library import OperationLibrary
from lumen.services.sessions import SessionService
from lumen.storage.base import MemorySessionStore
from lumen.synth.fixtures import flat_panel_scene, phone_camera
from tests.conftest import BUNDLED_OPS, PrefixProxy, calibration_for, jpeg_bytes, make_settings, px, render_cached

W, H = 1600, 1200
MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}


@pytest.fixture()
def app_and_library(tmp_path):
    from lumen.api.app import create_app

    lib = OperationLibrary(BUNDLED_OPS, tmp_path / "library")
    lib.persistent = False
    svc = SessionService(MemorySessionStore(), make_settings(), lib)
    return create_app(svc.settings, svc), lib


@pytest.fixture()
def client(app_and_library):
    from fastapi.testclient import TestClient

    with TestClient(app_and_library[0]) as c:
        yield c


def record(client, op_id="LOCAL-A", revision="A", status="draft"):
    rec = copy.deepcopy(client.get("/api/operations/DEMO-FLAT-01").json()["operation"])
    rec.update(operation_id=op_id, revision=revision, status=status)
    return rec


def test_validate_reports_what_is_missing_and_shows_the_values_in_mm(client):
    ops = client.get("/api/operations").json()
    assert ops["editable"] and ops["storage"] == {"editable": True, "persistent": False}
    rec = record(client)
    rec["placards"][0]["position"]["x"] = {"value": 9.5, "unit": "in"}
    rec["placards"][0]["tolerance"]["position"] = None
    rec["datums"]["side"]["positive"] = None
    out = client.post("/api/operations/validate", json={"operation": rec}).json()
    assert out["can_release"] is False and out["can_save"] is True and out["existing"] is None and out["problems"] == []
    assert out["summary"]["issues"] == {"datums": ["side_datum_sign_missing"], "placard:PLC-DEMO-101": ["missing_position_tolerance"]}
    assert out["placards_resolved"][0]["x_mm"] == pytest.approx(241.3) and out["placards_resolved"][0]["tol_position_mm"] is None, "a missing tolerance stays missing"
    # nothing was stored by asking
    assert len(client.get("/api/operations").json()["operations"]) == len(ops["operations"])
    # a value without a usable unit is a problem with its place and reason, not a guess
    rec["placards"][0]["position"]["x"] = {"value": 9.5, "unit": "cubits"}
    bad = client.post("/api/operations/validate", json={"operation": rec}).json()
    assert bad["can_save"] is False and bad["summary"] is None
    assert [p["where"] for p in bad["problems"]] == ["placards.0.position.x.unit"] and "cubits" in bad["problems"][0]["message"]
    r = client.post("/api/operations", json={"operation": rec})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_operation_record" and r.json()["error"]["details"]["problems"] == bad["problems"]
    # while the ID is still empty the rest is checked all the same
    rec = record(client, "")
    rec["placards"][0]["tolerance"]["rotation"] = None
    part = client.post("/api/operations/validate", json={"operation": rec}).json()
    assert [p["where"] for p in part["problems"]] == ["operation_id"] and "1 to 64 characters" in part["problems"][0]["message"]
    assert part["summary"]["issues"] == {"placard:PLC-DEMO-101": ["missing_rotation_tolerance"]} and part["can_save"] is False
    # fields that are not part of a record are named
    rec = record(client)
    rec["placards"][0]["tolerance"]["position_mm"] = 2
    extra = client.post("/api/operations/validate", json={"operation": rec}).json()
    assert extra["problems"] == [{"where": "placards.0.tolerance.position_mm", "message": "is not a field of an operation record"}]
    # the ID and revision of a bundled sample, or of a released revision, cannot be saved over
    same = client.post("/api/operations/validate", json={"operation": record(client, "DEMO-FLAT-01", "A", "released")}).json()
    assert same["existing"]["origin"] == "bundled" and same["can_save"] is False and same["can_release"] is False


def test_draft_release_revision_and_supersede_through_the_api(client, app_and_library):
    lib = app_and_library[1]
    rec = record(client)
    rec["synthetic"] = True  # a copy of a sample is not a sample
    saved = client.post("/api/operations", json={"operation": rec}).json()
    assert saved["origin"] == "local" and saved["summary"]["status"] == "draft" and saved["summary"]["synthetic"] is False
    listed = {(o["operation_id"], o["revision"]): o for o in client.get("/api/operations").json()["operations"]}
    assert listed[("LOCAL-A", "A")]["origin"] == "local" and listed[("DEMO-FLAT-01", "A")]["origin"] == "bundled"
    # a draft can be changed in place
    rec["placards"][0]["position"]["x"]["value"] = 251
    assert client.post("/api/operations", json={"operation": rec}).json()["placards_resolved"][0]["x_mm"] == 251.0
    # release, then the revision is fixed
    rec["status"] = "released"
    assert client.post("/api/operations", json={"operation": rec}).json()["summary"]["status"] == "released"
    rec["placards"][0]["position"]["x"]["value"] = 252
    r = client.post("/api/operations", json={"operation": rec})
    assert r.status_code == 409 and r.json()["error"]["code"] == "released_revision_immutable"
    assert client.post("/api/operations/delete", json={"operation_id": "LOCAL-A", "revision": "A"}).json()["error"]["code"] == "only_drafts_can_be_deleted"
    # the change goes into revision B; A is superseded, and the ID alone then means B
    rec["revision"] = "B"
    assert client.post("/api/operations", json={"operation": rec}).status_code == 200
    assert client.get("/api/operations/LOCAL-A").status_code == 409
    sup = client.post("/api/operations/supersede", json={"operation_id": "LOCAL-A", "revision": "A"}).json()
    assert sup["summary"]["status"] == "superseded" and sup["placards_resolved"][0]["x_mm"] == 251.0
    d = client.get("/api/operations/LOCAL-A").json()
    assert d["operation"]["revision"] == "B" and d["revisions"] == [{"revision": "A", "status": "superseded"}, {"revision": "B", "status": "released"}]
    # an incomplete record can be a draft, not a release
    inc = record(client, "LOCAL-B", "A", "released")
    inc["placards"][0]["tolerance"] = {}
    r = client.post("/api/operations", json={"operation": inc})
    assert r.status_code == 422 and r.json()["error"]["code"] == "incomplete_record_cannot_be_released" and "placard:PLC-DEMO-101" in r.json()["error"]["details"]["issues"]
    inc["status"] = "draft"
    assert client.post("/api/operations", json={"operation": inc}).status_code == 200
    assert client.post("/api/operations/delete", json={"operation_id": "LOCAL-B", "revision": "A"}).json() == {"deleted": True}
    assert client.get("/api/operations/LOCAL-B").status_code == 404
    # bundled samples are never changed
    for path in ("/api/operations/supersede", "/api/operations/delete"):
        assert client.post(path, json={"operation_id": "DEMO-FLAT-01", "revision": "A"}).json()["error"]["code"] == "bundled_record_immutable"
    assert client.post("/api/operations", json={"operation": record(client, "DEMO-FLAT-01", "A", "released")}).status_code == 409
    audit = client.get("/api/operations-audit", params={"operation_id": "LOCAL-A"}).json()["audit"]
    assert [(e["action"], e["revision"]) for e in audit] == [("saved", "A"), ("saved", "A"), ("saved", "A"), ("saved", "B"), ("superseded", "A")]
    assert lib.get("LOCAL-A").revision == "B"


def test_changes_are_attributed_to_the_connect_user(app_and_library):
    from fastapi.testclient import TestClient

    app, _ = app_and_library
    with TestClient(PrefixProxy(app, user="dana")) as c:
        rec = copy.deepcopy(c.get("/content/3f2a9c1e/api/operations/DEMO-FLAT-01").json()["operation"])
        rec.update(operation_id="LOCAL-U", revision="A", status="draft")
        assert c.post("/content/3f2a9c1e/api/operations", json={"operation": rec}).status_code == 200
        assert c.get("/content/3f2a9c1e/api/operations-audit").json()["audit"][-1]["author"] == "dana"


def test_a_session_measures_with_a_draft_only_as_a_preview_and_keeps_its_own_copy(client):
    scene = flat_panel_scene()
    cam = phone_camera((430, -720, 430), width=W, height=H, roll_deg=3, dist=(0.06, -0.12, 0.0004, -0.0003, 0.0))
    img = render_cached(scene, cam, seed=1).image
    rec = record(client, "LOCAL-S", "A", "draft")
    client.post("/api/operations", json={"operation": rec})

    def run(revision):
        sid = client.post("/api/sessions", json={"mode": "real"}).json()["id"]
        assert client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "LOCAL-S", "revision": revision, "placard_id": "PLC-DEMO-101"}).status_code == 200
        client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
        client.put(f"/api/sessions/{sid}/calibration", json={"profile": calibration_for(cam)})
        f = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(img), headers={"content-type": "image/jpeg"}).json()["frame"]
        client.post(f"/api/sessions/{sid}/register")
        fl = [px(cam, 60, 0).tolist(), px(cam, 560, 0).tolist()]
        sd = [px(cam, 0, 80).tolist(), px(cam, 0, 620).tolist()]
        client.post(f"/api/sessions/{sid}/datums/floor", json={"frame_id": f["id"], "p0": fl[0], "p1": fl[1]})
        client.post(f"/api/sessions/{sid}/datums/side", json={"frame_id": f["id"], "p0": sd[0], "p1": sd[1]})
        client.post(f"/api/sessions/{sid}/datums/confirm")
        return sid, client.get(f"/api/sessions/{sid}/report", params={"frame": f["id"]}).json()

    sid, rep = run("A")
    failed = {g["code"] for g in rep["gates"] if not g["passed"]}
    assert "operation_record_released" in failed and rep["status"] == "research_preview"
    # the session keeps the record as it was when it was chosen
    before = client.get(f"/api/sessions/{sid}").json()["operation"]
    rec["placards"][0]["position"]["x"]["value"] = 300
    client.post("/api/operations", json={"operation": rec})
    after = client.get(f"/api/sessions/{sid}").json()["operation"]
    assert after["content_hash"] == before["content_hash"] and after["operation"]["placards"][0]["position"]["x"]["value"] == 240
    # released: the same measurement is no longer held back by the record
    rec["status"] = "released"
    client.post("/api/operations", json={"operation": rec})
    _, rep2 = run("A")
    assert "operation_record_released" not in {g["code"] for g in rep2["gates"] if not g["passed"]}
