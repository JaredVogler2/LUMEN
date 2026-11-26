"""A job on a host that forgets everything between two requests: the client carries it.

``forgetful`` talks to ``StatelessHost``: every request is served by an application that was made for it and has
never seen another one, and a request or an answer above 4.5 MB is refused as a function platform refuses it.
Whatever works here does not depend on anything a server remembers.

The same job on a server that keeps it gives the same numbers (the first test), and the tests of live placement
run unchanged with a carried job in ``test_carried_placement.py`` and ``test_carried_hold.py``.
"""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.config import load_settings
from lumen.storage.carried import CarriedStore, pack
from tests.carry_client import CARRY_TYPE, CarryClient, StatelessHost, carried_app, forgetful_client, warm_client
from tests.conftest import jpeg_bytes, make_settings, render_cached
from tests.integration.test_placement_flow import MEASURED, SQUARE, live, placed, ready  # noqa: F401  (ready is a fixture)

PLACARD = "PLC-DEMO-101"


@pytest.fixture()
def forgetful():
    yield from forgetful_client()


@pytest.fixture()
def keeping(service):
    """A server of the ordinary kind: it keeps the job itself."""
    from lumen.api.app import create_app

    with TestClient(create_app(service.settings, service)) as c:
        yield c


@pytest.fixture()
def client(forgetful):
    """For ``ready`` (imported above): the job is built through the carrier."""
    return forgetful


def demo_job(c):
    """The demo job from the first request to the record; what a client sees of it."""
    sid = c.post("/api/sessions", json={"mode": "demo"}).json()["id"]
    assert c.post(f"/api/sessions/{sid}/demo/load", json={"scenario": "flat"}).status_code == 200
    reg = c.post(f"/api/sessions/{sid}/register")
    assert reg.status_code == 200, reg.text
    prop = c.post(f"/api/sessions/{sid}/datums/propose")
    assert prop.status_code == 200, prop.text
    assert c.post(f"/api/sessions/{sid}/datums/confirm").status_code == 200
    s = c.get(f"/api/sessions/{sid}").json()
    fid = s["registration"]["frames_used"][0]
    ov = c.get(f"/api/sessions/{sid}/overlay", params={"frame": fid, "view": "image", "uncertainty": "true", "placards": PLACARD})
    assert ov.status_code == 200, ov.text
    rect = c.get(f"/api/sessions/{sid}/frames/{fid}/rectified", params={"max": 1600})
    assert rect.status_code == 200 and rect.headers["content-type"] == "image/jpeg" and rect.content[:2] == b"\xff\xd8"
    vf = c.post(f"/api/sessions/{sid}/demo/verification").json()["frame"]["id"]
    ver = c.post(f"/api/sessions/{sid}/verify", json={"frame_id": vf, "placard_id": PLACARD, "snap": True})
    assert ver.status_code == 200, ver.text
    exp = c.get(f"/api/sessions/{sid}/export").json()
    return {"sid": sid, "session": ver.json()["session"], "overlay": ov.json(), "rect_spec": rect.headers.get("x-lumen-rect"), "export": exp, "frame": fid}


def test_a_job_on_a_host_that_forgets_everything_gives_what_a_keeping_server_gives(keeping, forgetful):
    kept = demo_job(keeping)
    carried = demo_job(forgetful)
    a, b = kept["session"], carried["session"]
    # the same registration, the same two lines, the same record
    assert b["registration"]["n_tags"] == a["registration"]["n_tags"] == 4 and b["registration"]["rms_px"] == pytest.approx(a["registration"]["rms_px"], abs=1e-6)
    for kind in ("floor", "side"):
        assert b["datums"][kind]["confirmed"] and b["datums"][kind]["n"] == pytest.approx(a["datums"][kind]["n"], abs=1e-9) and b["datums"][kind]["d"] == pytest.approx(a["datums"][kind]["d"], abs=1e-7)
    va, vb = a["verifications"][0], b["verifications"][0]
    assert vb["observation"] == va["observation"] == "indeterminate"
    for key in ("dx_mm", "dz_mm", "rotation_rad", "radial_mm"):
        assert vb[key] == pytest.approx(va[key], abs=1e-6), key
    assert vb["uncertainty"]["expanded_mm"] == pytest.approx(va["uncertainty"]["expanded_mm"], abs=0.02)
    # the same target on the photo, with the same status, and the same rectified picture's geometry
    ra, rb = kept["overlay"]["reports"][PLACARD], carried["overlay"]["reports"][PLACARD]
    assert rb["status"] == ra["status"] == "prototype_guidance" and [g["code"] for g in rb["gates"] if not g["passed"]] == [g["code"] for g in ra["gates"] if not g["passed"]]
    ta = next(p for p in kept["overlay"]["primitives"] if p["kind"] == "target")
    tb = next(p for p in carried["overlay"]["primitives"] if p["kind"] == "target")
    assert np.allclose(tb["corners"], ta["corners"], atol=1e-6)
    assert carried["rect_spec"] == kept["rect_spec"] and carried["rect_spec"], "the header that describes the rectified picture travels with it"
    assert carried["export"]["images_included"] is False and len(carried["export"]["verifications"]) == 1
    # ---- and the host: every request met an application that had seen nothing, and none was too large for a function
    host = forgetful.host
    # (eleven requests from the first to the export, none of them sent twice)
    assert host.requests == forgetful.round_trips == 11
    assert host.largest_request < 1_300_000 and host.largest_answer < 1_300_000, (host.largest_request, host.largest_answer)
    # what the client holds is the job: one document of some tens of kilobytes and the six photos (five and the record's)
    sid = carried["sid"]
    assert 40_000 < len(forgetful.states[sid][0]) < 200_000 and sorted(forgetful.blobs[sid]) == [f"f00{i}.jpg" for i in range(1, 7)]
    # photos went up only for the steps that look at them: all five to find the lines, one for the rectified picture,
    # one for the record; and no step had to ask for a photo it was not given
    assert forgetful.sent_pictures == 5 + 1 + 1 and forgetful.asked_again == 0, (forgetful.sent_pictures, forgetful.asked_again)
    assert forgetful.delete(f"/api/sessions/{sid}").json() == {"deleted": True}
    assert forgetful.states == {} and forgetful.blobs == {}


def test_live_frames_are_measured_against_a_job_the_client_brings(forgetful, ready):
    # `ready` used this module's client: photos of the bare panel went up one by one, the job came back each time
    sid, scene = ready
    assert len(forgetful.blobs[sid]) == 5
    sent_before = forgetful.sent_pictures
    meta: dict = {}
    seen = []
    for k, (dx, dz, rot) in enumerate([(12.0, -7.0, 3.0), (2.2, -1.0, 0.6), (0.4, -0.3, 0.1)]):
        out = live(forgetful, sid, render_cached(placed(scene, dx, dz, rot), SQUARE, seed=201 + k).image, meta)
        p = out["placement"]
        assert p["found"] and p["dx_mm"] == pytest.approx(dx, abs=0.12) and p["dz_mm"] == pytest.approx(dz, abs=0.12) and p["rotation_deg"] == pytest.approx(rot, abs=0.08)
        # nothing is held between two frames here: each is served by an application that never saw the one before
        assert "held" not in out["tracking"] and out["timing_ms"]["tags"] > 0
        seen.append((p["observation"], p["in_position"]))
        meta = {"placement_hint": {"corners_plane": p["corners_plane"], "polarity": p["polarity"]}, "edge_shift": out["tracking"]["edge_shift"]}
    assert seen == [("outside_tolerance_observed", False), ("indeterminate", False), ("within_tolerance_observed", True)]
    # a live frame brings the job's state and itself; no photo of the job travels with it
    assert forgetful.sent_pictures == sent_before
    assert forgetful.host.largest_request < 4_000_000


def test_a_step_asks_for_the_photos_it_needs_and_a_job_must_come_with_its_request(forgetful):
    sid = forgetful.post("/api/sessions", json={"mode": "demo"}).json()["id"]
    forgetful.post(f"/api/sessions/{sid}/demo/load", json={"scenario": "flat"})
    assert forgetful.post(f"/api/sessions/{sid}/register").status_code == 200
    # the lines are searched in the photos: without them the step says which one it misses
    before = forgetful.round_trips
    bare = forgetful._send("POST", f"api/sessions/{sid}/datums/propose", sid, b"", "", None, [], None)
    assert bare.status_code == 428 and bare.json()["error"]["code"] == "pictures_needed"
    assert bare.json()["error"]["details"]["id"] == sid and bare.json()["error"]["details"]["blobs"][0] in forgetful.blobs[sid]
    assert forgetful.round_trips == before + 1
    # nothing was changed by the request that failed: the state the client holds is the one from before
    state_before = forgetful.states[sid][0]
    # a client that does not know what the step needs is told photo by photo, and gets there
    slow = forgetful.request("POST", f"/api/sessions/{sid}/datums/propose", attach=[])
    assert slow.status_code == 200 and forgetful.asked_again == 5 and forgetful.states[sid][0] != state_before
    # one that knows sends them at once
    asked = forgetful.asked_again
    assert forgetful.post(f"/api/sessions/{sid}/datums/propose").status_code == 200 and forgetful.asked_again == asked
    # a request for a job that does not bring the job: there is none here
    http = forgetful.http
    r = http.post("/api/carry", content=pack({"method": "GET", "path": f"api/sessions/{sid}", "states": [], "blobs": [], "body": 0}, [b""]), headers={"content-type": CARRY_TYPE})
    assert r.status_code == 404 and r.json()["error"]["code"] == "session_not_found" and "did not come with the request" in r.json()["error"]["message"]
    # and one that does not come through the carrier at all
    r = http.get(f"/api/sessions/{sid}")
    assert r.status_code == 404 and "travels with the phone" in r.json()["error"]["message"]
    r = http.post("/api/sessions", json={})
    assert r.status_code == 404 and r.json()["error"]["code"] == "session_not_found"


def test_a_picture_that_is_not_the_one_that_was_measured_is_refused(forgetful):
    sid = forgetful.post("/api/sessions", json={"mode": "demo"}).json()["id"]
    forgetful.post(f"/api/sessions/{sid}/demo/load", json={"scenario": "flat"})
    forgetful.post(f"/api/sessions/{sid}/register")
    # the client sends the second photo under the name of the first
    forgetful.blobs[sid]["f001.jpg"] = forgetful.blobs[sid]["f002.jpg"]
    r = forgetful.post(f"/api/sessions/{sid}/datums/propose")
    assert r.status_code == 409 and r.json()["error"]["code"] == "picture_mismatch" and "f001" in r.json()["error"]["message"]
    fid = forgetful.state(sid)["registration"]["frames_used"][0]
    r = forgetful.get(f"/api/sessions/{sid}/frames/{fid}/rectified", attach=["f001.jpg"])
    assert r.status_code == 409 and r.json()["error"]["code"] == "picture_mismatch"


def test_only_requests_of_a_job_go_through_the_carrier_and_only_well_formed_ones(forgetful):
    http = forgetful.http

    def send(header, sections=(b"",), raw=None):
        return http.post("/api/carry", content=raw if raw is not None else pack(header, list(sections)), headers={"content-type": CARRY_TYPE})

    for path in ("api/config", "api/health", "api/carry", "static/js/main.js", "", "api/sessions/../config", "api/operations", "//example.invalid/api/sessions", "api/sessions/x y"):
        r = send({"method": "GET", "path": path, "states": [], "blobs": [], "body": 0})
        assert r.status_code == 400 and r.json()["error"]["code"] == "not_carried", path
    assert send({"method": "TRACE", "path": "api/sessions", "states": [], "blobs": [], "body": 0}).status_code == 400
    assert http.get("/api/carry").status_code == 405
    for raw in (b"", b"hello", b"LUMC\x01\x00\x00\x00\x05[1,2]", b"LUMC\x01\xff\xff\xff\xff{}", b"LUMC\x02\x00\x00\x00\x02{}"):
        r = send(None, raw=raw)
        assert r.status_code == 422 and r.json()["error"]["code"] == "not_a_carried_request", raw
    # an envelope that is shorter or longer than it says
    assert send({"method": "POST", "path": "api/sessions", "states": [], "blobs": [], "body": 10}, [b"abc"]).json()["error"]["code"] == "not_a_carried_request"
    assert send({"method": "POST", "path": "api/sessions", "states": [], "blobs": [], "body": 0}, [b"abc"]).json()["error"]["code"] == "not_a_carried_request"
    # a state that is not a job's, or not the job's it is sent for
    sid = forgetful.post("/api/sessions", json={}).json()["id"]
    raw_state = forgetful.states[sid][0]
    other = "A" * 24
    for st_id, payload, code in ((sid, b"[1]", "invalid_state"), (sid, b"\xff\xfe", "invalid_state"), (other, raw_state, "invalid_state"), ("short", raw_state, "session_not_found")):
        r = send({"method": "GET", "path": f"api/sessions/{sid}", "states": [{"id": st_id, "size": len(payload), "encoding": "identity"}], "blobs": [], "body": 0}, [payload, b""])
        assert r.status_code in (404, 422) and r.json()["error"]["code"] == code, (st_id, payload[:10], r.text)
    r = send({"method": "GET", "path": f"api/sessions/{sid}", "states": [{"id": sid, "size": 5, "encoding": "deflate"}], "blobs": [], "body": 0}, [b"nozip", b""])
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_state"
    # a state of the right form with content no job has: an error for this request, never a server that breaks
    import json as _json

    doc = _json.loads(raw_state)
    doc["frames"] = "many"
    bad = _json.dumps(doc).encode()
    r = send({"method": "GET", "path": f"api/sessions/{sid}", "states": [{"id": sid, "size": len(bad), "encoding": "identity"}], "blobs": [], "body": 0}, [bad, b""])
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_state", r.text
    # a picture for a job that did not come
    r = send({"method": "GET", "path": f"api/sessions/{sid}", "states": [], "blobs": [{"id": sid, "name": "f001.jpg", "size": 3}], "body": 0}, [b"abc", b""])
    assert r.status_code == 422


def test_with_a_secret_a_state_is_taken_back_only_as_this_server_wrote_it():
    for c in forgetful_client(state_secret="s3cret-of-this-installation"):
        sid = c.post("/api/sessions", json={}).json()["id"]
        raw, sig = c.states[sid]
        assert len(sig) == 64
        assert c.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": PLACARD}).status_code == 200
        assert c.states[sid][1] != sig, "every state has its own signature"
        good = c.states[sid]
        # one character changed
        c.states[sid] = (good[0].replace(b"DEMO-FLAT-01", b"DEMO-FLAT-02", 1), good[1])
        r = c.get(f"/api/sessions/{sid}")
        assert r.status_code == 409 and r.json()["error"]["code"] == "state_not_ours"
        # an older state of the same job is the server's own writing, and is taken (it is the client's to bring)
        c.states[sid] = (raw, sig)
        assert c.get(f"/api/sessions/{sid}").json()["operation"] is None
        # no signature at all
        c.states[sid] = (good[0], "")
        assert c.get(f"/api/sessions/{sid}").status_code == 409
        c.states[sid] = good
        assert c.get(f"/api/sessions/{sid}").json()["operation"]["operation"]["operation_id"] == "DEMO-FLAT-01"
    # a server with another secret does not take it either
    for other in forgetful_client(state_secret="another"):
        other.states[sid] = good
        other.blobs[sid] = {}
        assert other.get(f"/api/sessions/{sid}").status_code == 409


def test_what_does_not_fit_one_request_is_refused_in_words():
    from lumen.config import Limits

    for c in warm_client(limits=Limits(carry_max_request_bytes=600_000, carry_max_response_bytes=700_000)):
        sid = c.post("/api/sessions", json={"mode": "demo"}).json()["id"]
        # five photos of 185 kB each come back from the demo: more than this host's answer may be
        r = c.post(f"/api/sessions/{sid}/demo/load", json={"scenario": "flat"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "answer_too_large" and "0.7 MB" in r.json()["error"]["message"]
        assert c.blobs.get(sid) == {}, "an answer that was refused brought nothing"
        # and a request above the limit is refused before anything is read from it
        big = c.post(f"/api/sessions/{sid}/frames", content=b"\xff\xd8" + b"0" * 700_000, headers={"content-type": "image/jpeg"})
        assert big.status_code == 413 and big.json()["error"]["code"] == "request_too_large" and "0.6 MB" in big.json()["error"]["message"]


def test_a_video_needs_a_server_that_keeps_the_job(forgetful):
    sid = forgetful.post("/api/sessions", json={}).json()["id"]
    r = forgetful.post(f"/api/sessions/{sid}/video", content=b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, headers={"content-type": "video/mp4"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_available_here" and "photos" in " ".join(r.json()["error"]["suggestions"]).lower()
    cfg = forgetful.get("/api/config").json()
    assert cfg["sessions"]["kept_by"] == "client" and cfg["sessions"]["video"] is False and cfg["sessions"]["works"] is True
    # the sizes a browser keeps to, so that nine photos and the job fit one request
    s = cfg["sessions"]
    assert s["max_request_bytes"] == 4_000_000 and s["photo_max_edge"] == 1600 and 9 * s["photo_max_bytes"] + 300_000 <= s["max_request_bytes"]
    # no local records where no disk would keep them
    assert cfg["features"]["library_editable"] is False


def test_a_server_that_keeps_jobs_takes_no_carried_requests(keeping):
    r = keeping.post("/api/carry", content=pack({"method": "POST", "path": "api/sessions", "states": [], "blobs": [], "body": 2, "content_type": "application/json"}, [b"{}"]), headers={"content-type": CARRY_TYPE})
    assert r.status_code == 409 and r.json()["error"]["code"] == "job_kept_by_server"
    assert keeping.get("/api/config").json()["sessions"] == {"kept_by": "server", "works": True, "video": True}


def test_where_the_host_forgets_the_client_carries_unless_the_administrator_says_otherwise(monkeypatch):
    for var in ("VERCEL", "AWS_LAMBDA_FUNCTION_NAME", "FUNCTIONS_WORKER_RUNTIME", "NETLIFY", "LUMEN_STORAGE", "CONNECT_SERVER", "RSTUDIO_PRODUCT", "POSIT_PRODUCT"):
        monkeypatch.delenv(var, raising=False)
    assert load_settings().storage == "file" and load_settings().sessions == {"kept_by": "server", "works": True, "video": True}
    monkeypatch.setenv("VERCEL", "1")
    s = load_settings()
    assert s.storage == "client" and s.sessions["kept_by"] == "client" and s.sessions["works"] is True
    assert s.hosting["kind"] == "vercel" and s.hosting["keeps_sessions"] is False and s.hosting["message"] == ""
    with TestClient(carried_app()) as http:
        h = http.get("/api/health")
        assert h.status_code == 200 and h.json()["status"] == "ok" and h.json()["sessions"] == {"kept_by": "client"}
        assert h.json()["checks"]["host_keeps_sessions"] is False and h.json()["checks"]["jobs_can_be_kept"] is True
        assert h.json()["storage"]["adapter"] == "CarriedStore"
        assert '"sessions": "client"'.replace('"', "&quot;") in http.get("/").text
    # told to keep jobs on such a host, the server cannot work, and says how to change that
    monkeypatch.setenv("LUMEN_STORAGE", "file")
    s = load_settings()
    assert s.storage == "file" and s.sessions["works"] is False
    assert "forgets everything between two requests" in s.hosting["message"] and "remove LUMEN_STORAGE" in s.hosting["message"]
    monkeypatch.setenv("LUMEN_STORAGE", "client")
    monkeypatch.delenv("VERCEL")
    # ... and a host that could keep jobs may still leave them to the client
    assert load_settings().storage == "client" and load_settings().hosting["keeps_sessions"] is True
    assert isinstance(CarriedStore().list_ids(), list) and CarriedStore().list_ids() == []
    assert replace(make_settings(), storage="client").sessions["signed"] is False
