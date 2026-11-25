"""Capture services through the API: video frame selection, jobs, coaching, flash pairs (prompt 02)."""

import json
import time
from dataclasses import replace
from urllib.parse import quote

import numpy as np
import pytest

from lumen.config import Limits
from lumen.services.jobs import JobRunner
from lumen.synth import LightSpec
from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import calibration_for, jpeg_bytes, make_settings, px, render_cached, sweep_video_frames, write_video

MEASURED = {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}}


def new_session(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.put(f"/api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
    client.put(f"/api/sessions/{sid}/tags", json=MEASURED)
    return sid


def wait_job(client, sid, job, timeout=120.0):
    t0 = time.time()
    while job["state"] in ("queued", "running"):
        assert time.time() - t0 < timeout, "job did not finish"
        time.sleep(0.2)
        job = client.get(f"/api/sessions/{sid}/jobs/{job['id']}").json()
    return job


def meta(d):
    return {"x-lumen-meta": quote(json.dumps(d))}


@pytest.fixture(scope="module")
def sweep_file(tmp_path_factory):
    frames, cams, sharp = sweep_video_frames()
    path = write_video(str(tmp_path_factory.mktemp("cap") / "sweep.mp4"), frames, fps=12.0)
    return path, frames, cams, sharp


def test_video_upload_chooses_frames_then_registers(client, sweep_file):
    path, frames, cams, sharp = sweep_file
    sid = new_session(client)
    data = open(path, "rb").read()
    r = client.post(f"/api/sessions/{sid}/video", content=data, headers={"content-type": "video/mp4", **meta({"camera_label": "synthetic"})})
    assert r.status_code == 200, r.text
    job = wait_job(client, sid, r.json()["job"])
    assert job["state"] == "done", job
    res = job["result"]
    assert res["video"]["frames_decoded"] == len(frames) and res["candidates"] == 8
    assert 3 <= len(res["selected"]) <= 8 and not res["dropped_after_measurement"]
    s = client.get(f"/api/sessions/{sid}").json()
    assert len(s["frames"]) == len(res["selected"])
    assert all(f["source"] == "video_upload" and f["video_time_s"] is not None for f in s["frames"])
    assert all(len([m for m in f["markers"] if m["accepted"]]) == 4 for f in s["frames"])
    # the chosen frames are the sharp ones (middle of each hold), never a blurred neighbour
    times = sorted(f["video_time_s"] for f in s["frames"])
    sharp_times = [i / 12.0 for i in sharp]
    assert all(min(abs(t - st) for st in sharp_times) < 0.03 for t in times)
    # ... and they register: scale from the tags, 3-D checks from the spread of viewpoints
    reg = client.post(f"/api/sessions/{sid}/register").json()["registration"]
    assert reg["consistent"] and reg["n_tags"] == 4
    mv = reg["multiview"]
    assert mv["checks"]["diversity"]["max_view_angle_deg"] > 25.0
    # tag centres agree with the scene within half a millimetre (720p video frames, MPEG-4 compressed)
    lay = reg["layout"]
    g = lay["gauge_id"]
    truth = {3: (95, 125), 7: (455, 105), 11: (110, 495), 20: (450, 505)}
    P = {int(k): np.array(v["center"] if isinstance(v, dict) else v[:2]) for k, v in lay["poses"].items()}
    ids = sorted(P)
    d_est = np.array([np.linalg.norm(P[a] - P[b]) for a in ids for b in ids if a < b])
    d_true = np.array([np.linalg.norm(np.subtract(truth[a], truth[b])) for a in ids for b in ids if a < b])
    assert np.max(np.abs(d_est - d_true)) < 0.5, (d_est - d_true)
    assert g in ids


def test_video_upload_refuses_junk_and_oversize(service, library):
    from fastapi.testclient import TestClient

    from lumen.api.app import create_app
    from lumen.services.sessions import SessionService
    from lumen.storage.base import MemorySessionStore

    svc = SessionService(MemorySessionStore(), make_settings(limits=Limits(max_video_bytes=4000)), library)
    with TestClient(create_app(svc.settings, svc)) as c:
        sid = new_session(c)
        r = c.post(f"/api/sessions/{sid}/video", content=b"definitely not a video", headers={"content-type": "video/mp4"})
        assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_video"
        big = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 8000
        r = c.post(f"/api/sessions/{sid}/video", content=big, headers={"content-type": "video/mp4"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        r = c.post(f"/api/sessions/{sid}/video", content=b"", headers={"content-type": "video/mp4"})
        assert r.status_code == 422
        assert c.get(f"/api/sessions/{sid}").json()["frames"] == []
        assert c.get(f"/api/sessions/{sid}/jobs/jdoesnotexist").status_code == 404


def test_undecodable_video_fails_the_job_with_a_reason(client):
    sid = new_session(client)
    import os

    r = client.post(f"/api/sessions/{sid}/video", content=b"\x00\x00\x00\x1cftypisom" + os.urandom(3000), headers={"content-type": "video/mp4"})
    assert r.status_code == 200
    job = wait_job(client, sid, r.json()["job"])
    assert job["state"] == "failed" and job["error"]["code"] == "video_not_decodable" and job["error"]["suggestions"]


def test_job_runner_progress_cancel_failure_and_lost_worker(service):
    sid = service.create().id
    runner = JobRunner(service._load, service._mutate)
    gate = {"go": False}

    def slow(job):
        while not gate["go"]:
            if job.cancelled():
                from lumen.errors import OperationCancelled

                raise OperationCancelled("cancelled", "stop")
            job.progress(0.3, "Working", force=True)
            time.sleep(0.05)
        return {"answer": 42}

    j = runner.submit(sid, "video", slow)
    assert j["state"] == "queued"
    # a second job of the same kind is refused while the first runs
    from lumen.errors import ConflictError

    with pytest.raises(ConflictError):
        runner.submit(sid, "video", slow)
    time.sleep(0.2)
    assert runner.get(sid, j["id"])["state"] == "running" and runner.get(sid, j["id"])["stage"] == "Working"
    gate["go"] = True
    for _ in range(100):
        if runner.get(sid, j["id"])["state"] == "done":
            break
        time.sleep(0.05)
    done = runner.get(sid, j["id"])
    assert done["state"] == "done" and done["result"] == {"answer": 42} and done["progress"] == 1.0

    gate["go"] = False
    j2 = runner.submit(sid, "video", slow)
    time.sleep(0.15)
    runner.cancel(sid, j2["id"])
    for _ in range(100):
        if runner.get(sid, j2["id"])["state"] == "cancelled":
            break
        time.sleep(0.05)
    assert runner.get(sid, j2["id"])["state"] == "cancelled"

    def boom(job):
        raise RuntimeError("secret internal detail")

    j3 = runner.submit(sid, "video", boom)
    for _ in range(100):
        if runner.get(sid, j3["id"])["state"] == "failed":
            break
        time.sleep(0.05)
    err = runner.get(sid, j3["id"])["error"]
    assert err["code"] == "internal_error" and "secret" not in json.dumps(err)

    # a job whose process died stops heart-beating and is reported as lost, not left "running"
    def freeze(s):
        s.jobs["jlost"] = {"id": "jlost", "kind": "video", "state": "running", "stage": "x", "progress": 0.4, "created": "", "updated": "", "heartbeat": time.time() - 600, "result": None, "error": None}

    service._mutate(sid, freeze)
    lost = runner.get(sid, "jlost")
    assert lost["state"] == "failed" and lost["error"]["code"] == "worker_lost"
    runner.shutdown()


def add(client, sid, img, **m):
    r = client.post(f"/api/sessions/{sid}/frames", content=jpeg_bytes(img), headers={"content-type": "image/jpeg", **(meta(m) if m else {})})
    assert r.status_code == 200, r.text
    return r.json()["frame"]


def test_capture_advice_follows_the_capture(client):
    scene = flat_panel_scene()
    sid = new_session(client)
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    assert not adv["ready"] and adv["next"]["code"] == "first_photo"

    stations = sweep_stations(5, distance=760, height=430, span_deg=50)
    cams = [phone_camera(e, width=1600, height=1200, roll_deg=r) for e, r in stations]
    # three photos from (almost) the same spot on the left end of the arc
    e0 = stations[0][0]
    near = [phone_camera((e0[0] + dx, e0[1], e0[2]), width=1600, height=1200) for dx in (0.0, 12.0, 24.0)]
    add(client, sid, render_cached(scene, near[0], seed=1).image)
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    assert adv["ready"] and not adv["recommended"] and adv["next"]["code"] == "more_views" and adv["next"]["arrow"] in ("left", "right")
    for k in (1, 2):
        add(client, sid, render_cached(scene, near[k], seed=1 + k).image)
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    assert adv["n_usable"] == 3 and adv["spread_deg"] < 5.0
    nb = next(i for i in adv["items"] if i["code"] == "narrow_baseline")
    # the photos were taken from the left of the panel: the room to move is to the right
    assert nb["arrow"] == "right" and not adv["recommended"]
    # two photos from clearly different places resolve it
    add(client, sid, render_cached(scene, cams[2], seed=2).image)
    add(client, sid, render_cached(scene, cams[4], seed=4).image)
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    assert adv["spread_deg"] > 30.0 and adv["recommended"] and adv["next"]["code"] == "analyse"
    assert adv["tags_seen"] == [3, 7, 11, 20] == adv["common_tags"]


def test_advice_names_the_missing_floor_after_analysis(client):
    scene = flat_panel_scene()
    sid = new_session(client)
    tgt = (270.0, 0.0, 520.0)
    cams = [phone_camera(e, target=tgt, width=1600, height=1200, roll_deg=r, hfov_deg=50.0) for e, r in sweep_stations(4, distance=700, height=560, span_deg=40, target=tgt)]
    for i, c in enumerate(cams):
        add(client, sid, render_cached(scene, c, seed=i).image)
    client.post(f"/api/sessions/{sid}/register")
    client.post(f"/api/sessions/{sid}/datums/propose")
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    item = next(i for i in adv["items"] if i["code"] == "floor_missing")
    assert item["arrow"] == "down" and adv["spread_source"] == "registered"


def test_advice_reports_tag_problems_with_a_direction(client):
    scene = flat_panel_scene()
    sid = new_session(client)
    far = phone_camera((270.0, -2600.0, 420.0), width=1600, height=1200)  # tags only ~25 px across
    f = add(client, sid, render_cached(scene, far, seed=0).image)
    assert all(not m["accepted"] for m in f["markers"])
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    small = next(i for i in adv["items"] if i["code"] == "tag_small")
    assert small["arrow"] == "closer" and not adv["ready"]


def test_advice_mentions_small_tags_and_a_missing_lens_profile(client):
    scene = flat_panel_scene()
    # usable but small: about 45 px per tag, too few pixels per cell for the tags' own edge-shift estimate
    sid = new_session(client)
    mid = phone_camera((270.0, -1500.0, 420.0), width=1600, height=1200)
    f = add(client, sid, render_cached(scene, mid, seed=0).image)
    acc = [m for m in f["markers"] if m["accepted"]]
    assert len(acc) == 4 and not any(m["bias_applied"] for m in acc) and all(m["scale_mode"] is not None and len(m["scale_mode"]) == 8 for m in acc)
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    low = next(i for i in adv["items"] if i["code"] == "tags_low_resolution")
    assert low["level"] == "warn" and low["arrow"] == "closer" and "Full-res photo" in low["text"]
    # the session has no lens profile: said once, as information (it is not something a photo can fix)
    prof = next(i for i in adv["items"] if i["code"] == "no_lens_profile")
    assert prof["level"] == "info" and adv["next"]["code"] != "no_lens_profile"
    # at working distance neither hint about the tags appears
    sid2 = new_session(client)
    near = phone_camera((330.0, -720.0, 430.0), width=1600, height=1200)
    add(client, sid2, render_cached(scene, near, seed=1).image)
    codes = {i["code"] for i in client.get(f"/api/sessions/{sid2}/capture/advice").json()["items"]}
    assert "tags_low_resolution" not in codes and "tag_small" not in codes and "no_lens_profile" in codes
    client.put(f"/api/sessions/{sid2}/calibration", json={"profile": calibration_for(near)})
    assert "no_lens_profile" not in {i["code"] for i in client.get(f"/api/sessions/{sid2}/capture/advice").json()["items"]}


def test_flash_pairs_are_checked_for_motion_and_exposure(client):
    scene = flat_panel_scene()
    lit = replace(scene, light=replace(scene.light, flash=True, flash_strength=0.5))
    sid = new_session(client)
    cam = phone_camera((330.0, -720.0, 430.0), width=1600, height=1200)
    moved = phone_camera((330.0, -720.0, 430.0), target=(300.0, 0.0, 300.0), width=1600, height=1200)  # hand turned: aim shifted 30 mm
    a = add(client, sid, render_cached(scene, cam, seed=1).image, torch="off", pair_id="p1")
    b = add(client, sid, render_cached(lit, cam, seed=2).image, torch="on", pair_id="p1")
    add(client, sid, render_cached(scene, cam, seed=3).image, torch="off", pair_id="p2")
    add(client, sid, render_cached(lit, moved, seed=4).image, torch="on", pair_id="p2")
    add(client, sid, render_cached(scene, cam, seed=5).image, torch="off", pair_id="p3")
    assert a["torch"] == "off" and b["torch"] == "on" and a["pair_id"] == "p1"
    adv = client.get(f"/api/sessions/{sid}/capture/advice").json()
    pairs = {p["pair_id"]: p for p in adv["pairs"]}
    p1, p2, p3 = pairs["p1"], pairs["p2"], pairs["p3"]
    assert p1["status"] == "aligned" and p1["usable"] and p1["max_shift_px"] < 0.3
    assert p1["brightness_on_over_off"] > 1.05 and p1["exposure"]["source"] == "unknown"
    assert p2["status"] == "moved" and not p2["usable"] and p2["max_shift_px"] > 20.0 and 20.0 < p2["shift_on_panel_mm"] < 40.0
    assert "moved" in p2["reasons"][0] and "retake" in p2["reasons"][0]
    assert p3["status"] == "incomplete"
    assert any(i["code"] == "flash_pair" for i in adv["items"])
    # the operator can correct the recorded light state of an uploaded photo
    fid = a["id"]
    s = client.patch(f"/api/sessions/{sid}/frames/{fid}", json={"torch": "unknown"}).json()
    assert next(f for f in s["frames"] if f["id"] == fid)["torch"] == "unknown"
    assert client.patch(f"/api/sessions/{sid}/frames/{fid}", json={"torch": "sideways"}).status_code == 422
