"""Deployment behaviour that can be checked without a Posit Connect server (prompt 14).

Simulated only: a path-stripping proxy in front of the app, proxy headers,
several worker processes sharing one session directory, request limits and
the bundled assets.  An actual deployment has NOT been run (docs/POSIT_CONNECT.md).
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lumen.api.app import create_app
from lumen.config import Limits
from lumen.operations.library import OperationLibrary
from lumen.services.sessions import SessionService
from lumen.storage.base import FileSessionStore, MemorySessionStore
from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
from tests.conftest import BUNDLED_OPS, PrefixProxy, jpeg_bytes, make_settings, render_cached

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "lumen" / "static"
PREFIX = "/content/3f2a9c1e"


def service(**kw):
    return SessionService(MemorySessionStore(), make_settings(**kw), OperationLibrary(BUNDLED_OPS))


def base_href(html: str):
    m = re.search(r'<base href="([^"]*)">', html)
    return m.group(1) if m else None


def test_index_takes_its_base_from_the_proxy_and_never_from_guesswork(client):
    r = client.get("/", headers={"RStudio-Connect-App-Base-URL": "https://connect.example.invalid/content/3f2a9c1e"})
    assert base_href(r.text) == "https://connect.example.invalid/content/3f2a9c1e/"
    assert base_href(client.get("/", headers={"X-Forwarded-Prefix": "/apps/lumen"}).text) == "/apps/lumen/"
    # a prefix header that is not a plain path is ignored
    for bad in ("//evil.example/x", "/a/../b", "javascript:alert(1)", "/" + "a" * 300):
        assert base_href(client.get("/", headers={"X-Forwarded-Prefix": bad}).text) is None, bad
    # nothing known: no <base> is invented; a nonce-protected script adds the missing slash in the browser
    r = client.get("/")
    assert base_href(r.text) is None
    nonce = re.search(r'<script nonce="([^"]+)">', r.text).group(1)
    csp = r.headers["content-security-policy"]
    assert f"'nonce-{nonce}'" in csp and "unsafe-eval" not in csp and "http:" not in csp and "https:" not in csp
    assert "default-src 'self'" in csp and "object-src 'none'" in csp and "frame-ancestors 'self'" in csp
    assert client.get("/").text.count(nonce) == 0, "a fresh nonce per response"
    assert r.headers["x-content-type-options"] == "nosniff" and "camera=(self)" in r.headers["permissions-policy"]
    assert "no-store" in r.headers["cache-control"]


def test_configured_base_path_and_asgi_root_path(library):
    svc = service(base_path="/research/lumen")
    with TestClient(create_app(svc.settings, svc)) as c:
        assert base_href(c.get("/").text) == "/research/lumen/"
    svc2 = service()
    with TestClient(create_app(svc2.settings, svc2), root_path="/mounted") as c:
        assert base_href(c.get("/").text) == "/mounted/"


@pytest.mark.parametrize("announce", [True, False])
def test_everything_works_behind_a_path_stripping_proxy(announce):
    svc = service()
    proxy = PrefixProxy(create_app(svc.settings, svc), PREFIX, announce=announce)
    with TestClient(proxy) as c:
        assert c.get("/").status_code == 404 and c.get("/api/health").status_code == 404  # only the content path exists
        r = c.get(PREFIX + "/")
        assert r.status_code == 200
        assert (base_href(r.text) is not None) == announce
        if announce:
            assert base_href(r.text).endswith(PREFIX + "/")
        # every asset the page names is relative, and resolves under the prefix
        # (announced base: plain tags; unknown base: the bootstrap script adds them after fixing the base)
        refs = re.findall(r'(?:href|src)="([^"#]+)"', r.text) + re.findall(r"(?:href|src):'([^']+)'", r.text)
        refs = [x for x in refs if not x.startswith(("http://", "https://"))]
        assert len(refs) >= 3 and all(not x.startswith("/") for x in refs), refs
        if not announce:
            assert "<link" not in r.text and 'src="' not in r.text, "no asset may be requested before the base is known"
        for ref in refs:
            assert c.get(f"{PREFIX}/{ref}").status_code == 200, ref
        assert c.get(PREFIX + "/api/health").json()["status"] == "ok"
        sid = c.post(PREFIX + "/api/sessions", json={"mode": "demo"}).json()["id"]
        s = c.post(f"{PREFIX}/api/sessions/{sid}/demo/load", json={"scenario": "flat"}).json()
        fid = s["frames"][0]["id"]
        img = c.get(f"{PREFIX}/api/sessions/{sid}/frames/{fid}/image?max=320")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        assert c.post(f"{PREFIX}/api/sessions/{sid}/register").json()["registration"]["consistent"]
        # the app saw prefix-free paths only
        assert all(not p.startswith("/content") for p in proxy.seen)


EXTERNAL_URL = re.compile(r"""(?:https?:)?//[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+[^\s'"`)]*""")
ROOT_ABSOLUTE = re.compile(r"""['"`(](/(?:api|static)/[^'"`)]*)""")
MODULE_IMPORT = re.compile(r"""import\s[^;]*?from\s+['"]([^'"]+)['"]""")


def test_the_url_patterns_used_below_do_find_what_they_look_for():
    assert EXTERNAL_URL.search('src="https://cdn.example.com/lib.js"') and EXTERNAL_URL.search("url(//fonts.example.org/a.woff)")
    assert ROOT_ABSOLUTE.search("fetch('/api/health')") and ROOT_ABSOLUTE.search('href="/static/css/x.css"')
    assert not ROOT_ABSOLUTE.search("fetch('api/health')") and not EXTERNAL_URL.search("a // a comment about x.y")
    assert MODULE_IMPORT.search("import { h } from './dom.js';").group(1) == "./dom.js"


def test_bundled_assets_contain_no_absolute_or_external_urls():
    """Single origin: no CDN, no absolute paths that would break under a sub-path."""
    bad = []
    n_files = n_imports = 0
    for f in sorted(STATIC.rglob("*")):
        if f.suffix not in (".js", ".css", ".html", ".svg"):
            continue
        n_files += 1
        text = f.read_text(encoding="utf-8")
        for m in EXTERNAL_URL.finditer(text):
            url = m.group(0)
            if "www.w3.org/" in url:  # XML namespace identifiers, never fetched
                continue
            bad.append((f.name, url))
        for m in ROOT_ABSOLUTE.finditer(text):
            bad.append((f.name, m.group(1)))
        if f.suffix == ".js":
            assert "eval(" not in text and "new Function(" not in text, f.name
            for m in MODULE_IMPORT.finditer(text):
                n_imports += 1
                assert m.group(1).startswith("."), (f.name, m.group(1))  # only relative module imports
    assert n_files >= 15 and n_imports >= 30  # the scan really covered the application
    assert not bad, bad


def test_two_worker_processes_can_serve_one_session_from_a_shared_directory(tmp_path):
    """Posit Connect may run several processes of one content item; they share only the session directory."""
    lib = OperationLibrary(BUNDLED_OPS)
    a = SessionService(FileSessionStore(tmp_path / "sessions"), make_settings(), lib)
    b = SessionService(FileSessionStore(tmp_path / "sessions"), make_settings(), lib)
    scene = flat_panel_scene()
    cams = [phone_camera(e, width=1280, height=960, roll_deg=r) for e, r in sweep_stations(3, distance=740, height=430, span_deg=40)]
    sid = a.create().id
    b.set_operation(sid, "DEMO-FLAT-01", None, "PLC-DEMO-101")
    a.set_tag_sizes(sid, {"nominal_mm": 50.0, "measured_mm": {"3": 50.0, "7": 50.0, "11": 50.0, "20": 50.0}})
    for i, cam in enumerate(cams):
        (a if i % 2 else b).add_frame(sid, jpeg_bytes(render_cached(scene, cam, ss=2, seed=i).image))
    reg = a.register(sid)
    assert reg["consistent"] and len(reg["frames_used"]) == 3
    # the other process has no cached solution or images: it rebuilds them from the shared store
    out = b.propose_datums(sid)
    assert set(out["applied"]) == {"floor", "side"}
    ov = a.overlay(sid, reg["reference_frame_id"])
    assert any(p["kind"].startswith("target") for p in ov["primitives"])
    # a job started in one process is visible to the other
    job = a.jobs.submit(sid, "video", lambda j: {"ok": True})
    for _ in range(100):
        st = b.jobs.get(sid, job["id"])
        if st["state"] == "done":
            break
        import time

        time.sleep(0.05)
    assert st["state"] == "done" and st["result"] == {"ok": True}
    assert b.delete(sid) and not (tmp_path / "sessions" / sid).exists()
    a.jobs.shutdown()


def test_sessions_are_bound_to_the_connect_user_who_created_them():
    svc = service()
    app = create_app(svc.settings, svc)
    cred = lambda u: {"RStudio-Connect-Credentials": json.dumps({"user": u, "groups": ["g"]})}  # noqa: E731
    with TestClient(app) as c:
        sid = c.post("/api/sessions", json={}, headers=cred("alice")).json()["id"]
        assert c.get(f"/api/sessions/{sid}", headers=cred("alice")).status_code == 200
        other = c.get(f"/api/sessions/{sid}", headers=cred("bob"))
        assert other.status_code == 404 and other.json()["error"]["code"] == "session_not_found"
        assert c.delete(f"/api/sessions/{sid}", headers=cred("bob")).status_code == 404
        assert c.get("/api/config", headers=cred("alice")).json()["user"] == "alice"
        # a malformed header is ignored, not trusted
        assert c.get("/api/config", headers={"RStudio-Connect-Credentials": "not json"}).json()["user"] == ""


def test_request_limits_and_health():
    svc = service(limits=Limits(max_image_bytes=200_000, max_video_bytes=300_000, max_frames_per_session=2))
    with TestClient(create_app(svc.settings, svc)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}/frames", content=b"x" * 400_000, headers={"content-type": "image/jpeg"})
        assert r.status_code == 413
        h = c.get("/api/health")
        assert h.status_code == 200
        body = h.json()
        assert body["status"] == "ok" and body["checks"]["apriltag_36h11"] and body["checks"]["jpeg_codec"] and body["checks"]["storage"]
        assert body["dependencies"]["opencv"] and "notice" in body
        cfg = c.get("/api/config").json()
        assert cfg["limits"]["max_frames"] == 2 and cfg["features"]["assistant"] is False
        # nothing in the configuration answer is a secret or a filesystem path
        text = json.dumps(cfg)
        assert "api_key" not in text.lower() and "/tmp" not in text and "data_dir" not in text


def test_committed_manifest_matches_the_files_that_are_deployed():
    """Git-backed publishing deploys what manifest.json lists; a manifest that lags behind the code deploys old code."""
    import hashlib

    p = ROOT / "manifest.json"
    if not p.exists():
        pytest.skip("no manifest.json: run scripts/write_manifest.sh before git-backed publishing")
    m = json.loads(p.read_text())
    assert m["metadata"]["appmode"] == "python-fastapi" and m["metadata"]["entrypoint"] == "app:app"
    assert m["python"]["version"].startswith(("3.12", "3.13")), m["python"]["version"]
    listed = set(m["files"])
    expected = {"app.py", "requirements.txt"} | {
        str(f.relative_to(ROOT)).replace("\\", "/") for f in (ROOT / "lumen").rglob("*") if f.is_file() and "__pycache__" not in f.parts
    }
    # an installation may add its organisation's logo (brand/logo.*, see brand/README.md); nothing else is deployed
    listed -= {f for f in listed if re.fullmatch(r"brand/logo\.(svg|png|webp|jpe?g)", f) and (ROOT / f).is_file()}
    assert listed == expected, f"run scripts/write_manifest.sh (missing {sorted(expected - listed)[:5]}, extra {sorted(listed - expected)[:5]})"
    stale = [f for f, meta in m["files"].items() if hashlib.md5((ROOT / f).read_bytes()).hexdigest() != meta["checksum"]]
    assert not stale, f"run scripts/write_manifest.sh: {len(stale)} deployed file(s) changed since the manifest was written, e.g. {stale[:4]}"
    # nothing that must not be deployed
    assert not any(f.startswith(("tests/", "docs/", "prompt_series/", "scripts/", ".venv")) for f in m["files"])


def test_a_host_that_keeps_no_sessions_is_named_instead_of_failing_photo_by_photo(monkeypatch, library):
    """Seen on a test deployment on a function platform: every request ran in a new instance with an empty disk,
    photos would not load and each step answered "Unknown or expired session".  The application cannot work
    there; it has to say so."""
    for var in ("VERCEL", "AWS_LAMBDA_FUNCTION_NAME", "FUNCTIONS_WORKER_RUNTIME", "NETLIFY", "CONNECT_SERVER", "RSTUDIO_PRODUCT", "POSIT_PRODUCT"):
        monkeypatch.delenv(var, raising=False)
    svc = service()
    with TestClient(create_app(svc.settings, svc)) as c:
        cfg = c.get("/api/config").json()
        assert cfg["hosting"] == {"kind": "process", "name": "", "keeps_sessions": True, "message": ""}
        h = c.get("/api/health")
        assert h.status_code == 200 and h.json()["checks"]["host_keeps_sessions"] is True
        e = c.get("/api/sessions/AAAAAAAAAAAAAAAAAAAAAAAA").json()["error"]
        assert e["code"] == "session_not_found" and e["suggestions"] == []
    monkeypatch.setenv("CONNECT_SERVER", "https://connect.example.invalid")
    assert service().settings.hosting["kind"] == "posit_connect" and service().settings.hosting["keeps_sessions"] is True
    monkeypatch.delenv("CONNECT_SERVER")
    monkeypatch.setenv("VERCEL", "1")
    svc = service()
    with TestClient(create_app(svc.settings, svc)) as c:
        host = c.get("/api/config").json()["hosting"]
        assert host["kind"] == "vercel" and host["keeps_sessions"] is False
        assert "lost between requests" in host["message"] and "Posit Connect" in host["message"]
        h = c.get("/api/health")
        assert h.status_code == 503 and h.json()["status"] == "degraded" and h.json()["checks"]["host_keeps_sessions"] is False
        # a session that is gone is explained by what the host is, not as an expired session
        e = c.get("/api/sessions/AAAAAAAAAAAAAAAAAAAAAAAA").json()["error"]
        assert e["code"] == "session_not_found" and "lost between requests" in e["suggestions"][0] and e["details"]["hosting"] == "vercel"
