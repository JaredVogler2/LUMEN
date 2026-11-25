#!/usr/bin/env python3
"""Smoke test of a running LUMEN instance over HTTP (standard library only).

    python scripts/smoke_test.py http://127.0.0.1:8000/
    python scripts/smoke_test.py https://connect.example.org/content/<guid>/ --api-key "$CONNECT_API_KEY"

It runs the bundled *simulated* demo session through the real pipeline of the
target server: health check, demo photos, registration, automatic datums,
overlay with its uncertainty, live-frame tracking, verification, export, and
deletion of the session.  Nothing but the demo data is sent.  Exit status 0
means every step answered as expected.

The camera, the torch and a real phone are NOT covered: see
docs/POSIT_CONNECT.md for the manual checklist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base: str, api_key: str | None, timeout: float) -> None:
        self.base = base if base.endswith("/") else base + "/"
        self.api_key = api_key
        self.timeout = timeout

    def call(self, method: str, path: str, body: bytes | dict | None = None, content_type: str | None = None, raw: bool = False):
        url = self.base + path.lstrip("/")
        data = None
        headers = {"Accept": "application/json"}
        if isinstance(body, dict):
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif body is not None:
            data = body
            headers["Content-Type"] = content_type or "application/octet-stream"
        if self.api_key:
            headers["Authorization"] = f"Key {self.api_key}"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - operator-supplied URL
                payload = resp.read()
                return (resp.status, payload, dict(resp.headers)) if raw else (resp.status, json.loads(payload or b"null"), dict(resp.headers))
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                return exc.code, json.loads(payload), dict(exc.headers)
            except ValueError:
                return exc.code, {"error": {"code": f"http_{exc.code}", "message": payload[:200].decode("utf-8", "replace")}}, dict(exc.headers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base_url", help="address of the application, e.g. https://connect.example.org/content/<guid>/")
    ap.add_argument("--api-key", default=None, help="Posit Connect API key (sent as 'Authorization: Key ...'); omit for a local server")
    ap.add_argument("--scenario", default="flat", choices=["flat", "curve"])
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--keep", action="store_true", help="do not delete the demo session at the end")
    args = ap.parse_args(argv)
    c = Client(args.base_url, args.api_key, args.timeout)
    results: list[tuple[str, bool, str]] = []
    sid = None

    def step(name: str, fn):
        t = time.perf_counter()
        try:
            ok, note = fn()
        except Exception as exc:  # report and continue: later steps show what else is broken
            ok, note = False, f"{type(exc).__name__}: {exc}"
        results.append((name, bool(ok), f"{note}  [{time.perf_counter() - t:.1f} s]"))
        print(("PASS  " if ok else "FAIL  ") + name + " — " + results[-1][2], flush=True)
        return ok

    state: dict = {}

    def health():
        st, b, _ = c.call("GET", "api/health")
        state["health"] = b
        return st == 200 and b.get("status") == "ok" and b["checks"].get("apriltag_36h11"), f"status {st} {b.get('status')}, version {b.get('version')}, OpenCV {b.get('dependencies', {}).get('opencv')}, storage {b.get('storage', {}).get('adapter')}"

    def index():
        st, payload, headers = c.call("GET", "", raw=True)
        html = payload.decode("utf-8", "replace")
        csp = headers.get("Content-Security-Policy") or headers.get("content-security-policy") or ""
        return st == 200 and "LUMEN" in html and "script-src 'self'" in csp, f"status {st}, base tag {'present' if '<base ' in html else 'added in the browser'}, CSP {'present' if csp else 'MISSING'}"

    def assets():
        bad = []
        for a in ("static/js/main.js", "static/css/lumen.css", "static/img/icon.svg"):
            st, _, _ = c.call("GET", a, raw=True)
            if st != 200:
                bad.append(f"{a}: {st}")
        return not bad, "all served" if not bad else "; ".join(bad)

    def demo():
        nonlocal sid
        st, s, _ = c.call("POST", "api/sessions", {"mode": "demo"})
        if st != 200:
            return False, f"cannot create a demo session: {st} {s.get('error', {}).get('code')}"
        sid = s["id"]
        st, s, _ = c.call("POST", f"api/sessions/{sid}/demo/load", {"scenario": args.scenario})
        n = len(s.get("frames", [])) if st == 200 else 0
        tags = sorted({m["tag_id"] for f in s.get("frames", []) for m in f["markers"] if m["accepted"]}) if st == 200 else []
        return st == 200 and n >= 3 and len(tags) >= 2, f"{n} simulated photos, accepted tags {tags}"

    def register():
        st, r, _ = c.call("POST", f"api/sessions/{sid}/register")
        reg = r.get("registration", {})
        state["reg"] = reg
        return st == 200 and reg.get("consistent"), f"{reg.get('n_tags')} tags, {len(reg.get('frames_used', []))} photos, rms {reg.get('rms_px', float('nan')):.3f} px"

    def datums():
        st, r, _ = c.call("POST", f"api/sessions/{sid}/datums/propose")
        applied = (r.get("proposals") or {}).get("applied", {})
        st2, _, _ = c.call("POST", f"api/sessions/{sid}/datums/confirm")
        return st == 200 and st2 == 200 and set(applied) == {"floor", "side"}, f"automatic datums {sorted(applied)}"

    def overlay():
        fid = state["reg"]["reference_frame_id"]
        st, ov, _ = c.call("GET", f"api/sessions/{sid}/overlay?frame={fid}")
        rep = next(iter((ov.get("reports") or {}).values()), {})
        tgt = [p for p in ov.get("primitives", []) if p["kind"].startswith("target")]
        state["fid"] = fid
        u = (rep.get("uncertainty") or {}).get("expanded_mm")
        return st == 200 and bool(tgt) and rep.get("status") in ("prototype_guidance", "research_preview"), f"status {rep.get('status')}, expanded uncertainty {u if u is None else round(u, 2)} mm, simulated={ov.get('simulated')}"

    def image_and_track():
        st, img, hdr = c.call("GET", f"api/sessions/{sid}/frames/{state['fid']}/image?max=1280", raw=True)
        if st != 200 or not img.startswith(b"\xff\xd8"):
            return False, f"frame image: status {st}"
        st, ov, _ = c.call("POST", f"api/sessions/{sid}/track", img, "image/jpeg")
        t = ov.get("tracking", {})
        return st == 200 and t.get("state") == "tracking", f"image {len(img) // 1024} kB; tracking {t.get('state')} with tags {t.get('used_tags')}, server time {ov.get('timing_ms', {}).get('total')} ms"

    def verify():
        st, r, _ = c.call("POST", f"api/sessions/{sid}/demo/verification")
        if st == 404:
            return True, "this scenario has no simulated verification photo (skipped)"
        if st != 200:
            return False, f"status {st} {r.get('error', {}).get('code')}"
        st, out, _ = c.call("POST", f"api/sessions/{sid}/verify", {"frame_id": r["frame"]["id"]})
        v = out.get("verification", {})
        return st == 200 and v.get("dx_mm") is not None, f"measured dx {v.get('dx_mm'):+.2f} mm, dz {v.get('dz_mm'):+.2f} mm ({v.get('observation')}); the demo placard is installed 1.4 mm right and 0.9 mm low"

    def export():
        st, ex, _ = c.call("GET", f"api/sessions/{sid}/export")
        return st == 200 and ex.get("images_included") is False and ex.get("simulated") is True, f"{len(json.dumps(ex)) // 1024} kB summary, images_included={ex.get('images_included')}"

    def cleanup():
        if args.keep:
            return True, f"session {sid} kept (--keep)"
        st, r, _ = c.call("DELETE", f"api/sessions/{sid}")
        st2, _, _ = c.call("GET", f"api/sessions/{sid}")
        return st == 200 and r.get("deleted") and st2 == 404, "session and photos deleted"

    print(f"LUMEN smoke test against {c.base}  (simulated demo data only)\n")
    step("health", health)
    step("index page and content security policy", index)
    step("bundled assets", assets)
    if step("demo session with simulated photos", demo):
        if step("registration", register):
            step("datums", datums)
            if step("overlay and uncertainty", overlay):
                step("frame image and live tracking", image_and_track)
            step("verification", verify)
            step("export", export)
        step("delete session", cleanup)
    failed = [n for n, ok, _ in results if not ok]
    print("\n" + ("ALL STEPS PASSED" if not failed else f"FAILED: {', '.join(failed)}"))
    print("Not covered here: camera, torch, video upload from a phone, real photos. See docs/POSIT_CONNECT.md.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
