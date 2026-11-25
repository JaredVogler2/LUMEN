#!/usr/bin/env python3
"""Generate the bundled demo photos (deterministic).

    python scripts/make_demo_assets.py

Writes ``lumen/demo_assets/<scenario>/*.jpg`` and ``manifest.json``.  The
images are computer-generated pictures of invented panels; the manifest
records the exact scene and camera so the files can be regenerated and so
tests can compare results with ground truth.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumen.synth import LightSpec, render  # noqa: E402
from lumen.synth.fixtures import curved_panel_scene, demo_placard, flat_panel_scene, phone_camera, sweep_stations  # noqa: E402

OUT = ROOT / "lumen" / "demo_assets"
W, H = 1600, 1200
DIST = (0.06, -0.12, 0.0004, -0.0003, 0.0)
GENERATED = "2026-11-25T00:00:00Z"


def calibration(cam, scenario: str) -> dict:
    return {
        "id": f"demo-{scenario}-synthetic",
        "label": "Synthetic renderer camera (exact parameters)",
        "method": "synthetic_truth",
        "provenance": "simulated",
        "width": cam.width,
        "height": cam.height,
        "fx": cam.fx,
        "fy": cam.fy,
        "cx": cam.cx,
        "cy": cam.cy,
        "dist": list(cam.dist),
        "rms_px": 0.0,
        "n_views": 0,
        "coverage": 1.0,
        "param_sigma": [0.5, 0.5, 0.5, 0.5, 0.0005, 0.002, 0.0001, 0.0001, 0.0],
        "device": "synthetic",
        "lens": "synthetic 67 deg",
        "capture_mode": "photo",
        "created": GENERATED,
        "notes": "Exact parameters of the synthetic camera; simulated, not a measurement.",
    }


def write(scenario: str, title: str, description: str, operation_id: str, placard_id: str, shots: list, tag_ids: list[int]) -> None:
    d = OUT / scenario
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    cal = None
    for k, (name, role, scene, cam) in enumerate(shots):
        r = render(scene, cam, ss=3, seed=100 + k, return_jpeg=False)
        import cv2

        ok, buf = cv2.imencode(".jpg", r.image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        assert ok
        (d / name).write_bytes(buf.tobytes())
        frames.append({"file": name, "role": role, "truth": {"camera": r.truth["camera"], "tags": r.truth["tags"], "placards": r.truth["placards"]}})
        cal = cal or calibration(cam, scenario)
        print(f"  {scenario}/{name}: {len(buf) / 1024:.0f} kB")
    manifest = {
        "title": title,
        "description": description,
        "generated": GENERATED,
        "notice": "Computer-generated images of an invented panel. Simulated data.",
        "operation_id": operation_id,
        "revision": "A",
        "placard_id": placard_id,
        "tag_sizes": {"nominal_mm": 50.0, "measured_mm": {str(i): 50.0 for i in tag_ids}, "measured_u_mm": 0.05, "nominal_tolerance_mm": 1.0},
        "calibration": cal,
        "frames": frames,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def main() -> None:
    light = LightSpec(ambient=0.78)
    # ---------------------------------------------------------------- flat
    scene = flat_panel_scene(light=light)
    stations = sweep_stations(5, distance=760.0, height=430.0, span_deg=46.0)
    shots = []
    for k, (eye, roll) in enumerate(stations):
        cam = phone_camera(eye, width=W, height=H, roll_deg=roll, dist=DIST, blur_sigma=0.9)
        shots.append((f"reg_{k + 1}.jpg", "registration", scene, cam))
    # verification photo: a placard installed 1.4 mm right, 0.9 mm low, rotated 0.7 deg
    placed = flat_panel_scene(light=light, placards=(demo_placard(240.0 + 1.4, 300.0 - 0.9, rotation_deg=0.7),))
    cam_v = phone_camera((330.0, -690.0, 380.0), width=W, height=H, roll_deg=-2.0, dist=DIST, blur_sigma=0.9)
    shots.append(("verify_1.jpg", "verification", placed, cam_v))
    write(
        "flat",
        "Flat panel with joint seam",
        "Five photos of an invented flat panel that meets the floor, with a vertical joint seam and four 50 mm tags. Matches sample operation DEMO-FLAT-01.",
        "DEMO-FLAT-01",
        "PLC-DEMO-101",
        shots,
        [3, 7, 11, 20],
    )
    # --------------------------------------------------------------- curve
    scene_c = curved_panel_scene(light=light)
    shots = []
    for k, (eye, roll) in enumerate(sweep_stations(7, distance=740.0, height=420.0, span_deg=62.0, target=(230.0, 0.0, 300.0))):
        cam = phone_camera(eye, target=(230.0, 0.0, 300.0), width=W, height=H, roll_deg=roll, dist=DIST, blur_sigma=0.9)
        shots.append((f"reg_{k + 1}.jpg", "registration", scene_c, cam))
    write(
        "curve",
        "Panel with rounded side edge",
        "Seven photos swept around an invented panel whose left side bends away with a 35 mm radius. Matches sample operation DEMO-CURVE-01.",
        "DEMO-CURVE-01",
        "PLC-DEMO-201",
        shots,
        [3, 7, 11, 20],
    )


if __name__ == "__main__":
    main()
