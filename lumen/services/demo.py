"""Demo mode: bundled synthetic photos run through the *real* pipeline.

Nothing is mocked: the demo session uploads computer-generated images of an
invented panel (``lumen/demo_assets``, produced by
``scripts/make_demo_assets.py``) through the same ingestion, detection,
registration and gating code as real photos.  The only differences are that the
session is marked ``mode = "demo"`` (everything is labelled *simulated* in the
UI and in exports) and that tag sizes and the lens profile come from the
manifest instead of from a caliper and a calibration target.

Demo content can only be loaded into a session that was created in demo mode,
so simulated frames can never be mixed into a real session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import ConflictError, NotFoundError
from ..schemas import Calibration, Provenance, Session, SessionMode
from .sessions import SessionService

ASSET_DIR = Path(__file__).resolve().parent.parent / "demo_assets"


def scenarios() -> list[dict[str, Any]]:
    out = []
    if not ASSET_DIR.is_dir():
        return out
    for d in sorted(p for p in ASSET_DIR.iterdir() if p.is_dir()):
        m = d / "manifest.json"
        if m.exists():
            data = json.loads(m.read_text(encoding="utf-8"))
            out.append({"id": d.name, "title": data.get("title", d.name), "description": data.get("description", ""), "frames": len(data.get("frames", []))})
    return out


def load_demo(svc: SessionService, sid: str, scenario: str) -> Session:
    s = svc.get(sid)
    if s.mode != SessionMode.DEMO.value:
        raise ConflictError("not_a_demo_session", "Demo content can only be loaded into a demo session.")
    if not scenario.replace("_", "").replace("-", "").isalnum():
        raise NotFoundError("unknown_scenario", "Unknown demo scenario.")
    d = ASSET_DIR / scenario
    mpath = d / "manifest.json"
    if not mpath.exists():
        raise NotFoundError("unknown_scenario", f"Demo scenario {scenario!r} is not bundled.")
    man = json.loads(mpath.read_text(encoding="utf-8"))
    svc.set_operation(sid, man["operation_id"], man.get("revision"), man.get("placard_id"))
    svc.set_tag_sizes(sid, man["tag_sizes"])
    if man.get("calibration"):
        cal = Calibration(**man["calibration"])
        cal.provenance = Provenance.SIMULATED
        svc.set_calibration(sid, cal)
    # the simulated photos report the same camera and lens as the simulated profile
    device = str((man.get("calibration") or {}).get("device") or "synthetic renderer")
    lens = str((man.get("calibration") or {}).get("lens") or "")
    for fr in man["frames"]:
        if fr.get("role", "registration") != "registration":
            continue  # verification photos are added on the Verify step
        data = (d / fr["file"]).read_bytes()
        svc.add_frame(sid, data, role="registration", source="demo", client={"camera_label": device, "lens": lens, "captured_at": man.get("generated", "")})

    def remember(x: Session) -> None:
        x.research["demo_scenario"] = scenario

    svc._mutate(sid, remember)
    return svc.get(sid)


def add_demo_verification(svc: SessionService, sid: str) -> Any:
    """Add the scenario's bundled photo of the installed placard (demo sessions only)."""
    s = svc.get(sid)
    if s.mode != SessionMode.DEMO.value:
        raise ConflictError("not_a_demo_session", "Demo content can only be loaded into a demo session.")
    scenario = str(s.research.get("demo_scenario") or "")
    if not scenario:
        raise ConflictError("no_demo_loaded", "No demo scenario is loaded in this session.")
    data = demo_frame_bytes(scenario, "verification")
    cal = s.calibration
    return svc.add_frame(sid, data, role="verification", source="demo", client={"camera_label": cal.device if cal else "synthetic renderer", "lens": cal.lens if cal else ""})


def demo_frame_bytes(scenario: str, role: str) -> bytes:
    d = ASSET_DIR / scenario
    mpath = d / "manifest.json"
    if not mpath.exists():
        raise NotFoundError("unknown_scenario", "Unknown demo scenario.")
    man = json.loads(mpath.read_text(encoding="utf-8"))
    for fr in man["frames"]:
        if fr.get("role") == role:
            return (d / fr["file"]).read_bytes()
    raise NotFoundError("demo_frame_missing", f"The demo scenario has no {role} photo.")
