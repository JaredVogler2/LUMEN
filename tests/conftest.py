"""Shared fixtures.

Synthetic photos are rendered by ``lumen.synth`` (an independent ray caster)
and cached on disk by content hash, so the first run renders (a few seconds
per image) and later runs are fast.  Nothing in the tests uses real photos.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

os.environ.setdefault("LUMEN_STORAGE", "memory")

from lumen.config import Limits, Settings  # noqa: E402
from lumen.operations.library import OperationLibrary  # noqa: E402
from lumen.services.sessions import SessionService  # noqa: E402
from lumen.storage.base import MemorySessionStore  # noqa: E402
from lumen.synth import CameraSpec, Scene, project_points, render, scene_truth  # noqa: E402
from lumen.synth.render import RenderResult  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".pytest_cache" / "lumen_renders"
BUNDLED_OPS = ROOT / "lumen" / "operations" / "data" / "operations.json"


def render_cached(scene: Scene, cam: CameraSpec, *, ss: int = 3, seed: int = 0, jpeg: int | None = None) -> RenderResult:
    """Deterministic render with an on-disk cache keyed by the full inputs."""
    key = hashlib.sha256(repr((scene, cam, ss, seed, jpeg, "v3")).encode()).hexdigest()[:24]
    CACHE.mkdir(parents=True, exist_ok=True)
    png = CACHE / f"{key}.png"
    if png.exists():
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        truth = scene_truth(scene, cam)
        res = RenderResult(image=img, truth=truth, exposure=cam.exposure)
    else:
        res = render(scene, cam, ss=ss, seed=seed)
        cv2.imwrite(str(png), res.image)
    if jpeg is not None:
        ok, buf = cv2.imencode(".jpg", res.image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg])
        res.jpeg = buf.tobytes()
        res.image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return res


def jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def px(cam: CameraSpec, x: float, z: float, y: float = 0.0) -> np.ndarray:
    """Ground-truth pixel of a world point (default: on the panel face)."""
    uv, _ = project_points(cam, np.array([[x, y, z]]))
    return uv[0]


def truth_tag(res: RenderResult, tag_id: int) -> dict[str, Any]:
    return next(t for t in res.truth["tags"] if t["id"] == tag_id)


def calibration_for(cam: CameraSpec, cid: str = "truth") -> dict[str, Any]:
    return {
        "id": cid, "label": "synthetic truth", "method": "synthetic_truth", "provenance": "simulated",
        "width": cam.width, "height": cam.height, "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
        "dist": list(cam.dist), "param_sigma": [0.5, 0.5, 0.5, 0.5, 0.0005, 0.002, 0.0001, 0.0001, 0.0],
    }


def make_settings(**kw: Any) -> Settings:
    lim = kw.pop("limits", Limits())
    return Settings(limits=lim, storage="memory", **kw)


@pytest.fixture()
def library() -> OperationLibrary:
    return OperationLibrary(BUNDLED_OPS)


@pytest.fixture()
def service(library: OperationLibrary) -> SessionService:
    return SessionService(MemorySessionStore(), make_settings(), library)


@pytest.fixture()
def client(service: SessionService):
    from fastapi.testclient import TestClient

    from lumen.api.app import create_app

    app = create_app(service.settings, service)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def demo_manifest() -> dict[str, Any]:
    return json.loads((ROOT / "lumen" / "demo_assets" / "flat" / "manifest.json").read_text())


def camera_from_truth(c: dict[str, Any]) -> CameraSpec:
    return CameraSpec(
        width=c["width"], height=c["height"], fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"], dist=tuple(c["dist"]),
        R_wc=tuple(map(tuple, c["R_wc"])), C=tuple(c["C"]),
    )
