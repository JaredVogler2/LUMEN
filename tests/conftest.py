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
        # the exposure an auto-exposing camera chose is not stored with the picture: unknown on a cache hit
        res = RenderResult(image=img, truth=truth, exposure=float("nan") if cam.auto_exposure else cam.exposure)
    else:
        res = render(scene, cam, ss=ss, seed=seed)
        # write-then-rename, so a concurrent test process never reads a partial file
        tmp = CACHE / f"{key}.{os.getpid()}.tmp.png"
        cv2.imwrite(str(tmp), res.image)
        os.replace(tmp, png)
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


def sweep_video_frames(n_views: int = 8, hold: int = 3, width: int = 1280, height: int = 720, span_deg: float = 44.0, scene: Scene | None = None, blur_px: int = 15) -> tuple[list[np.ndarray], list[CameraSpec], list[int]]:
    """Frames of a synthetic hand sweep for video tests.

    ``n_views`` rendered viewpoints are each held for ``hold`` frames; only the
    middle frame of a hold is sharp, its neighbours carry horizontal motion
    blur (the hand moving between stations).  Returns (frames, camera per
    frame, indices of the sharp frames).
    """
    from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations

    scene = scene or flat_panel_scene()
    target = (270.0, 0.0, 300.0)
    frames: list[np.ndarray] = []
    cams: list[CameraSpec] = []
    sharp: list[int] = []
    kernel = np.zeros((1, blur_px), np.float32)
    kernel[:] = 1.0 / blur_px
    for v, (eye, roll) in enumerate(sweep_stations(n_views, distance=720, height=430, span_deg=span_deg, target=target)):
        cam = phone_camera(eye, target=target, width=width, height=height, roll_deg=roll)
        img = render_cached(scene, cam, ss=2, seed=v).image
        for h in range(hold):
            if h == hold // 2:
                sharp.append(len(frames))
                frames.append(img.copy())
            else:
                frames.append(cv2.filter2D(img, -1, kernel))
            cams.append(cam)
    return frames, cams, sharp


def write_video(path: str, frames: list[np.ndarray], fps: float = 12.0, fourcc: str = "mp4v") -> str:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    assert vw.isOpened(), f"cannot write {fourcc} video in this environment"
    for f in frames:
        vw.write(f)
    vw.release()
    return path


class PrefixProxy:
    """A path-stripping reverse proxy in front of the app, as Posit Connect is.

    Requests must arrive under ``prefix``; the prefix is removed before the
    app sees the path (the app never learns it from the URL).  With
    ``announce`` the proxy adds Connect's ``RStudio-Connect-App-Base-URL``
    header; without it the app has to cope on its own.
    """

    def __init__(self, app: Any, prefix: str = "/content/3f2a9c1e", announce: bool = True, user: str | None = None) -> None:
        self.app, self.prefix, self.announce, self.user = app, prefix.rstrip("/"), announce, user
        self.seen: list[str] = []

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if path != self.prefix and not path.startswith(self.prefix + "/"):
            await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"not this content item"})
            return
        inner = path[len(self.prefix):] or "/"
        self.seen.append(inner)
        new = dict(scope)
        new["path"] = inner
        new["raw_path"] = inner.encode()
        new["root_path"] = ""
        headers = [(k, v) for k, v in scope["headers"] if k not in (b"rstudio-connect-app-base-url", b"rstudio-connect-credentials")]
        if self.announce:
            host = dict(scope["headers"]).get(b"host", b"connect.example.invalid").decode()
            headers.append((b"rstudio-connect-app-base-url", f"http://{host}{self.prefix}/".encode()))
        if self.user:
            headers.append((b"rstudio-connect-credentials", json.dumps({"user": self.user, "groups": []}).encode()))
        new["headers"] = headers
        await self.app(new, receive, send)
