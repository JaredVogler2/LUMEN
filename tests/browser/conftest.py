"""Browser test fixtures: a real server on a local port and headless Chromium.

The tests drive the same single-page app an operator uses.  They are skipped
(not failed) when no Chromium binary is available.  Camera tests use
Chromium's fake capture device fed with a synthetic video file; that proves
the browser code path, not iPhone Safari behaviour (see docs/PHYSICAL_TEST_PROTOCOL.md).
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

CHROMIUM = os.environ.get("LUMEN_CHROMIUM", "/opt/pw-browsers/chromium")
ARTIFACTS = Path(__file__).parent / "_artifacts"

pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LiveServer:
    def __init__(self, app: Any) -> None:
        import uvicorn

        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> "LiveServer":
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


def make_app(**settings_kw: Any) -> Any:
    from lumen.api.app import create_app
    from lumen.operations.library import OperationLibrary
    from lumen.services.sessions import SessionService
    from lumen.storage.base import MemorySessionStore
    from tests.conftest import BUNDLED_OPS, make_settings

    settings = make_settings(**settings_kw)
    return create_app(settings, SessionService(MemorySessionStore(), settings, OperationLibrary(BUNDLED_OPS)))


@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    srv = LiveServer(make_app()).start()
    yield srv
    srv.stop()


@pytest.fixture(scope="session")
def playwright_instance() -> Iterator[Any]:
    pw = pytest.importorskip("playwright.sync_api")
    if not Path(CHROMIUM).exists():
        pytest.skip(f"no Chromium binary at {CHROMIUM}")
    with pw.sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright_instance: Any) -> Iterator[Any]:
    b = playwright_instance.chromium.launch(executable_path=CHROMIUM)
    yield b
    b.close()


PHONE = {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 1, "has_touch": True, "is_mobile": True}
PHONE_LANDSCAPE = {"viewport": {"width": 844, "height": 390}, "device_scale_factor": 1, "has_touch": True, "is_mobile": True}
DESKTOP = {"viewport": {"width": 1280, "height": 900}, "device_scale_factor": 1}


@pytest.fixture()
def page(browser: Any, live_server: LiveServer) -> Iterator[Any]:
    ctx = browser.new_context(**PHONE)
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "favicon" not in m.text else None)
    pg._lumen_errors = errors  # type: ignore[attr-defined]
    pg.goto(live_server.url)
    pg.wait_for_selector('[data-step="identify"]')
    yield pg
    ctx.close()


def shot(page: Any, name: str) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / f"{name}.png"
    page.screenshot(path=str(path))
    return path


def load_demo_to_place(page: Any) -> None:
    """Drive the demo session from the first step (Job) to the Place step."""
    page.click('[data-action="demo"]')
    page.wait_for_selector('[data-action="analyze"]', timeout=180_000)
    page.click('[data-action="analyze"]')
    # the Lines step searches automatically on first entry
    page.wait_for_selector('[data-datum="floor"] .chip.ok', timeout=180_000)
    page.wait_for_selector('[data-datum="side"] .chip.ok', timeout=60_000)
    page.wait_for_selector('.busy', state="detached", timeout=60_000)
    # confirming the lines leads straight to the placement
    page.click('[data-action="confirm-datums"]')
    page.wait_for_selector('[data-step="place"][aria-current="step"]', timeout=120_000)
    page.wait_for_selector('[data-action="center-target"]', timeout=120_000)
    page.wait_for_selector(".status-hero", timeout=120_000)


def open_scale_check(page: Any) -> None:
    """The square-on view with the rule measurement: a tool of the Lines step, not a stop of its own."""
    page.click('[data-step="datums"]')
    page.click('[data-action="scale-check"]')
    page.wait_for_selector('[data-region="scale-check"]', timeout=60_000)
    assert page.get_attribute('[data-step="datums"]', "aria-current") == "step", "the tool is shown as part of the Lines step"


# ---------------------------------------------------------------- fake camera
def write_y4m(path: str, frames: list, fps: int = 10) -> str:
    """Write BGR frames as a YUV4MPEG2 file (what Chromium's fake camera plays in a loop)."""
    import cv2

    h, w = frames[0].shape[:2]
    assert w % 2 == 0 and h % 2 == 0
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for f in frames:
            fh.write(b"FRAME\n")
            fh.write(cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).tobytes())
    return path


def camera_frames(n_views: int = 5, width: int = 960, height: int = 540, span_deg: float = 44.0) -> tuple[list, list]:
    """Synthetic panel views for the fake camera (rendered once, cached on disk)."""
    from lumen.synth.fixtures import flat_panel_scene, phone_camera, sweep_stations
    from tests.conftest import render_cached

    scene = flat_panel_scene()
    target = (270.0, 0.0, 300.0)
    cams = [phone_camera(e, target=target, width=width, height=height, roll_deg=r) for e, r in sweep_stations(n_views, distance=700, height=430, span_deg=span_deg, target=target)]
    return [render_cached(scene, c, ss=2, seed=i).image for i, c in enumerate(cams)], cams


def launch_with_camera(playwright_instance: Any, y4m: str) -> Any:
    return playwright_instance.chromium.launch(
        executable_path=CHROMIUM,
        args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-video-capture={y4m}"],
    )


def open_app(browser: Any, url: str, init_script: str | None = None, **ctx_kw: Any) -> Any:
    ctx = browser.new_context(**({**PHONE, **ctx_kw}))
    if init_script:
        ctx.add_init_script(init_script)
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "favicon" not in m.text else None)
    pg._lumen_errors = errors  # type: ignore[attr-defined]
    pg.goto(url)
    pg.wait_for_selector('[data-step="identify"]')
    return pg


def to_capture_step(page: Any, operation: str = "DEMO-FLAT-01") -> None:
    """Choose an operation, enter measured tag sizes and open the Photos step."""
    page.click(f'[data-operation="{operation}"]')
    page.wait_for_selector('[data-action="continue"]:not([disabled])')
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="add-tag"]')
    for tid in (3, 7, 11, 20):
        page.fill('input[aria-label="Tag ID to add"]', str(tid))
        page.fill('input[aria-label="Measured edge in millimetres"]', "50")
        page.click('[data-action="add-tag"]')
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="live-camera"]')


def assert_clean_text(page: Any) -> None:
    """No programming artefacts in what the operator reads (stray null / undefined / object text)."""
    text = page.evaluate("() => document.body.innerText")
    for bad in ("null", "undefined", "[object", "NaN"):
        assert bad not in text, f"stray {bad!r} on screen: ..." + text[max(0, text.find(bad) - 80): text.find(bad) + 80].replace("\n", " | ")
