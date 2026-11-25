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
    """Drive the demo session from Identify to the Place screen."""
    page.click('[data-action="demo"]')
    page.wait_for_selector('[data-action="analyze"]', timeout=180_000)
    page.click('[data-action="analyze"]')
    # the Datums screen searches automatically on first entry
    page.wait_for_selector('[data-datum="floor"] .chip.ok', timeout=180_000)
    page.wait_for_selector('[data-datum="side"] .chip.ok', timeout=60_000)
    page.wait_for_selector('.busy', state="detached", timeout=60_000)
    page.click('[data-action="confirm-datums"]')
    page.wait_for_selector(".status-hero", timeout=120_000)
    page.click('[data-action="continue"]')
    page.wait_for_selector('[data-action="center-target"]', timeout=120_000)
    page.wait_for_selector(".status-hero", timeout=120_000)
