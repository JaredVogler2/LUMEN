"""FastAPI application factory.

Single-origin design for Posit Connect (docs/POSIT_CONNECT.md):

* every URL the browser uses is *relative*; the server never emits an absolute
  path, so the app works at ``/``, under ``/content/<guid>/`` and behind any
  path-stripping reverse proxy;
* when the external base URL is known (Connect's ``RStudio-Connect-App-Base-URL``
  header, ``X-Forwarded-Prefix``, ``LUMEN_BASE_PATH`` or the ASGI ``root_path``)
  it is written into ``<base href>``; otherwise a tiny nonce-protected inline
  script repairs a missing trailing slash;
* all assets are bundled in ``lumen/static`` — no CDN, no telemetry;
* API responses are never cached; static assets carry a version query.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import ALGORITHM_VERSION, RESEARCH_NOTICE, __version__
from ..config import Settings, get_settings
from ..errors import LumenError
from ..operations.library import get_library
from ..services.sessions import SessionService
from ..storage.base import get_store

log = logging.getLogger("lumen")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def external_base(request: Request, settings: Settings) -> str | None:
    """External base path of the app ending in ``/``, or ``None`` if unknown."""
    for header in ("rstudio-connect-app-base-url", "posit-connect-app-base-url"):
        v = request.headers.get(header)
        if v:
            return v.rstrip("/") + "/"
    v = request.headers.get("x-forwarded-prefix")
    if v and v.startswith("/") and "//" not in v and ".." not in v and len(v) < 200:
        return v.rstrip("/") + "/"
    if settings.base_path:
        return settings.base_path.rstrip("/") + "/"
    root = request.scope.get("root_path") or ""
    if root:
        return root.rstrip("/") + "/"
    return None


def _csp(nonce: str) -> str:
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' blob: data:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "worker-src 'self' blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'self'"
    )


def create_app(settings: Settings | None = None, service: SessionService | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = FastAPI(
        title="LUMEN",
        version=__version__,
        description="Optical placard positioning research prototype. " + RESEARCH_NOTICE,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.service = service or SessionService(get_store(), settings, get_library())
    app.state.started = time.time()

    # ------------------------------------------------------------ errors
    @app.exception_handler(LumenError)
    async def _lumen_error(_: Request, exc: LumenError) -> JSONResponse:
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.http_status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail), "details": {}, "suggestions": []}},
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        ref = secrets.token_hex(4)
        log.exception("unexpected error ref=%s", ref)
        return JSONResponse(
            {"error": {"code": "internal_error", "message": f"Unexpected server error (reference {ref}).", "details": {}, "suggestions": ["Retry; if it persists, report the reference."]}},
            status_code=500,
        )

    # -------------------------------------------------------- middleware
    @app.middleware("http")
    async def _guards(request: Request, call_next: Any) -> Response:
        lim = settings.limits
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                n = int(cl)
            except ValueError:
                n = -1
            cap = max(lim.max_image_bytes, lim.max_video_bytes, lim.max_json_bytes)
            if n < 0 or n > cap:
                return JSONResponse({"error": {"code": "payload_too_large", "message": "Request body is too large.", "details": {}, "suggestions": []}}, status_code=413)
        response = await call_next(request)
        path = request.url.path
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        if "/api/" in path or path.endswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # -------------------------------------------------------------- index
    index_template = (STATIC_DIR / "index.html").read_text(encoding="utf-8") if (STATIC_DIR / "index.html").exists() else "<!doctype html><title>LUMEN</title><p>UI assets missing."

    def render_index(request: Request) -> HTMLResponse:
        nonce = secrets.token_urlsafe(16)
        base = external_base(request, settings)
        base_tag = f'<base href="{_attr(base)}">' if base else ""
        fix = (
            f'<script nonce="{nonce}">(function(){{var p=location.pathname;'
            "if(!document.querySelector('base')&&!/\\/$/.test(p)&&!/\\.html?$/.test(p)){"
            "var b=document.createElement('base');b.href=p+'/';document.head.appendChild(b);}})();</script>"
        )
        boot = {
            "version": __version__,
            "algorithm": ALGORITHM_VERSION,
            "notice": RESEARCH_NOTICE,
            "demo": settings.demo_enabled,
            "assistant": bool(settings.llm_enabled),
        }
        html = (
            index_template.replace("<!--LUMEN_BASE-->", base_tag + fix)
            .replace("__LUMEN_NONCE__", nonce)
            .replace("__LUMEN_VERSION__", __version__)
            .replace("__LUMEN_BOOT__", _attr(json.dumps(boot)))
        )
        return HTMLResponse(html, headers={"Content-Security-Policy": _csp(nonce), "Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        return render_index(request)

    @app.get("/index.html", include_in_schema=False)
    def index_html(request: Request) -> HTMLResponse:
        return render_index(request)

    from .routes import router

    app.include_router(router, prefix="/api")
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def _attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
