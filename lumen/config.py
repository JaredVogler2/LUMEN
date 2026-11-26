"""Runtime configuration.

Everything is read from environment variables (``LUMEN_*``) so the same bundle
runs locally and on Posit Connect, where variables are set per content item.
No secret has a default and none is ever logged or returned by the API.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _default_data_dir() -> Path:
    base = _env("LUMEN_DATA_DIR")
    if base:
        return Path(base)
    # TMPDIR is set per content process by Posit Connect; getuid keeps local
    # multi-user machines from sharing a directory.
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"lumen-sessions-{uid}"


#: image kinds accepted for a brand logo (shown inside <img>, where an SVG cannot run scripts) and its size limit
BRAND_LOGO_TYPES = {".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
BRAND_LOGO_MAX_BYTES = 512 * 1024
#: an installation's own artwork, placed beside the application before it is deployed (see brand/README.md)
BRAND_DIR = Path(__file__).resolve().parent.parent / "brand"


def bundled_brand_logo(folder: Path | None = None) -> str:
    """Path of ``logo.<svg|png|webp|jpg|jpeg>`` in the installation's brand folder, or "" if there is none."""
    d = BRAND_DIR if folder is None else Path(folder)
    for ext in BRAND_LOGO_TYPES:
        f = d / f"logo{ext}"
        if f.is_file():
            return str(f)
    return ""


@dataclass(frozen=True)
class Limits:
    """Hard bounds that keep one request / one session from exhausting a worker."""

    max_image_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    #: Stills are analysed at most at this long-edge size (area-averaged above it).
    max_analysis_long_edge: int = 4096
    max_video_bytes: int = 300 * 1024 * 1024
    max_video_seconds: float = 60.0
    max_video_candidates: int = 90
    max_frames_per_session: int = 24
    max_sessions: int = 40
    session_ttl_seconds: int = 4 * 3600
    #: Wall-clock budget of one analysis job.
    job_timeout_seconds: float = 120.0
    #: Wall-clock budget of one video (decode, frame selection, tag measurement).
    video_job_timeout_seconds: float = 300.0
    #: Wall-clock budget of one synchronous request (detection, overlay, track).
    request_timeout_seconds: float = 30.0
    #: Concurrent CPU-heavy computations per worker process.
    max_concurrent_jobs: int = 2
    max_json_bytes: int = 2 * 1024 * 1024
    max_track_long_edge: int = 1600
    #: Where the browser carries the job (storage "client"): the largest request and the largest answer.  The
    #: defaults fit under the 4.5 MB that Vercel allows a function (AWS Lambda and Netlify allow 6 MB).
    carry_max_request_bytes: int = 4_000_000
    carry_max_response_bytes: int = 4_300_000
    #: ... and the size to which the browser reduces a photo there, so that all photos of a job fit in one request
    carry_photo_max_edge: int = 1600


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    #: Where a job lives while it is open.  "file" and "memory": with this server.  "client": with the browser
    #: that runs it, which sends it along with every request (for hosts that keep nothing between requests).
    storage: str = "file"
    #: With storage "client": a secret with which the job's state is signed, so that a state this server did not
    #: write, or that was changed, is refused.  Empty = states are checked for their form only.
    state_secret: str = ""
    data_dir: Path = field(default_factory=_default_data_dir)
    #: Directory of the editable operation library ("" = bundled read-only sample).
    operations_dir: str = ""
    #: no local operation records at all (the bundled sample only)
    operations_readonly: bool = False
    #: LUMEN_DATA_DIR was set by the administrator (otherwise sessions and local records are in a temporary directory)
    data_dir_configured: bool = False
    #: External base path when it cannot be discovered from proxy headers.
    base_path: str = ""
    demo_enabled: bool = True
    #: Optional LLM assistant (disabled unless explicitly configured).
    llm_enabled: bool = False
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key_env: str = "LUMEN_LLM_API_KEY"
    llm_timeout_seconds: float = 20.0
    #: Experimental research features; diagnostics only, never guidance.
    photometric_enabled: bool = True
    #: JSON file describing a physically validated measurement domain ("" = none).
    validated_domain_file: str = ""
    #: Branding of this installation.  The name of the organisation, an image file with its logo and its primary
    #: colour are settings of a deployment, not part of the repository: official artwork is supplied by whoever
    #: runs the installation and has the right to use it.  The logo is the file named by LUMEN_BRAND_LOGO, or
    #: ``brand/logo.<svg|png|webp|jpg>`` beside the application.
    brand_name: str = ""
    brand_logo: str = ""
    brand_color: str = ""
    log_level: str = "INFO"

    @property
    def on_posit_connect(self) -> bool:
        return bool(os.environ.get("CONNECT_SERVER") or os.environ.get("RSTUDIO_PRODUCT") == "CONNECT" or os.environ.get("POSIT_PRODUCT") == "CONNECT")

    @property
    def brand(self) -> dict[str, object]:
        """What the interface shows as the installation's brand (see ``brand_name`` and its neighbours).

        A logo is offered only if the file exists, is an image of a kind a browser shows inside ``<img>`` and is
        small; a colour only if it is a plain ``#rrggbb``.  Anything else is left out, never guessed.
        """
        logo = None
        if self.brand_logo:
            p = Path(self.brand_logo)
            try:
                if p.is_file() and p.suffix.lower() in BRAND_LOGO_TYPES and p.stat().st_size <= BRAND_LOGO_MAX_BYTES:
                    logo = f"brand/logo?v={int(p.stat().st_mtime)}"
            except OSError:
                logo = None
        color = self.brand_color.strip().lower()
        ok = len(color) == 7 and color.startswith("#") and all(c in "0123456789abcdef" for c in color[1:])
        return {"name": self.brand_name.strip()[:60], "logo": logo, "color": color if ok else None}

    @property
    def sessions(self) -> dict[str, object]:
        """Who keeps a job while it is open, and what that means for the browser.

        ``kept_by`` is "server" (the ordinary way: this process holds the photos and the analysis for the minutes
        a job takes) or "client" (the browser carries them and sends what a step needs; see ``storage/carried.py``).
        ``works`` is false only for a server that is meant to keep jobs on a host that keeps nothing.
        """
        carried = self.storage == "client"
        lim = self.limits
        out: dict[str, object] = {"kept_by": "client" if carried else "server", "works": bool(carried or self.hosting["keeps_sessions"]), "video": not carried}
        if carried:
            views = 9  # the most photos a capture asks for; all of them go into one request when the lines are found
            room = max(lim.carry_max_request_bytes - 300_000, 500_000)
            out.update({
                "max_request_bytes": lim.carry_max_request_bytes,
                "photo_max_edge": lim.carry_photo_max_edge,
                "photo_max_bytes": int(room // views),
                "single_max_bytes": int(min(room, 1_500_000)),
                "signed": bool(self.state_secret),
            })
        return out

    @property
    def hosting(self) -> dict[str, object]:
        """Where this process runs, as far as its environment says, and whether a session can survive there.

        LUMEN keeps a session (its photos, the registration, the records) with the process that serves it: in
        files under ``data_dir`` or in memory.  That needs requests of one session to reach a place that still has
        them.  A function platform starts instances as it pleases, each with a disk of its own that is empty and
        is thrown away; a session made by one request is unknown to the next.  The application cannot work
        there, and it has to say so instead of failing photo by photo.

        ``keeps_sessions`` is ``False`` where the platform is known not to keep them, ``True`` otherwise (which is
        not a promise: it says only that nothing known speaks against it).
        """
        host = forgetful_host()
        if host is not None:
            kind, name = host
            if self.storage == "client":
                # the browser carries the job: the host's forgetting does not matter
                return {"kind": kind, "name": name, "keeps_sessions": False, "message": ""}
            return {
                "kind": kind, "name": name, "keeps_sessions": False,
                "message": (
                    f"This copy of LUMEN runs on {name} and is set to keep each job on the server (LUMEN_STORAGE). "
                    "This host forgets everything between two requests: a job is unknown again seconds after it was "
                    "started, and nothing after the first step can work. Nothing has to be stored for long: remove "
                    "LUMEN_STORAGE, and the phone carries the job. Or run LUMEN where one process keeps running: "
                    "Posit Connect (docs/POSIT_CONNECT.md), a container or a virtual machine."
                ),
            }
        if self.on_posit_connect:
            return {"kind": "posit_connect", "name": "Posit Connect", "keeps_sessions": True, "message": ""}
        return {"kind": "process", "name": "", "keeps_sessions": True, "message": ""}


#: platforms that start a new, empty copy of the program for a request as they please
_FORGETFUL = (
    ("vercel", "Vercel (serverless functions)", "VERCEL"),
    ("aws_lambda", "AWS Lambda", "AWS_LAMBDA_FUNCTION_NAME"),
    ("azure_functions", "Azure Functions", "FUNCTIONS_WORKER_RUNTIME"),
    ("netlify", "Netlify Functions", "NETLIFY"),
)


def forgetful_host() -> tuple[str, str] | None:
    """(kind, name) of the function platform this process runs on, or ``None``."""
    for kind, name, var in _FORGETFUL:
        if os.environ.get(var):
            return kind, name
    return None


def load_settings() -> Settings:
    lim = Limits(
        max_image_bytes=_env_int("LUMEN_MAX_IMAGE_MB", 25) * 1024 * 1024,
        max_image_pixels=_env_int("LUMEN_MAX_IMAGE_MEGAPIXELS", 50) * 1_000_000,
        max_analysis_long_edge=_env_int("LUMEN_MAX_ANALYSIS_LONG_EDGE", 4096),
        max_video_bytes=_env_int("LUMEN_MAX_VIDEO_MB", 300) * 1024 * 1024,
        max_video_seconds=_env_float("LUMEN_MAX_VIDEO_SECONDS", 60.0),
        max_video_candidates=_env_int("LUMEN_MAX_VIDEO_CANDIDATES", 90),
        max_frames_per_session=_env_int("LUMEN_MAX_FRAMES", 24),
        max_sessions=_env_int("LUMEN_MAX_SESSIONS", 40),
        session_ttl_seconds=_env_int("LUMEN_SESSION_TTL_SECONDS", 4 * 3600),
        job_timeout_seconds=_env_float("LUMEN_JOB_TIMEOUT_SECONDS", 120.0),
        video_job_timeout_seconds=_env_float("LUMEN_VIDEO_JOB_TIMEOUT_SECONDS", 300.0),
        request_timeout_seconds=_env_float("LUMEN_REQUEST_TIMEOUT_SECONDS", 30.0),
        max_concurrent_jobs=max(1, _env_int("LUMEN_MAX_CONCURRENT_JOBS", 2)),
        max_track_long_edge=_env_int("LUMEN_MAX_TRACK_LONG_EDGE", 1600),
        carry_max_request_bytes=max(600_000, int(_env_float("LUMEN_CARRY_MAX_REQUEST_MB", 4.0) * 1_000_000)),
        carry_max_response_bytes=max(600_000, int(_env_float("LUMEN_CARRY_MAX_RESPONSE_MB", 4.3) * 1_000_000)),
        carry_photo_max_edge=max(640, _env_int("LUMEN_CARRY_PHOTO_MAX_EDGE", 1600)),
    )
    # A job is kept by the server wherever a server can keep one.  On a platform that forgets everything between
    # two requests the browser carries it, unless the administrator says otherwise.
    storage = (_env("LUMEN_STORAGE", "") or "").lower()
    if storage not in ("file", "memory", "client"):
        storage = "client" if forgetful_host() is not None else "file"
    return Settings(
        limits=lim,
        storage=storage,
        state_secret=_env("LUMEN_STATE_SECRET", "") or "",
        data_dir=_default_data_dir(),
        operations_dir=_env("LUMEN_OPERATIONS_DIR", "") or "",
        operations_readonly=_env_bool("LUMEN_OPERATIONS_READONLY", False),
        data_dir_configured=bool(_env("LUMEN_DATA_DIR")),
        base_path=(_env("LUMEN_BASE_PATH", "") or "").rstrip("/"),
        demo_enabled=_env_bool("LUMEN_DEMO_ENABLED", True),
        llm_enabled=_env_bool("LUMEN_LLM_ENABLED", False),
        llm_base_url=_env("LUMEN_LLM_BASE_URL", "") or "",
        llm_model=_env("LUMEN_LLM_MODEL", "") or "",
        llm_api_key_env=_env("LUMEN_LLM_API_KEY_ENV", "LUMEN_LLM_API_KEY") or "LUMEN_LLM_API_KEY",
        llm_timeout_seconds=_env_float("LUMEN_LLM_TIMEOUT_SECONDS", 20.0),
        photometric_enabled=_env_bool("LUMEN_PHOTOMETRIC_ENABLED", True),
        validated_domain_file=_env("LUMEN_VALIDATED_DOMAIN_FILE", "") or "",
        brand_name=_env("LUMEN_BRAND_NAME", "") or "",
        brand_logo=_env("LUMEN_BRAND_LOGO", "") or bundled_brand_logo(),
        brand_color=_env("LUMEN_BRAND_COLOR", "") or "",
        log_level=(_env("LUMEN_LOG_LEVEL", "INFO") or "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    """For tests that change the environment."""
    get_settings.cache_clear()
