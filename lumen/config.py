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
    #: Wall-clock budget of one synchronous request (detection, overlay, track).
    request_timeout_seconds: float = 30.0
    #: Concurrent CPU-heavy computations per worker process.
    max_concurrent_jobs: int = 2
    max_json_bytes: int = 2 * 1024 * 1024
    max_track_long_edge: int = 1600


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    storage: str = "file"  # "file" | "memory"
    data_dir: Path = field(default_factory=_default_data_dir)
    #: Directory of the editable operation library ("" = bundled read-only sample).
    operations_dir: str = ""
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
    #: Retain uploaded photos after the session ends (default: never).
    retain_images: bool = False
    #: JSON file describing a physically validated measurement domain ("" = none).
    validated_domain_file: str = ""
    log_level: str = "INFO"

    @property
    def on_posit_connect(self) -> bool:
        return bool(os.environ.get("CONNECT_SERVER") or os.environ.get("RSTUDIO_PRODUCT") == "CONNECT" or os.environ.get("POSIT_PRODUCT") == "CONNECT")


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
        request_timeout_seconds=_env_float("LUMEN_REQUEST_TIMEOUT_SECONDS", 30.0),
        max_concurrent_jobs=max(1, _env_int("LUMEN_MAX_CONCURRENT_JOBS", 2)),
        max_track_long_edge=_env_int("LUMEN_MAX_TRACK_LONG_EDGE", 1600),
    )
    storage = (_env("LUMEN_STORAGE", "file") or "file").lower()
    if storage not in ("file", "memory"):
        storage = "file"
    return Settings(
        limits=lim,
        storage=storage,
        data_dir=_default_data_dir(),
        operations_dir=_env("LUMEN_OPERATIONS_DIR", "") or "",
        base_path=(_env("LUMEN_BASE_PATH", "") or "").rstrip("/"),
        demo_enabled=_env_bool("LUMEN_DEMO_ENABLED", True),
        llm_enabled=_env_bool("LUMEN_LLM_ENABLED", False),
        llm_base_url=_env("LUMEN_LLM_BASE_URL", "") or "",
        llm_model=_env("LUMEN_LLM_MODEL", "") or "",
        llm_api_key_env=_env("LUMEN_LLM_API_KEY_ENV", "LUMEN_LLM_API_KEY") or "LUMEN_LLM_API_KEY",
        llm_timeout_seconds=_env_float("LUMEN_LLM_TIMEOUT_SECONDS", 20.0),
        photometric_enabled=_env_bool("LUMEN_PHOTOMETRIC_ENABLED", True),
        retain_images=_env_bool("LUMEN_RETAIN_IMAGES", False),
        validated_domain_file=_env("LUMEN_VALIDATED_DOMAIN_FILE", "") or "",
        log_level=(_env("LUMEN_LOG_LEVEL", "INFO") or "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    """For tests that change the environment."""
    get_settings.cache_clear()
