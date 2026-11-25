"""Error model.

Every failure that can reach an operator carries a stable machine-readable
``code``, a plain-language ``message`` and, where it helps, ``suggestions`` for
the next capture.  The API layer converts these into a uniform JSON body:

    {"error": {"code": ..., "message": ..., "details": {...}, "suggestions": [...]}}

LUMEN fails closed: a geometry or gating error never degrades into a plausible
looking overlay.
"""

from __future__ import annotations

from typing import Any


class LumenError(Exception):
    """Base class for all expected (operator-facing) errors."""

    http_status: int = 400
    default_code: str = "lumen_error"

    def __init__(
        self,
        code: str | None = None,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = code or self.default_code
        self.message = message or self.code
        self.details = details or {}
        self.suggestions = suggestions or []
        if http_status is not None:
            self.http_status = http_status
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
        }


class LumenValidationError(LumenError):
    """Malformed or semantically invalid input (units, schemas, ranges)."""

    http_status = 422
    default_code = "validation_error"


class NotFoundError(LumenError):
    http_status = 404
    default_code = "not_found"


class LimitExceededError(LumenError):
    """A configured resource bound was hit (size, count, rate, time)."""

    http_status = 413
    default_code = "limit_exceeded"


class ConflictError(LumenError):
    """The request is valid but the session is not in a state that allows it."""

    http_status = 409
    default_code = "conflict"


class GeometryError(LumenError):
    """The observations do not support the requested geometric result.

    This is the fail-closed path: missing scale, ambiguous pose, degenerate or
    inconsistent geometry, unsupported surface.
    """

    http_status = 422
    default_code = "geometry_unsupported"


class CalibrationError(LumenError):
    http_status = 422
    default_code = "calibration_error"


class UnsupportedMediaError(LumenError):
    http_status = 415
    default_code = "unsupported_media"


class OperationTimeout(LumenError):
    http_status = 504
    default_code = "timeout"


class OperationCancelled(LumenError):
    http_status = 409
    default_code = "cancelled"


class FeatureDisabledError(LumenError):
    http_status = 403
    default_code = "feature_disabled"
