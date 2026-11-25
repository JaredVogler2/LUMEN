"""Unit handling.

Internally LUMEN uses millimetres and radians only.  Every value that enters
from a configuration file, a form or an API request carries an explicit unit
and is converted exactly once, at the boundary, by the helpers below.  There is
deliberately no "default unit": a missing or unknown unit is an error, because a
silently assumed unit is a classic way to misplace a placard by a factor of
25.4.
"""

from __future__ import annotations

import math
from typing import Final

from .errors import LumenValidationError

# Exact definitions (the inch is exactly 25.4 mm by international agreement).
_LENGTH_TO_MM: Final[dict[str, float]] = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "ft": 304.8,
    "um": 0.001,
}

_LENGTH_ALIASES: Final[dict[str, str]] = {
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "inch": "in",
    "inches": "in",
    '"': "in",
    "foot": "ft",
    "feet": "ft",
    "micron": "um",
    "microns": "um",
    "micrometer": "um",
    "micrometre": "um",
    "µm": "um",
}

_ANGLE_TO_RAD: Final[dict[str, float]] = {
    "rad": 1.0,
    "deg": math.pi / 180.0,
    "mrad": 0.001,
}

_ANGLE_ALIASES: Final[dict[str, str]] = {
    "radian": "rad",
    "radians": "rad",
    "degree": "deg",
    "degrees": "deg",
    "°": "deg",
    "milliradian": "mrad",
    "milliradians": "mrad",
}

LENGTH_UNITS: Final[tuple[str, ...]] = tuple(_LENGTH_TO_MM)
ANGLE_UNITS: Final[tuple[str, ...]] = tuple(_ANGLE_TO_RAD)


def normalize_length_unit(unit: str | None) -> str:
    """Return the canonical length unit token or raise."""
    if unit is None or not isinstance(unit, str) or not unit.strip():
        raise LumenValidationError(
            "unit_missing", "A length unit is required (mm, cm, m, in, ft, um)."
        )
    token = unit.strip()
    key = token if token in _LENGTH_TO_MM else token.lower()
    key = _LENGTH_ALIASES.get(key, key)
    if key not in _LENGTH_TO_MM:
        raise LumenValidationError(
            "unit_unknown",
            f"Unknown length unit {unit!r}. Supported: {', '.join(LENGTH_UNITS)}.",
        )
    return key


def normalize_angle_unit(unit: str | None) -> str:
    """Return the canonical angle unit token or raise."""
    if unit is None or not isinstance(unit, str) or not unit.strip():
        raise LumenValidationError(
            "unit_missing", "An angle unit is required (deg, rad, mrad)."
        )
    token = unit.strip()
    key = token if token in _ANGLE_TO_RAD else token.lower()
    key = _ANGLE_ALIASES.get(key, key)
    if key not in _ANGLE_TO_RAD:
        raise LumenValidationError(
            "unit_unknown",
            f"Unknown angle unit {unit!r}. Supported: {', '.join(ANGLE_UNITS)}.",
        )
    return key


def _finite(value: float, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise LumenValidationError("not_a_number", f"{what} must be a number.") from exc
    if not math.isfinite(v):
        raise LumenValidationError("not_finite", f"{what} must be finite.")
    return v


def to_mm(value: float, unit: str | None) -> float:
    """Convert ``value`` expressed in ``unit`` to millimetres."""
    return _finite(value, "Length") * _LENGTH_TO_MM[normalize_length_unit(unit)]


def from_mm(value_mm: float, unit: str | None) -> float:
    """Convert millimetres to ``unit``."""
    return _finite(value_mm, "Length") / _LENGTH_TO_MM[normalize_length_unit(unit)]


def to_rad(value: float, unit: str | None) -> float:
    """Convert ``value`` expressed in ``unit`` to radians."""
    return _finite(value, "Angle") * _ANGLE_TO_RAD[normalize_angle_unit(unit)]


def from_rad(value_rad: float, unit: str | None) -> float:
    """Convert radians to ``unit``."""
    return _finite(value_rad, "Angle") / _ANGLE_TO_RAD[normalize_angle_unit(unit)]


def convert_length(value: float, from_unit: str | None, to_unit: str | None) -> float:
    return from_mm(to_mm(value, from_unit), to_unit)


def wrap_angle(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    t = math.fmod(theta + math.pi, 2.0 * math.pi)
    if t <= 0.0:
        t += 2.0 * math.pi
    return t - math.pi


def format_length(value_mm: float, unit: str = "mm", digits: int | None = None) -> str:
    """Human readable length for the UI and reports."""
    u = normalize_length_unit(unit)
    v = from_mm(value_mm, u)
    if digits is None:
        digits = {"mm": 2, "cm": 3, "m": 4, "in": 3, "ft": 4, "um": 0}[u]
    return f"{v:.{digits}f} {u}"
