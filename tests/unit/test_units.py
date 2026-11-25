import math

import pytest

from lumen.errors import LumenValidationError
from lumen.units import convert_length, format_length, from_mm, from_rad, to_mm, to_rad, wrap_angle


def test_inch_is_exactly_25_4_mm():
    assert to_mm(1, "in") == 25.4
    assert to_mm(16.5, "in") == pytest.approx(419.1, abs=1e-12)
    assert from_mm(25.4, "in") == 1.0


@pytest.mark.parametrize("value,unit,mm", [(1, "mm", 1.0), (1, "cm", 10.0), (1, "m", 1000.0), (1, "ft", 304.8), (250, "um", 0.25), (2, "inches", 50.8), (3, "Millimetres", 3.0)])
def test_length_table(value, unit, mm):
    assert to_mm(value, unit) == pytest.approx(mm, rel=1e-15)


def test_round_trip_all_units():
    for u in ("mm", "cm", "m", "in", "ft", "um"):
        assert from_mm(to_mm(123.456, u), u) == pytest.approx(123.456, rel=1e-14)
    assert convert_length(12, "in", "ft") == pytest.approx(1.0)


def test_angles():
    assert to_rad(180, "deg") == pytest.approx(math.pi)
    assert to_rad(1000, "mrad") == pytest.approx(1.0)
    assert from_rad(math.pi / 2, "degrees") == pytest.approx(90.0)


@pytest.mark.parametrize("unit", [None, "", "  ", "furlong", "px", 5])
def test_missing_or_unknown_unit_is_an_error(unit):
    with pytest.raises(LumenValidationError) as e:
        to_mm(1.0, unit)  # type: ignore[arg-type]
    assert e.value.code in ("unit_missing", "unit_unknown")


def test_there_is_no_default_unit():
    # a bare number must never silently become millimetres
    with pytest.raises(LumenValidationError):
        to_mm(50.0, None)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc", None])
def test_non_finite_values_rejected(bad):
    with pytest.raises(LumenValidationError):
        to_mm(bad, "mm")  # type: ignore[arg-type]


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(0.1) == pytest.approx(0.1)
    assert -math.pi < wrap_angle(-math.pi + 1e-9) <= math.pi


def test_format_length():
    assert format_length(25.4, "in") == "1.000 in"
    assert format_length(12.345, "mm") == "12.35 mm" or format_length(12.345, "mm") == "12.34 mm"
