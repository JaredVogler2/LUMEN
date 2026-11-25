"""Named synthetic scenes and camera stations.

These are the *only* image sources in the repository: generic invented panels
used by the tests, the benchmark and the demo mode.  The world frame of each
scene coincides with the local engineering frame of the matching sample
operation in ``lumen/operations/data/operations.json`` (``x`` from the side
datum, ``z`` from the floor datum), so ground truth is simply "world x, z".
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .scene import (
    CameraSpec,
    CurvedEdge,
    FloorSpec,
    LightSpec,
    PanelSpec,
    Placard,
    Scene,
    Seam,
    TagSticker,
)

#: default tag constellation of the demo scenes: (id, centre (x, z), rotation deg)
DEMO_TAGS = ((3, (95.0, 125.0), 12.0), (7, (455.0, 105.0), -25.0), (11, (110.0, 495.0), 80.0), (20, (450.0, 505.0), 190.0))


def demo_tags(size: float = 50.0, ids: tuple[int, ...] | None = None, **kw: object) -> tuple[TagSticker, ...]:
    out = []
    for tid, c, rot in DEMO_TAGS:
        if ids is not None and tid not in ids:
            continue
        out.append(TagSticker(tid, c, rot, size=size, **kw))  # type: ignore[arg-type]
    return tuple(out)


def flat_panel_scene(
    *,
    tags: tuple[TagSticker, ...] | None = None,
    placards: tuple[Placard, ...] = (),
    seam_x: float = 0.0,
    extra_seams: tuple[Seam, ...] = (),
    light: LightSpec | None = None,
    panel_kw: dict | None = None,
    floor_kw: dict | None = None,
) -> Scene:
    """Flat panel meeting the floor, with a vertical joint seam at ``x = seam_x``.

    Matches sample operation ``DEMO-FLAT-01``: floor datum ``z = 0`` is the
    floor/panel interface, side datum ``x = 0`` is the joint seam.
    """
    panel = PanelSpec(x_min=-400.0, x_max=1400.0, z_min=0.0, z_max=1100.0, **(panel_kw or {}))
    seams = (Seam((seam_x, 0.0), (seam_x, 1100.0), width=2.0, darkness=0.7),) + tuple(extra_seams)
    return Scene(
        panel=panel,
        floor=FloorSpec(**(floor_kw or {})),
        stickers=demo_tags() if tags is None else tags,
        placards=placards,
        seams=seams,
        light=light or LightSpec(ambient=0.8),
    )


def curved_panel_scene(
    *,
    radius: float = 35.0,
    sweep_deg: float = 90.0,
    taper_deg: float = 0.0,
    return_length: float = 150.0,
    tags: tuple[TagSticker, ...] | None = None,
    placards: tuple[Placard, ...] = (),
    seams: tuple[Seam, ...] = (),
    light: LightSpec | None = None,
    panel_kw: dict | None = None,
    floor_kw: dict | None = None,
) -> Scene:
    """Panel whose left side bends away in a cylinder; tangent line at ``x = 0``.

    Matches sample operation ``DEMO-CURVE-01``.
    """
    panel = PanelSpec(
        x_max=1400.0,
        z_min=0.0,
        z_max=1100.0,
        left_edge=CurvedEdge(position=0.0, radius=radius, sweep_deg=sweep_deg, taper_deg=taper_deg, return_length=return_length),
        **(panel_kw or {}),
    )
    return Scene(
        panel=panel,
        floor=FloorSpec(**(floor_kw or {})),
        stickers=demo_tags() if tags is None else tags,
        placards=placards,
        seams=seams,
        light=light or LightSpec(ambient=0.8),
    )


def demo_placard(x: float = 240.0, z: float = 300.0, width: float = 120.0, height: float = 60.0, rotation_deg: float = 0.0, name: str = "PLC-DEMO-101", **kw: object) -> Placard:
    return Placard(name, (x, z), width, height, rotation_deg, **kw)  # type: ignore[arg-type]


def phone_camera(
    eye: tuple[float, float, float],
    target: tuple[float, float, float] = (270.0, 0.0, 300.0),
    *,
    width: int = 1600,
    height: int = 1200,
    hfov_deg: float = 67.0,
    roll_deg: float = 0.0,
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    **kw: object,
) -> CameraSpec:
    """A handheld-phone-like camera: ~67° horizontal field of view in landscape."""
    base = {"blur_sigma": 0.8, "noise_shot": 0.0006, "noise_read": 0.004}
    base.update(kw)  # type: ignore[arg-type]
    return CameraSpec.look_at(width, height, hfov_deg, eye, target, roll_deg, dist, **base)


def sweep_stations(n: int = 7, distance: float = 720.0, height: float = 430.0, span_deg: float = 50.0, target: tuple[float, float, float] = (270.0, 0.0, 300.0)) -> list[tuple[tuple[float, float, float], float]]:
    """Camera centres (and a small roll) on an arc in front of the panel."""
    out = []
    for k in range(n):
        a = np.deg2rad(-span_deg / 2 + span_deg * k / max(n - 1, 1))
        x = target[0] + distance * np.sin(a)
        y = -distance * np.cos(a)
        z = height + 60.0 * np.cos(2.3 * k)
        out.append(((float(x), float(y), float(z)), float(4.0 * np.sin(1.7 * k))))
    return out


def with_noise(cam: CameraSpec, level: str = "low") -> CameraSpec:
    table = {
        "none": (0.0, 0.0),
        "low": (0.0006, 0.004),
        "mid": (0.0015, 0.006),
        "high": (0.004, 0.012),
    }
    s, r = table[level]
    return replace(cam, noise_shot=s, noise_read=r)
