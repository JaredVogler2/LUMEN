"""Scene description for the deterministic synthetic renderer.

World frame of a synthetic scene (all lengths in millimetres):

* ``+X`` runs along the floor seam, to the viewer's right,
* ``+Z`` runs up the panel, perpendicular to the seam, in the panel plane,
* ``+Y`` points *into* the panel (away from the viewer); the camera sits at
  ``Y < 0``.  The flat part of the panel is the plane ``Y = 0``.

The panel may end on either side in a cylindrical bend that curves away from
the viewer.  Everything here is generic, invented geometry: no real aircraft
panel, part number or dimension is represented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Color = tuple[float, float, float]


@dataclass(frozen=True)
class CurvedEdge:
    """Cylindrical bend at one side of the panel.

    ``position`` is the X coordinate of the tangent line (where flat becomes
    curved) at floor level ``Z = 0``.  ``taper_deg`` tilts the tangent line and
    cylinder axis inside the panel plane (0 = perpendicular to the floor seam).
    """

    position: float
    radius: float = 30.0
    sweep_deg: float = 90.0
    taper_deg: float = 0.0
    return_length: float = 0.0


@dataclass(frozen=True)
class PanelSpec:
    x_min: float = -50.0  # used when there is no left curved edge
    x_max: float = 900.0  # used when there is no right curved edge
    z_min: float = 0.0  # bottom edge of the panel (> 0 leaves a gap above the floor)
    z_max: float = 900.0
    albedo: Color = (0.74, 0.74, 0.72)
    texture_contrast: float = 0.0  # 0 = perfectly uniform decor
    texture_scale: float = 6.0  # mm per noise cell
    spec_strength: float = 0.0  # Blinn-Phong k_s
    shininess: float = 60.0
    left_edge: CurvedEdge | None = None
    right_edge: CurvedEdge | None = None
    back_albedo: Color = (0.08, 0.08, 0.08)


@dataclass(frozen=True)
class FloorSpec:
    present: bool = True
    dihedral_deg: float = 90.0  # angle between panel "up" and the floor direction
    depth: float = 1500.0
    x_min: float = -1500.0
    x_max: float = 2500.0
    albedo: Color = (0.30, 0.31, 0.34)
    texture_contrast: float = 0.25
    texture_scale: float = 9.0
    spec_strength: float = 0.0
    shininess: float = 30.0


@dataclass(frozen=True)
class TrimSpec:
    """A kick strip fixed to the bottom of the panel (proud of the panel plane)."""

    height: float = 60.0
    thickness: float = 8.0
    x_min: float = -2000.0
    x_max: float = 3000.0
    albedo: Color = (0.20, 0.20, 0.22)


@dataclass(frozen=True)
class TagSticker:
    tag_id: int
    center: tuple[float, float]  # surface coordinates on the host, mm
    rotation_deg: float = 0.0  # counter-clockwise as seen by the viewer
    size: float = 50.0  # TRUE black-square (detection) edge length, mm
    margin: float = 5.0  # white film around the black square (left/right/top)
    margin_bottom: float = 12.0
    host: Literal["panel", "floor"] = "panel"
    lift: float = 0.0  # raised parallel to the panel by this many mm (shim)
    white: float = 0.90
    black: float = 0.045
    spec_strength: float = 0.0
    shininess: float = 80.0
    label: bool = True  # small dark blocks under the tag, like a printed ID


@dataclass(frozen=True)
class Placard:
    name: str
    center: tuple[float, float]  # panel surface coordinates of the placard centre
    width: float
    height: float
    rotation_deg: float = 0.0
    albedo: Color = (0.93, 0.93, 0.90)
    border: float = 3.0
    border_albedo: Color = (0.75, 0.10, 0.08)
    text_lines: int = 3
    text_albedo: Color = (0.10, 0.10, 0.12)
    corner_radius: float = 0.0
    spec_strength: float = 0.0
    shininess: float = 90.0
    #: (corner index 0=TL 1=TR 2=BR 3=BL, leg length mm): corner folded over.
    curl: tuple[int, float] | None = None


@dataclass(frozen=True)
class Seam:
    """A line painted on / cut into the panel surface (in surface coordinates)."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    width: float = 1.5
    darkness: float = 0.6  # multiplicative albedo drop at the centre (0..1)
    soft: float = 0.0  # > 0 gives a smooth (lighting-gradient-like) profile, sigma in mm


@dataclass(frozen=True)
class Box:
    """Axis-aligned opaque box (occluder / shadow caster)."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    albedo: Color = (0.15, 0.15, 0.16)


@dataclass(frozen=True)
class LightSpec:
    ambient: float = 0.75  # uniform ambient term (pixel value of a white Lambertian surface)
    #: Optional directional key light: (direction the light travels, strength).
    directional: tuple[tuple[float, float, float], float] | None = None
    flash: bool = False
    flash_strength: float = 0.9  # contribution at ``flash_ref_distance``, normal incidence
    flash_ref_distance: float = 500.0
    flash_offset_cam: tuple[float, float, float] = (-12.0, 9.0, 0.0)  # mm, camera frame
    flash_beam_exponent: float = 1.0  # cos^m beam roll-off around the optical axis


@dataclass(frozen=True)
class BackgroundSpec:
    y: float = 900.0  # far wall behind the panel
    albedo: Color = (0.16, 0.17, 0.19)
    texture_contrast: float = 0.35
    texture_scale: float = 60.0
    void: Color = (0.02, 0.02, 0.02)


@dataclass(frozen=True)
class Scene:
    panel: PanelSpec = field(default_factory=PanelSpec)
    floor: FloorSpec = field(default_factory=FloorSpec)
    trim: TrimSpec | None = None
    stickers: tuple[TagSticker, ...] = ()
    placards: tuple[Placard, ...] = ()
    seams: tuple[Seam, ...] = ()
    boxes: tuple[Box, ...] = ()
    light: LightSpec = field(default_factory=LightSpec)
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    texture_seed: int = 7


@dataclass(frozen=True)
class CameraSpec:
    """Intrinsics, pose and imaging pipeline parameters of one synthetic shot."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    #: ``T_world_cam`` rotation (columns = camera x/y/z axes in world) and centre.
    R_wc: tuple[tuple[float, float, float], ...] = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    C: tuple[float, float, float] = (300.0, -600.0, 300.0)
    exposure: float = 1.0
    auto_exposure: bool = False
    auto_exposure_target: float = 0.42  # mean linear value AE aims for
    blur_sigma: float = 0.6  # px, lens/defocus PSF
    motion_blur_px: float = 0.0
    motion_blur_deg: float = 0.0
    noise_shot: float = 0.0  # variance = shot * signal + read^2 (linear domain)
    noise_read: float = 0.0
    vignette: float = 0.0  # 0..1 strength of cos^4-like falloff
    sharpen: float = 0.0  # unsharp-mask amount applied after gamma (ISP-like halo)
    tone_gamma: float = 1.0  # extra power applied to encoded values (non-sRGB rendering)
    tone_contrast: float = 0.0  # S-curve strength applied to encoded values
    jpeg_quality: int | None = None
    white_balance: Color = (1.0, 1.0, 1.0)

    @staticmethod
    def look_at(
        width: int,
        height: int,
        hfov_deg: float,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        roll_deg: float = 0.0,
        dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        **kw: object,
    ) -> "CameraSpec":
        f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
        e = np.asarray(eye, dtype=float)
        fwd = np.asarray(target, dtype=float) - e
        fwd /= np.linalg.norm(fwd)
        upv = np.asarray(up, dtype=float)
        r = np.cross(fwd, upv)
        if np.linalg.norm(r) < 1e-9:
            r = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        r /= np.linalg.norm(r)
        d = np.cross(fwd, r)
        R = np.stack([r, d, fwd], axis=1)
        if roll_deg:
            a = np.deg2rad(roll_deg)
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            R = R @ Rz
        return CameraSpec(
            width=width,
            height=height,
            fx=float(f),
            fy=float(f),
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
            dist=dist,
            R_wc=tuple(tuple(float(x) for x in row) for row in R),
            C=(float(e[0]), float(e[1]), float(e[2])),
            **kw,  # type: ignore[arg-type]
        )
