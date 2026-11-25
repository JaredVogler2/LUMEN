"""Local engineering frame built from two datum lines, and placard geometry.

Everything here lives in the 2-D *plane frame* (millimetres on the target
plane, see :mod:`lumen.vision.planar`).  A local frame assigns coordinates
``(x, z)`` to plane points:

* ``z`` is governed by the **floor datum** (``z = 0`` on it),
* ``x`` is governed by the **side datum** (``x = 0`` on it),
* the positive directions are *configured*, never guessed from gravity or
  image "up": each datum declares whether its coordinate grows toward the
  panel or away from it, and "toward the panel" is the side of the datum line
  that contains a caller-supplied interior point (normally the tag centroid).

Three axis policies exist because a floor seam and a side edge are not
guaranteed to be perpendicular:

``floor_primary``
    ``+x`` runs *along the floor line*; ``+z`` is perpendicular to it.  The
    side line only fixes the origin.  This orthogonalises the frame and is
    allowed only while the measured angle between the datums is within the
    configured limit of the nominal angle.
``side_primary``
    ``+z`` runs along the side line; ``+x`` is perpendicular to it.
``perpendicular_offsets``
    ``x`` is the perpendicular distance from the side line and ``z`` the
    perpendicular distance from the floor line — what a technician measures
    with a ruler from each datum.  No orthogonality is assumed at all.

The raw angle between the datum lines is always preserved in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lines import Line2D

AXIS_POLICIES = ("floor_primary", "side_primary", "perpendicular_offsets")
REFERENCE_POINTS = (
    "center",
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
    "bottom_center",
    "top_center",
    "left_center",
    "right_center",
)

# offset of the rectangle centre from the reference point, in units of
# (half width, half height) along (width axis, height axis)
_REF_TO_CENTER = {
    "center": (0.0, 0.0),
    "bottom_left": (1.0, 1.0),
    "bottom_right": (-1.0, 1.0),
    "top_left": (1.0, -1.0),
    "top_right": (-1.0, -1.0),
    "bottom_center": (0.0, 1.0),
    "top_center": (0.0, -1.0),
    "left_center": (1.0, 0.0),
    "right_center": (-1.0, 0.0),
}


@dataclass
class LocalFrame:
    """Affine map between local ``(x, z)`` and plane ``(X, Y)`` millimetres."""

    #: 3x3 homogeneous matrix: ``[X, Y, 1]^T = M @ [x, z, 1]^T``
    M: np.ndarray
    origin: np.ndarray  # plane coordinates of (x, z) = (0, 0)
    #: unit plane direction of increasing x at constant z, and of increasing z at constant x
    ex: np.ndarray
    ez: np.ndarray
    policy: str
    #: "left" / "right": where the side datum lies when facing the panel with +z up
    side_of_panel: str
    raw_angle_deg: float  # angle between the two datum lines, 0..180 measured from +x-ish to +z-ish
    nominal_angle_deg: float
    axis_disagreement_deg: float
    orthogonalized: bool
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    x_positive: str = "toward_panel"
    z_positive: str = "toward_panel"

    @property
    def Minv(self) -> np.ndarray:
        return np.linalg.inv(self.M)

    def to_plane(self, xz: np.ndarray) -> np.ndarray:
        p = np.asarray(xz, dtype=float)
        return p @ self.M[:2, :2].T + self.M[:2, 2]

    def to_local(self, q: np.ndarray) -> np.ndarray:
        Mi = self.Minv
        p = np.asarray(q, dtype=float)
        return p @ Mi[:2, :2].T + Mi[:2, 2]

    @property
    def ortho_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Orthonormal pair used for metric constructions (placard rectangle):
        the ``+x`` direction and the perpendicular on the ``+z`` side."""
        ex = self.ex / np.linalg.norm(self.ex)
        perp = np.array([-ex[1], ex[0]])
        if float(perp @ self.ez) < 0:
            perp = -perp
        return ex, perp

    def to_dict(self) -> dict[str, Any]:
        return {
            "M": self.M.tolist(),
            "origin": self.origin.tolist(),
            "ex": self.ex.tolist(),
            "ez": self.ez.tolist(),
            "policy": self.policy,
            "side_of_panel": self.side_of_panel,
            "raw_angle_deg": round(self.raw_angle_deg, 4),
            "nominal_angle_deg": self.nominal_angle_deg,
            "axis_disagreement_deg": round(self.axis_disagreement_deg, 4),
            "orthogonalized": self.orthogonalized,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "x_positive": self.x_positive,
            "z_positive": self.z_positive,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LocalFrame":
        return LocalFrame(
            M=np.asarray(d["M"], dtype=float),
            origin=np.asarray(d["origin"], dtype=float),
            ex=np.asarray(d["ex"], dtype=float),
            ez=np.asarray(d["ez"], dtype=float),
            policy=str(d["policy"]),
            side_of_panel=str(d["side_of_panel"]),
            raw_angle_deg=float(d["raw_angle_deg"]),
            nominal_angle_deg=float(d["nominal_angle_deg"]),
            axis_disagreement_deg=float(d["axis_disagreement_deg"]),
            orthogonalized=bool(d["orthogonalized"]),
            valid=bool(d["valid"]),
            reasons=list(d.get("reasons", [])),
            x_positive=str(d.get("x_positive", "toward_panel")),
            z_positive=str(d.get("z_positive", "toward_panel")),
        )


def build_local_frame(
    floor: Line2D,
    side: Line2D,
    interior_point: np.ndarray,
    *,
    policy: str,
    nominal_angle_deg: float = 90.0,
    max_axis_disagreement_deg: float | None = None,
    x_positive: str = "toward_panel",
    z_positive: str = "toward_panel",
    expected_side: str | None = None,
    min_intersection_angle_deg: float = 20.0,
) -> LocalFrame:
    """Construct the local frame.  Never raises for geometric problems: an
    unusable frame is returned with ``valid = False`` and explicit reasons, so
    the caller can show the evidence and still block guidance."""
    if policy not in AXIS_POLICIES:
        raise ValueError(f"unknown axis policy {policy!r}")
    reasons: list[str] = []
    ip = np.asarray(interior_point, dtype=float)
    f = floor.oriented_towards(ip)  # normal -> panel side ("up")
    s = side.oriented_towards(ip)  # normal -> panel interior
    if abs(float(f.signed_distance(ip))) < 1e-6 or abs(float(s.signed_distance(ip))) < 1e-6:
        reasons.append("interior_point_on_datum")

    # directions along each line, pointing toward the panel interior / upward
    t_floor = f.direction if float(f.direction @ s.n) >= 0 else -f.direction
    t_side = s.direction if float(s.direction @ f.n) >= 0 else -s.direction
    cosang = float(np.clip(t_floor @ t_side, -1.0, 1.0))
    raw_angle = float(np.degrees(np.arccos(cosang)))
    disagreement = abs(raw_angle - nominal_angle_deg)

    origin = f.intersect(s, min_sin=1e-9)
    sin_between = float(np.sin(np.radians(raw_angle)))
    if origin is None or abs(sin_between) < np.sin(np.radians(min_intersection_angle_deg)):
        reasons.append("datums_nearly_parallel")
        origin = np.zeros(2) if origin is None else origin

    orthogonalized = False
    if policy == "floor_primary":
        ex, ez = t_floor, f.n
        orthogonalized = True
        M2 = np.stack([ex, ez], axis=1)
        off = origin
    elif policy == "side_primary":
        ex, ez = s.n, t_side
        orthogonalized = True
        M2 = np.stack([ex, ez], axis=1)
        off = origin
    else:  # perpendicular_offsets:  n_s.q - d_s = x ;  n_f.q - d_f = z
        A = np.stack([s.n, f.n])
        if abs(float(np.linalg.det(A))) < 1e-9:
            M2 = np.eye(2)
            off = origin
        else:
            Ainv = np.linalg.inv(A)
            M2 = Ainv
            off = Ainv @ np.array([s.d, f.d])
        ex, ez = t_floor, t_side

    if orthogonalized:
        if max_axis_disagreement_deg is None:
            reasons.append("axis_limit_not_configured")
        elif disagreement > max_axis_disagreement_deg:
            reasons.append("axis_disagreement_exceeds_limit")

    if x_positive == "away_from_panel":
        M2 = M2 * np.array([-1.0, 1.0])
        ex = -ex
    elif x_positive != "toward_panel":
        reasons.append("x_sign_not_configured")
    if z_positive == "away_from_panel":
        M2 = M2 * np.array([1.0, -1.0])
        ez = -ez
    elif z_positive != "toward_panel":
        reasons.append("z_sign_not_configured")

    # Where is the side datum for someone facing the panel with "up" = away from the floor?
    # In the (right-handed, y-up as viewed) plane frame: inward normal of the side
    # line 90° clockwise from the floor's upward normal  <=>  side datum on the left.
    cross = float(s.n[0] * f.n[1] - s.n[1] * f.n[0])
    side_of_panel = "left" if cross > 0 else "right"
    if expected_side in ("left", "right") and expected_side != side_of_panel:
        reasons.append("side_datum_on_unexpected_side")

    M = np.eye(3)
    M[:2, :2] = M2
    M[:2, 2] = off
    return LocalFrame(
        M=M,
        origin=np.asarray(off, dtype=float),
        ex=np.asarray(ex, dtype=float),
        ez=np.asarray(ez, dtype=float),
        policy=policy,
        side_of_panel=side_of_panel,
        raw_angle_deg=raw_angle,
        nominal_angle_deg=float(nominal_angle_deg),
        axis_disagreement_deg=float(disagreement),
        orthogonalized=orthogonalized,
        valid=not reasons,
        reasons=reasons,
        x_positive=x_positive,
        z_positive=z_positive,
    )


@dataclass(frozen=True)
class RectTarget:
    """A rectangular placard footprint in the plane frame (true rectangle)."""

    corners: np.ndarray  # (4, 2) plane mm: TL, TR, BR, BL in local semantics
    center: np.ndarray
    reference: np.ndarray  # plane position of the configured reference point
    width: float
    height: float
    angle: float
    w_dir: np.ndarray
    h_dir: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": self.corners.tolist(),
            "center": self.center.tolist(),
            "reference": self.reference.tolist(),
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
        }


def target_rectangle(
    frame: LocalFrame,
    x: float,
    z: float,
    width: float,
    height: float,
    angle: float = 0.0,
    reference_point: str = "center",
) -> RectTarget:
    """Corners of a ``width x height`` placard whose reference point sits at
    local ``(x, z)``.

    ``angle`` rotates the placard from ``+x`` toward ``+z`` (radians).  "Left"
    means lower ``x`` and "bottom" lower ``z`` at zero angle, independent of
    how the frame happens to appear in a photo.
    """
    if reference_point not in _REF_TO_CENTER:
        raise ValueError(f"unknown reference point {reference_point!r}")
    if width <= 0 or height <= 0:
        raise ValueError("placard width and height must be positive")
    ex_o, ez_o = frame.ortho_axes
    c, s = np.cos(angle), np.sin(angle)
    w_dir = c * ex_o + s * ez_o
    h_dir = -s * ex_o + c * ez_o
    ref = frame.to_plane(np.array([x, z]))
    kx, kz = _REF_TO_CENTER[reference_point]
    center = ref + kx * 0.5 * width * w_dir + kz * 0.5 * height * h_dir
    hw, hh = 0.5 * width, 0.5 * height
    corners = np.stack(
        [
            center - hw * w_dir + hh * h_dir,  # TL
            center + hw * w_dir + hh * h_dir,  # TR
            center + hw * w_dir - hh * h_dir,  # BR
            center - hw * w_dir - hh * h_dir,  # BL
        ]
    )
    return RectTarget(corners=corners, center=center, reference=ref, width=float(width), height=float(height), angle=float(angle), w_dir=w_dir, h_dir=h_dir)


def rectangle_pose_in_frame(frame: LocalFrame, corners_plane: np.ndarray) -> dict[str, float]:
    """Describe a measured quadrilateral (TL, TR, BR, BL in plane mm) in local
    terms: centre ``(x, z)``, width, height and rotation from ``+x`` toward ``+z``."""
    P = np.asarray(corners_plane, dtype=float).reshape(4, 2)
    center = P.mean(axis=0)
    ex_o, ez_o = frame.ortho_axes
    top = P[1] - P[0]
    bottom = P[2] - P[3]
    left = P[0] - P[3]
    right = P[1] - P[2]
    w_vec = 0.5 * (top + bottom)
    angle = float(np.arctan2(w_vec @ ez_o, w_vec @ ex_o))
    cxz = frame.to_local(center)
    return {
        "x": float(cxz[0]),
        "z": float(cxz[1]),
        "width": float(0.5 * (np.linalg.norm(top) + np.linalg.norm(bottom))),
        "height": float(0.5 * (np.linalg.norm(left) + np.linalg.norm(right))),
        "angle": angle,
    }
