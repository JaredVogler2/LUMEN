"""Synthetic photos of printed calibration boards (tests and demos only).

A :class:`PrintedBoard` is the target *as it physically exists on the paper*:
its true square size is what the renderer draws, independently of whatever an
operator later declares to the calibration code.  Photos are produced by
inverse mapping with the renderer's own camera maths
(:func:`lumen.synth.render.pixel_rays`, Newton inversion of the lens model),
while the ground-truth corner positions come from the forward model
(:func:`lumen.synth.render.project_points`).  Neither path uses
``cv2.projectPoints`` or ``lumen.geometry.camera``, so the calibration code is
tested against images it did not help to make.

Board frame (all lengths in millimetres)
----------------------------------------
* origin at the top-left *outer* corner of the printed pattern (not of the
  sheet), ``+x`` to the right and ``+y`` down as the print is read,
  ``+z`` into the paper (away from the viewer); the camera sits at ``z < 0``;
* inner corner ``(i, j)`` (column, row) is at ``((i + 1) s, (j + 1) s, 0)`` and
  has the id ``j * (squares_x - 1) + i`` — the numbering OpenCV uses for
  ChArUco corners;
* square ``(0, 0)`` (top-left) is black (white only for a ChArUco board in
  OpenCV's pre-4.6 "legacy" layout with an even number of rows).

Image formation (linear light until the last step)
--------------------------------------------------
reflectance bitmap -> pixel-area integration -> lighting/exposure -> vignette
-> PSF blur -> motion blur -> shot/read noise -> sRGB transfer -> optional ISP
sharpening -> 8 bit -> optional JPEG.  The tail of this pipeline mirrors
:func:`lumen.synth.render.render`.

Pixel-area integration.  The pattern is a bitmap whose every boundary lies on a
bitmap-pixel boundary; it is sampled bilinearly, so an edge is a symmetric ramp
one bitmap pixel wide (at most one picture pixel where the board is closest).
The board coordinates of every pixel centre are exact (one ray per pixel); a
pixel that sees an edge is integrated with ``ss * ss`` samples placed with the
local Jacobian of that map, on a sheared lattice whose projections on both
pixel axes are equally spaced.  All sampling patterns are symmetric, so edges
are not shifted on average; what remains is a zero-mean scatter of the edge
position from row to row.  Measured against the forward-projected truth
(``tests/unit/test_calibration.py``, ``ss = 5``, five poses): mean within
0.001 px, RMS 0.007 px, largest single-row deviation 0.015 px.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import cv2
import numpy as np

from .render import pixel_rays, project_points, srgb_encode
from .scene import CameraSpec

#: Resolution of the pattern bitmap relative to the picture, where the board is closest.
_BITMAP_PX_PER_IMAGE_PX = 1.0

#: Lens distortion of the project's synthetic phone (same values as
#: ``tests/unit/test_synth.py``): k1, k2, p1, p2, k3.
PHONE_DIST: tuple[float, float, float, float, float] = (0.08, -0.15, 0.0006, -0.0004, 0.0)


# --------------------------------------------------------------------------
# The printed target
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PrintedBoard:
    """A calibration target as printed (true dimensions, not declared ones)."""

    kind: Literal["checkerboard", "charuco"] = "charuco"
    squares_x: int = 9
    squares_y: int = 6
    #: TRUE edge length of one printed square
    square_mm: float = 30.0
    #: TRUE edge length of a marker's black border (ChArUco only)
    marker_mm: float | None = 21.0
    dictionary: str = "DICT_5X5_100"
    legacy_pattern: bool = False
    #: white paper around the pattern (rounded to whole bitmap pixels when drawn)
    margin_mm: float = 15.0
    white: float = 0.90  # linear reflectance of the paper
    black: float = 0.045  # linear reflectance of the ink
    #: a printer that scales the feed direction differently: the squares are
    #: ``square_mm`` wide and ``square_mm * stretch_y`` high
    stretch_y: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("checkerboard", "charuco"):
            raise ValueError(f"unknown board kind {self.kind!r}")
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("a board needs at least 2 x 2 squares")
        if not (self.square_mm > 0):
            raise ValueError("square_mm must be positive")
        if self.kind == "charuco":
            if self.marker_mm is None or not (0 < self.marker_mm < self.square_mm):
                raise ValueError("a ChArUco board needs 0 < marker_mm < square_mm")

    # ---------------------------------------------------------------- geometry
    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm * self.stretch_y

    @property
    def center_mm(self) -> tuple[float, float]:
        return 0.5 * self.width_mm, 0.5 * self.height_mm

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def corner_ids(self) -> np.ndarray:
        return np.arange(self.n_corners, dtype=np.int64)

    def corners_mm(self) -> np.ndarray:
        """True board-frame positions ``(N, 2)`` of the inner corners, by id."""
        i, j = np.meshgrid(np.arange(self.squares_x - 1), np.arange(self.squares_y - 1))
        return np.stack([(i.ravel() + 1.0) * self.square_mm, (j.ravel() + 1.0) * self.square_mm * self.stretch_y], axis=1)

    def sheet_outline_mm(self, margin_mm: float | None = None, per_edge: int = 1) -> np.ndarray:
        """Outline of the sheet (pattern plus margin), ``per_edge`` points per side."""
        m = self.margin_mm if margin_mm is None else margin_mm
        x0, y0, x1, y1 = -m, -m, self.width_mm + m, self.height_mm + m
        c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if per_edge <= 1:
            return c
        t = np.linspace(0.0, 1.0, per_edge, endpoint=False)[:, None]
        return np.concatenate([c[k] + t * (c[(k + 1) % 4] - c[k]) for k in range(4)])

    # ------------------------------------------------------------------ bitmap
    def exact_px_per_square(self, at_least: int = 96, limit: int = 2000) -> int:
        """Smallest bitmap resolution (pixels per square) ``>= at_least`` at which
        every boundary of the pattern falls on a bitmap-pixel boundary."""
        if self.kind == "checkerboard":
            return max(int(at_least), 1)
        assert self.marker_mm is not None
        cells = _dictionary(self.dictionary).markerSize + 2  # data bits + black border
        for p in range(max(int(at_least), cells), limit + 1):
            mk = p * self.marker_mm / self.square_mm
            mki = int(round(mk))
            if abs(mk - mki) < 1e-9 and mki % cells == 0 and (p - mki) % 2 == 0:
                return p
        raise ValueError(
            "marker_mm / square_mm does not give a pixel-exact ChArUco bitmap; use a ratio such as 0.7 "
            "(5x5 dictionaries) or 0.75 (4x4, 6x6 dictionaries)"
        )

    def pattern_bitmap(self, px_per_square: int) -> np.ndarray:
        """The pattern as ``uint8`` (0 = ink, 255 = paper), without margin."""
        p = int(px_per_square)
        if self.kind == "checkerboard":
            i, j = np.meshgrid(np.arange(self.squares_x), np.arange(self.squares_y))
            cells = np.where((i + j) % 2 == 0, 0, 255).astype(np.uint8)  # square (0, 0) is black
            return np.kron(cells, np.ones((p, p), np.uint8))
        assert self.marker_mm is not None
        board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), float(self.square_mm), float(self.marker_mm), _dictionary(self.dictionary))
        board.setLegacyPattern(bool(self.legacy_pattern))
        return board.generateImage((self.squares_x * p, self.squares_y * p), marginSize=0, borderBits=1)


def _dictionary(name: str) -> Any:
    code = getattr(cv2.aruco, name, None)
    if not name.startswith("DICT_") or not isinstance(code, int):
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(code)


@functools.lru_cache(maxsize=4)
def _sheet_bitmap(board: PrintedBoard, px_per_square: int) -> tuple[np.ndarray, int]:
    """Reflectance bitmap of the sheet (float32) and its margin in bitmap pixels."""
    pat = board.pattern_bitmap(px_per_square)
    m = max(int(round(board.margin_mm * px_per_square / board.square_mm)), 0)
    refl = np.where(pat > 127, np.float32(board.white), np.float32(board.black)).astype(np.float32)
    if m:
        refl = np.pad(refl, m, constant_values=np.float32(board.white))
    refl.setflags(write=False)
    return refl, m


# --------------------------------------------------------------------------
# Cameras looking at a board
# --------------------------------------------------------------------------
def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def phone_base(
    width: int = 1600,
    height: int = 1200,
    hfov_deg: float = 67.0,
    dist: tuple[float, float, float, float, float] = PHONE_DIST,
    *,
    pp_offset: tuple[float, float] = (0.0, 0.0),
    aspect: float = 1.0,
    **imaging: Any,
) -> CameraSpec:
    """Intrinsics and imaging parameters of a phone-like camera (pose unset).

    ``pp_offset`` moves the principal point away from the image centre and
    ``aspect`` is ``fy / fx``; both let a test check that these are *measured*
    and not assumed.  Imaging defaults are those of
    :func:`lumen.synth.fixtures.phone_camera`.
    """
    f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) * 0.5)
    kw: dict[str, Any] = {"blur_sigma": 0.8, "noise_shot": 0.0006, "noise_read": 0.004}
    kw.update(imaging)
    return CameraSpec(
        width=int(width),
        height=int(height),
        fx=float(f),
        fy=float(f * aspect),
        cx=(width - 1) * 0.5 + float(pp_offset[0]),
        cy=(height - 1) * 0.5 + float(pp_offset[1]),
        dist=tuple(float(x) for x in dist),  # type: ignore[arg-type]
        R_wc=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        C=(0.0, 0.0, -1.0),
        **kw,
    )


def board_camera(
    board: PrintedBoard,
    base: CameraSpec,
    *,
    distance_mm: float = 450.0,
    tilt_deg: float = 0.0,
    azimuth_deg: float = 0.0,
    roll_deg: float = 0.0,
    aim: tuple[float, float] = (0.5, 0.5),
    at: tuple[float, float] = (0.5, 0.5),
) -> CameraSpec:
    """``base`` (intrinsics, imaging) posed in front of ``board``.

    * ``tilt_deg`` is the angle between the optical axis and the board normal
      (0 = image plane parallel to the board); the board's far side lies toward
      ``azimuth_deg`` (0 = board ``+x``, 90 = board ``+y``) before the roll;
    * ``roll_deg`` rotates the camera about its optical axis;
    * the board point ``aim`` (fractions of the pattern width/height) appears at
      the image point ``at`` (fractions of the image width/height), at
      ``distance_mm`` along that viewing ray.

    The world frame of the returned camera is the board frame.
    """
    tilt, az, roll = np.deg2rad(tilt_deg), np.deg2rad(azimuth_deg), np.deg2rad(roll_deg)
    # board -> camera rotation: tilt about the in-plane axis perpendicular to the azimuth, then roll
    R_cb = _rot_z(roll) @ _rot_axis(np.array([-np.sin(az), np.cos(az), 0.0]), -tilt)
    cam0 = replace(base, R_wc=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), C=(0.0, 0.0, 0.0))
    uv = np.array([[at[0] * base.width - 0.5, at[1] * base.height - 0.5]])
    ray = pixel_rays(cam0, uv)[0]  # unit ray of the image point, camera frame
    A = np.array([aim[0] * board.width_mm, aim[1] * board.height_mm, 0.0])
    t_cb = float(distance_mm) * ray - R_cb @ A
    R_wc = R_cb.T
    C = -R_cb.T @ t_cb
    return replace(base, R_wc=tuple(tuple(float(x) for x in row) for row in R_wc), C=(float(C[0]), float(C[1]), float(C[2])))


def true_tilt_deg(cam: CameraSpec) -> float:
    """Angle between the optical axis and the board normal (board = world frame)."""
    return float(np.degrees(np.arccos(np.clip(abs(np.asarray(cam.R_wc, dtype=float)[2, 2]), 0.0, 1.0))))


def true_pose(cam: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` with ``x_cam = R @ x_board + t`` for a camera made by :func:`board_camera`."""
    R_wc = np.asarray(cam.R_wc, dtype=float)
    return R_wc.T, -R_wc.T @ np.asarray(cam.C, dtype=float)


# --------------------------------------------------------------------------
# Pixel -> board mapping
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _normalised_lattice(width: int, height: int, fx: float, fy: float, cx: float, cy: float, dist: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted normalised coordinates of every pixel centre (``float64``).

    Depends on the intrinsics only, so one lattice serves every pose of a
    camera.  Computed with :func:`lumen.synth.render.pixel_rays` (about 1 s
    for 1600 x 1200) and kept in memory for the two most recent cameras
    (16 bytes per pixel each).
    """
    cam0 = CameraSpec(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, R_wc=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), C=(0.0, 0.0, 0.0))  # type: ignore[arg-type]
    xn = np.empty((height, width))
    yn = np.empty((height, width))
    us = np.arange(width, dtype=float)
    rows = max(1, 400_000 // width)
    for r0 in range(0, height, rows):
        r1 = min(height, r0 + rows)
        uu, vv = np.meshgrid(us, np.arange(r0, r1, dtype=float))
        d = pixel_rays(cam0, np.stack([uu.ravel(), vv.ravel()], axis=1))
        xn[r0:r1] = (d[:, 0] / d[:, 2]).reshape(r1 - r0, width)
        yn[r0:r1] = (d[:, 1] / d[:, 2]).reshape(r1 - r0, width)
    xn.setflags(write=False)
    yn.setflags(write=False)
    return xn, yn


def board_coordinates(cam: CameraSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Board-plane point ``(X, Y)`` in mm seen by every pixel centre, and a mask
    of pixels whose ray meets the plane in front of the camera."""
    xn, yn = _normalised_lattice(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, tuple(cam.dist))
    R = np.asarray(cam.R_wc, dtype=float)
    C = np.asarray(cam.C, dtype=float)
    dz = R[2, 0] * xn + R[2, 1] * yn + R[2, 2]
    hit = dz * (-C[2]) > 1e-12
    t = -C[2] / np.where(hit, dz, 1.0)
    X = C[0] + t * (R[0, 0] * xn + R[0, 1] * yn + R[0, 2])
    Y = C[1] + t * (R[1, 0] * xn + R[1, 1] * yn + R[1, 2])
    return X, Y, hit


def _sample_offsets(ss: int) -> np.ndarray:
    """``ss * ss`` sub-pixel offsets in (-0.5, 0.5)^2: a sheared grid whose
    projections on both pixel axes are equally spaced and whose mean is zero
    (a pattern with a non-zero mean would shift every edge)."""
    n = ss * ss
    a, b = np.meshgrid(np.arange(ss), np.arange(ss), indexing="ij")
    iu = (a * ss + b).ravel()
    iv = (b * ss + (ss - 1 - a)).ravel()
    return np.stack([(iu + 0.5) / n - 0.5, (iv + 0.5) / n - 0.5], axis=1)


# --------------------------------------------------------------------------
# The photo
# --------------------------------------------------------------------------
@dataclass
class BoardPhoto:
    image: np.ndarray  # uint8 BGR (OpenCV channel order)
    board: PrintedBoard
    camera: CameraSpec
    #: ids and TRUE pixel positions of every inner corner (also those outside the picture)
    corner_ids: np.ndarray
    corners_px: np.ndarray
    #: which corners are inside the picture (and on the physical branch of the lens model)
    corner_visible: np.ndarray
    #: the sheet outline in pixels and whether sheet / pattern are completely inside the picture
    outline_px: np.ndarray
    sheet_inside: bool
    pattern_inside: bool
    tilt_deg: float
    meta: dict[str, Any] = field(default_factory=dict)

    def truth(self) -> dict[str, Any]:
        R, t = true_pose(self.camera)
        c = self.camera
        return {
            "camera": {"width": c.width, "height": c.height, "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "dist": list(c.dist)},
            "R_cam_board": R.tolist(),
            "t_cam_board": t.tolist(),
            "tilt_deg": self.tilt_deg,
            "corner_ids": self.corner_ids.tolist(),
            "corners_px": self.corners_px.tolist(),
            "corner_visible": self.corner_visible.tolist(),
            "sheet_inside": self.sheet_inside,
            "pattern_inside": self.pattern_inside,
            "frame": "board: origin top-left outer pattern corner, +x right, +y down, +z into the paper; mm",
            **self.meta,
        }


def project_board_points(cam: CameraSpec, xy_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True pixel positions of board points and which of them are really seen.

    A point is *seen* when it is in front of the camera, inside the picture
    and the ray of its pixel leads back to it.  The last test matters because
    a polynomial lens model folds over far outside the field of view, where
    the forward projection can land inside the picture although no ray of the
    camera reaches the point.
    """
    xy = np.asarray(xy_mm, dtype=float).reshape(-1, 2)
    Xw = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
    uv, depth = project_points(cam, Xw)
    ok = depth > 1e-6
    ok &= (uv[:, 0] >= -0.5) & (uv[:, 0] <= cam.width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= cam.height - 0.5)
    if np.any(ok):
        d = pixel_rays(cam, uv[ok])
        C = np.asarray(cam.C, dtype=float)
        t = -C[2] / d[:, 2]
        back = C[:2] + t[:, None] * d[:, :2]
        good = np.linalg.norm(back - xy[ok], axis=1) < 1e-5
        idx = np.flatnonzero(ok)
        ok[idx[~good]] = False
    return uv, ok


def _remap(src: np.ndarray, mx: np.ndarray, my: np.ndarray, border: float) -> np.ndarray:
    """Bilinear lookup for a flat list of coordinates (any length)."""
    n = mx.size
    width = 8192
    rows = -(-n // width)
    if rows * width != n:
        mx = np.concatenate([mx, np.full(rows * width - n, -1e6, np.float32)])
        my = np.concatenate([my, np.full(rows * width - n, -1e6, np.float32)])
    out = cv2.remap(src, mx.reshape(rows, width), my.reshape(rows, width), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border))
    return out.ravel()[:n]


def _central_differences(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(da/du, da/dv)`` of a map sampled at the pixel centres."""
    gu = np.empty_like(a)
    gv = np.empty_like(a)
    gu[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    gu[:, 0] = a[:, 1] - a[:, 0]
    gu[:, -1] = a[:, -1] - a[:, -2]
    gv[1:-1] = 0.5 * (a[2:] - a[:-2])
    gv[0] = a[1] - a[0]
    gv[-1] = a[-1] - a[-2]
    return gu, gv


def board_reflectance(board: PrintedBoard, cam: CameraSpec, *, ss: int = 5, background: float = 0.30, adaptive: bool = True) -> np.ndarray:
    """Pixel-area-integrated linear reflectance seen by ``cam`` (``float32``).

    ``adaptive=False`` integrates every pixel with ``ss * ss`` samples instead
    of only those whose neighbourhood shows an edge (slower, same result).
    """
    X, Y, hit = board_coordinates(cam)
    # bitmap resolution: about one bitmap pixel per picture pixel where the board is closest
    sub = (slice(4, None, 8), slice(4, None, 8))
    Xs, Ys = X[:, 1:-1][sub], Y[:, 1:-1][sub]
    on = (hit[:, 2:] & hit[:, :-2])[sub] & (Xs > 0) & (Xs < board.width_mm) & (Ys > 0) & (Ys < board.height_mm)
    mm_per_px = 0.5 * np.hypot(X[:, 2:] - X[:, :-2], Y[:, 2:] - Y[:, :-2])[sub]
    finest = float(np.min(mm_per_px[on])) if np.any(on) else board.square_mm / 64.0
    p = board.exact_px_per_square(int(np.clip(np.ceil(_BITMAP_PX_PER_IMAGE_PX * board.square_mm / max(finest, 1e-9)), 48, 320)))
    bitmap, m = _sheet_bitmap(board, p)
    s = p / board.square_mm
    bx = np.where(hit, X * s + (m - 0.5), -1e6).astype(np.float32)
    by = np.where(hit, Y * (s / board.stretch_y) + (m - 0.5), -1e6).astype(np.float32)
    base = cv2.remap(bitmap, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(background))
    if ss <= 1:
        return base
    if adaptive:
        k3 = np.ones((3, 3), np.uint8)
        edge = (cv2.dilate(base, k3) - cv2.erode(base, k3)) > 1e-4
        idx = np.flatnonzero(cv2.dilate(edge.astype(np.uint8), k3).ravel())
    else:
        idx = np.arange(base.size)
    if idx.size == 0:
        return base
    # sub-pixel samples: the exact map at the pixel centre plus its local Jacobian
    (bxu, bxv), (byu, byv) = _central_differences(bx), _central_differences(by)
    cx0, cy0 = bx.ravel()[idx], by.ravel()[idx]
    jxu, jxv, jyu, jyv = bxu.ravel()[idx], bxv.ravel()[idx], byu.ravel()[idx], byv.ravel()[idx]
    acc = np.zeros(idx.size, dtype=np.float32)
    offsets = _sample_offsets(ss).astype(np.float32)
    for ou, ov in offsets:
        acc += _remap(bitmap, cx0 + ou * jxu + ov * jxv, cy0 + ou * jyu + ov * jyv, background)
    out = base.copy()
    out.ravel()[idx] = acc / np.float32(len(offsets))
    return out


@functools.lru_cache(maxsize=1)
def _srgb_lut() -> np.ndarray:
    return srgb_encode(np.arange(65536, dtype=np.float64) / 65535.0).astype(np.float32)


def _encode(lin: np.ndarray) -> np.ndarray:
    """sRGB transfer through a 16-bit table built from the renderer's ``srgb_encode``."""
    q = np.clip(np.rint(lin * 65535.0), 0, 65535).astype(np.uint16)
    return _srgb_lut()[q]


def render_board(
    board: PrintedBoard,
    cam: CameraSpec,
    *,
    ss: int = 5,
    seed: int = 0,
    background: float = 0.30,
    ambient: float = 0.8,
    shading: tuple[float, float] = (0.0, 0.0),
    glare: tuple[float, float, float, float] | None = None,
    adaptive: bool = True,
) -> BoardPhoto:
    """Photo of ``board`` taken by ``cam`` (a camera whose world frame is the board frame).

    ``shading`` is a relative illumination gradient across the picture
    (left-to-right, top-to-bottom), e.g. ``(0.2, -0.1)``.  ``glare`` adds a
    specular highlight ``(u, v, sigma_px, strength)``: a Gaussian blob of linear
    light centred on pixel ``(u, v)`` (strength 1 clips paper and ink alike).
    The result is fully deterministic for given inputs.
    """
    refl = board_reflectance(board, cam, ss=ss, background=background, adaptive=adaptive)
    H, W = refl.shape
    lin = refl * np.float32(ambient)
    if shading[0] or shading[1]:
        u = np.linspace(-0.5, 0.5, W, dtype=np.float32)[None, :]
        v = np.linspace(-0.5, 0.5, H, dtype=np.float32)[:, None]
        lin = lin * (1.0 + np.float32(shading[0]) * u + np.float32(shading[1]) * v)

    if glare is not None:
        gu, gv, gs, strength = glare
        uu = (np.arange(W, dtype=np.float32) - np.float32(gu)) ** 2
        vv = (np.arange(H, dtype=np.float32) - np.float32(gv)) ** 2
        lin = lin + np.float32(strength) * np.exp(-(vv[:, None] + uu[None, :]) / np.float32(2.0 * gs * gs))

    exposure = float(cam.exposure)
    if cam.auto_exposure:
        exposure = cam.auto_exposure_target / max(float(np.mean(lin)), 1e-6)
    lin = lin * np.float32(exposure)
    if cam.vignette > 0.0:
        us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        r2 = ((us - cam.cx) / cam.fx) ** 2 + ((vs - cam.cy) / cam.fy) ** 2
        lin = lin * (1.0 - cam.vignette + cam.vignette / (1.0 + r2) ** 2).astype(np.float32)

    lin = np.ascontiguousarray(lin, dtype=np.float32)
    if cam.blur_sigma > 0.0:
        lin = cv2.GaussianBlur(lin, (0, 0), cam.blur_sigma, borderType=cv2.BORDER_REPLICATE)
    if cam.motion_blur_px > 1.0:
        L = int(round(cam.motion_blur_px))
        k = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.float32)
        a = np.deg2rad(cam.motion_blur_deg)
        for q in np.linspace(-L / 2.0, L / 2.0, 4 * L + 1):
            k[int(round(L + q * np.sin(a))), int(round(L + q * np.cos(a)))] += 1.0
        k /= k.sum()
        lin = cv2.filter2D(lin, -1, k, borderType=cv2.BORDER_REPLICATE)

    # neutral scene: the three channels differ only by white balance and noise
    wb = np.asarray(cam.white_balance, dtype=np.float32)
    rgb = lin[:, :, None] * wb[None, None, :]
    if cam.noise_shot > 0.0 or cam.noise_read > 0.0:
        rng = np.random.default_rng(seed)
        var = np.float32(cam.noise_shot) * np.clip(rgb, 0.0, None) + np.float32(cam.noise_read**2)
        rgb = rgb + rng.standard_normal(rgb.shape, dtype=np.float32) * np.sqrt(var)

    enc = _encode(rgb)
    if cam.tone_gamma != 1.0:
        enc = np.power(np.clip(enc, 0.0, 1.0), np.float32(cam.tone_gamma))
    if cam.tone_contrast != 0.0:
        kc = np.float32(cam.tone_contrast)
        enc = np.clip(enc + kc * (enc - 0.5) * (1.0 - np.abs(2.0 * enc - 1.0)) * 2.0 * enc * (1 - enc) * 4.0, 0.0, 1.0)
    if cam.sharpen > 0.0:
        soft = cv2.GaussianBlur(enc, (0, 0), 1.2, borderType=cv2.BORDER_REPLICATE)
        enc = np.clip(enc + np.float32(cam.sharpen) * (enc - soft), 0.0, 1.0)
    img8 = np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(img8[:, :, ::-1])
    if cam.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(cam.jpeg_quality)])
        if not ok:  # pragma: no cover - OpenCV always encodes valid input
            raise RuntimeError("JPEG encoding failed")
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    corners_px, visible = project_board_points(cam, board.corners_mm())
    outline_px, outline_ok = project_board_points(cam, board.sheet_outline_mm(per_edge=8))
    _, pattern_ok = project_board_points(cam, board.sheet_outline_mm(margin_mm=0.0, per_edge=8))
    return BoardPhoto(
        image=bgr,
        board=board,
        camera=cam,
        corner_ids=board.corner_ids(),
        corners_px=corners_px,
        corner_visible=visible,
        outline_px=outline_px,
        sheet_inside=bool(np.all(outline_ok)),
        pattern_inside=bool(np.all(pattern_ok)),
        tilt_deg=true_tilt_deg(cam),
        meta={"render": {"ss": ss, "seed": seed, "exposure": exposure, "background": background, "ambient": ambient}},
    )


# --------------------------------------------------------------------------
# View sets
# --------------------------------------------------------------------------
def inside_picture(board: PrintedBoard, cam: CameraSpec, what: Literal["sheet", "pattern"] = "sheet", margin_px: float = 4.0) -> bool:
    """Is the whole sheet (or the whole pattern) inside the picture, ``margin_px`` from its border?"""
    uv, ok = project_board_points(cam, board.sheet_outline_mm(margin_mm=None if what == "sheet" else 0.0, per_edge=6))
    m = margin_px - 0.5
    return bool(np.all(ok & (uv[:, 0] >= m) & (uv[:, 0] <= cam.width - 1 - m) & (uv[:, 1] >= m) & (uv[:, 1] <= cam.height - 1 - m)))


def view_set(
    board: PrintedBoard,
    base: CameraSpec,
    n: int = 14,
    *,
    seed: int = 0,
    tilt_deg: tuple[float, float] = (18.0, 42.0),
    fill: tuple[float, float] = (0.42, 0.62),
    rolls_deg: tuple[float, ...] = (0.0, 0.0, 90.0, 0.0, 180.0, 0.0, -90.0),
    roll_jitter_deg: float = 12.0,
    spread: float = 0.46,
    require: Literal["sheet", "pattern", "any"] = "sheet",
    same_tilt: tuple[float, float] | None = None,
) -> list[CameraSpec]:
    """``n`` deterministic camera stations for a calibration capture.

    The first station is nearly fronto-parallel; the others tilt the board by
    an angle from ``tilt_deg`` toward azimuths that go once around the circle,
    put it in different regions of the picture (a spiral reaching ``spread`` of
    the picture size from its centre) and vary roll and distance.  ``fill`` is
    the range of the board's long side as a fraction of the picture width.

    ``require`` says what must be inside the picture: the whole ``"sheet"``
    (a checkerboard needs this), the ``"pattern"``, or nothing (``"any"``: the
    board may be cut off, which lets partial ChArUco views reach the picture
    corners).  ``same_tilt=(tilt, azimuth)`` gives every station the same
    board orientation and no roll (a degenerate set, for tests).
    """
    rng = np.random.default_rng(seed)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    long_side = max(board.width_mm, board.height_mm)
    out: list[CameraSpec] = []
    for k in range(n):
        if same_tilt is not None:
            tilt, az, roll = float(same_tilt[0]), float(same_tilt[1]), 0.0
        elif k == 0:
            tilt, az = float(rng.uniform(2.0, 6.0)), float(rng.uniform(0.0, 360.0))
            roll = float(rng.uniform(-roll_jitter_deg, roll_jitter_deg))
        else:
            tilt = float(tilt_deg[0] + (tilt_deg[1] - tilt_deg[0]) * ((k * 0.618034) % 1.0))
            az = float((360.0 * (k - 1) / max(n - 1, 1) + rng.uniform(-15.0, 15.0)) % 360.0)
            roll = float(rolls_deg[k % len(rolls_deg)] + rng.uniform(-roll_jitter_deg, roll_jitter_deg))
        rad = spread * np.sqrt((k + 0.5) / n)
        ang = golden * k + 0.7
        at = (0.5 + rad * np.cos(ang), 0.5 + rad * np.sin(ang))
        dist_mm = base.fx * long_side / (float(rng.uniform(*fill)) * base.width)
        cam = board_camera(board, base, distance_mm=dist_mm, tilt_deg=tilt, azimuth_deg=az, roll_deg=roll, at=at)
        if require != "any":
            for _ in range(80):
                if inside_picture(board, cam, require):
                    break
                # pull the board toward the picture centre and step back a little
                at = (0.5 + 0.92 * (at[0] - 0.5), 0.5 + 0.92 * (at[1] - 0.5))
                dist_mm *= 1.02
                cam = board_camera(board, base, distance_mm=dist_mm, tilt_deg=tilt, azimuth_deg=az, roll_deg=roll, at=at)
            else:  # pragma: no cover - defensive
                raise ValueError("could not fit the board into the picture")
        out.append(cam)
    return out
