"""Intrinsic camera calibration from photos of a printed board (prompt 03).

Why this module exists
----------------------
The web platform gives a page no calibrated intrinsics.  The capture APIs this
application uses (``getUserMedia`` / ``MediaStreamTrack`` settings and
capabilities, ``<input type="file" capture>``) describe a stream by width,
height, frame rate, facing mode and, on some devices, zoom or torch; they carry
no focal length in pixels, no principal point and no distortion coefficients.
EXIF holds at best a nominal 35 mm-equivalent focal length.  Intrinsics
therefore have to be *measured* for every device / lens / zoom / resolution /
orientation that is used, from photos of a target of known geometry, and a
profile must only ever be applied to frames of exactly that configuration.

What is measured, and what is not
---------------------------------
* A board is described by a :class:`BoardSpec`: checkerboard or ChArUco, the
  number of squares and the **measured** square edge (a caliper or steel rule
  across many squares, divided by their number — not the size the printer was
  asked for).
* The intrinsics ``fx, fy, cx, cy`` and the distortion coefficients are
  *scale-free*: declaring a wrong square size changes none of them.  It scales
  the board poses and every millimetre figure of the planar validation.  Low
  reprojection error therefore says nothing about metric scale.
* Not scale-free is the *shape* of the board: a print that is stretched along
  one paper direction is taken for a different ``fy / fx``.  The pitch along y
  can be declared separately, and an undeclared stretch is detected from the
  photos and refused.
* Model (identical to :mod:`lumen.geometry.camera`, OpenCV order): pinhole with
  ``fx, fy, cx, cy`` and Brown–Conrady ``k1, k2, p1, p2, k3``.  ``k1, k2`` are
  always estimated; the tangential pair and ``k3`` are added only when photos
  that were *not* used for the fit are predicted better with them and the new
  coefficients are significant.  Terms that are not estimated are exactly 0.
* Uncertainties: the reported standard deviations combine the least-squares
  value, a leave-one-photo-out jackknife and — because a term fixed at zero is
  an assumption — how far each parameter moves when all five distortion terms
  are estimated (:attr:`CalibrationResult.param_sigma`).  They do not cover
  errors common to all photos (a bent board, a wrong board description).

Conventions
-----------
* Pixels: integer ``(u, v)`` is the centre of a pixel, ``u`` right, ``v`` down.
* Board frame, millimetres: origin at the top-left *outer* corner of the
  pattern, ``+x`` right, ``+y`` down as printed, ``+z`` into the paper.  Inner
  corner ``(i, j)`` (column, row) is at ``((i + 1) s, (j + 1) s, 0)`` and has
  the id ``j * (squares_x - 1) + i`` (OpenCV's ChArUco corner numbering).
  Square ``(0, 0)`` is black (on a checkerboard this is what tells the two
  ends of an asymmetric board apart).
* Angles in the results are in degrees (they are operator-facing); everything
  metric is in millimetres.

Pipeline
--------
1. :func:`detect_board` — find and index the corners (``findChessboardCornersSB``
   or ``CharucoDetector``, at reduced size for large pictures), then localise
   every corner in the full picture with ``cornerSubPix`` in a window sized
   from the local square size.  Corners whose surroundings are not evenly lit
   (glare, shadow edges) are dropped, because the refinement relies on the
   point symmetry of a chessboard corner.  Returns the points with quality
   figures (coverage, tilt, blur, partial visibility).
2. :func:`calibrate_camera` — one camera configuration, several views:
   screening, outlier views, a check that the board is as square as declared,
   distortion-model selection, uncertainties, coverage, leave-one-view-out
   error; fails closed with reasons and concrete capture advice.
3. :func:`validate_planar` — a board photo that was *not* used for the
   calibration is measured through the lens model with a homography; residuals
   are reported in pixels and in millimetres, with and without the distortion
   model.
4. :func:`calibration_summary` — one JSON-serialisable document of all of it.

Accuracy of the building blocks on synthetic photos is recorded in
``tests/unit/test_calibration.py``; nothing here has been run on real photos.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import cv2
import numpy as np

from ..errors import CalibrationError, LumenValidationError
from ..geometry.camera import CameraModel
from ..geometry.homography import apply_h, dlt_homography, h_local_jacobian, normalization_transform, normalize_h
from ..geometry.optimize import lm_solve
from .imaging import linear_luminance, to_gray

#: Order of the intrinsic parameters in every ``param_*`` list (schema order).
PARAM_NAMES: tuple[str, ...] = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")

WEB_API_NOTE = (
    "Browsers do not provide calibrated intrinsics: the capture APIs report width, height, frame rate, facing mode "
    "and sometimes zoom, but no focal length in pixels, principal point or distortion. This profile was measured "
    "from photos of a printed target and applies only to the same device, lens, zoom, resolution and orientation."
)

BOARD_KINDS = ("checkerboard", "charuco")


# --------------------------------------------------------------------------
# A. Board description
# --------------------------------------------------------------------------
def _aruco_dictionary(name: str) -> Any:
    code = getattr(cv2.aruco, name, None) if isinstance(name, str) and name.startswith("DICT_") else None
    if not isinstance(code, int):
        return None
    return cv2.aruco.getPredefinedDictionary(code)


@dataclass(frozen=True)
class BoardSpec:
    """A printed calibration target as declared by the operator.

    ``square_mm`` (and ``marker_mm``) are the **measured** edge lengths.  Use
    :meth:`from_measured_span` when the measurement was taken across several
    squares.  ``size_provenance="nominal"`` records that the size was not
    measured; the calibration is unaffected (intrinsics are scale-free) but
    every millimetre figure derived from this board is then only nominal.

    Printers often scale the two paper directions differently by a few tenths
    of a percent.  Unlike a common scale error this *does* change the
    intrinsics (it is taken for a different ``fy / fx``), so the pitch along
    the board's y can be given separately as ``square_y_mm``;
    :func:`calibrate_camera` refuses photos that show such a stretch when it
    was not declared.
    """

    kind: str
    squares_x: int
    squares_y: int
    square_mm: float
    marker_mm: float | None = None
    dictionary: str | None = None
    #: ChArUco layout of OpenCV < 4.6 for boards with an even number of rows
    legacy_pattern: bool = False
    size_provenance: str = "measured"
    #: measured pitch of the squares along board y, when it differs from ``square_mm`` (along x)
    square_y_mm: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", self.kind.strip().lower())
        problem = self._problem()
        if problem is not None:
            code, message, suggestions = problem
            raise LumenValidationError("invalid_board", message, details={"problem": code, "board": self._raw()}, suggestions=suggestions)
        object.__setattr__(self, "squares_x", int(self.squares_x))
        object.__setattr__(self, "squares_y", int(self.squares_y))
        object.__setattr__(self, "square_mm", float(self.square_mm))
        object.__setattr__(self, "square_y_mm", None if self.square_y_mm is None or float(self.square_y_mm) == float(self.square_mm) else float(self.square_y_mm))
        if self.kind == "charuco":
            object.__setattr__(self, "marker_mm", float(self.marker_mm))  # type: ignore[arg-type]
        else:
            object.__setattr__(self, "marker_mm", None)
            object.__setattr__(self, "dictionary", None)
            object.__setattr__(self, "legacy_pattern", False)

    # -------------------------------------------------------------- validation
    def _raw(self) -> dict[str, Any]:
        return {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)) for k, v in self.__dict__.items()}

    def _problem(self) -> tuple[str, str, list[str]] | None:
        def is_int(x: Any) -> bool:
            return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

        def is_num(x: Any) -> bool:
            return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and bool(np.isfinite(x))

        if self.kind not in BOARD_KINDS:
            return "kind", f"Unknown board kind {self.kind!r}; use 'checkerboard' or 'charuco'.", []
        if not is_int(self.squares_x) or not is_int(self.squares_y):
            return "squares", "squares_x and squares_y must be whole numbers of squares (not inner corners).", []
        lo = 4 if self.kind == "checkerboard" else 3
        if min(self.squares_x, self.squares_y) < lo or max(self.squares_x, self.squares_y) > 60:
            return "squares", f"A {self.kind} needs between {lo} and 60 squares along each side (count squares, not inner corners).", []
        if (self.squares_x - 1) * (self.squares_y - 1) < 12:
            return "squares", "The board has fewer than 12 inner corners; too few points per photo for a calibration.", ["Use a board with at least 5 x 4 squares."]
        if not is_num(self.square_mm) or not (2.0 <= float(self.square_mm) <= 500.0):
            return "square_mm", "square_mm must be the measured edge of one square in millimetres (2 to 500 mm).", ["Measure across several squares with a caliper or steel rule and divide by their number."]
        if self.size_provenance not in ("measured", "nominal"):
            return "size_provenance", "size_provenance must be 'measured' or 'nominal'.", []
        if self.square_y_mm is not None and (not is_num(self.square_y_mm) or abs(float(self.square_y_mm) / float(self.square_mm) - 1.0) > 0.03):
            return "square_y_mm", "square_y_mm is the measured pitch of the same squares along y; it can differ from square_mm by a print stretch of at most 3 %.", ["Give both pitches in millimetres, each measured across several squares."]
        if self.kind == "charuco":
            if not is_num(self.marker_mm) or not (0.0 < float(self.marker_mm) < float(self.square_mm)):  # type: ignore[arg-type]
                return "marker_mm", "A ChArUco board needs marker_mm (the measured edge of a marker's black square), smaller than square_mm.", []
            ratio = float(self.marker_mm) / float(self.square_mm)  # type: ignore[arg-type]
            if not (0.3 <= ratio <= 0.9):
                return "marker_mm", f"marker_mm / square_mm is {ratio:.2f}; ChArUco boards have markers of 0.3 to 0.9 of the square.", ["Check that both lengths are in millimetres and that marker_mm is the black marker square, not a single bit."]
            d = _aruco_dictionary(self.dictionary)  # type: ignore[arg-type]
            if d is None:
                return "dictionary", f"Unknown ArUco dictionary {self.dictionary!r}; give the OpenCV name, e.g. 'DICT_5X5_100'.", []
            need = (self.squares_x * self.squares_y) // 2
            have = int(d.bytesList.shape[0])
            if need > have:
                return "dictionary", f"{self.dictionary} has {have} markers but this board needs {need}.", ["Use the dictionary the board was generated with."]
        return None

    @property
    def flags(self) -> list[str]:
        """Machine-readable caveats of a valid description."""
        out: list[str] = []
        if self.kind == "checkerboard":
            if (self.squares_x + self.squares_y) % 2 == 0:
                out.append("rotation_ambiguous_180")
            if self.squares_x == self.squares_y:
                out.append("rotation_ambiguous_90")
        if self.kind == "charuco" and str(self.dictionary).upper() == "DICT_APRILTAG_36H11":
            out.append("dictionary_shared_with_scene_tags")
        if self.size_provenance == "nominal":
            out.append("square_size_nominal")
        return out

    def notes(self) -> dict[str, str]:
        """Operator-readable explanation of every flag."""
        text = {
            "rotation_ambiguous_180": (
                f"A {self.squares_x} x {self.squares_y} checkerboard looks the same after a half turn, so which corner is the origin "
                "cannot be told from a photo. The intrinsics are not affected, but board poses and corner ids are only defined up to "
                "that half turn. Use a board with an even number of squares along one side and an odd number along the other "
                "(for example 10 x 7), or a ChArUco board."
            ),
            "rotation_ambiguous_90": (
                "A checkerboard with the same number of squares along both sides cannot be told from itself turned by a quarter turn. "
                "Use different numbers of squares along the two sides, or a ChArUco board."
            ),
            "dictionary_shared_with_scene_tags": (
                "The board uses the AprilTag 36h11 dictionary, the same as the scale tags of a measuring session; never leave the "
                "board in a measurement photo."
            ),
            "square_size_nominal": (
                "The square size is the nominal print size, not a measurement. Intrinsics are unaffected; millimetre figures from this "
                "board are nominal until the print is measured."
            ),
        }
        return {f: text[f] for f in self.flags}

    # ---------------------------------------------------------------- geometry
    @property
    def inner(self) -> tuple[int, int]:
        """Number of inner corners along x and y."""
        return self.squares_x - 1, self.squares_y - 1

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def pitch_y_mm(self) -> float:
        """Pitch of the squares along board y (equal to ``square_mm`` unless declared otherwise)."""
        return self.square_mm if self.square_y_mm is None else self.square_y_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.pitch_y_mm

    @property
    def clean_fraction(self) -> float:
        """Half-size of the feature-free zone around an inner corner, in squares."""
        if self.kind == "charuco":
            return 0.5 * (1.0 - float(self.marker_mm) / self.square_mm)  # type: ignore[arg-type]
        return 0.5

    def corner_xy(self, ids: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        """Board-frame positions ``(N, 2)`` in mm of the inner corners ``ids`` (all by default)."""
        nx = self.squares_x - 1
        k = np.arange(self.n_corners) if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
        if k.size and (k.min() < 0 or k.max() >= self.n_corners):
            raise LumenValidationError("invalid_corner_id", f"Corner ids must be in 0..{self.n_corners - 1} for this board.")
        return np.stack([(k % nx + 1.0) * self.square_mm, (k // nx + 1.0) * self.pitch_y_mm], axis=1)

    def object_points(self, ids: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        """Board-frame positions ``(N, 3)`` in mm, ``z = 0``."""
        xy = self.corner_xy(ids)
        return np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)

    def outline_xy(self) -> np.ndarray:
        """The four outer corners of the pattern (TL, TR, BR, BL), mm."""
        w, h = self.width_mm, self.height_mm
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])

    def cv_board(self) -> Any:
        """The matching ``cv2.aruco.CharucoBoard`` (ChArUco only)."""
        if self.kind != "charuco":
            raise LumenValidationError("invalid_board", "Only a ChArUco board has an OpenCV board object.")
        b = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), float(self.square_mm), float(self.marker_mm), _aruco_dictionary(self.dictionary))  # type: ignore[arg-type]
        b.setLegacyPattern(bool(self.legacy_pattern))
        return b

    # ------------------------------------------------------------------- (de)serialisation
    def with_square_mm(self, square_mm: float, provenance: str | None = None) -> "BoardSpec":
        """Same layout with another declared square size (marker scaled alike)."""
        k = float(square_mm) / self.square_mm
        return BoardSpec(
            self.kind, self.squares_x, self.squares_y, float(square_mm), None if self.marker_mm is None else self.marker_mm * k, self.dictionary, self.legacy_pattern,
            provenance or self.size_provenance, None if self.square_y_mm is None else self.square_y_mm * k,
        )

    @staticmethod
    def from_measured_span(
        kind: str,
        squares_x: int,
        squares_y: int,
        span_mm: float,
        span_squares: int,
        *,
        span_y_mm: float | None = None,
        span_y_squares: int | None = None,
        marker_ratio: float | None = None,
        dictionary: str | None = None,
        legacy_pattern: bool = False,
    ) -> "BoardSpec":
        """Board whose square size is a measured length across ``span_squares`` squares along x.

        Give ``span_y_mm`` across ``span_y_squares`` squares as well to record the
        print's pitch along y.  ``marker_ratio`` is ``marker edge / square edge``
        of the design (a ratio survives any print scaling along x).
        """

        def pitch(length: float | None, n: int | None) -> float:
            if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool) and n >= 1 and length is not None and np.isfinite(length) and length > 0):
                raise LumenValidationError("invalid_board", "A measured span needs a positive length in millimetres and a whole number of squares.", details={"problem": "span"})
            return float(length) / int(n)

        sq = pitch(span_mm, span_squares)
        sq_y = None if span_y_mm is None and span_y_squares is None else pitch(span_y_mm, span_y_squares)
        return BoardSpec(kind, squares_x, squares_y, sq, None if marker_ratio is None else sq * float(marker_ratio), dictionary, legacy_pattern, "measured", sq_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_mm": self.square_mm,
            "square_y_mm": self.square_y_mm,
            "marker_mm": self.marker_mm,
            "dictionary": self.dictionary,
            "legacy_pattern": self.legacy_pattern,
            "size_provenance": self.size_provenance,
            "inner_corners": list(self.inner),
            "flags": self.flags,
            "notes": self.notes(),
            "frame": "origin at the top-left outer pattern corner, +x right, +y down, +z into the paper; mm",
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "BoardSpec":
        try:
            return BoardSpec(
                kind=d["kind"],
                squares_x=d["squares_x"],
                squares_y=d["squares_y"],
                square_mm=d["square_mm"],
                marker_mm=d.get("marker_mm"),
                dictionary=d.get("dictionary"),
                legacy_pattern=bool(d.get("legacy_pattern", False)),
                size_provenance=d.get("size_provenance", "measured"),
                square_y_mm=d.get("square_y_mm"),
            )
        except KeyError as exc:
            raise LumenValidationError("invalid_board", f"Board description lacks {exc.args[0]!r}.") from None


@dataclass(frozen=True)
class CaptureConfig:
    """What the client reported about how a photo was taken.

    None of this is trusted for geometry; it is used to keep photos of
    different camera configurations apart and is stored with the profile.
    """

    device: str = ""
    lens: str = ""
    zoom: float | None = None
    capture_mode: str = ""

    def key(self) -> tuple[str, str, float | None, str]:
        return (self.device.strip().lower(), self.lens.strip().lower(), None if self.zoom is None else round(float(self.zoom), 3), self.capture_mode.strip().lower())

    def to_dict(self) -> dict[str, Any]:
        return {"device": self.device, "lens": self.lens, "zoom": self.zoom, "capture_mode": self.capture_mode}


# --------------------------------------------------------------------------
# Small geometric helpers
# --------------------------------------------------------------------------
def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography ``dst ~ H @ src`` minimising the squared distance in ``dst``.

    Normalised DLT (:func:`lumen.geometry.homography.dlt_homography`) followed
    by Levenberg–Marquardt on the geometric error, in normalised coordinates.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    H0 = dlt_homography(src, dst)
    if src.shape[0] == 4:
        return H0
    Ts, Td = normalization_transform(src), normalization_transform(dst)
    s, d = apply_h(Ts, src), apply_h(Td, dst)
    Hn = Td @ H0 @ np.linalg.inv(Ts)
    if abs(Hn[2, 2]) < 1e-12 * np.linalg.norm(Hn):
        return H0
    Hn = Hn / Hn[2, 2]
    x, y, n = s[:, 0], s[:, 1], s.shape[0]

    def fn(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = h[6] * x + h[7] * y + 1.0
        u = (h[0] * x + h[1] * y + h[2]) / w
        v = (h[3] * x + h[4] * y + h[5]) / w
        J = np.zeros((2 * n, 8))
        J[:n, 0], J[:n, 1], J[:n, 2] = x / w, y / w, 1.0 / w
        J[:n, 6], J[:n, 7] = -u * x / w, -u * y / w
        J[n:, 3], J[n:, 4], J[n:, 5] = x / w, y / w, 1.0 / w
        J[n:, 6], J[n:, 7] = -v * x / w, -v * y / w
        return np.concatenate([u - d[:, 0], v - d[:, 1]]), J

    res = lm_solve(fn, Hn.ravel()[:8], max_iter=40)
    H = np.linalg.inv(Td) @ np.append(res.x, 1.0).reshape(3, 3) @ Ts
    return normalize_h(H)


def _vanishing(H: np.ndarray, width: int, height: int) -> dict[str, Any]:
    """What one board->image homography says about the board's tilt, before any calibration.

    * ``distance``: pixels from the image centre to the vanishing line of the
      board plane (``inf`` when the board is parallel to the image plane).  With
      the principal point at the image centre it equals ``f / tan(tilt)``;
    * ``direction_deg``: the direction in the picture (u right, v down) in which
      the board recedes;
    * ``a, b``: Zhang's two constraints on the focal length, ``a / f^2 + b = 0``
      (square pixels, principal point at the image centre), scaled alike.
    """
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    Hc = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]) @ np.asarray(H, dtype=float)
    out: dict[str, Any] = {"distance": float("inf"), "direction_deg": None, "a": np.zeros(2), "b": np.zeros(2)}
    scale = float(np.sqrt(0.5 * (Hc[0, 0] ** 2 + Hc[1, 0] ** 2 + Hc[0, 1] ** 2 + Hc[1, 1] ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        return out
    h1, h2 = Hc[:, 0] / scale, Hc[:, 1] / scale
    out["a"] = np.array([h1[0] * h2[0] + h1[1] * h2[1], (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)])
    out["b"] = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    try:
        line = np.linalg.inv(Hc)[2]  # vanishing line of the plane, image coordinates relative to the centre
    except np.linalg.LinAlgError:
        return out
    g = float(np.hypot(line[0], line[1]))
    if np.isfinite(g) and g > 1e-15 * max(abs(line[2]), 1e-300) and abs(line[2]) > 0:
        out["distance"] = abs(float(line[2])) / g
        toward = -np.sign(line[2]) * line[:2] / g
        out["direction_deg"] = float(np.degrees(np.arctan2(toward[1], toward[0])))
    return out


def _focal_from_perspective(geo: Sequence[dict[str, Any]], width: int, height: int) -> float | None:
    """Focal length (px) that the perspective of one or several views agrees on, or ``None``.

    Every view gives two equations ``a / f^2 + b = 0``.  Real perspective makes
    ``a`` and ``b`` proportional with a negative factor; in views without
    perspective both are noise.  A value is returned only when the two are
    strongly anti-correlated (``rho < -0.9``) and the result is plausible.
    """
    a = np.concatenate([g["a"] for g in geo]) if geo else np.zeros(0)
    b = np.concatenate([g["b"] for g in geo]) if geo else np.zeros(0)
    saa, sbb, sab = float(a @ a), float(b @ b), float(a @ b)
    if saa <= 0 or sbb <= 0 or sab / np.sqrt(saa * sbb) > -0.9:
        return None
    f = float((saa / sbb) ** 0.25)
    return f if 0.2 * max(width, height) < f < 8.0 * max(width, height) else None


def _tilt_from_vanishing(geo: dict[str, Any], f: float) -> float:
    return float(np.degrees(np.arctan(f / geo["distance"]))) if np.isfinite(geo["distance"]) and geo["distance"] > 0 else 0.0


def _side_name(direction_deg: float | None) -> str | None:
    """Which side of the picture a direction (u right, v down) points to."""
    if direction_deg is None:
        return None
    a = (direction_deg + 360.0) % 360.0
    return ("right", "bottom", "left", "top")[int(((a + 45.0) % 360.0) // 90.0)]


def _hull_fraction(pts: np.ndarray, width: int, height: int) -> float:
    if pts.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull) / float(width * height))


# --------------------------------------------------------------------------
# B. Detection in one photo
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectConfig:
    #: the board is *found* at most at this long-edge size; corners are always
    #: localised in the full-resolution picture
    chessboard_detect_long_edge: int = 1024
    charuco_detect_long_edge: int = 2000
    #: a ChArUco view needs this many corners (or all of a smaller board)
    min_charuco_corners: int = 12
    #: a checkerboard is found as a whole; of its corners at least this fraction
    #: must survive the per-corner checks (window inside the picture, even lighting)
    min_checkerboard_fraction: float = 0.5
    #: squares smaller than this in the picture are not measured (px)
    min_square_px: float = 14.0
    #: cornerSubPix half-window as a fraction of the local square size
    refine_window_frac: float = 0.25
    #: ... but never beyond this fraction of the marker-free zone of a ChArUco corner
    refine_clean_frac: float = 0.8
    refine_min_half: int = 3
    refine_max_half: int = 32
    #: a corner that moves further than this during refinement is dropped
    #: (pixels, plus 10 % of the local square size)
    max_refine_shift_px: float = 3.0
    #: a corner is dropped when the brightness of the two squares of the same
    #: colour that meet in it differs, across the refinement window, by more than
    #: this fraction of the black/white step (glare, a shadow edge, something on
    #: a square).  Measured on synthetic photos: clean ones stay below 0.015
    #: (0.03 with heavy noise); in photos with a highlight, corners between 0.04
    #: and 0.06 were 0.1 px off (RMS), above 0.1 they were 0.2 px off and more.
    max_corner_asymmetry: float = 0.04
    #: edge spread (Gaussian sigma, px) above which a photo is called blurred / refused
    blur_warn_px: float = 2.0
    blur_reject_px: float = 3.5
    #: relative difference between the declared marker size and the one seen in the
    #: picture (through the squares) above which a ChArUco description is refused
    marker_size_tolerance: float = 0.08
    #: field of view assumed only for the pre-calibration tilt estimate
    nominal_hfov_deg: float = 65.0


@dataclass
class BoardQuality:
    n_expected: int = 0
    n_detected: int = 0
    #: every inner corner of the board was measured
    complete: bool = False
    #: the printed pattern reaches beyond the picture
    partly_outside: bool = False
    #: expected corners that should be inside the picture but were not measured
    missing_inside: int = 0
    #: corners found but discarded (window off the picture, refinement did not settle)
    n_dropped: int = 0
    #: fraction of the image area inside the convex hull of the points
    coverage: float = 0.0
    #: bounding box of the points as fractions of the picture (u0, v0, u1, v1)
    bbox: tuple[float, float, float, float] | None = None
    tilt_deg: float | None = None
    tilt_source: str = ""  # "estimated_nominal_focal" | "estimated_homography_focal" | "calibrated"
    far_side: str | None = None
    far_direction_deg: float | None = None
    min_square_px: float | None = None
    #: Gaussian-equivalent edge spread in pixels (larger = more blurred)
    blur_px: float | None = None
    #: ratio of the edge spread across the two edge families (> 1.6 suggests motion blur)
    blur_anisotropy: float | None = None
    #: the profiles were too short to see the full edge: ``blur_px`` is a lower bound
    blur_saturated: bool = False
    #: Michelson contrast of the squares in linear light; are the white squares clipped?
    contrast: float | None = None
    white_clipped: bool = False
    border_distance_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        def r(x: float | None, n: int = 3) -> float | None:
            return None if x is None or not np.isfinite(x) else round(float(x), n)

        return {
            "n_expected": self.n_expected,
            "n_detected": self.n_detected,
            "complete": self.complete,
            "partly_outside": self.partly_outside,
            "missing_inside": self.missing_inside,
            "n_dropped": self.n_dropped,
            "coverage": r(self.coverage, 4),
            "bbox": None if self.bbox is None else [round(float(x), 4) for x in self.bbox],
            "tilt_deg": r(self.tilt_deg, 2),
            "tilt_source": self.tilt_source,
            "far_side": self.far_side,
            "far_direction_deg": r(self.far_direction_deg, 1),
            "min_square_px": r(self.min_square_px, 1),
            "blur_px": r(self.blur_px, 3),
            "blur_anisotropy": r(self.blur_anisotropy, 2),
            "blur_saturated": self.blur_saturated,
            "contrast": r(self.contrast, 3),
            "white_clipped": self.white_clipped,
            "border_distance_px": r(self.border_distance_px, 1),
        }


@dataclass
class BoardObservation:
    """The measured corners of one board in one photo."""

    view_id: str
    board: BoardSpec
    image_size: tuple[int, int]  # (width, height)
    ok: bool
    ids: np.ndarray  # (N,) corner ids
    image_points: np.ndarray  # (N, 2) raw pixels (as recorded by the lens)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    quality: BoardQuality = field(default_factory=BoardQuality)
    capture: CaptureConfig | None = None
    details: dict[str, Any] = field(default_factory=dict)
    #: board mm -> raw pixels, fitted to the measured corners (lens distortion ignored)
    homography: np.ndarray | None = None
    #: hash of the analysed picture ("" for observations made from stored points)
    image_id: str = ""

    @property
    def object_points(self) -> np.ndarray:
        """Board-frame positions ``(N, 3)`` of the measured corners, mm, ``z = 0``."""
        return self.board.object_points(self.ids)

    @property
    def n_points(self) -> int:
        return int(self.ids.size)

    @property
    def fingerprint(self) -> str:
        """Identity of this measurement (used to tell held-out photos from used ones).

        Built from the points at the 0.0001 px resolution at which :meth:`to_dict`
        stores them, so it survives a round trip through JSON.
        """
        h = hashlib.sha1()
        h.update(repr((self.board.kind, self.board.squares_x, self.board.squares_y, tuple(int(x) for x in self.image_size))).encode())
        h.update(np.ascontiguousarray(self.ids, dtype=np.int64).tobytes())
        h.update(np.rint(np.asarray(self.image_points, dtype=np.float64) * 1e4).astype(np.int64).tobytes())
        return h.hexdigest()[:16]

    def to_dict(self, points: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "view_id": self.view_id,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "suggestions": list(self.suggestions),
            "image_size": list(self.image_size),
            "n_points": self.n_points,
            "quality": self.quality.to_dict(),
            "capture": None if self.capture is None else self.capture.to_dict(),
            "fingerprint": self.fingerprint,
            "image_id": self.image_id,
            "details": self.details,
        }
        if points:
            d["ids"] = self.ids.astype(int).tolist()
            d["image_points"] = (np.rint(np.asarray(self.image_points, dtype=np.float64) * 1e4) / 1e4).tolist()
            d["object_points"] = np.round(self.object_points, 6).tolist()
        return d

    @staticmethod
    def from_dict(board: BoardSpec, d: dict[str, Any]) -> "BoardObservation":
        """Rebuild an observation stored with :meth:`to_dict` (points included).

        The stored verdict (``ok``, ``reasons``, quality figures) is taken as it
        is: keep stored observations on the server side.
        """
        try:
            size = (int(d["image_size"][0]), int(d["image_size"][1]))
            ids = np.asarray(d["ids"], dtype=np.int64).reshape(-1)
            pts = np.asarray(d["image_points"], dtype=np.float64).reshape(-1, 2)
        except (KeyError, TypeError, ValueError, IndexError):
            raise LumenValidationError("invalid_observation", "A stored observation needs image_size, ids and image_points.") from None
        if ids.size != pts.shape[0] or not np.all(np.isfinite(pts)) or np.unique(ids).size != ids.size or min(size) < 16:
            raise LumenValidationError("invalid_observation", "Stored ids and image_points do not match.")
        board.corner_xy(ids)  # validates the ids against this board
        q = d.get("quality") or {}
        quality = BoardQuality(**{k: (tuple(v) if k == "bbox" and v is not None else v) for k, v in q.items() if k in BoardQuality.__dataclass_fields__})
        cap = d.get("capture")
        capture = None if not cap else CaptureConfig(str(cap.get("device") or ""), str(cap.get("lens") or ""), cap.get("zoom"), str(cap.get("capture_mode") or ""))
        obs = BoardObservation(
            str(d.get("view_id", "")), board, size, bool(d.get("ok", False)), ids, pts,
            [str(x) for x in d.get("reasons", [])], [str(x) for x in d.get("warnings", [])], [str(x) for x in d.get("suggestions", [])],
            quality, capture, dict(d.get("details") or {}), image_id=str(d.get("image_id", "")),
        )
        nx = board.squares_x - 1
        if ids.size >= 4 and np.unique(ids % nx).size >= 2 and np.unique(ids // nx).size >= 2:
            obs.homography = fit_homography(board.corner_xy(ids), pts)
        return obs


def _reduced(gray: np.ndarray, long_edge: int) -> tuple[np.ndarray, float, float]:
    H, W = gray.shape[:2]
    if max(H, W) <= long_edge:
        return gray, 1.0, 1.0
    s = long_edge / float(max(H, W))
    small = cv2.resize(gray, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    return small, small.shape[1] / float(W), small.shape[0] / float(H)


def _to_full(c: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Pixel centres of a reduced image -> full-resolution pixel coordinates."""
    q = np.asarray(c, dtype=np.float64).reshape(-1, 2).copy()
    q[:, 0] = (q[:, 0] + 0.5) / sx - 0.5
    q[:, 1] = (q[:, 1] + 0.5) / sy - 0.5
    return q


def _find_chessboard(gray: np.ndarray, nx: int, ny: int, cfg: DetectConfig) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Coarse inner corners ``(ny * nx, 2)`` in full-resolution pixels, row-major."""
    small, sx, sy = _reduced(gray, cfg.chessboard_detect_long_edge)
    info: dict[str, Any] = {"detect_scale": round(sx, 4)}
    attempts: list[tuple[np.ndarray, float, float, int]] = [(small, sx, sy, 0), (small, sx, sy, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)]
    if sx < 1.0:
        attempts.append((gray, 1.0, 1.0, cv2.CALIB_CB_EXHAUSTIVE))
    for k, (img, ax, ay, flags) in enumerate(attempts):
        ok, c = cv2.findChessboardCornersSB(img, (nx, ny), flags=flags)
        if ok and c is not None and c.shape[0] == nx * ny:
            info["attempt"] = k
            info["detect_scale"] = round(ax, 4)
            return _to_full(c, ax, ay), info
    # Nothing of the declared size: is a smaller grid visible (board cut off or covered)?
    try:
        ok, c, meta = cv2.findChessboardCornersSBWithMeta(small, (3, 3), cv2.CALIB_CB_LARGER)
    except cv2.error:  # pragma: no cover - defensive
        ok, c, meta = False, None, None
    if ok and meta is not None:
        info["partial_grid"] = [int(meta.shape[1]), int(meta.shape[0])]
    return None, info


def _canonical_chessboard(gray: np.ndarray, pts: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Put detected corners into the board's own numbering.

    Two things are fixed: a mirrored numbering (the board is seen from the
    front, so ``x`` cross ``y`` must point into the picture), and — when the
    board is not symmetric under a half turn — which end is the origin, from
    the colour of the squares (square ``(0, 0)`` is black, hence the square
    between the first four inner corners is black as well).
    """
    g = pts.reshape(ny, nx, 2)
    info: dict[str, Any] = {"mirrored_numbering": False, "half_turn_applied": False, "origin_square": "unknown"}
    ex = np.mean(g[:, 1:] - g[:, :-1], axis=(0, 1))
    ey = np.mean(g[1:] - g[:-1], axis=(0, 1))
    if ex[0] * ey[1] - ex[1] * ey[0] < 0:
        g = g[:, ::-1]
        info["mirrored_numbering"] = True
    # mean code value at the centre of every inner cell
    cen = 0.25 * (g[:-1, :-1] + g[:-1, 1:] + g[1:, :-1] + g[1:, 1:])
    side = np.minimum(np.linalg.norm(g[:-1, 1:] - g[:-1, :-1], axis=2), np.linalg.norm(g[1:, :-1] - g[:-1, :-1], axis=2))
    H, W = gray.shape[:2]
    val = np.zeros(cen.shape[:2])
    for j in range(cen.shape[0]):
        for i in range(cen.shape[1]):
            r = int(max(1, 0.15 * side[j, i]))
            u, v = int(round(cen[j, i, 0])), int(round(cen[j, i, 1]))
            val[j, i] = float(np.mean(gray[max(v - r, 0) : min(v + r + 1, H), max(u - r, 0) : min(u + r + 1, W)]))
    jj, ii = np.meshgrid(np.arange(cen.shape[0]), np.arange(cen.shape[1]), indexing="ij")
    even = (ii + jj) % 2 == 0
    if even.any() and (~even).any():
        diff = float(val[even].mean() - val[~even].mean())  # negative: even cells are the dark ones
        info["origin_square"] = "black" if diff < 0 else "white"
        if diff > 0 and (nx + ny) % 2 == 1:
            g = g[::-1, ::-1]
            info["half_turn_applied"] = True
            info["origin_square"] = "black"
    return np.ascontiguousarray(g).reshape(-1, 2), info


def _refine_corners(gray: np.ndarray, pts: np.ndarray, half: np.ndarray) -> np.ndarray:
    """``cornerSubPix`` with one window half-size per corner."""
    out = pts.astype(np.float32).copy()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    for h in np.unique(half):
        m = half == h
        sel = np.ascontiguousarray(out[m].reshape(-1, 1, 2))
        out[m] = cv2.cornerSubPix(gray, sel, (int(h), int(h)), (-1, -1), crit).reshape(-1, 2)
    return out.astype(np.float64)


def _local_square_px(H: np.ndarray, xy: np.ndarray, square_mm: float) -> np.ndarray:
    """Shortest image extent of one square at each board point (px)."""
    J = h_local_jacobian(H, xy)
    return np.linalg.svd(J, compute_uv=False)[:, -1] * square_mm


def _corner_asymmetry(lin: np.ndarray, pts: np.ndarray, ex: np.ndarray, ey: np.ndarray, clean: float, half: np.ndarray) -> np.ndarray:
    """Illumination asymmetry across each corner's refinement window, as a fraction of the black/white step.

    A chessboard corner is point symmetric: the two squares of one colour that
    meet in it are equally bright, whatever the blur, tone curve or (uniform)
    lighting.  ``cornerSubPix`` relies on that symmetry.  A highlight, a shadow
    edge or something lying on a square breaks it and moves the result.

    The mean linear intensity of the four squares is taken ``0.6 clean`` squares
    out along both diagonals (patches that grow with the square, placed to
    sub-pixel), the difference between opposite squares is divided by the
    black/white step and scaled from the distance between the two patches to
    the width ``2 half`` of the refinement window.  ``ex, ey`` are the image
    vectors of one square along board x and y; ``clean`` is the half-size of the
    feature-free zone around a corner in squares.  Corners whose patches leave
    the picture cannot be judged and get 0.
    """
    n = pts.shape[0]
    sq = np.minimum(np.linalg.norm(ex, axis=1), np.linalg.norm(ey, axis=1))
    r = np.clip(np.floor(0.3 * clean * sq), 2, 12).astype(int)
    d = 0.6 * clean
    diag = (d * (ex + ey), d * (ex - ey))
    centres = np.stack([pts + diag[0], pts - diag[0], pts + diag[1], pts - diag[1]], axis=1)  # (n, 4, 2)
    H, W = lin.shape[:2]
    pad = int(r.max()) + 2
    u0, v0 = max(int(np.floor(centres[..., 0].min())) - pad, 0), max(int(np.floor(centres[..., 1].min())) - pad, 0)
    u1, v1 = min(int(np.ceil(centres[..., 0].max())) + pad + 1, W), min(int(np.ceil(centres[..., 1].max())) + pad + 1, H)
    if u1 - u0 < 3 or v1 - v0 < 3:
        return np.zeros(n)
    crop = np.ascontiguousarray(lin[v0:v1, u0:u1])
    m = np.zeros((n, 4), dtype=np.float32)
    for k in np.unique(r):
        sel = r == k
        box = cv2.blur(crop, (2 * int(k) + 1, 2 * int(k) + 1), borderType=cv2.BORDER_REPLICATE)
        m[sel] = cv2.remap(box, (centres[sel][..., 0] - u0).astype(np.float32), (centres[sel][..., 1] - v0).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inside = np.all((centres[..., 0] - r[:, None] >= 0) & (centres[..., 1] - r[:, None] >= 0) & (centres[..., 0] + r[:, None] <= W - 1) & (centres[..., 1] + r[:, None] <= H - 1), axis=1)
    step = np.abs(0.5 * (m[:, 0] + m[:, 1]) - 0.5 * (m[:, 2] + m[:, 3]))
    per_px = np.maximum(np.abs(m[:, 0] - m[:, 1]) / (2.0 * np.linalg.norm(diag[0], axis=1)), np.abs(m[:, 2] - m[:, 3]) / (2.0 * np.linalg.norm(diag[1], axis=1)))
    return np.where(inside, per_px * 2.0 * half / np.maximum(step, 1e-6), 0.0)


def _edge_blur(lin: np.ndarray, board: BoardSpec, ids: np.ndarray, pts: np.ndarray, square_px: np.ndarray) -> dict[str, Any]:
    """Edge spread across the square boundaries between neighbouring corners.

    Profiles are taken in linear light, perpendicular to the edge, around its
    middle (away from the corners and, on a ChArUco board, inside the
    marker-free zone).  The 10–90 % rise of a Gaussian edge is ``2.563 sigma``.
    """
    nx = board.squares_x - 1
    index = {int(k): n for n, k in enumerate(ids)}
    pairs: list[tuple[int, int, int]] = []
    for k in index:
        if (k % nx) + 1 < nx and k + 1 in index:
            pairs.append((index[k], index[k + 1], 0))
        if k + nx in index:
            pairs.append((index[k], index[k + nx], 1))
    if not pairs:
        return {}
    a = pts[[p[0] for p in pairs]]
    b = pts[[p[1] for p in pairs]]
    fam = np.array([p[2] for p in pairs])
    sq = np.minimum(square_px[[p[0] for p in pairs]], square_px[[p[1] for p in pairs]])
    L = np.clip(min(0.45, 0.8 * board.clean_fraction) * sq, 0.0, 16.0)
    keep = L >= 3.0
    if not np.any(keep):
        return {}
    a, b, fam, L = a[keep], b[keep], fam[keep], L[keep]
    mid = 0.5 * (a + b)
    t = b - a
    length = np.linalg.norm(t, axis=1)
    t = t / length[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
    frac = np.linspace(-1.0, 1.0, 49)
    along = np.array([-0.15, -0.075, 0.0, 0.075, 0.15])
    # sample points: (edge, along, across, 2)
    P = mid[:, None, None, :] + (along[None, :, None, None] * length[:, None, None, None]) * t[:, None, None, :] + (frac[None, None, :, None] * L[:, None, None, None]) * nrm[:, None, None, :]
    mx = P[..., 0].reshape(len(L), -1).astype(np.float32)
    my = P[..., 1].reshape(len(L), -1).astype(np.float32)
    prof = cv2.remap(lin, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(L), along.size, frac.size).mean(axis=1)
    n_end = 6
    flip = prof[:, -n_end:].mean(axis=1) < prof[:, :n_end].mean(axis=1)
    prof = np.where(flip[:, None], prof[:, ::-1], prof)
    lo = np.median(prof[:, :n_end], axis=1)
    hi = np.median(prof[:, -n_end:], axis=1)
    amp = hi - lo
    good = amp > 0.03
    if not np.any(good):
        return {}
    q = (prof - lo[:, None]) / np.where(good, amp, 1.0)[:, None]

    def crossing(level: float, first: bool) -> np.ndarray:
        above = q >= level
        if first:  # first sample at or above the level
            k = np.argmax(above, axis=1)
        else:  # last sample below the level, then step to the next
            k = q.shape[1] - np.argmax(~above[:, ::-1], axis=1)
        k = np.clip(k, 1, q.shape[1] - 1)
        r = np.arange(q.shape[0])
        q0, q1 = q[r, k - 1], q[r, k]
        w = np.clip((level - q0) / np.where(np.abs(q1 - q0) > 1e-9, q1 - q0, 1.0), 0.0, 1.0)
        return frac[k - 1] + w * (frac[k] - frac[k - 1])

    width = (crossing(0.9, False) - crossing(0.1, True)) * L  # px
    good &= width > 0
    sigma = width / 2.563
    saturated = width > 0.8 * 2.0 * L
    med = [float(np.median(sigma[good & (fam == f)])) if np.any(good & (fam == f)) else np.nan for f in (0, 1)]
    vals = [m for m in med if np.isfinite(m)]
    if not vals:
        return {}
    out: dict[str, Any] = {
        "blur_px": float(max(vals)),
        "blur_anisotropy": float(max(vals) / max(min(vals), 1e-6)) if len(vals) == 2 else None,
        "blur_saturated": bool(np.mean(saturated[good]) > 0.3),
        "contrast": float(np.median((hi[good] - lo[good]) / np.clip(hi[good] + lo[good], 1e-6, None))),
        "white_level": float(np.median(hi[good])),
        "n_edges": int(good.sum()),
    }
    return out


def _capture_hint(board: BoardSpec) -> str:
    if board.kind == "checkerboard":
        return "Keep the whole checkerboard, with its white margin, inside the picture."
    return "Keep at least a third of the board inside the picture, in focus and evenly lit."


def observation_from_points(
    board: BoardSpec,
    image_size: tuple[int, int],
    ids: np.ndarray | Sequence[int],
    image_points: np.ndarray,
    *,
    view_id: str = "",
    capture: CaptureConfig | None = None,
    cfg: DetectConfig | None = None,
) -> BoardObservation:
    """Observation from already measured corners (stored points, simulations).

    Applies the same geometric checks as :func:`detect_board`; the figures
    that need the picture itself (blur, contrast) stay empty.
    """
    cfg = cfg or DetectConfig()
    width, height = int(image_size[0]), int(image_size[1])
    ids_a = np.asarray(ids, dtype=np.int64).reshape(-1)
    pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if width < 16 or height < 16:
        raise LumenValidationError("invalid_image_size", "image_size must be (width, height) in pixels.")
    if ids_a.size != pts.shape[0] or not np.all(np.isfinite(pts)) or np.unique(ids_a).size != ids_a.size:
        raise LumenValidationError("invalid_points", "ids and image_points must have the same length, finite coordinates and no repeated id.")
    board.corner_xy(ids_a)  # validates the ids
    order = np.argsort(ids_a)
    return _finish_observation(board, (width, height), ids_a[order], pts[order], None, cfg, view_id, capture, {}, 0)


def _finish_observation(
    board: BoardSpec,
    size: tuple[int, int],
    ids: np.ndarray,
    pts: np.ndarray,
    lin: np.ndarray | None,
    cfg: DetectConfig,
    view_id: str,
    capture: CaptureConfig | None,
    details: dict[str, Any],
    n_dropped: int,
) -> BoardObservation:
    width, height = size
    q = BoardQuality(n_expected=board.n_corners, n_detected=int(ids.size), n_dropped=int(n_dropped))
    obs = BoardObservation(view_id, board, size, False, ids, pts, quality=q, capture=capture, details=details)
    nx = board.squares_x - 1
    need = min(cfg.min_charuco_corners, board.n_corners) if board.kind == "charuco" else int(np.ceil(cfg.min_checkerboard_fraction * board.n_corners))
    if ids.size < max(need, 6):
        uneven = int(details.get("dropped_uneven_lighting", 0))
        if uneven >= max(3, n_dropped // 2):
            obs.reasons.append("glare_or_uneven_lighting")
            obs.suggestions.append(f"{uneven} corners lie in a highlight, a shadow edge or under something covering the board. Light the board evenly, avoid reflections and keep fingers off the pattern.")
        else:
            obs.reasons.append("too_few_corners")
            obs.suggestions.append(f"Only {ids.size} of {board.n_corners} corners could be measured. " + _capture_hint(board))
        return obs
    if np.unique(ids % nx).size < 2 or np.unique(ids // nx).size < 2:
        obs.reasons.append("corners_collinear")
        obs.suggestions.append("The measured corners lie on one line of the board; show more of the board.")
        return obs
    xy = board.corner_xy(ids)
    H = fit_homography(xy, pts)
    square_px = _local_square_px(H, xy, board.square_mm)
    q.min_square_px = float(square_px.min())
    q.coverage = _hull_fraction(pts, width, height)
    q.bbox = (float((pts[:, 0].min() + 0.5) / width), float((pts[:, 1].min() + 0.5) / height), float((pts[:, 0].max() + 0.5) / width), float((pts[:, 1].max() + 0.5) / height))
    q.border_distance_px = float(min(pts[:, 0].min() + 0.5, pts[:, 1].min() + 0.5, width - 0.5 - pts[:, 0].max(), height - 0.5 - pts[:, 1].max()))
    q.complete = bool(ids.size == board.n_corners)
    # where the whole pattern is, according to the homography of the measured part
    outline = apply_h(H, board.outline_xy())
    inside = (outline[:, 0] >= -0.5) & (outline[:, 0] <= width - 0.5) & (outline[:, 1] >= -0.5) & (outline[:, 1] <= height - 0.5)
    q.partly_outside = bool(not np.all(inside))
    if not q.complete:
        missing = np.setdiff1d(np.arange(board.n_corners), ids)
        pm = apply_h(H, board.corner_xy(missing))
        m = 0.25 * float(np.median(square_px))
        q.missing_inside = int(np.sum((pm[:, 0] > m) & (pm[:, 0] < width - 1 - m) & (pm[:, 1] > m) & (pm[:, 1] < height - 1 - m)))
    # tilt estimated from this photo alone: focal length from its own perspective when
    # there is enough of it, otherwise from a nominal field of view
    obs.homography = H
    geo = _vanishing(H, width, height)
    f_nom = 0.5 * width / np.tan(np.deg2rad(cfg.nominal_hfov_deg) * 0.5)
    f_own = _focal_from_perspective([geo], width, height) if f_nom / geo["distance"] > np.tan(np.deg2rad(10.0)) else None
    q.tilt_deg = _tilt_from_vanishing(geo, f_own or f_nom)
    q.tilt_source = "estimated_homography_focal" if f_own else "estimated_nominal_focal"
    q.far_direction_deg = geo["direction_deg"] if q.tilt_deg >= 3.0 else None
    q.far_side = _side_name(q.far_direction_deg)
    if lin is not None:
        b = _edge_blur(lin, board, ids, pts, square_px)
        if b:
            q.blur_px, q.blur_anisotropy, q.blur_saturated, q.contrast = b["blur_px"], b["blur_anisotropy"], b["blur_saturated"], b["contrast"]
            q.white_clipped = bool(b["white_level"] >= 0.98)
    if q.min_square_px < cfg.min_square_px:
        obs.reasons.append("board_too_small_in_picture")
        obs.suggestions.append(f"The smallest square is {q.min_square_px:.0f} px wide; move closer or tilt less (at least {cfg.min_square_px:.0f} px).")
    if q.blur_px is not None and q.blur_px > cfg.blur_reject_px:
        obs.reasons.append("too_blurred")
        obs.suggestions.append(f"Edges are spread over about {q.blur_px:.1f} px. Hold the phone still, let it focus on the board and retake.")
    elif q.blur_px is not None and q.blur_px > cfg.blur_warn_px:
        obs.warnings.append("blurred")
    if q.blur_anisotropy is not None and q.blur_anisotropy > 1.6 and (q.blur_px or 0.0) > 1.5:
        obs.warnings.append("motion_blur_suspected")
    if q.contrast is not None and q.contrast < 0.3:
        obs.warnings.append("low_contrast")
    if q.white_clipped:
        obs.warnings.append("overexposed")
    if q.partly_outside:
        obs.warnings.append("board_partly_outside_picture")
    if q.missing_inside > 0:
        obs.warnings.append("corners_missing_inside_picture")
    if n_dropped:
        obs.warnings.append("corners_dropped")
    if details.get("dropped_uneven_lighting", 0):
        obs.warnings.append("glare_or_uneven_lighting")
    obs.ok = not obs.reasons
    return obs


def detect_board(
    image: np.ndarray,
    board: BoardSpec,
    cfg: DetectConfig | None = None,
    *,
    view_id: str = "",
    capture: CaptureConfig | None = None,
) -> BoardObservation:
    """Find ``board`` in one photo and measure its inner corners to sub-pixel.

    ``image`` is an 8-bit BGR or grey picture, upright, exactly as it will be
    analysed (pixel-centre convention).  A ChArUco board may be partly visible;
    a checkerboard must be completely inside the picture, because a part of a
    checkerboard cannot be told from any other part of it.

    The returned observation has ``ok=False`` with ``reasons`` when the photo
    cannot be used; it never raises for a photo that merely shows no board.
    """
    cfg = cfg or DetectConfig()
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3) or min(image.shape[:2]) < 32 or (image.ndim == 3 and image.shape[2] != 3):
        raise LumenValidationError("invalid_image", "detect_board needs an 8-bit grey or BGR image of at least 32 x 32 pixels.")
    gray = np.ascontiguousarray(to_gray(image))
    height, width = gray.shape[:2]
    size = (int(width), int(height))
    nx, ny = board.inner
    details: dict[str, Any] = {}

    image_id = hashlib.sha1(gray).hexdigest()[:16]

    def failed(reason: str, suggestion: str, **q: Any) -> BoardObservation:
        quality = BoardQuality(n_expected=board.n_corners, **q)
        return BoardObservation(view_id, board, size, False, np.zeros(0, np.int64), np.zeros((0, 2)), [reason], [], [suggestion], quality, capture, details, image_id=image_id)

    if board.kind == "checkerboard":
        coarse, info = _find_chessboard(gray, nx, ny, cfg)
        details.update(info)
        if coarse is None:
            if "partial_grid" in info:
                gx, gy = info["partial_grid"]
                return failed(
                    "checkerboard_partly_visible",
                    f"Only a {gx} x {gy} corner grid was found where {nx} x {ny} was declared. A checkerboard must be completely inside the "
                    "picture with its white margin and nothing covering or glaring on it (or the description does not match the board); "
                    "use a ChArUco board to photograph the picture corners.",
                    partly_outside=True,
                )
            return failed("board_not_found", "No checkerboard of the declared size was found. " + _capture_hint(board) + " Check squares_x / squares_y (count squares, not inner corners).")
        coarse, order = _canonical_chessboard(gray, coarse, nx, ny)
        details.update(order)
        ids = np.arange(nx * ny, dtype=np.int64)
    else:
        small, sx, sy = _reduced(gray, cfg.charuco_detect_long_edge)
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.CharucoDetector(board.cv_board(), cv2.aruco.CharucoParameters(), dp)
        cc, cid, mc, mid = det.detectBoard(small)
        details["detect_scale"] = round(sx, 4)
        details["n_markers"] = 0 if mid is None else int(len(mid))
        if cid is None or len(cid) == 0:
            if details["n_markers"] >= 4:
                return failed(
                    "too_few_corners",
                    f"{details['n_markers']} markers were read but they do not form the declared board. Check squares_x / squares_y and, for a board "
                    "with an even number of rows made with OpenCV older than 4.6, legacy_pattern.",
                )
            if details["n_markers"]:
                return failed("too_few_corners", f"Only {details['n_markers']} markers were read. " + _capture_hint(board))
            return failed("board_not_found", "No marker of the board's dictionary was found. " + _capture_hint(board) + " Check the dictionary name.")
        markers = np.stack([_to_full(m, sx, sy) for m in mc]) if mc is not None and len(mc) else np.zeros((0, 4, 2))
        ids = np.asarray(cid, dtype=np.int64).reshape(-1)
        coarse = _to_full(cc, sx, sy)
        order_idx = np.argsort(ids)
        ids, coarse = ids[order_idx], coarse[order_idx]

    # ---- sub-pixel localisation in the full-resolution picture
    xy = board.corner_xy(ids)
    lin = linear_luminance(gray)
    n_dropped = 0
    if ids.size >= 4 and np.unique(ids % nx).size >= 2 and np.unique(ids // nx).size >= 2:
        H0 = dlt_homography(xy, coarse)
        J = h_local_jacobian(H0, xy) * board.square_mm  # image vectors of one square along board x, y
        sq = np.linalg.svd(J, compute_uv=False)[:, -1]
        frac = min(cfg.refine_window_frac, cfg.refine_clean_frac * board.clean_fraction)
        half = np.clip(np.rint(frac * sq), cfg.refine_min_half, cfg.refine_max_half).astype(int)
        # the window (plus the gradient border cornerSubPix needs) must be inside the
        # picture: near the border it shrinks, and below the minimum the corner is dropped
        room = np.minimum.reduce([coarse[:, 0], coarse[:, 1], width - 1 - coarse[:, 0], height - 1 - coarse[:, 1]])
        half = np.minimum(half, np.floor(room - 2.0).astype(int))
        fits = half >= cfg.refine_min_half
        refined = coarse.copy()
        if np.any(fits):
            refined[fits] = _refine_corners(gray, coarse[fits], half[fits])
        shift = np.linalg.norm(refined - coarse, axis=1)
        settled = shift <= cfg.max_refine_shift_px + 0.1 * sq
        even = _corner_asymmetry(lin, refined, J[:, :, 0], J[:, :, 1], board.clean_fraction, np.maximum(half, cfg.refine_min_half)) <= cfg.max_corner_asymmetry
        keep = fits & settled & even
        n_dropped = int(np.sum(~keep))
        details["dropped_uneven_lighting"] = int(np.sum(fits & settled & ~even))
        if np.any(fits):
            details["refine_half_px"] = [int(half[fits].min()), int(half[fits].max())]
        ids, pts = ids[keep], refined[keep]
    else:
        pts = coarse
    obs = _finish_observation(board, size, ids, pts, lin, cfg, view_id, capture, details, n_dropped)
    obs.image_id = image_id
    if board.kind == "checkerboard":
        obs.warnings.extend(f for f in board.flags if f.startswith("rotation_ambiguous"))
    elif obs.n_points >= 6 and markers.shape[0] and not ({"too_few_corners", "corners_collinear"} & set(obs.reasons)):
        # the declared marker size against the picture: marker edges mapped to the board by the corners' homography
        Hinv = np.linalg.inv(fit_homography(board.corner_xy(obs.ids), obs.image_points))
        mm = apply_h(Hinv, markers.reshape(-1, 2)).reshape(-1, 4, 2)
        measured = float(np.median(np.linalg.norm(np.roll(mm, -1, axis=1) - mm, axis=2)))
        details["marker_mm_measured"] = round(measured, 2)
        if abs(measured / float(board.marker_mm) - 1.0) > cfg.marker_size_tolerance:  # type: ignore[arg-type]
            obs.ok = False
            # the blur figure looked for clean paper where this board has markers: drop it
            obs.reasons = [r for r in obs.reasons if r != "too_blurred"] + ["marker_size_mismatch"]
            obs.warnings = [w for w in obs.warnings if w not in ("blurred", "motion_blur_suspected")]
            obs.suggestions = [t for t in obs.suggestions if not t.startswith("Edges are spread")]
            obs.quality.blur_px = obs.quality.blur_anisotropy = None
            obs.suggestions.append(
                f"With squares of {board.square_mm:g} mm the markers in the picture measure about {measured:.1f} mm, but marker_mm is declared as "
                f"{board.marker_mm:g} mm. Measure the black square of a marker and one chessboard square again."
            )
    return obs


# --------------------------------------------------------------------------
# C. Calibration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationConfig:
    #: usable photos needed at all / below which a warning is given
    min_views: int = 6
    recommended_views: int = 10
    #: a set whose largest board tilt is below this cannot fix the focal length
    fronto_max_tilt_deg: float = 10.0
    #: RMS angular spread of the board normals below which all photos count as "the same tilt"
    min_tilt_spread_deg: float = 6.0
    #: a photo counts as tilted from here on; fewer than ``min_tilted_views`` is a warning
    tilted_deg: float = 15.0
    min_tilted_views: int = 3
    #: refusal limits after the solve
    max_rel_sigma_f: float = 0.01
    max_sigma_pp_frac: float = 0.01  # of the long image edge
    max_rms_px: float = 0.6
    #: outlier photos: per-view RMS above ``outlier_factor`` x the median and above ``outlier_min_px``
    outlier_factor: float = 3.0
    outlier_min_px: float = 0.1
    max_outlier_fraction: float = 0.3
    #: coverage: an image point is "near" a calibration point within this fraction of the diagonal
    coverage_radius_frac: float = 0.05
    min_coverage: float = 0.35
    warn_coverage: float = 0.6
    #: outermost point / distance to the farthest picture corner below which the
    #: distortion model is reported as extrapolated
    min_radial_reach: float = 0.75
    #: distortion terms are added when held-out squared error drops by this fraction ...
    select_min_gain: float = 0.02
    #: ... consistently over the held-out photos (paired t) ...
    select_min_t: float = 2.5
    #: ... and a new coefficient is this many standard deviations from zero
    select_min_z: float = 3.0
    #: leave-one-view-out is used for model selection up to this many photos, k-fold above
    lovo_max_views: int = 24
    kfold: int = 12
    #: held-out RMS / fit RMS above this is reported
    max_heldout_ratio: float = 1.6
    #: field of view assumed for the tilt estimates before the solve, when the
    #: photos themselves show no consistent perspective
    nominal_hfov_deg: float = 65.0
    #: |fy / fx - 1| above this (and above four standard deviations) is reported
    aspect_tolerance: float = 0.002
    #: a board is refused as "not square" when the photos fit squares whose y/x
    #: pitch ratio differs from the declared one by more than this, by more than
    #: ``stretch_min_z`` standard deviations, and the fit improves to below
    #: ``stretch_max_rms_ratio`` of the declared board's RMS
    max_board_stretch: float = 0.001
    stretch_min_z: float = 5.0
    stretch_max_rms_ratio: float = 0.95
    #: disagreement (px) between the chosen and the full distortion model, anywhere
    #: in the picture, above which the result is flagged as model dependent
    max_model_dependence_px: float = 1.0


#: distortion-term groups beyond the always-estimated k1, k2
_TERM_GROUPS: dict[str, tuple[str, ...]] = {"tangential": ("p1", "p2"), "k3": ("k3",)}


def _flags_for(groups: frozenset[str]) -> int:
    f = 0
    if "tangential" not in groups:
        f |= cv2.CALIB_ZERO_TANGENT_DIST
    if "k3" not in groups:
        f |= cv2.CALIB_FIX_K3
    return f


def _terms(groups: frozenset[str]) -> list[str]:
    est = ["k1", "k2"]
    if "tangential" in groups:
        est += ["p1", "p2"]
    if "k3" in groups:
        est += ["k3"]
    return est


@dataclass
class _Solution:
    camera: CameraModel
    params: np.ndarray  # 9
    sigma: np.ndarray  # 9 (formal, from OpenCV)
    rms: float
    R: list[np.ndarray]
    t: list[np.ndarray]
    residuals: list[np.ndarray]  # per view (N, 2): projected - observed

    @property
    def view_rms(self) -> np.ndarray:
        return np.array([float(np.sqrt(np.mean(np.sum(r * r, axis=1)))) for r in self.residuals])


def _residuals(cam: CameraModel, R: np.ndarray, t: np.ndarray, obj: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Reprojection residuals with the project's own camera model."""
    uv, _ = cam.project(obj @ R.T + t)
    return uv - img


def _solve(views: Sequence[BoardObservation], size: tuple[int, int], groups: frozenset[str]) -> _Solution:
    objs = [v.object_points.astype(np.float32) for v in views]
    imgs = [v.image_points.astype(np.float32) for v in views]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10)
    rms, K, d, rvecs, tvecs, s_int, _s_ext, _pv = cv2.calibrateCameraExtended(objs, imgs, size, None, None, flags=_flags_for(groups), criteria=crit)
    dist = np.zeros(5)
    dd = np.asarray(d, dtype=float).ravel()
    dist[: min(5, dd.size)] = dd[:5]
    params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist], dtype=float)
    if not np.all(np.isfinite(params)) or params[0] <= 0 or params[1] <= 0:
        raise cv2.error("calibration diverged")
    cam = CameraModel(size[0], size[1], float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), tuple(float(x) for x in dist), source=views[0].board.kind)  # type: ignore[arg-type]
    Rs = [cv2.Rodrigues(np.asarray(r, dtype=float))[0] for r in rvecs]
    ts = [np.asarray(t, dtype=float).ravel() for t in tvecs]
    res = [_residuals(cam, R, t, v.object_points, v.image_points) for R, t, v in zip(Rs, ts, views)]
    sig = np.asarray(s_int, dtype=float).ravel()[:9]
    return _Solution(cam, params, np.where(np.isfinite(sig), sig, np.inf), float(rms), Rs, ts, res)


def _view_pose(cam: CameraModel, v: BoardObservation) -> tuple[np.ndarray, np.ndarray] | None:
    ok, rvec, tvec = cv2.solvePnP(v.object_points, v.image_points, cam.K, cam.dist_array, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0], np.asarray(tvec, dtype=float).ravel()


def _heldout(views: Sequence[BoardObservation], size: tuple[int, int], groups: frozenset[str], folds: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """Held-out mean squared residual per view, and the parameter vector of every fold.

    For each fold the intrinsics are estimated without its views; each left-out
    view then only gets its own pose (6 parameters) fitted.
    """
    mse = np.full(len(views), np.nan)
    params: list[np.ndarray | None] = []
    for fold in folds:
        train = [v for i, v in enumerate(views) if i not in set(fold.tolist())]
        try:
            sol = _solve(train, size, groups)
        except cv2.error:
            params.append(None)
            continue
        params.append(sol.params)
        for i in fold:
            pose = _view_pose(sol.camera, views[i])
            if pose is not None:
                r = _residuals(sol.camera, pose[0], pose[1], views[i].object_points, views[i].image_points)
                mse[i] = float(np.mean(np.sum(r * r, axis=1)))
    return mse, params


def _board_stretch(views: Sequence[BoardObservation], size: tuple[int, int], start: CameraModel) -> dict[str, Any]:
    """Would squares that are longer (or shorter) along y explain the photos better?

    The calibration is repeated with the board's y coordinates multiplied by
    ``s`` (all five distortion terms free); the squared reprojection error is a
    parabola in ``s`` whose minimum and curvature give the stretch the photos
    ask for and its standard deviation.  With boards tilted in several
    directions a stretched print is not absorbed by ``fy / fx`` (measured:
    0.1 % is found to 0.005 %).  The result is relative to the declared pitches.
    """
    imgs = [v.image_points.astype(np.float32) for v in views]
    base = [v.object_points.astype(np.float32) for v in views]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10)

    def cost(sy: float) -> float:
        objs = [o * np.array([1.0, sy, 1.0], np.float32) for o in base]
        # started from the solution for the declared board: a small stretch moves it little
        return float(cv2.calibrateCamera(objs, imgs, size, start.K.copy(), start.dist_array.copy(), flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)[0]) ** 2

    xs = [0.995, 1.0, 1.005]
    ys = [cost(x) for x in xs]
    curv = 0.0
    for _ in range(2):
        k = np.argsort(ys)[:3]
        c = np.polyfit(np.array(xs)[k], np.array(ys)[k], 2)
        if not c[0] > 0:
            break
        curv = float(c[0])
        xs.append(float(np.clip(-c[1] / (2.0 * c[0]), 0.9, 1.1)))
        ys.append(cost(xs[-1]))
    best = int(np.argmin(ys))
    n = sum(v.n_points for v in views)
    sigma = float(np.sqrt(ys[best] / (2.0 * n * curv))) if curv > 0 else float("inf")
    return {"y_over_x": float(xs[best]), "sigma": sigma, "rms_declared_px": float(np.sqrt(ys[1])), "rms_best_px": float(np.sqrt(ys[best]))}


def _pose_figures(R: np.ndarray, t: np.ndarray, board: BoardSpec) -> dict[str, Any]:
    """Tilt, far side, roll and distance of a board pose ``x_cam = R x_board + t``."""
    n = R[:, 2] * (1.0 if R[2, 2] >= 0 else -1.0)
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    far = -n[:2]
    centre = R @ np.array([0.5 * board.width_mm, 0.5 * board.height_mm, 0.0]) + t
    return {
        "normal": n,
        "tilt_deg": tilt,
        "far_direction_deg": float(np.degrees(np.arctan2(far[1], far[0]))) if tilt >= 3.0 else None,
        "roll_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "distance_mm": float(np.linalg.norm(centre)),
    }


def _normal_spread_deg(normals: np.ndarray) -> float:
    """RMS angle between the board normals and their mean direction."""
    if normals.shape[0] < 2:
        return 0.0
    m = normals.sum(axis=0)
    m = m / max(np.linalg.norm(m), 1e-12)
    ang = np.arccos(np.clip(normals @ m, -1.0, 1.0))
    return float(np.degrees(np.sqrt(np.mean(ang * ang))))


def _model_dependence(a: CameraModel, b: CameraModel, points: np.ndarray, radius_px: float) -> dict[str, Any]:
    """Disagreement between two lens models that no homography can absorb (px).

    Both models undistort a grid over the whole picture; the two results are
    registered with a homography fitted where there are calibration points
    (that removes everything a pose or a planar mapping would absorb) and the
    remaining distance is reported near the points and over the whole picture.
    """
    us = np.linspace(0.0, a.width - 1.0, 25)
    vs = np.linspace(0.0, a.height - 1.0, 19)
    g = np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2)
    ia, ib = a.undistort_pixels(g), b.undistort_pixels(g)
    out: dict[str, Any] = {"max_px_near_points": None, "max_px_whole_picture": None}
    if points.shape[0] == 0 or not (np.all(np.isfinite(ia)) and np.all(np.isfinite(ib))):
        return out
    d2 = np.min(np.sum((g[:, None, :] - points[None, ::3, :]) ** 2, axis=2), axis=1)
    near = d2 <= radius_px**2
    if near.sum() < 8:
        return out
    diff = np.linalg.norm(apply_h(fit_homography(ia[near], ib[near]), ia) - ib, axis=1)
    out.update(max_px_near_points=round(float(diff[near].max()), 4), max_px_whole_picture=round(float(diff.max()), 4))
    return out


_REGIONS = ("top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right")


def coverage_report(points: np.ndarray, width: int, height: int, radius_frac: float = 0.05, principal_point: tuple[float, float] | None = None) -> dict[str, Any]:
    """Where in the picture the calibration points are.

    ``fraction`` is the share of the image area that lies within
    ``radius_frac`` x diagonal of a calibration point (evaluated on a raster of
    about 4 px).  ``regions`` gives the same share for a 3 x 3 division of the
    picture; ``grid`` counts the points on an occupancy grid of about 8 cells
    along the long side.  ``radial_reach`` is how far out the points go from
    the principal point, relative to the farthest picture corner.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    step = max(1, int(round(max(width, height) / 400.0)))
    w, h = -(-width // step), -(-height // step)
    radius = radius_frac * float(np.hypot(width, height))
    mask = np.full((h, w), 255, np.uint8)
    if pts.shape[0]:
        iu = np.clip(((pts[:, 0] + 0.5) / step).astype(int), 0, w - 1)
        iv = np.clip(((pts[:, 1] + 0.5) / step).astype(int), 0, h - 1)
        mask[iv, iu] = 0
        near = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * step <= radius
    else:
        near = np.zeros((h, w), dtype=bool)
    regions: dict[str, float] = {}
    for k, name in enumerate(_REGIONS):
        r, c = divmod(k, 3)
        sub = near[r * h // 3 : (r + 1) * h // 3, c * w // 3 : (c + 1) * w // 3]
        regions[name] = round(float(sub.mean()), 3) if sub.size else 0.0
    cols = 8 if width >= height else max(1, int(round(8.0 * width / height)))
    rows = 8 if height > width else max(1, int(round(8.0 * height / width)))
    grid = np.zeros((rows, cols), dtype=int)
    if pts.shape[0]:
        gi = np.clip(((pts[:, 0] + 0.5) / width * cols).astype(int), 0, cols - 1)
        gj = np.clip(((pts[:, 1] + 0.5) / height * rows).astype(int), 0, rows - 1)
        np.add.at(grid, (gj, gi), 1)
    cx, cy = principal_point if principal_point is not None else ((width - 1) * 0.5, (height - 1) * 0.5)
    corners = np.array([[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]])
    r_corner = float(np.max(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)))
    reach = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) / r_corner) if pts.shape[0] else 0.0
    empty = [name for name, f in regions.items() if f < 0.5]
    if not empty:
        note = "Calibration points lie near every part of the picture."
    else:
        note = "Few or no calibration points near: " + ", ".join(empty) + ". The lens model is extrapolated there."
    return {
        "fraction": round(float(near.mean()), 4),
        "radius_px": round(radius, 1),
        "regions": regions,
        "empty_regions": empty,
        "grid": grid.tolist(),
        "grid_shape": [cols, rows],
        "radial_reach": round(reach, 3),
        "n_points": int(pts.shape[0]),
        "note": note,
    }


@dataclass
class CalibrationResult:
    ok: bool
    reasons: list[str]
    warnings: list[str]
    suggestions: list[str]
    board: BoardSpec
    image_size: tuple[int, int]
    capture: CaptureConfig
    #: the lens model; ``None`` when the calibration was refused
    camera: CameraModel | None = None
    rms_px: float | None = None
    n_views: int = 0
    n_views_given: int = 0
    n_points: int = 0
    #: Standard uncertainties of fx, fy, cx, cy, k1, k2, p1, p2, k3.  ``formal``
    #: is the least-squares value of the chosen model (OpenCV) and ``jackknife``
    #: the spread over leaving one photo out (it sees errors common to a photo).
    #: When distortion terms were fixed at zero, that is an assumption: the
    #: reported value then also contains ``model_shift`` (how far the parameter
    #: moves when all five terms are estimated) and the amount by which the
    #: five-term model's own variance exceeds the formal one,
    #: ``sqrt(max(formal, jackknife)^2 + model_shift^2 + max(formal_full^2 - formal^2, 0))``.
    #: So it never claims more than the full model could, while the estimate
    #: itself stays the one of the simpler model.  For a term fixed at zero this
    #: is what the photos would allow it to be.
    param_sigma: list[float] | None = None
    param_sigma_formal: list[float] | None = None
    param_sigma_jackknife: list[float] | None = None
    param_model_shift: list[float] | None = None
    distortion: dict[str, Any] = field(default_factory=dict)
    views: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    tilt: dict[str, Any] = field(default_factory=dict)
    heldout: dict[str, Any] = field(default_factory=dict)
    #: the board's squareness as the photos see it (see ``_board_stretch``)
    board_check: dict[str, Any] = field(default_factory=dict)
    #: identity of the observations (their points) and of the pictures the estimate was made from
    used_fingerprints: tuple[str, ...] = ()
    used_images: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ queries
    @property
    def coverage_fraction(self) -> float | None:
        return self.coverage.get("fraction") if self.coverage else None

    def matches(self, width: int, height: int, capture: CaptureConfig | None = None) -> tuple[bool, str]:
        """May this calibration be applied to a picture of this geometry / configuration?"""
        w, h = self.image_size
        if (int(width), int(height)) != (w, h):
            if (int(height), int(width)) == (w, h):
                return False, f"calibrated at {w}x{h}, picture is {width}x{height}: the same sensor mode in the other orientation is a different configuration"
            return False, f"calibrated at {w}x{h}, picture is {width}x{height}"
        if capture is not None:
            a, b = self.capture.key(), capture.key()
            for name, x, y in zip(("device", "lens", "zoom", "capture_mode"), a, b):
                if x not in ("", None) and y not in ("", None) and x != y:
                    return False, f"calibrated for {name} {x!r}, picture reports {y!r}"
        return True, "same camera configuration"

    def to_dict(self) -> dict[str, Any]:
        def r(seq: list[float] | None) -> list[float | None] | None:
            # six significant digits; a non-finite sigma (parameter not determined) becomes null
            return None if seq is None else [float(f"{x:.6g}") if np.isfinite(x) else None for x in seq]

        cam = self.camera
        intr = None
        if cam is not None:
            intr = {
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "dist": list(cam.dist),
                "hfov_deg": round(float(np.degrees(cam.hfov)), 3),
                "vfov_deg": round(float(np.degrees(cam.vfov)), 3),
                "aspect_ratio": round(cam.fy / cam.fx, 6),
                "max_distortion_px": round(cam.max_distortion_px(), 2),
            }
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "suggestions": list(self.suggestions),
            "method": self.board.kind,
            "board": self.board.to_dict(),
            "image_size": list(self.image_size),
            "capture": self.capture.to_dict(),
            "intrinsics": intr,
            "param_names": list(PARAM_NAMES),
            "param_sigma": r(self.param_sigma),
            "param_sigma_formal": r(self.param_sigma_formal),
            "param_sigma_jackknife": r(self.param_sigma_jackknife),
            "param_model_shift": r(self.param_model_shift),
            "param_sigma_definition": "sqrt(max(formal, jackknife)^2 + model_shift^2 + max(formal_of_full_model^2 - formal^2, 0))",
            "distortion_model": self.distortion,
            "rms_px": None if self.rms_px is None else round(self.rms_px, 4),
            "n_views": self.n_views,
            "n_views_given": self.n_views_given,
            "n_points": self.n_points,
            "views": self.views,
            "excluded": self.excluded,
            "coverage": self.coverage,
            "tilt": self.tilt,
            "heldout": self.heldout,
            "board_check": self.board_check,
            "notes": list(self.notes),
        }

    def to_profile(
        self,
        profile_id: str,
        *,
        label: str = "",
        device: str | None = None,
        lens: str | None = None,
        zoom: float | None = None,
        capture_mode: str | None = None,
        created: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Dictionary for :class:`lumen.schemas.Calibration` (provenance ``measured``).

        Raises :class:`CalibrationError` for a refused calibration: a profile
        is never produced from one.
        """
        if not self.ok or self.camera is None or self.param_sigma is None:
            raise CalibrationError(
                self.reasons[0] if self.reasons else "calibration_refused",
                "The calibration was refused; no profile can be made from it.",
                details={"reasons": list(self.reasons)},
                suggestions=list(self.suggestions),
            )
        cam = self.camera
        est = ", ".join(self.distortion.get("estimated", []))
        fixed = ", ".join(self.distortion.get("fixed_zero", [])) or "none"
        auto = (
            f"{self.board.kind} target, {self.n_views} photos, {self.n_points} points; estimated {est}; fixed at zero: {fixed}; "
            f"RMS {self.rms_px:.3f} px, leave-one-view-out {self.heldout.get('rms_px', float('nan')):.3f} px; "
            f"coverage {self.coverage.get('fraction', 0.0):.0%}" + (f"; warnings: {', '.join(self.warnings)}" if self.warnings else "") + ". " + WEB_API_NOTE
        )
        return {
            "id": str(profile_id),
            "label": label or f"{self.board.kind} calibration {self.image_size[0]}x{self.image_size[1]}",
            "method": self.board.kind,
            "provenance": "measured",
            "width": int(self.image_size[0]),
            "height": int(self.image_size[1]),
            "fx": float(cam.fx),
            "fy": float(cam.fy),
            "cx": float(cam.cx),
            "cy": float(cam.cy),
            "dist": [float(x) for x in cam.dist],
            "rms_px": float(self.rms_px),  # type: ignore[arg-type]
            "n_views": int(self.n_views),
            "coverage": float(self.coverage.get("fraction", 0.0)),
            "param_sigma": [float(x) for x in self.param_sigma],
            "device": self.capture.device if device is None else device,
            "lens": self.capture.lens if lens is None else lens,
            "zoom": self.capture.zoom if zoom is None else zoom,
            "capture_mode": self.capture.capture_mode if capture_mode is None else capture_mode,
            "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": (notes + " " if notes else "") + auto,
        }


def _one_configuration(obs: Sequence[BoardObservation]) -> tuple[BoardSpec, tuple[int, int], CaptureConfig]:
    """All photos must show one board and come from one camera configuration."""
    board = obs[0].board
    for o in obs[1:]:
        if o.board != board:
            raise CalibrationError(
                "mixed_boards",
                "The photos were analysed with different board descriptions.",
                details={"boards": [o.board.to_dict() for o in obs if o.board != board][:1] + [board.to_dict()]},
                suggestions=["Calibrate with one board and one description of it."],
            )
    sizes: dict[tuple[int, int], list[str]] = {}
    for k, o in enumerate(obs):
        sizes.setdefault((int(o.image_size[0]), int(o.image_size[1])), []).append(o.view_id or f"#{k}")
    if len(sizes) > 1:
        keys = sorted(sizes, key=lambda s: -len(sizes[s]))
        transposed = any((b, a) in sizes for (a, b) in sizes if a != b)
        msg = "The photos do not all have the same pixel size: " + ", ".join(f"{len(sizes[s])} x {s[0]}x{s[1]}" for s in keys) + "."
        if transposed:
            msg += " Landscape and portrait photos of the same sensor mode are different camera configurations."
        raise CalibrationError(
            "mixed_camera_configurations",
            msg,
            details={"image_sizes": {f"{s[0]}x{s[1]}": sizes[s] for s in keys}, "orientation_mix": transposed},
            suggestions=[
                "Calibrate one configuration at a time: same camera, lens, zoom, resolution and phone orientation for every photo.",
                "Calibrate in the orientation you will measure in; turn the board, not the phone, to vary the roll.",
            ],
        )
    # capture settings reported by the client: every setting that is reported must agree
    names = ("device", "lens", "zoom", "capture_mode")
    seen: dict[str, dict[Any, Any]] = {n: {} for n in names}  # normalised value -> value as reported
    for o in obs:
        if o.capture is not None:
            for name, norm in zip(names, o.capture.key()):
                if norm not in ("", None):
                    seen[name].setdefault(norm, getattr(o.capture, name))
    clash = {name: sorted(str(v) for v in vals.values()) for name, vals in seen.items() if len(vals) > 1}
    if clash:
        raise CalibrationError(
            "mixed_camera_configurations",
            "The photos report different capture settings: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in clash.items()) + ".",
            details={"capture": clash},
            suggestions=["Calibrate one configuration at a time: same camera, lens, zoom, resolution and phone orientation for every photo."],
        )
    one = {name: (next(iter(vals.values())) if vals else None) for name, vals in seen.items()}
    cap = CaptureConfig(device=one["device"] or "", lens=one["lens"] or "", zoom=one["zoom"], capture_mode=one["capture_mode"] or "")
    return board, next(iter(sizes)), cap


_SIDES = ("left", "right", "top", "bottom")


def _tilt_report(figs: list[dict[str, Any]], cfg: CalibrationConfig) -> dict[str, Any]:
    tilts = np.array([f["tilt_deg"] for f in figs]) if figs else np.zeros(0)
    normals = np.array([f["normal"] for f in figs]) if figs else np.zeros((0, 3))
    sides = {s: 0 for s in _SIDES}
    for f in figs:
        if f["tilt_deg"] >= cfg.tilted_deg and f.get("far_direction_deg") is not None:
            sides[_side_name(f["far_direction_deg"])] += 1  # type: ignore[index]
    rolls = np.deg2rad([f["roll_deg"] for f in figs if f.get("roll_deg") is not None])
    # spread of the board's in-picture orientation, counting a half turn as the same orientation
    roll_spread = float(np.degrees(0.5 * np.sqrt(max(-2.0 * np.log(max(abs(np.mean(np.exp(2j * rolls))), 1e-12)), 0.0)))) if len(rolls) else 0.0
    return {
        "min_deg": round(float(tilts.min()), 2) if tilts.size else None,
        "max_deg": round(float(tilts.max()), 2) if tilts.size else None,
        "median_deg": round(float(np.median(tilts)), 2) if tilts.size else None,
        "spread_deg": round(_normal_spread_deg(normals), 2),
        "n_tilted": int(np.sum(tilts >= cfg.tilted_deg)),
        "tilted_from_deg": cfg.tilted_deg,
        "far_sides": sides,
        "missing_far_sides": [s for s in _SIDES if sides[s] == 0],
        "roll_spread_deg": round(roll_spread, 1),
    }


def _tilt_advice(tilt: dict[str, Any], n_more: int = 0) -> str:
    missing = tilt.get("missing_far_sides") or []
    want = [f"one with the {s} side of the board farther from the camera" for s in missing] or ["tilted toward different sides"]
    lead = f"Add at least {max(n_more, len(missing), 3)} photos" if n_more or len(missing) >= 3 else f"Add {max(len(missing), 1)} or more photos"
    return f"{lead} with the board tilted 25–45° to the camera: " + ", ".join(want) + " (sides as they appear in the picture). Keep the board sharp and in the picture."


def calibrate_camera(observations: Sequence[BoardObservation], cfg: CalibrationConfig | None = None) -> CalibrationResult:
    """Intrinsics of ONE camera configuration from several board photos.

    Raises :class:`CalibrationError` when the input is not one configuration
    (no photos, different boards, different pixel sizes / orientations or
    capture settings).  Everything else is reported in the result:
    ``ok=False`` with ``reasons`` and concrete ``suggestions`` when the photos
    do not determine the camera, in which case ``camera`` is ``None`` and no
    profile can be made.

    Stages (each leaves its figures in the result, also when refused):

    1. *Screening* — photos without a usable detection and repeated photos are
       set aside (``excluded``, stage ``detection`` / ``screening``).
    2. *Sufficiency, before any solve* — enough photos; the board tilted in at
       least one of them (``fronto_parallel_only``) and not at the same tilt in
       all (``same_tilt``).  Tilts come from each photo's vanishing line.
    3. *Outlier photos* — with all five distortion terms free, the photo with
       the largest reprojection RMS is removed while it exceeds
       ``outlier_factor`` x the median and ``outlier_min_px``; at most
       ``max_outlier_fraction`` of the photos and never below ``min_views``
       (then: ``inconsistent_views``).  Every removal is listed with its figures.
    4. *Board squareness* — an undeclared stretch of the print along one
       direction is estimated from the photos (``board_not_square``).
    5. *Distortion model* — ``k1, k2``; ``p1, p2`` and ``k3`` only when held-out
       photos support them (``distortion_model``).
    6. *Uncertainty, held-out error, coverage, tilt diversity.*
    7. *Gates* — focal length and principal point determined, reprojection
       error, coverage; everything else is a warning with advice.
    """
    cfg = cfg or CalibrationConfig()
    obs = list(observations)
    if not obs:
        raise CalibrationError("no_views", "No calibration photos were given.", suggestions=["Take 10 to 15 photos of the board from different directions."])
    board, size, capture = _one_configuration(obs)
    width, height = size
    res = CalibrationResult(False, [], [], [], board, size, capture, n_views_given=len(obs))
    res.notes.append(WEB_API_NOTE)
    res.warnings.extend(board.flags)
    res.notes.extend(board.notes().values())

    # ---- screening of the single photos
    usable: list[BoardObservation] = []
    index_of = {id(o): k for k, o in enumerate(obs)}
    seen: set[str] = set()
    for k, o in enumerate(obs):
        keys = {o.fingerprint} | ({"image:" + o.image_id} if o.image_id else set())
        if not (o.ok and o.n_points >= 6):
            res.excluded.append({"view_id": o.view_id or f"#{k}", "index": k, "stage": "detection", "reasons": list(o.reasons) or ["too_few_corners"], "detail": "; ".join(o.suggestions)})
        elif keys & seen:
            res.excluded.append({"view_id": o.view_id or f"#{k}", "index": k, "stage": "screening", "reasons": ["duplicate_photo"], "detail": "the same picture (or the same measured points) was given before; it is counted once"})
        else:
            usable.append(o)
            seen |= keys

    def view_row(o: BoardObservation, k: int) -> dict[str, Any]:
        q = o.quality
        return {
            "view_id": o.view_id or f"#{k}",
            "used": False,
            "n_points": o.n_points,
            "tilt_deg": None if q.tilt_deg is None else round(q.tilt_deg, 2),
            "tilt_source": q.tilt_source,
            "far_side": q.far_side,
            "blur_px": None if q.blur_px is None else round(q.blur_px, 3),
            "hull_coverage": round(q.coverage, 4),
            "warnings": list(o.warnings),
        }

    all_pts = np.concatenate([o.image_points for o in usable]) if usable else np.zeros((0, 2))
    res.coverage = coverage_report(all_pts, width, height, cfg.coverage_radius_frac)

    # Tilts before any solve: the principal point is taken at the image centre and the
    # focal length from what the perspective of all photos agrees on (Zhang's closed
    # form); if they carry no consistent perspective, from a nominal field of view.
    geo = [_vanishing(o.homography, width, height) if o.homography is not None else _vanishing(np.eye(3), width, height) for o in usable]
    f_nominal = 0.5 * width / np.tan(np.deg2rad(cfg.nominal_hfov_deg) * 0.5)
    f_pre = _focal_from_perspective(geo, width, height)
    pre = []
    for g in geo:
        tilt = _tilt_from_vanishing(g, f_pre or f_nominal)
        t, a_ = np.deg2rad(tilt), np.deg2rad(g["direction_deg"] or 0.0)
        pre.append({"tilt_deg": tilt, "far_direction_deg": g["direction_deg"] if tilt >= 3.0 else None, "roll_deg": None, "normal": np.array([-np.sin(t) * np.cos(a_), -np.sin(t) * np.sin(a_), np.cos(t)])})
    res.tilt = _tilt_report(pre, cfg)
    res.tilt["source"] = "estimated before calibration: principal point at the image centre, focal length " + (f"{f_pre:.0f} px from the perspective of the photos" if f_pre else f"{f_nominal:.0f} px assumed (the photos show no consistent perspective)")
    res.views = [view_row(o, k) for k, o in enumerate(obs)]

    def refuse(reason: str, suggestion: str) -> None:
        if reason not in res.reasons:
            res.reasons.append(reason)
        if suggestion and suggestion not in res.suggestions:
            res.suggestions.append(suggestion)

    if len(usable) < cfg.min_views:
        refuse("too_few_views", f"{len(usable)} usable photos; at least {cfg.min_views} are needed ({cfg.recommended_views} or more recommended). " + _tilt_advice(res.tilt, cfg.min_views - len(usable)))
    if usable and (res.tilt["max_deg"] or 0.0) < cfg.fronto_max_tilt_deg:
        refuse(
            "fronto_parallel_only",
            f"Every photo looks at the board almost straight on (largest tilt about {res.tilt['max_deg']:.0f}°). In such photos a longer focal "
            "length cannot be told from a larger distance, so the focal length is not determined. " + _tilt_advice(res.tilt, 4),
        )
    elif len(usable) >= 2 and res.tilt["spread_deg"] < cfg.min_tilt_spread_deg:
        side = next((s for s in _SIDES if res.tilt["far_sides"][s]), None)
        refuse(
            "same_tilt",
            f"All photos see the board at practically the same tilt (about {res.tilt['median_deg']:.0f}°"
            + (f", {side} side away" if side else "")
            + "). Photos of the same board orientation carry the information of a single photo: focal length and principal point cannot be "
            "separated. " + _tilt_advice(res.tilt, 4) + " Add one photo straight on as well.",
        )
    for e in res.excluded:
        res.views[e["index"]]["excluded"] = list(e["reasons"])
    if res.reasons:
        _coverage_advice(res, cfg)
        return res

    # ---- outlier photos (judged with the full five-coefficient model, so that a
    # missing distortion term is not mistaken for a bad photo)
    full = frozenset(_TERM_GROUPS)
    views = list(usable)
    max_removed = int(np.floor(cfg.max_outlier_fraction * len(usable)))
    removed = 0
    try:
        while True:
            sol = _solve(views, size, full)
            vr = sol.view_rms
            med = float(np.median(vr))
            worst = int(np.argmax(vr))
            limit = max(cfg.outlier_factor * med, cfg.outlier_min_px)
            if vr[worst] <= limit:
                break
            if removed >= max_removed or len(views) - 1 < cfg.min_views:
                refuse(
                    "inconsistent_views",
                    f"Photo {views[worst].view_id or '#' + str(index_of[id(views[worst])])} disagrees with the others ({vr[worst]:.2f} px against a typical {med:.2f} px) and too few "
                    "photos would remain without it. Retake the set: board flat on a rigid backing, phone still, same zoom throughout.",
                )
                break
            bad = views.pop(worst)
            removed += 1
            res.excluded.append(
                {
                    "view_id": bad.view_id or f"#{index_of[id(bad)]}",
                    "index": index_of[id(bad)],
                    "stage": "calibration",
                    "reasons": ["reprojection_outlier"],
                    "rms_px": round(float(vr[worst]), 4),
                    "median_rms_px": round(med, 4),
                    "detail": f"reprojection RMS {vr[worst]:.3f} px is {vr[worst] / max(med, 1e-9):.1f} times the median of the set ({med:.3f} px)"
                    + ("; the photo is blurred" if "blurred" in bad.warnings or "motion_blur_suspected" in bad.warnings else ""),
                }
            )
    except cv2.error as exc:
        refuse("solver_failed", "The photos do not determine a camera model. " + _tilt_advice(res.tilt, 4))
        res.notes.append(f"OpenCV: {str(exc).strip().splitlines()[-1][:200]}")
        _coverage_advice(res, cfg)
        return res
    full_sol = sol

    # ---- is the board as square as declared?  (a stretched print biases fy / fx)
    try:
        st = _board_stretch(views, size, full_sol.camera)
    except cv2.error:
        st = None
    if st is not None:
        off = st["y_over_x"] - 1.0
        found = abs(off) > cfg.max_board_stretch and abs(off) > cfg.stretch_min_z * st["sigma"] and st["rms_best_px"] < cfg.stretch_max_rms_ratio * st["rms_declared_px"]
        res.board_check = {
            "y_over_x": round(st["y_over_x"], 6),
            "sigma": None if not np.isfinite(st["sigma"]) else float(f"{st['sigma']:.2g}"),
            "rms_declared_px": round(st["rms_declared_px"], 4),
            "rms_best_px": round(st["rms_best_px"], 4),
            "stretched": bool(found),
            "note": "y_over_x is the ratio of the squares' pitch along y to that along x which fits the photos best, relative to the declared pitches",
        }
        if found:
            pitch_y = board.pitch_y_mm * st["y_over_x"]
            refuse(
                "board_not_square",
                f"The photos fit a board whose squares are {abs(off):.2%} {'longer' if off > 0 else 'shorter'} along y than declared "
                f"(reprojection RMS {st['rms_best_px']:.3f} px instead of {st['rms_declared_px']:.3f} px). Printers often stretch one paper direction. "
                f"Measure the print across several squares along both sides and declare both pitches (square_y_mm would be about {pitch_y:.3f} mm "
                f"if square_mm = {board.square_mm:g} mm is right), or print again at 100 % scale. Until then fy / fx would be wrong by about that amount."
                + (" Photos listed as outliers may only disagree because of this." if removed else ""),
            )

    # ---- distortion model: k1, k2 always; tangential and k3 only if held-out photos support them
    n = len(views)
    if n <= cfg.lovo_max_views:
        folds = [np.array([i]) for i in range(n)]
        cv_name = "leave-one-view-out"
    else:
        folds = [np.arange(k, n, cfg.kfold) for k in range(cfg.kfold)]
        cv_name = f"{cfg.kfold}-fold over views"
    weights = np.array([v.n_points for v in views], dtype=float)
    cache: dict[frozenset[str], tuple[_Solution, np.ndarray, list[np.ndarray | None]]] = {}

    def evaluate(groups: frozenset[str]) -> tuple[_Solution, np.ndarray, list[np.ndarray | None]]:
        if groups not in cache:
            s = _solve(views, size, groups)
            mse, pars = _heldout(views, size, groups, folds)
            cache[groups] = (s, mse, pars)
        return cache[groups]

    def heldout_rms(mse: np.ndarray) -> float:
        m = np.isfinite(mse)
        return float(np.sqrt(np.sum(mse[m] * weights[m]) / max(np.sum(weights[m]), 1.0))) if np.any(m) else float("nan")

    selection: list[dict[str, Any]] = []
    try:
        current: frozenset[str] = frozenset()
        s0, mse0, _ = evaluate(current)
        selection.append({"terms": _terms(current), "rms_px": round(s0.rms, 4), "heldout_rms_px": round(heldout_rms(mse0), 4), "decision": "base model"})
        while True:
            best: tuple[float, frozenset[str], dict[str, Any]] | None = None
            base_sol, base_mse, _ = evaluate(current)
            for g in _TERM_GROUPS:
                if g in current:
                    continue
                cand = current | {g}
                try:
                    cs, cm, _ = evaluate(cand)
                except cv2.error:
                    continue
                m = np.isfinite(base_mse) & np.isfinite(cm)
                gain = 1.0 - float(np.sum(cm[m] * weights[m]) / max(np.sum(base_mse[m] * weights[m]), 1e-300))
                diff = base_mse[m] - cm[m]
                sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
                tval = float(np.mean(diff) / (sd / np.sqrt(diff.size))) if sd > 0 else 0.0
                idx = [PARAM_NAMES.index(p) for p in _TERM_GROUPS[g]]
                z = float(np.max(np.abs(cs.params[idx]) / np.where(cs.sigma[idx] > 0, cs.sigma[idx], np.inf)))
                supported = gain >= cfg.select_min_gain and tval >= cfg.select_min_t and z >= cfg.select_min_z
                row = {
                    "terms": _terms(cand),
                    "added": list(_TERM_GROUPS[g]),
                    "rms_px": round(cs.rms, 4),
                    "heldout_rms_px": round(heldout_rms(cm), 4),
                    "heldout_gain": round(gain, 4),
                    "paired_t": round(tval, 2),
                    "max_z": round(z, 2),
                    "decision": "supported" if supported else "not supported",
                }
                selection.append(row)
                if supported and (best is None or gain > best[0]):
                    best = (gain, cand, row)
            if best is None:
                break
            best[2]["decision"] = "added"
            current = best[1]
        sol, mse, fold_params = evaluate(current)
    except cv2.error as exc:
        refuse("solver_failed", "The photos do not determine a camera model. " + _tilt_advice(res.tilt, 4))
        res.notes.append(f"OpenCV: {str(exc).strip().splitlines()[-1][:200]}")
        _coverage_advice(res, cfg)
        return res

    estimated = _terms(current)
    res.distortion = {
        "estimated": estimated,
        "fixed_zero": [p for p in ("k1", "k2", "p1", "p2", "k3") if p not in estimated],
        "selection": selection,
        "cross_validation": cv_name,
        "rule": (
            f"k1 and k2 are always estimated. p1, p2 (together) and k3 are added only if the held-out squared error drops by at least "
            f"{cfg.select_min_gain:.0%}, the drop is consistent over the held-out photos (paired t >= {cfg.select_min_t}) and a new coefficient is at "
            f"least {cfg.select_min_z} standard deviations from zero."
        ),
    }

    # ---- leave-one-view-out error and jackknife spread for the chosen model
    if cv_name != "leave-one-view-out":
        mse, fold_params = _heldout(views, size, current, [np.array([i]) for i in range(n)])
    good = [p for p in fold_params if p is not None]
    jack = None
    if len(good) == n and n >= 3:
        P = np.array(good)
        jack = np.sqrt((n - 1) / n * np.sum((P - P.mean(axis=0)) ** 2, axis=0))
    free = np.array([True, True, True, True] + [p in estimated for p in ("k1", "k2", "p1", "p2", "k3")])
    formal = np.where(free, sol.sigma, 0.0)
    if jack is not None:
        jack = np.where(free, jack, 0.0)
        sigma = np.maximum(formal, jack)
    else:
        sigma = formal
        res.warnings.append("jackknife_unavailable")
    # A term that is fixed at zero is an assumption, not knowledge: how far the
    # parameters move when it is estimated too is part of their uncertainty
    # (see CalibrationResult.param_sigma) ...
    if current != full:
        shift = np.abs(full_sol.params - sol.params)
        # ... and the full model's own, larger variance: the shift above is only known that well
        wider = np.sqrt(np.clip(np.where(np.isfinite(full_sol.sigma), full_sol.sigma, 0.0) ** 2 - formal**2, 0.0, None))
    else:
        shift = wider = np.zeros(9)
    sigma = np.sqrt(sigma**2 + shift**2 + wider**2)
    ho = heldout_rms(mse)
    res.heldout = {
        "method": "leave-one-view-out: intrinsics from the other photos, only the pose of the left-out photo is fitted",
        "rms_px": round(ho, 4),
        "max_view_rms_px": round(float(np.sqrt(np.nanmax(mse))), 4) if np.any(np.isfinite(mse)) else None,
        "ratio_to_fit": round(ho / max(sol.rms, 1e-12), 3),
    }

    # ---- per-view figures from the solved poses
    figs = [_pose_figures(R, t, board) for R, t in zip(sol.R, sol.t)]
    vr = sol.view_rms
    used_ids = {id(v): k for k, v in enumerate(views)}
    for e in res.excluded:
        res.views[e["index"]]["excluded"] = list(e["reasons"])
    for k, o in enumerate(obs):
        row = res.views[k]
        j = used_ids.get(id(o))
        if j is None:
            continue
        f = figs[j]
        row.update(
            used=True,
            rms_px=round(float(vr[j]), 4),
            max_px=round(float(np.max(np.linalg.norm(sol.residuals[j], axis=1))), 4),
            heldout_rms_px=None if not np.isfinite(mse[j]) else round(float(np.sqrt(mse[j])), 4),
            tilt_deg=round(f["tilt_deg"], 2),
            tilt_source="calibrated",
            far_side=_side_name(f["far_direction_deg"]),
            roll_deg=round(f["roll_deg"], 1),
            distance_mm=round(f["distance_mm"], 1),
        )
    res.tilt = _tilt_report(figs, cfg)
    res.tilt["source"] = "calibrated board poses"
    dists = [f["distance_mm"] for f in figs]
    res.tilt["distance_mm"] = [round(float(min(dists)), 1), round(float(max(dists)), 1)]
    pts_used = np.concatenate([v.image_points for v in views])
    res.coverage = coverage_report(pts_used, width, height, cfg.coverage_radius_frac, (sol.camera.cx, sol.camera.cy))

    res.rms_px = sol.rms
    res.n_views = n
    res.n_points = int(pts_used.shape[0])
    res.param_sigma_formal = [float(x) for x in formal]
    res.param_sigma_jackknife = None if jack is None else [float(x) for x in jack]
    res.param_model_shift = [float(x) for x in shift]
    res.param_sigma = [float(x) for x in sigma]
    if current != full:
        dep = _model_dependence(sol.camera, full_sol.camera, pts_used, res.coverage["radius_px"])
        dep["compared_with"] = _terms(full)
        res.distortion["model_dependence"] = dep
    res.used_fingerprints = tuple(v.fingerprint for v in views)
    res.used_images = tuple(v.image_id for v in views if v.image_id)

    # ---- gates (fail closed) and advice
    f_mean = 0.5 * (sol.camera.fx + sol.camera.fy)
    rel_f = float(max(sigma[0], sigma[1]) / f_mean) if np.all(np.isfinite(sigma[:2])) else float("inf")
    pp = float(max(sigma[2], sigma[3]) / max(width, height)) if np.all(np.isfinite(sigma[2:4])) else float("inf")
    if (res.tilt["max_deg"] or 0.0) < cfg.fronto_max_tilt_deg:
        refuse("fronto_parallel_only", "Every photo looks at the board almost straight on, so the focal length is not determined. " + _tilt_advice(res.tilt, 4))
    elif res.tilt["spread_deg"] < cfg.min_tilt_spread_deg:
        refuse("same_tilt", "All photos see the board at practically the same tilt; focal length and principal point cannot be separated. " + _tilt_advice(res.tilt, 4))
    if rel_f > cfg.max_rel_sigma_f:
        refuse("focal_length_not_determined", f"The focal length is only known to {rel_f:.1%} (limit {cfg.max_rel_sigma_f:.1%}). " + _tilt_advice(res.tilt, 3))
    if pp > cfg.max_sigma_pp_frac:
        refuse("principal_point_not_determined", f"The principal point is only known to {max(sigma[2], sigma[3]):.0f} px. " + _tilt_advice(res.tilt, 3) + " Turn the board by a quarter turn for some of them.")
    if not np.all(np.isfinite(sigma)):
        refuse("uncertainty_not_determined", "The uncertainty of the camera model could not be computed from these photos. " + _tilt_advice(res.tilt, 3))
    if sol.rms > cfg.max_rms_px:
        refuse("high_reprojection_error", f"The photos fit the camera model only to {sol.rms:.2f} px RMS (limit {cfg.max_rms_px} px). Check that the board is flat and rigid, the description matches the print and the zoom did not change.")
    if n < cfg.recommended_views:
        res.warnings.append("few_views")
        res.suggestions.append(f"{n} photos were used; {cfg.recommended_views} or more give a steadier result.")
    if res.tilt["n_tilted"] < cfg.min_tilted_views:
        res.warnings.append("few_tilted_views")
        res.suggestions.append(f"Only {res.tilt['n_tilted']} photos are tilted by {cfg.tilted_deg:.0f}° or more. " + _tilt_advice(res.tilt))
    elif len(res.tilt["missing_far_sides"]) >= 2:
        res.warnings.append("tilt_directions_missing")
        res.suggestions.append(_tilt_advice(res.tilt))
    if res.tilt["roll_spread_deg"] < 15.0:
        res.warnings.append("no_roll_diversity")
        res.suggestions.append("In every photo the board has the same orientation in the picture. Add two or three photos with the board turned by a quarter turn (do not turn the phone).")
    if res.heldout["ratio_to_fit"] > cfg.max_heldout_ratio:
        res.warnings.append("heldout_error_high")
        res.suggestions.append("Photos that were left out are predicted clearly worse than the fit suggests; add photos so that every part of the picture is seen by several of them.")
    aspect = sol.camera.fy / sol.camera.fx
    if abs(aspect - 1.0) > max(cfg.aspect_tolerance, 4.0 * float(np.hypot(sigma[0], sigma[1]) / f_mean)):
        res.warnings.append("aspect_ratio_not_unity")
        checked = res.board_check.get("sigma")
        if checked is not None and checked < 0.25 * abs(aspect - 1.0):
            res.suggestions.append(f"fy / fx = {aspect:.4f}. The photos show the board to be as square as declared (to {checked:.3%}), so the pixels of this capture mode are not square; that is part of the profile.")
        else:
            res.suggestions.append(f"fy / fx = {aspect:.4f}. Either the pixels are not square in this capture mode or the print is stretched: measure the board along both sides, and include photos tilted in different directions and with the board turned by a quarter turn.")
    if removed:
        res.warnings.append("views_excluded")
    dep_px = res.distortion.get("model_dependence", {}).get("max_px_whole_picture")
    if dep_px is not None and dep_px > cfg.max_model_dependence_px:
        res.warnings.append("distortion_model_dependent")
        res.suggestions.append(
            f"Distortion terms that these photos cannot determine would change the lens model by up to {dep_px:.1f} px in parts of the picture "
            "without calibration points. Measure only where there are calibration points, or add photos that reach the picture corners."
        )
    _coverage_advice(res, cfg)
    res.ok = not res.reasons
    if res.ok:
        res.camera = sol.camera
    else:
        res.notes.append("The estimate is withheld because the calibration was refused.")
    return res


def _coverage_advice(res: CalibrationResult, cfg: CalibrationConfig) -> None:
    cov = res.coverage
    if not cov:
        return
    where = ", ".join(cov["empty_regions"])
    hint = "Add photos with the board in these parts of the picture" + (" (a ChArUco board may reach beyond the picture edge)." if res.board.kind == "charuco" else " (a ChArUco board, which may be cut off by the picture edge, reaches the corners more easily).")
    if cov["fraction"] < cfg.min_coverage:
        if "insufficient_coverage" not in res.reasons:
            res.reasons.append("insufficient_coverage")
            res.suggestions.append(f"Calibration points lie near only {cov['fraction']:.0%} of the picture (at least {cfg.min_coverage:.0%} needed); nothing near: {where}. " + hint)
    elif cov["fraction"] < cfg.warn_coverage or cov["empty_regions"]:
        code = "low_coverage" if cov["fraction"] < cfg.warn_coverage else "regions_without_points"
        if code not in res.warnings:
            res.warnings.append(code)
            res.suggestions.append(f"Calibration points lie near {cov['fraction']:.0%} of the picture; few or none near: {where}. " + hint)
    if cov.get("radial_reach", 1.0) < cfg.min_radial_reach and "distortion_extrapolated" not in res.warnings:
        res.warnings.append("distortion_extrapolated")
        res.suggestions.append(f"The outermost calibration point is at {cov['radial_reach']:.0%} of the way to the picture corner; beyond it the distortion model is an extrapolation.")


# --------------------------------------------------------------------------
# D. Planar validation against an independent board photo
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanarCheckConfig:
    #: residual RMS (px) accepted as "consistent with a plane": this many times
    #: the calibration RMS, but never below the floor
    max_rms_ratio: float = 2.5
    rms_floor_px: float = 0.08


def _planar_figures(board_xy: np.ndarray, px: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    """Homography fit of board mm -> pixels and what it leaves over."""
    H = fit_homography(board_xy, px)
    r_px = apply_h(H, board_xy) - px
    back = apply_h(np.linalg.inv(H), px)
    r_mm = back - board_xy
    e_px = np.linalg.norm(r_px, axis=1)
    e_mm = np.linalg.norm(r_mm, axis=1)
    # distances between all pairs of measured points versus the board geometry
    iu, ju = np.triu_indices(board_xy.shape[0], k=1)
    d_true = np.linalg.norm(board_xy[iu] - board_xy[ju], axis=1)
    d_meas = np.linalg.norm(back[iu] - back[ju], axis=1)
    err = d_meas - d_true
    k = int(np.argmax(np.abs(err)))
    span = int(np.argmax(d_true))
    return {
        "rms_px": float(np.sqrt(np.mean(e_px**2))),
        "max_px": float(e_px.max()),
        "rms_mm": float(np.sqrt(np.mean(e_mm**2))),
        "max_mm": float(e_mm.max()),
        "max_distance_error_mm": float(abs(err[k])),
        "max_distance_error_signed_mm": float(err[k]),
        "max_distance_error_pair": [int(ids[iu[k]]), int(ids[ju[k]])],
        "max_distance_error_length_mm": float(d_true[k]),
        "rms_distance_error_mm": float(np.sqrt(np.mean(err**2))),
        "span_mm": float(d_true[span]),
        "span_error_mm": float(err[span]),
        "homography": H,
    }


@dataclass
class PlanarValidation:
    ok: bool
    reasons: list[str]
    warnings: list[str]
    #: True / False when it could be checked against the calibration's own photos, else None
    independent: bool | None
    view_id: str
    n_points: int
    #: homography residuals after undistorting with the lens model
    with_model: dict[str, Any]
    #: the same on the raw pixels (no lens model at all)
    without_model: dict[str, Any]
    #: without / with (RMS in pixels); > 1 means the lens model helps
    improvement: float
    #: residual RMS that would be expected from the calibration alone, and the limit applied
    expected_rms_px: float | None
    limit_rms_px: float
    board: BoardSpec
    region: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def block(b: dict[str, Any]) -> dict[str, Any]:
            return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in b.items() if k != "homography"}

        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "independent": self.independent,
            "view_id": self.view_id,
            "n_points": self.n_points,
            "with_model": block(self.with_model),
            "without_model": block(self.without_model),
            "improvement": round(self.improvement, 3),
            "expected_rms_px": None if self.expected_rms_px is None else round(self.expected_rms_px, 4),
            "limit_rms_px": round(self.limit_rms_px, 4),
            "square_mm": self.board.square_mm,
            "size_provenance": self.board.size_provenance,
            "region": self.region,
            "notes": list(self.notes),
        }


def validate_planar(
    calibration: "CalibrationResult | CameraModel | dict[str, Any]",
    observation: BoardObservation,
    cfg: PlanarCheckConfig | None = None,
) -> PlanarValidation:
    """How well can a plane be measured through this lens model?

    ``observation`` is a board photo that was **not** used for the
    calibration.  Its corners are undistorted with the lens model, a
    homography is fitted from the board's known geometry to them, and what is
    left over is reported: residuals in pixels and in millimetres on the board
    (RMS and maximum) and the worst error of any distance between two board
    points.  The same is done on the raw pixels, so that the effect of the
    distortion model is visible.  Once the pixels are undistorted the homography
    needs nothing else from the calibration (it absorbs the pose and any scale
    of the focal length): this is the check behind using homographies on a
    planar target when the distortion is characterised but no 3-D calibration
    is relied on.

    ``calibration`` may be a :class:`CalibrationResult` (independence is then
    verified against the photos it was made from), a
    :class:`~lumen.geometry.camera.CameraModel` or a profile dictionary.

    The millimetre figures use the declared ``square_mm`` and scale with it; a
    homography cannot see a wrong square size.  Raises
    :class:`CalibrationError` when the photo is of another camera
    configuration or shows no usable board.
    """
    cfg = cfg or PlanarCheckConfig()
    reasons: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    independent: bool | None = None
    expected: float | None = None
    if isinstance(calibration, CalibrationResult):
        if not calibration.ok or calibration.camera is None:
            raise CalibrationError("calibration_refused", "The calibration was refused; there is no lens model to validate.", details={"reasons": calibration.reasons}, suggestions=calibration.suggestions)
        cam = calibration.camera
        match, why = calibration.matches(observation.image_size[0], observation.image_size[1], observation.capture)
        if not match:
            raise CalibrationError("mismatched_camera_configuration", f"The validation photo is not of the calibrated configuration: {why}.", suggestions=["Take the validation photo with the same camera, lens, zoom, resolution and orientation."])
        independent = observation.fingerprint not in calibration.used_fingerprints and observation.image_id not in calibration.used_images
        expected = calibration.rms_px
    else:
        cam = calibration if isinstance(calibration, CameraModel) else CameraModel.from_dict(calibration)
        if (cam.width, cam.height) != tuple(observation.image_size):
            raise CalibrationError("mismatched_camera_configuration", f"The lens model is for {cam.width}x{cam.height} but the validation photo is {observation.image_size[0]}x{observation.image_size[1]}.", suggestions=["Take the validation photo with the same camera, lens, zoom, resolution and orientation."])
        if isinstance(calibration, dict) and calibration.get("rms_px") is not None:
            expected = float(calibration["rms_px"])
        warnings.append("independence_not_verified")
        notes.append("Only a CalibrationResult knows which photos it was made from; make sure this photo was not one of them.")
    if not observation.ok or observation.n_points < 8:
        raise CalibrationError("validation_board_unusable", "The validation photo does not show enough of the board.", details={"reasons": observation.reasons}, suggestions=observation.suggestions or ["Retake the validation photo with the board sharp and inside the picture."])

    board = observation.board
    xy = board.corner_xy(observation.ids)
    raw = observation.image_points
    with_model = _planar_figures(xy, cam.undistort_pixels(raw), observation.ids)
    without = _planar_figures(xy, raw, observation.ids)
    limit = max(cfg.max_rms_ratio * (expected or 0.0), cfg.rms_floor_px)
    if independent is False:
        reasons.append("observation_used_for_calibration")
        notes.append("This photo was used for the calibration; its residuals are not an independent check.")
    if with_model["rms_px"] > limit:
        reasons.append("planar_residuals_high")
        notes.append(f"After undistortion the board is not explained by a plane to within {limit:.2f} px (RMS {with_model['rms_px']:.2f} px): the lens model does not hold here, or the board is not flat.")
    if board.size_provenance == "nominal":
        warnings.append("square_size_nominal")
        notes.append("Millimetre figures assume the nominal print size.")
    q = observation.quality
    cov = coverage_report(raw, observation.image_size[0], observation.image_size[1], 0.05, (cam.cx, cam.cy))
    region = {"hull_coverage": round(q.coverage, 4), "bbox": None if q.bbox is None else [round(float(x), 4) for x in q.bbox], "radial_reach": cov["radial_reach"]}
    notes.append("The check holds for the part of the picture this board covers.")
    return PlanarValidation(
        ok=not reasons,
        reasons=reasons,
        warnings=warnings,
        independent=independent,
        view_id=observation.view_id,
        n_points=observation.n_points,
        with_model=with_model,
        without_model=without,
        improvement=float(without["rms_px"] / max(with_model["rms_px"], 1e-12)),
        expected_rms_px=expected,
        limit_rms_px=float(limit),
        board=board,
        region=region,
        notes=notes,
    )


# --------------------------------------------------------------------------
# E. Summary
# --------------------------------------------------------------------------
def calibration_summary(
    result: CalibrationResult,
    validations: Sequence[PlanarValidation] = (),
    observations: Sequence[BoardObservation] | None = None,
) -> dict[str, Any]:
    """Everything about one calibration as a JSON-serialisable document."""
    doc: dict[str, Any] = {
        "ok": result.ok,
        "status": "calibrated" if result.ok else "refused",
        "reasons": list(result.reasons),
        "warnings": list(result.warnings),
        "suggestions": list(result.suggestions),
        "calibration": result.to_dict(),
        "planar_validation": [v.to_dict() for v in validations],
        "planar_validated": bool(validations) and all(v.ok for v in validations),
        "web_api_note": WEB_API_NOTE,
        "units": {"length": "mm", "image": "px (integer coordinates are pixel centres, u right, v down)", "angles": "deg"},
    }
    if observations is not None:
        doc["observations"] = [o.to_dict(points=False) for o in observations]
    return doc
