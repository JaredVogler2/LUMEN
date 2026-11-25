"""Image decoding, orientation normalisation and radiometric helpers.

Two facts drive this module:

* A phone JPEG stores pixels in a display-referred, gamma-like encoding.  A
  blurred black/white edge is only *symmetric* in linear light, so every
  sub-pixel edge measurement in LUMEN is made on linearised intensities
  (inverse sRGB transfer).  Measuring on the encoded values shifts each edge
  toward its dark side and makes every AprilTag look smaller than it is — a
  pure scale bias that no amount of averaging removes (quantified in
  docs/BUILD_LOG.md).
* EXIF orientation must be applied exactly once, and the resulting pixel
  geometry is what every later stage (and the browser overlay) refers to.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..errors import LimitExceededError, UnsupportedMediaError

# 8-bit code value -> linear light (sRGB electro-optical transfer function).
_x = np.arange(256, dtype=np.float64) / 255.0
SRGB_TO_LINEAR_LUT = np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4).astype(np.float32)
del _x

_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),  # confirmed below
    (b"BM", "image/bmp"),
)


def sniff_image_type(data: bytes) -> str | None:
    """Identify an image container from magic bytes (never from the filename)."""
    head = data[:16]
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            if mime == "image/webp" and head[8:12] != b"WEBP":
                continue
            return mime
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypmsf1", b"ftypheim", b"ftypheis"):
        return "image/heic"
    return None


def linearize(gray_or_bgr_u8: np.ndarray) -> np.ndarray:
    """8-bit sRGB-encoded image -> float32 linear light in [0, 1]."""
    return SRGB_TO_LINEAR_LUT[gray_or_bgr_u8]


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def linear_luminance(bgr: np.ndarray) -> np.ndarray:
    """Linear-light luminance (Rec.709 weights) as float32, from 8-bit BGR/grey."""
    if bgr.ndim == 2:
        return SRGB_TO_LINEAR_LUT[bgr]
    lin = SRGB_TO_LINEAR_LUT[bgr]
    return (0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]).astype(np.float32)


@dataclass
class ImageMeta:
    """What we know about how a still was captured (all optional, never assumed)."""

    mime: str = "application/octet-stream"
    native_width: int = 0  # stored pixel size before EXIF orientation
    native_height: int = 0
    exif_orientation: int = 1
    orientation_applied: bool = False
    make: str | None = None
    model: str | None = None
    lens_model: str | None = None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    digital_zoom: float | None = None
    exposure_time_s: float | None = None
    iso: int | None = None
    f_number: float | None = None
    flash_fired: bool | None = None
    datetime_original: str | None = None
    sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _exif_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        try:
            num, den = v  # rational tuple
            return float(num) / float(den) if den else None
        except Exception:
            return None


def read_exif(data: bytes) -> dict[str, Any]:
    """Best-effort EXIF read with Pillow.  Never raises."""
    out: dict[str, Any] = {}
    try:
        from PIL import ExifTags, Image

        with Image.open(io.BytesIO(data)) as im:
            out["native_width"], out["native_height"] = im.size
            exif = im.getexif()
            if not exif:
                return out
            base = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            try:
                sub = exif.get_ifd(ExifTags.IFD.Exif)
                base.update({ExifTags.TAGS.get(k, str(k)): v for k, v in sub.items()})
            except Exception:
                pass
            out["orientation"] = int(base.get("Orientation", 1) or 1)
            for key, name in (
                ("Make", "make"),
                ("Model", "model"),
                ("LensModel", "lens_model"),
                ("DateTimeOriginal", "datetime_original"),
            ):
                if base.get(key):
                    out[name] = str(base[key]).strip().strip("\x00")
            for key, name in (
                ("FocalLength", "focal_length_mm"),
                ("FocalLengthIn35mmFilm", "focal_length_35mm"),
                ("DigitalZoomRatio", "digital_zoom"),
                ("ExposureTime", "exposure_time_s"),
                ("FNumber", "f_number"),
            ):
                if key in base:
                    f = _exif_float(base[key])
                    if f is not None and f > 0:
                        out[name] = f
            iso = base.get("ISOSpeedRatings", base.get("PhotographicSensitivity"))
            if iso is not None:
                try:
                    out["iso"] = int(iso[0] if isinstance(iso, (tuple, list)) else iso)
                except Exception:
                    pass
            if "Flash" in base:
                try:
                    out["flash_fired"] = bool(int(base["Flash"]) & 1)
                except Exception:
                    pass
    except Exception:
        return out
    return out


@dataclass
class DecodedImage:
    bgr: np.ndarray  # uint8, upright (EXIF orientation applied)
    meta: ImageMeta
    #: Factor applied to the upright native image to obtain ``bgr`` (1.0 = native).
    processing_scale: float = 1.0

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])


def decode_image(
    data: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
    max_long_edge: int | None = None,
) -> DecodedImage:
    """Validate and decode an uploaded still.

    * type is sniffed from magic bytes,
    * EXIF orientation is applied by OpenCV exactly once,
    * images above ``max_long_edge`` are reduced with area averaging and the
      scale is recorded so calibration data can be rescaled exactly.
    """
    if len(data) == 0:
        raise UnsupportedMediaError("empty_upload", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise LimitExceededError(
            "image_too_large",
            f"Image is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB.",
        )
    mime = sniff_image_type(data)
    if mime is None:
        raise UnsupportedMediaError(
            "unsupported_image_type",
            "Unsupported image type. Use JPEG or PNG.",
        )
    if mime == "image/heic":
        raise UnsupportedMediaError(
            "heic_not_supported",
            "HEIC images cannot be decoded on this server.",
            suggestions=[
                "On iPhone: Settings → Camera → Formats → Most Compatible, or let the browser convert the photo to JPEG.",
            ],
        )
    exif = read_exif(data)
    nw, nh = int(exif.get("native_width", 0)), int(exif.get("native_height", 0))
    if nw and nh and nw * nh > max_pixels:
        raise LimitExceededError(
            "image_too_many_pixels",
            f"Image has {nw * nh / 1e6:.1f} MP; the limit is {max_pixels / 1e6:.1f} MP.",
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # applies EXIF orientation
    if bgr is None or bgr.size == 0:
        raise UnsupportedMediaError("image_decode_failed", "The image could not be decoded.")
    if bgr.shape[0] * bgr.shape[1] > max_pixels:
        raise LimitExceededError("image_too_many_pixels", "Image exceeds the pixel limit.")
    orientation = int(exif.get("orientation", 1) or 1)
    meta = ImageMeta(
        mime=mime,
        native_width=nw or int(bgr.shape[1]),
        native_height=nh or int(bgr.shape[0]),
        exif_orientation=orientation,
        orientation_applied=orientation not in (0, 1),
        make=exif.get("make"),
        model=exif.get("model"),
        lens_model=exif.get("lens_model"),
        focal_length_mm=exif.get("focal_length_mm"),
        focal_length_35mm=exif.get("focal_length_35mm"),
        digital_zoom=exif.get("digital_zoom"),
        exposure_time_s=exif.get("exposure_time_s"),
        iso=exif.get("iso"),
        f_number=exif.get("f_number"),
        flash_fired=exif.get("flash_fired"),
        datetime_original=exif.get("datetime_original"),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    scale = 1.0
    if max_long_edge is not None:
        long_edge = max(bgr.shape[0], bgr.shape[1])
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            new_w = int(round(bgr.shape[1] * scale))
            new_h = int(round(bgr.shape[0] * scale))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # exact per-axis scale after rounding is what calibration must use
            scale = new_w / float(meta.native_width if orientation < 5 else meta.native_height)
    return DecodedImage(bgr=bgr, meta=meta, processing_scale=scale)


def encode_jpeg(bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------
# Frame quality metrics
# --------------------------------------------------------------------------
@dataclass
class FrameQuality:
    sharpness: float  # robust gradient-energy measure, resolution-normalised
    mean: float  # mean code value 0..1
    clipped_high: float  # fraction of pixels >= 250
    clipped_low: float  # fraction of pixels <= 5
    contrast: float  # p95 - p5 of code values, 0..1

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def frame_quality(bgr: np.ndarray, work_long_edge: int = 640) -> FrameQuality:
    """Cheap whole-frame quality metrics, comparable across resolutions.

    Sharpness is the variance of the Laplacian evaluated at a fixed working
    resolution, so a 12 MP still and a 1080p video frame of the same scene with
    the same *angular* blur score alike.
    """
    g = to_gray(bgr)
    s = work_long_edge / float(max(g.shape))
    if s < 1.0:
        g = cv2.resize(g, (int(round(g.shape[1] * s)), int(round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    cdf = np.cumsum(hist) / total
    p5 = float(np.searchsorted(cdf, 0.05)) / 255.0
    p95 = float(np.searchsorted(cdf, 0.95)) / 255.0
    return FrameQuality(
        sharpness=float(lap.var()),
        mean=float(g.mean()) / 255.0,
        clipped_high=float(hist[250:].sum()) / total,
        clipped_low=float(hist[:6].sum()) / total,
        contrast=p95 - p5,
    )


def focal_px_from_exif(meta: ImageMeta, width: int, height: int) -> float | None:
    """Coarse focal length prior (pixels) from the 35 mm-equivalent EXIF tag.

    The 35 mm equivalent is defined on the 43.27 mm frame diagonal.  This is an
    *inferred prior* (typically within a few percent); it is never treated as a
    calibration.
    """
    f35 = meta.focal_length_35mm
    if not f35 or f35 <= 0:
        return None
    diag_px = math.hypot(width, height)
    return float(f35) * diag_px / 43.2666
