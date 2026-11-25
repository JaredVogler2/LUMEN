"""Frame ingestion: decode, quality metrics, calibration matching, tag detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..config import Settings
from ..geometry.camera import CameraModel
from ..schemas import Calibration, Frame, FrameQualityModel, FrameRole, FrameSource, MarkerObservation, Provenance, TorchState
from ..vision.imaging import DecodedImage, decode_image, focal_px_from_exif, frame_quality
from ..vision.markers import MarkerConfig, TagDetection, TagSizeTable, detect_tags
from ..vision.planar import TagObs


def camera_for_frame(
    cal: Calibration | None, width: int, height: int, native_w: int, native_h: int, scale: float,
    *, device: str = "", lens: str = "", zoom: float | None = None,
) -> tuple[CameraModel | None, str]:
    """Camera model for a frame, or ``None`` with the reason it does not apply.

    A profile is applied only to frames whose *upright native* pixel geometry
    equals the profile's (the analysis image may be an exact, recorded uniform
    downscale of it).  Anything else — a different resolution, aspect ratio or
    orientation — is a different capture configuration and is refused, because
    crop, binning and electronic stabilisation change the intrinsics in ways a
    web page cannot observe.

    Where both the profile and the frame say which camera, lens or zoom was
    used and they differ, the profile is refused as well: a phone's lenses
    often deliver pictures of the same pixel size with different intrinsics.
    A value missing on either side is not a reason to refuse (the browser
    does not always report it).
    """
    if cal is None:
        return None, "no calibration profile attached"
    for what, have, want in (("camera", device, cal.device), ("lens", lens, cal.lens)):
        if have and want and have.strip().lower() != want.strip().lower():
            return None, f"profile is for {what} {want!r} but this picture reports {have!r}"
    if zoom is not None and cal.zoom is not None and abs(float(zoom) - float(cal.zoom)) > 1e-3:
        return None, f"profile is for zoom {cal.zoom:g} but this picture reports zoom {float(zoom):g}"
    upright_native = (native_w, native_h)
    if upright_native != (cal.width, cal.height):
        if (native_h, native_w) == (cal.width, cal.height):
            return None, (
                f"profile is for {cal.width}x{cal.height} but the frame is {native_w}x{native_h}: "
                "same sensor mode in the other orientation — calibrate in the orientation you measure in"
            )
        return None, f"profile is for {cal.width}x{cal.height} but the frame is {native_w}x{native_h}"
    cam = CameraModel(cal.width, cal.height, cal.fx, cal.fy, cal.cx, cal.cy, tuple((list(cal.dist) + [0.0] * 5)[:5]), source=cal.method)  # type: ignore[arg-type]
    if (width, height) != (cal.width, cal.height):
        sx = width / float(cal.width)
        sy = height / float(cal.height)
        if abs(sx - sy) > 2e-3:
            return None, "analysis image is not a uniform rescale of the calibrated geometry"
        cam = cam.scaled(sx, sy)
        cam = CameraModel(width, height, cam.fx, cam.fy, cam.cx, cam.cy, cam.dist, source=cal.method)
    return cam, f"profile {cal.id} applied"


def marker_from_detection(d: TagDetection) -> MarkerObservation:
    return MarkerObservation(
        tag_id=d.tag_id,
        corners_px=np.asarray(d.corners_px, dtype=float).tolist(),
        corners_ideal=np.asarray(d.corners_ideal, dtype=float).tolist(),
        cov=np.asarray(d.cov, dtype=float).tolist(),
        corner_sigma_px=d.corner_sigma_px,
        size_mm=d.size_mm,
        size_provenance=Provenance.MEASURED if d.size_provenance == "measured" else Provenance.NOMINAL,
        accepted=d.accepted,
        reject_reasons=list(d.reject_reasons),
        flags=list(d.flags),
        refined=d.refined,
        edge_px=[float(e) for e in d.edge_px],
        sigma_blur_px=d.sigma_blur_px,
        contrast=d.contrast,
        bit_errors=d.bit_errors,
        decode_margin=d.decode_margin,
        bias_px=d.bias_px,
        bias_sigma_px=None if not np.isfinite(d.bias_sigma_px) else float(d.bias_sigma_px),
        bias_applied=d.bias_applied,
        scale_mode=None if d.scale_mode is None else [float(x) for x in d.scale_mode],
        max_sagitta_px=d.max_sagitta_px,
        saturation=d.saturation,
        quality=d.quality(),
    )


@dataclass(frozen=True)
class EdgeShiftPool:
    """The dark-side edge shift measured on the tags that are large enough for it (px), to use on those that are not."""

    delta_px: float
    sigma_px: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"delta_px": round(self.delta_px, 4), "sigma_px": round(self.sigma_px, 4), "n": self.n}


#: Part of a pooled shift that is kept as uncertainty: the shift is not exactly the same on every tag.
POOL_MODEL_FRACTION = 0.35
POOL_FLOOR_PX = 0.03
#: Tags whose own estimates disagree by more than this are not pooled at all.
POOL_MAX_SPREAD_PX = 0.12


def edge_shift_pool(markers: Iterable[MarkerObservation]) -> EdgeShiftPool | None:
    """Pooled edge shift of the accepted tags that carry their own estimate.

    The shift comes from lens blur and from the camera's tone and sharpening
    processing, which are the same for every tag of a photo and nearly the same
    for every photo of a capture.  A tag too small for its own estimate
    therefore takes the pooled value; what is kept as its uncertainty is the
    scatter between the donors, a fraction of the value itself and a floor.
    """
    d = [(float(m.bias_px), float(m.bias_sigma_px)) for m in markers if m.accepted and m.bias_applied and m.bias_sigma_px]
    if not d:
        return None
    v = np.array([x[0] for x in d])
    w = 1.0 / np.maximum(np.array([x[1] for x in d]), 1e-3) ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    spread = float(np.sqrt(np.sum(w * (v - mean) ** 2) / np.sum(w))) if len(d) > 1 else 0.0
    if spread > POOL_MAX_SPREAD_PX:
        return None
    sigma = float(np.sqrt(1.0 / np.sum(w) + spread**2 + (POOL_MODEL_FRACTION * mean) ** 2 + POOL_FLOOR_PX**2))
    return EdgeShiftPool(mean, sigma, len(d))


def tag_obs(m: MarkerObservation, pool: EdgeShiftPool | None = None) -> TagObs:
    """Observation for the solvers.  With ``pool``, a tag without its own edge-shift estimate is corrected with
    the pooled one: its border moves outward by that amount and the uncertainty of the amount goes into its
    covariance as one common move of the four border lines."""
    corners = np.asarray(m.corners_ideal, dtype=float)
    cov = np.asarray(m.cov, dtype=float)
    pooled = False
    if pool is not None and not m.bias_applied and m.scale_mode is not None:
        g = np.asarray(m.scale_mode, dtype=float)
        corners = corners + pool.delta_px * g.reshape(4, 2)
        cov = cov + np.outer(g, g) * pool.sigma_px**2
        pooled = True
    return TagObs(m.tag_id, corners, cov, float(m.size_mm), bool(m.bias_applied), pooled)


def capture_identity(img: DecodedImage, client: dict[str, Any] | None) -> dict[str, Any]:
    """Which camera / lens / zoom a picture reports (browser stream label or EXIF); empty when unknown."""
    client = client or {}
    return {
        "device": str(client.get("camera_label") or img.meta.model or "")[:120],
        "lens": str(client.get("lens") or img.meta.lens_model or "")[:120],
        "zoom": _opt_float(client.get("zoom")),
    }


def detect_in_image(
    img: DecodedImage,
    cal: Calibration | None,
    sizes: TagSizeTable,
    cfg: MarkerConfig | None = None,
    client: dict[str, Any] | None = None,
) -> tuple[list[TagDetection], CameraModel | None, str]:
    cam, note = camera_for_frame(cal, img.width, img.height, _upright(img)[0], _upright(img)[1], img.processing_scale, **capture_identity(img, client))
    dets = detect_tags(img.bgr, camera=cam, sizes=sizes, cfg=cfg)
    return dets, cam, note


def _upright(img: DecodedImage) -> tuple[int, int]:
    """Native pixel size after EXIF orientation (what the frame would be at scale 1)."""
    m = img.meta
    if m.exif_orientation in (5, 6, 7, 8):
        return m.native_height, m.native_width
    return m.native_width, m.native_height


def decode_for_session(data: bytes, settings: Settings) -> DecodedImage:
    lim = settings.limits
    return decode_image(
        data,
        max_bytes=lim.max_image_bytes,
        max_pixels=lim.max_image_pixels,
        max_long_edge=lim.max_analysis_long_edge,
    )


def build_frame(
    *,
    frame_id: str,
    index: int,
    blob: str,
    data_len: int,
    img: DecodedImage,
    dets: list[TagDetection],
    cal: Calibration | None,
    cal_note: str,
    cam: CameraModel | None,
    role: FrameRole,
    source: FrameSource,
    client: dict[str, Any] | None,
) -> Frame:
    client = client or {}
    q = frame_quality(img.bgr)
    up_w, up_h = _upright(img)
    exif: dict[str, Any] = {
        k: v
        for k, v in (
            ("make", img.meta.make),
            ("model", img.meta.model),
            ("lens_model", img.meta.lens_model),
            ("focal_length_mm", img.meta.focal_length_mm),
            ("focal_length_35mm", img.meta.focal_length_35mm),
            ("digital_zoom", img.meta.digital_zoom),
            ("exposure_time_s", img.meta.exposure_time_s),
            ("iso", img.meta.iso),
            ("f_number", img.meta.f_number),
            ("flash_fired", img.meta.flash_fired),
            ("datetime_original", img.meta.datetime_original),
        )
        if v is not None
    }
    f_prior = focal_px_from_exif(img.meta, img.width, img.height)
    if f_prior is not None:
        exif["focal_px_prior"] = round(f_prior, 1)
    torch = str(client.get("torch", "unknown")).lower()
    if torch not in ("on", "off", "unknown"):
        torch = "unknown"
    if torch == "unknown" and img.meta.flash_fired is not None:
        torch = "on" if img.meta.flash_fired else "off"
    notes: list[str] = []
    if img.processing_scale != 1.0:
        notes.append(f"analysed at {img.processing_scale:.4f}x of native resolution")
    if q.clipped_high > 0.05:
        notes.append("large overexposed area")
    if q.mean < 0.12:
        notes.append("very dark image")
    return Frame(
        id=frame_id,
        index=index,
        role=role,
        source=source,
        width=img.width,
        height=img.height,
        native_width=up_w,
        native_height=up_h,
        processing_scale=img.processing_scale,
        exif_orientation=img.meta.exif_orientation,
        orientation_applied=img.meta.orientation_applied,
        camera_label=str(client.get("camera_label") or img.meta.model or "")[:120],
        lens=str(client.get("lens") or img.meta.lens_model or "")[:120],
        zoom=_opt_float(client.get("zoom")),
        facing_mode=str(client.get("facing_mode", ""))[:24],
        torch=TorchState(torch),
        captured_at=str(client.get("captured_at", ""))[:40],
        exif=exif,
        quality=FrameQualityModel(**q.to_dict()),
        sha256=img.meta.sha256,
        n_bytes=data_len,
        blob=blob,
        markers=[marker_from_detection(d) for d in dets],
        selected=True,
        pair_id=(str(client["pair_id"])[:40] if client.get("pair_id") else None),
        notes=notes,
        calibration_id=(cal.id if (cal is not None and cam is not None) else ""),
        calibration_note=cal_note,
        video_time_s=_opt_float(client.get("video_time_s")),
    )


def _opt_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def sizes_from_state(d: dict[str, Any] | None) -> TagSizeTable:
    return TagSizeTable.from_dict(d or {})
