"""Frame ingestion: decode, quality metrics, calibration matching, tag detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Settings
from ..geometry.camera import CameraModel
from ..schemas import Calibration, Frame, FrameQualityModel, FrameRole, FrameSource, MarkerObservation, Provenance, TorchState
from ..vision.imaging import DecodedImage, decode_image, focal_px_from_exif, frame_quality
from ..vision.markers import MarkerConfig, TagDetection, TagSizeTable, detect_tags
from ..vision.planar import TagObs


def camera_for_frame(cal: Calibration | None, width: int, height: int, native_w: int, native_h: int, scale: float) -> tuple[CameraModel | None, str]:
    """Camera model for a frame, or ``None`` with the reason it does not apply.

    A profile is applied only to frames whose *upright native* pixel geometry
    equals the profile's (the analysis image may be an exact, recorded uniform
    downscale of it).  Anything else — a different resolution, aspect ratio or
    orientation — is a different capture configuration and is refused, because
    crop, binning and electronic stabilisation change the intrinsics in ways a
    web page cannot observe.
    """
    if cal is None:
        return None, "no calibration profile attached"
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
        max_sagitta_px=d.max_sagitta_px,
        saturation=d.saturation,
        quality=d.quality(),
    )


def tag_obs(m: MarkerObservation) -> TagObs:
    return TagObs(m.tag_id, np.asarray(m.corners_ideal, dtype=float), np.asarray(m.cov, dtype=float), float(m.size_mm))


def detect_in_image(
    img: DecodedImage,
    cal: Calibration | None,
    sizes: TagSizeTable,
    cfg: MarkerConfig | None = None,
) -> tuple[list[TagDetection], CameraModel | None, str]:
    cam, note = camera_for_frame(cal, img.width, img.height, _upright(img)[0], _upright(img)[1], img.processing_scale)
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
        camera_label=str(client.get("camera_label", ""))[:120],
        lens=str(client.get("lens", img.meta.lens_model or ""))[:120],
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
