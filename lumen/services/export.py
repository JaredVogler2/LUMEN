"""Session summary export (prompt 11).

The export is a self-describing JSON document: coordinate semantics, the
operation snapshot with its content hash, tag sizes and their provenance, the
calibration profile, quality flags, gate results with the uncertainty budget,
verification results and the algorithm version.  It never contains image data;
photos stay on the server only until the session ends.
"""

from __future__ import annotations

from typing import Any

from .. import ALGORITHM_VERSION, RESEARCH_NOTICE, __version__
from ..schemas import SessionMode
from .context import utcnow
from .sessions import SessionService

COORDINATES = {
    "image": "Pixels of a frame as analysed (after EXIF orientation); pixel-centre convention, u right, v down.",
    "plane": "Millimetres on the target plane in the gauge tag's frame: +x to the tag's right, +y to its top, viewed from the front.",
    "local": "Engineering frame (x, z) in millimetres: z = 0 on the floor datum, x = 0 on the side datum; signs as configured in the operation record.",
    "tag_size": "Edge of the black square of an AprilTag 36h11 marker (boundary between black border and white margin).",
    "angles": "Radians internally; rotation is positive from +x toward +z.",
}


def build_export(svc: SessionService, sid: str) -> dict[str, Any]:
    s = svc.get(sid)
    reports: dict[str, Any] = {}
    if s.operation:
        op = svc._operation(s)
        for p in op.placards:
            try:
                reports[p.placard_id] = svc._gate_report(s, None, p.placard_id, True).model_dump(mode="json")
            except Exception as exc:  # never let one placard break the export
                reports[p.placard_id] = {"error": f"{type(exc).__name__}: {exc}"}
    frames = []
    for f in s.frames:
        frames.append(
            {
                "id": f.id,
                "index": f.index,
                "role": f.role,
                "source": f.source,
                "width": f.width,
                "height": f.height,
                "native_width": f.native_width,
                "native_height": f.native_height,
                "processing_scale": f.processing_scale,
                "exif_orientation": f.exif_orientation,
                "torch": f.torch,
                "sha256": f.sha256,
                "selected": f.selected,
                "quality": f.quality.model_dump(),
                "calibration_id": f.calibration_id,
                "calibration_note": f.calibration_note,
                "notes": f.notes,
                "markers": [
                    {
                        "tag_id": m.tag_id,
                        "accepted": m.accepted,
                        "reject_reasons": m.reject_reasons,
                        "flags": m.flags,
                        "corners_px": m.corners_px,
                        "corner_sigma_px": m.corner_sigma_px,
                        "size_mm": m.size_mm,
                        "size_provenance": m.size_provenance,
                        "edge_px": m.edge_px,
                        "sigma_blur_px": m.sigma_blur_px,
                        "bias_px": m.bias_px,
                        "bias_applied": m.bias_applied,
                    }
                    for m in f.markers
                ],
            }
        )
    return {
        "document": "LUMEN session summary",
        "exported_at": utcnow(),
        "notice": RESEARCH_NOTICE,
        "status_semantics": "Statuses describe measurement evidence under the stated gates. They are not acceptance decisions.",
        "simulated": s.mode == SessionMode.DEMO.value,
        "app_version": __version__,
        "algorithm_version": ALGORITHM_VERSION,
        "session": {"id": s.id, "mode": s.mode, "created": s.created, "updated": s.updated, "operator": s.operator},
        "coordinates": COORDINATES,
        "operation": s.operation,
        "active_placard_id": s.active_placard_id,
        "tag_sizes": s.tag_sizes,
        "calibration": None if s.calibration is None else s.calibration.model_dump(mode="json"),
        "frames": frames,
        "registration": s.registration,
        "datums": {k: v.model_dump(mode="json") for k, v in s.datums.items()},
        "local_frame": None if s.local_frame is None else s.local_frame.model_dump(mode="json"),
        "surface": s.surface.model_dump(mode="json"),
        "reports": reports,
        "verifications": [v.model_dump(mode="json") for v in s.verifications],
        "policy": svc.policy.to_dict(),
        "events": [e.model_dump(mode="json") for e in s.events],
        "images_included": False,
    }
