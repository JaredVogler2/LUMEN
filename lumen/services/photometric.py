"""Research-only use of the photometric (phone torch) analysis (prompt 07).

Nothing computed here feeds a gate, a datum or a target.  The result is kept
under ``session.research["photometric"]`` and shown as an experiment next to
the estimate of the outline (silhouette) method, so the two can be compared.
``docs/PHOTOMETRIC_ABLATION.md`` says where the method helps, where it does
not, and where it was wrong without noticing on synthetic photos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ConflictError, FeatureDisabledError
from ..geometry.camera import CameraModel
from ..geometry.lines import Line2D
from ..schemas import Session
from ..vision.photometric import EVIDENCE_LABEL, EdgeHint, FlashView, analyse_flash_sweep, combine_with_silhouette, tag_exclusion_polygons
from .context import utcnow

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

#: cross-sections kept in the stored result (each is a few hundred numbers)
MAX_PROFILES = 4
MAX_EXTENT_MM = 225.0


def photometric_research(svc: "SessionService", sid: str) -> dict[str, Any]:
    """Run the torch-shading analysis on the registered photos and store it as research evidence."""
    if not svc.settings.photometric_enabled:
        raise FeatureDisabledError("photometric_disabled", "The flashlight experiment is switched off on this deployment.")
    s = svc.get(sid)
    sol, _views = svc._require_registration(s)
    mv = (s.registration or {}).get("multiview")
    if not mv or "skipped" in mv:
        raise ConflictError(
            "needs_camera_positions", "The flashlight experiment needs several registered photos from different places.",
            suggestions=["Take three or more photos with the light on while moving sideways around the rounded edge, then analyse again."],
        )
    side = s.datums.get("side")
    if side is None or len(side.endpoints) != 2:
        raise ConflictError("no_side_datum", "Set the side datum first: the experiment looks at the shading next to it.", suggestions=["Search for the datum lines on the Lines step."])

    frames = {f.id: f for f in s.frames}
    ctxs = svc._frame_contexts(sid, s, sol, mv)
    intr = mv["intrinsics"]
    views: list[FlashView] = []
    index: dict[str, int] = {}
    for fid in sorted(ctxs):
        fc, f = ctxs[fid], frames[fid]
        if fc.pose is None:
            continue
        cam = fc.camera or CameraModel(f.width, f.height, intr["fx"], intr["fy"], intr["cx"], intr["cy"], tuple(intr.get("dist", (0.0,) * 5)))
        index[fid] = len(views)
        torch = f.torch.value if hasattr(f.torch, "value") else str(f.torch)
        views.append(FlashView(svc._image(sid, f).bgr, cam, fc.pose, torch, None, fid, fc.tag_blur_px))
    # flash-off partners: frames that share a pair id
    by_pair: dict[str, list[str]] = {}
    for fid in index:
        if frames[fid].pair_id:
            by_pair.setdefault(frames[fid].pair_id, []).append(fid)
    for fids in by_pair.values():
        if len(fids) == 2:
            a, b = index[fids[0]], index[fids[1]]
            views[a].pair, views[b].pair = b, a

    ends = np.asarray(side.endpoints, dtype=float)
    mid = ends.mean(axis=0)
    direction = (ends[1] - ends[0]) / max(float(np.linalg.norm(ends[1] - ends[0])), 1e-9)
    interior = np.mean([sol.layout.corners(t).mean(axis=0) for t in sol.layout.poses], axis=0)
    inward = np.array([-direction[1], direction[0]])
    if inward @ (interior - mid) < 0:
        inward = -inward
    half = min(0.5 * float(np.linalg.norm(ends[1] - ends[0])), MAX_EXTENT_MM)
    ddef = svc._datum_def(s, "side")
    radius = None if (ddef is None or ddef.radius is None) else float(ddef.radius.mm)
    hint = EdgeHint(point=(float(mid[0]), float(mid[1])), direction=(float(direction[0]), float(direction[1])), inward=(float(inward[0]), float(inward[1])), radius_mm=radius, extent_mm=(-half, half))
    polygons = tag_exclusion_polygons([sol.layout.corners(t) for t in sol.layout.poses])

    with svc.cpu.slot(svc.settings.limits.job_timeout_seconds):
        res = analyse_flash_sweep(views, hint, exclude_polygons=polygons)
    out = res.to_dict()
    out["profiles"] = out.get("profiles", [])[:MAX_PROFILES]

    # how far the estimate is from the side datum in use (positive = toward the flat face)
    datum_line = Line2D(np.asarray(side.n, dtype=float), float(side.d)).oriented_towards(interior)
    offset = None
    comparison = None
    if res.tangent is not None:
        t = res.tangent.oriented_towards(interior)
        offset = float(datum_line.signed_distance(mid) - t.signed_distance(mid))
        if side.feature_kind == "cylinder_model" and side.sigma_perp_mm:
            fit = side.image_fit or {}
            comparison = combine_with_silhouette(
                res, datum_line, float(side.sigma_perp_mm), float(np.degrees(side.sigma_angle_rad or 0.0)),
                radius_mm=fit.get("radius_mm"), sigma_radius_mm=None, reference_point=mid,
            )
    record = {
        "label": EVIDENCE_LABEL,
        "guidance_enabled": False,
        "computed_at": utcnow(),
        "n_photos": len(views),
        "n_torch_on": sum(1 for v in views if v.torch == "on"),
        "side_datum": {"feature_kind": side.feature_kind, "provenance": str(side.provenance.value if hasattr(side.provenance, "value") else side.provenance)},
        "offset_from_side_datum_mm": None if offset is None else round(offset, 3),
        "comparison_with_outline_method": comparison,
        "result": out,
        "notice": "Research evidence only. This result is not used for any datum, check or target.",
    }

    def fn(x: Session) -> Session:
        x.research = {**(x.research or {}), "photometric": record}
        return x

    svc._mutate(sid, fn, "photometric_research", {"status": res.status, "n_torch_on": record["n_torch_on"]})
    return record
