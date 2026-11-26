"""Assisted placement: the placard measured in a live frame while it is being positioned.

The operator holds the placard near the target; every live frame is registered
against the tags (see ``live.py``) and the placard's outline is measured in that
same frame.  What comes back is how far the placard is from where the record
puts it, which way to move it, and whether it can be called "in position".

What "in position" means here
-----------------------------
The same rule as for a verification photo: the measured deviation plus the
expanded uncertainty of this frame lies inside the record's tolerance.  Three
things are particular to a placard that is still in a hand:

* **It may not lie on the surface.**  A placard held a few millimetres off the
  panel looks displaced in the picture by that height times the tangent of the
  viewing angle.  Height shows as size: the placard measures too large.  A
  placard that measures too large (or too small) is not judged at all, and the
  height that the size test cannot exclude enters the uncertainty as a bound.
  The bound grows with the viewing angle, so an oblique view cannot give "in
  position" for a tight tolerance, and the answer says so.
* **A hand covers part of it.**  A side that is partly hidden is still a line
  when enough of it is seen (a third, here); a covered corner is the
  intersection of its two sides.  A side that is hidden altogether is placed
  from the opposite, measured side and the placard's size in the record
  (``complete_from_size``): an assumption, which costs uncertainty (the part
  may not have exactly that size, and its size can then be checked in one
  direction only).  All of that is accepted while positioning.  The result is
  called *clean* only when the whole outline is measured, which is what
  recording asks for.
* **Nothing is stored.**  This is guidance.  The record of the placement is a
  verification of a stored photo, made by the ordinary verification path.

No tolerance in the record means no judgement: the deviation is shown and
nothing is called right or wrong.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import RESEARCH_NOTICE
from ..errors import LumenError
from ..geometry.homography import apply_h, local_scale_px_per_mm, pose_from_homography
from ..policy.gates import _largest_component_hint
from ..schemas import PlacementSpec, Uncertainty, Verification
from ..units import wrap_angle
from ..vision.placard import SIDE_NAMES, PlacardMeasurement, _corners_from_sides, _rect_corners, complete_from_size, foreign_sides, measure_placard
from .context import utcnow
from .verification import DOUBTFUL_OUTLINE, _evaluate

if TYPE_CHECKING:  # pragma: no cover
    from .sessions import SessionService

#: while a hand holds the placard, a partly hidden side, a covered corner or a side placed by the record's size does
#: not stand in the way of "in position"
HAND_FLAGS = ("edge_partly_hidden", "corner_lifted_or_hidden", "corner_not_directly_seen", "side_from_record_size")
LIVE_DOUBTFUL = tuple(f for f in DOUBTFUL_OUTLINE if f not in HAND_FLAGS) + ("not_flat_on_surface", "size_not_seen")
#: what in a measurement says that something covers part of the placard
HAND_SIGNS = ("edge_partly_hidden", "corner_lifted_or_hidden", "corner_not_directly_seen", "side_not_measured", "partial_perimeter")
#: a side counts as measured in a live frame when this fraction of it lies on one straight edge (a hand covers the rest)
LIVE_MIN_SIDE_SUPPORT = 0.3
#: how far the real size of a part may differ from the record's, where a hidden side is placed by it (mm).  An
#: assumption of the live guidance only; a record measures all four sides.
RECORD_SIZE_ALLOWANCE_MM = 1.0
SIZE_TERM = "hidden side placed by the size in the record (live)"
HELD_TERM = "registration held from an earlier frame of the standing phone (live)"
#: margin on the size test beyond the edge-definition bound (mm): fit noise of two sides
SIZE_MARGIN_MM = 0.15
#: added to the viewing angle when the camera of the live stream is not calibrated (degrees)
UNCALIBRATED_ANGLE_MARGIN_DEG = 5.0
#: name of the budget term for a placard that may not lie on the surface yet
LIFT_TERM = "placard not known to lie on the surface (live)"
#: "in position" is called hard to reach when the uncertainty leaves less than this share of the tolerance (and less than TIGHT_MM)
TIGHT_FRACTION = 0.3
TIGHT_MM = 0.6


def _view_geometry(loc: Any, H: np.ndarray, width: int, height: int, point_plane: np.ndarray) -> tuple[float | None, float | None, bool]:
    """Angle between the line of sight to ``point_plane`` and the panel's normal (degrees), the distance (mm), and
    whether the camera's geometry is known (calibrated stream) or assumed (a phone's main camera)."""
    pose, known = loc.pose, loc.pose is not None
    if pose is None:
        f = 0.75 * max(width, height)  # horizontal field of view of about 67 degrees
        K = np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
        try:
            pose = pose_from_homography(K, H)
        except (np.linalg.LinAlgError, ValueError):
            return None, None, False
    c = -np.asarray(pose.R).T @ np.asarray(pose.t)  # camera centre in plane coordinates
    v = c - np.array([point_plane[0], point_plane[1], 0.0])
    dist = float(np.linalg.norm(v))
    if not np.isfinite(dist) or dist < 1.0:
        return None, None, known
    return float(np.degrees(np.arccos(min(abs(float(v[2])) / dist, 1.0)))), dist, known


def _hint(meta: dict[str, Any]) -> tuple[np.ndarray | None, int]:
    """Where the placard was in the previous frame (plane millimetres), echoed back by the client."""
    h = meta.get("placement_hint") if isinstance(meta, dict) else None
    if not isinstance(h, dict):
        return None, 0
    try:
        pts = np.asarray(h.get("corners_plane"), dtype=float)
        pol = int(h.get("polarity") or 0)
    except (TypeError, ValueError):
        return None, 0
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)) or float(np.max(np.abs(pts))) > 1e5:
        return None, 0
    return pts, int(np.sign(pol))


def live_placement(
    svc: "SessionService", s: Any, frame: Any, bgr: np.ndarray, cam: Any, loc: Any, rep: dict[str, Any] | None, pid: str | None, meta: dict[str, Any],
    held_px: float | None = None,
) -> dict[str, Any]:
    """Measure the placard in this live frame and say where it is relative to its target.

    ``held_px``: the registration ``loc`` is that of an earlier frame of a phone that stands still, and the tags
    were found within this many pixels of where they were then (``live.py``).  What that check cannot see is a
    term of the reading's budget.
    """
    out: dict[str, Any] = {"placard_id": pid, "found": False, "state": "no_target", "in_position": False, "clean": False, "notice": RESEARCH_NOTICE}
    lf = svc._local_frame(s)
    if not pid or not s.operation or lf is None or not lf.valid:
        out["message"] = "There is no target yet: choose the placard and confirm the two lines first."
        return out
    try:
        spec = PlacementSpec(**svc._operation(s).placard(pid).resolved())
    except LumenError:
        out["message"] = "The placard is not defined in the record."
        return out
    if not svc._spec_complete(spec):
        out["message"] = "The record does not define this placard completely."
        return out
    if rep is None or rep.get("status") == "blocked":
        out["state"] = "blocked"
        out["message"] = (rep or {}).get("reasons", ["The target cannot be established in this view."])[0] if rep else "The target cannot be established in this view."
        return out
    if "high_residual" in (loc.flags or []):
        out["state"] = "blocked"
        out["message"] = "A registered tag appears to have moved: this view is not used."
        return out

    rect = svc._target(lf, spec)
    H = np.asarray(loc.H, dtype=float)
    ppm = float(np.median(local_scale_px_per_mm(H, rect.corners)))
    tgt_ideal = apply_h(H, rect.corners)
    tgt_px = cam.distort_pixels(tgt_ideal) if cam is not None else tgt_ideal
    out["target_px"] = np.asarray(tgt_px, dtype=float).round(2).tolist()
    out["tolerance_mm"] = spec.tol_position_mm
    out["tolerance_rotation_deg"] = None if spec.tol_rotation_rad is None else float(np.degrees(spec.tol_rotation_rad))
    out["px_per_mm"] = round(ppm, 4)
    # what this view is worth, whether or not a placard is in it: the status and the uncertainty of the target itself
    out["status"] = rep.get("status")
    out["target_expanded_mm"] = None if not rep.get("uncertainty") else float(rep["uncertainty"]["expanded_mm"])

    # ---- find it: near where it was a moment ago, else by searching around the target
    cfg = replace(svc._placard_cfg, min_side_support=LIVE_MIN_SIDE_SUPPORT, wide_side_retry=False)
    seed_plane, pol = _hint(meta)
    if seed_plane is not None and float(np.linalg.norm(seed_plane.mean(axis=0) - rect.center)) > cfg.search_mm + 20.0:
        # further from this target than a search would ever look: it is not this placard (another one that was
        # followed before, or a stale answer after the target changed)
        seed_plane = None

    def measure(seed: np.ndarray | None, seed_pol: int, was_whole: bool) -> PlacardMeasurement:
        kw: dict[str, Any] = {} if seed is None else {"seed_corners_ideal": apply_h(H, seed), "seed_polarity": seed_pol}
        got = measure_placard(bgr, cam, H, rect.corners, cfg=cfg, **kw)
        if len(got.sides) == 4 and any(f.startswith(HAND_SIGNS) for f in got.flags):
            # Something covers part of the placard.  A straight edge of that something (a hand, a tool) can pass
            # for a side: a side that does not have the panel outside and the placard inside, like the others,
            # is taken out and treated as hidden.
            for k in foreign_sides(bgr, cam, got.sides):
                sd = got.sides[k]
                sd.ok, sd.reason = False, "edge_of_something_else"
                got.flags = [f for f in got.flags if not f.endswith(f":{sd.name}")] + [f"side_not_measured:{sd.name}"]
                got.proposal.setdefault("foreign_sides", []).append(SIDE_NAMES[k])
            if got.proposal.get("foreign_sides"):
                got.corners_ideal, got.corners_cov = _corners_from_sides(got.sides)
                n_left = sum(sd.ok for sd in got.sides)
                got.ok = bool(got.ok and n_left >= 3)
                if not got.ok and not got.reason:
                    got.reason = "too_few_sides_measured"
                if n_left == 3 and "partial_perimeter" not in got.flags:
                    got.flags.append("partial_perimeter")
        if not got.complete and sum(sd.ok for sd in got.sides) >= 2:
            # a hand hides a side: place it by the placard's size, where the picture shows something covering it.
            # Two hidden sides only for a placard that was seen whole a moment ago.
            done = complete_from_size(got, H, rect.width, rect.height, allow_two=was_whole, image=bgr, camera=cam)
            if done and any(v is not None and v > 0.5 for v in (got.proposal.get("placed_side_edge") or {}).values()):
                # A side that was placed shows the placard's own edge: it is not covered, it was only not found
                # where it was looked for (the placard moved further between two frames than a side is followed).
                # Measured from where the outline now is, it is a side like the others.
                sign = int(got.proposal.get("edge_polarity") or seed_pol or 0)
                again = measure_placard(bgr, cam, H, rect.corners, cfg=cfg, seed_corners_ideal=got.corners_ideal, seed_polarity=sign)
                if again.complete and sum(sd.ok for sd in again.sides) > sum(1 for sd in got.sides if sd.ok and sd.method != "record_size"):
                    again.proposal["measured_again_from_placed_outline"] = True
                    return again
            plain = {k: v for k, v in (got.proposal.get("placed_side_plain") or {}).items() if v is not None and v > 0.5}
            if not done and plain and sum(sd.ok for sd in got.sides) == 3:
                # three sides are there and the fourth is not where this size puts it, with nothing over it
                got.flags.append("not_the_recorded_size:" + sorted(plain)[0])
        return got

    def usable(g: PlacardMeasurement) -> bool:
        return bool(g.ok and g.complete and not any(f.startswith("size_differs_from_record") for f in g.flags))

    def n_measured(g: PlacardMeasurement) -> int:
        return sum(1 for sd in g.sides if sd.ok and sd.method != "record_size")

    m, method = None, "searched"
    if seed_plane is not None:
        m, method = measure(seed_plane, pol, True), "followed"
        if not usable(m):
            m = None  # it moved further than a side scan reaches, or the seed led onto print inside it: search
    if m is None:
        m, method = measure(None, 0, False), "searched"
        rivals = [a for a in (m.proposal.get("alternatives") or []) if float(a.get("ratio", 0.0)) >= 0.5][:3]
        if rivals and (not usable(m) or n_measured(m) < 4 or "ambiguous_proposal" in m.flags):
            # The search found other places that look like a rectangle of this size (a hand's straight edge
            # will do).  Each is measured the same way; a place is the placard only if its outline is complete.
            E = np.asarray(rect.corners, dtype=float)
            ex, ey = (E[1] - E[0]) / rect.width, (E[0] - E[3]) / rect.height
            others, tried = [], []
            for a in rivals:
                c = E.mean(axis=0) + float(a["offset_mm"][0]) * ex + float(a["offset_mm"][1]) * ey
                g = measure(_rect_corners(c, rect.width, rect.height, float(np.radians(a["angle_deg"])), ex, ey), 0, False)
                tried.append(g)
                if usable(g):
                    others.append(g)
            Hinv = np.linalg.inv(H)

            def same_place(g1: PlacardMeasurement, g2: PlacardMeasurement) -> bool:
                """Two measurements of one and the same outline (a search result a few millimetres beside
                another one locks onto the same edges)."""
                return float(np.max(np.linalg.norm(apply_h(Hinv, g1.corners_ideal) - apply_h(Hinv, g2.corners_ideal), axis=1))) < 3.0

            first_ok = usable(m)
            places: list[PlacardMeasurement] = []  # the distinct other places that hold up, best measured first
            for g in sorted(others, key=n_measured, reverse=True):
                if not (first_ok and same_place(g, m)) and not any(same_place(g, h) for h in places):
                    places.append(g)
            if first_ok and not places:
                # no other place holds up: what looked like a second candidate is not one
                m.flags = [f for f in m.flags if f != "ambiguous_proposal"]
                m.proposal["rivals_rejected"] = len(rivals)
            elif places and (not first_ok or n_measured(places[0]) > n_measured(m)):
                if len(places) > 1 and n_measured(places[1]) == n_measured(places[0]) and "ambiguous_proposal" not in places[0].flags:
                    places[0].flags.append("ambiguous_proposal")  # two other places hold up equally: not decided
                m = places[0]
                m.proposal["chosen_from_rivals"] = True
            elif places:
                if "ambiguous_proposal" not in m.flags:
                    m.flags.append("ambiguous_proposal")  # another place holds up as well as this one
            elif not _opposite_fit(m):
                # nothing can be completed.  A place where two opposite sides are the record's distance apart is
                # still the placard, with its other two sides covered: say that, not "nothing found"
                m = next((g for g in tried if _opposite_fit(g)), m)
    out["method"] = method
    if not m.ok:
        out["state"] = "searching"
        out["reason"] = m.reason
        seen = [SIDE_NAMES[sd.index] for sd in m.sides if sd.ok]
        out["message"] = {
            "expected_area_outside_image": "The target is not inside the picture: aim at it.",
            "too_few_sides_measured": _covered(seen, m.flags),
        }.get(m.reason, "Bring the placard to the dashed outline.")
        return out

    base = Verification(id="live", frame_id="live", placard_id=pid, created=utcnow(), notice=RESEARCH_NOTICE, registration_flags=list(loc.flags or []), observation="no_evidence")
    detail: dict[str, Any] = {"px_per_mm": ppm}
    flags = list(m.flags)
    if rep.get("status") == "research_preview":
        flags.append("registration_preview_only")

    # ---- is it lying on the surface?  Height shows as size, in the directions in which both sides were measured.
    C = m.corners_ideal
    size_bound = 2.0 * svc._placard_cfg.edge_definition_bound_px / ppm
    measured = [bool(sd.ok and sd.method != "record_size") for sd in m.sides]
    width_seen, height_seen = bool(measured[1] and measured[3]), bool(measured[0] and measured[2])
    size_err: list[float] = []
    extra: list[tuple[str, float, str]] = []
    if m.complete:
        P = apply_h(np.linalg.inv(H), C)
        if width_seen:
            size_err += [float(np.linalg.norm(P[1] - P[0]) - rect.width), float(np.linalg.norm(P[2] - P[3]) - rect.width)]
        if height_seen:
            size_err += [float(np.linalg.norm(P[0] - P[3]) - rect.height), float(np.linalg.norm(P[1] - P[2]) - rect.height)]
        if size_err and max(abs(e) for e in size_err) > size_bound + SIZE_MARGIN_MM:
            flags.append("not_flat_on_surface")
        if not size_err:
            flags.append("size_not_seen")  # two neighbouring sides only: nothing says whether it lies on the surface
        n_assumed = int(not width_seen) + int(not height_seen)
        if n_assumed:
            b = 0.5 * RECORD_SIZE_ALLOWANCE_MM * float(np.sqrt(n_assumed))
            extra.append((SIZE_TERM, b, f"the part may differ from the record's size by up to {RECORD_SIZE_ALLOWANCE_MM:.1f} mm; half of that moves its centre"))
    if held_px is not None:
        extra.append((HELD_TERM, float(held_px) / ppm, f"the tags were where they were in the frame that was measured, to within {held_px:g} px"))
    angle, dist, known = _view_geometry(loc, H, int(frame.width), int(frame.height), rect.center)
    lift_bound = None
    seen_dims = [d for d, ok in ((rect.width, width_seen), (rect.height, height_seen)) if ok]
    if angle is not None and dist is not None and seen_dims:
        lift_bound = (size_bound + SIZE_MARGIN_MM) * dist / max(seen_dims)
        eff = min(angle + (0.0 if known else UNCALIBRATED_ANGLE_MARGIN_DEG), 80.0)
        extra.append((
            LIFT_TERM, float(lift_bound * np.tan(np.radians(eff))),
            f"a height of up to {lift_bound:.1f} mm does not show in its size; seen {eff:.0f} degrees off the normal it displaces the outline by this much",
        ))
    base_unc = Uncertainty.model_validate(rep["uncertainty"]) if rep.get("uncertainty") else None
    with_msg = _evaluate(svc, base, detail, m, H=H, cam=cam, lf=lf, rect=rect, spec=spec, base_unc=base_unc, ppm=ppm, manual=False, flags=flags, extra_bounds=extra, doubtful=LIVE_DOUBTFUL)
    base.flags = flags

    hands = any(f.startswith(HAND_FLAGS + ("side_not_measured", "partial_perimeter")) for f in flags)
    clean = bool(m.complete and all(measured) and not any(f.startswith(DOUBTFUL_OUTLINE + LIVE_DOUBTFUL + HAND_FLAGS + ("side_not_measured", "partial_perimeter")) for f in flags))
    U = None if base.uncertainty is None else float(base.uncertainty.expanded_mm)
    rot_U = detail.get("rotation_expanded_rad")
    tol, tol_rot = spec.tol_position_mm, spec.tol_rotation_rad
    reachable = None if (tol is None or U is None) else bool(U < tol and (tol_rot is None or rot_U is None or rot_U < tol_rot))
    # how far from its target the placard may be measured and still be called in position, and what limits that
    room = None if (tol is None or U is None) else float(tol - U)
    limit_hint = None
    if base.uncertainty is not None and base.uncertainty.components:
        top = max(base.uncertainty.components, key=lambda c: c.variance_mm2)
        limit_hint = (
            "Largest contribution: the placard may not lie flat yet and is seen at an angle. Hold the phone square to the placard."
            if top.name == LIFT_TERM else _largest_component_hint(base.uncertainty)
        )
    out.update({
        "found": True, "state": "found", "complete": bool(m.complete),
        "sides": [("measured" if ok else ("record_size" if sd.ok else "hidden")) for sd, ok in zip(m.sides, measured)],
        "corners_px": base.corners_px, "corners_plane": base.corners_plane, "polarity": int(m.proposal.get("edge_polarity") or pol or 0),
        "dx_mm": base.dx_mm, "dz_mm": base.dz_mm, "radial_mm": base.radial_mm,
        "rotation_deg": None if base.rotation_rad is None else float(np.degrees(base.rotation_rad)),
        "width_error_mm": base.width_error_mm, "height_error_mm": base.height_error_mm,
        "size_error_mm": [round(e, 3) for e in size_err], "size_bound_mm": round(size_bound + SIZE_MARGIN_MM, 3),
        "expanded_mm": U, "rotation_expanded_deg": None if rot_U is None else float(np.degrees(rot_U)),
        "observation": base.observation, "position_observation": detail.get("position_observation"), "rotation_observation": detail.get("rotation_observation"),
        "reachable": reachable, "in_position": base.observation == "within_tolerance_observed",
        "room_mm": None if room is None else round(room, 3),
        # the terms of this reading's budget that a stored photo would not have
        "live_terms": [{"name": name, "bound_mm": round(float(b), 4)} for name, b, _ in extra],
        "tight": bool(room is not None and 0.0 < room < min(TIGHT_FRACTION * tol, TIGHT_MM)), "limit_hint": limit_hint,
        "hands": hands, "clean": clean, "flags": flags, "doubtful": [f for f in flags if f.startswith(LIVE_DOUBTFUL)],
        "view_angle_deg": None if angle is None else round(angle, 1), "distance_mm": None if dist is None else round(dist, 0),
        "view_assumed": not known, "lift_bound_mm": None if lift_bound is None else round(float(lift_bound), 2),
        "status": rep.get("status"),
    })

    # ---- which way to move it, on the screen and in the record's axes
    if m.complete:
        raw = np.asarray(base.corners_px, dtype=float)
        tp = np.asarray(tgt_px, dtype=float)
        d = tp.mean(axis=0) - raw.mean(axis=0)
        a_t = np.arctan2(*(tp[1] - tp[0])[::-1])
        a_m = np.arctan2(*(raw[1] - raw[0])[::-1])
        out["move"] = {
            "x_mm": None if base.dx_mm is None else -base.dx_mm, "z_mm": None if base.dz_mm is None else -base.dz_mm,
            "px": [float(d[0]), float(d[1])],
            # positive: turn it clockwise as seen on the screen (image rows grow downward)
            "turn_screen_deg": float(np.degrees(wrap_angle(a_t - a_m))),
        }
    out["message"] = _message(out, with_msg)
    return out


def _opposite_fit(g: PlacardMeasurement) -> bool:
    """Two opposite sides were measured, the record's distance apart (and that is all there is of the outline)."""
    sign = g.proposal.get("edge_polarity")
    tried = next((t for t in g.proposal.get("polarities_tried") or [] if t.get("polarity") == sign), None)
    mismatch = None if tried is None else tried.get("opposite_sides_mismatch")
    return bool(mismatch is not None and mismatch <= 0.03 and sum(sd.ok for sd in g.sides) == 2)


def _covered(seen: list[str], flags: list[str] | None = None) -> str:
    """What to uncover when the outline cannot be completed (``seen``: the sides that were measured)."""
    odd = next((f.split(":")[1] for f in flags or [] if f.startswith("not_the_recorded_size:")), None)
    if odd:
        return f"The {odd} side is not where a placard of the recorded size has it, and nothing covers it: is this the right part?"
    hidden = [n for n in SIDE_NAMES if n not in seen]
    if set(hidden) in ({"left", "right"}, {"top", "bottom"}):
        return f"The {hidden[0]} and {hidden[1]} sides are both covered: uncover one of them."
    return "Too little of the placard's outline is visible: uncover more of it."


def _message(p: dict[str, Any], fallback: str) -> str:
    if not p.get("complete"):
        return _covered([n for n, st in zip(SIDE_NAMES, p.get("sides") or []) if st != "hidden"], p.get("flags"))
    if any(f.startswith("not_flat_on_surface") for f in p["flags"]):
        e = max(p.get("size_error_mm") or [0.0], key=abs)
        if e > 0:
            return f"The placard measures {abs(e):.1f} mm too large: it is not lying on the surface yet."
        return f"The outline measures {abs(e):.1f} mm smaller than the placard: something covers a side, or it is another part."
    tol, U = p.get("tolerance_mm"), p.get("expanded_mm")
    if tol is None:
        return "The record gives no tolerance: the deviation is shown, nothing is judged."
    if p.get("reachable") is False:
        ang = p.get("view_angle_deg")
        tip = p.get("limit_hint") or ("Hold the phone square to the placard." if (ang is not None and ang > 12.0) else "Move closer.")
        return f"From here the reading is good to ±{U:.1f} mm, too coarse for ±{tol:.1f} mm. {tip}"
    if p.get("in_position"):
        return "In position. Let go to record." if p.get("hands") else "In position."
    if p.get("doubtful") and p.get("position_observation") == "within" and p.get("rotation_observation") in (None, "within"):
        # the numbers are inside the tolerance, but they come from an outline that is not trusted
        if "size_not_seen" in p["doubtful"]:
            return "Too much of the placard is covered to judge its position: uncover one more side."
        return "The outline is not clear enough here to judge the position."
    parts = []
    if p.get("position_observation") != "within" and p.get("radial_mm") is not None:
        parts.append(f"move {p['radial_mm']:.1f} mm")
    if p.get("rotation_observation") not in (None, "within") and p.get("rotation_deg") is not None:
        parts.append(f"turn {abs(p['rotation_deg']):.1f}°")
    if not parts:
        return "Close: hold still."
    return (", ".join(parts)).capitalize() + "."
