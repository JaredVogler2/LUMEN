# Conventions

One page of definitions used by every module. Code comments refer to this
file instead of repeating it.

## Units

| Quantity | Internal unit | Notes |
|---|---|---|
| Length | millimetre | Converted once, where a value enters (`lumen/units.py`). A length without a unit is refused, never assumed. |
| Angle | radian | Degrees appear only in the interface, in records and in report text. |
| Image position | pixel | See below. |
| Time | second | |

Display units (mm or inches) are a setting of the interface. Calculations and
stored results are always in millimetres.

## Pixels

OpenCV pixel-centre convention: integer `(u, v)` is the centre of the pixel
in column `u`, row `v`; `u` grows to the right and `v` downward; the image
covers `[-0.5, width - 0.5] × [-0.5, height - 0.5]`.

*Ideal* pixels are pixels after the lens model has been removed (what a
pinhole camera with the same matrix `K` would record). All plane mappings
(homographies) work on ideal pixels. Raw pixels are what the photo contains.

A photo is analysed upright (EXIF orientation applied once on decoding). When
a photo is reduced before analysis, `processing_scale` relates analysed to
native pixels and a lens profile is scaled with it; a profile is never applied
to a photo of another native size, orientation, camera, lens or zoom.

## Transforms

`T_a_b` maps coordinates in frame `b` to frame `a`: `x_a = R_a_b x_b + t_a_b`.
A homography named `H` maps plane millimetres to ideal pixels unless its name
says otherwise.

## AprilTags

* Family 36h11, IDs 0 to 586.
* **Tag size** is the edge of the black square (the outer edge of the black
  border). It is not the size of the sticker, the white margin or the number
  on the packaging. Declared sizes are "measured" (entered per tag, from a
  caliper) or "nominal" (one value for all, from the packaging); nominal sizes
  cannot reach the status `prototype_guidance` ("Ready").
* Corner order: top-left, top-right, bottom-right, bottom-left in the tag's
  own upright orientation (indices 0 to 3).
* A tag's frame: origin at its centre, `+x` to its right, `+y` to its top, in
  the plane of the tag.

## Frames on the panel

**Plane frame.** Millimetres on the target plane. It is the frame of one tag
of the registration, the *gauge tag*: origin at its centre, `+x` its right,
`+y` its top. It has no meaning for the operator; it only fixes the layout of
the tags.

**Local frame `(x, z)`.** The frame of the operation record:

* `z = 0` on the floor datum, `x = 0` on the side datum;
* the direction in which each coordinate grows is stated by the record
  ("toward the panel" or "away from the panel"), never taken from gravity, the
  phone's sensors or the top of the picture;
* how the two datum lines define the axes is the record's axis rule:
  `floor_primary` (x along the floor line, z perpendicular to it),
  `side_primary` (z along the side line, x perpendicular), or
  `perpendicular_offsets` (each coordinate is the perpendicular distance from
  its own datum; no right angle is assumed);
* the measured angle between the datum lines is always kept and compared with
  the record's nominal angle.

A datum has an optional configured offset from the feature that is observed
(along the positive axis), for cases where the datum is not the visible line
itself.

Which line of a feature is the datum:

* an edge between two surfaces (the foot of the panel on the floor): the edge;
* a seam, a joint or a scribed line up to 6 mm wide: its **centre line**,
  whether a photo shows it as a thin line or resolves it into its two edges.
  A record that means one edge of such a seam states the offset of half its
  width;
* anything wider: its two edges are two features, and the operator chooses.

**Placard rectangle.** Width along local `+x` at rotation 0, height along
`+z`. A positive rotation turns the width direction from `+x` toward `+z`
(whether that looks clockwise in a photo depends on which side the side datum
is on). The record's reference point (centre, a corner, or the middle of a
side) is the point the offsets locate. "Left" means lower `x` and "bottom"
lower `z` at rotation 0, whatever the photo looks like; corners are reported
top-left, top-right, bottom-right, bottom-left in that sense.

## Where a value comes from

Every stored value that matters carries one of these labels, and the
interface shows it:

| Label | Meaning |
|---|---|
| measured | computed from the pixels of this session by a measurement algorithm (a fitted edge, a decoded tag, a calibration from board photos) |
| inferred | estimated indirectly (a model fit, a focal length re-estimated from the tags) |
| manual | entered or drawn by the operator and not fitted to the image |
| configured | taken from the operation record |
| nominal | a catalogue or packaging value that was not measured |
| simulated | from computer-generated photos (the demo session, the benchmark) |
| unknown | not stated |

## Status of a result

| Status (API, export) | On the screen | Meaning |
|---|---|---|
| `blocked` | Cannot measure yet | A required condition is missing or contradicted. No target outline is drawn. |
| `research_preview` | Check needed | Geometry can be computed but a qualifying check is not met. The outline is drawn dashed and labelled. |
| `prototype_guidance` | Ready | Every computed check passed. The checks are backed by synthetic tests only; on an installation without a validated domain the screen adds "Evaluation only: not validated". |
| `validated` | Ready · validated | As above and inside a physically validated domain supplied by an administrator. None ships with LUMEN, so this status cannot occur. |

The status values are the interface between server, export and tests and do
not change with the wording of the screen.

What kind of results an installation gives is shown by one indicator in the
header of every screen, by the watermark on every picture with a result, and
beside the status:

| Indicator | When | Watermark |
|---|---|---|
| Demo | the session runs on computer-generated photos (`mode: demo`) | DEMO · SIMULATED DATA |
| Evaluation | real photos, no validated domain supplied (every installation today) | EVALUATION ONLY · NOT ACCEPTANCE |
| Validated | `LUMEN_VALIDATED_DOMAIN_FILE` names a validated domain | MEASUREMENT EVIDENCE · NOT ACCEPTANCE |

Steps of the workflow and their names on the screen: `identify` Job,
`markers` Tags, `capture` Photos, `datums` Lines, `place` Place, `verify`
Records. `rectified` (the straight-on view with the scale check) is a tool
opened from Lines and has no stop of its own.

Each check in a report has a `severity`: `blocking` (failing it gives
`blocked`), `warning` (failing it gives `research_preview`) or `info`. A
check with `caveat: true` passed but states something that was **not**
checked, for example that the scale rests on the declared tag sizes alone or
that a datum is on the face by the record's word only. Caveats do not change
the status; the interface shows them with an "i" instead of a tick.

## Uncertainty

For a target position the report gives:

* `u_random_mm`: repeatability (one standard deviation) propagated from the
  pixel noise of this capture;
* `bias_bound_mm`: the combination of systematic bounds (tag size, residual
  edge shift, lens model, panel flatness, edge definition), each treated as a
  rectangular distribution; the list of terms is part of the report;
* `expanded_mm = k × sqrt(u_random² + Σ bound_i² / 3)` with `k = 2`.

No probability is attached to the expanded value. How often it contained the
true error has been examined on synthetic data only
(`docs/benchmark/RESULTS.md`).

## A placard in a live frame

What the live placement answers for one frame (`placement` in the answer of
`POST …/track?placement=true`):

**Deviation.** `dx_mm`, `dz_mm`, `rotation_deg`: measured placard minus
target, in the local frame, with the same signs as a verification.
`move.x_mm`, `move.z_mm` are their negatives. `move.px` is the vector from
the placard's centre to the target's centre in the pixels of that frame
(`u` right, `v` down), and `move.turn_screen_deg` is positive when the placard
has to be turned clockwise as seen on the screen. The arrow, the dial and the
level on the screen use these two, so that "left" on the screen is left in
the picture whatever the orientation of the record's axes.

**In position.** `in_position` is true when, for this frame, the distance
from the target plus the expanded uncertainty is inside the position
tolerance of the record, and the rotation plus its expanded uncertainty is
inside the rotation tolerance. It is the same rule as "within" in a
verification. Without a tolerance in the record nothing is judged.
`room_mm` is tolerance minus expanded uncertainty: how far from its target
the placard may measure and still be in position.

**What is particular to a placard that is still being handled:**

| Field or flag | Meaning |
|---|---|
| `not_flat_on_surface` | the outline measures larger (or smaller) than the record's size by more than its measurement allows. A placard held off the panel is nearer to the camera and looks larger. Nothing is judged |
| budget term "placard not known to lie on the surface (live)" | a height too small to show in the size still displaces the outline when seen at an angle. Its bound: the largest height the size test would miss, times the tangent of the viewing angle (plus 5° when the stream has no lens profile) |
| `sides` | per side (top, right, bottom, left): `measured`, `record_size` (hidden; placed from the opposite side and the record's size), `hidden` |
| budget term "hidden side placed by the size in the record (live)" | a part may differ from the record's size; 1.0 mm is allowed for, of which half moves the centre. An assumption of the guidance only |
| `size_not_seen` | no two opposite sides were measured, so nothing shows whether the placard lies on the surface. Not judged |
| `hands` | part of the outline is covered (a partly hidden side, a corner not seen, a side placed by size) |
| `tracking.held` | the phone stands still: no tag was measured in this frame. The registration is that of an earlier frame, checked on the tags that can be seen (`tags_seen`; `tags_covered` could not be compared). `shift_px` is the largest shift found; with `moved_along` the registration was carried along a shift of up to 0.5 px on which at least three tags agree |
| budget term "registration held from an earlier frame of the standing phone (live)" | what that check cannot see: 0.1 px, or 0.06 px when the registration was moved along |
| `live_terms` | the terms of this reading's budget that a stored photo would not have (lying flat, a side placed by size, a held registration), with their bounds |
| `clean` | all four sides measured, nothing covered, no doubt about the outline. Required for the automatic record |

A side counts as measured in a live frame when at least 30 % of it lies on
one straight edge (55 % in a verification photo). A side is placed by the
record's size only where the picture shows something covering it; if the
place where it would be looks like bare panel or bare placard, it is not
placed, and three sides with a fourth that is nowhere point to another part.
A straight edge that has something other than the panel outside it (a hand,
a tool, or the placard's own surface, as at the ends of a block of print) is
not taken for a side.

**The record.** Nothing of the above is stored. The record of a placement is
a verification: a stored photo at the camera's full size, measured with all
four sides, without any of the assumptions above.

## Identifiers

| Thing | Form |
|---|---|
| Session | random URL-safe token; knowing it is what gives access |
| Frame | `f` + three-digit counter within the session (`f001`) |
| Operation record | `operation_id` + `revision`; content identified by a SHA-256 hash that the session stores with its own copy |
| Lens profile | `id` generated at calibration; stored in the browser of the phone it belongs to |
