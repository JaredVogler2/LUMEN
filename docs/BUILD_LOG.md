# LUMEN build log

Chronological record of what was built, what was measured and what did not
work. Every number in this file was produced by running the code in this
repository on the date shown; nothing here is a physical measurement.
Physical accuracy on real panels, real iPhones and a real Posit Connect server
is **NOT RUN** unless a section says otherwise.

Conventions: lengths in mm, angles in degrees unless stated, pixel errors in
pixels of the analysed image. "Synthetic" means images from `lumen/synth`
(independent ray caster, linear-light pipeline, sRGB encoding).

---

## 2026-11-25 — Environment

| Item | Value |
|---|---|
| Python used for development | 3.12 (venv); 3.11 and 3.13 also present on the build host |
| OpenCV | `opencv-python-headless==4.14.0.94` (manylinux_2_28 wheel, no system OpenCV) |
| AprilTag dictionary | `cv2.aruco.DICT_APRILTAG_36h11` present, 587 codes, 6x6 data bits (verified by import) |
| Other pins | see `requirements.txt` |
| Browser for tests | Chromium build 1243 via Playwright 1.63.0 (headless) |

References checked on this date (through a fetch tool; quotes in
`docs/POSIT_CONNECT.md` and `docs/CONVENTIONS.md`):

* Posit Connect FastAPI user guide, publishing-CLI pages, process management
  and configuration appendix.
* AprilRobotics/apriltag README (definition of tag size: "the distance between
  the detection corners, or alternately, the length of the edge between the
  white border and the black border").
* The sticker-sheet product page named in the README states 60 x 65 mm
  stickers with a "50mm x 50mm" tag but does **not** say which edge the 50 mm
  refers to. LUMEN therefore treats 50 mm as nominal only and asks for a
  caliper measurement of the black-square edge.
* MDN pages for `getUserMedia`, `getCapabilities`, `MediaTrackConstraints`.
  The browser-compatibility tables did not come through the fetch, so torch
  support on iPhone Safari is **not established from documentation**; the app
  probes it at run time and reports "unknown" honestly.

## 2026-11-25 — Marker corner localisation (prompt 03)

Scene: flat panel, four 50 mm tags at arbitrary rotations, 1280x960 synthetic
image, tags 63–90 px on a side, lens blur sigma 0.7–1.5 px.

**Finding 1 — stock corner refinement is biased, and the bias is a scale
error.** Against renderer ground truth (4 tags, 16 corners):

| OpenCV `cornerRefinementMethod` | RMS px | mean radial error px (− = tag looks smaller) |
|---|---|---|
| NONE | 0.958 | −0.838 |
| SUBPIX | 0.660 | −0.653 |
| CONTOUR | 1.413 | −1.403 |
| APRILTAG | 0.823 | −0.487 |

A blurred black/white edge is only symmetric in linear light; in the
gamma-encoded image the mid-grey crossing sits inside the black region, so
every border edge moves inward. −0.65 px per edge on an 83 px tag is −1.6 %
of scale, i.e. −8 mm at 500 mm. This is not noise and does not average out
over tags or frames.

**Change made:** `lumen/vision/edgefit.py` fits a Gaussian edge-spread
function to the actual pixel samples of each border edge, on sRGB-decoded
(linear) intensities, in lens-corrected coordinates, with a re-centred band.
Corners are intersections of adjacent fitted lines.

| Condition (1280x960, 4 tags) | RMS px | max px | mean radial px |
|---|---|---|---|
| clean, no distortion | 0.009 | 0.017 | +0.004 |
| distortion k1=0.08 k2=−0.15, calibrated | 0.015 | 0.031 | +0.004 |
| same distortion, *uncalibrated* (straight-line fit) | 0.027 | 0.062 | +0.005 |
| blur sigma 1.5 px | 0.010 | 0.026 | −0.004 |
| shot+read noise | 0.016 | 0.024 | −0.006 |
| noise + JPEG quality 80 | 0.021 | 0.036 | −0.003 |
| motion blur 4 px | 0.008 | 0.016 | −0.004 |

**Finding 2 — a wrong tone curve brings the bias back; interior cell
boundaries make it observable.** Real phones do not apply a pure sRGB curve
(local tone mapping, sharpening). With a deliberately mismatched rendering:

| Rendering | radial error without correction | with polarity-balanced correction | estimated shift |
|---|---|---|---|
| unsharp mask 1.0 after gamma | −0.160 px | −0.005 px | 0.10–0.12 px |
| extra gamma 1.25 | +0.217 px | −0.002 px | −0.15 px |
| extra gamma 0.8 | −0.202 px | −0.003 px | +0.14 px |
| S-curve 0.6 + unsharp 0.6 | −0.435 px | −0.005 px | 0.30–0.31 px |
| 2560x1920, gamma 1.25, blur 1.4 | (not run) | −0.000 px | −0.23 px |

**Change made:** every tag contains black→white and white→black boundaries at
known grid positions. A residual non-linearity moves all of them toward their
dark side by the same amount `delta`; fitting all interior boundaries jointly
(two families, for directional blur) isolates `delta`, which is then removed
from the border. 35 % of the correction plus 0.01 px is kept as uncertainty
because halos are not a pure shift. Limitation: needs cells of at least
6.5 px; smaller tags are flagged `bias_not_estimated`.

**Corner covariance check** (24 noise seeds per row, 1600x1200, formal
covariance from the edge fits, no inflation):

| Noise | empirical std px | formal prediction px | variance ratio emp/pred |
|---|---|---|---|
| low | 0.0078 | 0.0086 | 0.82 |
| mid | 0.0143 | 0.0148 | 0.95 |
| high | 0.0233 | 0.0237 | 0.99 |
| mid + JPEG 75 | 0.0178 | 0.0142 | 1.60 |

Defaults chosen from this: variance inflation 1.6 (covers JPEG-correlated
noise), deterministic floor 0.012 px.

Cost: about 150–230 ms for 4 tags without the interior-boundary pass and
450–650 ms with it (pure NumPy, one core). Too slow for live tracking as is;
see the live-tracking entry for what was done about it.

## 2026-11-25 — Planar solver (prompt 04)

Scene: flat panel, four 50 mm tags spanning about 400 x 350 mm, camera at
about 700 mm, 1600x1200, noise on. Error = distance between the estimated and
true plane position of 56 probe points spread over 650 x 560 mm.

| Case | mean mm | p95 mm | max mm | notes |
|---|---|---|---|---|
| 4 tags, homography (no K) | 0.030 | 0.064 | 0.066 | reduced chi² 0.09 with the early, over-wide covariance |
| 4 tags, pose mode with true K | 0.025 | 0.045 | 0.051 | |
| 2 tags, homography | 0.045 | 0.095 | 0.126 | cross-check weak: size ratio sigma 1.1–1.5 % |
| 1 tag, homography | 0.630 | 1.771 | 2.226 | extrapolation from one 50 mm square |
| 1 tag, pose mode with true K | 0.098 | 0.201 | 0.243 | ambiguity margin delta-chi² 4330 |
| 4 tags, distortion calibrated | 0.049 | 0.113 | 0.117 | |
| 4 tags, same distortion **uncalibrated** | 1.648 | 2.910 | 3.360 | flagged `high_chi2` (64) |
| 4 tags, oblique view | 0.100 | 0.235 | 0.299 | |
| one tag raised 6 mm | 0.146 | 0.358 | 0.413 | raised tag detected by leave-one-out and excluded |
| one tag 48 mm declared as 50 mm | 0.023 | 0.047 | 0.054 | mis-sized tag detected and excluded |

Decisions taken from this table:

* Homography mode is the default whenever a view has two or more tags: it
  needs no intrinsics and is immune to focus breathing / OIS shifts.
* A single tag is accepted only with a caveat (`single_tag_no_independent_check`).
* Uncharacterised lens distortion is the dominant error without a
  calibration; it must appear in the uncertainty budget as a systematic bound.

## 2026-11-25 — Milestone 1: working web application (prompts 01, 04, 09 baseline, 10 baseline)

What runs: FastAPI app (`app.py`), relative-URL single-page UI, session store,
operation library, photo upload, tag measurement, planar registration,
operator-drawn datums snapped to image edges, local frame, rectified view,
placement overlay (grid, axes, dimensions, target), gate report and
uncertainty budget, simulated demo session, JSON export.

Measured through the HTTP API on synthetic photos (`tests/integration/test_api_flow.py`):

| Flow | Result |
|---|---|
| single calibrated photo, 4 measured tags, datums drawn 3–4 px off and snapped | target corners within 0.15 mm of truth (typ. 0.02 mm); inside the reported expanded uncertainty |
| datums drawn on the *rectified* view instead of the photo | same accuracy; image and overlay verified to share one rectification |
| demo session, 5 photos, joint registration | target within 0.2 mm of truth in **every** photo, not only the one the datums were drawn on; status `prototype_guidance` (labelled simulated) |
| uncalibrated, nominal tag sizes | `research_preview`; budget shows "nominal tags" and "lens distortion: not characterised" |
| no datums / parallel datums / incomplete record / inconsistent tags | `blocked`; no grid and no target outline are returned at all |

First uncertainty budget (1600x1200 synthetic photo, target at x 240, z 300 mm,
4 measured tags, no calibration): expanded 1.03 mm (k=2) = repeatability
0.13 mm + bounds for per-tag size 0.36, edge bias 0.45, uncharacterised
distortion 0.23, flatness 0.30, common caliper error 0.20 mm. The true error in
that image was 0.02 mm: on synthetic data the budget is dominated by assumed
systematic bounds that the renderer does not contain. Coverage with those
effects injected is examined in the uncertainty entry below.

Defects found by looking at screenshots and fixed in this milestone:

* the rectified picture and its overlay were requested at different sizes and
  did not line up — now both name the same size and the client refuses to draw
  if the two specifications differ;
* HUD controls were hidden behind the canvas; toast icons were unbounded;
* labels collided on a phone screen — labels are now queued and placed by
  priority (target and dimensions first; tag IDs and grid ticks yield).

Cost on one core: tag measurement about 0.8 s for a 1600x1200 photo with four
tags (0.35 s without interior-boundary bias estimation); registration 0.13 s;
overlay with uncertainty 0.5 s.

Test count at this point: 227 passed (unit + API integration), about 3 min on
first run (rendering) and under 1 min with the render cache.

## 2026-11-25 — Multi-view localisation and bundle adjustment (prompt 05)

`lumen/vision/multiview.py`: SE(3) bundle adjustment (left-multiplicative
updates, analytic Jacobians, Huber loss, whitened by the corner covariances),
gauge = one tag, every tag metric. Initialisation keeps both IPPE branches of
each tag observation and chooses tag-to-tag transforms by consensus across
views.

Synthetic sweep, 6 views on a 55° arc at about 740 mm, 1280x960, tags about
65–80 px, noise on (`tests/unit/test_multiview.py`, exploratory run):

| Case | camera-centre error (max) | tag-centre error (max) | Outcome |
|---|---|---|---|
| 6 views, K known | 0.16 mm | 0.10 mm | coplanar (could have seen 0.6 mm / 0.18°); held-out ratio 1.10 |
| 2 views (13° apart) | 0.63 mm | 0.41 mm | coplanar, resolving power 1.4 mm |
| 3 views within 1.6° | 0.10 mm | 0.12 mm | `narrow_baseline`; coplanarity **not established** (power 1.7 mm) |
| no calibration, focal length self-calibrated | 0.17 mm | 0.10 mm | f = 966.91 ± 0.17 px, true 966.93 |
| one tag raised 8 mm | 0.17 mm | 0.10 mm | `not_coplanar`; raised tag identified, measured 8.6 mm off |
| one tag declared 52 mm (true 50) | 0.26 mm | 0.14 mm | excluded as `declared_size_inconsistent` |
| one tag moved 6.7 mm after the third photo | < 1.0 mm | < 0.3 mm | excluded as `moved_or_misread` |
| all sizes declared 4 % large | — | — | same reprojection error; distances 4.0 % off (low residual ≠ accuracy) |

Single view versus multi-view: tag-pose covariance trace falls by more than
half from 1 to 6 views; tag-centre error is lower in the multi-view solution
(asserted in `test_multi_view_beats_single_view_for_the_tag_layout`).

What did not work at first:

* Coplanarity against a plane fitted to *all* tags hid a raised tag (the plane
  tilted toward it: 2.3 mm reported for an 8 mm lift). Each tag is now compared
  with the plane of the others, and off-plane tags are found by asking which
  removal leaves the most coplanar set.
* A mis-sized tag was first reported as "moved". The solver now frees the tag
  sizes before labelling; if a different size explains the residuals the
  reason is `declared_size_inconsistent`.

Integration (`tests/integration/test_multiview_flow.py`): registration now runs
the 3-D adjustment whenever there are two or more photos of one camera
configuration. It excludes moved/raised/mis-sized tags from the planar
solution, records the surface model (`verified_multiview`, with the measured
out-of-plane bound feeding the uncertainty budget) and, with an attached lens
profile, re-estimates the focal length from the tags: a profile whose focal
length was 6 % wrong was flagged with a measured difference of −5.7 %.
Without a profile the focal length is self-calibrated for the 3-D checks only
(labelled inferred; it does not satisfy the calibration gate).

Cost: about 2.3 s for 6 views including held-out re-solves and the free-size
test (single core).

## 2026-11-25 — Automatic datum geometry and rounded edges (prompt 06)

`lumen/vision/datums.py` finds straight features in every photo, matches them
across photos and classifies each by triangulation rather than by appearance:

* **on the panel face** — the same physical line in every photo, lying in the
  tag plane (a seam, a joint, the panel/floor junction);
* **off the plane** — a stationary physical line at a measured height (the
  front edge of a raised trim): offered, but choosing it blocks guidance;
* **view dependent** — the line moves on the panel as the camera moves (the
  silhouette of a rounded edge, a shadow, a highlight): never offered as a
  planar datum;
* **unverified** — seen in one photo only: preview at most.

Floor/side pairs are chosen from the configured nominal angle and the
configured side of the panel (a handedness in the panel's own frame). Nothing
uses gravity, the image "down" direction or the phone's orientation.

A rounded panel edge is fitted as a cylinder from its silhouettes in several
photos. For a camera at in-plane distance `c_e` from the tangent line and
height `c_h` above the face, the silhouette appears on the face plane at
`phi = atan2(c_h + R, c_e) - asin(R / hypot(c_e, c_h + R))` around the axis;
the fit solves for the tangent line and the radius together.

Measured on synthetic 1600x1200 photos, 5–7 per sweep, tags measured,
calibrated (script output, 2026-11-25; numbers are simulation results, not
physical accuracy):

| Case | Result |
|---|---|
| Flat panel, floor step + joint seam | both found without drawing; floor line within 0.014 mm and seam within 0.021 mm of truth; raw angle 89.998°; target outline 0.03 mm from truth |
| Floor trim 8 mm proud, 60 mm tall | trim/panel junction classified on-plane; lower front edge classified off-plane at 8.29 mm (true 8.0); the hidden floor line is not invented |
| Extra seam 45 mm beside the datum seam | side datum reported ambiguous; `research_preview` until the operator chooses |
| Floor below every photo | no floor datum, capture hint, `blocked` |
| Rounded edge R = 20 / 35 / 60 mm | R = 20.01 ± 0.17 / 35.03 ± 0.17 / 60.04 ± 0.27 mm; tangent line within 0.017 / 0.026 / 0.035 mm; target 0.03 mm from truth |
| Rounded edge, one photo | silhouette is not the tangent line: `research_preview`, gate `datum_side_needs_3d` |
| Edge leaning 5° | raw angle 85.001° kept, disagreement 4.999° > 1° limit: `blocked` (not orthogonalised) |
| Two identical seams left and right | resolved only by the operation's configured side |

What did not work at first:

* The silhouette formula had a sign error (found by comparing with a
  brute-force tangent search, now a unit test).
* The floor line was contaminated by the part of the edge beyond the bend;
  lines are now trimmed to their straight support before fitting.
* A cylinder fit returned R = 24 for a true 20 mm when one photo was taken from
  beyond the edge (bend over 87°). Such views, and outliers, are now excluded;
  the fit never drops below half of the photos.
* At higher resolution a silhouette passed as a stationary off-plane line
  (residual 0.69 mm against a 0.8 mm limit). The limit is now 0.25 mm.
* One tag without an edge-bias estimate applied the uncorrected 0.25 px bound
  to every tag (2.9 mm in the budget). The bound is now per observation.

Cost: proposing datums for 5 photos takes about 3.9 s on one core.

Not covered: seams that are curved, occluded along most of their length, or
visible only as a faint texture change; real materials; real lighting.

## 2026-11-25 — Browser tests and the display mapping

`tests/browser` drives the application in headless Chromium (phone portrait
390x844 and landscape 844x390): first-run states, operation selection with
units, the demo walk-through to guidance, theme switch, reload/resume, session
deletion and basic accessibility (names on controls, 44 px touch targets).

`test_pixel_mapping.py` checks the drawn overlay against the picture itself:
tag corners returned by the server are located independently in a screenshot
of the canvas and compared with where the viewer says it drew them.

Finding: everything was drawn about 1 CSS pixel too high (constant −0.97 px in
y) because the canvas was sized from its bordered container. After sizing from
the canvas content box the mean error is (0.01, 0.00) px at 3x zoom,
(0.02, −0.07) px at 6x and (0.24, −0.07) px at 10x, i.e. at the resolution of
the screenshot-based check.

The page's content security policy forbids `eval`, so the tests pass functions
(not expression strings) to the browser.

Test count at this point: 273 unit/integration + 9 browser, all passing.
