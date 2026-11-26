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

## 2026-11-25 — Camera, video and capture coaching (prompt 02)

**In-app camera** (`lumen/static/js/camera.js`, `views/camera_sheet.js`): rear
camera through `getUserMedia` (no audio, `playsinline`, opened only by a button
press), still capture of the full stream frame, recorded sweep, flash on/off
pair, camera/lens choice, permission and interruption recovery. The camera is
stopped when its screen closes and when the page goes to the background.

**Torch**: the control exists only when the browser lists the capability. The
light state written into a photo is `on`/`off` only if the browser reports it
or the picture got measurably brighter/darker; otherwise it is `unknown`. A
flash pair without a confirmed light is discarded.

**Video** (`lumen/vision/video.py`): one decoding pass; the sharpest frame of
each time window becomes a candidate; tags are located in each candidate;
frames are then chosen greedily for viewpoint spread, weighted by sharpness,
tag coverage and exposure. No random numbers: the same file gives the same
frames. Viewpoints for the ranking use an assumed field of view and are used
for nothing else. The video is never stored; chosen frames go through the
normal photo path. Processing runs as a background job whose state lives in
the session record, so any worker process can answer a status poll.

Measured (synthetic 2 s sweep, 24 frames 1280x720 at 12 fps, 8 viewpoints on a
44° arc, two of every three frames motion-blurred):

| Check | Result |
|---|---|
| candidates | exactly the 8 unblurred frames |
| estimated spread of the chosen frames | 45.1° (true 44°); pairwise angles within 8° of truth |
| registration from the chosen frames | tag-to-tag distances within 0.06 mm of truth |
| analysis time | 0.16 s decode and candidates, 4 ms selection; whole job 4.2 s (8 frames measured) |
| H.264-in-MOV and HEVC-in-MP4 stored sideways with rotation metadata (built with the ffmpeg command) | decoded upright, tags within 3 px of the original |
| static video | flagged `narrow_baseline`, no padding with look-alike frames beyond the minimum of 3 |
| junk, oversize, undecodable uploads | refused with reasons (415 / 413 / failed job) |

**Coaching** (`lumen/services/capture.py`): after every photo the server says
what is missing — more views, which way there is room to move, tags too
small/clipped/oblique, glare, a tag missing from the newest photo, and (after
analysis) a floor line or side edge that no photo shows. Two identical
features are reported as something more photos cannot resolve.

**Flash pairs**: a pair is `aligned` when the tags moved less than
max(1 px, 0.08 % of the long edge) between light-off and light-on; otherwise
`moved` with the shift in pixels and millimetres. Exposure ratio is reported
from EXIF when both photos have it and as *unknown* for browser captures.

Browser tests (headless Chromium, fake camera fed with a synthetic video of
the panel; 8 tests): photo capture, five-photo sweep registering end to end,
recorded WebM sweep through the server, torch reported / torch silent,
permission refusal, interruption and background stop.

Not run: any real phone. iPhone Safari behaviour (stream resolution, torch
capability, MediaRecorder container, lens switching) is unverified.

## 2026-11-25 — Live guidance (prompt 09)

Each live frame is sent to the server, measured and localised against the
registered tag layout, and returned with the overlay for *that frame*; the
viewer shows the same frame with it. Nothing is drawn when a frame cannot be
localised, when the target is outside the picture, when a registered tag has
moved, or when no answer has arrived for 2.5 s. Live frames are never stored.

Measured (synthetic, registration from five 1600x1200 photos, live frames
1280x960 from viewpoints not used for registration):

| Check | Result |
|---|---|
| target outline versus truth | 0.03 mm |
| time per frame on one busy core | tags 320–400 ms, localisation 5–14 ms, overlay with uncertainty 410–440 ms, or 50–65 ms when the uncertainty of a nearly identical view is reused |
| lost states | no tag / tags too small (arrow: closer) / unknown tags / target off-screen (arrow toward it) / tag moved |
| 960x540 browser stream | outline within 0.8 mm; status `research_preview` (tags too small for the edge-bias estimate: 1.04 mm expanded against a 1.0 mm limit) |

Uncertainty is reused only for the same tags seen from a view whose projected
points moved less than 2 % of the frame diagonal, and for at most 6 s.

Live frames reduced by the browser are matched to a lens profile through the
stream size the browser reports; without it they count as uncalibrated.

## 2026-11-25 — Verification after installation (prompt 11)

`lumen/vision/placard.py` finds the installed placard near the target and
measures its outline; `lumen/services/verification.py` turns that into signed
deviations in the local frame with an uncertainty, and a guard-banded
observation (`within` / `outside` / `indeterminate`), never an acceptance.

Outline measurement alone (true mapping, 1600x1200, 1.66 px/mm):

| Case | Corner error (max) | Centre error |
|---|---|---|
| on target / shifted / rotated / 40 mm away | 0.11–0.18 mm | 0.003–0.006 mm |
| plain white placard, no border | 0.09 mm | 0.013 mm |
| rounded corners (6 mm) | 0.15 mm | 0.005 mm |
| one corner folded over 12 mm | 0.12 mm | 0.003 mm; flagged, gap measured 12.1 mm |
| corner covered by a block | < 0.4 mm | located from the two sides, flagged |
| narrower part than recorded | width within 0.4 mm | flagged `size_differs_from_record` |
| no placard | — | `placard_not_found` |

Whole chain through the API (registration, datums, new viewpoint; target
±2.0 mm, ±1°):

| Installed at (dx, dz, rotation) | Measured | Expanded U | Observation |
|---|---|---|---|
| +0.3, −0.2, 0° | +0.320, −0.178, 0.000° | 0.92 mm | within |
| +1.4, −0.9, 0° | +1.423, −0.885, −0.001° | 0.92 mm | indeterminate |
| +4.0, +2.5, 0° | +4.014, +2.518, −0.010° | 0.92 mm | outside |
| +0.2, +0.1, 2.5° | +0.225, +0.115, +2.505° | 0.92 mm | outside (rotation) |
| −0.4, +0.3, −0.5° | −0.379, +0.314, −0.504° | 0.92 mm | within |

The measured size is 0.10–0.13 mm small on every placard: each edge sits about
0.1 px toward its dark side. The budget carries a 0.25 px bound per side for
this; it cancels in the centre.

What did not work at first:

* The search locked onto the printed border 3 mm inside the edge. Both edge
  signs are now measured and the outline whose size matches the record wins.
* A proposal found in bare panel texture is now "not found" when no side can
  be measured.
* The edge-noise estimate included the edge itself and failed in a narrow
  scan; it now uses the outer end of the profiles.
* A browser test showed the outline centred on a hidden (zero-size) canvas;
  the picture area is now laid out before anything is centred in it.

Evidence is refused when the photo is one of the registration photos, cannot
be registered, or shows a moved tag. A clean number from an outline with
quality flags is reported as indeterminate, not within.

Test count at this point: 327 unit/integration + 22 browser, all passing
(calibration and photometric modules are still in progress and not counted).

## 2026-11-25 — Lens profiles from board photos (prompt 03)

A web page is not given the camera's lens parameters, so they are measured:
`lumen/vision/calibration.py` (board detection, calibration, refusals,
uncertainty, held-out check), `lumen/synth/boards.py` (simulated board
photos), `lumen/services/calibration.py` and `/api/calibration/*` (a
workspace of board photos that is not a measurement session), and the
Settings screen (download a board, enter its *measured* square size, take or
choose photos, coverage map, calibrate, check on one more photo, save on the
phone, attach to a session).

Simulated 1600x1200 phone camera with the principal point 7.3 / −5.1 px off
centre, fy/fx = 1.001 and distortion (0.08, −0.15, 0.0006, −0.0004); 12
photos, one more held out. Figures are from a run made for this entry, with
a different random seed than the tests:

| | ChArUco 9x6, 30 mm | Checkerboard 10x7, 25 mm |
|---|---|---|
| corner position against the true projection, RMS (max) | 0.033 px (0.10) | 0.027 px (0.07) |
| reprojection RMS of the fit | 0.031 px | 0.026 px |
| focal length error fx, fy | +0.02, +0.02 px | −0.07, −0.06 px |
| principal point error cx, cy | −0.17, −0.01 px | −0.02, −0.09 px |
| distortion k1, k2 found | 0.07995, −0.14999 | 0.08011, −0.15012 |
| stated standard uncertainty of fx, cx | 0.14, 0.14 px | 0.10, 0.10 px |
| held-out board fitted as a plane, with the lens model | 0.034 px (0.015 mm) | 0.027 px (0.011 mm) |
| the same without a lens model | 0.21 px (0.09 mm, 0.37 mm worst distance) | 0.22 px (0.09 mm, 0.37 mm) |
| fitted against true lens over the whole picture, after the best plane mapping | 0.07 px max | 0.05 px max |

Both runs carry the notes "few or no calibration points near three picture
corners" and "distortion is an extrapolation beyond the outermost point":
the profile is usable, and says where it is weakest.

Whole chain through the API (board photos → profile → five panel photos
through the same lens → target outline): outline within 0.3 mm of truth and
inside the stated uncertainty; the focal length the tags imply agrees with
the profile within 0.3 %.

Refusals covered by tests (58 unit, 4 API, 1 browser): no board in the photo,
too few photos, all photos square-on, all photos with the same tilt, photos
that contradict each other, a print stretched in one direction, mixed pixel
sizes in one calibration, a "check" photo that was already used for the
calibration, a checkerboard cut off by the picture edge.

A profile is applied only to pictures of exactly its pixel size and
orientation, and not to a picture that reports a different camera, lens or
zoom (`camera_for_frame`); such a picture is measured as uncalibrated and
says why. Lens profiles are stored in the browser of the phone they belong
to; the server keeps calibration photos only as long as a session.

What did not work at first:

* A test compared two identical calibrations to 1e-9; multi-threaded linear
  algebra differs by 5e-9 between runs. Tolerance is now 1e-6.
* The Settings screen printed `null[object HTMLDivElement]`: the browser's
  own `append()` writes `null` and arrays as text. Elements made by the `h()`
  helper now skip and flatten them, and browser tests fail on any stray
  `null`, `undefined`, `NaN` or `[object` in what the operator reads.
* The coverage map was empty: the board extent is reported as fractions of
  the picture and was drawn as pixels. A test now reads the canvas.

Not done: nothing here has been run with a real phone or a real print
(physical protocol: NOT RUN). The simulated lens has no rolling shutter,
focus breathing or optical image stabilisation; a phone that refocuses
between calibration and measurement changes its focal length by an amount
this profile does not know.

## 2026-11-25 — Independent scale check (prompts 03, 10)

Every length comes from the declared tag sizes, so an error common to all of
them (the sticker measured instead of its black square, a wrong nominal) is
invisible to every tag-to-tag check. `lumen/services/reference.py` compares
one ruler measurement between two marked tag corners with the same distance
from the registration; a contradiction is a blocking gate
(`independent_length_check`).

| Declared (true 50 mm) | Tag-to-tag checks | Target error | Report before the ruler | After the ruler |
|---|---|---|---|---|
| 50 mm | consistent | < 0.2 mm | prototype guidance | agrees, unchanged |
| 52 mm on all four | consistent | 12–22 mm (4 % of the distance to the datums) | prototype guidance — a false result with a clean report | blocked, no outline drawn, "+4 %" stated |
| corrected to 50 mm | consistent | < 0.2 mm | prototype guidance | re-checked against the new registration, agrees |

Without the ruler measurement the app cannot detect this fault. The check is
offered on the Rectified screen and is not mandatory.

## 2026-11-25 — Code scanning for identification (prompt 08)

`lumen/vision/barcode.py` reads codes with zxing-cpp (plain, contrast
equalised, enlarged, and rotated in steps for 1-D codes) and falls back to
OpenCV's QR decoder; `/api/scan` matches the text exactly against the
operation library. A scan selects a record only on exactly one exact match;
two codes in view are offered for choice; an unknown code selects nothing.
A code never supplies a dimension or the scale.

| Case (simulated label photos, blur 1.2 px, noise) | Result |
|---|---|
| QR, Data Matrix (8 px modules), Code 128, Code 39 (3 px narrow bar), tilted 14° with keystone | read, outline at the label |
| Code 128 at 0°, 14°, −31°, 47°, 90°, 118° | read in thorough mode; the quick mode used by the live loop does not search rotations (the same code at 31° is not read) |
| two QR labels in one picture | both, once each |
| AprilTags, blank picture | no code |
| QR without zxing-cpp | read by the OpenCV fallback |

Limit found: about 3 px per narrowest bar is needed; smaller codes are not
read and the screen says to move closer.

## 2026-11-25 — Optional assistant (prompt 12)

Off unless `LUMEN_LLM_ENABLED=1` and an HTTPS endpoint are configured; no
part of measurement, gating or verification calls it. It explains the gate
report of a session in words. Replies are discarded unless they fit the
schema, quote the provided sources, contain only numbers that occur in the
provided context, and contain no acceptance or override wording. Tested with
a mock adapter only (14 tests); the round trip to a real endpoint is an
opt-in test that has NOT been run.

## 2026-11-25 — Deployment readiness (prompt 14)

NOT DEPLOYED: no Posit Connect server was available. What was tested
locally instead:

| Check | Result |
|---|---|
| app behind a path-stripping proxy that sets `RStudio-Connect-App-Base-URL` (with and without trailing slash, with and without the header) | every page, asset and API call stays inside the content path (API tests and a browser walkthrough) |
| bundled files | no absolute or external URLs |
| two worker processes, one shared data directory | a session created by one is served by the other; job state is in the session store |
| sessions and Connect users | a session is bound to the user who created it |
| `scripts/smoke_test.py` | 11 steps pass against a sub-path; fails against a dead address |
| `rsconnect write-manifest fastapi` (rsconnect-python 1.34.1) in a scratch copy | manifest with 94 files; `manifest.json` is generated at release, not committed |

`docs/POSIT_CONNECT.md` lists what remains NOT RUN: publishing, the iPhone
camera and torch through the Connect address, process scaling and timeouts
on a real server.

## 2026-11-25 — Entering and revising operation records (prompt 08)

The library was readable and had revisions; it could not be edited from the
app. Added: a record editor on the Identify screen
(`lumen/static/js/views/operation_editor.js`), `/api/operations/validate`,
`/supersede`, `/delete`, a change log, and a gate.

| Rule | How it is kept |
|---|---|
| nothing is invented | an empty field is sent as "not stated"; the server lists it as missing and the measurement stays a research preview |
| units | each length keeps the unit it was entered in; the editor shows the millimetre values the measurement will use (9.5 in → 241.30 mm in the test); a unit the form does not offer (cm, ft, rad) is kept, not replaced |
| revisions | a draft can be edited or deleted; a released revision is fixed; a correction is a new revision; the old one is marked superseded, its values unchanged |
| unreviewed values | new gate `operation_record_released`: a draft or superseded record cannot reach guidance status |
| codes | a scanned or typed code means the released revision; drafts and superseded revisions of the same operation no longer make it ambiguous; the same code on two operations still asks the operator |
| who and when | every save, supersede and delete is logged with the Connect user and the content hash |
| sessions | a session keeps its own copy of the record it started with (checked: editing the draft afterwards does not change the session) |
| several worker processes | saves are made under a file lock and other processes re-read the file when it changes (4 writers × 6 records: none lost) |

Local records are stored in `LUMEN_OPERATIONS_DIR`, or next to the sessions
when that is not set. The second case is temporary storage; the screen says
so and offers export. `LUMEN_OPERATIONS_READONLY=true` turns entry off.

What did not work at first:

* The check-while-typing answered a new record (empty ID) with a raw schema
  error. It now answers with the place and reason of each unreadable entry
  and still lists what is missing in the rest.
* Unit errors lost their place in the record (the unit check raised its own
  error type); they are now reported as `placards.0.position.x.unit: Unknown
  length unit 'cubits'`.
* Typing a barcode text into the search field selected the right record and
  filtered the list down to nothing; the filter is cleared on an exact match.
* After superseding revision A, the panel code matched A and B and was
  "ambiguous". Fixed as in the table.
* The form squeezed three fields into one row on a phone. Two per row now.

Not done: no CSV import, no approval workflow (anyone with access to the app
can release a record; the log shows who did), no OCR of printed identifiers.

Test count at this point: 440 unit and integration tests (439 pass, 1 opt-in
test of a real assistant endpoint skipped) and 33 browser tests, all passing.
In the full browser run one test of the new record editor failed: the editor
learned from a delayed browser event that a section had been opened, and
rebuilt the form before the event arrived. The editor now reads the state of
its sections from the page before rebuilding; that test then passed four
times in a row on its own.

## 2026-11-25 — Held-out layouts find two faults in the multi-photo path (prompts 10, 13)

Before the first full benchmark run, six of its held-out layouts were run at
1280x960, a smaller size than any test uses, with in-camera processing added
(sharpening, a tone curve, JPEG). Two things were wrong. Both were in code
whose tests passed.

**1. More photos made the stated uncertainty worse.** A tag under about
52 px cannot have its edge shift estimated. Such a tag had full weight in the
scale, and the budget charged 0.25 px of unknown shift to it. A few small
tags among fifteen observations doubled the stated uncertainty and, in one
case, tripled the error.

First attempt, discarded: give such a tag a large uncertainty of its own in
the corner covariance, so that its size counts for little. That fixed the
multi-photo figures and broke the single-photo check that finds a raised
tag. The unknown shift is almost the same for every tag of a photo, so it
cancels when tags are compared with each other; treating it as independent
noise made that comparison blind.

Change kept: the shift is a property of the camera's blur and processing, so
a tag too small for its own estimate takes the value measured on the larger
tags of the same photo, or of all registration photos when its own photo has
none (`edge_shift_pool`, `tag_obs`). The scatter between the donors, 35 % of
the value and a floor are kept as its uncertainty; donors that disagree by
more than 0.12 px are not pooled. The budget charges 0.12 px to a tag
corrected this way, 0.05 px to a tag with its own estimate, and 0.25 px when
no tag of the capture was large enough to measure anything. On a test photo
with a strong tone curve the border of a 36 mm tag is off by more than
0.08 px uncorrected and by less than 0.06 px after the pooled correction.

**2. Four of six multi-photo runs were blocked: "the floor datum is not on
the panel face".** It was on the face. Three causes:

* heights were measured from the face of one 50 mm tag (the reference tag of
  the bundle adjustment); its tilt error of about 0.1 degrees is a millimetre
  at the floor line;
* the height was compared with a fixed 1 mm limit, whatever the photos could
  resolve;
* all photos look down at the floor line from similar heights; a shift of the
  edge by a fraction of a pixel, the same in every photo, then looks exactly
  like a height.

Change: heights are measured from the plane fitted through all tags. Each
height gets a margin: what a 0.25 px common edge shift could fake, plus two
standard deviations of the line fits and of that plane at that place. A
feature is "on the face" only if that is shown at both ends, "off the face"
only if the whole feature is clear of it, and otherwise "not settled by these
photos"; it is then taken as the record declares it, and the screen says so.

Same six layouts, same photos, before and after both changes (U = expanded
uncertainty, error = largest target corner error, both mm):

| Layout, processing | Single photo, before | Single photo, after | All photos, before | All photos, after |
|---|---|---|---|---|
| 1, plain | 1.09 / 0.26 | 1.09 / 0.26 | preview, 2.10 / 0.12 | preview, 1.17 / 0.14 |
| 2, sharpened | 1.15 / 0.16 | 1.15 / 0.16 | preview, 2.39 / 0.54 | guidance, 0.99 / 0.15 |
| 3, tone curve, tags 34–49 px | 4.23 / 1.62 | 1.93 / 1.31 | **blocked** | preview, 1.44 / 0.33 |
| 4, sharpened, contrast, JPEG 85 | 1.63 / 0.84 | 1.63 / 0.84 | **blocked** | preview, 1.21 / 0.26 |
| 5, plain | 1.29 / 0.08 | 1.29 / 0.08 | **blocked** | preview, 1.23 / 0.17 |
| 6, sharpened | 1.53 / 0.24 | 1.53 / 0.24 | **blocked** | preview, 1.35 / 0.38 |

Every error is inside its stated uncertainty. At this size the status is a
preview because the uncertainty exceeds half of the ±2 mm sample tolerance.
The closest case is layout 3 from a single photo: error 1.31 mm against
1.93 mm. Its tags are 34 to 49 px across the photos, at or below the 40 px
the gate `tag_resolution` asks for, and the capture screen now warns about
small tags before the analysis ("The tags are small in the photos: move
closer, or use Full-res photo").

Datum heights after the change (true value 0): side seams −0.04 to +0.10 mm,
resolved to ±0.6 to ±0.8 mm, reported "verified in 3-D"; floor lines up to
1 mm at their far ends, resolved to ±1.1 to ±1.4 mm, reported "not settled".
On the test scene at 1600x1200 the floor line is likewise "not settled"
(±0.8 mm); at 4032x3024 it is verified (±0.3 mm). The 8 mm proud trim of the
fault tests is still reported as off the face (8.0 mm, margin under 3 mm).
Three tests that expected "verified" for a horizontal line seen from one
height were changed to accept "not settled"; the side seam must still verify.

What this says about the earlier results: the multi-photo figures recorded
above in this log were measured on the test layouts only, where neither
fault showed. The held-out set did its job. Because these six layouts have
now been looked at, the benchmark report also contains a second set from
another seed that was not looked at before its run.

`ALGORITHM_VERSION` is now `lumen-algo-2026.11.1`.

## 2026-11-25 — A rendering fault in the benchmark photos, and one bad observation (prompts 05, 13)

A control case of the benchmark (correct tag sizes, five photos) registered
only two of four tags; the other two were excluded as "moved or misread".
Nothing had moved.

Cause, in two steps:

* In two of twenty tag observations one border line was rotated by 0.3
  degrees (0.2 px at its ends). The edge fit was right; the picture was
  wrong. The benchmark rendered with a regular 2x2 grid of samples per pixel.
  Along an edge at 45 degrees or at a slope of 1:2 every pixel meets that grid
  at the same phase and is rounded the same way, so the whole edge is
  displaced. The affected borders had exactly those slopes.
* The multi-photo solver answers an observation that does not fit by
  excluding its tag from the whole session.

Changes:

* The renderer has a second sampling pattern, `pattern="lattice"`: the
  samples of a pixel sit on a golden-ratio lattice that is shifted
  differently in every pixel, so rounding no longer adds up along an edge.
  Displacement of the three affected border lines, at their two ends (px):

  | Sampling | tag 41, photo 5 | tag 102, photo 4 | tag 102, photo 4, next side |
  |---|---|---|---|
  | 2x2 grid (benchmark until now) | +0.19 / −0.20 | +0.14 / −0.14 | −0.04 / −0.19 |
  | 3x3 grid (what the tests use) | −0.05 / −0.04 | −0.04 / +0.05 | +0.03 / −0.13 |
  | 4x4 grid | −0.02 / −0.00 | −0.03 / −0.01 | +0.06 / −0.10 |
  | lattice, 4 samples | +0.02 / −0.05 | +0.04 / −0.03 | −0.01 / +0.01 |
  | lattice, 9 samples (benchmark now) | +0.00 / −0.05 | +0.01 / −0.03 | +0.01 / +0.02 |

  The benchmark photos were rendered again with the lattice. The unit and
  integration tests keep the 3x3 grid they were written with; part of the
  corner errors recorded for them earlier in this log may therefore be
  errors of the picture rather than of the measurement.
* A tag that does not fit in exactly one photo and is seen in at least three
  is no longer excluded: that one observation is left out, the tag is kept,
  and the report says so ("Left out because it did not fit the other photos
  of the same tag"). A tag that does not fit in two or more photos is
  excluded as before (the moved-tag test still passes).

Why the tests had not shown it: they use one tag layout and a 3x3 grid, and
none of its borders fell on such a slope badly enough to cross a test
threshold.

## 2026-11-25 — Flashlight experiment (prompt 07)

`lumen/vision/photometric.py` fits the start of a rounded edge, its direction
and its radius to the shading of photos taken with the phone light on,
together with exposure, beam shape, lens fall-off, ambient light, torch
position, gloss and tone curve, and decides from its own evidence whether the
result is supported. The full ablation with every case is in
`docs/PHOTOMETRIC_ABLATION.md`. Summary of its held-out set (synthetic
photos, one run; I re-ran the 23 tests of the module myself: all pass):

| Compared with the outline method | Tangent line, median error | Photometric closer in |
|---|---|---|
| on the same torch-on photos (39 cases) | 0.04 mm against 0.58 mm | 37 of 39 |
| against torch-off photos from the same places (17 cases) | 0.09 mm against 0.02 mm | 5 of 17 |

* It helps when the sweep was taken with the light on, most in a dark room.
  The outline of a lit rounded edge is found 0.4 to 0.9 mm too far in,
  because the surface darkens toward its outline when the light sits next to
  the lens.
* It does not help when photos without the light exist.
* It was wrong without refusing on a surface with smooth brightness
  variation (−1.1 mm at a stated ±0.3 mm) and for the radius under a mild
  tone curve (+0.5 mm at ±0.16). The renderer and the fitted model contain
  the same physics, so these numbers are an upper limit on how good it is.

Decision: the module stays research evidence. In the application it is a
step that can be run from the Datums screen for rounded-edge operations
(`POST /api/sessions/{id}/research/photometric`); the result is stored under
`research`, labelled "research evidence only", shown next to the outline
method, and never used for a datum, a check or a target (a test compares the
datums and the whole check report before and after). `LUMEN_PHOTOMETRIC_ENABLED=false`
removes it.

Two things the ablation changed in the main path:

* Heights and outlines were measured from the face of one tag; the ablation
  measured up to 0.27 mm of height error at the edge from that. The plane
  through all tags is used now (entry above).
* A rounded-edge datum fitted to torch-on photos now carries a note that
  such outlines are found too far in and that photos without the light
  should be used.

Guided sweep: the capture coach already directs the sweep (more views, step
sideways, floor or edge missing); the experiment adds its own requests
("take the sweep with the light on", "add a view", "move toward the edge")
to its card. Nothing was tried on a real phone: whether a web page can
switch the torch at all depends on the browser, and the camera sheet says so
when it cannot.

## 2026-11-25 — Clean installs on other Python versions (prompts 14, 15)

`requirements.txt` and `pyproject.toml` said Python 3.11 to 3.13. That was
not true:

| Python | Clean environment from `requirements-dev.txt` | Tests |
|---|---|---|
| 3.11.15 | **fails**: the pinned numpy 2.5.5 needs Python 3.12 or newer | — |
| 3.12 | installs (the development environment) | full suite |
| 3.13.18 | installs; OpenCV 4.14.0, numpy 2.5.5, SciPy 1.18.1, FastAPI 0.165.0, pydantic 2.14.4 | 249 of 249 in a representative part of the suite (geometry, units, library, gates, storage, local frame, planar solver, tags, codes, API flow, library flow, deployment) |

The files and documents now say Python 3.12 or 3.13. Other versions of the
libraries, and OpenCV 5, were not tried.

## 2026-11-25 — First full benchmark: four faults behind "handled" (prompts 05, 06, 10, 13)

`scripts/run_benchmark.py` was run in full for the first time (algorithm
`lumen-algo-2026.11.1`, photos 1600x1200, rendered with the lattice sampling
of the entry above): 12 layouts of set 9131 with 17 fault cases and coverage
trials, and 12 layouts of set 9132, which had not been looked at.

| First run | single photo, hand-drawn datums | all photos, automatic datums |
|---|---|---|
| set 9131: corner error, median / max | 0.21 / 0.89 mm | 0.13 / 0.42 mm |
| set 9132: corner error, median / max | 0.15 / 0.38 mm | 0.10 / 0.61 mm |
| error within the stated uncertainty | 24 of 24 | 24 of 24 |
| wrong result shown as guidance | 0 | 0 |
| fault cases "handled as expected" | — | 17 of 17 |

The last line was not true in the sense that matters. The verdict column said
"handled" because the outcome was safe (blocked, or a preview). The columns
beside it, read case by case, showed four cases that were wrong inside:

| Case | What the report showed | What had happened |
|---|---|---|
| proud trim | preview, error 61 mm, only `uncertainty_vs_tolerance` failed | The trim's front edge was measured 7.9 mm in front of the face (true 8.0) and rejected as a datum. The trim's upper edge, which does lie on the face, was then taken as the floor line without a word. It was a preview only because the uncertainty exceeded half the tolerance at this photo size; with larger tags it would have been shown as guidance, 61 mm from where the record's floor line puts the target. |
| raised tag | blocked: "floor datum NOT on the target plane", "side datum NOT on the target plane", uncertainty 6.8 mm | The raised tag had been identified and left out of the planar solution, but the 3-D solution, its plane "through all tags", the flatness figures and the datum heights measured from that plane still contained it. The plane was tilted and 3 mm off; the datum lines appeared 8.4 and 3.6 mm behind it. Blocked for a reason that was false. |
| moved tag | blocked, tags 41 and 317 set aside | The tag that had moved was tag 58. Three tags were in all five photos and a fourth in one. The solver chose the tag to leave out by the size of its residuals; the adjustment had spread the contradiction over all of them and the largest share fell on a sound tag. With that one out, the two that remained contradicted each other one against one, and the second sound tag went as well. The moved tag was kept, together with the tag seen in a single photo, which could not contradict it. |
| leaning side edge | blocked: "Missing datum: floor", with the hint to tilt the phone down | The floor line was in all seven photos, 900 mm of it. The line finder had merged the foot of the panel with the edge of the floor beyond the panel's corner into one segment of 1490 px, about a quarter of it beyond the corner. The edge fit over the whole diverged, the thin-line fit did not hold piece by piece, and the floor line was not offered at all. The case never reached the question it is about (two datums that do not meet at the record's angle). |

Why the benchmark had not caught them: its judge accepted "blocked or
preview" for these faults, and at 1600x1200 almost every result is a preview
anyway. The unit tests had not caught them either: on the test layout (all
four tags in every photo, so that a moved tag is one against three; a floor
line that ends inside the picture) every one of these paths gives the right
answer.

Changes:

| Fault | Change |
|---|---|
| a line on the face that is the edge of a raised part | For every datum candidate the 3-D height is solved together with the place on the plane directly under it. A line that was applied automatically and has a fixed edge at another depth right beside it (within 30 mm in space), or further out with little depth against its distance (a face continuing past the line, as a trim's front does), carries the new check `datum_floor_no_step_beside` / `datum_side_no_step_beside`: the result stays a preview and the Datums screen asks "Is this the right line?". The operator's own choice of the line ("Choose detected line", or drawing it) settles it; the system cannot know which feature a record means. Features of a floor that meets the panel at the line have the opposite proportion (much depth, little distance) and do not raise it. |
| plane after a tag was set aside | When the 3-D check names tags off the common plane, the capture is solved again without them (up to twice). Plane, flatness figures and datum heights now describe the tags that carry the result. If the rest still does not lie on one plane, the surface is "violated" and the result is blocked. |
| which tag to leave out | As in the plane check, the question is asked the other way round: each tag is left out in turn and the one without which the rest agree is chosen (fewest remaining contradictions, then their size). Residual size only breaks ties. |
| a segment that runs over two features | When neither model fits a merged segment, its middle half and its two halves are measured in turn; the line of a part that fits is laid over the whole segment and the stretch that follows it is kept and measured again. A feature found in another photo may be described by either model (edge or thin line) as long as it is the one used in the home photo. |
| the judge | A fault with a check of its own must fail that check (seven cases). Where the fault is about one tag, a drawn result counts only when that tag was set aside, and the report shows whether exactly that tag was left out. A refusal still counts as handled. |

Reading the remaining cases the same way gave two smaller findings:

* Glossy tags under a flash: after two tags had been set aside, the solver
  still flagged observations that contradict each other
  (`inconsistent_observations`), and no check looked at that flag. New check
  `photos_agree` (warning): the result stays a preview.
* Faint seam, floor outside the photos: one line was found and its partner
  was not, and the hint said "No floor line was found: tilt the phone down"
  although the line that was missing could as well be the other one. The hint
  now says what is known: one line, no second one that fits it.

Two things the report did not say are now said in the list of checks, as
entries that pass but carry an "i" instead of a tick (`caveat` in the API):

* `scale_source`: no independent length was measured, so the scale rests on
  the declared tag sizes alone. This is the one fault of the benchmark that
  no check on the photos can see (all sizes wrong in the same way: 17.9 mm of
  error with a clean report).
* `tags_left_out`: which tags were set aside, why, and how many remain.

The notes that already existed (coplanarity not established, a single
observation left out, a datum on the face by the record's word only) are
marked the same way. A caveat does not change the status.

Also fixed: `RuntimeWarning: overflow encountered in exp` in the edge fit
(a wild trial step of the solver; the value was clamped afterwards, the
logarithm is clamped now). The result of the fit is unchanged.

`ALGORITHM_VERSION` is now `lumen-algo-2026.11.2`.

### After the changes

Both sets and the fault cases again, with the final code (`lumen-algo-2026.11.2`,
runs of 2026-11-25 19:47 and 19:59 UTC, photos 1600x1200, four shared cores;
reports in `docs/benchmark/`). These are the runs after the change of the
next entry as well; that change did not alter any figure here:

| Final run | single photo, hand-drawn datums | all photos, automatic datums |
|---|---|---|
| set 9131, 12 layouts: guidance / preview | 3 / 9 | 2 / 10 |
| set 9131: corner error, median / 95th / max | 0.21 / 0.70 / 0.89 mm | 0.13 / 0.37 / 0.42 mm |
| set 9131: stated expanded uncertainty, median / max | 1.24 / 3.06 mm | 1.07 / 2.07 mm |
| set 9132, 12 layouts: guidance / preview | 3 / 9 | 6 / 6 |
| set 9132: corner error, median / 95th / max | 0.15 / 0.34 / 0.38 mm | 0.10 / 0.53 / 0.61 mm |
| set 9132: stated expanded uncertainty, median / max | 1.22 / 2.10 mm | 1.07 / 1.61 mm |
| rotation error, largest | 0.04° | 0.04° |
| error within the stated uncertainty | 24 of 24 | 24 of 24 |
| largest error ÷ stated uncertainty | 0.45 | 0.61 |
| wrong result shown as guidance | 0 | 0 |
| time per capture, median: tags / registration / datums / overlay | 0.6 / 0.0 / 0.4 / 0.2 s | 3.4 / 1.1 / 11.7 / 0.4 s |

The nominal figures are the same as in the first run, to the last digit
shown: none of the changes touches a sound capture. No nominal layout has a
tag set aside, a raised edge beside a datum or an unresolved contradiction;
one has a single observation left out.

Why most results are previews at this photo size: the stated uncertainty
exceeds half of the sample record's ±2 mm tolerance in 18 of 24 single-photo
and 15 of 24 multi-photo results, and in 3 multi-photo results a second line
of similar strength makes the side datum a question for the operator (in the
layout that was looked at, the outer edge of the panel, which runs parallel
to the seam 400 mm away). The
largest term of the budget is the bound on residual edge shift in 24 of the
48 results, the per-tag size measurement in 12 (0.05 mm on a 50 mm tag is
0.1 % of every distance from the tags), pixel noise in 8 and the bound on
panel flatness in 4.

By in-camera processing, both sets together (corner error, median / max):

| Processing of the photo | single photo | all photos |
|---|---|---|
| none | 0.16 / 0.22 mm | 0.08 / 0.37 mm |
| sharpened 0.6 | 0.13 / 0.35 mm | 0.08 / 0.20 mm |
| tone curve | 0.16 / 0.30 mm | 0.13 / 0.22 mm |
| sharpened 0.9, contrast curve, JPEG 85 | 0.34 / 0.89 mm | 0.38 / 0.61 mm |

The heaviest processing accounts for the largest errors in both modes. Tag
borders are corrected for the edge shift it causes; datum edges are not, and
the budget carries a bound for that.

Fault cases after the changes (17 of 17 handled, none shown as guidance, no
failure), with the reason beside the verdict:

| Case | Outcome | Reason shown |
|---|---|---|
| moved tag | preview, error 0.09 mm (stated ±1.79) | tag 58, the one that moved, set aside; the other three measure |
| one size wrong | preview, 0.11 mm (±1.11) | the mis-sized tag set aside |
| raised tag | preview, 0.16 mm (±1.98) | the raised tag set aside; datums on the face of the other three |
| leaning side edge | blocked | `local_frame_valid`: the datums do not meet at the record's angle |
| proud trim | preview | `datum_floor_no_step_beside`; the outline is 61 mm from where the hidden floor line would put it, and the operator is asked which line is meant |
| glossy tags, flash | preview, 0.64 mm (±2.21) | two tags set aside, and `photos_agree`: contradictions remain |
| all sizes wrong | recorded only: 17.87 mm at a stated ±1.35 | nothing: no check on the photos can see it. A preview here only because of the uncertainty. `scale_source` now says so on every result without a rule measurement |
| the same with one rule measurement | blocked | `independent_length_check` |
| no tag; floor outside the photos; faint seam | refused or blocked | no scale; a datum is missing |
| no measured sizes; twin seams; curved target; draft record | preview or blocked | the check that belongs to each |
| duplicate ID; correct sizes with a rule | preview, 0.08 and 0.04 mm | only the uncertainty against the tolerance |

Coverage trials (errors drawn from what the budget says it covers): the
stated uncertainty contained the error in 39 of 40 single-photo trials
(largest error ÷ stated uncertainty 1.13) and in 10 of 10 multi-photo trials
(0.99).

## 2026-11-25 — Full resolution: a seam measured at one of its edges (prompts 04, 06, 13)

The benchmark had only ever been run at 1600x1200. Four layouts of the first
set were then run at 4032x3024, the size of a phone's full-resolution photo,
with scenes, camera positions and in-camera processing unchanged:

| First run at 4032x3024 | single photo, hand-drawn datums | all photos, automatic datums |
|---|---|---|
| corner error, median / max | 1.04 / 1.18 mm | 1.03 / 1.07 mm |
| stated expanded uncertainty, median | 0.70 mm | 0.65 mm |
| error within the stated uncertainty | 0 of 4 | 0 of 4 |
| rotation error, largest | 0.01° | 0.00° |

Every target was displaced by one millimetre without rotation, in both
modes: one cause for all. The side datum of these scenes is a seam 2 mm wide.
At 1600x1200 (1.36 px/mm) it is 2.7 px wide and is fitted as a thin line, at
its centre (measured offset 0.000 mm). At 4032x3024 (3.43 px/mm) it is 6.9 px
wide. The snap looks for the strongest transition, which is one of the seam's
two edges, and the edge model fits that edge without complaint. The datum was
1.00 mm from the seam's centre: half its width. The floor line, an edge, was
within 0.001 mm in the same photo.

This is not a peculiarity of rendered photos. A joint a few millimetres wide
is resolved by any 12-megapixel photo taken from arm's length; on a phone it
would have been the normal case, and the error was outside the stated
uncertainty.

Change:

* `fit_trench` in `lumen/vision/edgefit.py`: a box of half-width `h` blurred
  by `sigma` on a base level, that is, two opposite edges with the same level
  outside both. Accepted only when the edges are resolved (`h >= 1.2 sigma`);
  a thinner line stays with the thin-line model.
* When a snap has chosen an edge, it looks across that edge for a partner of
  the opposite direction and comparable strength within 6 mm
  (`max_line_width_mm`), with the same level outside both. If there is one,
  the two are fitted together and the centre line is returned as the line;
  the single edge stays available. An edge between two different surfaces has
  different levels on its two sides and stays an edge.
* Used in the automatic search, in finding a line again in the other photos
  and in the manual snap. The datum card says "a line about 2.0 mm wide,
  resolved into its two edges: the datum is its centre line".
* `docs/CONVENTIONS.md` now states which line of a feature is the datum.

After the change, the same photos (run of 2026-11-25 20:19 UTC,
`docs/benchmark/RESULTS_FULL_RES.md`):

| After the change, 4032x3024 | single photo, hand-drawn datums | all photos, automatic datums |
|---|---|---|
| guidance / preview | 4 / 0 | 2 / 2 |
| corner error, median / max | 0.03 / 0.14 mm | 0.02 / 0.12 mm |
| stated expanded uncertainty, median / max | 0.70 / 0.90 mm | 0.65 / 0.68 mm |
| error within the stated uncertainty | 4 of 4 | 4 of 4 |
| largest error ÷ stated uncertainty | 0.15 | 0.18 |
| time, median: tags / registration / datums / overlay | 1.3 / 0.1 / 2.4 / 0.2 s | 7.1 / 1.3 / 91 / 0.4 s |

The two multi-photo previews are the two layouts in which the outer edge of
the panel competes with the seam for the side datum; they are previews at
1600x1200 for the same reason. At this photo size the largest term of the
budget is the caliper uncertainty of the tag sizes in seven of the eight
results.

At 1600x1200 nothing changed: both sets and the fault cases were run again
after this change, and every figure of the entry above is the same.

What this is and is not. Four layouts, one seam width (2 mm, 6.9 px). Other
widths were tried as synthetic profiles only: lines 1 to 19 px wide at three
blurs, snapped from either edge, come out within 0.1 px of their centre
(`tests/unit/test_edgefit.py`); 4 to 5 px wide in a sharp photo is the worst
place, up to 0.08 px. Finding the datums automatically took 74 to 103 s per
capture at this size. And the fault itself says something about the
benchmark: one photo size had been taken for the method.

## 2026-11-25 — Deployment manifest; tests on the final code (prompts 14, 15)

`scripts/write_manifest.sh` as it stood after the last change did not run:
one of its exclusion patterns contained `**` twice, which the publishing tool
rejects. The sentence in the deployment guide about a manifest of 94 files
came from a run before those patterns were added. The script is corrected and
`manifest.json` (Python 3.12.3, pip 24.0, 102 files: `app.py`,
`requirements.txt` and the `lumen` package) is now in the repository; a test
fails when it no longer matches the deployed files. Publishing from it:
NOT RUN.

Tests on the final code:

| Suite | Result |
|---|---|
| unit and integration (`tests/unit`, `tests/integration`), 500 tests | 499 pass; 1 skipped, the opt-in call to a real language-model endpoint. In the complete run (13.9 min) 498 passed and the manifest test failed, as it should: the manifest had been written before the last change to the code. With the manifest written again, the ten deployment tests pass. |
| browser (`tests/browser`: Chromium, phone-sized window, fake camera), 37 tests | 37 pass (7.9 min) |
| Python 3.13.18 in a clean environment, 14 test files (gates, datums, edge fits, multi-view, geometry, units, library, planar solver, tags, local frame; datum, multi-view, deployment and library flows) | 249 pass |

The code under test did not change between the start of these runs and the
commit (checked by a hash of the difference to the last commit).

## 2026-11-25 — Live placement: the placard is measured while it is being positioned (prompts 09, 10, 11)

The workflow so far ended with "place it, then take a photo and see". That is
measuring after the fact, and what it finds is rework. The requirement is the
other way round: the camera watches while the placard is positioned, says
which way to move it, turns green when it is right and announces it.

It was not a redesign. Identify, Markers, Capture, Datums and Rectified are
unchanged, and so are registration, datums, the uncertainty budget and the
verification. What changed is the last third: the Place step now runs a loop,
and the Verify step mostly shows a record that already exists.

### What a live frame answers

`POST …/track?placement=true` registers the frame against the tag layout as
before and then measures the placard in the same frame with the outline
measurement of the verification (`vision/placard.py`). The answer says where
the placard is relative to its target, which way to move it on the screen,
and whether it is in position. "In position" is the rule of a verification:
deviation plus expanded uncertainty of this frame inside the tolerance of the
record. Nothing is stored.

Three things are particular to a placard that is still in a hand, and each
was settled by asking what the picture can and cannot show:

* **It may not lie on the surface.** A placard 8 mm off the panel, 640 mm
  from the camera, is displaced in the picture by 8 mm times the tangent of
  the viewing angle. Height shows as size: it measures 1.3 % too large. A
  placard that does not measure its size is not judged, and the height that
  the size test cannot exclude (2.6 mm at 1280×720 and this distance) enters
  the uncertainty as a bound that grows with the viewing angle: 0.35 mm
  square-on, 2.2 mm from 33° to the side, which alone is more than the
  tolerance. From the side the answer is "too coarse here: hold the phone
  square to the placard", not a green that a lifted corner could fake.
* **Hands cover it.** See the next section.
* **Green must not be the record.** The record is a stored photo at the
  camera's full size, measured by the ordinary verification with all four
  sides, taken by the browser once the placard has been in position with
  nothing covering it for three answers in a row and at least one second. It
  is announced as recorded only if that measurement is inside the tolerance
  too; a photo that does not confirm it is deleted again.

### Hands

The first version stopped reading as soon as a side was covered, which is the
normal state of a placard that is being positioned. Each of the following was
found with a probe that puts skin-coloured boxes over the rendered placard,
and each is now a test.

| What happened | Why | Change |
|---|---|---|
| a hand over a whole side: no reading | three sides are not an outline | a hidden side is placed from the opposite, measured side and the record's size (`complete_from_size`). It costs uncertainty: 1.0 mm is allowed for the part's own size, and the size (lying flat) can then be checked in one direction only |
| the hand's straight edge, 12 mm inside the placard, was measured as the side: 15 mm "too small", position 4 mm off | a side that is not found near the outline was looked for three times further away (meant for parts of another size) | not in live frames: there a side that is not where it should be is hidden |
| a side placed where nothing covered it (part 20 mm narrower than the record) | placing by size assumed a hand | a side is placed only where the picture shows something over it. Where the place looks like bare panel or bare placard it is not placed, and three sides with a fourth that is nowhere give "is this the right part?" (`plain_fraction`) |
| with a side hidden, the outline was the printed border 3 mm inside the edge | the border's inner edge has the other sign; with one true side missing it offered as many sides | among incomplete candidates the one whose opposite sides are the record's distance apart is the placard, before the number of sides counts |
| the ends of the three text bars passed for the left side | a third of a side's length is enough in a live frame | an edge with the placard's own colour outside it is not a side; nor is one with something that is not the panel outside it (`foreign_sides`) |
| a search with a hand in the picture found two places alike and was "not clear" | the search only reported rivals | every rival place is measured the same way; a place is the placard only if its outline is complete. Rivals that lock onto the same outline confirm it |
| the follower lost the placard after 4 to 5 mm of movement, although it scans ±7 mm | the noise level was taken from the outer end of each profile, where the edge now was | the calmer of the two ends is taken; 6 mm is followed |
| after a larger jump one side was not found, was placed by size and drawn as covered, though it was in plain view | placing by size did not look whether the side is to be seen | a placed side that shows the placard's own edge is measured from where the outline now is (`edge_fraction`) |
| followed with the wrong sign after a side went under a hand | both signs were tried in every frame | a placard that is followed keeps the sign of its edges |
| everything above worked in the tests and not in the browser | the browser's video path stores colour at half resolution; the first pixels inside an edge then carry the smeared colour of the border, differently on every side | the panel is sampled 3 to 6 px outside a side, the placard 6 to 10 px inside; hue is compared closely and brightness coarsely (a hand also shades what is next to it) |

What still gives no reading, on purpose: both ends gripped completely
("The right and left sides are both covered: uncover one of them"), and two
neighbouring sides covered on a placard that was not seen whole a moment
before. With two neighbouring sides placed by size the position is shown but
not judged, because nothing then says whether it lies flat.

### What it costs per frame

A live frame used to cost a second. Two things are carried from frame to
frame, and both are said in the answer: the edge shift of the tag borders is
a property of the stream, so it is measured on one frame and handed back by
the client with the next ones (refreshed every five seconds); and the placard
is followed from where it was. Measured on the development machine (shared
cores, one server process, frames 1280×720):

| | server time |
|---|---|
| first frame (edge shift, uncertainty, search) | 0.9 s |
| followed frame | 0.17 s |
| no placard in view (searched every frame) | 0.27 s |
| a covered placard that has to be searched for again (search plus rivals) | 0.42 s |
| through headless Chromium on the same machine | about 5 results per second, each shown 0.2 s after its picture was taken |
| from "in position, hands off" to the chime | about 2 s (one second of rest, then 0.7 s for the photo and its measurement) |

Frames are sent at 1280 px in placement, not 1600: the budget of a live
reading is not limited by pixel size (1.074 mm at 1280×720 against 1.052 mm
at 1600×900 in the same view).

### What limits "in position"

Read from the budget of a live reading, same panel, ±2 mm tolerance:

| Registration photos | Live reading | Room left for "in position" |
|---|---|---|
| with a lens profile, placard whole | ±1.07 mm (edge shift 0.44, tag size 0.35 and 0.20, lifting 0.35, flatness 0.33, live lens 0.16) | ±0.93 mm |
| with a lens profile, one side under a hand | ±1.41 mm (lifting 0.70, because the size is then checked in one direction only; size of the part 0.50) | ±0.59 mm |
| without a lens profile | ±1.84 mm (uncharacterised lens 1.36) | ±0.16 mm |

The 1.36 mm is not the live picture's lens. It is the allowance of 1 %
distortion acting on the datum lines of the registration photos, which run
out to the picture's edges. The live frame's own share is 0.16 mm, because
the placard sits among the tags. So the profile that matters is the one for
the photos, and a phone without one will hang at "almost" on a ±2 mm
tolerance. The answer now says so ("It counts as in position only within
±0.2 mm …: uncharacterised lens distortion") instead of leaving the
operator to guess. Estimating the distortion from the capture itself would
remove the largest term for every uncalibrated phone; not done.

### On the phone

The live view shows each analysed frame with the placard's measured outline
(a side placed by size is dashed) and an arrow to the target. Beside it: a
dial on which the ring is the tolerance and the disc is the placard, as large
as the uncertainty, so that "in position" is "the disc is inside the ring";
a level for the rotation; the distance and the turn in large figures; one
line that says what to do. The eyes are on the placard, so the state is also
heard: beeps that speed up, a steady tone in position, a chime for the
record. Every sound follows a reading and stops by itself 1.3 s after the
last one, so a lost connection cannot leave "in position" sounding.

### Found on the way

* **Deleting a verification photo discarded the registration.**
  `remove_frame` invalidated everything for any photo. On the Verify screen
  "Delete this photo" therefore threw away the analysis and all other
  verification results; the browser test only looked at the list of
  verifications, which was empty either way. A verification photo now takes
  only its own result with it.
* **A state class named like an existing one.** The live panel in its
  "guide" state took the absolute positioning of the capture screen's
  `.guide` hint and lay over the picture. Seen in the first screenshot;
  state classes now carry a prefix.
* **The test deployment on a function platform.** The screenshots that came
  with the request showed broken photo tiles and "Unknown or expired
  session". The platform's logs showed why: requests arriving together were
  served by separate, freshly started instances, each with an empty disk.
  LUMEN keeps a session with its process; it cannot work there, and live
  placement, which asks one process several times a second, even less. The
  application now recognises such a host, says so on the first screen and in
  the error, and reports it in the health check
  (`docs/POSIT_CONNECT.md`, section 5). A photo that cannot be loaded is a
  labelled tile instead of the browser's broken-image mark.

### Tests

`ALGORITHM_VERSION` is now `lumen-algo-2026.11.3`. The benchmark was not run
again: it measures registration, datums and the target, and none of that code
changed (the changes are in the placard outline, the live path and the
deletion of photos). The outline measurement of the verification changed in
two places that can alter a result (the noise level of a side profile, and
which incomplete outline is preferred); its tests pass unchanged.

| Suite | Result |
|---|---|
| unit and integration | 521 tests: 520 pass, 1 skipped (the opt-in call to a real language-model endpoint); 13.4 min |
| browser (Chromium, phone-sized window, fake camera) | 42 tests: 42 pass; 9.0 min |

New: 13 tests of the live placement through the API, 6 of the outline with
hands, 4 browser tests with films of a placard (arriving and resting; far
away; under a hand; absent), one of the deletion, and two of the host notice
(API and browser).

### Not run

No phone, no hand, no placard, no loudspeaker. The hands are boxes, the
placard is a rendered rectangle with a border and three bars of print, the
video is a film in a fake camera. Whether real skin is told from a real side,
whether the tones are heard in a workshop, how many answers per second a
deployed server gives over a real network: `docs/PHYSICAL_TEST_PROTOCOL.md`,
section 8.

## 2026-11-26 — A phone that stands still: the registration is held (prompts 05, 09)

Five readings a second, each 0.2 s old, is a slide show beside a hand that
moves a placard. Of the 0.17 s a frame cost, 0.12 s went into measuring the
tags and redrawing what follows from them, in every frame, although during
placement the phone is on a stand and the tags are in the same pixels each
time. And whenever an arm covered a tag the frame was registered on what was
left, with a different result.

### The check

The registration of one frame is kept, with the picture around each tag
(`vision/steady.py`). A later frame is asked one question per tag: is this
still the same picture, in the same place? The answer is a least-squares fit
of the kept patch to the new one with four unknowns: a shift in x and y, a
gain and an offset. Measured on rendered frames through JPEG:

| The new frame | Shift found (px) | Explained (R²) |
|---|---|---|
| the same view, other noise, the placard elsewhere | 0.000 to 0.005 | 1.000 |
| the picture shifted by 0.05 / 0.1 / 0.3 / 1.0 px | 0.05 / 0.10 / 0.30 / 1.03 | 1.000 |
| shifted by 3 px | 3.3 | 0.98 |
| 20 % brighter, 30 % darker | under 0.006 | 0.998 or more |
| an arm over two tags | none for those two | 0.00 for those two, 1.00 for the third |
| the camera 0.1 mm / 1 mm to the side (re-aimed) | 0.02 / 0.12 to 0.23 | 1.000 |

Three tags are compared in about a millisecond. A shift of a few pixels is
recognised by its size, not by a poor fit: three pixels are still explained
to 98 %. What must not happen is a large shift that looks like a small,
well-explained one. In 504 comparisons of pictures shifted by 0.6 to 30 px in
twelve directions none did: wherever the estimate came out under half a
pixel, at most 2 % of the picture was explained.
(The first unit test of this failed by 20 %. The estimate was right: the test
shifted its pictures with an OpenCV warp, which places a picture only to
1/32 px.)

### The rule

* At least two tags compare (R² of 0.97 or more) and none has moved more than
  0.1 px: the kept registration is used as it is. The frame is *held*.
* They have moved up to 0.5 px and at least three agree on how (a similarity,
  to 0.06 px): the registration is moved along. This is a stand that gives a
  little when the panel is touched.
* One tag alone cannot show a movement that leaves that tag where it is. It
  is accepted for three seconds after two tags last agreed (an arm passing).
* Anything else: the tags are measured afresh, and that frame becomes the new
  reference. A held registration is also measured afresh every five seconds
  when no tag is covered; the uncertainty of the view is then taken over,
  because the same tags in the same pixels have the same uncertainty.
* A tag that cannot be compared is not asked: covered, moved, or changed, it
  takes no part, and the registration made when it was in view stands.

The answer says that the frame was held, by which tags, which were covered
and how far they had moved; the reading's budget has a term for what the
check cannot see (0.1 px, or 0.06 px when moved along: 0.07 mm here).

### What it buys

Server time per 1280×720 frame, development machine:

| | measured in every frame | held |
|---|---|---|
| placard followed | 0.17 s | 0.04 s (0.03 s when it rests: one measuring pass is enough for a placard that has not moved) |
| no placard in view | 0.27 s | 0.14 s |
| a covered placard searched for again | 0.42 s | 0.33 s |
| in a browser on the same machine | 5 readings a second, 0.2 s old | 12 a second (the page asks for no more than one every 80 ms), about 40 ms old |

A phone standing for 16 s: 262 frames, 258 of them held at a median of 42 ms;
the routine re-measurements took 0.16 s each, the first frame 0.93 s and one
more 0.49 s (the first re-measurement, whose tags were measured in another
way than those of the first frame, so that its uncertainty was evaluated
again).

And what the speed does not show: with an arm over two of the three tags in
view, the held reading is the one made with all three; measured afresh, the
same picture has one tag and cannot say "in position".

### Limits

The check sees the picture around the tags. A change that leaves it in place
and still moves the target in the frame is not seen. For a rigid panel and a
rigid phone there is none; a panel that flexes between the tags when it is
pressed would be one. The held registration lives in the memory of one
process; several processes each keep their own, which is correct and slower.

### Tests

| Suite | Result |
|---|---|
| unit and integration | 535 tests: 534 pass, 1 skipped (the opt-in call to a real language-model endpoint); 14.5 min |
| browser | 42 tests: 42 pass; 9.0 min |

New: 9 tests of the patch comparison, 5 of the held registration through the
API (a standing phone; an arm over one and over two tags; a phone moved 2 mm;
a stand that gives 0.2 mm; a tag knocked aside and a change of placard), and
the browser test of a far placard now also requires that the film's standing
camera is recognised as such.

Not run: a phone on a real stand. How still a clamped phone is, in pixels,
while somebody presses on the panel beside it, decides how often this applies.
