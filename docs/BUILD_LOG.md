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
