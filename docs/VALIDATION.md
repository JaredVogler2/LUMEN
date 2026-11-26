# Validation

**What exists: software tests and a benchmark on computer-generated photos.
What does not exist: any measurement with a real phone, a real print or a real
panel.** LUMEN therefore has no validated domain, its status `validated`
cannot occur, and nothing below is evidence of physical accuracy. The
physical study is specified in `docs/PHYSICAL_TEST_PROTOCOL.md` and is NOT RUN.

## 1. Layers of evidence

| Layer | What it shows | Where | Run |
|---|---|---|---|
| Unit tests | each algorithm against ground truth it cannot see: tag corners, edge fits, planar solver, bundle adjustment, datums, placard outline, calibration, video selection, codes, units, gates | `tests/unit` | `python -m pytest tests/unit` |
| Integration tests | whole flows through the HTTP API on rendered photos, compared with the renderer's truth; refusals; limits; storage; sub-path deployment | `tests/integration` | `python -m pytest tests/integration` |
| Browser tests | the interface on a phone-sized viewport with a fake camera: walkthrough, live guidance, live placement (a film of a placard brought to its target), verification, scanning, record editor, calibration, sub-path | `tests/browser` | `python -m pytest tests/browser` |
| Benchmark | accuracy, stated uncertainty and fault handling on layouts the tests never use | `lumen/validation/benchmark.py` | `python scripts/run_benchmark.py` |
| Photometric ablation | the experimental flashlight analysis against the edge-only method | `scripts/photometric_ablation.py` | see `docs/PHOTOMETRIC_ABLATION.md` |
| Smoke test | a running instance end to end | `scripts/smoke_test.py` | `python scripts/smoke_test.py <address>` |

Test counts and the results of each run are in `docs/BUILD_LOG.md`.

## 2. The synthetic photos

`lumen/synth` is a ray caster written separately from the measurement code
(its own camera mathematics; a test cross-checks the two). It renders a flat
panel with seams, a floor at an angle, an optional rounded side edge, proud
trim, AprilTag stickers anywhere on the panel (optionally raised or glossy),
placards, a torch near the lens, and calibration boards.

Modelled: perspective and lens distortion (five coefficients), principal
point offset, exposure and auto-exposure, lens blur, motion blur, shot and
read noise, vignetting, the sRGB curve with optional extra tone curve and
contrast, sharpening halos, JPEG compression, specular highlights, cast
shadows of simple blocks.

The benchmark photos are rendered with nine samples per pixel on a lattice
that is shifted from pixel to pixel. A regular sample grid, which the unit
and integration tests still use, displaces edges at 45 degrees and similar
slopes by up to 0.2 px; in a benchmark that looked like tags that had moved
(`docs/BUILD_LOG.md`).

Not modelled: rolling shutter, focus breathing and refocusing between photos,
optical image stabilisation, chromatic aberration, local tone mapping and
multi-frame fusion of phone cameras, lens dirt, real surface textures and
curvature of "flat" panels, sticker lift and stretch, mixed or flickering
light, the operator's hand.

Every bound in the uncertainty budget that stands for something on the second
list (residual edge shift, lens model beyond the profile, panel flatness,
edge definition) is an assumption. On synthetic photos these bounds make the
stated uncertainty several times larger than the true error. That margin says
nothing about a real phone.

## 3. Held-out configurations

Thresholds were chosen while looking at one tag layout (IDs 3, 7, 11, 20) and
one family of camera sweeps. The benchmark draws its own: other IDs,
positions and rotations of three to five tags, three to seven camera
stations at other distances and heights, other lens distortion, blur, noise
and exposure, and four kinds of in-camera processing, from a seeded generator
(seed 9131).

The set was not untouched. Before the first full run, six of its layouts were
run at a reduced size and exposed two faults in the multi-photo path, which
were then fixed (`docs/BUILD_LOG.md`, "Held-out layouts find two faults").
Reading the fault cases of the first full report later led to four more
(same file, "First full benchmark: four faults behind 'handled'"). The report
therefore also contains a second set from another seed (9132). No change was
ever made because of a result of that set: it was run once before anything
of it had been seen, and once more with the final algorithm version. Compare
the two
sets.

## 4. What the benchmark measures

Two modes on the same photos:

* **single photo, hand-drawn datums:** one photo; the datum lines are drawn
  with a few pixels of error and fitted to the edge;
* **all photos, automatic datums:** every photo; datums found automatically
  and checked in 3-D.

Four layouts of the first set are also run at 4032x3024, the size of a
phone's full-resolution photo (`RESULTS_FULL_RES.md`). A feature can look
different at another photo size. The first run at full resolution had every
target 1.0 mm off and outside its stated uncertainty: the 2 mm side seam is
seven pixels wide there, it was taken as two edges, and one of them became
the datum (`docs/BUILD_LOG.md`). Nothing at 1600x1200 could have shown that.

For each: the largest target corner error, the centre (translation) error and
the rotation error against the renderer's truth; the status; the stated
expanded uncertainty and whether it contains the error; false acceptance
(shown as guidance although the error exceeds both the uncertainty and the
tolerance); unexpected failures; and time per stage.

Fault cases, each with what a correct system does:

| Fault | Expected |
|---|---|
| no tag on the panel | refuse |
| one tag moved between photos | set exactly that tag aside and measure with the rest, or refuse |
| one tag declared 2 mm too large | set exactly that tag aside and measure with the rest, or refuse |
| all tags declared 2 mm too large | cannot be seen from photos: recorded as a wrong result with a clean report |
| the same, with one rule measurement between two tags | blocked |
| correct sizes with a rule measurement (control) | guidance |
| no measured sizes entered | no guidance |
| a second identical seam beside the datum seam | no guidance without the operator's choice |
| floor line in no photo | no guidance; the line is not invented |
| side edge leaning 5 degrees | blocked, because the two datums do not meet at the record's angle |
| one tag on an 8 mm shim | set exactly that tag aside and measure with the rest, or refuse |
| side seam barely visible | accurate or refused |
| proud trim hiding the floor line; the trim's upper edge is on the panel face | no guidance until the operator has chosen the line |
| two stickers with the same ID | accurate or refused |
| glossy tags under a strong flash | accurate or refused; contradictions that remain keep it a preview |
| record declares a curved target surface | blocked |
| record never released | preview only |

A safe outcome is not the same as a right one. At the benchmark's photo size
the stated uncertainty alone often makes a result a preview, and "blocked"
can have a wrong cause. The first full report counted four cases as handled
that were not right inside: a trim's upper edge taken for the floor line, a
reference plane that still contained the tag that had been set aside, a
moved tag that was kept while two sound ones were thrown out, and a floor
line that was in every photo and was not found (`docs/BUILD_LOG.md`). The
judging was tightened afterwards:

* for seven cases (undeclared sizes, twin seams, leaning side edge, proud
  trim, rule measurement, curved target, unreleased record) the case counts
  as handled only when the check that belongs to the fault is among the
  failed ones;
* where the fault is about one tag, a drawn result counts only when that tag
  was set aside, and the report shows whether exactly that tag was left out
  (✓) or a sound one went with it or instead of it (✗).

A refusal still counts as handled in every fault case: refusing is safe. The
"set aside" column is there so that a refusal for the wrong reason is seen.

Coverage trials repeat one layout with errors drawn from what the budget
says it covers (tag sizes, lens profile, image noise, the operator's hand) and
count how often the stated uncertainty contains the error.

## 5. Results

The numbers are in `docs/benchmark/` (generated; do not edit):

| File | Content |
|---|---|
| `RESULTS.md`, `results.json` | first set (seed 9131): 12 layouts, the fault cases, coverage trials |
| `RESULTS_SECOND_SET.md`, `results_second_set.json` | second set (seed 9132): 12 layouts |
| `RESULTS_FULL_RES.md`, `results_full_res.json` | four layouts of the first set at 4032x3024 |

They are restated in `docs/BUILD_LOG.md` with the date of the run. Rerun the
benchmark after any change that bumps `ALGORITHM_VERSION`.

How to read them:

* An error smaller than the stated uncertainty on synthetic photos is
  expected (section 2) and is not a safety margin.
* "Guidance" in the benchmark means every computed check passed on a
  synthetic photo. It does not mean the result would be right on a panel.
* The one case that is wrong with a clean report (all tag sizes wrong in the
  same way) is a property of the method, not a defect of one run: scale comes
  only from the declared sizes. The rule measurement that catches it is
  optional in the interface.
* The largest errors of the nominal cases belong to the heaviest in-camera
  processing (strong sharpening, contrast curve and JPEG 85 together). Tag
  borders are corrected for the edge shift this causes; datum edges are not,
  and the budget carries a bound for it.
* "Handled" in the fault table is about the outcome. Read the "set aside" and
  "failed checks" columns for whether the reason was the right one.
* `RESULTS_FULL_RES.md` repeats four layouts at 4032x3024, the size of a
  phone's full-resolution photo. Part of the stated uncertainty at 1600x1200
  is pixel-related; the full-resolution run shows how much of it remains.
  Four layouts with one seam width are a spot check, not a study.
* Latency was measured on a shared four-core build machine.

## 6. Live placement: what the tests show

Live placement is not in the benchmark. It is covered by tests against the
renderer's truth, on the same panel as the other tests (target 120 × 60 mm,
±2.0 mm, ±1°; live frames 1280×720 from 640 mm, a viewpoint the registration
did not use):

| Question | Test | Result |
|---|---|---|
| Are the numbers those of the placard? | a placard at six known places on its way to the target (`tests/integration/test_placement_flow.py`) | deviations within 0.12 mm and 0.08° of the truth, with their signs; the arrow on the screen within 1 px of the true direction |
| Is "in position" the verification rule? | same sequence | outside at 35, 14 and 6 mm; neither at 2.4 mm (uncertainty ±1.1 mm against ±2 mm); in position at 0.7 and 0.2 mm |
| A placard held off the surface | a sheet of the placard's size 8 mm in front of the panel, exactly over the target | measures 1.5 mm too large; not judged |
| A height too small to show in the size | a sheet 2 mm off the surface, seen 30° from the side: displaced by about 1 mm in the picture | the displacement is inside the stated uncertainty; not called in position |
| A hand over a corner, a side, half the placard, the middle of both ends | boxes of skin colour, 40 mm thick, in the rendered frame | numbers within 0.3 mm of the truth; in position where the placard is, never "clean" |
| A hand's straight edge parallel to a side, 12 mm inside it | same | not taken for the side: the side is placed by the record's size |
| Both ends gripped, corners and all | same | no reading: "The right and left sides are both covered" |
| A part 20 mm narrower than the record | rendered placard | no reading that calls a side hidden: "is this the right part?" |
| No lens profile for the registration photos | same panel without a profile | uncertainty ±1.84 mm; the answer says that only ±0.16 mm of the tolerance is left, and why |
| Following from frame to frame | a seed up to 6 mm from the placard (`tests/unit/test_placard.py`) | all four sides found, centre within 0.05 mm |
| A jump of 5.5 mm between two frames | the follower misses one side, which is in plain view | that side is measured from where the outline now is; it is not drawn as covered |
| The whole loop in a browser | a film in a fake camera: the placard arrives and rests (`tests/browser/test_placement.py`) | told to move with the distance, green in position, the record taken by itself from a full-size frame and found within tolerance, the camera released; the tones commanded in this order: beeps, the chime of reaching it, the steady tone, the chime of the record |
| A reading that is no longer current | answers stop arriving while in position | the tone ends within 1.3 s and the reading is withdrawn |
| A phone that stands still (`tests/integration/test_hold_flow.py`) | the next frames from the same camera | no tag is measured; the reading is that of the same picture measured afresh, within 0.05 mm |
| An arm over one tag, then over two of the three in view | boxes over the tags | the registration made with all three is kept; measured afresh, the same picture has one tag and cannot say "in position" |
| The phone is moved 2 mm (3 px) | another camera position | measured afresh, right reading; held again from there |
| A stand that gives 0.2 mm (0.3 px) | another camera position | the three tags agree on the shift and the registration is moved along; the reading stays within 0.08 mm (unmoved it would be 0.2 mm off) |
| A tag knocked 4 mm aside while the phone stands | another scene | that tag is not asked; the reading is unchanged |
| The patch comparison itself (`tests/unit/test_steady.py`) | known shifts of 0.05 to 0.5 px, changes of light, a hand, another tag | shifts within 0.03 px; light changes nothing; anything else is "not the same picture" |

What these tests are not: a hand, a placard with a real surface, a phone's
video stream with its exposure and focus changes, a real network, or anybody
hearing a tone. See section 9.

## 7. The interface and an installation's brand

What an operator reads and how the application looks are tested in a browser
on a phone-sized window, not judged by anyone who will use it:

| Question | Test | Result |
|---|---|---|
| Is it said, on every screen, what kind of results these are? | the indicator in the header, its explanation, the watermark on the picture and the line beside the status (`tests/browser/test_workflow.py`, `test_camera.py`) | "Evaluation" with real photos, "Demo" with simulated ones; "Evaluation only: not validated" beside "Ready" and "Check needed"; nothing calls real photos simulated |
| Can the text be read? | contrast of body text and of the main button in both themes (`tests/browser/test_branding.py`, `test_workflow.py`) | 7:1 or better for body text; 7:1 (light) and 4.5:1 (dark) or better on the main button; touch targets 40 px or more, the main buttons 56 px |
| Does an installation's logo, name and colour appear, and nothing else? | a logo drawn by the test, a name, a dark and a light colour (`tests/browser/test_branding.py`, `tests/integration/test_branding.py`) | the logo loads from inside the content address and fits a phone's header; the colour is on the rule and the main button; a light colour is darkened for text and buttons to 4.5:1; a file of another kind, a missing or an oversized one and a colour that is not `#rrggbb` are ignored |
| Can a supplied logo do anything but be shown? | the headers of `brand/logo` | served as a picture with a policy that allows no script and no request |
| Is any organisation's artwork in the repository? | what git tracks under `brand/` and the application's own pictures | the note only; LUMEN's own mark only |
| Does the phone's screen stay on during live placement? | a wake lock that records what is asked of it (`tests/browser/test_placement.py`) | asked for when live placement starts, given back when it stops |

What these tests are not: an operator with gloves in a hangar, the light
there, or the judgement of the organisation whose logo it will carry.

## 8. A host that keeps nothing between two requests

On a function platform the browser carries the job (`docs/FUNCTION_HOSTS.md`).
The stand-in for such a platform in the tests is harsher than any real one:
every request is served by an application made for that request, and a
request or an answer above 4.5 MB is refused.

| Question | Test | Result |
|---|---|---|
| Does a job give the same numbers when the browser carries it? | the demo job on a server that keeps it and on the stand-in (`tests/integration/test_carried_flow.py`) | the same registration, lines, target on the photo, status and record, to the last digit that is compared (1e-6); 11 requests, the largest 1.1 MB |
| Do the existing tests hold with a carried job? | the tests of live placement (13), of the held registration (5) and of lens calibration (4), unchanged, through the carrier (`test_carried_placement.py`, `test_carried_hold.py`, `test_carried_calibration.py`) | all pass |
| Live frames against a job that every request brings anew | three frames of a placard arriving, each served by an application that has seen nothing | the right numbers, in position at the end; no photo travels with a live frame |
| A step that needs photos which were not sent | the search for the lines without them | names the photo it misses (428); nothing is changed by the failed request; sent photo by photo it gets there, and a client that knows the step sends them at once |
| A photo that is not the one that was measured | another photo under the frame's name | refused: "not the one that was measured" |
| A state that this server did not write | with a secret: one character changed, no signature, another server's secret; without: not a job, another job's, a form no job has | refused each time, as an error of that request |
| What does not fit | a request and an answer above the host's limit | refused in words that say what to do |
| In a browser (`tests/browser/test_carried.py`) | the demo; five photos of 2880 × 2160 from files; a reload in the middle; the straight-on view; live placement from a film to the automatic record | photos shown from the browser's own copies; each photo arrives at 1600 × 1200 and under 0.41 MB with all four tags measured; the job is back after the reload; five photos were sent once, for the search for the lines, and no step had to ask; after "End session" nothing of the job is in the browser |
| What it costs in time | live readings on one core, no network | 30 ms with the server keeping the job; 36 ms with the job carried and the same application answering; 1.3 s when every frame meets a new application |

Beyond the tests, the same client and Chromium with a fake camera were run
once against a preview deployment on Vercel (Hobby plan, from a data centre):
the demo job with the same numbers, a live reading every 0.15 s with 23 of 24
answered from a held registration, live placement from the film to the
automatic record in 9.6 s (`docs/FUNCTION_HOSTS.md` has the table). That is
one deployment on one day.

What none of this is: a phone; a mobile network; another function platform.
See section 9.

## 9. What has not been tested at all

| Item | Status |
|---|---|
| Accuracy, repeatability and coverage on real hardware | NOT RUN |
| Any iPhone: camera, torch, recording, live guidance, live placement and its sound in Safari | NOT RUN |
| Live placement with a real hand on a real placard: whether hands are told from sides, how fast it follows, whether "in position" agrees with a reference measurement | NOT RUN |
| Lens calibration with a real print | NOT RUN |
| Reading real bar codes at working distance | NOT RUN |
| Posit Connect deployment | NOT RUN |
| More than one operator | NOT RUN |
| The optional assistant against a real endpoint | NOT RUN |
| The interface with the people who will use it: whether six steps and these words are as plain as intended; the screen wake lock on a phone | NOT RUN |
| A function platform from a phone: Vercel was tried once from a data centre with a test client and with Chromium (section 8); a phone on a mobile network, and any other platform | NOT RUN |
