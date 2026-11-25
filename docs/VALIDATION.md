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
| Browser tests | the interface on a phone-sized viewport with a fake camera: walkthrough, live guidance, verification, scanning, record editor, calibration, sub-path | `tests/browser` | `python -m pytest tests/browser` |
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

## 6. What has not been tested at all

| Item | Status |
|---|---|
| Accuracy, repeatability and coverage on real hardware | NOT RUN |
| Any iPhone: camera, torch, recording, live guidance in Safari | NOT RUN |
| Lens calibration with a real print | NOT RUN |
| Reading real bar codes at working distance | NOT RUN |
| Posit Connect deployment | NOT RUN |
| More than one operator | NOT RUN |
| The optional assistant against a real endpoint | NOT RUN |
