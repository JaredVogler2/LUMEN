# LUMEN

Browser-based optical measurement and placard positioning. A phone's rear
camera, a few commercially available AprilTag stickers of measured size, and
two datum features on the panel are enough to show where a placard goes and,
afterwards, to measure where it is. No CAD model, no native app, no LiDAR, no
NFC, no extra optical hardware.

> **Research prototype.** Everything LUMEN shows is measurement evidence, not
> product acceptance. All accuracy figures in this repository come from
> computer-generated photos. Nothing has been measured with a real phone, a
> real print or a real panel, and nothing has been deployed to a Posit Connect
> server: those steps are written down and marked NOT RUN
> (`docs/PHYSICAL_TEST_PROTOCOL.md`, `docs/POSIT_CONNECT.md`).

## What it does

| Step | What happens |
|---|---|
| Identify | choose the operation and placard, or scan a bar code / QR code (exact matches only); records can be entered and revised in the app |
| Markers | enter the caliper-measured edge of each tag's black square; scale comes from nothing else |
| Capture | photos, or a video from which sharp, well-spread frames are selected; coaching on distance, angle, glare, coverage |
| Datums | floor line and side line are found and fitted to sub-pixel edges; a rounded side edge is modelled from its outline in several photos; the operator confirms |
| Rectified | metric straight-on view with a grid from the datums; optional rule measurement that catches a wrong tag size |
| Place | the target outline on the photo and on live video, with distances, uncertainty and a status that says what was and was not checked |
| Verify | the installed placard's outline is measured and compared with the target: signed deviations with uncertainty; export; delete |

How it stays honest:

* the frame of reference is built only from the datums and signs the record
  defines; gravity, the phone's sensors and "up in the picture" are not used;
* tags are placed anywhere on the flat face; their layout is solved, cross-checked
  between tags and, with several photos, checked for coplanarity in 3-D;
* missing or contradictory geometry gives "no guidance" with a reason and a
  next step, never a guess; no datum, offset or tolerance is invented;
* every result carries an uncertainty budget and the list of checks behind its
  status; simulated data is labelled on every screen;
* a flashlight (photometric) analysis of rounded edges and an optional
  language-model assistant exist as clearly separated experiments that never
  feed the guidance.

## Run it

Python 3.12 or 3.13 (the pinned NumPy does not install on older versions).

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

```bash
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> and choose **Load demo session** to walk
through all seven steps on simulated photos. To use a phone's camera the page
must be served over HTTPS; deploying to Posit Connect provides that
(`docs/POSIT_CONNECT.md`).

Tests, benchmark and smoke test:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python scripts/run_benchmark.py
```

```bash
.venv/bin/python scripts/smoke_test.py http://127.0.0.1:8000/
```

## Documents

| Document | For |
|---|---|
| `docs/OPERATOR_QUICKSTART.md` | using the app, step by step |
| `docs/TROUBLESHOOTING.md` | messages and what to do about them |
| `docs/POSIT_CONNECT.md` | publishing on Posit Connect, settings, storage, smoke tests |
| `docs/DEVELOPER_HANDOFF.md` | architecture diagram, module map, rules, what is not done |
| `docs/CONVENTIONS.md` | units, pixels, frames, labels, status words |
| `docs/VALIDATION.md` | what was tested, how, and what the results do and do not show |
| `docs/benchmark/RESULTS.md` | synthetic benchmark results (generated) |
| `docs/PHOTOMETRIC_ABLATION.md` | flashlight experiment against the edge-only method |
| `docs/PHYSICAL_TEST_PROTOCOL.md` | the physical study that still has to be run |
| `docs/ASSISTANT.md` | the optional assistant: what is sent, how replies are checked |
| `docs/BUILD_LOG.md` | what was built, measured results, and what did not work |
| `prompt_series/` | the requirements this implementation follows |

## Data in this repository

Only generic, invented content: synthetic panel images, a sample operation
library marked as synthetic, no real parts, dimensions, photos, credentials
or internal services. Use an approved environment for real manufacturing
data, and validate physical accuracy, surface compatibility and tooling
controls before any production use.

## Original brief

The three sections that follow are the brief this repository started from, kept word for word. What was actually built, measured and left undone is described above and in `docs/`.

## Build with Fable 5.2
Read every file in `prompt_series/` before coding. Execute in numeric order, starting with 00_MASTER.md. Maintain a working app at each milestone. Improve the proposed algorithms and UX when evidence supports it; document decisions and test results. Do not stop at design documents or pretend unimplemented features work.

Target: single-origin FastAPI web application deployed to Posit Connect, CPU-first Python/OpenCV/NumPy/SciPy and locally bundled JavaScript/HTML/CSS. Verify target platform dependencies and browser behavior. Use synthetic fixtures and generic configurations only; no employer data, photos, credentials or internal requirements in this repository.

## Workflow
Identify operation, place 2–4 distinct known-size AprilTags near the target, capture overlapping images or video of the side boundary/curvature and floor interface, establish scale and local geometry, confirm datums, calculate specified target corners, show registered placement guidance, then measure observed placement from a verification image. Photometric analysis is a research extension to reliable planar metrology. Require explicit coordinate definitions and fail closed on ambiguous or unvalidated geometry. All outputs are prototype guidance, not product acceptance.

## Reference links
- https://docs.posit.co/connect/user/fastapi/
- https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/getCapabilities
- https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- https://github.com/AprilRobotics/apriltag
- https://texxary-shop.com/products/april-tags-sticker-sheet
- https://arxiv.org/abs/2303.14190 (research background)

Use only approved environments for real manufacturing data and validate actual physical accuracy, surface compatibility and tooling controls before production use.
