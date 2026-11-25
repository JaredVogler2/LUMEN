# Photometric (phone torch) module: ablation

Status: **research evidence only**. `lumen/vision/photometric.py` never
produces guidance, is not wired into any gate, and the planar workflow does
not need it. Everything below was measured on **synthetic photos** from
`lumen/synth` on 2026-11-25. No real photo, no real phone torch and no real
panel was involved; physical accuracy is **NOT RUN**.

Short answer (section 4.10 has the long one): on torch-on photos the
photometric fit is far better than silhouettes from those same photos (median
tangent error 0.04 mm against 0.58 mm); it is no better than silhouettes from
torch-off photos (0.09 mm against 0.02 mm); and on a surface with smooth
albedo variation it returns a wrong answer (about -1.1 mm) without refusing,
unless aligned flash-off partners are used in ratio mode.

**After this run** (same day): the application no longer measures heights
from the face of one gauge tag. It now uses the plane fitted through all
tags, which is the variant called "plane through all tags" in group G below;
the "tag pipeline" columns of the other groups were measured with the single
gauge tag and are therefore slightly pessimistic about the application as it
is now. The module is wired into the application as a research step
(`lumen/services/photometric.py`, a card on the Datums screen); its result is
stored and displayed and is never used for a datum, a check or a target. On
the test sweep through the application (four torch-on photos, true radius
35 mm) it reported radius 35.01 ± 0.11 mm and a start of the bend 0.34 mm
further out than the outline method on the same photos, 1.3 standard
deviations apart.

Reproduce the tables with

    .venv/bin/python scripts/photometric_ablation.py --write-doc

(renders are cached in `.pytest_cache/lumen_renders`; raw numbers go to
`.pytest_cache/photometric_ablation.json`). Unit tests:
`tests/unit/test_photometric.py`.

## 1. What is compared

* **(a) edge-only baseline.** Silhouettes of the rounded edge in several
  photos, cylinder fit (`fit_cylinder_from_silhouettes`), run through the
  real session service exactly as the application runs it (tag detection,
  registration, datum proposal). The configured radius of the demo operation
  is removed so that the baseline has to find the radius itself. Run on the
  same torch-on photos as the photometric module and, where stated, on
  ambient-only photos of the same sweep (**a'**). Its stated tangent sigma in
  the tables is the root sum of squares of the fit's `sigma_tangent_mm` and
  `rms_mm`.
* **(b) photometric.** `analyse_flash_sweep` on the torch-on photos with
  camera poses from the real tag pipeline (`detect_tags` +
  `solve_multiview`), a hint that is 5 mm and 1.5 degrees off, radius
  unknown. The plane frame handed to the module is the frame of the gauge tag
  that the session service picks (seen in most photos, then largest in the
  image), which is what `FrameContext.pose` of the service holds; group G
  compares other ways of fixing that frame. The module fits the tangent line,
  the axis direction and the radius together with per-view exposure, beam
  shape, lens fall-off, ambient level, torch position, gloss and an effective
  tone exponent, and decides from conditioning, variance estimates, a held-out
  view and a stationarity test whether the result is `supported`, `needs_view`
  (with a specific request) or `unsupported`.
* **(c) combined.** `combine_with_silhouette`: inverse-variance mean of (a)
  and (b) for tangent offset, direction and radius, *only* when the two agree
  within three combined standard deviations; otherwise nothing is combined and
  the disagreement is reported, i.e. the photometric estimate then acts as a
  consistency check. The two use the same photos and poses, so they are not
  independent and the combined sigma is optimistic by construction.

Errors are against the renderer's ground truth: tangent offset (mm, at the
middle of the analysed stretch, + toward the flat face), direction (degrees)
and radius (mm).

## 2. Protocol

* **Development set** (used to design the model, the masks and every
  threshold): the standard curved demo panel with radius 20, 35 or 60 mm, a
  sweep of 7 stations on a 62 degree arc at 760 mm (subsets of it), ambient 0
  to 0.8, torch 0.9 at 500 mm with beam exponent 1 placed 12 mm left of and
  9 mm below the lens, lens blur 0.8 px, low noise, no vignetting, mostly
  renderer poses; plus stress scenes (gloss, texture, paint strips, seams,
  vignetting, tone curves, a block casting shadows, flat panels, one-sided
  room light, flash-off partners). The unit tests use this set.
* **Held-out set** (the tables below; defined in
  `scripts/photometric_ablation.py::build_cases`): other camera rigs (6
  stations on a 54 degree arc at 680 mm, a 12 degree arc at 700 mm, a 60
  degree arc at 880 mm), other aim point and stretch of the edge, edge tapered
  by 1.5 degrees, torch 1.1 with beam exponent 2 placed 9 mm right of and
  13 mm below the lens, ambient 0.2, vignetting 0.15, blur 1.0 px, "mid"
  noise, other noise seeds, poses from the tag pipeline. These rigs, torch,
  lens and noise settings and seeds were not used during development. The
  *kinds* of stress (gloss, texture, strips, seams, tone curves, shadows,
  one-sided light, partners) are the same kinds as in the development set,
  with other parameters; the tag pipeline was used in development for one
  test and one check (section 5, item 16).
* The held-out set was started four times and completed twice.
  1. *First start*: stopped after 26 of 107 cases by a crash of the *script*
     (the tag pipeline finds no tag in the deliberately under-exposed photos
     and raises; the script now records that the photometric step is never
     reached). While it was running, three defects of the flash-off
     "difference" formulation and of the gloss term were found **on the
     development set** (section 5, items 11 to 13) and fixed. None of the 26
     cases has flash-off partners.
  2. *Second start*, with the final module: stopped by hand after 34 cases to
     add pose-attribution cases and the head-to-head table to the script.
  3. *First complete run* (115 cases). It showed that the photometric result
     depends on which tag defines the plane frame. The **script** was then
     extended: the gauge tag follows the rule of the session service in every
     group (before, groups without a baseline used the default of the
     multi-view solver), group G compares four ways of getting poses, and the
     error of the plane frame at the edge is recorded.
  4. *Second complete run* (127 cases): every number below.

  `lumen/vision/photometric.py` is byte-identical in starts 2 to 4. No
  threshold, default or model option was changed after any held-out result
  had been seen; the unfavourable held-out results are reported as they are
  (section 4). Statuses, reasons and printed numbers of the 26 and 34 cases of
  the first two starts are the same as in the final run, except for the
  880 mm rig (0.01 mm in the printed digits) and one 0.001 mm; for the
  comparison of the two complete runs see section 6, "Moving ground".
* Each case has one noise seed. Cases share the panel, the tag constellation
  and the rigs, so the coverage figures of section 3 are not from independent
  samples.

## 3. Results

<!-- BEGIN GENERATED TABLES (scripts/photometric_ablation.py) -->

Cells are `tangent offset error mm ± stated sigma / direction error deg / radius error mm ± stated sigma`, measured at the middle of the analysed stretch.  `[...]` marks an estimate that the module withheld (status not `supported`): it is shown to tell what trusting it would have cost.

### A. Radius, number and spread of views

| case | (a) edge-only, same torch photos | (a') edge-only, ambient photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|---|
| R 20 mm, 6 views, 54 deg | +0.74 ± 0.14 / -0.001 / +0.38 ± 0.23 | +0.01 ± 0.15 / +0.009 / +0.00 ± 0.23 | supported: overexposed_views | -0.02 ± 0.10 / +0.003 ± 0.010 / -0.03 ± 0.06 | 1.1 | not combined: disagree (z = 4.3) ; -0.01 ± 0.08 / -0.03 ± 0.06 |
| R 20 mm, 4 views | +0.71 ± 0.16 / +0.000 / +0.40 ± 0.24 | +0.01 ± 0.16 / +0.009 / -0.01 ± 0.24 | supported: overexposed_views | +0.12 ± 0.11 / -0.001 ± 0.011 / +0.07 ± 0.07 | 0.8 | not combined: disagree (z = 3.1) ; +0.08 ± 0.09 / +0.07 ± 0.07 |
| R 20 mm, 3 views | +0.66 ± 0.12 / +0.004 / +0.26 ± 0.16 | +0.06 ± 0.12 / +0.010 / +0.02 ± 0.16 | supported | +0.05 ± 0.11 / +0.009 ± 0.010 / +0.04 ± 0.07 | 0.6 | not combined: disagree (z = 3.8) ; +0.06 ± 0.08 / +0.03 ± 0.07 |
| R 20 mm, 2 views over the flat face | +0.85 ± 0.22 / +0.006 / +0.50 ± 0.38 | -0.02 ± 0.22 / -0.004 / +0.04 ± 0.38 | supported | +0.15 ± 0.11 / +0.009 ± 0.011 / +0.03 ± 0.07 | 0.4 | +0.28 ± 0.10 / +0.05 ± 0.07 ; +0.12 ± 0.10 / +0.03 ± 0.07 |
| R 35 mm, 6 views, 54 deg | +0.59 ± 0.11 / +0.001 / +0.26 ± 0.15 | +0.02 ± 0.11 / +0.008 / +0.02 ± 0.15 | supported: overexposed_views | -0.01 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.2 | not combined: disagree (z = 3.9) ; +0.01 ± 0.07 / -0.01 ± 0.09 |
| R 35 mm, 4 views | +0.64 ± 0.16 / +0.003 / +0.38 ± 0.24 | -0.01 ± 0.16 / +0.008 / -0.04 ± 0.25 | supported: overexposed_views | +0.12 ± 0.12 / -0.003 ± 0.011 / +0.10 ± 0.12 | 0.9 | +0.30 ± 0.10 / +0.16 ± 0.11 ; +0.07 ± 0.10 / +0.07 ± 0.11 |
| R 35 mm, 3 views | +0.70 ± 0.23 / +0.003 / +0.46 ± 0.40 | +0.06 ± 0.12 / +0.010 / +0.02 ± 0.17 | supported | +0.04 ± 0.11 / +0.007 ± 0.011 / +0.02 ± 0.12 | 0.7 | +0.16 ± 0.10 / +0.06 ± 0.11 ; +0.05 ± 0.08 / +0.02 ± 0.10 |
| R 35 mm, 2 views over the flat face | +0.76 ± 0.23 / +0.009 / +0.48 ± 0.40 | -0.02 ± 0.23 / -0.003 / +0.06 ± 0.40 | supported | +0.14 ± 0.11 / +0.007 ± 0.011 / -0.01 ± 0.13 | 0.4 | +0.26 ± 0.10 / +0.04 ± 0.12 ; +0.11 ± 0.10 / -0.00 ± 0.12 |
| R 60 mm, 6 views, 54 deg | +0.58 ± 0.16 / +0.003 / +0.35 ± 0.25 | +0.04 ± 0.11 / +0.010 / +0.04 ± 0.17 | supported | -0.01 ± 0.11 / +0.007 ± 0.011 / +0.01 ± 0.18 | 1.0 | not combined: disagree (z = 3.1) ; +0.01 ± 0.08 / +0.03 ± 0.12 |
| R 60 mm, 4 views | +0.60 ± 0.24 / +0.006 / +0.49 ± 0.42 | +0.01 ± 0.23 / +0.010 / -0.01 ± 0.42 | supported | +0.14 ± 0.13 / +0.003 ± 0.011 / +0.14 ± 0.20 | 1.0 | +0.24 ± 0.11 / +0.20 ± 0.18 ; +0.11 ± 0.11 / +0.11 ± 0.18 |
| R 60 mm, 3 views | +0.40 ± 0.26 / +0.006 / +0.13 ± 0.33 | +0.11 ± 0.26 / +0.011 / +0.08 ± 0.33 | supported | +0.04 ± 0.12 / +0.010 ± 0.012 / +0.12 ± 0.25 | 1.4 | +0.10 ± 0.11 / +0.12 ± 0.20 ; +0.05 ± 0.11 / +0.10 ± 0.20 |
| R 60 mm, 2 views over the flat face | +0.74 ± 0.23 / +0.010 / +0.58 ± 0.42 | -0.01 ± 0.24 / -0.005 / +0.07 ± 0.42 | supported | +0.09 ± 0.12 / +0.009 ± 0.012 / -0.06 ± 0.22 | 0.5 | +0.23 ± 0.11 / +0.08 ± 0.19 ; +0.07 ± 0.11 / -0.03 ± 0.19 |
| R 20 mm, 3 views within 12 deg | +0.66 ± 0.41 / +0.006 / +0.25 ± 0.50 | -0.00 ± 0.39 / +0.004 / +0.00 ± 0.48 | supported | -0.11 ± 0.10 / +0.026 ± 0.011 / -0.21 ± 0.08 | 0.8 | -0.07 ± 0.10 / -0.20 ± 0.08 ; -0.11 ± 0.10 / -0.21 ± 0.08 |
| R 35 mm, 3 views within 12 deg | +0.57 ± 0.42 / +0.006 / +0.23 ± 0.53 | -0.01 ± 0.40 / +0.004 / +0.00 ± 0.50 | supported | -0.09 ± 0.11 / +0.025 ± 0.011 / -0.22 ± 0.12 | 0.8 | -0.05 ± 0.10 / -0.19 ± 0.12 ; -0.08 ± 0.10 / -0.20 ± 0.12 |
| R 60 mm, 3 views within 12 deg | +0.41 ± 0.44 / +0.007 / +0.12 ± 0.57 | -0.03 ± 0.42 / +0.003 / -0.02 ± 0.55 | supported | -0.06 ± 0.12 / +0.020 ± 0.011 / -0.15 ± 0.19 | 0.6 | -0.02 ± 0.12 / -0.12 ± 0.18 ; -0.05 ± 0.12 / -0.13 ± 0.18 |
| R 35 mm, 6 views from 880 mm | +0.56 ± 0.13 / +0.005 / +0.25 ± 0.20 | +0.02 ± 0.13 / +0.007 / +0.01 ± 0.20 | supported: overexposed_views | +0.23 ± 0.12 / +0.006 ± 0.012 / +0.24 ± 0.12 | 1.1 | +0.38 ± 0.09 / +0.24 ± 0.10 ; +0.13 ± 0.09 / +0.18 ± 0.10 |
| R 35 mm, 1 view | fails (no_silhouette_found) | not run | needs_view: too_few_views | [-0.10 ± 0.12 / +0.008 ± 0.022 / -0.37 ± 0.57] | 0.3 | - |

### B. Ambient level

| case | (a) edge-only, same torch photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|
| ambient 0.0 (torch 1.1 at 500 mm) | +1.86 ± 4.22 / -0.553 / -0.17 ± 5.29 | supported: overexposed_views | +0.08 ± 0.10 / +0.013 ± 0.011 / +0.07 ± 0.11 | 1.3 | not combined: disagree (z = 4.5) |
| ambient 0.5 (torch 1.1 at 500 mm) | +0.27 ± 0.15 / +0.005 / +0.14 ± 0.23 | supported | +0.03 ± 0.11 / +0.010 ± 0.011 / +0.07 ± 0.11 | 1.0 | +0.11 ± 0.09 / +0.09 ± 0.10 |
| ambient 1.0 (torch 1.1 at 500 mm) | +0.15 ± 0.15 / +0.006 / +0.07 ± 0.23 | supported | -0.02 ± 0.12 / +0.013 ± 0.013 / +0.08 ± 0.12 | 1.0 | +0.05 ± 0.09 / +0.08 ± 0.11 |
| ambient 2.5 (torch 1.1 at 500 mm) | +0.07 ± 0.15 / +0.005 / +0.03 ± 0.23 | supported | -0.09 ± 0.17 / +0.014 ± 0.018 / +0.09 ± 0.16 | 1.4 | -0.00 ± 0.11 / +0.07 ± 0.13 |

### C. Gloss

| case | (a) edge-only, same torch photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|
| gloss strength 0.2, shininess 60 | +0.58 ± 0.11 / +0.002 / +0.28 ± 0.15 | supported: overexposed_views | +0.00 ± 0.10 / +0.010 ± 0.010 / -0.01 ± 0.11 | 1.1 | not combined: disagree (z = 3.8) |
| gloss strength 0.5, shininess 25 | +0.58 ± 0.11 / +0.002 / +0.28 ± 0.15 | supported: overexposed_views | +0.02 ± 0.10 / +0.010 ± 0.011 / -0.02 ± 0.11 | 1.3 | not combined: disagree (z = 3.7) |
| gloss strength 1.0, shininess 150 | +0.58 ± 0.11 / +0.001 / +0.28 ± 0.15 | supported: overexposed_views | -0.02 ± 0.11 / +0.008 ± 0.010 / -0.00 ± 0.11 | 1.5 | not combined: disagree (z = 3.9) |

### D. Exposure differences between views

| case | (a) edge-only, same torch photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|
| fixed exposures x0.6 .. x1.7 | +0.44 ± 0.20 / +0.012 / +0.06 ± 0.23 | supported | -0.00 ± 0.10 / -0.005 ± 0.011 / -0.01 ± 0.11 | 0.9 | +0.09 ± 0.09 / +0.00 ± 0.10 |
| fixed exposures all x3 (clipping) | +0.43 ± 0.15 / +0.010 / +0.47 ± 0.23 | needs_view: edge_region_not_usable, overexposed | - | 0.2 | not combined (photometric_not_supported) |
| fixed exposures all x0.04 (dark) | fails (error: GeometryError: no_tags: No accepted AprilTag in the selected frames: there is no metric scale.) | not_run: no_tag_poses (no_tags) | - | 0.0 | - |

### E. Albedo and material near the edge (no flash-off partner)

| case | (a) edge-only, same torch photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|
| paint strip (albedo 0.52) from 40 mm on the bend to 16 mm on the face | +0.85 ± 0.19 / -0.010 / +0.69 ± 0.22 | unsupported: surface_fixed_pattern_at_edge, overexposed_views | [+8.39 ± 0.10 / +1.520 ± 0.010 / -27.39 ± 0.02] | 1.8 | not combined (photometric_not_supported) |
| paint strip on the face, 27 to 53 mm from the edge | +0.60 ± 0.15 / +0.001 / +0.37 ± 0.23 | unsupported: model_does_not_explain_brightness, heldout_inconsistent, surface_not_uniform | [-5.74 ± 13.17 / +0.023 ± 3.036 / -5.36 ± 10.19] | 2.5 | not combined (photometric_not_supported) |
| sharp dark line 14 mm from the edge | +0.60 ± 0.15 / +0.001 / +0.37 ± 0.23 | supported: overexposed_views | +0.02 ± 0.10 / -0.000 ± 0.010 / +0.02 ± 0.11 | 1.1 | not combined: disagree (z = 3.2) |
| soft 3 % dark band 9 mm from the edge | +0.60 ± 0.15 / +0.001 / +0.37 ± 0.23 | unsupported: surface_not_uniform, overexposed_views | [-1.42 ± 0.92 / -0.021 ± 0.209 / -0.72 ± 0.70] | 1.0 | not combined (photometric_not_supported) |
| texture 4 %, 6 mm grain | +0.59 ± 0.15 / +0.001 / +0.35 ± 0.23 | supported: overexposed_views | -0.12 ± 0.23 / +0.008 ± 0.034 / -0.11 ± 0.21 | 1.1 | +0.38 ± 0.13 / +0.10 ± 0.16 |
| albedo variation 8 %, 120 mm patches | +0.59 ± 0.15 / +0.000 / +0.37 ± 0.23 | supported | -1.09 ± 0.31 / -0.031 ± 0.057 / +0.18 ± 0.37 | 1.6 | not combined: disagree (z = 4.9) |

### F. Low silhouette contrast

| case | (a) edge-only, same torch photos | (a') edge-only, ambient photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|---|
| wall 110 mm behind, albedo 0.27 (matches the limb in torch photos) | +0.17 ± 2.04 / +0.057 / -1.25 ± 2.21 | +0.01 ± 0.11 / -0.000 / +0.00 ± 0.15 | supported | +0.05 ± 0.10 / +0.004 ± 0.011 / +0.01 ± 0.11 | 1.1 | +0.05 ± 0.10 / +0.01 ± 0.11 ; +0.03 ± 0.08 / +0.01 ± 0.09 |
| wall 110 mm behind, same albedo as the panel | -0.80 ± 0.15 / +0.008 / -0.81 ± 0.23 | fails (no_silhouette_found) | unsupported: radius_out_of_range, transition_not_stationary, surface_not_uniform | [+34.57 ± 12.53 / -0.733 ± 2.093 / +465.00 ± 5.56] | 5.2 | not combined (photometric_not_supported) ; - |
| dark room (ambient 0): the limb is as dark as the background | +1.86 ± 4.22 / -0.553 / -0.17 ± 5.29 | not run | supported: overexposed_views | +0.08 ± 0.10 / +0.013 ± 0.011 / +0.07 ± 0.11 | 1.3 | not combined: disagree (z = 4.5) |

### G. Pose error of the real tag pipeline (same photos, four ways of getting poses)

| case | (b) photometric: status | (b) estimate | time s |
|---|---|---|---|
| R 20 mm, 6 views, renderer poses | supported: overexposed_views | -0.02 ± 0.10 / -0.003 ± 0.010 / -0.02 ± 0.06 | 1.3 |
| R 20 mm, 6 views, tag pipeline | supported: overexposed_views | -0.02 ± 0.10 / +0.003 ± 0.010 / -0.03 ± 0.06 | 1.0 |
| R 60 mm, 6 views, renderer poses | supported | -0.02 ± 0.11 / +0.001 ± 0.011 / +0.02 ± 0.18 | 1.0 |
| R 60 mm, 6 views, tag pipeline | supported | -0.01 ± 0.11 / +0.007 ± 0.011 / +0.01 ± 0.18 | 1.2 |
| R 35 mm, 6 views, renderer poses | supported: overexposed_views | -0.02 ± 0.10 / -0.000 ± 0.010 / -0.02 ± 0.11 | 1.4 |
| R 35 mm, 6 views, tag pipeline | supported: overexposed_views | -0.01 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.4 |
| R 35 mm, 6 views, tag pipeline, gauge tag of the multi-view solver | supported: overexposed_views | -0.01 ± 0.10 / -0.002 ± 0.011 / -0.00 ± 0.11 | 1.1 |
| R 35 mm, 6 views, tag pipeline, plane through all tags | supported: overexposed_views | +0.00 ± 0.10 / +0.004 ± 0.011 / -0.01 ± 0.11 | 1.1 |
| R 35 mm, 4 views, renderer poses | supported: overexposed_views | -0.02 ± 0.10 / -0.004 ± 0.011 / -0.00 ± 0.11 | 0.9 |
| R 35 mm, 4 views, tag pipeline | supported: overexposed_views | +0.12 ± 0.12 / -0.003 ± 0.011 / +0.10 ± 0.12 | 1.2 |
| R 35 mm, 4 views, tag pipeline, gauge tag of the multi-view solver | supported: overexposed_views | -0.07 ± 0.10 / +0.002 ± 0.011 / -0.04 ± 0.12 | 1.0 |
| R 35 mm, 4 views, tag pipeline, plane through all tags | supported: overexposed_views | -0.01 ± 0.11 / -0.004 ± 0.011 / -0.00 ± 0.12 | 0.9 |
| R 35 mm, 3 views, renderer poses | supported | -0.01 ± 0.10 / -0.002 ± 0.013 / -0.01 ± 0.11 | 0.6 |
| R 35 mm, 3 views, tag pipeline | supported | +0.04 ± 0.11 / +0.007 ± 0.011 / +0.02 ± 0.12 | 0.8 |
| R 35 mm, 3 views, tag pipeline, gauge tag of the multi-view solver | supported | -0.02 ± 0.11 / +0.005 ± 0.014 / +0.00 ± 0.13 | 0.6 |
| R 35 mm, 3 views, tag pipeline, plane through all tags | supported | +0.02 ± 0.11 / +0.002 ± 0.013 / -0.00 ± 0.12 | 0.8 |
| R 35 mm, 2 views over the flat face, renderer poses | supported | +0.04 ± 0.11 / -0.009 ± 0.011 / -0.04 ± 0.12 | 0.5 |
| R 35 mm, 2 views over the flat face, tag pipeline | supported | +0.14 ± 0.11 / +0.007 ± 0.011 / -0.01 ± 0.13 | 0.4 |
| R 35 mm, 2 views over the flat face, tag pipeline, gauge tag of the multi-view solver | supported | +0.07 ± 0.11 / -0.013 ± 0.011 / -0.02 ± 0.13 | 0.4 |
| R 35 mm, 2 views over the flat face, tag pipeline, plane through all tags | supported | +0.12 ± 0.11 / -0.007 ± 0.011 / -0.04 ± 0.13 | 0.4 |
| R 35 mm, 3 views within 12 deg, renderer poses | supported | +0.03 ± 0.11 / +0.001 ± 0.011 / -0.05 ± 0.11 | 0.7 |
| R 35 mm, 3 views within 12 deg, tag pipeline | supported | -0.09 ± 0.11 / +0.025 ± 0.011 / -0.22 ± 0.12 | 0.8 |
| R 35 mm, 3 views within 12 deg, tag pipeline, gauge tag of the multi-view solver | supported | +0.02 ± 0.11 / +0.002 ± 0.011 / -0.07 ± 0.11 | 0.9 |
| R 35 mm, 3 views within 12 deg, tag pipeline, plane through all tags | supported | +0.01 ± 0.11 / +0.020 ± 0.011 / -0.08 ± 0.12 | 0.9 |
| R 35 mm, 6 views from 880 mm, renderer poses | supported: overexposed_views | +0.04 ± 0.11 / +0.000 ± 0.011 / +0.03 ± 0.11 | 1.0 |
| R 35 mm, 6 views from 880 mm, tag pipeline | supported: overexposed_views | +0.23 ± 0.12 / +0.006 ± 0.012 / +0.24 ± 0.12 | 1.1 |
| R 35 mm, 6 views from 880 mm, tag pipeline, gauge tag of the multi-view solver | supported: overexposed_views | -0.02 ± 0.11 / +0.003 ± 0.011 / -0.03 ± 0.11 | 1.0 |
| R 35 mm, 6 views from 880 mm, tag pipeline, plane through all tags | supported: overexposed_views | +0.03 ± 0.11 / +0.001 ± 0.011 / +0.04 ± 0.11 | 0.9 |

### G, continued: error of the plane frame at the edge

The module takes the poses as exact.  `frame at the edge` is where the point that the estimated poses call "middle of the edge, on the face" really lies (renderer camera poses as reference, mean over the photos): across the edge (+ toward the face) and out of the panel.  R = 35 mm.

| photos | poses | gauge tag: id, distance to the middle of the edge | frame at the edge: across mm | frame at the edge: height mm | (b) tangent error mm | (b) radius error mm |
|---|---|---|---|---|---|---|
| 6 views | renderer poses | - | 0 (given) | 0 (given) | -0.02 | -0.02 |
| 6 views | tag pipeline: frame of the gauge tag the session service picks | 20, 469 mm | -0.02 | +0.00 | -0.01 | -0.02 |
| 6 views | tag pipeline: frame of the gauge tag the multi-view solver picks | 3, 236 mm | -0.00 | +0.02 | -0.01 | -0.00 |
| 6 views | tag pipeline: plane fitted through all tags | 20, 469 mm | -0.02 | +0.01 | +0.00 | -0.01 |
| 4 views | renderer poses | - | 0 (given) | 0 (given) | -0.02 | -0.00 |
| 4 views | tag pipeline: frame of the gauge tag the session service picks | 20, 469 mm | -0.01 | +0.17 | +0.12 | +0.10 |
| 4 views | tag pipeline: frame of the gauge tag the multi-view solver picks | 3, 236 mm | +0.00 | -0.05 | -0.07 | -0.04 |
| 4 views | tag pipeline: plane fitted through all tags | 20, 469 mm | -0.01 | -0.01 | -0.01 | -0.00 |
| 3 views | renderer poses | - | 0 (given) | 0 (given) | -0.01 | -0.01 |
| 3 views | tag pipeline: frame of the gauge tag the session service picks | 20, 469 mm | -0.02 | +0.04 | +0.04 | +0.02 |
| 3 views | tag pipeline: frame of the gauge tag the multi-view solver picks | 3, 236 mm | +0.03 | +0.02 | -0.02 | +0.00 |
| 3 views | tag pipeline: plane fitted through all tags | 20, 469 mm | -0.02 | +0.00 | +0.02 | -0.00 |
| 2 views over the flat face | renderer poses | - | 0 (given) | 0 (given) | +0.04 | -0.04 |
| 2 views over the flat face | tag pipeline: frame of the gauge tag the session service picks | 20, 469 mm | -0.03 | +0.14 | +0.14 | -0.01 |
| 2 views over the flat face | tag pipeline: frame of the gauge tag the multi-view solver picks | 3, 236 mm | +0.04 | +0.11 | +0.07 | -0.02 |
| 2 views over the flat face | tag pipeline: plane fitted through all tags | 20, 469 mm | -0.03 | +0.09 | +0.12 | -0.04 |
| 3 views within 12 deg | renderer poses | - | 0 (given) | 0 (given) | +0.03 | -0.05 |
| 3 views within 12 deg | tag pipeline: frame of the gauge tag the session service picks | 20, 469 mm | -0.00 | -0.26 | -0.09 | -0.22 |
| 3 views within 12 deg | tag pipeline: frame of the gauge tag the multi-view solver picks | 3, 236 mm | -0.00 | -0.02 | +0.02 | -0.07 |
| 3 views within 12 deg | tag pipeline: plane fitted through all tags | 20, 469 mm | -0.00 | -0.05 | +0.01 | -0.08 |
| 6 views from 880 mm | renderer poses | - | 0 (given) | 0 (given) | +0.04 | +0.03 |
| 6 views from 880 mm | tag pipeline: frame of the gauge tag the session service picks | 20, 469 mm | +0.01 | +0.27 | +0.23 | +0.24 |
| 6 views from 880 mm | tag pipeline: frame of the gauge tag the multi-view solver picks | 3, 236 mm | -0.02 | -0.12 | -0.02 | -0.03 |
| 6 views from 880 mm | tag pipeline: plane fitted through all tags | 20, 469 mm | +0.00 | -0.04 | +0.03 | +0.04 |

Over the 35 distinct supported runs of groups A to D, F and I (plain panel, tag pipeline, gauge tag of the session service): frame height at the edge -0.27 to +0.27 mm (rms 0.12); correlation with the photometric tangent error +0.74; rms tangent error 0.08 mm, after removing the straight-line dependence on the frame height 0.05 mm (slope +0.49 mm per mm).

### H. Flash-off partners: none, difference, ratio

| case | (a) edge-only, same torch photos | (a') edge-only, ambient photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|---|
| plain panel; no partner | not run | not run | supported: overexposed_views | -0.01 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.1 |  |
| plain panel; partners, difference | not run | not run | supported: overexposed_views | -0.16 ± 0.12 / +0.000 ± 0.011 / -0.10 ± 0.11 | 2.0 |  |
| plain panel; partners, ratio | not run | not run | supported: overexposed_views | -0.01 ± 0.11 / +0.003 ± 0.010 / -0.03 ± 0.11 | 1.2 |  |
| paint strip over the edge; no partner | not run | not run | unsupported: surface_fixed_pattern_at_edge, overexposed_views | [+7.93 ± 0.16 / +1.528 ± 0.017 / -29.78 ± 0.36] | 2.5 |  |
| paint strip over the edge; partners, difference | not run | not run | unsupported: surface_fixed_pattern_at_edge, transition_not_stationary, overexposed_views | [+9.57 ± 1.16 / +0.089 ± 0.106 / +7.59 ± 1.06] | 1.7 |  |
| paint strip over the edge; partners, ratio | not run | not run | supported: overexposed_views | -0.01 ± 0.11 / +0.012 ± 0.011 / -0.02 ± 0.11 | 1.1 |  |
| albedo variation 8 %, 120 mm patches; no partner | not run | not run | supported | -1.14 ± 0.30 / -0.024 ± 0.058 / +0.19 ± 0.39 | 1.1 |  |
| albedo variation 8 %, 120 mm patches; partners, difference | not run | not run | supported | -0.73 ± 0.30 / -0.014 ± 0.062 / +0.32 ± 0.41 | 1.9 |  |
| albedo variation 8 %, 120 mm patches; partners, ratio | not run | not run | supported | -0.01 ± 0.10 / +0.012 ± 0.011 / -0.02 ± 0.11 | 1.2 |  |
| room light from one side (ambient 0.2 + directional 0.35); no partner | +0.13 ± 0.11 / +0.013 / -0.12 ± 0.15 | -0.03 ± 0.11 / +0.013 / -0.15 ± 0.15 | unsupported: transition_not_stationary, surface_not_uniform | [-4.66 ± 0.86 / -0.010 ± 0.150 / -6.31 ± 0.58] | 1.2 | not combined (photometric_not_supported) ; not combined (photometric_not_supported) |
| room light from one side (ambient 0.2 + directional 0.35); partners, difference | not run | not run | supported | -0.09 ± 0.13 / +0.009 ± 0.015 / -0.05 ± 0.19 | 1.7 |  |
| room light from one side (ambient 0.2 + directional 0.35); partners, ratio | not run | not run | unsupported: transition_not_stationary | [+2.69 ± 0.95 / +0.038 ± 0.020 / +4.01 ± 0.76] | 2.1 |  |
| plain panel; partners taken 45 mm away | not run | not run | supported: pair_misaligned, overexposed_views | -0.04 ± 0.10 / +0.004 ± 0.010 / -0.04 ± 0.11 | 1.3 |  |

### I. Radiometry and sensor effects

| case | (a) edge-only, same torch photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|
| tone: extra gamma 1.2 | +0.55 ± 0.34 / +0.021 / -0.02 ± 0.39 | supported: overexposed_views | -0.02 ± 0.11 / +0.017 ± 0.011 / -0.01 ± 0.11 | 1.2 | +0.03 ± 0.10 / -0.01 ± 0.10 |
| tone: extra gamma 0.8 | +0.37 ± 0.11 / +0.009 / +0.10 ± 0.15 | supported: overexposed_views | -0.10 ± 0.11 / +0.016 ± 0.011 / -0.06 ± 0.11 | 1.5 | not combined: disagree (z = 3.0) |
| tone: S-curve 0.2 | +1.04 ± 0.15 / +0.012 / +0.67 ± 0.23 | supported: overexposed_views | +0.11 ± 0.12 / +0.008 ± 0.012 / +0.50 ± 0.16 | 1.6 | not combined: disagree (z = 4.8) |
| tone: S-curve 0.5 | +1.48 ± 0.15 / +0.012 / +1.05 ± 0.23 | unsupported: surface_fixed_pattern_at_edge, transition_not_stationary, overexposed_views | [+1.19 ± 0.64 / +0.133 ± 0.111 / +4.54 ± 1.30] | 2.0 | not combined (photometric_not_supported) |
| ISP sharpening 0.8 | +0.42 ± 0.15 / +0.007 / +0.19 ± 0.23 | supported: overexposed_views | -0.04 ± 0.10 / +0.014 ± 0.011 / +0.01 ± 0.11 | 1.0 | +0.11 ± 0.09 / +0.04 ± 0.10 |
| lens vignetting 0.6 | +0.63 ± 0.15 / +0.013 / +0.34 ± 0.23 | supported | +0.01 ± 0.10 / +0.014 ± 0.011 / +0.04 ± 0.11 | 1.1 | not combined: disagree (z = 3.3) |
| JPEG quality 70 | +0.64 ± 0.15 / +0.009 / +0.33 ± 0.23 | supported | -0.10 ± 0.10 / +0.016 ± 0.012 / -0.06 ± 0.11 | 1.3 | not combined: disagree (z = 4.0) |
| high noise | +0.48 ± 0.18 / +0.022 / +0.10 ± 0.21 | supported: overexposed_views | -0.01 ± 0.11 / +0.019 ± 0.012 / +0.04 ± 0.11 | 1.0 | +0.13 ± 0.10 / +0.05 ± 0.10 |
| lens distortion (calibrated) | +0.45 ± 0.16 / +0.007 / +0.10 ± 0.19 | supported: overexposed_views | +0.02 ± 0.10 / -0.005 ± 0.011 / +0.05 ± 0.11 | 1.1 | +0.14 ± 0.09 / +0.06 ± 0.09 |
| torch beam exponent 5 | +0.36 ± 0.29 / +0.025 / -0.00 ± 0.34 | supported: overexposed_views | +0.01 ± 0.10 / +0.016 ± 0.011 / +0.04 ± 0.11 | 1.1 | +0.05 ± 0.10 / +0.03 ± 0.10 |
| torch 30 mm from the lens | +0.43 ± 0.25 / +0.019 / +0.04 ± 0.28 | supported: overexposed_views | -0.01 ± 0.11 / +0.015 ± 0.011 / +0.02 ± 0.11 | 1.4 | +0.05 ± 0.10 / +0.02 ± 0.10 |

### J. Cases that have to be refused, shadows

| case | (a) edge-only, same torch photos | (a') edge-only, ambient photos | (b) photometric: status | (b) estimate | time s | (c) combined with (a) / (a') |
|---|---|---|---|---|---|---|
| flat panel, hint in the middle of the face | not run | not run | unsupported: no_shading_transition | - | 0.3 |  |
| flat panel, hint on a dark joint line | not run | not run | unsupported: no_shading_transition | - | 0.2 |  |
| flat panel, 60 mm proud block: hint at its torch shadow | not run | not run | unsupported: surface_fixed_pattern_at_edge, radius_out_of_range | - | 2.1 |  |
| rounded edge with a 45 mm proud block next to it (shadow, occlusion) | +0.39 ± 0.19 / +0.009 / -0.00 ± 0.22 | not run | supported: overexposed_views | +0.00 ± 0.11 / +0.009 ± 0.011 / -0.02 ± 0.11 | 1.4 | +0.10 ± 0.10 / -0.02 ± 0.10 |
| bend of only 45 degrees, then a straight flange | +106.78 ± 0.59 / +0.015 / +264.15 ± 1.19 | not run | unsupported: surface_fixed_pattern_at_edge | [+53.00 ± 0.10 / -0.558 ± 0.408 / +192.47 ± 10.91] | 2.8 | not combined (photometric_not_supported) |
| ambient photos only (torch off) | not run | +0.02 ± 0.11 / +0.008 / +0.02 ± 0.15 | needs_view: no_torch_on_views | - | 0.0 | not combined (photometric_not_supported) |
| torch state not recorded | not run | not run | needs_view: no_torch_on_views, torch_state_unknown | - | 0.0 |  |

### K. Model options

| case | (b) photometric: status | (b) estimate | time s |
|---|---|---|---|
| plain, 6 views | default | supported: overexposed_views | -0.01 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.3 |
| plain, 2 views | default | supported | +0.14 ± 0.11 / +0.007 ± 0.011 / -0.01 ± 0.13 | 0.5 |
| tone gamma 1.2 | default | supported: overexposed_views | -0.02 ± 0.11 / +0.017 ± 0.011 / -0.01 ± 0.11 | 1.3 |
| room light from one side | default | unsupported: transition_not_stationary, surface_not_uniform | [-4.66 ± 0.86 / -0.010 ± 0.150 / -6.31 ± 0.58] | 1.3 |
| plain, 6 views | tone exponent fixed | supported: overexposed_views | -0.00 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.1 |
| plain, 2 views | tone exponent fixed | supported | +0.14 ± 0.11 / +0.006 ± 0.011 / +0.06 ± 0.13 | 0.4 |
| tone gamma 1.2 | tone exponent fixed | supported: overexposed_views | -0.28 ± 0.19 / +0.010 ± 0.026 / -0.29 ± 0.15 | 1.2 |
| room light from one side | tone exponent fixed | unsupported: transition_not_stationary, surface_not_uniform | [-4.77 ± 0.81 / -0.012 ± 0.150 / -6.38 ± 0.60] | 1.2 |
| plain, 6 views | no lens fall-off | supported: overexposed_views | -0.08 ± 0.10 / +0.004 ± 0.012 / -0.10 ± 0.11 | 0.9 |
| plain, 2 views | no lens fall-off | supported | +0.10 ± 0.11 / +0.003 ± 0.011 / -0.06 ± 0.13 | 0.4 |
| tone gamma 1.2 | no lens fall-off | supported: overexposed_views | -0.08 ± 0.11 / +0.015 ± 0.012 / -0.08 ± 0.11 | 1.2 |
| room light from one side | no lens fall-off | unsupported: transition_not_stationary, surface_not_uniform | [-5.59 ± 0.74 / -0.039 ± 0.127 / -6.40 ± 0.51] | 0.9 |
| plain, 6 views | no 2nd-order beam/lens terms | supported: overexposed_views | -0.03 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 0.8 |
| plain, 2 views | no 2nd-order beam/lens terms | supported | +0.13 ± 0.11 / +0.006 ± 0.011 / -0.01 ± 0.13 | 0.4 |
| tone gamma 1.2 | no 2nd-order beam/lens terms | supported: overexposed_views | -0.03 ± 0.11 / +0.016 ± 0.012 / -0.01 ± 0.11 | 0.9 |
| room light from one side | no 2nd-order beam/lens terms | unsupported: transition_not_stationary, surface_not_uniform | [-4.85 ± 0.70 / -0.021 ± 0.137 / -6.19 ± 0.53] | 1.6 |
| plain, 6 views | torch at the lens (offset fixed) | supported: overexposed_views | +1.02 ± 0.43 / -0.078 ± 0.072 / -0.69 ± 0.27 | 1.4 |
| plain, 2 views | torch at the lens (offset fixed) | unsupported: heldout_inconsistent | [+1.08 ± 0.21 / +0.016 ± 0.040 / -1.38 ± 0.19] | 0.5 |
| tone gamma 1.2 | torch at the lens (offset fixed) | supported: overexposed_views | +1.01 ± 0.42 / -0.066 ± 0.072 / -0.69 ± 0.29 | 1.4 |
| room light from one side | torch at the lens (offset fixed) | supported | -6.25 ± 0.84 / -0.131 ± 0.085 / -6.66 ± 0.59 | 1.3 |
| plain, 6 views | beam tilt free | supported: overexposed_views | -0.00 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.3 |
| plain, 2 views | beam tilt free | supported | +0.13 ± 0.11 / +0.008 ± 0.011 / -0.00 ± 0.14 | 0.5 |
| tone gamma 1.2 | beam tilt free | supported: overexposed_views | -0.02 ± 0.10 / +0.015 ± 0.011 / +0.00 ± 0.11 | 1.2 |
| room light from one side | beam tilt free | unsupported: heldout_inconsistent | [-4.71 ± 0.70 / +0.003 ± 0.096 / -6.39 ± 0.54] | 2.8 |
| plain, 6 views | albedo gradient free | supported: overexposed_views | -0.01 ± 0.10 / +0.006 ± 0.011 / -0.02 ± 0.11 | 1.4 |
| plain, 2 views | albedo gradient free | supported | +0.13 ± 0.11 / +0.006 ± 0.011 / +0.03 ± 0.13 | 0.5 |
| tone gamma 1.2 | albedo gradient free | supported: overexposed_views | -0.02 ± 0.10 / +0.017 ± 0.011 / -0.00 ± 0.11 | 1.2 |
| room light from one side | albedo gradient free | unsupported: transition_not_stationary | [-5.40 ± 0.61 / -0.018 ± 0.062 / -7.02 ± 0.53] | 1.5 |
| plain, 6 views | ambient harmonics free | supported: overexposed_views | -0.00 ± 0.11 / +0.007 ± 0.011 / -0.06 ± 0.14 | 1.1 |
| plain, 2 views | ambient harmonics free | supported | +0.13 ± 0.12 / +0.007 ± 0.012 / -0.05 ± 0.34 | 0.6 |
| tone gamma 1.2 | ambient harmonics free | supported: overexposed_views | -0.03 ± 0.11 / +0.016 ± 0.011 / -0.03 ± 0.22 | 1.5 |
| room light from one side | ambient harmonics free | supported | -0.01 ± 0.12 / +0.008 ± 0.012 / +0.04 ± 0.16 | 1.5 |

### Head to head: photometric against edge-only

Cases of the groups above in which the edge-only baseline was run (group K and renderer-pose runs excluded; a run that appears twice is counted once).  `photometric` means status `supported`; an estimate the module withheld counts as no estimate.

| comparison | cases | both give an estimate | only photometric | only edge-only | neither |
|---|---|---|---|---|---|
| (b) against (a), same torch photos | 49 | 39 | 0 | 8 | 2 |
| (b) against (a'), ambient photos from the same stations | 20 | 17 | 0 | 2 | 1 |

Where both give an estimate:

| comparison | quantity | n | edge-only: median, max abs error | photometric: median, max abs error | photometric closer to truth in |
|---|---|---|---|---|---|
| (b) against (a), same torch photos | tangent mm | 39 | 0.578, 1.856 | 0.043, 1.087 | 37 of 39 |
| (b) against (a), same torch photos | direction deg | 39 | 0.006, 0.553 | 0.009, 0.031 | 14 of 39 |
| (b) against (a), same torch photos | radius mm | 39 | 0.264, 1.253 | 0.042, 0.499 | 34 of 39 |
| (b) against (a'), ambient photos from the same stations | tangent mm | 17 | 0.019, 0.114 | 0.089, 0.231 | 5 of 17 |
| (b) against (a'), ambient photos from the same stations | direction deg | 17 | 0.008, 0.011 | 0.007, 0.026 | 10 of 17 |
| (b) against (a'), ambient photos from the same stations | radius mm | 17 | 0.020, 0.080 | 0.060, 0.241 | 5 of 17 |

### Do the stated uncertainties cover the errors?

Over all cases above except group K; a run that appears in two groups is counted once.  `within 1 sigma` should be near 0.68 and `within 2 sigma` near 0.95 for honest, Gaussian uncertainties.

| method | quantity | n | median abs error | max abs error | within 1 sigma | within 2 sigma | worst error / sigma |
|---|---|---|---|---|---|---|---|
| (a') edge-only, ambient photos | tangent mm | 18 | 0.019 | 0.114 | 1.00 | 1.00 | 0.5 |
| (a') edge-only, ambient photos | direction deg | 18 | 0.008 | 0.013 | 0.39 | 0.89 | 2.9 |
| (a') edge-only, ambient photos | radius mm | 18 | 0.020 | 0.150 | 1.00 | 1.00 | 1.0 |
| (a) edge-only, torch photos | tangent mm | 47 | 0.587 | 106.775 | 0.11 | 0.28 | 182.2 |
| (a) edge-only, torch photos | direction deg | 47 | 0.006 | 0.553 | 0.43 | 0.77 | 4.4 |
| (a) edge-only, torch photos | radius mm | 47 | 0.282 | 264.146 | 0.40 | 0.87 | 222.5 |
| (b) photometric, supported | tangent mm | 47 | 0.043 | 1.141 | 0.79 | 0.94 | 3.8 |
| (b) photometric, supported | direction deg | 47 | 0.009 | 0.031 | 0.70 | 0.96 | 2.5 |
| (b) photometric, supported | radius mm | 47 | 0.043 | 0.499 | 0.91 | 0.94 | 3.0 |
| (b) photometric, supported, gauge tag of the solver (group G only) | tangent mm | 6 | 0.022 | 0.070 | 1.00 | 1.00 | 0.7 |
| (b) photometric, supported, gauge tag of the solver (group G only) | direction deg | 6 | 0.003 | 0.013 | 0.83 | 1.00 | 1.2 |
| (b) photometric, supported, gauge tag of the solver (group G only) | radius mm | 6 | 0.022 | 0.065 | 1.00 | 1.00 | 0.6 |
| (b) photometric, supported, plane through all tags (group G only) | tangent mm | 6 | 0.016 | 0.117 | 0.83 | 1.00 | 1.1 |
| (b) photometric, supported, plane through all tags (group G only) | direction deg | 6 | 0.004 | 0.020 | 0.83 | 1.00 | 1.8 |
| (b) photometric, supported, plane through all tags (group G only) | radius mm | 6 | 0.025 | 0.085 | 1.00 | 1.00 | 0.7 |
| (b) photometric, supported, renderer poses (group G only) | tangent mm | 8 | 0.019 | 0.038 | 1.00 | 1.00 | 0.4 |
| (b) photometric, supported, renderer poses (group G only) | direction deg | 8 | 0.001 | 0.009 | 1.00 | 1.00 | 0.8 |
| (b) photometric, supported, renderer poses (group G only) | radius mm | 8 | 0.017 | 0.047 | 1.00 | 1.00 | 0.4 |
| (b) photometric, withheld (if it had been trusted) | tangent mm | 11 | 5.738 | 53.003 | 0.18 | 0.36 | 530.0 |
| (c) combined with edge-only on ambient photos | tangent mm | 17 | 0.069 | 0.132 | 0.76 | 1.00 | 1.5 |
| (c) combined with edge-only on torch photos | tangent mm | 24 | 0.108 | 0.380 | 0.46 | 0.71 | 4.2 |

### Axis direction from normalised-brightness gradients alone

Model-free estimate (groups A to C, 24 cases): median abs error 0.074 deg, max 0.422 deg.  Joint model fit on the same cases: median 0.009 deg, max 0.026 deg.

### Torch position and beam recovered from the photos

Truth of the base configuration: torch 9 mm right of and 13 mm below the lens, beam exponent 2, lens fall-off `1 - 0.15 + 0.15 cos^4` (to first order an exponent v1 = 0.6 in the model's `cos^v1` form), ambient / torch = 0.2 / 1.1 = 0.182, tone exponent 1.  Beam and lens fall-off multiply the same torch light: without ambient light, and when the ambient part is measured by flash-off partners (difference), only their product is identified; in the ratio of a pair the lens fall-off cancels.

| case | status | torch offset x, y mm (sigma) | beam exponent m, m2 | lens v1, v2 | ambient / torch | tone exponent |
|---|---|---|---|---|---|---|
| R 35 mm, 6 views, 54 deg | supported | +9.1, +12.7 (0.288, 0.155) | 2.09, -0.02 | +0.50, +0.70 | 0.180 | 0.999 |
| R 35 mm, 3 views | supported | +8.9, +12.3 (0.476, 0.214) | 2.23, +0.43 | +0.42, +0.44 | 0.175 | 0.992 |
| R 35 mm, 2 views over the flat face | supported | +10.6, +11.1 (1.557, 0.62) | 2.27, -0.42 | +0.46, +0.89 | 0.161 | 0.950 |
| ambient 0.0 (torch 1.1 at 500 mm) | supported | +7.6, +10.9 (0.563, 1.042) | 2.93, -0.28 | -0.38, +1.03 | 0.000 | 1.004 |
| plain panel; partners, difference | supported | +10.9, +11.6 (0.313, 0.085) | 2.53, +0.11 | +0.00, +0.00 | 0.155 | 0.959 |
| plain panel; partners, ratio | supported | +9.4, +13.1 (0.425, 0.116) | 1.98, -0.12 | +0.00, +0.00 | 0.181 | 0.996 |
| torch beam exponent 5 | supported | +9.4, +12.4 (0.21, 0.11) | 5.18, -0.55 | +0.43, +0.88 | 0.172 | 0.986 |
| torch 30 mm from the lens | supported | -24.7, +14.1 (0.342, 0.192) | 2.20, -0.35 | +0.43, +0.92 | 0.170 | 0.986 |

### Runtime

Single process on the machine that ran this script (4 cores, shared with other jobs); model evaluations are NumPy on one core.

| step | n | median s | max s |
|---|---|---|---|
| photometric analysis, 1 torch-on views in the fit | 1 | 0.27 | 0.27 |
| photometric analysis, 2 torch-on views in the fit | 15 | 0.45 | 0.60 |
| photometric analysis, 3 torch-on views in the fit | 14 | 0.78 | 1.38 |
| photometric analysis, 4 torch-on views in the fit | 7 | 0.94 | 1.23 |
| photometric analysis, 5 torch-on views in the fit | 2 | 1.32 | 1.35 |
| photometric analysis, 6 torch-on views in the fit | 82 | 1.23 | 5.16 |
| tag detection + bundle adjustment, per photo | 118 | 0.43 | 0.55 |
| edge-only baseline through the session service (register + propose) | 70 | 6.59 | 12.08 |

Script run: 127 cases, 342 photos; rendering 7 s (cached photos are reused), everything else 898 s.

<!-- END GENERATED TABLES -->

## 4. Reading of the results

All statements refer to the tables above (second complete run: 127 cases, 95
of them outside the option study K; of those 95, 77 are `supported`, 13
`unsupported`, 4 `needs_view` and 1 is never reached because the tag pipeline
finds no tag).

### 4.1 Does the photometric cue improve on the edge-only method?

**Against silhouettes from the same torch-on photos: yes, clearly, for
position and radius.** With the light next to the lens the shading falls to
zero at the occluding contour, the silhouette is found inside the true
contour, and the edge-only tangent line lands +0.40 to +0.85 mm toward the
face in group A (stated 0.11 to 0.44 mm) and +1.86 mm in a dark room. Over the
39 cases in which both methods give an estimate, the median absolute tangent
error is 0.58 mm edge-only and 0.04 mm photometric; the photometric estimate
is closer to the truth in 37 of 39 cases for the tangent line and in 34 of 39
for the radius (median 0.26 against 0.04 mm). Only 11 % of the edge-only
tangent errors on torch photos lie within one stated sigma. The bias shrinks
as room light takes over (+0.59, +0.27, +0.15, +0.07 mm at ambient 0.2, 0.5,
1.0, 2.5); at ambient 2.5 the two methods are equally good (+0.07 against
-0.09 mm).

**Against silhouettes from ambient photos taken at the same stations: no.**
Edge-only on torch-off photos reaches a median of 0.02 mm (max 0.11 mm) for
the tangent line and 0.02 mm (max 0.08 mm) for the radius; photometric 0.09 mm
(max 0.23 mm) and 0.06 mm (max 0.24 mm). The photometric estimate is closer to
the truth in 5 of 17 cases. Whenever torch-off photos can be taken in which
the rounded edge stands out against the background, the torch adds nothing to
the geometric estimate. The planar and the edge-only workflow do not need it.

**Direction of the edge: no consistent gain in either comparison.** Median
0.006 to 0.008 degrees edge-only, 0.007 to 0.009 degrees photometric.

**There is no case in which only the photometric module returns an
estimate.** In 8 cases with a baseline on torch photos the module refuses
while the edge-only method still answers, with errors of +0.43, +0.85, +0.60,
+0.60, -0.80, +1.48, +0.13 and +106.78 mm: seven of those eight edge-only
answers are poor themselves. In 2 cases neither answers (one view; photos too
dark for the tags).

Where the photometric estimate is the better one *available*:

* torch-on photos are all there is (the sweep was taken with the torch, or the
  room is dark): -0.11 to +0.23 mm in group A instead of +0.40 to +0.85 mm,
  +0.08 mm instead of +1.86 mm in the dark room;
* the background is as dark as the limb in the torch photos (group F, wall
  with albedo 0.27): edge-only +0.17 mm but with a stated +- 2.0 mm and a
  radius of -1.25 +- 2.2 mm; photometric +0.05 +- 0.10 mm. Ambient photos of
  the same scene are again better (+0.01 mm);
* little parallax: with three views inside 12 degrees the edge-only sigma is
  0.4 mm (its errors on ambient photos are small all the same: 0.03 mm or
  less), while the photometric estimate states 0.10 to 0.12 mm (errors -0.06
  to -0.11 mm) because the shading of the bend does not need parallax. Two
  views are enough in these renders (+0.09 to +0.15 mm).

It does **not** rescue the case in which the wall behind has the albedo of
the panel: ambient silhouettes fail (`no_silhouette_found`), torch silhouettes
are 0.80 mm off at a stated 0.15 mm, and the photometric module refuses (a lit
wall behind the limb is not in its model; trusting its withheld numbers would
have cost 35 mm).

### 4.2 Are the stated uncertainties honest?

For `supported` results with tag-pipeline poses (47 distinct runs): tangent
line 79 % within one stated sigma and 94 % within two, radius 91 % / 94 %,
direction 70 % / 96 %. That is near Gaussian coverage in the bulk, with three
kinds of outliers that the uncertainty does not announce:

* smooth albedo variation without ratio partners: -1.09 and -1.14 mm at a
  stated 0.30 mm (3.5 and 3.8 sigma), -0.73 mm with "difference" partners
  (2.4 sigma); see 4.5;
* a mild S-shaped tone curve: radius +0.50 mm at a stated 0.16 mm (3.0
  sigma), tangent line unaffected (+0.11 mm);
* the plane frame of the tag pipeline (4.3): radius -0.21 mm at a stated
  0.08 mm (2.7 sigma) and direction +0.026 degrees at 0.011 (2.5 sigma) with
  the 12-degree rig; tangent +0.23 mm and radius +0.24 mm at 0.12 mm (1.9 and
  2.1 sigma) with the 880 mm rig.

With renderer poses (8 runs) every error is within one sigma (at most
0.04 mm, 0.009 degrees, 0.05 mm): the stated sigma is then dominated by its
floors (0.10 mm, 0.01 degrees) and is conservative.

The edge-only baseline, for comparison: on ambient photos its tangent and
radius sigmas are conservative (all errors within one sigma) and its direction
sigma is optimistic (39 % / 89 %); on torch photos its uncertainty does not
cover its bias (11 % / 28 % for the tangent line).

### 4.3 What limits the accuracy with real tag poses

Group G separates the photometric model from the poses. With renderer poses
the tangent error is -0.02 to +0.04 mm for every view set. With the tag
pipeline it is -0.09 to +0.23 mm, and the table "error of the plane frame at
the edge" shows why. The module is given `T_cam_plane` in the frame of one
50 mm tag, the gauge. The session service picks tag 20 as gauge, 469 mm from
the middle of the edge; a tilt of its estimated plane of only 0.3 to 0.6 mrad
puts the assumed face 0.14 to 0.27 mm above or below the real one at the edge
(4 views +0.17, 2 views +0.14, 12-degree rig -0.26, 880 mm rig +0.27 mm). The
in-plane error of the frame is small (0.04 mm or less across the edge). Over
the 35 distinct supported plain-panel runs the frame height at the edge has an
rms of 0.12 mm, its correlation with the photometric tangent error is +0.74,
and removing the linear dependence (slope +0.49 mm per mm) takes the rms
tangent error from 0.08 to 0.05 mm.

Two other ways of fixing the frame, on the same photos. They are
**exploratory**: they were added to the script after the first complete run
had shown the effect and are not part of what was fixed before the held-out
runs.

* The gauge tag the multi-view solver picks by default (tag 3, 236 mm from
  the middle of the edge): frame height -0.12 to +0.11 mm, tangent error -0.07
  to +0.07 mm, radius error 0.07 mm or less.
* The service's gauge tag, but the plane fitted through the corners of all
  tags: frame height -0.05 to +0.09 mm, tangent error -0.01 to +0.12 mm (the
  +0.12 mm is the two-view set), radius error 0.08 mm or less.

The module cannot see this error: it takes the poses as exact, and its
tangent sigma has a floor of 0.10 mm for that reason. The remedy is upstream:
hand over a plane frame that averages over the tag constellation, or the
covariance of the frame.

### 4.4 Views, ambient level, gloss, exposure

* **Number and spread of views** (A, G): 6, 4, 3 or 2 views and a 12-degree
  spread all give `supported` results; the differences between them are
  pose-frame effects (4.3), not conditioning (renderer poses: -0.02 to
  +0.04 mm throughout). One view is refused as `needs_view` with the request
  `add_view` (about 150 mm across the edge). Its withheld estimate (-0.10 mm)
  would have been fine here, but nothing cross-checks it.
* **Ambient level** (B): `supported` from a dark room to ambient 2.5; errors
  +0.08, -0.01, +0.03, -0.02, -0.09 mm at ambient 0, 0.2, 0.5, 1.0, 2.5; the
  stated sigma grows from 0.10 to 0.17 mm as the torch share falls.
* **Gloss** (C): no measurable effect (+0.00, +0.02, -0.02 mm); the specular
  core is masked and the lobe is fitted.
* **Exposure** (D): different exposures per view are absorbed by the
  per-view levels (-0.00 mm). Photos clipped on the face are refused
  (`needs_view`, request `reframe_edge`). At exposure x0.04 the tag pipeline
  finds no tag: nothing runs, the edge-only baseline included.
* `overexposed_views` on a `supported` row is a note, not a refusal: part of
  the edge region is clipped in some photo and those cells were masked.

### 4.5 Albedo, material, flash-off partners

Without a partner (E, H):

* paint strip across the edge: refused (`surface_fixed_pattern_at_edge`);
  trusting the withheld numbers would have cost +8.4 mm and 1.5 degrees;
* paint strip on the face, 27 to 53 mm from the edge: refused (three reasons;
  withheld -5.7 mm);
* sharp dark line 14 mm from the edge: masked as a surface-fixed outlier,
  result +0.02 mm;
* soft 3 % band 9 mm from the edge: refused (`surface_not_uniform`; withheld
  -1.4 mm);
* fine texture (4 %, 6 mm grain): `supported`, -0.12 mm, sigma raised to
  0.23 mm by the surface-pattern allowance;
* **smooth albedo variation (8 %, 120 mm patches): `supported` and wrong,
  -1.09 mm at a stated 0.31 mm.** The surface-pattern allowance comes out at
  0.22 mm, below its limit of 0.5 mm; held-out ratio 1.17, stationarity
  0.14 mm: no check fires. The development scene of this kind (10 %, 150 mm)
  was refused. A patch as wide as the analysed strip changes the brightness of
  the flat face over the same distance as the shading of the bend does, and
  the fit absorbs part of it into the transition.

With flash-off partners taken from a rest (4 mm apart; group H), tangent error
in mm (the generated table has the sigmas):

| scene | no partner | difference | ratio |
|---|---|---|---|
| plain panel | -0.01 | -0.16 | -0.01 |
| paint strip across the edge | refused | refused | -0.01 |
| smooth albedo variation | **-1.14, `supported`** | **-0.73, `supported`** | -0.01 |
| room light from one side | refused | -0.09 | refused |

The ratio of a pair removes paint and texture; the difference of a pair
removes uneven room light. Neither does both, and the module cannot tell from
the photos which one a scene needs: `pair_mode` is the researcher's choice. On
a plain panel under even light, partners are useless (ratio) or slightly
harmful (difference: -0.16 mm at a stated 0.12 mm). Partners taken 45 mm away
are rejected (`pair_misaligned`) and the result falls back to the torch-only
fit (-0.04 mm).

Uneven room light without partners is refused (`transition_not_stationary`,
`surface_not_uniform`; withheld -4.7 mm). The edge-only method is not
disturbed by it (+0.13 mm on the torch photos, -0.03 mm on ambient photos).

### 4.6 Radiometry and sensor effects (I)

An extra gamma of 1.2 or 0.8 is absorbed by the fitted tone exponent (-0.02,
-0.10 mm; with the exponent fixed the gamma 1.2 photos give -0.28 mm and a
radius error of -0.29 mm, group K). An S-curve of 0.2 leaves the tangent line
alone (+0.11 mm) but shifts the radius by +0.50 mm at a stated 0.16 mm without
any check firing. An S-curve of 0.5 is refused (withheld +1.2 mm, radius
+4.5 mm). Sharpening, vignetting 0.6, JPEG 70, high noise, calibrated lens
distortion, a narrower beam and a torch 30 mm from the lens stay within -0.10
to +0.02 mm.

### 4.7 Refusals and shadows (J)

A flat panel, with the hint on the bare face or on a joint line, gives
`no_shading_transition`. The torch shadow of a 60 mm block on a flat panel is
refused with no estimate at all; which check fires depends on the poses
(`surface_fixed_pattern_at_edge` and `radius_out_of_range` here,
`heldout_inconsistent` with the other gauge tag in the first complete run). A
block that casts shadows next to a real rounded edge is masked (+0.00 mm). A
45-degree bend followed by a straight flange is refused (withheld +53 mm); the
edge-only baseline reports +106.8 mm at a stated 0.59 mm there, because the
silhouette is the end of the flange. Without torch-on photos, or with the
torch state unknown, the result is `needs_view` with the request
`torch_on_sweep`, and the edge-only result from the same photos is untouched
(+0.02 mm).

### 4.8 Combining the two estimates (c)

Inverse-variance combination does not pay.

* With silhouettes from the *same torch photos* the combination is refused in
  15 of 39 cases (disagreement above three combined sigmas; that test does
  expose the silhouette bias, for instance in all three gloss cases and in the
  dark room). Where it is not refused it drags the photometric estimate toward
  the biased silhouette: median error 0.11 mm, max 0.38 mm, 46 % within one
  stated sigma, against a median of 0.04 mm for the photometric estimate
  alone.
* With silhouettes from *ambient photos* the combination (median 0.07 mm, max
  0.13 mm) is worse than the ambient silhouette alone (0.02 mm, max 0.11 mm).

The useful part of (c) is the disagreement test as a diagnostic, not the
combined number.

### 4.9 Model-free cue, nuisance parameters, options, runtime

* **Normalised-brightness gradients alone** give the axis direction with a
  median error of 0.07 degrees and up to 0.42 degrees: about ten times worse
  than the joint fit (0.009 degrees) and than the silhouettes. They serve as a
  starting value and as a diagnostic (`gradient_axis`), not as an estimate.
* **Torch position and beam** are recovered from the photos themselves:
  offset (9, 13) mm as (9.1, 12.7) with 6 views, (8.9, 12.3) with 3 and
  (10.6, 11.1) with 2; offset (-26, 15) mm as (-24.7, 14.1); beam exponent 2
  as 2.09 to 2.27 and 5 as 5.18; ambient/torch ratio 0.182 as 0.161 to 0.180.
  Lens fall-off is separated from the beam only through the ambient light
  (0.42 to 0.50 for an equivalent 0.6); in a dark room and with "difference"
  partners only the product is determined (beam 2.93 with lens -0.38; beam
  2.53 with lens 0). This describes the simple beam of the renders; it is not
  a characterisation of a phone torch.
* **Options** (K). Fitting the tone exponent and the lens fall-off helps (4.6;
  without lens fall-off the plain case gives -0.08 mm and a radius error of
  -0.10 mm instead of -0.01 and -0.02 mm). Second-order beam and lens terms
  and the albedo gradient make no difference here. **Fixing the torch at the
  lens is harmful**: +1.02 mm at a stated 0.43 mm on the plain case, and
  `supported` with -6.25 mm under one-sided room light. **A free beam tilt is
  unsafe under uneven light**: -4.71 mm, refused here by the held-out test,
  but returned as `supported` (-4.74 +- 0.70 mm) with the other gauge tag in
  the first complete run. Free ambient harmonics rescue one-sided room light
  without partners (-0.01 mm) at the price of a radius sigma of 0.34 mm with
  two views; during development the same option biased the radius with two
  views, so it stays off by default.
* **Runtime.** Photometric analysis, single core, NumPy: 0.27 s for one view,
  0.45 s for two, 0.8 s for three, 1.2 s for six torch-on views (median; up to
  5.2 s for a refused case whose left-out fits are repeated). Tag detection
  and bundle adjustment: 0.43 s per photo. Edge-only baseline through the
  session service: 6.6 s per sweep. The complete script took 898 s for the 127
  cases with a warm render cache, on a machine shared with another test suite;
  the first start spent 876 s rendering with three processes (most of the 342
  photos; a few were already cached).

### 4.10 Bottom line

* **Helps**: when the sweep was taken with the torch on, and most of all in a
  dark room or against a background as dark as the limb. It removes a
  silhouette bias of 0.4 to 1.9 mm, and its disagreement with the silhouettes
  says that the silhouettes of torch photos are biased.
* **Useless**: when torch-off photos with a visible silhouette exist
  (edge-only 0.02 mm, photometric 0.09 mm median); for the direction of the
  edge; as a model-free gradient estimate; as an inverse-variance partner of
  the silhouette fit; flash-off partners on a plain panel under even light.
* **Harmful if trusted**: smooth albedo variation without ratio partners
  (about -1.1 mm reported as `supported` +- 0.3 mm); the radius under a mild
  S-shaped tone curve (+0.5 mm at +- 0.16 mm); "difference" partners on a
  patterned surface; the options "torch at the lens" and "beam tilt"; the
  combination with silhouettes from the same torch photos; and any withheld
  estimate (median error 5.7 mm, up to 53 mm, over the 11 refused cases that
  still carry numbers).

None of this has been checked on a real photo (section 7). The module stays
research evidence with guidance disabled.

## 5. What did not work during development, and what was changed

All numbers in this section are from development-set runs (7 stations, 62
degree arc, 760 mm, R = 35 mm, ambient 0.25 unless stated; renderer poses).

1. **Limb contamination.** Using bend cells up to 85 degrees of bend and
   seen as obliquely as cos 0.15 biased the tangent line by +0.41 mm (formal
   sigma 0.04): lens blur mixes the dark limb with whatever lies behind it.
   With cells limited to a view cosine of at least 0.4 and 70 degrees of bend
   the bias fell to +0.07 mm. The silhouette itself is never used.
2. **Lens blur shifts a kink.** Remaining bias +0.041 mm without lens blur,
   +0.070 mm at 0.8 px, +0.157 mm at 1.6 px (0.045 mm per px squared). The
   model now carries a second-difference blur term per view (tag-edge blur
   plus pixel and sampling footprint): +0.037 mm at 0.8 px.
3. **Solver cost.** `scipy.optimize.least_squares` spent 3.0 s of 6.8 s in 29
   dense SVDs of an 18000 x 19 Jacobian. Replaced by a bounded
   Levenberg-Marquardt on the normal equations with Huber weights.
4. **First guess of the transition.** "Where the normalised brightness leaves
   1", scanned from the flat side, fired at the far end of the window (the
   nominal flat-field did not fit) and later at a paint strip 27 to 57 mm from
   the edge (fit ended 28 mm off). Now the flat field is fitted first and the
   guess is the start of the brightness *plateau* walking in from the limb;
   the strip case then gives -0.01 mm.
5. **Lens fall-off is not beam shape.** With vignetting 0.5 a model that
   folded lens fall-off into the torch beam gave -0.36 to -0.72 mm and -0.10
   degrees at a stated 0.03 degrees. Vignetting multiplies ambient *and* torch
   light, the beam only the torch light. With a separate fall-off term:
   +0.04 mm.
6. **Tone curve.** With the tone exponent fixed at sRGB, an extra gamma of
   1.15 / 0.85 gave -0.21 / +0.32 mm and a radius of 34.7 / 35.2 mm at stated
   sigmas of 0.12 mm (2 to 3 sigma). An effective tone exponent is now fitted
   with a weak prior (1 +- 0.15): +0.03 mm, exponent 1.156 +- 0.004.
7. **Slow, non-converging fits.** Adding the tone exponent and lens fall-off
   made the problem a curved valley: 1000 to 1600 model evaluations, 4 to 7 s,
   "not converged" at ambient 0.8 (condition number 3600, leave-one-out
   ratios of 40). The solver now works with the *level* of each view (log mean
   modelled brightness) instead of its gain, which decorrelates the lighting
   parameters from the gains: condition numbers 30 to 200, about 1 s for 7
   views, and a staged start (lighting on the flat face, then shape, then
   everything).
8. **A strong S-curve passed as "supported".** Tone contrast 0.4 gave
   +1.9 mm (exponent fixed) or +0.8 mm (exponent fitted) at stated 0.3 to
   0.5 mm. The held-out-view test (ratio limit 1.8, no absolute escape) and
   the per-view stationarity test (0.5 mm and four of the view's own sigmas)
   now refuse it.
9. **Surface patterns were not in the uncertainty.** Smooth albedo variation
   (10 %, 150 mm patches): -1.09 mm at a stated 0.27 mm. A faint dark band
   (2 %) 10 mm from the edge: -1.10 mm at 0.49 mm. The residual that stays
   fixed on the flat face across views is now turned into an allowance
   (coherent band next to the edge, plus the uncertainty of band means with a
   block-length scan for patterns that are smooth along the edge); both cases
   are refused as `surface_not_uniform`.
10. **First-harmonic ambient light is not a safe default.** It rescues
    one-sided room light with 7 views (0.01 to 0.04 mm) but with 2 views it
    moved the radius by 0.8 to 1.6 mm at 2 to 3 stated sigmas. It stays an
    option (`fit_ambient_harmonics`).
11. **Flash-off partners, difference formulation, attempt 1.** One ambient
    scale per pair: the tone exponent drifted to 0.66 and the radius by
    +0.4 mm, because the per-pair scale removes the cross-view tie between
    ambient level and torch level. Flash-off photos all show the same static
    scene, so they are now scaled to each other on the flat face and one
    shared scale remains.
12. **Attempt 2.** Still exponent 0.83 and -0.26 mm (-0.50 mm at ambient 0.8,
    2.8 sigma). Two causes: noise of the flash-off image acts as noise in a
    regressor (it is now smoothed along the edge and median-filtered across
    it, which keeps kinks and steps), and lens fall-off cannot be separated
    from the beam when the ambient part is measured instead of modelled (it
    is no longer fitted in that case). After both: -0.05 to +0.03 mm and an
    exponent of 0.95 to 1.01 on the development pairs.
13. **A "gloss" lobe at the broad end of its range** (shininess at the lower
    bound) acted as a second diffuse term and pulled the tone exponent to 0.65
    with two pairs (refused by the held-out test). Such a lobe is now dropped
    and the fit repeated: -0.02 mm.
14. **One-step leave-one-out** (a single Gauss-Newton step) gave ratios of 40
    to 130 when the problem was poorly conditioned. With up to four views, or
    a condition number above 1500, the left-out problems are re-solved.
15. **Beam tilt** as a free parameter made a two-view case fail to converge
    and is not exercised by the renderer (its beam is on the optical axis):
    off by default, unverified.
16. **Tag-pipeline poses** added up to 0.16 mm on the development sweep at a
    stated 0.06 mm. The module receives poses without uncertainty, so the
    stated tangent sigma has a floor of 0.10 mm.

## 6. Failure modes

Observed on the held-out set (section 3).

**Wrong and not announced** (status `supported`, error well outside the stated
sigma):

1. Smooth albedo variation at the scale of the analysed strip, torch-only or
   with "difference" partners: -1.1 mm and -0.7 mm at a stated 0.3 mm. The
   surface-pattern allowance under-estimates such patterns (0.22 mm). Only
   ratio partners remove the effect.
2. A mild S-shaped tone curve: radius +0.5 mm at a stated 0.16 mm. One tone
   exponent cannot represent an S-curve. The tangent line survives because it
   is located by where the brightness starts to depart from the flat
   prediction; the radius depends on how fast it departs.
3. A plane frame of the tag pipeline that is tilted by a fraction of a
   milliradian: up to 0.23 mm in the tangent line and 0.24 mm in the radius
   (about two stated sigmas; 2.7 for one radius). The module cannot see it.
4. Options away from the defaults: torch position fixed at the lens (+1.0 mm;
   -6.3 mm under one-sided light, `supported`); free beam tilt under one-sided
   light (`supported` with -4.7 mm in one of two runs).
5. Inverse-variance combination with silhouettes from the same torch photos:
   up to +0.38 mm, 4.2 combined sigmas.

**Refused** (no estimate is returned; the withheld numbers stay available in
`conditioning.withheld_estimate`): one view; no torch-on photo or unknown
torch state; clipped face; flat panel; joint line; torch shadow on a flat
panel; paint strip across the edge (torch-only and with "difference"
partners) or on the face; soft dark band next to the edge; one-sided room
light (torch-only and with "ratio" partners); strong S-curve; 45-degree bend
with flange; wall of panel albedo close behind the edge. Where a refused case
still carries numbers they were wrong by 1.2 to 53 mm, except in the one-view
case (-0.10 mm), which is refused only because nothing cross-checks it. The reason
codes of one and the same scene can change with small changes of the poses
(torch shadow, group J): they are hints for the researcher, not a
classification.

**Cannot run**: photos in which the tags are not detected (exposure x0.04).
There are no poses, for either method.

**Requests for another view** are specific for: too few views (`add_view`,
about 150 mm across the edge), no torch-on photo (`torch_on_sweep`), edge
region clipped, too dark or out of frame (`reframe_edge`), weak torch share
(`increase_torch_share`), bend not covered (`move_toward_edge`), weak
conditioning (`add_view` or `closer_view`), unusable partners
(`retake_pair_from_rest`). For surface patterns, uneven light and tone curves
the module refuses and says why; another torch photo does not fix those, and
the suggestion text says so.

**Slow cases**: a refused fit can take 2 to 5 s instead of about 1 s (the
left-out fits are repeated).

**Moving ground**: the tag pipeline and the session service were being edited
by another engineer while these runs were made. Between the two complete runs
(22 minutes apart) the photometric results of 50 of the 51 cases with
unchanged inputs were identical to six decimals and one moved by 0.008 mm (a
change in the tag detector's covariance); the edge-only tangent errors moved
by up to 0.008 mm and its radius errors by up to 0.04 mm (0.09 and 0.13 mm in
the clipped-exposure and the 45-degree case). The tables are those of the
working tree as it was at 16:55 UTC on 2026-11-25.

## 7. What the synthetic photos do not contain, and what is not verified

The renderer (`lumen/synth/render.py::_shade`) is a single-bounce model: a
uniform ambient term that does not depend on the surface normal, one optional
distant directional light, a point torch with inverse-square fall-off, a
`cos^m` beam about the optical axis, Lambert shading, a Blinn-Phong lobe and
shadows cast by boxes only. The fitted model contains the same terms. That is
an inverse crime: the tables show what the method can do when its physics is
right, not what a phone will give. Not in the photos, and therefore **not
tested**:

* inter-reflection (floor and wall light the bend; a bright floor near a
  bottom corner is typical);
* ambient light that depends on the normal in any way other than one distant
  source (ceiling panels, windows, the operator's body blocking light);
* a real torch: beam not centred on the optical axis, hot spot and rings,
  colour, PWM flicker with a rolling shutter, brightness drift; and whether a
  web page can switch it at all on the target phone (not established, see
  docs/BUILD_LOG.md);
* a real camera pipeline: local tone mapping, noise reduction, multi-frame
  fusion, lens-shading correction, white balance and exposure changing inside
  a pair. Only global curves (gamma, S-curve), global sharpening, JPEG and
  radial vignetting were rendered;
* real surfaces: non-Lambertian paint (sheen at grazing angles), Fresnel
  gloss, embossed decor with micro-shadowing, dirt, protective film;
* real geometry: a bend that is not a circular arc, spring-back, a face that
  is not flat near the edge. A bend of 45 degrees followed by a straight
  flange is refused (group J); nothing else was tried;
* pose error beyond what the synthetic tag pipeline produces (true intrinsics,
  flat stickers, no rolling shutter);
* the Posit Connect runtime and browser capture path.

Also not verified: `fit_beam_axis` (the renderer cannot tilt the beam), the
light-offset estimate along the optical axis (fixed at 0), radii below 20 mm
and above 60 mm, tag layouts other than the demo constellation, photos other
than 1600 x 1200, more than 7 views, panels whose tags are not coplanar with
the face next to the edge, and whether the two exploratory frame variants of
group G hold beyond the six photo sets shown. The module is not called by the
session service or the API: the integration (building `FlashView`s from
frames, pairing by `pair_id`, storing the diagnostics) does not exist yet.

"Characterise the real device flashlight" was **not feasible** here (no
device). The module estimates torch position and beam shape as nuisance
parameters and reports them with standard deviations; the table "Torch
position and beam recovered from the photos" shows how well that works on
renders. Whether those estimates mean anything on a phone is unknown.

## 8. How to read a result

`PhotometricResult.to_dict()` carries: `status`, machine-readable `reasons`,
operator-readable `suggestions`, `requests` (codes such as `add_view`,
`move_toward_edge`, `closer_view`, `increase_torch_share`,
`retake_pair_from_rest`, `torch_on_sweep`, `reframe_edge` with a direction in
the plane frame, for a guided sweep), the tangent line, axis direction and
radius with standard deviations (`None` where not identified), `conditioning`
(singular values, shape correlations, the three variance estimates, the
surface-pattern allowance, per-view transition offsets and, when the result
was withheld, the withheld numbers for inspection), `heldout`, `nuisance`
(beam, lens fall-off, ambient ratio, torch offset, gloss, tone exponent),
`gradient_axis` (the model-free axis estimate), per-view diagnostics (role,
gain, exposure ratio of a pair, mask fractions, held-out error) and a few
cross-section profiles (position, measured / predicted-flat brightness, model).
It is labelled `research_evidence_only` and `guidance_enabled` is always
`false`.
