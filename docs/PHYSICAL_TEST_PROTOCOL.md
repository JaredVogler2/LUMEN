# Physical test protocol

**Status: NOT RUN.** No part of this protocol has been carried out. Every
accuracy figure in this repository comes from computer-generated images
(`docs/BUILD_LOG.md`, `docs/benchmark/RESULTS.md`). Until the tables at the end of
this document are filled in from real measurements, LUMEN has no validated
domain and its output is research evidence only.

The protocol answers four questions, in this order:

1. How far is a LUMEN target position from the position found by an
   independent instrument (bias)?
2. How much does it vary between repeated captures and between operators
   (repeatability, reproducibility)?
3. Does the uncertainty LUMEN states contain the error as often as it should
   (coverage)?
4. Do the checks stop the faults they are meant to stop (seeded faults)?

Use non-production panels only. Do not photograph production hardware,
drawings or documents for this study, and do not store real part data in this
repository.

## 1. Equipment

| Item | Requirement | Record |
|---|---|---|
| Flat panel | rigid, at least 600 × 600 mm, matt or satin face, with a straight visible seam or scribed line as side datum and a straight lower edge or floor line as floor datum | flatness measured (section 2) |
| Panel with rounded side edge | same, one side bending away with a uniform radius; two radii if available (for example about 25 mm and 50 mm) | radius measured (section 2) |
| AprilTag stickers | family 36h11, nominal 50 mm black square, commercially printed, matt; at least 8 different IDs | each one measured (section 2) |
| Calibration board | ChArUco board from the app (Settings), printed and mounted flat | square and marker size measured |
| Phones | the managed iPhone models that would be used, default rear camera, Safari | model, iOS version, lens used |
| Reference instrument | expanded uncertainty at most one quarter of the tolerance under study, calibrated; a coordinate measuring machine, a height gauge on a surface plate, or a laser tracker | type, serial number, calibration date, stated uncertainty |
| Caliper | resolution 0.01 mm, calibrated | serial number, calibration date |
| Steel rule or tape | class I or II, for the independent scale check | class, calibration |
| Light meter | lux | model |
| Target marks | 10 or more fine cross marks or small printed dots at known nominal places on each panel, spread over the area where placards would go, including places far from the tags | reference coordinates (section 2) |

## 2. Reference measurements (before any capture)

Made by someone who does not operate the phone in section 4, and kept away
from the operators until the analysis.

1. **Tags.** Measure the black square of every sticker, not the sticker
   paper: width and height, three times each, after it is applied to the
   panel. Record the mean and the range. Enter the mean as the measured size.
2. **Panel flatness.** Deviation from a plane over the measured area
   (straightedge and feeler gauges, or the reference instrument). Record the
   largest deviation. The uncertainty budget assumes 0.3 mm unless told
   otherwise.
3. **Datum features.** Straightness of the floor line and of the side seam;
   the angle between them; for rounded edges the radius and where the flat
   face ends (tangent line), measured with the reference instrument.
4. **Target marks.** Coordinates of each mark in the frame defined by the two
   datums exactly as the operation record defines it (sign, offsets, axis
   rule). Three repeats; record the reference uncertainty.
5. **Operation records.** Enter one record per panel in the app with these
   datums and with each mark as a "placard" of small size whose reference
   point is its centre. The tolerance entered is the engineering need under
   study (section 6), not a value taken from LUMEN.

## 3. Factors

| Factor | Levels |
|---|---|
| Panel | flat; rounded edge radius 1; rounded edge radius 2 |
| Distance from phone to panel | 0.4 m, 0.7 m, 1.0 m |
| Viewing angle to the panel normal | about 0°, 20°, 40° |
| Roll of the phone | 0°, about 15°, 90° |
| Lighting | bright diffuse (over 500 lx); dim (under 100 lx); one-sided raking light; phone torch on |
| Number and placement of tags | 2, 4, 6; around the target area, and all on one side of it |
| Capture | one photo; five to nine photos from different sides; video sweep |
| Lens profile | made on the day; none |
| Operator | at least 3, trained with the quick start only |
| Phone | at least 2 units |
| Setup | tags removed and applied again: at least 3 times |

A full crossing is not practical. Use two studies.

**Study A, repeatability and reproducibility (gauge R&R, crossed).** One
panel, one lighting, one distance and angle that represent intended use. Parts
are the 10 target marks. 3 operators × 3 repeats × 10 marks = 90 results per
capture mode, each repeat a completely new session (new photos, datums found
again). Repeat for the rounded-edge panel.

**Study B, sensitivity.** One operator. Vary distance, angle, roll, lighting,
tag placement and lens profile one factor at a time around the Study A
condition, and in the corner combinations that are expected to be worst
(far + oblique + dim; tags on one side + target far away). 3 repeats each.

## 4. Procedure for one capture

1. Start a new session, choose the record and the mark. Enter the measured
   tag sizes.
2. Attach the lens profile (or none, as the run sheet says).
3. Capture as the run sheet says. Do not look at earlier results.
4. Analyse, review the datums, confirm. Do not correct a datum by hand unless
   the app asks for it; if it is corrected, note it.
5. Make the independent scale check with the steel rule if the run sheet
   includes it.
6. Export the session summary (JSON). The export holds the detections, the
   registration, the datums, the uncertainty budget, the gate report and the
   software version. Name the file with the run number only.
7. Note light level, distance, anything unusual, and the time taken.

For verification runs (placard already applied): apply a test label at a
position measured with the reference instrument, then use the Verify step and
export.

## 5. Seeded faults

Each fault is set up without telling the operator. A fault counts as stopped
when the app shows no target outline or says "blocked" with the right reason.

| Fault | Expected reaction |
|---|---|
| Tag size entered 2 mm too large for one tag | that tag is excluded or the tags are reported inconsistent |
| The same wrong size entered for all tags | not detectable by the tags; must be caught by the independent scale check |
| One tag moved between photos | reported as moved; no result from those photos |
| One tag on the trim or the curved edge | reported as not on the plane |
| Tag sizes of the sticker paper instead of the black square | caught only by the independent scale check |
| Wrong side seam chosen where two seams are visible | ambiguity reported, or the datum check fails |
| Floor line hidden | no floor datum; blocked |
| Glare on a tag | tag rejected |
| Lens profile of another zoom setting | profile refused or flagged as inconsistent |
| Phone refocuses to a much closer distance than at calibration | to be observed: the app has no check for this |

Record for each: detected or not, and what the app said.

## 6. Thresholds

To be filled in by the engineering owner before the study, from the actual
need of the operation under study. LUMEN does not supply these numbers and the
defaults in `lumen/policy/gates.py` are placeholders.

| Quantity | Value | Source |
|---|---|---|
| Position tolerance of the operation, ± mm | NOT SET | |
| Rotation tolerance, ± degrees | NOT SET | |
| Largest acceptable share of the tolerance used by measurement variation (6 × combined R&R standard deviation ÷ tolerance width) | NOT SET | |
| Largest acceptable bias, mm | NOT SET | |
| Smallest acceptable coverage of the stated expanded uncertainty | NOT SET | |
| Largest acceptable rate of results shown as guidance that are outside tolerance | NOT SET | |
| Faults in section 5 that must be stopped every time | NOT SET | |

Sample size: to show with 95 % confidence that a failure rate is below *p*
when no failure is observed, about 3 ÷ *p* independent trials are needed (300
trials for 1 %). Study A alone cannot support small failure rates.

## 7. Analysis

Error of one result = LUMEN position − reference position, separately for x
and z, and the rotation difference. Keep the sign.

* **Bias:** mean error per axis with its confidence interval, per condition.
  Report it separately from the spread; do not fold it into one number.
* **Repeatability and reproducibility:** analysis of variance of the crossed
  study (mark, operator, mark × operator, repeat). Report the standard
  deviations of repeat (repeatability), operator and interaction
  (reproducibility) and their combination, and the share of the tolerance from
  section 6.
* **Coverage:** share of results whose error magnitude is within the expanded
  uncertainty the app stated, with a Clopper-Pearson interval. Also list the
  largest budget term for results that were not covered.
* **Guidance status:** of the results shown as "prototype guidance", the share
  outside tolerance (false acceptance), with its interval; of the results
  shown as "research preview" or "blocked", the share that would have been
  inside tolerance (lost results).
* **Sensitivity:** error and stated uncertainty against each factor of
  Study B; note where the error grows faster than the stated uncertainty.
* **Faults:** table of section 5.
* **Time:** median and longest time per capture.

Keep the raw exports, the reference sheets and the analysis script together,
outside this public repository if they contain anything that is not generic.

## 8. Separate system tests

| Test | Status |
|---|---|
| **Live placement, agreement with a reference.** Ten placements per operator and phone with live placement, phone on a stand: after each automatic record, measure the placard's position with the reference instrument. Report the difference between the record and the reference (bias, spread), the share of records whose stated uncertainty covers that difference, and any record announced as within tolerance that the reference puts outside it | NOT RUN |
| **Live placement, hands.** With the placard held as an installer holds it (one end, both ends, from above, with a squeegee): how often a reading is shown, how often a side is drawn dashed, and every case in which the shown position is off by more than its stated uncertainty while a hand is on the placard (compare with the record taken after letting go). Gloves of the panel's colour are a required case | NOT RUN |
| **Live placement, not flat.** The placard held 2, 5 and 10 mm off the surface over the target, square-on and from 30°: is it refused as not flat, or is the displacement inside the stated uncertainty? | NOT RUN |
| **Live placement, timing and sound.** Results per second and delay on the deployed server over the site's network; time from letting go to the chime; whether the tones are heard in the workshop with the ring switch in both positions | NOT RUN |
| Managed iPhone, Safari: camera permission, rear camera selection, photo, video, torch control, live guidance, through the deployed HTTPS address | NOT RUN |
| Posit Connect: publish, start-up, sub-path addresses, session storage with more than one process, time limits, upload limits (see `docs/POSIT_CONNECT.md`, section 9) | NOT RUN |
| Lens profile made with a real print and a real phone, checked on a second print | NOT RUN |
| Bar code and QR reading of real labels at working distance | NOT RUN |

## 9. Results

| Item | Value |
|---|---|
| Date, place, people | NOT RUN |
| Panels, tags, phones (models, iOS versions) | NOT RUN |
| Reference instrument, uncertainty, calibration date | NOT RUN |
| Number of sessions in Study A / Study B / fault trials | NOT RUN |
| Bias x, z (mm), rotation (degrees) | NOT RUN |
| Repeatability, reproducibility (mm) | NOT RUN |
| Share of tolerance | NOT RUN |
| Coverage of the stated uncertainty | NOT RUN |
| False acceptance rate at guidance status | NOT RUN |
| Faults stopped | NOT RUN |
| Location of raw data and analysis | NOT RUN |

## 10. From results to a validated domain

Only after section 9 is complete and meets section 6 may an administrator
describe the tested range in a file and name it in
`LUMEN_VALIDATED_DOMAIN_FILE`:

```json
{
  "id": "study identifier",
  "evidence": "report number, date, where the raw data is kept",
  "limits": {
    "min_tag_edge_px": 0,
    "max_view_angle_deg": 0,
    "max_target_offset_mm": 0,
    "requires_calibration": true,
    "min_tags": 0,
    "max_expanded_uncertainty_mm": 0
  }
}
```

Each limit is the edge of what was actually tested (the zeros above are
placeholders, not suggestions). A result is reported as "validated" only when
every gate passes and every limit is met; a limit name the software does not
know makes the check fail. No such file ships with LUMEN.
