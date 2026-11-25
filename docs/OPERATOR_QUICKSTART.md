# Operator quick start

LUMEN shows where a placard goes on a panel, from photos taken with a phone,
and afterwards measures where it actually is. It is a **research prototype**:
what it shows is measurement evidence, not an acceptance decision. Nothing in
it has been validated on real hardware (`docs/PHYSICAL_TEST_PROTOCOL.md`).

## What you need

* A phone with a rear camera and a current browser. Open the address your
  administrator gave you; it must start with `https://`.
* Two to six AprilTag stickers (family 36h11, about 50 mm), each with a
  different ID.
* A caliper to measure the black square of each sticker. A steel rule for the
  optional scale check.
* Good, even light. No glare on the stickers.

## Try it without a panel

On the first screen choose **Load demo session** (a flat panel with a seam) or
**Demo with a rounded edge**. Each loads computer-generated photos of an
invented panel and runs the real measurement on them. Every screen says
"simulated" while a demo is open.

## The seven steps

The row of numbered steps at the top shows where you are. A step opens when
the ones before it have what it needs.

### 1. Identify

Pick the operation and the placard, or press **Scan** and point the camera at
the bar code or QR code of the work order, panel or placard. LUMEN selects a
record only when the code matches exactly one. Read the datum descriptions on
this screen: they tell you which line is the floor datum and which is the side
datum. If the record is marked *incomplete* or *draft* you can look at a
preview but not get guidance.

### 2. Markers

Stick the tags flat on the flat face of the panel:

* around the place where the placard goes, not all on one side of it;
* not on trim, not on the floor, not on a curved part, not over a seam;
* each one fully visible, flat, without bubbles.

Measure the **black square** of each tag with the caliper (the picture on the
screen shows which edge) and enter the value next to its ID. Do not enter the
size of the sticker or the number on the packaging. Tags without a measured
size keep the result at "research preview".

### 3. Capture

Press **Open camera**. Hold the phone 50 to 90 cm from the panel.

* One photo is enough for a flat panel when all tags, the floor line and the
  side seam are in it.
* Three to nine photos from different sides (move about a step between them)
  let LUMEN check that the tags are on one plane and are required for a
  rounded side edge. **Sweep video** does the same: record up to 30 s while
  moving slowly around the panel; LUMEN picks sharp frames from it.
* The coach line under the camera picture says what is missing: closer,
  further left, hold still, glare.

Then press **Analyse**. Do not touch the tags between photos. If a tag was
moved, start the capture again.

### 4. Datums

LUMEN proposes the floor line and the side line and shows how sure it is.
Zoom in and check that each line lies on the feature the record names. If a
line is wrong, choose its drawing tool and drag along the correct edge; with
"snap" on, the line is fitted to the edge under it. Press **Confirm datums**.

If two similar edges are close together LUMEN asks instead of choosing.

Each datum shows what is known about it lying on the panel face:
"verified on the face in 3-D" (the photos show it), "declared on the face
(not verified)" (the record says so and these photos cannot tell, which is
normal for a floor line photographed from one height), or "NOT on the face"
(the photos show it is not; it cannot be used).

For a rounded side edge the screen also shows the fitted radius. The card
"Flashlight experiment" is research only: it compares a second, experimental
estimate with the first and never changes a datum.

### 5. Rectified

The panel face as seen straight on, with a millimetre grid from the datums.
Things that are not on the face (floor, trim, the curved edge) look displaced
here; that is expected.

**Independent scale check (recommended):** every length LUMEN reports comes
from the tag sizes you entered. If all of them are wrong in the same way, no
photo can show it. Press *Show the two corners*, measure that distance on the
panel with a steel rule, type it in and press **Check**. If it contradicts the
tags, no target is drawn until the sizes are corrected.

### 6. Place

The outline of the placard on the photo, with its distances from the datums.
The coloured banner says what you are looking at:

| Banner | Meaning | What to do |
|---|---|---|
| No guidance | something required is missing or contradictory; no outline | read the reason and the suggestion under it |
| Research preview | an outline, dashed; at least one check is not met | open the list of checks; usually: measure the tags, add a lens profile, take more photos, move closer |
| Prototype guidance | every check passed | use it as guidance within this study |

The stated uncertainty (± mm) is a property of this capture. It is compared
with the tolerance in the record; LUMEN never supplies a tolerance of its own.

**Live guidance** shows the outline on the moving camera picture. The outline
is drawn only for a frame that was just measured; when tracking is lost it
disappears and the screen says why. A still photo is always labelled
"Frozen photo — not live".

Several placards on the same panel: choose the next one with the placard
buttons in this step. The photos and datums stay valid.

### 7. Verify

After the placard is on the panel, take one more photo (same tags, not
moved). LUMEN finds the placard's outline and reports how far it is from the
target, in millimetres along x and z and in degrees, with an uncertainty:

* **within**: inside the tolerance even allowing for the uncertainty;
* **outside**: outside it even allowing for the uncertainty;
* **indeterminate**: too close to call, or the outline had a problem (hidden
  corner, glare, curled edge). Take a better photo.

Check the drawn outline against the picture, then press **Confirm measured
outline** and **Export summary**. The summary is a JSON file with the
measurements, the checks and the software version. Photos are not in it.

## Lens profile (once per phone)

Menu → **Settings & camera calibration**. Download the board, print it,
mount it flat, measure its squares, and take 10 to 15 photos of it as the
screen describes. Save the profile; it stays on this phone. Attach it to a
session from the same screen. Without a profile results stay at "research
preview".

Use the same camera (lens), zoom and orientation for the board photos and for
the panel photos. A profile is not applied to photos that report another
camera, zoom or size.

## When you are done

Menu → **End session and delete photos**. Sessions are also deleted
automatically some hours after their last use.

## What LUMEN does not do

* It does not decide whether a part is acceptable.
* It does not know which way is down: directions come from the two datums in
  the record.
* It does not measure on curved surfaces. A rounded edge can serve as a datum;
  the placard itself must be on the flat face.
* It does not invent a missing datum, offset or tolerance.
* It cannot notice that every tag size was entered wrong in the same way
  unless you make the scale check.
