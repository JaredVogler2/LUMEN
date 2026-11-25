# Troubleshooting

For operators and for whoever runs the server. Deployment problems on Posit
Connect are in `docs/POSIT_CONNECT.md`, section 10. Behaviour on a real
iPhone has not been tested (NOT RUN); the phone entries below follow the
published browser behaviour and the messages the app shows.

## Camera

| What you see | Likely cause | What to do |
|---|---|---|
| "Camera access was refused" | permission denied for this site, or the page is not served over HTTPS | allow the camera for the site in the browser's site settings and reload; check that the address starts with `https://` (or is `localhost`) |
| "No rear camera was found" | no camera, or the browser hides it | use **Choose photos** and take the pictures with the phone's camera app |
| "The camera is in use or was interrupted" | another app holds the camera, or a call came in | close the other app, open the camera again |
| "The camera did not start" | a permission prompt is waiting, or the camera is slow to start | answer the prompt, try again |
| The picture is from the wide or tele lens | the browser chose another rear camera | open the camera options (gear) in the camera sheet and pick the main camera |
| No torch button | the browser or the phone does not offer torch control to web pages | use room light; torch photos are needed only for the experimental flashlight diagnostics |
| Recording is not offered | the browser cannot record this camera | take single photos from several sides instead |
| Photos look soft | motion, or the lens is focused elsewhere | hold still, tap to focus if the phone allows, more light |

## Tags

| Message | Meaning | What to do |
|---|---|---|
| too small in the image | a tag edge is under the smallest size that can be measured | move closer, or use full-resolution photos instead of video frames |
| cut off at the image border | part of the tag is outside the picture | re-aim or step back |
| seen at too steep an angle | the tag is strongly foreshortened | face the panel more directly |
| glare | highlights are clipped on the tag | change the angle a little; turn off the torch |
| bent or distorted | the tag's edges are not straight | re-stick it flat; do not place tags on curved parts |
| decode mismatch | the pattern does not match the ID after refinement | clean the tag; check for a partly covered tag |
| edge disturbed | something crosses the tag's border (a seam, a shadow edge, a finger) | move the tag or the obstruction |
| "Tags contradict each other" | sizes or positions do not fit one plane | check each entered size against the right ID; all tags must be flat on the same face; do not move tags between photos |
| "A registered tag appears to have moved" | a tag is somewhere else than in the registration photos | start the capture again |
| "A tag was excluded" on the Datums screen, "A tag was left out of the measurement" in the list of checks | one tag contradicts all the others (it moved between photos, is not on the same plane, or its entered size is wrong) and was set aside; the result rests on the remaining tags and the reason is given beside the ID | look at that tag: re-stick it flat, correct its size, or capture again without touching the tags. With fewer tags the uncertainty is larger |
| Two stickers with the same ID | the ID is ambiguous | remove one |

## Analysis and datums

| What you see | What to do |
|---|---|
| "No accepted AprilTag: there is no metric scale" | at least one measured tag must be fully visible and accepted |
| "Only one tag: nothing checks its size, flatness or the plane assumption" | add a second tag on the other side of the target |
| "Two different camera poses explain the tags almost equally well" | add a tag, or take the photo from closer or from more to the side |
| "The views are too similar to verify coplanarity" | move further between photos (about a step) |
| No floor line or side line is proposed | make sure the line is in at least one photo with the tags; draw it by hand with snap on |
| "Only one straight line was found in the photos" (or "N straight lines were found … but no two of them fit together") | a floor line and a side line that meet at the record's angle are both needed, and one of them is too faint, hidden or outside the photos. From one line LUMEN cannot tell which. Retake with both clearly in view, or draw the lines by hand with snap on |
| "A competing edge of similar strength nearby" | zoom in and draw the line on the edge the record names |
| "Is this the right line?" on a datum (check: "A fixed edge about … mm in front of the panel face runs … the floor datum") | the line LUMEN found lies on the face, but a raised or recessed edge runs along it, as at the top of a kick strip or beside a moulding. The line may be the edge of that part and not the feature the record means, and LUMEN cannot know which. Look at the photo: if it is the right line, tap **Choose detected line** and select it yourself; if not, draw the datum on the right feature. Until then the result stays a preview |
| A datum is marked "declared on the face (not verified)" with the note "the photos do not settle whether this line lies on the panel face" | nothing is wrong: all photos saw that line from similar heights (or similar sides), so they cannot show whether it is on the face to a millimetre, and the record's statement is used. To have it verified, add photos from clearly different heights for a horizontal line, or from further left and right for a vertical one |
| A datum is marked "NOT on the face" | the photos show the whole feature clear of the panel face (a proud trim, an edge behind the face): choose a feature on the face |
| "Left out because it did not fit the other photos of the same tag" | one tag was mis-measured in one photo (glare on an edge, something in front of it); that observation is ignored and the tag is kept. If it happens for several tags, retake the photo |
| "The local frame cannot be established" | the two lines are nearly parallel or their angle is outside the record's limit: check that the right features were chosen |
| Rounded side edge: "a single planar view cannot locate it" | take five to nine photos around the edge so that its outline is seen from different sides |
| "Contradiction" in the scale check | the declared tag sizes and the rule measurement disagree: measure the black squares again, check the rule reading, correct the sizes |
| The target is "outside this frame" | take a photo that shows both the tags and the place of the placard |

## Status stays at "research preview"

Open the list of checks on the Place screen; the failed ones are at the top,
each with what to do. The usual ones:

| Check | Fix |
|---|---|
| tag sizes not measured | enter the caliper value for each tag |
| no lens profile | make one in Settings and attach it; use the same camera, zoom and orientation |
| lens profile disagrees with the tags | the profile is from another camera or zoom: recalibrate or detach it |
| datums not confirmed | review and confirm them |
| uncertainty exceeds the share of the tolerance | move closer or use full-resolution photos (larger tags), put tags around the target, add photos; the list of budget terms shows which term dominates. Tags under about 52 px across cannot measure their own edge shift: the Capture screen warns ("The tags are small in the photos") |
| target far from the tags | put a tag near the target |
| tag observations still contradict the other photos | glare on a tag, or more than one tag was moved, or too few tags are left to tell which one is wrong: retake without glare and without touching the tags; with a third or fourth tag a faulty one can be identified and set aside |
| "Is this the right line?" (a raised edge beside a datum) | see "Analysis and datums" above: choose the line yourself |
| record incomplete or draft | the record has to be completed and released |

Entries with an "i" in that list passed, but say what was not checked: for
example that the scale rests on the entered tag sizes alone because no rule
measurement was made, or that a datum is on the face by the record's word
only. They do not lower the status; each says how to close the gap.

## Verification

| What you see | What to do |
|---|---|
| No placard is found | the placard must be within a few centimetres of the target and clearly different from the panel; take the photo square-on with even light |
| "indeterminate" although the number looks fine | the outline had a problem (hidden corner, glare, curl, wrong size) or the uncertainty reaches the tolerance: retake, or mark the corners by hand and say so in the record |
| The photo is refused as evidence | a registration photo cannot show the installed placard: take a new photo after installing it; the tags must be the registered ones and must not have moved |

## Sessions and uploads

| What you see | Cause |
|---|---|
| "Session not found" after a pause | sessions are deleted some hours after their last change (default 4 h), when the server is redeployed, or when another server node without shared storage answers |
| Upload refused: too large | photo over the size limit (default 25 MB / 50 megapixels) or video over the limit (default 300 MB / 60 s analysed) |
| "The server is busy with other measurements" | all computation slots are in use; try again in a few seconds |
| A video stays at "processing" | the job has a time limit (default 5 min); a lost worker is reported after 45 s without progress: upload again |
| Records entered in the library disappeared | they were kept in temporary storage: set `LUMEN_OPERATIONS_DIR` to a persistent directory; keep exported JSON files |

## For the administrator

* `GET <base>/api/health` reports the software version and self-checks (tag
  detection on a generated tag, image codecs, storage, the operation
  library). `degraded` names the failed check.
* `python scripts/smoke_test.py <address>` walks through a simulated session
  on a running instance and prints each step.
* Logs: an unexpected error is logged with its traceback and a short
  reference; the same reference is shown to the user ("Unexpected server
  error (reference 1a2b3c4d)"). Refusals the app expects (no tag, blocked
  geometry, limits) are answers, not log entries. Request lines come from the
  web server's own access log; they contain session IDs as part of the
  address, so treat that log as confidential.
* Nothing in LUMEN needs outbound network access unless the optional
  assistant is enabled (`docs/ASSISTANT.md`).
