# LUMEN on a host that keeps nothing (Vercel and other function platforms)

LUMEN is made for Posit Connect (`docs/POSIT_CONNECT.md`). This page is about
the other kind of host: a function platform such as Vercel, AWS Lambda,
Netlify or Azure Functions, which is a convenient place to try the
application on a phone and is not where it is meant to run at work.

Status: tested against a stand-in for such a host, and tried once on a
preview deployment on Vercel (Hobby plan, 2026-11-26), from a data centre,
with the test client and with Chromium and a fake camera (results below).
**Not tried on a phone.**

## What is different there

Placing one placard takes several requests in a row: the photos, the analysis
that finds the tags and the two lines, then live frames measured against
that analysis. Something has to remember the analysis for those few minutes.

| | Posit Connect, a container, a virtual machine | a function platform |
|---|---|---|
| what answers a request | a process that keeps running | whatever copy of the program the platform starts or reuses; possibly one that has never seen the job |
| who keeps the job while it is open | the server (in memory, or in a temporary directory shared by its processes) | **the phone** |
| what is left on the server after a request | the job, until it ends or expires | nothing |

Where the phone keeps the job, every request that belongs to it brings the
job's state (a document of 60 to 90 kB, 20 to 30 kB compressed) and the
photos that step looks at; the answer brings the state back as the request
left it. The server computes, as before, and remembers nothing. The
measurement code is the same and gives the same numbers
(`tests/integration/test_carried_flow.py`).

LUMEN chooses by itself: on a platform it recognises as a function platform
(`VERCEL`, `AWS_LAMBDA_FUNCTION_NAME`, `FUNCTIONS_WORKER_RUNTIME`, `NETLIFY`
in its environment) the phone carries the job; everywhere else the server
keeps it. `LUMEN_STORAGE` overrides the choice.

Nothing is stored for long in either case, and there is no database in
either. Photos are for the job: on a server they go when the session ends or
expires; on the phone they go when the session ends, and a job that was left
behind (a closed tab) is removed the next time the page is opened, two hours
after it was last used.

## What it costs

The phone carrying the job does **not** make anything faster. The measuring
is done on the server either way; the phone only holds the data.

| | server keeps the job | phone carries the job |
|---|---|---|
| photos | as the camera made them (analysed up to 4096 px) | reduced by the browser to 1600 px and at most 0.41 MB each, so that the nine photos a capture may have fit into one request together |
| what that means for the result | the benchmark at 4032 px: stated uncertainty about 0.7 mm | the benchmark at 1600 px: about 1.1 to 1.2 mm; enough to place a placard to ±2 mm with a lens profile, usually "Check needed" |
| a video, the recorded sweep | yes | no: a video is worked on after its upload has ended, which needs a server that still has the job then |
| records entered in the app | yes | no: the bundled sample records only (there is no disk that would keep them) |
| a lens profile | for the photos as taken | has to be made on the same host: it is for the reduced size |
| one live reading, server time, phone on a stand | 30 ms | 36 ms when the same copy of the program answers; 1.3 s from a copy that has never seen the job, which solves the registration again |
| sent with every live frame beside the frame | nothing | the job's state, 23 kB |
| a whole job, first request to export | | 11 requests; the largest 1.1 MB (the five demo photos) |

(Server times are of this development machine, one core, no network.)

How fast live placement is on such a platform therefore depends on how often
the platform answers with a copy that already knows the job. That is the
platform's decision. On the one deployment tried it almost always did (next
section).

## On Vercel, measured once

One preview deployment of this code on Vercel's Hobby plan, reached from a
data centre (so with a better connection than a phone has). First with the
test client (`tests/carry_client.py`, the same client as in the tests), then
with Chromium, a phone-sized window and a fake camera playing the film of a
placard that is brought to its target.

| | |
|---|---|
| the page opens (first request, instances cold) | 6.8 s |
| the five demo photos, tags measured | 3.6 to 4.4 s |
| registration | 1.3 s |
| finding the lines (the five photos go up, 0.9 MB) | 15.9 to 18.9 s |
| the target on a photo; the straight-on picture | 0.7 s each |
| a live reading, request to answer | median 0.15 s (0.13 to 0.19); the server's own share 0.04 s |
| the first reading on an instance that has not seen the job | 1.4 to 1.6 s |
| readings answered from a held registration, after the first six | 23 of 24 |
| sent with each live frame | 92 kB (frame 69 kB, job 23 to 25 kB) |
| live placement from the film in Chromium | guiding after 2.4 s, in position after 5.2 s, recorded by itself after 9.6 s; 17 readings; the tones commanded in order |
| the record | 0.33 mm and 0.20 mm from the target (the film ends at 0.30 and 0.20) |
| a reload in the middle of the job | the job is back, from the browser's own storage |
| the whole job with the test client | 42 requests, 30 of them live frames; 7 photos sent; no step had to ask for a photo |

The numbers of the demo job are those of every other run of it (the record
reads 1.44 mm and -0.90 mm against 1.4 and -0.9).

What this does not say: how it is from a phone on a mobile network (each
live frame is 92 kB up), on another plan or another day, or with several
people at once.

## The limits of the platform

As Vercel documents them (read on 2026-11-26; they are the platform's to
change): a request and an answer of at most 4.5 MB; with Fluid compute a
function may run 300 s on every plan; 1 vCPU and 2 GB on the Hobby plan;
several requests may share one instance, and nothing promises that two
requests in a row reach the same one.

LUMEN keeps under 4.0 MB per request and 4.3 MB per answer by default. The
slowest step measured here, finding the lines in five photos, took 12 to 16 s
on one core.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_STORAGE` | `client` on a recognised function platform, else `file` | `file` or `memory`: the server keeps the job. `client`: the phone carries it |
| `LUMEN_STATE_SECRET` | empty | with `client`: a secret with which the server signs each state it hands out. A state it did not write, or that was changed, is refused. Without it a state is checked for its form only. Set it for anything but a private trial |
| `LUMEN_CARRY_MAX_REQUEST_MB` / `LUMEN_CARRY_MAX_RESPONSE_MB` | `4.0` / `4.3` | the largest request and answer through the carrier |
| `LUMEN_CARRY_PHOTO_MAX_EDGE` | `1600` | the size to which the browser reduces a photo |

A server that is told to keep jobs (`LUMEN_STORAGE=file` or `memory`) on a
function platform cannot work. It says so on its first screen and reports
`degraded` from `api/health`.

## What a state from a phone is trusted with

A state that comes back from a browser is an input like any other. It is read
by the same models that read a stored session and refused if it does not fit
them. A photo is taken only if it has the checksum that the state records for
its frame, so nothing is measured in another picture than the one the tags
were found in. Requests through the carrier may address the routes of a
session or of a calibration workspace and nothing else.

Without `LUMEN_STATE_SECRET`, somebody who writes a state by hand can make
the server compute on it, and can make an export say what they like. An
export was never proof of anything by itself (it is a file on the
operator's device); with the secret set, at least the server will not work
from a state it did not write.

## If something does not work on the phone

Menu → **Diagnostics** → "Where the job is kept" lists the last requests: what
was asked, how much was sent, how long the answer took and how it ended. A
step that sends megabytes and takes long is the connection; one that sends
little and takes long is the server.

| What you see | Meaning |
|---|---|
| "This step would send … MB at once, more than this server takes" | too many or too large photos for one request: remove one and analyse again |
| "A photo of this job is no longer on this phone" | the browser's storage was cleared: take the photos again |
| "The job did not come with the request" | the page was open from before the server changed its way of keeping jobs: reload |
| readings come once a second or slower | the platform answers with copies of the program that do not know the job; nothing in the application can change that |

## Tested, and not

Tested (`tests/integration/test_carried_flow.py`, `test_carried_placement.py`,
`test_carried_hold.py`, `test_carried_calibration.py`, `tests/browser/test_carried.py`,
`tests/unit/test_carried.py`): the whole job against a server that is made
anew for every single request and refuses anything above 4.5 MB; the same
numbers as a server that keeps the job; the existing tests of live
placement, of the held registration and of lens calibration, unchanged, with
the job carried; in a browser the demo, phone-sized photos being reduced, a
reload in the middle of a job, live placement to the automatic record, and
nothing left in the browser after the session ends.

Tried once on Vercel as described above. Not run: a phone; a mobile network;
any other function platform.
