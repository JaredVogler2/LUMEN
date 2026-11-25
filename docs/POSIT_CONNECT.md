# Deploying LUMEN on Posit Connect

> **Status: NOT DEPLOYED.** No Posit Connect server was available while LUMEN
> was built. Everything below marked *simulated* was tested locally; everything
> marked *NOT RUN* still has to be done on your server. Do not describe LUMEN as
> deployed until the checklist in section 9 has been completed on the real
> system.

LUMEN is one FastAPI application that serves both its API and its web
interface from a single origin. It needs no GPU, no build step, no CDN, no
database and no other service.

## 1. What Connect needs

Facts taken from the Posit documentation (consulted 2026-11-25; check them
against the version of your server):

| Item | Value | Source |
|---|---|---|
| Content type | FastAPI (`python-fastapi` in the manifest) | <https://docs.posit.co/connect/user/fastapi/> |
| Entry point | `app:app` (module `app.py`, object `app`) | <https://docs.posit.co/connect/user/publishing-cli-apps/index.html#understanding-entrypoints> |
| Dependencies | `requirements.txt` in the bundle | <https://docs.posit.co/connect/user/fastapi/> |
| Deploy command | `rsconnect deploy fastapi` | <https://docs.posit.co/rsconnect-python/commands/deploy/> |
| Manifest command | `rsconnect write-manifest fastapi` | <https://docs.posit.co/rsconnect-python/commands/write-manifest/> |
| Base address header | `RStudio-Connect-App-Base-URL` | <https://docs.posit.co/connect/user/fastapi/> |
| User header | `RStudio-Connect-Credentials` (JSON: user, groups) | <https://docs.posit.co/connect/user/fastapi/> |
| Process settings | Min/Max processes, Max connections per process, Load factor, timeouts | <https://docs.posit.co/connect/admin/process-management/> |

Connect "may launch multiple processes" for one content item and does not
coordinate them; LUMEN is built for that (section 5).

Python: 3.11, 3.12 or 3.13 (`pyproject.toml`: `>=3.11,<3.14`). Dependencies
are pinned to exact versions in `requirements.txt`; all of them install as
binary wheels on x86-64 Linux. `opencv-python-headless` is used, so no display
libraries (`libGL`) are required. Clean installs and test runs on this
repository are recorded in `docs/BUILD_LOG.md`.

## 2. What goes into the bundle

Deploy: `app.py`, `requirements.txt`, and the whole `lumen/` package (it
contains the web interface under `lumen/static`, the sample operation library
and the simulated demo photos).

Leave out: `tests/`, `docs/`, `prompt_series/`, `scripts/`,
`requirements-dev.txt`, `pyproject.toml`, virtual environments and caches.

## 3. Publishing

Install the publishing tool on your workstation (it is not a runtime
dependency):

```bash
pip install rsconnect-python
```

### 3a. From the command line

```bash
rsconnect add --name myserver --server https://connect.example.org --api-key "$CONNECT_API_KEY"
```

```bash
rsconnect deploy fastapi --name myserver --entrypoint app:app --title "LUMEN (research prototype)" \
  -x "tests/**" -x "docs/**" -x "prompt_series/**" -x "scripts/**" \
  -x "requirements-dev.txt" -x "pyproject.toml" -x "README.md" .
```

The option names were checked against `rsconnect-python` 1.34.1
(`rsconnect deploy fastapi --help`). The deployment itself: **NOT RUN**.

If the server's Python differs in patch version from yours, add
`--override-python-version 3.12` (major.minor form needs Connect 2025.03.0 or
later, according to the tool's help text).

### 3b. From a manifest (git-backed content)

```bash
scripts/write_manifest.sh /path/to/python3.12
```

This writes `manifest.json` (content type `python-fastapi`, entry point
`app:app`, the Python version of the interpreter given, every deployed file
with a checksum). It was run on a copy of this repository and produced a valid
manifest of 94 files with the test and documentation folders excluded. The
manifest must be regenerated whenever a deployed file changes. Publishing from
it: **NOT RUN**.

## 4. Settings to make in Connect

**Access.** Restrict the content to named users or a group. LUMEN ties each
session to the Connect user name that created it (`RStudio-Connect-Credentials`):
another signed-in user who learns a session id gets "not found". Do not publish
the content as "Anyone - no login required".

**HTTPS.** Browsers only open the camera on a secure (HTTPS) page. Connect must
be reached over HTTPS from the phone.

**Runtime (per content item).** Starting points, to be adjusted from
observation:

| Setting | Suggested | Why |
|---|---|---|
| Max processes | 2–3 | each process handles up to `LUMEN_MAX_CONCURRENT_JOBS` (default 2) heavy computations at once |
| Min processes | 1 | avoids a cold start (about 1 s import, first health check 0.2 s) when an operator opens the page |
| Max connections per process | 20 (default) | live guidance holds one request at a time per phone |
| Load factor | 0.5 (default) | |
| Idle timeout | 10–30 minutes | with Min processes 0 and the default (5 s) a process may be stopped between two photos; that loses no data (sessions are on disk) but the next request waits for a new process |
| Connection / read timeout | default (1 h) | a 300 MB video upload over a slow link can take minutes |
| Initial timeout | default (120 s) | |

Measured for one process on this build machine (4 shared cores, busy; both
simulated demo sessions at 1600×1200):

| Quantity | Value |
|---|---|
| memory after import | 99 MB |
| peak memory, 5-photo session | 505 MB |
| peak memory, 7-photo session with the rounded-edge model | 656 MB |
| CPU time for a whole session (detect, register, datums, overlay, track, verify) | 32–34 s |

Allow about **1 GB of memory per process**. If your server enforces a memory
limit per process, set it no lower than 1.5 GB.

## 5. Session storage and several processes

A session is one JSON file plus its photos in a directory.

* Default: `FileSessionStore` under `$TMPDIR/lumen-sessions-<uid>` (or
  `LUMEN_DATA_DIR`). Writes are atomic and guarded by a file lock. All processes
  of the content item **on the same host** see the same directory, so any of
  them can serve any request. Tested by running two service instances on one
  directory (*simulated*, `tests/integration/test_deployment.py`).
* Background video jobs record their progress in the session file, so a status
  poll may be answered by a different process than the one doing the work. A
  job whose process died is reported as failed after 45 s instead of hanging.
* Decoded images and solver results are cached per process and rebuilt on
  demand; losing a process loses no data.

This storage is a **cache with a time limit, not durable storage**. Posit's
documentation advises against relying on the content's filesystem for
persistence, and so does LUMEN: sessions are deleted `LUMEN_SESSION_TTL_SECONDS`
after their last change (default 4 hours), when the operator ends them, or when
Connect replaces the directory. Export the session summary before then.

**More than one Connect node, or off-host (Kubernetes) execution: NOT RUN.**
The processes then do not share a local directory. Use one of:

1. set `LUMEN_DATA_DIR` to a directory on a volume shared by all nodes;
2. run a single process (Max processes = 1) on a single node;
3. write a storage adapter for a database or object store: implement the
   abstract methods of `lumen.storage.base.SessionStore` (create, load, save,
   delete, exists, put/get/delete blob, list ids, lock) and return it from
   `get_store()`.
   No such adapter ships with LUMEN.

`LUMEN_STORAGE=memory` keeps sessions inside one process and must not be used
with more than one process.

## 6. Environment variables

Set these in the content's **Vars** panel. None is required. Nothing secret
has a default, and no value is returned by the API.

| Variable | Default | Meaning |
|---|---|---|
| `LUMEN_DATA_DIR` | `$TMPDIR/lumen-sessions-<uid>` | session directory (see section 5) |
| `LUMEN_STORAGE` | `file` | `file` or `memory` |
| `LUMEN_SESSION_TTL_SECONDS` | `14400` | sessions and their photos are deleted this long after the last change |
| `LUMEN_MAX_SESSIONS` | `40` | new sessions are refused beyond this |
| `LUMEN_MAX_FRAMES` | `24` | photos per session |
| `LUMEN_MAX_IMAGE_MB` / `LUMEN_MAX_IMAGE_MEGAPIXELS` | `25` / `50` | upload limits for a photo |
| `LUMEN_MAX_ANALYSIS_LONG_EDGE` | `4096` | larger photos are reduced (area-averaged) before analysis |
| `LUMEN_MAX_VIDEO_MB` / `LUMEN_MAX_VIDEO_SECONDS` | `300` / `60` | upload limit; only this much of a video is analysed |
| `LUMEN_MAX_VIDEO_CANDIDATES` | `90` | candidate frames examined per video |
| `LUMEN_MAX_TRACK_LONG_EDGE` | `1600` | size of live-guidance frames |
| `LUMEN_MAX_CONCURRENT_JOBS` | `2` | heavy computations at once per process; others get "server busy" |
| `LUMEN_REQUEST_TIMEOUT_SECONDS` | `30` | wait for a free computation slot |
| `LUMEN_JOB_TIMEOUT_SECONDS` / `LUMEN_VIDEO_JOB_TIMEOUT_SECONDS` | `120` / `300` | budgets for analysis and for one video |
| `LUMEN_OPERATIONS_DIR` | empty | directory of your own operation records; empty = the bundled read-only sample |
| `LUMEN_DEMO_ENABLED` | `true` | offer the simulated demo session |
| `LUMEN_BASE_PATH` | empty | external path of the app, only for proxies that send no base header |
| `LUMEN_POLICY_FILE` | empty | JSON file overriding gate thresholds |
| `LUMEN_VALIDATED_DOMAIN_FILE` | empty | JSON description of a physically validated domain; none ships with LUMEN, so the status `validated` cannot occur |
| `LUMEN_PHOTOMETRIC_ENABLED` | `true` | experimental flashlight diagnostics (never guidance) |
| `LUMEN_LLM_ENABLED` | `false` | optional assistant; see `docs/ASSISTANT.md` before enabling |
| `LUMEN_LLM_BASE_URL`, `LUMEN_LLM_MODEL`, `LUMEN_LLM_API_KEY_ENV`, `LUMEN_LLM_TIMEOUT_SECONDS` | empty / `LUMEN_LLM_API_KEY` / `20` | assistant endpoint (HTTPS only), model, name of the variable holding the key |
| `LUMEN_LOG_LEVEL` | `INFO` | |

## 7. Address handling behind the proxy

Connect serves the content under `/content/<guid>/` and removes that prefix
before the request reaches the application.

* Every address the browser uses is relative. The server never writes an
  absolute path into the page, the scripts or the style sheet (checked by a
  test over all bundled files).
* When the proxy states the external address (`RStudio-Connect-App-Base-URL`,
  or `X-Forwarded-Prefix`, or `LUMEN_BASE_PATH`), it is written into
  `<base href>`.
* When it does not, a small script protected by the page's content security
  policy adds the missing trailing slash and only then requests the style
  sheet and the application script.

*Simulated* in a real browser behind a path-stripping proxy, with and without
the header and with and without the trailing slash: the demo session ran to
placement guidance in all four cases with no request outside the content path
(`tests/browser/test_subpath.py`). With a real Connect server: **NOT RUN**.

## 8. Security and privacy summary

* Single origin; content security policy `default-src 'self'`, scripts only
  from the app itself, no `eval`, no external requests, no telemetry.
* Uploads: type checked from the file's first bytes, size and pixel limits,
  bounded analysis size; videos are decoded once and deleted.
* Session ids are 144 random bits; session and file names are validated
  against fixed patterns before touching the filesystem.
* Photos exist only inside their session and go with it (time limit, operator's
  "End session", or deletion through the API). Live-guidance frames and scanned
  code pictures are never stored. The exported summary contains no image.
* The optional assistant is off; when off, no code path contacts any other
  host.
* Authentication and transport security are Connect's. LUMEN adds the
  per-user session binding described above.

## 9. Smoke tests

### 9a. Local

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
.venv/bin/python scripts/smoke_test.py http://127.0.0.1:8000/
```

Expected: eleven lines starting with `PASS` and `ALL STEPS PASSED`. The script
uses only simulated demo data and deletes its session.

The in-app camera needs HTTPS. From a phone on the same network, plain
`http://<computer>:8000` will show the interface but the browser will refuse
the camera; photo and video *upload* still work. To try the camera locally,
serve HTTPS (for example `uvicorn ... --ssl-keyfile key.pem --ssl-certfile
cert.pem` with a certificate the phone trusts).

### 9b. On Connect — **NOT RUN**

1. Publish (section 3) and apply the settings of section 4.
2. `python scripts/smoke_test.py https://<server>/content/<guid>/ --api-key "$CONNECT_API_KEY"`
   — all steps must pass. The `health` line reports the storage adapter and the
   OpenCV version actually installed on the server.
3. Open the content address in a desktop browser while signed in; run the demo
   session to the Verify step.
4. Open the same session address as a different user: it must not open.
5. With Max processes ≥ 2, repeat step 2 several times in parallel
   (`for i in 1 2 3 4; do python scripts/smoke_test.py ... & done; wait`): all
   must pass (sessions are shared through the directory, not through one process).
6. Restart the content; a session started before the restart must resume from
   the browser (it is on disk) until its time limit.

### 9c. On the managed iPhone — **NOT RUN**

Record the iOS version, the browser and the phone model with each result.

| # | Check | Expected |
|---|---|---|
| 1 | Open the Connect address in Safari, sign in | LUMEN's Identify screen; yellow research banner |
| 2 | Capture → Open camera | permission prompt; after "Allow" a live rear-camera picture, full frame visible |
| 3 | Camera options | stream size shown; if several rear cameras are listed, a fixed-lens one can be chosen |
| 4 | Light button | present only if Safari offers the torch; note whether it is, and whether photos record `on`, `off` or `unknown` |
| 5 | Take 5 photos walking around a panel with tags | each uploads; coaching line changes; count increases |
| 6 | Sweep video (if offered) | records, uploads, frames are chosen; note the container (MP4/WebM) from Diagnostics |
| 7 | Full-resolution photo (system camera) | HEIC or JPEG is accepted and upright |
| 8 | Use a video (recorded with the system camera, portrait and landscape) | frames upright; chosen frames show the tags |
| 9 | Lock the phone / switch apps while the camera is open | camera stops; "Resume camera" works |
| 10 | Rotate the phone | layout follows; a warning appears if the stream size changes mid-session |
| 11 | Analyse → Datums → Place → Live guidance | outline appears only while tags are in view; disappears when they leave |
| 12 | Verify with a new photo | deviations shown; "not an acceptance decision" visible |
| 13 | Menu → End session | photos deleted; reopening the address starts fresh |

## 10. Troubleshooting

| Symptom | Likely cause and remedy |
|---|---|
| Page loads without styling, or scripts 404 | a proxy in front of Connect rewrites paths without sending a base header: set `LUMEN_BASE_PATH` to the external path |
| "Camera access was refused" or no camera prompt | page not on HTTPS; or camera blocked for the site (Safari: "aA" → Website Settings); or an embedding page lacks camera permission |
| `server_busy` (503) under load | raise Max processes, or `LUMEN_MAX_CONCURRENT_JOBS` if the host has spare cores |
| Session "not found" after a while | time limit reached (`LUMEN_SESSION_TTL_SECONDS`), content redeployed, or a second node without a shared `LUMEN_DATA_DIR` |
| Video "could not be decoded" | codec not supported by the bundled decoder; record in the phone's "Most compatible" format or upload photos |
| Health check reports `degraded` | the response lists which self-check failed (tag detection, JPEG codec, storage, operation library) |
| Import error for `cv2` on the server | the server built OpenCV from source instead of using the wheel; make sure pip may use binary wheels |

## 11. Not run

* Any deployment to a Posit Connect server (publishing, manifest publishing,
  Vars, process settings, access control, user header, sub-path on the real proxy).
* Several Connect nodes; off-host execution.
* Any real iPhone (camera, torch, recording format, orientation, HEIC upload).
* Load beyond a single operator.
