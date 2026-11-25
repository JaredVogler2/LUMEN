"""Benchmark cases, runner and report.

Held-out configurations
-----------------------
The unit and integration tests use one tag layout (IDs 3, 7, 11, 20 at fixed
places) and camera arcs from ``sweep_stations``.  Thresholds in the code were
chosen while looking at those.  The benchmark draws its own layouts (other
IDs, positions, rotations), camera stations, lens distortion, blur, noise and
exposure from a seeded generator (seed 9131), so its configurations were not
seen when thresholds were set.

Modes compared on the same photos
---------------------------------
``single_manual``  one photo; datum lines drawn by a simulated operator (the
                   true lines with a few pixels of error) and snapped to the edge.
``multi_auto``     all photos; datums found automatically and checked in 3-D.

(The flashlight module has its own ablation: docs/PHOTOMETRIC_ABLATION.md.)

Metrics
-------
* corner error: largest distance between a drawn target corner and its true
  position, in millimetres on the panel;
* rotation error of the target outline;
* coverage: fraction of drawn targets whose corner error is within the
  reported expanded uncertainty;
* false acceptance: status ``prototype_guidance`` although the corner error
  exceeds both the expanded uncertainty and the record's tolerance;
* latency of each stage; failures (unexpected errors).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..errors import LumenError
from ..operations.library import OperationLibrary, parse_operation
from ..services.sessions import SessionService
from ..storage.base import MemorySessionStore
from ..synth import Box, CameraSpec, LightSpec, Scene, Seam, TagSticker, TrimSpec, project_points, render, scene_truth
from ..synth.fixtures import curved_panel_scene, flat_panel_scene, phone_camera

HELD_OUT_SEED = 9131
TAG_POOL = (41, 58, 102, 233, 317, 440, 501, 77)
#: PLC-DEMO-101 of the sample operation DEMO-FLAT-01, in scene coordinates (x, z): TL, TR, BR, BL
TARGET_101 = ((180.0, 330.0), (300.0, 330.0), (300.0, 270.0), (180.0, 270.0))
#: PLC-DEMO-201 of DEMO-CURVE-01
TARGET_201 = ((100.0, 345.0), (200.0, 345.0), (200.0, 295.0), (100.0, 295.0))
TOLERANCE_MM = 2.0


@dataclass
class Case:
    name: str
    kind: str  # "nominal" or the name of the fault
    scene: Scene
    cams: list[CameraSpec]
    operation: str = "DEMO-FLAT-01"
    revision: str | None = None
    placard: str = "PLC-DEMO-101"
    #: the operator also measures the suggested distance between two tag corners with a rule (the true distance)
    reference: bool = False
    target: tuple[tuple[float, float], ...] = TARGET_101
    calibrated: bool = True
    #: sizes the operator declares (id -> mm); the true size is in the scene
    declared: dict[int, float] = field(default_factory=dict)
    #: per-photo scene overrides (index -> scene), e.g. a tag moved after photo 3
    scene_at: dict[int, Scene] = field(default_factory=dict)
    #: what a correct system does: "accurate" (an outline within its stated uncertainty, as guidance or as a
    #: preview with a reason), "not_guidance" (blocked or preview), "accurate_or_refused", ...
    expect: str = "accurate"
    #: checks that have to be among the failed ones: the fault must be caught for its own reason, not because
    #: some unrelated check (say, the uncertainty at this photo size) happens to withhold guidance as well
    must_fail: tuple[str, ...] = ()
    #: IDs of the stickers the fault is about (the one that was moved, raised or mis-sized).  A result may only
    #: be drawn when these were set aside; the report also shows whether a sound tag was thrown out with them.
    faulty_tags: tuple[int, ...] = ()
    note: str = ""
    seed: int = 0


# --------------------------------------------------------------------------
# rendering with an on-disk cache
# --------------------------------------------------------------------------
class RenderCache:
    """Benchmark photos, rendered once and kept on disk.

    The photos use the renderer's ``"lattice"`` sampling.  With the regular sub-pixel grid, tag borders that
    happen to lie at 45 degrees or at a slope of 1:2 are rendered up to 0.2 px away from where they are (every
    pixel along such an edge is rounded the same way).  The measurement then reports, correctly, an edge that
    is not where the scene description says, and a tag looks moved.  That is a property of the picture
    generator, not of a camera, and must not be in a benchmark.
    """

    def __init__(self, directory: str | Path | None, ss: int = 3, pattern: str = "lattice") -> None:
        self.dir = Path(directory) if directory else None
        self.ss = ss
        self.pattern = pattern
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, scene: Scene, cam: CameraSpec, seed: int) -> np.ndarray:
        if self.dir is None:
            return render(scene, cam, ss=self.ss, seed=seed, pattern=self.pattern).image
        key = hashlib.sha256(repr((scene, cam, self.ss, self.pattern, seed, "bench-v2")).encode()).hexdigest()[:24]
        f = self.dir / f"{key}.png"
        if f.exists():
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        img = render(scene, cam, ss=self.ss, seed=seed, pattern=self.pattern).image
        tmp = self.dir / f"{key}.{time.time_ns()}.tmp.png"
        cv2.imwrite(str(tmp), img)
        tmp.replace(f)
        return img


def _jpeg(img: np.ndarray, q: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return buf.tobytes()


def _profile(cam: CameraSpec) -> dict[str, Any]:
    return {
        "id": "bench-truth", "label": "synthetic truth", "method": "synthetic_truth", "provenance": "simulated",
        "width": cam.width, "height": cam.height, "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "dist": list(cam.dist),
        "param_sigma": [0.5, 0.5, 0.5, 0.5, 0.0005, 0.002, 0.0001, 0.0001, 0.0],
    }


def _px(cam: CameraSpec, x: float, z: float) -> np.ndarray:
    uv, _ = project_points(cam, np.array([[x, 0.0, z]]))
    return uv[0]


# --------------------------------------------------------------------------
# case generation
# --------------------------------------------------------------------------
def _layout(rng: np.random.Generator, n: int, keep_out: tuple[tuple[float, float], ...], x_range=(45.0, 560.0), z_range=(70.0, 560.0)) -> tuple[TagSticker, ...]:
    """``n`` non-overlapping 50 mm tags around (not on) the target region."""
    ko = np.array(keep_out)
    lo, hi = ko.min(axis=0) - 45.0, ko.max(axis=0) + 45.0
    ids = list(rng.choice(TAG_POOL, size=n, replace=False))
    out: list[TagSticker] = []
    tries = 0
    while len(out) < n and tries < 4000:
        tries += 1
        c = np.array([rng.uniform(*x_range), rng.uniform(*z_range)])
        if lo[0] < c[0] < hi[0] and lo[1] < c[1] < hi[1]:
            continue
        if any(np.linalg.norm(c - np.array(t.center)) < 95.0 for t in out):
            continue
        out.append(TagSticker(int(ids[len(out)]), (float(c[0]), float(c[1])), float(rng.uniform(0, 360))))
    if len(out) < n:
        raise RuntimeError("could not place the tags")
    # spread check: the tags must surround the target reasonably (not all on one side)
    return tuple(out)


def _stations(rng: np.random.Generator, n: int, target: tuple[float, float, float], size: tuple[int, int], dist_k: tuple[float, ...], blur: float, noise: tuple[float, float], exposure_jitter: float) -> list[CameraSpec]:
    distance = rng.uniform(560.0, 860.0)
    height = rng.uniform(330.0, 600.0)
    span = np.deg2rad(rng.uniform(26.0, 58.0)) if n > 1 else 0.0
    centre = np.deg2rad(rng.uniform(-12.0, 12.0))
    cams = []
    for k in range(n):
        a = centre + (span * (k / (n - 1) - 0.5) if n > 1 else 0.0)
        eye = (target[0] + distance * np.sin(a) + rng.uniform(-25, 25), -distance * np.cos(a) + rng.uniform(-30, 30), height + rng.uniform(-70, 70))
        cams.append(
            phone_camera(
                tuple(float(v) for v in eye), target=target, width=size[0], height=size[1], roll_deg=float(rng.uniform(-11, 11)),
                dist=dist_k, blur_sigma=blur, noise_shot=noise[0], noise_read=noise[1], exposure=float(1.0 + rng.uniform(-exposure_jitter, exposure_jitter)),
            )
        )
    return cams


#: In-camera processing of the nominal cases, by case number (no random draw is spent on it).  Phones sharpen and
#: tone-map their pictures; both move edges, which is what the edge-shift correction of the tag measurement is for.
ISP_VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("plain", {}),
    ("sharpened 0.6", {"sharpen": 0.6}),
    ("tone curve", {"tone_gamma": 1.15, "tone_contrast": 0.3}),
    ("sharpened 0.9 + contrast, JPEG 85", {"sharpen": 0.9, "tone_contrast": 0.4, "jpeg_quality": 85}),
)


def nominal_cases(n: int = 12, seed: int = HELD_OUT_SEED, size: tuple[int, int] = (1600, 1200)) -> list[Case]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_tags = int(rng.choice([3, 4, 4, 5]))
        tags = _layout(rng, n_tags, TARGET_101)
        n_views = int(rng.choice([3, 5, 5, 7]))
        k1 = float(rng.uniform(0.0, 0.11))
        dist_k = (k1, float(-rng.uniform(0.0, 2.0) * k1), 0.0, 0.0, 0.0)
        blur = float(rng.uniform(0.75, 1.35))
        noise = [(0.0006, 0.004), (0.0015, 0.006)][int(rng.integers(0, 2))]
        scene = flat_panel_scene(tags=tags, light=LightSpec(ambient=float(rng.uniform(0.66, 0.86))))
        cams = _stations(rng, n_views, (float(rng.uniform(230, 300)), 0.0, float(rng.uniform(270, 330))), size, dist_k, blur, noise, 0.15)
        isp_name, isp = ISP_VARIANTS[i % len(ISP_VARIANTS)]
        cams = [replace(c, **isp) for c in cams]
        out.append(Case(f"nominal-{i + 1:02d}", "nominal", scene, cams, declared={t.tag_id: 50.0 for t in tags}, seed=1000 + i,
                        note=f"{n_tags} tags, {n_views} photos, k1 {k1:.3f}, blur {blur:.2f} px, {isp_name}"))
    return out


def negative_cases(seed: int = HELD_OUT_SEED + 1, size: tuple[int, int] = (1600, 1200)) -> list[Case]:
    """Faults a correct system must refuse, flag, or survive without a wrong confident answer."""
    rng = np.random.default_rng(seed)
    out: list[Case] = []

    def base(n_tags: int = 4, n_views: int = 5) -> tuple[tuple[TagSticker, ...], list[CameraSpec]]:
        tags = _layout(rng, n_tags, TARGET_101)
        cams = _stations(rng, n_views, (265.0, 0.0, 300.0), size, (0.05, -0.08, 0.0, 0.0, 0.0), 0.9, (0.0006, 0.004), 0.1)
        return tags, cams

    tags, cams = base()
    out.append(Case("no-scale", "no_tags", flat_panel_scene(tags=()), cams, declared={}, expect="not_guidance", seed=2001, note="no tag on the panel: nothing gives scale"))

    tags, cams = base()
    sc = flat_panel_scene(tags=tags)
    moved = flat_panel_scene(tags=tuple(replace(t, center=(t.center[0] + 8.0, t.center[1] - 3.0)) if k == 1 else t for k, t in enumerate(tags)))
    out.append(Case("moved-tag", "moved_tag", sc, cams, declared={t.tag_id: 50.0 for t in tags}, scene_at={3: moved, 4: moved}, expect="accurate_or_refused", faulty_tags=(tags[1].tag_id,), seed=2002, note="one tag re-stuck 8.5 mm away before the last two photos"))

    tags, cams = base()
    out.append(Case("one-size-wrong", "wrong_size_one", flat_panel_scene(tags=tags), cams, declared={t.tag_id: (52.0 if k == 0 else 50.0) for k, t in enumerate(tags)}, expect="accurate_or_refused", faulty_tags=(tags[0].tag_id,), seed=2003, note="one tag declared 52 mm (true 50)"))

    tags, cams = base()
    out.append(Case("all-sizes-wrong", "wrong_size_all", flat_panel_scene(tags=tags), cams, declared={t.tag_id: 52.0 for t in tags}, expect="undetectable_without_reference", seed=2004, note="every tag declared 52 mm (true 50): a common scale error that no tag-to-tag check can see"))

    tags, cams = base()
    out.append(Case("undeclared-tags", "nominal_sizes", flat_panel_scene(tags=tags), cams, declared={}, expect="not_guidance", must_fail=("tag_sizes_measured",), seed=2005, note="no measured sizes entered: nominal 50 mm only"))

    tags, cams = base()
    sym = flat_panel_scene(tags=tags, extra_seams=(Seam((-48.0, 0.0), (-48.0, 1100.0), width=2.0, darkness=0.7),))
    out.append(Case("twin-seams", "ambiguous_side", sym, cams, declared={t.tag_id: 50.0 for t in tags}, expect="not_guidance", must_fail=("datum_side_unambiguous",), seed=2006, note="a second identical seam 48 mm beside the datum seam"))

    tags = _layout(rng, 4, TARGET_101, z_range=(330.0, 700.0))
    cams_hi = _stations(rng, 5, (265.0, 0.0, 560.0), size, (0.05, -0.08, 0.0, 0.0, 0.0), 0.9, (0.0006, 0.004), 0.1)
    cams_hi = [phone_camera(tuple(c.C[:2]) + (640.0,), target=(265.0, 0.0, 600.0), width=size[0], height=size[1], hfov_deg=46.0, dist=c.dist) for c in cams_hi]
    out.append(Case("floor-not-in-view", "missing_origin", flat_panel_scene(tags=tags), cams_hi, declared={t.tag_id: 50.0 for t in tags}, expect="not_guidance", seed=2007, note="no photo shows the floor line"))

    tags_c = _layout(rng, 4, TARGET_201, x_range=(60.0, 520.0))
    cams_c = _stations(rng, 7, (230.0, 0.0, 300.0), size, (0.0, 0.0, 0.0, 0.0, 0.0), 0.9, (0.0006, 0.004), 0.05)
    out.append(Case("leaning-edge", "nonorthogonal_axes", curved_panel_scene(taper_deg=5.0, tags=tags_c), cams_c, operation="DEMO-CURVE-01", placard="PLC-DEMO-201", target=TARGET_201,
                    declared={t.tag_id: 50.0 for t in tags_c}, expect="not_guidance", must_fail=("local_frame_valid",), seed=2008, note="side edge leans 5 degrees: axes are not perpendicular"))

    tags, cams = base()
    lifted = flat_panel_scene(tags=tuple(replace(t, lift=8.0) if k == 2 else t for k, t in enumerate(tags)))
    out.append(Case("raised-tag", "off_plane_tag", lifted, cams, declared={t.tag_id: 50.0 for t in tags}, expect="accurate_or_refused", faulty_tags=(tags[2].tag_id,), seed=2009, note="one tag sits on an 8 mm shim"))

    tags, cams = base()
    faint = flat_panel_scene(tags=tags)
    faint = replace(faint, seams=(Seam((0.0, 0.0), (0.0, 1100.0), width=1.2, darkness=0.06),))
    out.append(Case("faint-seam", "low_contrast", faint, cams, declared={t.tag_id: 50.0 for t in tags}, expect="accurate_or_refused", seed=2010, note="side seam is barely visible (6 % darker, 1.2 mm wide)"))

    tags, cams = base()
    trim = replace(flat_panel_scene(tags=tags), trim=TrimSpec(height=60.0, thickness=8.0))
    out.append(Case("proud-trim", "off_plane_floor_edge", trim, cams, declared={t.tag_id: 50.0 for t in tags}, expect="not_guidance", must_fail=("datum_floor_no_step_beside",), seed=2011,
                    note="an 8 mm proud trim hides the floor line; the trim's upper edge lies on the panel face and looks like a floor line"))

    tags, cams = base()
    dup = flat_panel_scene(tags=tuple(tags) + (TagSticker(tags[0].tag_id, (610.0, 620.0), 30.0),))
    out.append(Case("duplicate-id", "duplicate_id", dup, cams, declared={t.tag_id: 50.0 for t in tags}, expect="accurate_or_refused", seed=2012, note="two stickers carry the same ID"))

    tags, cams = base()
    glare = flat_panel_scene(tags=tuple(replace(t, spec_strength=2.5, shininess=30.0) for t in tags), light=LightSpec(ambient=0.55, flash=True, flash_strength=1.6))
    out.append(Case("flash-glare", "glare", glare, cams, declared={t.tag_id: 50.0 for t in tags}, expect="accurate_or_refused", seed=2013, note="glossy tags lit by a strong flash"))

    tags, cams = base()
    out.append(Case("all-sizes-wrong-ruler", "wrong_size_all_with_reference", flat_panel_scene(tags=tags), cams, declared={t.tag_id: 52.0 for t in tags}, reference=True, expect="not_guidance", must_fail=("independent_length_check",), seed=2014,
                    note="every tag declared 52 mm (true 50), and the operator measures one distance between two tags with a rule"))

    tags, cams = base()
    out.append(Case("true-sizes-ruler", "reference_agrees", flat_panel_scene(tags=tags), cams, declared={t.tag_id: 50.0 for t in tags}, reference=True, expect="accurate", seed=2015,
                    note="correct sizes and a rule measurement that agrees (control for the case above)"))

    tags, cams = base()
    out.append(Case("curved-target", "off_plane_target", flat_panel_scene(tags=tags), cams, operation=CURVED_TARGET_ID, revision="A", declared={t.tag_id: 50.0 for t in tags}, expect="not_guidance", must_fail=("surface_supported",), seed=2016,
                    note="the record says the placard sits on a curved surface: outside the planar method"))

    tags, cams = base()
    out.append(Case("draft-record", "unreleased_record", flat_panel_scene(tags=tags), cams, operation=DRAFT_ID, revision="A", declared={t.tag_id: 50.0 for t in tags}, expect="not_guidance", must_fail=("operation_record_released",), seed=2017,
                    note="complete record that was never released"))
    return out


CURVED_TARGET_ID = "BENCH-CURVED-TARGET"
DRAFT_ID = "BENCH-DRAFT"


def benchmark_library(directory: str | Path) -> OperationLibrary:
    """The bundled sample records plus the records two fault cases need, in a scratch directory."""
    lib = OperationLibrary(Path(__file__).resolve().parent.parent / "operations" / "data" / "operations.json", Path(directory))
    base = lib.get("DEMO-FLAT-01").model_dump(mode="json")
    for op_id, patch in ((CURVED_TARGET_ID, {"surface": {"target": "curved", "notes": "benchmark fault record"}}), (DRAFT_ID, {})):
        if not lib.revisions(op_id):
            rec = json.loads(json.dumps(base))
            rec.update(operation_id=op_id, revision="A", status="draft", synthetic=False, title=f"Benchmark fault record ({op_id})", **patch)
            rec["panel"]["barcodes"] = []
            for pl in rec["placards"]:
                pl["barcodes"] = []
            lib.save(parse_operation(rec), author="benchmark")
    return lib


# --------------------------------------------------------------------------
# running a case
# --------------------------------------------------------------------------
def _service(library: OperationLibrary) -> SessionService:
    from ..config import Settings

    return SessionService(MemorySessionStore(), Settings(storage="memory"), library)


def run_case(case: Case, mode: str, library: OperationLibrary, shoot: Callable[[Scene, CameraSpec, int], np.ndarray], operator_px: float = 2.5,
             profile_error: np.ndarray | None = None) -> dict[str, Any]:
    """Run one case in one mode; returns a flat record (never raises for expected refusals).

    ``profile_error`` (fx, fy, cx, cy, k1, k2) is added to the attached lens
    profile, so that the profile differs from the camera that took the photos.
    """
    rec: dict[str, Any] = {"case": case.name, "kind": case.kind, "mode": mode, "expect": case.expect, "must_fail": list(case.must_fail), "faulty_tags": list(case.faulty_tags), "status": None, "error": None, "note": case.note}
    svc = _service(library)
    t = {"detect": 0.0, "register": 0.0, "datums": 0.0, "overlay": 0.0}
    try:
        sid = svc.create().id
        svc.set_operation(sid, case.operation, case.revision, case.placard)
        svc.set_tag_sizes(sid, {"nominal_mm": 50.0, "measured_mm": {str(k): v for k, v in case.declared.items()}})
        cams = case.cams if mode == "multi_auto" else case.cams[len(case.cams) // 2 : len(case.cams) // 2 + 1]
        idx = list(range(len(case.cams))) if mode == "multi_auto" else [len(case.cams) // 2]
        if case.calibrated:
            from ..schemas import Calibration

            prof = _profile(cams[0])
            if profile_error is not None:
                d = [float(x) for x in profile_error]
                prof["fx"] += d[0]; prof["fy"] += d[1]; prof["cx"] += d[2]; prof["cy"] += d[3]  # noqa: E702
                prof["dist"] = [prof["dist"][0] + d[4], prof["dist"][1] + d[5], *prof["dist"][2:]]
            svc.set_calibration(sid, Calibration(**prof))
        t0 = time.perf_counter()
        fids = []
        for k, cam in zip(idx, cams):
            f = svc.add_frame(sid, _jpeg(shoot(case.scene_at.get(k, case.scene), cam, case.seed + k)))
            fids.append(f.id)
        t["detect"] = time.perf_counter() - t0
        rec["n_photos"] = len(fids)
        t0 = time.perf_counter()
        reg = svc.register(sid)
        t["register"] = time.perf_counter() - t0
        rec["n_tags"] = reg["n_tags"]
        rec["excluded_tags"] = sorted(int(k) for k in (reg.get("excluded_tags") or {}))
        ref = reg["reference_frame_id"]
        cam_ref = cams[fids.index(ref)]
        if case.reference:
            sug = svc.reference_suggestion(sid)
            if sug is not None:
                truth_tags = {t["id"]: np.array(t["corners_world"])[:, [0, 2]] for t in scene_truth(case.scene, cam_ref)["tags"]}
                d_true = float(np.linalg.norm(truth_tags[sug["a"]["tag_id"]][sug["a"]["corner"]] - truth_tags[sug["b"]["tag_id"]][sug["b"]["corner"]]))
                r0 = svc.add_reference_distance(sid, sug["a"], sug["b"], d_true, 0.5, "steel rule (simulated: the true distance)")
                rec["reference"] = {"consistent": r0.get("consistent"), "relative_difference": r0.get("relative_difference")}
        t0 = time.perf_counter()
        if mode == "multi_auto":
            prop = svc.propose_datums(sid)
            rec["datums_found"] = sorted(prop["applied"])
            rec["ambiguous"] = bool(prop.get("ambiguous_side") or prop.get("ambiguous_floor"))
        else:
            rng = np.random.default_rng(case.seed + 77)
            lines = {"floor": ((60.0, 0.0), (560.0, 0.0)), "side": ((0.0, 80.0), (0.0, 600.0))}
            found = []
            for kind, (a, b) in lines.items():
                p0 = _px(cam_ref, *a) + rng.normal(0, operator_px, 2)
                p1 = _px(cam_ref, *b) + rng.normal(0, operator_px, 2)
                W, H = cam_ref.width, cam_ref.height
                if not (0 <= p0[0] < W and 0 <= p0[1] < H and 0 <= p1[0] < W and 0 <= p1[1] < H):
                    # the operator can only draw what the photo shows: clip the true line to the picture
                    ts = np.linspace(0, 1, 60)
                    pts = np.array([_px(cam_ref, a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])) for s in ts])
                    ok = (pts[:, 0] > 8) & (pts[:, 0] < W - 9) & (pts[:, 1] > 8) & (pts[:, 1] < H - 9)
                    if ok.sum() < 8:
                        continue
                    p0, p1 = pts[ok][0] + rng.normal(0, operator_px, 2), pts[ok][-1] + rng.normal(0, operator_px, 2)
                try:
                    svc.propose_datum(sid, kind, ref, p0.tolist(), p1.tolist(), snap=True)
                    found.append(kind)
                except LumenError as exc:
                    rec.setdefault("datum_errors", []).append(f"{kind}:{exc.code}")
            rec["datums_found"] = found
        try:
            svc.confirm_datums(sid)
        except LumenError:
            pass
        t["datums"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ov = svc.overlay(sid, ref)
        t["overlay"] = time.perf_counter() - t0
        rep = ov["reports"].get(case.placard, {})
        rec["status"] = rep.get("status")
        rec["failed_gates"] = [g["code"] for g in rep.get("gates", []) if not g["passed"]]
        unc = rep.get("uncertainty")
        rec["expanded_mm"] = None if not unc else float(unc["expanded_mm"])
        # the three largest terms of the budget, so a report can say what limits the result
        rec["budget_top"] = [] if not unc else [
            {"name": c["name"], "mm": round(float(c["value_mm"]), 3)} for c in sorted(unc.get("components", []), key=lambda c: -float(c["value_mm"]))[:3]
        ]
        tgt = next((p for p in ov["primitives"] if p["kind"].startswith("target") and p.get("corners") and p.get("active") is not False), None)
        if tgt is not None:
            truth = np.array([_px(cam_ref, x, z) for x, z in case.target])
            got = np.array(tgt["corners"])
            e = np.linalg.norm(got - truth, axis=1) / float(ov["px_per_mm"])
            rec["corner_error_mm"] = float(e.max())
            rec["centre_error_mm"] = float(np.linalg.norm(got.mean(axis=0) - truth.mean(axis=0)) / float(ov["px_per_mm"]))
            a_got = np.arctan2(*(got[1] - got[0])[::-1])
            a_tru = np.arctan2(*(truth[1] - truth[0])[::-1])
            rec["rotation_error_deg"] = float(abs(np.degrees((a_got - a_tru + np.pi) % (2 * np.pi) - np.pi)))
            if rec["expanded_mm"] is not None:
                rec["covered"] = bool(rec["corner_error_mm"] <= rec["expanded_mm"])
        rec["false_acceptance"] = bool(
            rec["status"] == "prototype_guidance" and rec.get("corner_error_mm") is not None
            and rec["corner_error_mm"] > max(rec.get("expanded_mm") or 0.0, TOLERANCE_MM)
        )
    except LumenError as exc:
        rec["status"] = "refused"
        rec["error"] = exc.code
        rec["false_acceptance"] = False
    except Exception as exc:  # an unexpected failure is a finding, not a crash of the benchmark
        rec["status"] = "failure"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["false_acceptance"] = False
    rec["seconds"] = {k: round(v, 2) for k, v in t.items()}
    rec["outcome_ok"] = judge(rec)
    return rec


def judge(rec: dict[str, Any]) -> bool:
    """Did the system do what a correct one should for this case?"""
    st, exp = rec["status"], rec["expect"]
    err, U = rec.get("corner_error_mm"), rec.get("expanded_mm")
    drawn_ok = err is not None and err <= max(U or 0.0, TOLERANCE_MM)
    if st == "failure":
        return False
    if exp == "accurate":
        # the capture is sound: an outline must be drawn and be right.  Whether it is shown as guidance or as a
        # preview depends on the uncertainty against the tolerance and is reported separately.
        return st in ("prototype_guidance", "validated", "research_preview") and drawn_ok
    if exp == "not_guidance":
        # a refusal has no list of checks; otherwise the check that belongs to this fault has to be among the failed ones
        named = st == "refused" or set(rec.get("must_fail") or ()) <= set(rec.get("failed_gates") or ())
        return st in ("blocked", "research_preview", "refused") and named
    if exp == "accurate_or_refused":
        if st in ("blocked", "refused"):
            return True  # refusing is always safe; whether the right tag was identified is shown beside it
        # a result that is drawn must not rest on the tag the fault is about
        aside = set(rec.get("faulty_tags") or ()) <= set(rec.get("excluded_tags") or ())
        return aside and ((err is not None and drawn_ok) or (st == "research_preview" and err is None))
    if exp == "undetectable_without_reference":
        return True  # recorded, not judged: see the report's note
    return False


# --------------------------------------------------------------------------
# seeded faults inside the budget's stated assumptions: coverage
# --------------------------------------------------------------------------
def coverage_trials(case: Case, library: OperationLibrary, shoot: Callable[[Scene, CameraSpec, int], np.ndarray], n: int = 40, seed: int = 4242, mode: str = "single_manual") -> list[dict[str, Any]]:
    """Repeat one case with errors drawn from what the budget says it covers.

    Per trial: every declared tag size is off by N(0, 0.05 mm) (the stated
    caliper uncertainty) plus a common offset within +-0.03 mm; the lens
    profile is off by its own stated sigmas; a fresh noise realisation of the
    photo; a fresh operator hand (2.5 px) for drawn lines.  The reported
    expanded uncertainty should then contain the true error in (nearly) all
    trials.  Panel flatness and edge-definition bounds are *not* exercised
    (the synthetic panel is flat), so the budget is expected to be conservative.
    """
    rng = np.random.default_rng(seed)
    out = []
    sig = np.array([0.5, 0.5, 0.5, 0.5, 0.0005, 0.002])
    for i in range(n):
        common = rng.uniform(-0.03, 0.03)
        declared = {k: float(50.0 + common + rng.normal(0, 0.05)) for k in case.declared}
        d = rng.normal(0, 1, 6) * sig

        c2 = replace(case, declared=declared, seed=case.seed + 100 * (i + 1), name=f"{case.name}#t{i + 1}")
        rec = run_case(c2, mode, library, shoot, operator_px=2.5, profile_error=d if case.calibrated else None)
        rec["trial"] = i + 1
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _stats(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in records}):
        rs = [r for r in records if r["mode"] == mode and r["kind"] == "nominal"]
        if not rs:
            continue
        drawn = [r for r in rs if r.get("corner_error_mm") is not None]
        cov = [r for r in drawn if "covered" in r]
        out[mode] = {
            "cases": len(rs),
            "status": {s: sum(r["status"] == s for r in rs) for s in sorted({str(r["status"]) for r in rs})},
            "corner_error_mm": _stats([r["corner_error_mm"] for r in drawn]),
            "centre_error_mm": _stats([r["centre_error_mm"] for r in drawn]),
            "rotation_error_deg": _stats([r["rotation_error_deg"] for r in drawn]),
            "expanded_mm": _stats([r["expanded_mm"] for r in cov]),
            "coverage": None if not cov else {"covered": sum(r["covered"] for r in cov), "of": len(cov)},
            "error_over_expanded": _stats([r["corner_error_mm"] / r["expanded_mm"] for r in cov if r["expanded_mm"]]),
            "false_acceptance": sum(r["false_acceptance"] for r in rs),
            "failures": sum(r["status"] == "failure" for r in rs),
            "seconds": {k: float(np.median([r["seconds"][k] for r in rs])) for k in ("detect", "register", "datums", "overlay")},
        }
    neg = [r for r in records if r["kind"] != "nominal"]
    out["negative"] = {
        "cases": len(neg),
        "handled_correctly": sum(r["outcome_ok"] for r in neg),
        "false_acceptance": sum(r["false_acceptance"] for r in neg),
        "failures": sum(r["status"] == "failure" for r in neg),
    }
    return out


def to_markdown(records: list[dict[str, Any]], summary: dict[str, Any], coverage: list[dict[str, Any]] | None, meta: dict[str, Any]) -> str:
    f2 = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    L = ["# Synthetic benchmark results", "", "Generated by `scripts/run_benchmark.py`. **Simulated data only**: these numbers describe the software on", "computer-generated photos. They are not physical accuracy and validate nothing about a real phone or panel.", ""]
    if meta.get("note"):
        L += [str(meta["note"]), ""]
    L += [f"* run: {meta.get('when')}  ·  algorithm {meta.get('algorithm')}  ·  seed {meta.get('seed')}  ·  photos {meta.get('size')}  ·  {meta.get('n_nominal')} held-out nominal cases, {meta.get('n_negative')} fault cases", f"* machine: {meta.get('machine')}", ""]
    L += ["## Nominal cases (calibrated lens, measured tags)", "", "| | single photo, hand-drawn datums | all photos, automatic datums |", "|---|---|---|"]
    a, b = summary.get("single_manual"), summary.get("multi_auto")

    def row(label: str, fn: Callable[[dict[str, Any]], str]) -> str:
        return f"| {label} | {fn(a) if a else '—'} | {fn(b) if b else '—'} |"

    st = lambda s: ", ".join(f"{k} {v}" for k, v in s["status"].items())  # noqa: E731
    q = lambda s, key: "—" if not s.get(key) else f"{s[key]['median']:.2f} / {s[key]['p95']:.2f} / {s[key]['max']:.2f}"  # noqa: E731
    L += [
        row("cases", lambda s: str(s["cases"])),
        row("status", st),
        row("corner error, mm (median / 95th / max)", lambda s: q(s, "corner_error_mm")),
        row("centre error, mm (median / 95th / max)", lambda s: q(s, "centre_error_mm")),
        row("rotation error, degrees (median / 95th / max)", lambda s: q(s, "rotation_error_deg")),
        row("expanded uncertainty, mm (median / 95th / max)", lambda s: q(s, "expanded_mm")),
        row("error within expanded uncertainty", lambda s: "—" if not s.get("coverage") else f"{s['coverage']['covered']} of {s['coverage']['of']}"),
        row("error ÷ expanded uncertainty (median / 95th / max)", lambda s: q(s, "error_over_expanded")),
        row("false acceptance", lambda s: str(s["false_acceptance"])),
        row("unexpected failures", lambda s: str(s["failures"])),
        row("time: tags / register / datums / overlay, s (median)", lambda s: " / ".join(f"{s['seconds'][k]:.1f}" for k in ("detect", "register", "datums", "overlay"))),
        "",
    ]
    L += ["### Each nominal case", "", "| case | configuration | mode | status | corner error mm | expanded mm | rotation ° | largest term of the budget |", "|---|---|---|---|---|---|---|---|"]
    for r in records:
        if r["kind"] == "nominal":
            top = (r.get("budget_top") or [None])[0]
            L.append(f"| {r['case']} | {r['note']} | {r['mode']} | {r['status']}{'' if not r.get('error') else ' (' + r['error'] + ')'} | {f2(r.get('corner_error_mm'))} | {f2(r.get('expanded_mm'))} | {f2(r.get('rotation_error_deg'))} | {'—' if not top else top['name'] + ' ' + format(top['mm'], '.2f') + ' mm'} |")
    faults = [r for r in records if r["kind"] != "nominal"]
    if faults:
        L += ["", "## Fault cases", "", "A fault case is handled correctly when the system refuses or withholds guidance, or — where the fault can be",
              "isolated (a moved, raised or mis-sized single tag) — excludes it and still gives an accurate result. Where a",
              "check is printed in bold, the case only counts as handled if that check is the one that fails: at this photo",
              "size the uncertainty alone often withholds guidance, and that would say nothing about the fault. Errors are",
              "measured against the target as the scene's true floor and side lines define it.", "",
              "\"Set aside\" lists the tags the solver left out; where the fault is about one tag, it is marked ✓ when exactly",
              "that tag was left out and ✗ when the faulty tag was kept or a sound one was thrown out with it.", "",
              "| case | fault | mode | status | set aside | failed checks | corner error mm | expanded mm | handled |", "|---|---|---|---|---|---|---|---|---|"]
        for r in faults:
            aside = ", ".join(str(t) for t in (r.get("excluded_tags") or []))
            if r.get("faulty_tags"):
                aside = (aside or "none") + (" ✓" if sorted(r.get("excluded_tags") or []) == sorted(r["faulty_tags"]) else " ✗")
            need = list(r.get("must_fail") or [])
            got = r.get("failed_gates") or []
            shown = [f"**{g}**" for g in need if g in got] + [g for g in got if g not in need][: max(0, 4 - len(need))]
            gates = ", ".join(shown) or (r.get("error") or "")
            handled = "recorded only" if r["expect"] in ("undetectable_without_reference",) else ("yes" if r["outcome_ok"] else "**NO**")
            L.append(f"| {r['case']} | {r['note']} | {r['mode']} | {r['status']} | {aside} | {gates} | {f2(r.get('corner_error_mm'))} | {f2(r.get('expanded_mm'))} | {handled} |")
        n = summary["negative"]
        known = [r["case"] for r in faults if r.get("false_acceptance") and r["expect"] == "undetectable_without_reference"]
        unexpected = n["false_acceptance"] - len(known)
        L += ["", f"Handled as expected: {n['handled_correctly']} of {n['cases']} runs. Wrong results shown as guidance: {n['false_acceptance']}"
              + (f" ({', '.join(known)}: the fault that no check on the photos can see; {unexpected} other)" if known else "")
              + f". Unexpected failures: {n['failures']}.", ""]
    if coverage:
        drawn = [r for r in coverage if r.get("corner_error_mm") is not None and r.get("expanded_mm")]
        ratio = [r["corner_error_mm"] / r["expanded_mm"] for r in drawn]
        L += ["## Coverage with seeded errors", "",
              "The same scene repeated with errors drawn from what the uncertainty budget says it covers: tag sizes off by",
              "N(0, 0.05 mm) each plus a common ±0.03 mm, lens profile off by its stated standard deviations, new image noise,",
              "and (single-photo mode) a new 2.5 px operator hand. Panel flatness and edge-definition bounds are not exercised.", ""]
        for mode in sorted({r["mode"] for r in coverage}):
            d = [r for r in drawn if r["mode"] == mode]
            if not d:
                continue
            rr = np.array([r["corner_error_mm"] / r["expanded_mm"] for r in d])
            ee = np.array([r["corner_error_mm"] for r in d])
            L.append(f"* **{mode}**: {len(d)} trials with a drawn target; error within the expanded uncertainty in {int((rr <= 1).sum())} of {len(d)}; "
                     f"corner error median {np.median(ee):.2f} mm, max {ee.max():.2f} mm; expanded uncertainty median {np.median([r['expanded_mm'] for r in d]):.2f} mm; "
                     f"error ÷ expanded: median {np.median(rr):.2f}, 95th {np.percentile(rr, 95):.2f}, max {rr.max():.2f}.")
        L += ["", f"Trials without a drawn target (refused or blocked): {len(coverage) - len(drawn)} of {len(coverage)}.", ""]
        _ = ratio
    L += ["## Reading these numbers", "",
          "* The expanded uncertainty is dominated by bounds for effects the synthetic photos do not contain (panel",
          "  flatness, residual edge bias, lens distortion beyond the model). On synthetic data the true error is therefore",
          "  much smaller than the reported uncertainty; that is expected and is not evidence of the same margin on a real panel.",
          "* `all-sizes-wrong` shows a fault no internal check can see: every tag declared 4 % too large scales the whole",
          "  result by 4 %, with a clean report. `all-sizes-wrong-ruler` is the same fault with one rule measurement",
          "  between two tag corners: the contradiction blocks the result. The rule measurement is optional in the interface.",
          "* Latency was measured on a shared build machine and varies with its load.", ""]
    return "\n".join(L)


def run_benchmark(library: OperationLibrary, cache_dir: str | Path | None, n_nominal: int = 12, size: tuple[int, int] = (1600, 1200), modes: tuple[str, ...] = ("single_manual", "multi_auto"),
                  with_negative: bool = True, coverage_n: int = 0, progress: Callable[[str], None] | None = None, seed: int = HELD_OUT_SEED) -> dict[str, Any]:
    shoot = RenderCache(cache_dir)
    say = progress or (lambda s: None)
    records: list[dict[str, Any]] = []
    noms = nominal_cases(n_nominal, seed=seed, size=size)
    for c in noms:
        for m in modes:
            say(f"{c.name} [{m}]")
            records.append(run_case(c, m, library, shoot))
    negs = negative_cases(size=size) if with_negative else []
    for c in negs:
        say(f"{c.name} [multi_auto]")
        records.append(run_case(c, "multi_auto", library, shoot))
    cov: list[dict[str, Any]] = []
    if coverage_n and noms:
        for m in modes:
            say(f"coverage trials [{m}]")
            cov += coverage_trials(noms[0], library, shoot, n=coverage_n if m == "single_manual" else max(4, coverage_n // 4), mode=m)
    summary = summarise(records)
    return {"records": records, "summary": summary, "coverage": cov, "n_nominal": len(noms), "n_negative": len(negs)}


def dump(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
