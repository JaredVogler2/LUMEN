#!/usr/bin/env python
"""Ablation of the experimental photometric (phone torch) module.

Compares, on synthetic photos with known geometry:

  (a) the edge-only baseline: cylinder fitted to silhouettes, run through the
      real session service (tag detection, registration, datum proposal);
  (b) the photometric module ``lumen.vision.photometric`` with camera poses
      from the real tag pipeline (``detect_tags`` + ``solve_multiview``); the
      plane frame is the frame of the gauge tag the session service picks
      (group G compares other ways of fixing that frame);
  (c) both combined by inverse variance (``combine_with_silhouette``), which
      refuses to combine when the two disagree.

Everything printed is measured in the run that prints it.  Nothing here is a
physical measurement: the photos come from ``lumen.synth``.

    .venv/bin/python scripts/photometric_ablation.py                 # everything (renders are cached on disk)
    .venv/bin/python scripts/photometric_ablation.py --groups A,F    # some groups
    .venv/bin/python scripts/photometric_ablation.py --list          # what would run
    .venv/bin/python scripts/photometric_ablation.py --write-doc     # also refresh the tables in docs/PHOTOMETRIC_ABLATION.md

The configurations below are the *held-out* set: camera rigs, light, lens and
noise settings differ from the scenes that were used while the module and its
thresholds were developed (see docs/PHOTOMETRIC_ABLATION.md, "Protocol").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LUMEN_STORAGE", "memory")

from lumen.errors import GeometryError  # noqa: E402
from lumen.geometry.camera import CameraModel  # noqa: E402
from lumen.geometry.lines import Line2D  # noqa: E402
from lumen.geometry.se3 import Pose  # noqa: E402
from lumen.synth import BackgroundSpec, Box, CameraSpec, LightSpec, Placard, Scene, Seam, render  # noqa: E402
from lumen.synth.fixtures import curved_panel_scene, flat_panel_scene, phone_camera, sweep_stations  # noqa: E402
from lumen.vision.markers import detect_tags  # noqa: E402
from lumen.vision.multiview import MVView, solve_multiview  # noqa: E402
from lumen.vision.photometric import (  # noqa: E402
    EdgeHint, FlashView, PhotometricConfig, PhotometricResult, analyse_flash_sweep, combine_with_silhouette, tag_exclusion_polygons,
)
from lumen.vision.planar import TagObs, choose_gauge  # noqa: E402

W, H = 1600, 1200
DOC = ROOT / "docs" / "PHOTOMETRIC_ABLATION.md"
BEGIN, END = "<!-- BEGIN GENERATED TABLES (scripts/photometric_ablation.py) -->", "<!-- END GENERATED TABLES -->"

try:  # share the on-disk render cache with the tests
    from tests.conftest import render_cached as _render_cached
except Exception:  # pragma: no cover - the script also runs without the test package
    _render_cached = None


def photo(scene: Scene, cam: CameraSpec, seed: int) -> np.ndarray:
    if _render_cached is not None:
        return _render_cached(scene, cam, seed=seed).image
    return render(scene, cam, seed=seed).image


def _render_job(job: tuple[Scene, CameraSpec, int]) -> None:
    photo(*job)


# --------------------------------------------------------------------------
# Held-out configurations
# --------------------------------------------------------------------------
#: lens / sensor of the held-out photos: some vignetting, more blur and noise than the development scenes
CAM_KW: dict[str, Any] = {"auto_exposure": True, "vignette": 0.15, "blur_sigma": 1.0, "noise_shot": 0.0015, "noise_read": 0.006}
#: torch of the held-out photos: other position, narrower beam and other strength than the development scenes
LIGHT_KW: dict[str, Any] = {"flash": True, "flash_strength": 1.1, "flash_offset_cam": (9.0, 13.0, 0.0), "flash_beam_exponent": 2.0}
Z_MID, HALF = 345.0, 215.0  # stretch of the edge that is analysed (scene z, mm)


def rig(n: int = 6, distance: float = 680.0, height: float = 390.0, span: float = 54.0, target: tuple[float, float, float] = (205.0, 0.0, 340.0), **kw: Any) -> list[CameraSpec]:
    ck = dict(CAM_KW)
    ck.update(kw)
    return [phone_camera(e, target=target, width=W, height=H, roll_deg=r, **ck) for e, r in sweep_stations(n, distance=distance, height=height, span_deg=span, target=target)]


def edge_scene(radius: float = 35.0, taper: float = 1.5, ambient: float = 0.2, light_kw: dict[str, Any] | None = None, **kw: Any) -> Scene:
    lk = dict(LIGHT_KW)
    lk.update(light_kw or {})
    return curved_panel_scene(radius=radius, taper_deg=taper, light=LightSpec(ambient=ambient, **lk), **kw)


@dataclass
class Case:
    key: str
    group: str
    label: str
    scene: Scene
    cams: list[CameraSpec]
    radius: float | None = 35.0  # None: there is no bend (the module must refuse)
    taper: float = 1.5
    x_edge: float = 0.0  # scene x of the hinted place at floor level (for scenes without a bend)
    #: "tags": real tag pipeline, plane frame = frame of the gauge tag the session service picks;
    #: "tags_solver": the same with the gauge tag the multi-view solver picks by default;
    #: "tags_plane": as "tags", but the plane is fitted through the corners of all tags;
    #: "truth": renderer poses
    pose: str = "tags"
    pairs: bool = False  # flash-off partners, taken from a rest
    off_jitter_mm: float = 4.0
    cfg: PhotometricConfig | None = None
    hint_radius: float | None = None
    baseline: tuple[str, ...] = ("torch",)  # which photo sets the edge-only baseline is run on
    torch: str = "on"  # torch state reported for the "on" photos
    seed0: int = 1000
    note: str = ""

    def off_scene(self) -> Scene:
        return replace(self.scene, light=replace(self.scene.light, flash=False))

    def off_cams(self) -> list[CameraSpec]:
        out = []
        for i, c in enumerate(self.cams):
            v = np.random.default_rng(self.seed0 + 500 + i).normal(size=3)
            v = v / np.linalg.norm(v) * self.off_jitter_mm
            out.append(replace(c, C=tuple(float(x) for x in np.asarray(c.C) + v)))
        return out

    def renders(self) -> list[tuple[Scene, CameraSpec, int]]:
        jobs = [(self.scene, c, self.seed0 + i) for i, c in enumerate(self.cams)]
        if self.pairs or "ambient" in self.baseline:
            jobs += [(self.off_scene(), c, self.seed0 + 100 + i) for i, c in enumerate(self.off_cams())]
        return jobs


def build_cases() -> list[Case]:
    cs: list[Case] = []
    wide = rig()
    sub = {"6 views, 54 deg": [0, 1, 2, 3, 4, 5], "4 views": [0, 2, 3, 5], "3 views": [1, 3, 5], "2 views over the flat face": [3, 5]}

    # A. radius x number / spread of views; baseline on the same torch photos and on ambient-only photos
    for R in (20.0, 35.0, 60.0):
        sc = edge_scene(radius=R)
        for name, idx in sub.items():
            cs.append(Case(f"A-R{R:.0f}-{len(idx)}v{'' if name != '2 views over the flat face' else 'f'}", "A", f"R {R:.0f} mm, {name}", sc, [wide[i] for i in idx], radius=R, baseline=("torch", "ambient"), seed0=1000 + idx[0]))
    narrow = rig(n=3, span=12.0, distance=700.0, height=410.0, target=(150.0, 0.0, 345.0))
    for R in (20.0, 35.0, 60.0):
        cs.append(Case(f"A-R{R:.0f}-narrow", "A", f"R {R:.0f} mm, 3 views within 12 deg", edge_scene(radius=R), narrow, radius=R, baseline=("torch", "ambient"), seed0=1100))
    far = rig(distance=880.0, height=470.0, span=60.0, target=(215.0, 0.0, 345.0))
    cs.append(Case("A-R35-far", "A", "R 35 mm, 6 views from 880 mm", edge_scene(), far, baseline=("torch", "ambient"), seed0=1200))
    cs.append(Case("A-R35-1v", "A", "R 35 mm, 1 view", edge_scene(), [wide[3]], baseline=("torch",), seed0=1003))

    # B. ambient level (torch strength fixed)
    for amb in (0.0, 0.5, 1.0, 2.5):
        cs.append(Case(f"B-amb{amb}", "B", f"ambient {amb} (torch 1.1 at 500 mm)", edge_scene(ambient=amb), wide, seed0=1300))

    # C. gloss
    for ks, sh in ((0.2, 60.0), (0.5, 25.0), (1.0, 150.0)):
        cs.append(Case(f"C-ks{ks}-sh{sh:.0f}", "C", f"gloss strength {ks}, shininess {sh:.0f}", edge_scene(panel_kw={"spec_strength": ks, "shininess": sh}), wide, seed0=1400))

    # D. exposure differences between views (no auto exposure)
    for name, exps in (("x0.6 .. x1.7", (0.8, 1.5, 0.6, 1.2, 1.7, 0.9)), ("all x3 (clipping)", (3.0,) * 6), ("all x0.04 (dark)", (0.04,) * 6)):
        cams = [replace(c, auto_exposure=False, exposure=e) for c, e in zip(wide, exps)]
        cs.append(Case(f"D-{name.split()[0]}{name.split()[1] if name.startswith('all') else ''}", "D", f"fixed exposures {name}", edge_scene(), cams, seed0=1500))

    # E. albedo / material near the edge (no flash-off partners)
    strip = Placard("strip", center=(-12.0, 345.0), width=56.0, height=760.0, albedo=(0.52, 0.52, 0.50), border=0.0, text_lines=0)
    strip_flat = Placard("strip", center=(40.0, 345.0), width=26.0, height=760.0, albedo=(0.52, 0.52, 0.50), border=0.0, text_lines=0)
    for key, label, kw in (
        ("E-strip-edge", "paint strip (albedo 0.52) from 40 mm on the bend to 16 mm on the face", {"placards": (strip,)}),
        ("E-strip-flat", "paint strip on the face, 27 to 53 mm from the edge", {"placards": (strip_flat,)}),
        ("E-seam", "sharp dark line 14 mm from the edge", {"seams": (Seam((14.0, 0.0), (14.0 + 1100.0 * np.tan(np.deg2rad(1.5)), 1100.0), width=2.0, darkness=0.5),)}),
        ("E-softseam", "soft 3 % dark band 9 mm from the edge", {"seams": (Seam((9.0, 0.0), (9.0 + 1100.0 * np.tan(np.deg2rad(1.5)), 1100.0), soft=5.0, darkness=0.03),)}),
        ("E-texture-fine", "texture 4 %, 6 mm grain", {"panel_kw": {"texture_contrast": 0.04}}),
        ("E-texture-coarse", "albedo variation 8 %, 120 mm patches", {"panel_kw": {"texture_contrast": 0.08, "texture_scale": 120.0}}),
    ):
        cs.append(Case(key, "E", label, edge_scene(**kw), wide, seed0=1600))

    # F. low silhouette contrast: a wall close behind the edge
    near_dark = BackgroundSpec(y=110.0, albedo=(0.27, 0.27, 0.27), texture_contrast=0.0)
    near_same = BackgroundSpec(y=110.0, albedo=(0.74, 0.74, 0.72), texture_contrast=0.0)
    cs.append(Case("F-wall-dim", "F", "wall 110 mm behind, albedo 0.27 (matches the limb in torch photos)", replace(edge_scene(), background=near_dark), wide, baseline=("torch", "ambient"), seed0=1700))
    cs.append(Case("F-wall-same", "F", "wall 110 mm behind, same albedo as the panel", replace(edge_scene(), background=near_same), wide, baseline=("torch", "ambient"), seed0=1700))
    cs.append(Case("F-dark-room", "F", "dark room (ambient 0): the limb is as dark as the background", edge_scene(ambient=0.0), wide, baseline=("torch",), seed0=1300))

    # G. where the pose error comes from: same photos, renderer poses and three ways of using the tag pipeline
    for R in (20.0, 60.0):
        cs.append(Case(f"G-R{R:.0f}-truth", "G", f"R {R:.0f} mm, 6 views, renderer poses", edge_scene(radius=R), wide, radius=R, pose="truth", baseline=(), seed0=1000))
        cs.append(Case(f"G-R{R:.0f}-tags", "G", f"R {R:.0f} mm, 6 views, tag pipeline", edge_scene(radius=R), wide, radius=R, baseline=(), seed0=1000))
    for key, name, cams, seed in (("6v", "6 views", wide, 1000), ("4v", "4 views", [wide[i] for i in (0, 2, 3, 5)], 1000), ("3v", "3 views", [wide[i] for i in (1, 3, 5)], 1001),
                                  ("2vf", "2 views over the flat face", [wide[3], wide[5]], 1003), ("narrow", "3 views within 12 deg", narrow, 1100),
                                  ("far", "6 views from 880 mm", far, 1200)):
        for pose, what in (("truth", "renderer poses"), ("tags", "tag pipeline"), ("tags_solver", "tag pipeline, gauge tag of the multi-view solver"),
                           ("tags_plane", "tag pipeline, plane through all tags")):
            cs.append(Case(f"G-R35-{key}-{pose}", "G", f"R 35 mm, {name}, {what}", edge_scene(), cams, pose=pose, baseline=(), seed0=seed))

    # H. flash-off partners: none / difference / ratio; plain, patterned, uneven room light
    directional = {"directional": ((0.55, 0.72, -0.42), 0.35)}
    for key, label, sc in (
        ("plain", "plain panel", edge_scene()),
        ("strip", "paint strip over the edge", edge_scene(placards=(strip,))),
        ("coarse", "albedo variation 8 %, 120 mm patches", edge_scene(panel_kw={"texture_contrast": 0.08, "texture_scale": 120.0})),
        ("dirlight", "room light from one side (ambient 0.2 + directional 0.35)", edge_scene(light_kw=directional)),
    ):
        seed = 1800 if key != "plain" else 1000
        cs.append(Case(f"H-{key}-none", "H", f"{label}; no partner", sc, wide, seed0=seed, baseline=("torch", "ambient") if key == "dirlight" else ()))
        cs.append(Case(f"H-{key}-diff", "H", f"{label}; partners, difference", sc, wide, pairs=True, cfg=PhotometricConfig(pair_mode="difference"), baseline=(), seed0=seed))
        cs.append(Case(f"H-{key}-ratio", "H", f"{label}; partners, ratio", sc, wide, pairs=True, cfg=PhotometricConfig(pair_mode="ratio"), baseline=(), seed0=seed))
    cs.append(Case("H-plain-misaligned", "H", "plain panel; partners taken 45 mm away", edge_scene(), wide, pairs=True, off_jitter_mm=45.0, baseline=(), seed0=1000))

    # I. radiometry the browser cannot promise, sensor effects
    for key, label, kw, lk in (
        ("gamma1.2", "tone: extra gamma 1.2", {"tone_gamma": 1.2}, None),
        ("gamma0.8", "tone: extra gamma 0.8", {"tone_gamma": 0.8}, None),
        ("scurve0.2", "tone: S-curve 0.2", {"tone_contrast": 0.2}, None),
        ("scurve0.5", "tone: S-curve 0.5", {"tone_contrast": 0.5}, None),
        ("sharpen", "ISP sharpening 0.8", {"sharpen": 0.8}, None),
        ("vignette0.6", "lens vignetting 0.6", {"vignette": 0.6}, None),
        ("jpeg70", "JPEG quality 70", {"jpeg_quality": 70}, None),
        ("noisehigh", "high noise", {"noise_shot": 0.004, "noise_read": 0.012}, None),
        ("distortion", "lens distortion (calibrated)", {"dist": (0.07, -0.13, 0.0005, -0.0004, 0.0)}, None),
        ("beam5", "torch beam exponent 5", {}, {"flash_beam_exponent": 5.0}),
        ("offset30", "torch 30 mm from the lens", {}, {"flash_offset_cam": (-26.0, 15.0, 0.0)}),
    ):
        cs.append(Case(f"I-{key}", "I", label, edge_scene(light_kw=lk), rig(**kw), seed0=1900))

    # J. cases that must be refused
    flat = flat_panel_scene(light=LightSpec(ambient=0.2, **LIGHT_KW))
    cs.append(Case("J-flat", "J", "flat panel, hint in the middle of the face", flat, wide, radius=None, taper=0.0, x_edge=210.0, baseline=(), seed0=2000))
    cs.append(Case("J-flat-seam", "J", "flat panel, hint on a dark joint line", flat, wide, radius=None, taper=0.0, x_edge=0.0, baseline=(), seed0=2000))
    wall = Box(lo=(150.0, -60.0, 60.0), hi=(190.0, 0.0, 700.0))
    cs.append(Case("J-flat-shadow", "J", "flat panel, 60 mm proud block: hint at its torch shadow", replace(flat, boxes=(wall,)), wide, radius=None, taper=0.0, x_edge=207.0, baseline=(), seed0=2000))
    block = Box(lo=(34.0, -45.0, 250.0), hi=(62.0, 0.0, 330.0))
    cs.append(Case("J-edge-block", "J", "rounded edge with a 45 mm proud block next to it (shadow, occlusion)", replace(edge_scene(), boxes=(block,)), wide, seed0=2100))
    cs.append(Case("J-bend45", "J", "bend of only 45 degrees, then a straight flange", edge_scene(sweep_deg=45.0), wide, seed0=2200))
    cs.append(Case("J-torch-off", "J", "ambient photos only (torch off)", edge_scene(), wide, torch="off", pairs=False, baseline=("ambient",), seed0=1000))
    cs.append(Case("J-torch-unknown", "J", "torch state not recorded", edge_scene(), wide, torch="unknown", baseline=(), seed0=1000))

    # K. model options (same photos as A-R35-6v and three stress cases)
    variants = (
        ("default", PhotometricConfig()),
        ("tone exponent fixed", PhotometricConfig(fit_tone_gamma=False)),
        ("no lens fall-off", PhotometricConfig(fit_lens_falloff=False)),
        ("no 2nd-order beam/lens terms", PhotometricConfig(fit_beam_shape2=False, fit_lens_falloff2=False)),
        ("torch at the lens (offset fixed)", PhotometricConfig(fit_light_offset=False)),
        ("beam tilt free", PhotometricConfig(fit_beam_axis=True)),
        ("albedo gradient free", PhotometricConfig(fit_albedo_gradient=True)),
        ("ambient harmonics free", PhotometricConfig(fit_ambient_harmonics=True)),
    )
    for vname, cfg in variants:
        tag = vname.replace(" ", "-")
        cs.append(Case(f"K-plain-{tag}", "K", f"plain, 6 views | {vname}", edge_scene(), wide, cfg=cfg, baseline=(), seed0=1000))
        cs.append(Case(f"K-2v-{tag}", "K", f"plain, 2 views | {vname}", edge_scene(), [wide[3], wide[5]], cfg=cfg, baseline=(), seed0=1003))
        cs.append(Case(f"K-gamma-{tag}", "K", f"tone gamma 1.2 | {vname}", edge_scene(), rig(tone_gamma=1.2), cfg=cfg, baseline=(), seed0=1900))
        cs.append(Case(f"K-dirlight-{tag}", "K", f"room light from one side | {vname}", edge_scene(light_kw=directional), wide, cfg=cfg, baseline=(), seed0=1800))
    return cs


# --------------------------------------------------------------------------
# Poses, truth, baseline
# --------------------------------------------------------------------------
R_WP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # scene face frame (x = X, y = Z, z = -Y) -> world


def model_of(c: CameraSpec) -> CameraModel:
    return CameraModel(c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.dist, source="synthetic_truth")


def gauge_in_scene(scene: Scene, gauge_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and origin of the gauge tag frame in the scene face frame (x, z)."""
    st = next(s for s in scene.stickers if s.tag_id == gauge_id)
    a = np.deg2rad(st.rotation_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), np.asarray(st.center, dtype=float)


def tag_poses(images: list[np.ndarray], cams: list[CameraSpec], gauge: int | None, own_gauge: bool = False) -> tuple[Any, list[float | None], int]:
    """Camera poses from the real tag pipeline.  Its world frame is the frame of one tag, the gauge: the one the
    session service uses (``gauge`` when the service was run on these photos, otherwise its rule ``choose_gauge``:
    seen in most photos, then largest in the image) or, with ``own_gauge``, the default of the multi-view solver
    (seen in most photos, then lowest id).  Also returns what the service rule gives for these detections."""
    views, blur = [], []
    for i, (im, c) in enumerate(zip(images, cams)):
        cm = model_of(c)
        dets = [d for d in detect_tags(im, camera=cm) if d.accepted]
        blur.append(float(np.median([d.sigma_blur_px for d in dets])) if dets else None)
        views.append(MVView(f"f{i}", [TagObs(d.tag_id, d.corners_ideal, d.cov, d.size_mm) for d in dets], cm, {d.tag_id: d.corners_px for d in dets}, (c.width, c.height)))
    rule = int(choose_gauge(views))  # raises GeometryError("no_tags") when there is no tag at all
    want = None if own_gauge else (gauge if gauge is not None else rule)
    return solve_multiview(views, gauge=want, checks=False), blur, rule


def plane_through_tags(sol: Any) -> Pose:
    """Frame of the plane fitted through the corners of all tags, given in the frame of the gauge tag: z along the
    fitted normal, x the gauge x axis projected into the plane, origin the gauge origin dropped onto the plane."""
    pts = np.vstack([sol.tag_corners_world(t) for t in sol.tags])
    c0 = pts.mean(axis=0)
    n = np.linalg.svd(pts - c0)[2][2]
    n = n if n[2] > 0 else -n
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n
    x /= np.linalg.norm(x)
    return Pose(np.column_stack([x, np.cross(n, x), n]), float(c0 @ n) * n)


def frame_error(c: "Case", cams: list[CameraSpec], poses: list[Pose | None], Rg: np.ndarray, cg: np.ndarray) -> list[float] | None:
    """Where the assumed plane frame really is at the middle of the edge (uses the renderer's camera poses): the
    point the estimated poses call "edge mid-point, on the face" lies this far from the true one, in mm
    (across the edge toward the face, along the edge, out of the panel); mean over the photos."""
    Rg3 = np.eye(3)
    Rg3[:2, :2] = Rg
    T_plane_gauge = Pose(Rg3, np.array([cg[0], cg[1], 0.0]))  # true pose of the gauge tag in the scene face frame
    T_world_gauge = Pose(R_WP, np.zeros(3)) @ T_plane_gauge
    p = T_plane_gauge.inv().apply(np.array([[c.x_edge + Z_MID * np.tan(np.deg2rad(c.taper)), Z_MID, 0.0]]))
    moved = []
    for cam, est in zip(cams, poses):
        if est is not None:
            true = Pose(np.asarray(cam.R_wc), np.asarray(cam.C)).inv() @ T_world_gauge
            moved.append(Rg3 @ ((true.inv() @ est).apply(p)[0] - p[0]))
    return None if not moved else [float(x) for x in np.mean(moved, axis=0)]


def line_errors(n: np.ndarray, d: float, Rg: np.ndarray, cg: np.ndarray, taper: float, x_edge: float) -> tuple[float, float]:
    """Tangent offset error at mid height (mm, + toward the flat face) and direction error (deg)."""
    ns = Rg @ np.asarray(n, dtype=float)
    ds = float(d) + float(ns @ cg)
    psi = np.deg2rad(taper)
    n_true = np.array([np.cos(psi), -np.sin(psi)])
    if float(ns @ n_true) < 0:
        ns, ds = -ns, -ds
    p_mid = np.array([x_edge + Z_MID * np.tan(psi), Z_MID])
    return float(ds - ns @ p_mid), float(np.degrees(np.arctan2(n_true[0] * ns[1] - n_true[1] * ns[0], float(ns @ n_true))))


_LIB: Any = None


def _library() -> Any:
    """The bundled operation library with the configured radius of the demo
    curve removed, so that the baseline has to find the radius itself."""
    global _LIB
    if _LIB is None:
        from lumen.operations.library import OperationLibrary

        src = ROOT / "lumen" / "operations" / "data" / "operations.json"
        data = json.loads(src.read_text())
        for op in data["operations"]:
            if op["operation_id"] == "DEMO-CURVE-01":
                op["datums"]["side"].pop("radius", None)
        path = Path(tempfile.mkdtemp(prefix="lumen-ablation-")) / "operations.json"
        path.write_text(json.dumps(data))
        _LIB = OperationLibrary(path)
    return _LIB


def silhouette_baseline(scene: Scene, cams: list[CameraSpec], images: list[np.ndarray]) -> dict[str, Any]:
    """Edge-only estimate through the session service (as the application runs it)."""
    from lumen.config import Limits, Settings
    from lumen.schemas import Calibration
    from lumen.services.sessions import SessionService
    from lumen.storage.base import MemorySessionStore

    t0 = time.perf_counter()
    out: dict[str, Any] = {"ok": False, "reason": ""}
    try:
        svc = SessionService(MemorySessionStore(), Settings(limits=Limits(), storage="memory"), _library())
        sid = svc.create().id
        svc.set_operation(sid, "DEMO-CURVE-01", None, "PLC-DEMO-201")
        svc.set_tag_sizes(sid, {"nominal_mm": 50.0, "measured_mm": {str(s.tag_id): float(s.size) for s in scene.stickers}})
        c = cams[0]
        svc.set_calibration(sid, Calibration(id="truth", label="synthetic truth", method="synthetic_truth", provenance="simulated", width=c.width, height=c.height,
                                             fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy, dist=list(c.dist), param_sigma=[0.5, 0.5, 0.5, 0.5, 0.0005, 0.002, 0.0001, 0.0001, 0.0]))
        for im in images:
            ok, buf = cv2.imencode(".png", im)
            svc.add_frame(sid, buf.tobytes())
        reg = svc.register(sid)
        cyl = svc.propose_datums(sid)["cylinder"]
        out["gauge"] = int(reg["layout"]["gauge_id"])
        if not cyl or not cyl.get("ok") or cyl.get("tangent") is None:
            out["reason"] = (cyl or {}).get("reason") or "no_cylinder"
        else:
            ends = np.asarray(cyl["endpoints"], dtype=float)
            out.update(ok=True, n=cyl["tangent"]["n"], d=cyl["tangent"]["d"], radius=cyl["radius_mm"], n_views=cyl["n_views"],
                       sigma_t=float(np.hypot(cyl["sigma_tangent_mm"], cyl["rms_mm"])), sigma_ang=cyl["sigma_angle_deg"], sigma_r=cyl["sigma_radius_mm"],
                       ref=ends.mean(axis=0).tolist(), radius_fixed=cyl["radius_fixed"])
    except Exception as exc:  # the service belongs to another part of the project: report, do not crash
        out["reason"] = f"error: {type(exc).__name__}: {exc}"[:160]
    out["time_s"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------
def run_case(c: Case) -> dict[str, Any]:
    row: dict[str, Any] = {"key": c.key, "group": c.group, "label": c.label, "n_views": len(c.cams), "radius": c.radius, "note": c.note, "pose": c.pose}
    # identical photo sets / identical runs appear in more than one group: count them once in the summaries
    row["photos_sig"] = hashlib.md5(repr((c.scene, c.cams, c.seed0)).encode()).hexdigest()[:12]
    row["run_sig"] = hashlib.md5(repr((c.scene, c.cams, c.seed0, c.pose, c.pairs, c.off_jitter_mm, c.cfg, c.hint_radius, c.torch)).encode()).hexdigest()[:12]
    on = [photo(c.scene, cam, c.seed0 + i) for i, cam in enumerate(c.cams)]
    off_cams = c.off_cams()
    off = [photo(c.off_scene(), cam, c.seed0 + 100 + i) for i, cam in enumerate(off_cams)] if (c.pairs or "ambient" in c.baseline or c.torch == "off") else []

    # (a) edge-only baseline
    base: dict[str, dict[str, Any]] = {}
    for which in c.baseline:
        b = silhouette_baseline(c.scene, c.cams if which == "torch" else off_cams, on if which == "torch" else off)
        if b["ok"]:
            Rg, cg = gauge_in_scene(c.scene, b["gauge"])
            b["err_t"], b["err_ang"] = line_errors(np.array(b["n"]), b["d"], Rg, cg, c.taper, c.x_edge)
            b["err_r"] = None if c.radius is None else b["radius"] - c.radius
        base[which] = b
    row["baseline"] = base
    gauge = next((b["gauge"] for b in base.values() if "gauge" in b), None)

    # (b) photometric, with poses from the tag pipeline (or from the renderer)
    t0 = time.perf_counter()
    if c.torch == "off":
        imgs, cams, states = off, off_cams, ["off"] * len(off)
    else:
        imgs, cams, states = list(on), list(c.cams), [c.torch] * len(on)
        if c.pairs:
            imgs, cams, states = imgs + off, cams + off_cams, states + ["off"] * len(off)
    photo_gauge: int | None = None
    if c.pose == "truth":
        Rg, cg = np.eye(2), np.zeros(2)  # the plane frame is the scene face frame itself
        T_world_plane = Pose(R_WP, np.zeros(3))
        poses = [Pose(np.asarray(cam.R_wc), np.asarray(cam.C)).inv() @ T_world_plane for cam in cams]
        blur: list[float | None] = [cam.blur_sigma for cam in cams]
        polys = tag_exclusion_polygons(true_tag_squares(c.scene))
    else:
        try:
            sol, blur, rule = tag_poses(imgs, cams, gauge, own_gauge=c.pose == "tags_solver")
        except GeometryError as exc:  # no usable tags: there are no poses, the photometric step is never reached
            row["time_pose_s"] = time.perf_counter() - t0
            row["photo"] = {"status": "not_run", "reasons": [f"no_tag_poses ({exc.code})"], "suggestion": "", "requests": [], "n_views_used": 0, "runtime_s": 0.0,
                            "nuisance": {}, "masks_max": {}, "roles": [], "gradient_axis_err_deg": None}
            row["combined"] = {}
            return row
        photo_gauge = int(sol.gauge_id)
        Rg, cg = gauge_in_scene(c.scene, sol.gauge_id)
        T_gp = plane_through_tags(sol) if c.pose == "tags_plane" else Pose.identity()  # plane frame -> frame of the gauge tag
        poses = [None if sol.cameras.get(f"f{i}") is None else sol.cameras[f"f{i}"] @ T_gp for i in range(len(cams))]
        polys = tag_exclusion_polygons([T_gp.inv().apply(sol.tag_corners_world(t))[:, :2] for t in sol.tags])
        row["pose_rms_px"] = sol.rms_px
        st = next(s_ for s_ in c.scene.stickers if s_.tag_id == sol.gauge_id)
        mid = np.array([c.x_edge + Z_MID * np.tan(np.deg2rad(c.taper)), Z_MID])
        row["gauge"] = {"id": photo_gauge, "service_rule": rule, "from_service": gauge, "distance_to_edge_mid_mm": float(np.linalg.norm(np.asarray(st.center, dtype=float) - mid))}
        row["frame_error_mm"] = frame_error(c, cams, poses, Rg, cg)
    row["time_pose_s"] = time.perf_counter() - t0
    row["n_photos"] = len(imgs)
    n_on = len(on) if c.torch != "off" else 0
    views: list[FlashView] = []
    where: dict[int, int] = {}
    for i, (im, cam, stt) in enumerate(zip(imgs, cams, states)):
        if poses[i] is None:
            continue  # a photo the bundle adjustment could not localise
        where[i] = len(views)
        views.append(FlashView(im, model_of(cam), poses[i], stt, None, f"{'on' if i < n_on else 'off'}{i % max(n_on, 1)}", blur[i]))
    if c.pairs and c.torch != "off":
        for i in range(n_on):
            if i in where and i + n_on in where:
                views[where[i]].pair, views[where[i + n_on]].pair = where[i + n_on], where[i]
    # hint: what the datum step or an operator would provide (5 mm and 1.5 degrees off, radius unknown)
    psi = np.deg2rad(c.taper + 1.5)
    p_s = np.array([c.x_edge + Z_MID * np.tan(np.deg2rad(c.taper)) - 5.0, Z_MID])
    hint = EdgeHint(point=tuple(Rg.T @ (p_s - cg)), direction=tuple(Rg.T @ np.array([np.sin(psi), np.cos(psi)])), inward=tuple(Rg.T @ np.array([1.0, 0.0])),
                    radius_mm=c.hint_radius, extent_mm=(-HALF, HALF))
    res = analyse_flash_sweep(views, hint, c.cfg, exclude_polygons=polys)
    row["photo"] = summarise(res, Rg, cg, c)

    # (c) both combined (only when the baseline of the same or the ambient photos exists and both are in one frame)
    comb: dict[str, Any] = {}
    for which, b in base.items():
        if not b.get("ok") or photo_gauge is None or b["gauge"] != photo_gauge:
            continue  # both estimates have to be in the frame of the same tag
        k = combine_with_silhouette(res, Line2D(np.array(b["n"]), b["d"]), b["sigma_t"], b["sigma_ang"], b["radius"], b["sigma_r"], reference_point=np.array(b["ref"]))
        if k.get("available"):
            k["err_t"], k["err_ang"] = line_errors(np.array(k["tangent"]["n"]), k["tangent"]["d"], Rg, cg, c.taper, c.x_edge)
            k["err_r"] = None if (c.radius is None or k.get("radius_mm") is None) else k["radius_mm"] - c.radius
        comb[which] = {kk: vv for kk, vv in k.items() if kk not in ("tangent", "tangent_point", "label")}
    row["combined"] = comb
    return row


def true_tag_squares(scene: Scene) -> list[np.ndarray]:
    """True tag squares in the scene face frame (exclusion polygons when renderer poses are used)."""
    out = []
    for st in scene.stickers:
        h = 0.5 * st.size
        a = np.deg2rad(st.rotation_deg)
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(np.array([[-h, h], [h, h], [h, -h], [-h, -h]]) @ Rm.T + np.asarray(st.center))
    return out


def summarise(res: PhotometricResult, Rg: np.ndarray, cg: np.ndarray, c: Case) -> dict[str, Any]:
    cond = res.conditioning
    out: dict[str, Any] = {
        "status": res.status, "reasons": res.reasons, "suggestion": res.suggestions[0] if res.suggestions else "", "requests": [r.get("code") for r in res.requests],
        "n_views_used": res.n_views_used, "runtime_s": res.runtime_s, "radius_source": res.radius_source,
        "condition": cond.get("condition_number_scaled"), "rms_rel": cond.get("rms_rel"), "stationarity_mm": cond.get("stationarity_mm"),
        "heldout_ratio": res.heldout.get("max_ratio"), "heldout_rms": res.heldout.get("max_heldout_rms_rel"),
        "fixed_pattern_mm": (cond.get("fixed_pattern") or {}).get("allowance_tangent_mm"),
        "sigma_formal": cond.get("sigma_formal"), "sigma_rows": cond.get("sigma_row_blocks"), "sigma_jack": cond.get("sigma_view_jackknife"),
        "gradient_axis_err_deg": None, "nuisance": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in res.nuisance.items()},
        "beam_m2": (res.nuisance.get("beam_exponent") or {}).get("second_order"),
        "light_offset_sigma": (res.nuisance.get("light_offset_cam_mm") or {}).get("sigma"),
        "masks_max": {}, "roles": sorted({v.role for v in res.views}),
    }
    for v in res.views:
        for k, x in v.mask_fractions.items():
            out["masks_max"][k] = max(out["masks_max"].get(k, 0.0), x)
    if res.gradient_axis.get("inward") is not None:
        n_ax = np.asarray(res.gradient_axis["inward"])
        out["gradient_axis_err_deg"] = line_errors(n_ax, float(n_ax @ np.asarray(res.gradient_axis["onset_point"])), Rg, cg, c.taper, c.x_edge)[1]
    est = None
    if res.tangent is not None:
        est = (res.tangent.n, res.tangent.d, res.radius_mm, res.sigma_tangent_mm, res.sigma_angle_deg, res.sigma_radius_mm, False)
    elif cond.get("withheld_estimate") and cond["withheld_estimate"]["tangent"] is not None:
        w = cond["withheld_estimate"]
        est = (np.array(w["tangent"]["n"]), w["tangent"]["d"], w["radius_mm"], w["sigma_tangent_mm"], w["sigma_angle_deg"], w["sigma_radius_mm"], True)
    if est is not None and c.radius is not None:
        n, d, R, st, sa, sr, withheld = est
        out["err_t"], out["err_ang"] = line_errors(n, d, Rg, cg, c.taper, c.x_edge)
        if R is None and cond.get("withheld_estimate"):
            R, sr = cond["withheld_estimate"]["radius_mm"], cond["withheld_estimate"]["sigma_radius_mm"]
            out["radius_withheld"] = True
        out.update(err_r=None if R is None else R - c.radius, sigma_t=st, sigma_ang=sa, sigma_r=sr, withheld=withheld)
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _pm(err: float | None, sig: float | None, nd: int = 2) -> str:
    if err is None or not np.isfinite(err):
        return "-"
    s = f"{err:+.{nd}f}"
    return s + (f" ± {sig:.{nd}f}" if sig is not None and np.isfinite(sig) else "")


def _base_cell(b: dict[str, Any] | None) -> str:
    if b is None:
        return "not run"
    if not b.get("ok"):
        return f"fails ({b.get('reason') or 'no fit'})"
    return f"{_pm(b['err_t'], b['sigma_t'])} / {b['err_ang']:+.3f} / {_pm(b['err_r'], b['sigma_r'])}"


def _photo_cell(p: dict[str, Any]) -> str:
    if "err_t" not in p:
        return "-"
    r = _pm(p.get("err_r"), p.get("sigma_r"))
    cell = f"{_pm(p['err_t'], p['sigma_t'])} / {_pm(p['err_ang'], p['sigma_ang'], 3)} / {r}"
    return f"[{cell}]" if p.get("withheld") else cell


def _status(p: dict[str, Any]) -> str:
    s = p["status"]
    extra = [r for r in p["reasons"]]
    return s if s == "supported" and not extra else f"{s}: {', '.join(extra)}"


def _comb_cell(k: dict[str, Any] | None) -> str:
    if not k:
        return "-"
    if not k.get("available"):
        if k.get("reason") == "estimates_disagree":
            return f"not combined: disagree (z = {max(k.get('z_offset') or 0, k.get('z_angle') or 0, k.get('z_radius') or 0):.1f})"
        return f"not combined ({k.get('reason')})"
    return f"{_pm(k['err_t'], k['sigma_tangent_mm'])} / {_pm(k.get('err_r'), k.get('sigma_radius_mm'))}"


TITLES = {
    "A": "A. Radius, number and spread of views",
    "B": "B. Ambient level",
    "C": "C. Gloss",
    "D": "D. Exposure differences between views",
    "E": "E. Albedo and material near the edge (no flash-off partner)",
    "F": "F. Low silhouette contrast",
    "G": "G. Pose error of the real tag pipeline (same photos, four ways of getting poses)",
    "H": "H. Flash-off partners: none, difference, ratio",
    "I": "I. Radiometry and sensor effects",
    "J": "J. Cases that have to be refused, shadows",
    "K": "K. Model options",
}


def tables(rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    legend = ("Cells are `tangent offset error mm ± stated sigma / direction error deg / radius error mm ± stated sigma`, "
              "measured at the middle of the analysed stretch.  `[...]` marks an estimate that the module withheld "
              "(status not `supported`): it is shown to tell what trusting it would have cost.")
    out.append(legend + "\n")
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g]
        out.append(f"### {TITLES.get(g, g)}\n")
        has_amb = any("ambient" in r["baseline"] for r in rs)
        has_torch = any("torch" in r["baseline"] for r in rs)
        head = ["case"]
        if has_torch:
            head.append("(a) edge-only, same torch photos")
        if has_amb:
            head.append("(a') edge-only, ambient photos")
        head += ["(b) photometric: status", "(b) estimate", "time s"]
        if has_torch or has_amb:
            head.append("(c) combined with (a) / (a')")
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for r in rs:
            p = r["photo"]
            cells = [r["label"]]
            if has_torch:
                cells.append(_base_cell(r["baseline"].get("torch")))
            if has_amb:
                cells.append(_base_cell(r["baseline"].get("ambient")))
            cells += [_status(p), _photo_cell(p), f"{p['runtime_s']:.1f}"]
            if has_torch or has_amb:
                cc = [_comb_cell(r["combined"].get(w)) for w in ("torch", "ambient") if w in r["baseline"]]
                cells.append(" ; ".join(cc))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        if g == "G":
            out.append(frame_table(rows))
    out.append(head_to_head(rows))
    out.append(coverage(rows))
    out.append(axis_table(rows))
    out.append(nuisance_table(rows))
    out.append(runtime(rows))
    return "\n".join(out)


def coverage(rows: list[dict[str, Any]]) -> str:
    """How often do the stated standard deviations cover the true error?"""

    def stats(pairs: list[tuple[float, float]]) -> str:
        if not pairs:
            return "0 | - | - | - | - | -"
        z = np.array([abs(e) / max(s, 1e-12) for e, s in pairs])
        e = np.array([abs(e) for e, _ in pairs])
        return f"{z.size} | {np.median(e):.3f} | {np.max(e):.3f} | {np.mean(z <= 1):.2f} | {np.mean(z <= 2):.2f} | {np.max(z):.1f}"

    sets: dict[str, dict[str, list[tuple[float, float]]]] = {}

    def add(method: str, qty: str, e: float | None, s_: float | None) -> None:
        if e is not None and s_ is not None and np.isfinite(e) and np.isfinite(s_) and s_ > 0:
            sets.setdefault(method, {}).setdefault(qty, []).append((e, s_))

    seen: set[str] = set()
    for r in rows:
        if r["group"] == "K":
            continue  # option variants are not the shipped configuration
        p = r["photo"]
        first = r["run_sig"] not in seen
        seen.add(r["run_sig"])
        if not first:
            p = {"status": "duplicate"}
        name_b = {"truth": "(b) photometric, supported, renderer poses (group G only)", "tags_solver": "(b) photometric, supported, gauge tag of the solver (group G only)",
                  "tags_plane": "(b) photometric, supported, plane through all tags (group G only)"}.get(r.get("pose"), "(b) photometric, supported")
        if p["status"] == "supported" and "err_t" in p:
            add(name_b, "tangent mm", p["err_t"], p["sigma_t"])
            add(name_b, "direction deg", p["err_ang"], p["sigma_ang"])
            if not p.get("radius_withheld"):
                add(name_b, "radius mm", p.get("err_r"), p.get("sigma_r"))
        elif "err_t" in p and p.get("sigma_t"):
            add("(b) photometric, withheld (if it had been trusted)", "tangent mm", p["err_t"], p["sigma_t"])
        for w, b in r["baseline"].items():
            sig = f"{w}:{r['photos_sig']}"
            if b.get("ok") and sig not in seen:
                seen.add(sig)
                name = "(a) edge-only, torch photos" if w == "torch" else "(a') edge-only, ambient photos"
                add(name, "tangent mm", b["err_t"], b["sigma_t"])
                add(name, "direction deg", b["err_ang"], b["sigma_ang"])
                add(name, "radius mm", b["err_r"], b["sigma_r"])
        for w, k in r["combined"].items():
            if k.get("available") and first:
                add(f"(c) combined with edge-only on {w} photos", "tangent mm", k["err_t"], k["sigma_tangent_mm"])
    out = ["### Do the stated uncertainties cover the errors?\n",
           "Over all cases above except group K; a run that appears in two groups is counted once.  `within 1 sigma` should be near 0.68 and "
           "`within 2 sigma` near 0.95 for honest, Gaussian uncertainties.\n",
           "| method | quantity | n | median abs error | max abs error | within 1 sigma | within 2 sigma | worst error / sigma |", "|---|---|---|---|---|---|---|---|"]
    for m in sorted(sets):
        for q, pairs in sets[m].items():
            out.append(f"| {m} | {q} | {stats(pairs)} |")
    return "\n".join(out) + "\n"


def head_to_head(rows: list[dict[str, Any]]) -> str:
    """Photometric against edge-only, case by case, wherever the edge-only baseline was run."""
    names = {"torch": "(b) against (a), same torch photos", "ambient": "(b) against (a'), ambient photos from the same stations"}
    seen: set[str] = set()
    counts: dict[str, list[int]] = {}
    acc: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for r in rows:
        if r["group"] == "K" or r.get("pose", "tags") != "tags":
            continue
        p = r["photo"]
        sup = p["status"] == "supported" and "err_t" in p
        for w, b in r["baseline"].items():
            sig = f"{w}:{r['run_sig']}"
            if sig in seen:
                continue
            seen.add(sig)
            ok = bool(b.get("ok"))
            counts.setdefault(w, [0, 0, 0, 0])[0 if (sup and ok) else 1 if sup else 2 if ok else 3] += 1
            if sup and ok:
                for q, ea, eb in (("tangent mm", b["err_t"], p["err_t"]), ("direction deg", b["err_ang"], p["err_ang"]), ("radius mm", b["err_r"], p.get("err_r"))):
                    if ea is not None and eb is not None and not (q == "radius mm" and p.get("radius_withheld")):
                        acc.setdefault(w, {}).setdefault(q, []).append((abs(float(ea)), abs(float(eb))))
    if not counts:
        return ""
    out = ["### Head to head: photometric against edge-only\n",
           "Cases of the groups above in which the edge-only baseline was run (group K and renderer-pose runs excluded; a run that appears twice is counted once).  "
           "`photometric` means status `supported`; an estimate the module withheld counts as no estimate.\n",
           "| comparison | cases | both give an estimate | only photometric | only edge-only | neither |", "|---|---|---|---|---|---|"]
    for w in ("torch", "ambient"):
        if w in counts:
            c = counts[w]
            out.append(f"| {names[w]} | {sum(c)} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")
    out += ["", "Where both give an estimate:\n",
            "| comparison | quantity | n | edge-only: median, max abs error | photometric: median, max abs error | photometric closer to truth in |", "|---|---|---|---|---|---|"]
    for w in ("torch", "ambient"):
        for q, pairs in acc.get(w, {}).items():
            a, b = np.array([x for x, _ in pairs]), np.array([y for _, y in pairs])
            out.append(f"| {names[w]} | {q} | {a.size} | {np.median(a):.3f}, {np.max(a):.3f} | {np.median(b):.3f}, {np.max(b):.3f} | {int(np.sum(b < a))} of {a.size} |")
    return "\n".join(out) + "\n"


def frame_table(rows: list[dict[str, Any]]) -> str:
    """Photometric error against the error of the plane frame at the edge, for each way of getting poses."""
    kinds = {"truth": "renderer poses", "tags": "tag pipeline: frame of the gauge tag the session service picks",
             "tags_solver": "tag pipeline: frame of the gauge tag the multi-view solver picks", "tags_plane": "tag pipeline: plane fitted through all tags"}
    sets = [r for r in rows if r["group"] == "G" and r.get("pose") == "truth" and r["radius"] == 35.0]
    if not sets:
        return ""
    out = ["### G, continued: error of the plane frame at the edge\n",
           "The module takes the poses as exact.  `frame at the edge` is where the point that the estimated poses call \"middle of the edge, on the face\" really lies "
           "(renderer camera poses as reference, mean over the photos): across the edge (+ toward the face) and out of the panel.  R = 35 mm.\n",
           "| photos | poses | gauge tag: id, distance to the middle of the edge | frame at the edge: across mm | frame at the edge: height mm | (b) tangent error mm | (b) radius error mm |",
           "|---|---|---|---|---|---|---|"]
    for t in sets:
        for kind in kinds:
            r = next((r for r in rows if r["group"] == "G" and r["photos_sig"] == t["photos_sig"] and r.get("pose") == kind), None)
            if r is None:
                continue
            p, g, fe = r["photo"], r.get("gauge"), r.get("frame_error_mm")
            gauge = "-" if not g else f"{g['id']}, {g['distance_to_edge_mid_mm']:.0f} mm"
            out.append(f"| {t['label'].replace(', renderer poses', '').replace('R 35 mm, ', '')} | {kinds[kind]} | {gauge} | {'0 (given)' if fe is None else f'{fe[0]:+.2f}'} | "
                       f"{'0 (given)' if fe is None else f'{fe[2]:+.2f}'} | {_pm(p.get('err_t'), None)}{'' if p['status'] == 'supported' else ' (withheld)'} | {_pm(p.get('err_r'), None)} |")
    tagrows = [r for r in rows if r.get("pose") == "tags" and r.get("frame_error_mm") and r["photo"]["status"] == "supported" and "err_t" in r["photo"] and r["group"] in "ABCDFI"]
    seen: set[str] = set()
    pts = []
    for r in tagrows:
        if r["run_sig"] not in seen:
            seen.add(r["run_sig"])
            pts.append((r["frame_error_mm"][2], r["photo"]["err_t"]))
    if len(pts) >= 5:
        h, e = np.array(pts).T
        out.append("")
        out.append(f"Over the {h.size} distinct supported runs of groups A to D, F and I (plain panel, tag pipeline, gauge tag of the session service): frame height at the edge "
                   f"{np.min(h):+.2f} to {np.max(h):+.2f} mm (rms {np.sqrt(np.mean(h ** 2)):.2f}); correlation with the photometric tangent error {np.corrcoef(h, e)[0, 1]:+.2f}; "
                   f"rms tangent error {np.sqrt(np.mean(e ** 2)):.2f} mm, after removing the straight-line dependence on the frame height {np.std(e - np.polyval(np.polyfit(h, e, 1), h)):.2f} mm "
                   f"(slope {np.polyfit(h, e, 1)[0]:+.2f} mm per mm).")
    return "\n".join(out) + "\n"


def axis_table(rows: list[dict[str, Any]]) -> str:
    """Axis direction from normalised-brightness gradients alone (no shape model)."""
    vals = [(r["label"], r["photo"]["gradient_axis_err_deg"], r["photo"].get("err_ang")) for r in rows if r["group"] in ("A", "B", "C") and r["photo"].get("gradient_axis_err_deg") is not None and r["radius"] is not None]
    if not vals:
        return ""
    g = np.array([abs(v[1]) for v in vals])
    m = np.array([abs(v[2]) for v in vals if v[2] is not None])
    return ("### Axis direction from normalised-brightness gradients alone\n\n"
            f"Model-free estimate (groups A to C, {g.size} cases): median abs error {np.median(g):.3f} deg, max {np.max(g):.3f} deg.  "
            f"Joint model fit on the same cases: median {np.median(m):.3f} deg, max {np.max(m):.3f} deg.\n")


def nuisance_table(rows: list[dict[str, Any]]) -> str:
    """Torch position and beam recovered from the photos (truth is known in the renders)."""
    pick = [r for r in rows if r["key"] in ("A-R35-6v", "A-R35-3v", "A-R35-2vf", "I-offset30", "I-beam5", "B-amb0.0", "H-plain-diff", "H-plain-ratio")]
    if not pick:
        return ""
    out = ["### Torch position and beam recovered from the photos\n",
           "Truth of the base configuration: torch 9 mm right of and 13 mm below the lens, beam exponent 2, lens fall-off `1 - 0.15 + 0.15 cos^4` "
           "(to first order an exponent v1 = 0.6 in the model's `cos^v1` form), ambient / torch = 0.2 / 1.1 = 0.182, tone exponent 1.  "
           "Beam and lens fall-off multiply the same torch light: without ambient light, and when the ambient part is measured by flash-off partners "
           "(difference), only their product is identified; in the ratio of a pair the lens fall-off cancels.\n",
           "| case | status | torch offset x, y mm (sigma) | beam exponent m, m2 | lens v1, v2 | ambient / torch | tone exponent |", "|---|---|---|---|---|---|---|"]
    for r in pick:
        nu, p = r["photo"]["nuisance"], r["photo"]
        if not nu:
            continue
        off, sg = nu.get("light_offset_cam_mm"), p.get("light_offset_sigma") or [None, None]
        be = nu.get("beam_exponent")
        lf = nu.get("lens_falloff") or [0.0, 0.0]
        rho = nu.get("ambient_to_torch_ratio")
        out.append(f"| {r['label']} | {p['status']} | {off[0]:+.1f}, {off[1]:+.1f} ({sg[0]}, {sg[1]}) | {be:.2f}, {p.get('beam_m2') or 0.0:+.2f} | {lf[0]:+.2f}, {lf[1]:+.2f} | {'-' if rho is None else f'{rho:.3f}'} | {nu.get('tone_gamma'):.3f} |")
    return "\n".join(out) + "\n"


def runtime(rows: list[dict[str, Any]]) -> str:
    by_n: dict[int, list[float]] = {}
    for r in rows:
        if r["photo"]["n_views_used"]:
            by_n.setdefault(int(r["photo"]["n_views_used"]), []).append(float(r["photo"]["runtime_s"]))
    pose = [r["time_pose_s"] / max(r.get("n_photos", r["n_views"]), 1) for r in rows if r.get("pose_rms_px") is not None]
    base = [b["time_s"] for r in rows for b in r["baseline"].values()]
    out = ["### Runtime\n", "Single process on the machine that ran this script (4 cores, shared with other jobs); model evaluations are NumPy on one core.\n",
           "| step | n | median s | max s |", "|---|---|---|---|"]
    for n in sorted(by_n):
        out.append(f"| photometric analysis, {n} torch-on views in the fit | {len(by_n[n])} | {np.median(by_n[n]):.2f} | {np.max(by_n[n]):.2f} |")
    if pose:
        out.append(f"| tag detection + bundle adjustment, per photo | {len(pose)} | {np.median(pose):.2f} | {np.max(pose):.2f} |")
    if base:
        out.append(f"| edge-only baseline through the session service (register + propose) | {len(base)} | {np.median(base):.2f} | {np.max(base):.2f} |")
    return "\n".join(out) + "\n"


def write_doc(text: str) -> None:
    """Replace the generated block of the write-up with ``text``."""
    doc = DOC.read_text() if DOC.exists() else f"{BEGIN}\n{END}\n"
    if BEGIN in doc and END in doc:
        DOC.write_text(doc[: doc.index(BEGIN) + len(BEGIN)] + "\n\n" + text + "\n" + doc[doc.index(END):])
        print(f"tables written to {DOC}")
    else:
        print(f"markers not found in {DOC}: nothing written")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="", help="comma separated group letters (default: all)")
    ap.add_argument("--jobs", type=int, default=2, help="processes used for rendering (default 2)")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    ap.add_argument("--out", default=str(ROOT / ".pytest_cache" / "photometric_ablation.json"), help="where the raw results go (JSON)")
    ap.add_argument("--write-doc", action="store_true", help=f"replace the generated block in {DOC.relative_to(ROOT)}")
    ap.add_argument("--from-json", default="", help="rebuild the tables from a previous --out file instead of running anything")
    args = ap.parse_args()
    if args.from_json:
        saved = json.loads(Path(args.from_json).read_text())
        text = tables(saved["rows"]) + saved.get("footer", "")
        print(text)
        if args.write_doc:
            write_doc(text)
        return
    cases = build_cases()
    if args.groups:
        want = {g.strip().upper() for g in args.groups.split(",")}
        cases = [c for c in cases if c.group in want]
    if args.list:
        for c in cases:
            print(f"{c.key:34s} {c.label}")
        print(len(cases), "cases")
        return
    t_all = time.perf_counter()
    jobs: dict[str, tuple[Scene, CameraSpec, int]] = {}
    for c in cases:
        for j in c.renders():
            jobs[repr(j)] = j
    print(f"{len(cases)} cases, {len(jobs)} photos (cached renders are reused)", flush=True)
    t0 = time.perf_counter()
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            pool.map(_render_job, list(jobs.values()), chunksize=1)
    else:
        for j in jobs.values():
            _render_job(j)
    t_render = time.perf_counter() - t0
    print(f"photos ready in {t_render:.0f} s", flush=True)
    rows = []
    for k, c in enumerate(cases):
        t0 = time.perf_counter()
        row = run_case(c)
        rows.append(row)
        p = row["photo"]
        print(f"[{k + 1}/{len(cases)}] {c.key:32s} {p['status']:11s} {_photo_cell(p):60s} {time.perf_counter() - t0:5.1f} s  {','.join(p['reasons'])}", flush=True)
    total = time.perf_counter() - t_all
    footer = f"\nScript run: {len(rows)} cases, {len(jobs)} photos; rendering {t_render:.0f} s (cached photos are reused), everything else {total - t_render:.0f} s.\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)  # raw results first: the tables can be rebuilt from them with --from-json
    out.write_text(json.dumps({"rows": rows, "render_s": t_render, "total_s": total, "footer": footer}, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o), indent=1))
    text = tables(rows)
    print("\n" + text + footer)
    print(f"raw results: {out}")
    if args.write_doc:
        write_doc(text + footer)


if __name__ == "__main__":
    main()
