"""Video ingestion: probing, candidate frames and deterministic frame selection.

A phone video is a long run of small, compressed, often motion-blurred frames.
Registration wants a handful of sharp frames that all show the tags and that
were taken from clearly different positions.  This module picks them.

Pipeline
--------
1. ``analyse_video`` decodes the file once.  Time is cut into windows; the
   sharpest frame of each window becomes a *candidate* (so a candidate is never
   the blurred middle of a hand movement when a sharper neighbour exists).
   Tags are located in every candidate with the quick detector.
2. ``estimate_viewpoints`` gives each candidate an approximate camera position
   relative to the tags.  The focal length of a browser/phone video is not
   known, so a typical one is assumed: the positions are only used to *rank
   viewpoint diversity* — never for measurement.
3. ``select_frames`` chooses up to ``k_max`` frames greedily: start from the
   best frame, then repeatedly add the candidate that is farthest (in viewing
   direction) from everything chosen so far, weighted by its quality.

Everything is deterministic: no random numbers, ties are broken by frame
index, so the same file always yields the same frames.

Selected frames then go through the normal photo path (full tag measurement);
the video itself is never stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from ..errors import LimitExceededError, OperationCancelled, UnsupportedMediaError
from ..geometry.homography import apply_h, dlt_homography, pose_from_homography
from .imaging import to_gray
from .markers import MarkerConfig, detect_tags, tag_plane_points

# --------------------------------------------------------------------------
# container sniffing
# --------------------------------------------------------------------------
_MP4_FIRST_ATOMS = (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot")


def sniff_video_type(head: bytes) -> str | None:
    """Container family from the first bytes: ``mp4`` (MP4/MOV), ``webm`` or ``avi``."""
    if len(head) >= 12 and head[4:8] in _MP4_FIRST_ATOMS:
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"  # Matroska / WebM
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    return None


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class VideoInfo:
    width: int = 0  # as decoded (after any rotation applied by the decoder)
    height: int = 0
    fps: float = 0.0
    fps_assumed: bool = False
    frames_decoded: int = 0
    duration_s: float = 0.0
    rotation_meta_deg: float = 0.0
    auto_oriented: bool = False
    codec: str = ""
    container: str = ""
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height, "fps": round(self.fps, 3), "fps_assumed": self.fps_assumed,
            "frames_decoded": self.frames_decoded, "duration_s": round(self.duration_s, 3),
            "rotation_meta_deg": self.rotation_meta_deg, "auto_oriented": self.auto_oriented,
            "codec": self.codec, "container": self.container, "truncated": self.truncated,
        }


@dataclass
class VideoCandidate:
    index: int
    time_s: float
    sharpness: float
    mean: float
    clipped_high: float
    clipped_low: float
    jpeg: bytes = b""
    #: tag id -> 4x2 corner pixels (quick detector, full-frame pixel coordinates)
    tags: dict[int, np.ndarray] = field(default_factory=dict)
    #: tags found but not usable in this frame, with the reason
    tag_issues: dict[int, list[str]] = field(default_factory=dict)
    min_tag_edge_px: float | None = None
    #: approximate camera centre in the rough tag-layout frame (nominal mm); ranking only
    center: np.ndarray | None = None
    #: rough layout-plane -> image homography behind ``center``
    H: np.ndarray | None = None
    #: intrinsics to use for this candidate instead of the assumed field of view
    K: np.ndarray | None = None
    reasons: list[str] = field(default_factory=list)  # why it cannot be used at all
    quality: float = 0.0

    @property
    def usable(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "time_s": round(self.time_s, 3), "sharpness": round(self.sharpness, 2),
            "mean": round(self.mean, 4), "clipped_high": round(self.clipped_high, 4),
            "tags": sorted(self.tags), "tag_issues": {str(k): v for k, v in sorted(self.tag_issues.items())},
            "min_tag_edge_px": None if self.min_tag_edge_px is None else round(self.min_tag_edge_px, 1),
            "quality": round(self.quality, 4), "reasons": list(self.reasons),
        }


@dataclass
class VideoAnalysis:
    info: VideoInfo
    candidates: list[VideoCandidate]
    window_s: float


@dataclass
class SelectionConfig:
    k_min: int = 3
    k_max: int = 9
    #: a frame softer than this fraction of the video's sharp frames is not used
    min_relative_sharpness: float = 0.35
    max_clipped_high: float = 0.25
    min_mean: float = 0.04
    #: tags smaller than this (pixels per edge) are too small to measure well
    min_tag_edge_px: float = 28.0
    #: stop adding frames when the best remaining one is closer than this to a chosen view
    min_new_angle_deg: float = 2.0
    #: horizontal field of view assumed for ranking viewpoints (never for measurement)
    assumed_hfov_deg: float = 66.0


@dataclass
class Selection:
    selected: list[int]  # indices into ``candidates`` (time order)
    notes: dict[int, str]  # candidate index -> why chosen
    rejected: dict[int, list[str]]  # candidate index -> reasons
    spread_deg: float | None
    reference_tags: list[int]
    flags: list[str]
    hints: list[str]

    def to_dict(self, cands: list[VideoCandidate]) -> dict[str, Any]:
        return {
            "selected": [{"candidate": i, "time_s": round(cands[i].time_s, 3), "frame_index": cands[i].index, "why": self.notes.get(i, "")} for i in self.selected],
            "rejected": [{"candidate": i, "time_s": round(cands[i].time_s, 3), "reasons": r} for i, r in sorted(self.rejected.items())],
            "spread_deg": None if self.spread_deg is None else round(self.spread_deg, 2),
            "reference_tags": self.reference_tags, "flags": self.flags, "hints": self.hints,
        }


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
def _sharpness_small(bgr: np.ndarray, work: int = 480) -> tuple[float, np.ndarray]:
    """Variance of the Laplacian at a fixed working size (comparable across resolutions)."""
    g = to_gray(bgr)
    s = work / float(max(g.shape))
    if s < 1.0:
        g = cv2.resize(g, (max(8, int(round(g.shape[1] * s))), max(8, int(round(g.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var()), g


def _fourcc_text(v: float) -> str:
    try:
        i = int(v)
        s = "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4))
        return s if s.isprintable() else ""
    except (ValueError, OverflowError):
        return ""


def analyse_video(
    path: str,
    *,
    max_candidates: int = 90,
    max_seconds: float = 60.0,
    max_pixels: int = 9_000_000,
    jpeg_quality: int = 95,
    marker_cfg: MarkerConfig | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> VideoAnalysis:
    """Decode ``path`` once and return one candidate per time window.

    ``max_seconds`` bounds the work: a longer recording is analysed up to that
    point and flagged ``truncated`` (the caller tells the operator).
    """
    with open(path, "rb") as fh:
        container = sniff_video_type(fh.read(16))
    if container is None:
        raise UnsupportedMediaError(
            "unsupported_video", "This file is not an MP4/MOV, WebM or AVI video.",
            suggestions=["Record with the phone camera (MOV/MP4) or use the in-app recorder."],
        )
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise UnsupportedMediaError(
            "video_not_decodable", "The video could not be opened by the decoder.",
            suggestions=["Record again in the phone's 'Most compatible' (H.264) format, or upload photos instead."],
        )
    try:
        info = VideoInfo(container=container)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (1.0 <= fps <= 480.0) or not math.isfinite(fps):
            fps, info.fps_assumed = 30.0, True
        info.fps = fps
        info.rotation_meta_deg = float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
        info.auto_oriented = bool(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))
        info.codec = _fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        n_reported = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        dur_reported = n_reported / fps if 1.0 <= n_reported < 1e7 else 0.0  # live recordings often report nothing useful
        expect = min(dur_reported, max_seconds) if dur_reported > 0 else 0.0
        window = max(0.25, expect / max_candidates) if expect > 0 else 0.25

        cfg = marker_cfg or MarkerConfig()
        cands: list[VideoCandidate] = []
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None  # sharpness, index, time, frame, small gray
        win_id = 0
        idx = -1
        last_t = 0.0

        def close_window() -> None:
            nonlocal best
            if best is None:
                return
            sharp, i, t, frame, small = best
            best = None
            hist = cv2.calcHist([small], [0], None, [256], [0, 256]).ravel()
            tot = float(hist.sum()) or 1.0
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            c = VideoCandidate(
                index=i, time_s=t, sharpness=sharp, mean=float(small.mean()) / 255.0,
                clipped_high=float(hist[250:].sum()) / tot, clipped_low=float(hist[:6].sum()) / tot,
                jpeg=buf.tobytes() if ok else b"",
            )
            edges = []
            for d in detect_tags(frame, cfg=cfg, refine=False):
                if d.accepted:
                    c.tags[d.tag_id] = np.asarray(d.corners_px, dtype=float)
                    edges.append(d.min_edge_px)
                else:
                    c.tag_issues.setdefault(d.tag_id, []).extend(d.reject_reasons)
            c.min_tag_edge_px = float(min(edges)) if edges else None
            cands.append(c)

        while True:
            if cancelled is not None and cancelled():
                raise OperationCancelled("cancelled", "Video analysis was cancelled.")
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx += 1
            if idx == 0:
                info.height, info.width = int(frame.shape[0]), int(frame.shape[1])
                if info.width * info.height > max_pixels:
                    raise LimitExceededError(
                        "video_resolution_too_large", f"Video frames of {info.width}x{info.height} exceed the limit.",
                        suggestions=["Record at 4K or lower."],
                    )
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            t = t_ms / 1000.0 if (t_ms > 0 and not info.fps_assumed) else idx / fps
            if t < last_t:  # non-monotonic container timestamps: fall back to counting
                t = idx / fps
            last_t = t
            if t > max_seconds:
                info.truncated = True
                break
            w = int(t / window)
            if w != win_id:
                close_window()
                win_id = w
                if len(cands) >= max_candidates:
                    # recording is longer than expected: keep the sharper of each
                    # neighbouring pair and double the window (deterministic)
                    merged = []
                    for a in range(0, len(cands) - 1, 2):
                        x, y = cands[a], cands[a + 1]
                        merged.append(x if x.sharpness >= y.sharpness else y)
                    if len(cands) % 2:
                        merged.append(cands[-1])
                    cands = merged
                    window *= 2.0
                    win_id = int(t / window)
            sharp, small = _sharpness_small(frame)
            if best is None or sharp > best[0]:
                best = (sharp, idx, t, frame, small)
            if progress is not None and idx % 15 == 0:
                denom = expect if expect > 0 else max_seconds
                progress(min(0.98, t / max(denom, 1e-6)), "Reading video")
        close_window()
        info.frames_decoded = idx + 1
        info.duration_s = last_t
        if info.frames_decoded == 0:
            raise UnsupportedMediaError(
                "video_not_decodable", "No frame could be decoded from this video.",
                suggestions=["Record again in the phone's 'Most compatible' (H.264) format, or upload photos instead."],
            )
        return VideoAnalysis(info=info, candidates=cands, window_s=window)
    finally:
        cap.release()


# --------------------------------------------------------------------------
# approximate viewpoints (ranking only)
# --------------------------------------------------------------------------
def estimate_viewpoints(cands: list[VideoCandidate], width: int, height: int, assumed_hfov_deg: float = 66.0, tag_size_mm: float = 50.0) -> list[int]:
    """Fill ``center`` of every candidate that sees tags of a rough common layout.

    The layout (tag positions on the panel) is taken from the candidate that
    sees the most tags at the largest size, in the frame of its biggest tag;
    it is extended with tags that only appear in other candidates.  Returns
    the tag ids of the layout.  All of this assumes a focal length and nominal
    tag sizes, which is good enough to tell "same place" from "moved 15
    degrees" and is used for nothing else.
    """
    f = 0.5 * width / math.tan(math.radians(assumed_hfov_deg) / 2.0)
    K = np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    with_tags = [c for c in cands if c.tags]
    if not with_tags:
        return []
    obj = tag_plane_points(tag_size_mm)

    def seed_score(c: VideoCandidate) -> tuple[float, float, int]:
        return (float(len(c.tags)), float(c.min_tag_edge_px or 0.0) * c.sharpness, -c.index)

    seed = max(with_tags, key=seed_score)
    ref_id = max(seed.tags, key=lambda i: (float(np.mean(np.linalg.norm(np.roll(seed.tags[i], -1, axis=0) - seed.tags[i], axis=1))), -i))
    H_seed = dlt_homography(obj, seed.tags[ref_id])
    layout: dict[int, np.ndarray] = {i: apply_h(np.linalg.inv(H_seed), q) for i, q in seed.tags.items()}
    layout[ref_id] = obj.copy()

    def solve(c: VideoCandidate) -> np.ndarray | None:
        ids = [i for i in sorted(c.tags) if i in layout]
        if not ids:
            return None
        P = np.concatenate([layout[i] for i in ids])
        Q = np.concatenate([c.tags[i] for i in ids])
        try:
            return dlt_homography(P, Q)
        except Exception:
            return None

    # extend the layout with tags unseen by the seed (a few passes are enough)
    for _ in range(3):
        grew = False
        for c in with_tags:
            new = [i for i in c.tags if i not in layout]
            if not new:
                continue
            H = solve(c)
            if H is None:
                continue
            Hi = np.linalg.inv(H)
            for i in sorted(new):
                layout[i] = apply_h(Hi, c.tags[i])
                grew = True
        if not grew:
            break
    centroid = np.mean(np.concatenate(list(layout.values())), axis=0)
    for c in with_tags:
        H = solve(c)
        if H is None:
            continue
        try:
            T = pose_from_homography(c.K if c.K is not None else K, H)
        except Exception:
            continue
        c.H = H
        ctr = T.center
        if not np.all(np.isfinite(ctr)) or abs(ctr[2]) < 1e-6:
            continue
        ctr = ctr.copy()
        ctr[:2] -= centroid
        ctr[2] = abs(ctr[2])
        c.center = ctr
    return sorted(layout)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = a / max(np.linalg.norm(a), 1e-9), b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(np.clip(float(ua @ ub), -1.0, 1.0))))


def _view_distance(a: VideoCandidate, b: VideoCandidate) -> float:
    """Viewpoint separation in "degrees": direction change plus a term for moving closer/farther."""
    assert a.center is not None and b.center is not None
    ang = _angle_deg(a.center, b.center)
    ra, rb = float(np.linalg.norm(a.center)), float(np.linalg.norm(b.center))
    return ang + 12.0 * abs(math.log(max(ra, 1e-6) / max(rb, 1e-6)))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def select_frames(an: VideoAnalysis, cfg: SelectionConfig | None = None, capacity: int | None = None) -> Selection:
    """Deterministic choice of registration frames (see the module docstring)."""
    cfg = cfg or SelectionConfig()
    cands = an.candidates
    k_max = cfg.k_max if capacity is None else max(0, min(cfg.k_max, capacity))
    flags: list[str] = []
    hints: list[str] = []
    rejected: dict[int, list[str]] = {}
    notes: dict[int, str] = {}
    if not cands or k_max == 0:
        return Selection([], {}, {}, None, [], ["no_candidates"] if not cands else ["no_capacity"], hints)

    sharp = np.array([c.sharpness for c in cands])
    ref_sharp = float(np.percentile(sharp, 90))
    all_ids = sorted({i for c in cands for i in c.tags})
    layout_ids = estimate_viewpoints(cands, an.info.width, an.info.height, cfg.assumed_hfov_deg)

    for k, c in enumerate(cands):
        c.reasons = []
        rel = c.sharpness / max(ref_sharp, 1e-9)
        if rel < cfg.min_relative_sharpness:
            c.reasons.append("motion_blur")
        if c.clipped_high > cfg.max_clipped_high:
            c.reasons.append("overexposed")
        if c.mean < cfg.min_mean:
            c.reasons.append("too_dark")
        if all_ids and not c.tags:
            c.reasons.append("no_tag_in_view" if not c.tag_issues else "tags_not_usable")
        if c.tags and (c.min_tag_edge_px or 0.0) < cfg.min_tag_edge_px:
            c.reasons.append("tags_too_small")
        cover = len(c.tags) / max(len(all_ids), 1) if all_ids else 1.0
        size = min(1.0, (c.min_tag_edge_px or cfg.min_tag_edge_px) / 60.0) if c.tags else 1.0
        expo = 1.0 - min(1.0, c.clipped_high / max(cfg.max_clipped_high, 1e-9)) * 0.5
        c.quality = float(min(rel, 1.2) / 1.2 * (0.35 + 0.65 * cover) * (0.5 + 0.5 * size) * expo)
        if c.reasons:
            rejected[k] = list(c.reasons)

    usable = [k for k, c in enumerate(cands) if c.usable]
    if not all_ids:
        flags.append("no_tags_seen")
        hints.append("No AprilTag was found anywhere in the video: make sure the tags are in view and in focus.")
    if not usable:
        flags.append("no_usable_frame")
        if any("motion_blur" in r for r in rejected.values()):
            hints.append("Every frame is blurred: move more slowly or add light.")
        return Selection([], notes, rejected, None, layout_ids, flags, hints)

    posed = [k for k in usable if cands[k].center is not None]
    chosen: list[int] = []
    if len(posed) >= 1:
        first = max(posed, key=lambda k: (cands[k].quality, -k))
        chosen.append(first)
        notes[first] = f"best frame: {len(cands[first].tags)} tag(s), sharpest in its {an.window_s:.2f} s window"
        while len(chosen) < k_max:
            best_k, best_s, best_d = None, -1.0, 0.0
            for k in posed:
                if k in chosen:
                    continue
                d = min(_view_distance(cands[k], cands[j]) for j in chosen)
                s = d * (0.4 + 0.6 * cands[k].quality)
                if s > best_s + 1e-12:
                    best_k, best_s, best_d = k, s, d
            if best_k is None or best_d < cfg.min_new_angle_deg:
                break
            chosen.append(best_k)
            notes[best_k] = f"{best_d:.0f}° from the nearest chosen view, {len(cands[best_k].tags)} tag(s)"
    # fill up to the minimum (or everything, if nothing could be posed) evenly in time
    if len(chosen) < min(cfg.k_min, k_max):
        pool = [k for k in usable if k not in chosen]
        want = min(cfg.k_min, k_max) - len(chosen)
        if not posed:
            want = min(k_max, max(cfg.k_min, min(len(pool), 5)))
        if pool and want > 0:
            times = np.array([cands[k].time_s for k in pool])
            targets = np.linspace(times.min(), times.max(), num=min(want, len(pool)) + 2)[1:-1] if len(pool) > want else times
            for tt in targets:
                rest = [k for k in pool if k not in chosen]
                if not rest:
                    break
                k = min(rest, key=lambda k: (abs(cands[k].time_s - tt) - 0.3 * an.window_s * cands[k].quality, k))
                chosen.append(k)
                notes[k] = "added to reach the minimum number of frames (no new viewpoint)"
    chosen = sorted(set(chosen), key=lambda k: cands[k].time_s)

    spread = None
    pc = [cands[k].center for k in chosen if cands[k].center is not None]
    if len(pc) >= 2:
        spread = max(_angle_deg(a, b) for i, a in enumerate(pc) for b in pc[i + 1:])
        if spread < 8.0:
            flags.append("narrow_baseline")
            hints.append("The camera hardly moved during the video: sweep sideways around the panel (20° or more).")
    if len(chosen) < cfg.k_min:
        flags.append("few_frames")
        hints.append(f"Only {len(chosen)} usable frame(s): record a longer, slower sweep.")
    if an.info.truncated:
        flags.append("truncated")
        hints.append("Only the first part of the recording was analysed (length limit).")
    lost = [i for i in all_ids if not all(i in cands[k].tags for k in chosen)]
    if lost and chosen:
        hints.append("Not every tag is visible in every chosen frame (IDs " + ", ".join(str(i) for i in lost) + "): keep all tags in view during the sweep.")
    return Selection(chosen, notes, rejected, spread, layout_ids, flags, hints)
