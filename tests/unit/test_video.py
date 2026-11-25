"""Video ingestion and deterministic frame selection (prompt 02)."""

import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from lumen.errors import UnsupportedMediaError
from lumen.vision.video import SelectionConfig, analyse_video, select_frames, sniff_video_type
from tests.conftest import sweep_video_frames, write_video


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    frames, cams, sharp = sweep_video_frames()
    path = str(tmp_path_factory.mktemp("video") / "sweep.mp4")
    write_video(path, frames, fps=12.0)
    return path, frames, cams, sharp


def test_container_sniffing():
    assert sniff_video_type(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00") == "mp4"
    assert sniff_video_type(b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00") == "mp4"
    assert sniff_video_type(b"\x1a\x45\xdf\xa3" + b"\x00" * 12) == "webm"
    assert sniff_video_type(b"RIFF\x00\x00\x00\x00AVI LIST") == "avi"
    assert sniff_video_type(b"\xff\xd8\xff\xe0" + b"\x00" * 12) is None  # a JPEG is not a video
    assert sniff_video_type(b"") is None


def test_not_a_video_is_refused(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"this is not a video at all, just text")
    with pytest.raises(UnsupportedMediaError):
        analyse_video(str(p))
    # right signature, garbage content: the decoder cannot open it
    q = tmp_path / "y.mp4"
    q.write_bytes(b"\x00\x00\x00\x1cftypisom" + os.urandom(400))
    with pytest.raises(UnsupportedMediaError):
        analyse_video(str(q))


def test_candidates_are_the_sharp_frames_of_each_window(sweep):
    path, frames, cams, sharp = sweep
    an = analyse_video(path)
    assert an.info.width == 1280 and an.info.height == 720 and an.info.frames_decoded == len(frames)
    assert abs(an.info.fps - 12.0) < 0.01 and not an.info.fps_assumed
    # 8 held views of 0.25 s each -> one window per view, and its sharpest frame is the unblurred one
    assert [c.index for c in an.candidates] == sharp
    assert all(set(c.tags) == {3, 7, 11, 20} for c in an.candidates)


def test_selection_is_deterministic_and_spreads_the_viewpoints(sweep):
    path, frames, cams, sharp = sweep
    a = analyse_video(path)
    b = analyse_video(path)
    sa, sb = select_frames(a, SelectionConfig(k_max=5)), select_frames(b, SelectionConfig(k_max=5))
    assert sa.selected == sb.selected and len(sa.selected) == 5
    chosen = [a.candidates[k].index for k in sa.selected]
    views = [sharp.index(i) for i in chosen]
    # both ends of the arc are chosen, and the views are spread rather than adjacent
    assert 0 in views and 7 in views
    assert sa.spread_deg is not None and 30.0 < sa.spread_deg < 60.0  # true arc: 44 degrees
    assert not sa.flags and sa.reference_tags == [3, 7, 11, 20]


def test_approximate_viewpoints_follow_the_true_camera_positions(sweep):
    path, frames, cams, sharp = sweep
    an = analyse_video(path)
    select_frames(an)
    est = np.array([c.center for c in an.candidates])
    assert all(c.center is not None for c in an.candidates)
    # true viewing directions relative to the panel centre, as angles along the arc
    true = np.array([cams[i].C for i in sharp]) - np.array([277.5, 0.0, 307.5])  # tag centroid on the panel
    def pair_angles(P):
        U = P / np.linalg.norm(P, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(U @ U.T, -1, 1)))
    err = np.abs(pair_angles(est) - pair_angles(true))
    # an assumed focal length distorts the angles a little; the ranking survives
    assert err.max() < 8.0


def test_capacity_and_minimum_are_respected(sweep):
    path, *_ = sweep
    an = analyse_video(path)
    assert len(select_frames(an, capacity=2).selected) == 2
    assert select_frames(an, capacity=0).selected == []
    sel = select_frames(an, SelectionConfig(k_max=9))
    assert 3 <= len(sel.selected) <= 8


def test_a_static_video_is_flagged_and_not_padded_with_duplicates(tmp_path):
    frames, cams, sharp = sweep_video_frames(n_views=1, hold=18)
    frames = [frames[sharp[0]].copy() for _ in range(18)]  # 1.5 s without any motion or blur
    path = write_video(str(tmp_path / "static.mp4"), frames, fps=12.0)
    an = analyse_video(path)
    sel = select_frames(an)
    assert "narrow_baseline" in sel.flags and any("hardly moved" in h for h in sel.hints)
    assert len(sel.selected) == 3  # the minimum, each labelled as adding no viewpoint
    assert sum("no new viewpoint" in sel.notes[k] for k in sel.selected) == 2


def test_blurred_and_tagless_frames_are_rejected_with_reasons(tmp_path):
    frames, cams, sharp = sweep_video_frames(n_views=4, hold=3)
    blank = np.full_like(frames[0], 128)
    cv2.randn(blank, 128, 6)
    seq = frames[:6] + [blank] * 3 + frames[6:]
    path = write_video(str(tmp_path / "gap.mp4"), seq, fps=12.0)
    an = analyse_video(path)
    sel = select_frames(an)
    reasons = {r for rs in sel.rejected.values() for r in rs}
    assert "no_tag_in_view" in reasons
    assert all(an.candidates[k].tags for k in sel.selected)


def test_video_without_any_tag_still_reports_honestly(tmp_path):
    rng = np.random.default_rng(3)
    frames = [np.clip(rng.normal(120, 30, (360, 640, 3)), 0, 255).astype(np.uint8) for _ in range(12)]
    path = write_video(str(tmp_path / "noise.mp4"), frames, fps=12.0)
    sel = select_frames(analyse_video(path))
    assert "no_tags_seen" in sel.flags and any("No AprilTag" in h for h in sel.hints)


def test_length_limit_truncates_and_says_so(sweep):
    path, frames, *_ = sweep
    an = analyse_video(path, max_seconds=1.0)
    assert an.info.truncated and an.info.frames_decoded <= 14
    assert "truncated" in select_frames(an).flags


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg command not available to build phone-like test files")
@pytest.mark.parametrize("codec,ext,transpose,rotation", [("libx264", "mov", "2", "-90"), ("libx265", "mp4", "1", "90")])
def test_phone_style_files_decode_and_rotation_metadata_is_applied(sweep, tmp_path, codec, ext, transpose, rotation):
    """H.264-in-MOV and HEVC-in-MP4 stored sideways with a display rotation, as phones write video.

    The frames are stored rotated by 90 degrees and the stream is tagged with
    the opposite display rotation; a correct decoder returns them upright.
    """
    path, frames, cams, sharp = sweep
    stored = str(tmp_path / "stored_sideways.mp4")
    out = str(tmp_path / f"phone.{ext}")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"transpose={transpose}", "-c:v", codec, "-pix_fmt", "yuv420p", "-crf", "16", stored], capture_output=True, text=True)
    if r.returncode == 0:
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", rotation, "-i", stored, "-c", "copy", out], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg could not build the test file: {r.stderr[:200]}")
    raw = cv2.VideoCapture(stored)
    assert (int(raw.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (720, 1280)  # really stored sideways
    raw.release()
    an = analyse_video(out)
    ref = analyse_video(path)
    assert an.info.frames_decoded == len(frames)
    assert (an.info.width, an.info.height) == (1280, 720), an.info.to_dict()
    assert an.info.rotation_meta_deg in (90.0, 270.0) and an.info.auto_oriented
    # upright and unmirrored: every tag is where it is in the original video
    a, b = an.candidates[0], ref.candidates[0]
    assert set(a.tags) == set(b.tags) == {3, 7, 11, 20}
    for i in a.tags:
        assert np.max(np.abs(a.tags[i] - b.tags[i])) < 3.0
    sel = select_frames(an)
    assert len(sel.selected) >= 3 and all(len(an.candidates[k].tags) == 4 for k in sel.selected)
