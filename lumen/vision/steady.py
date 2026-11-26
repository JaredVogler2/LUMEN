"""Has a tag moved in the picture since an earlier frame?  (For a phone that stands still.)

During live placement the phone is on a stand.  The tags are then in the same
pixels frame after frame, and measuring them again each time costs most of a
frame's time and fails for every tag that an arm happens to cover.  Instead,
the picture around each tag is kept from the frame in which the tags were
measured (the *reference*), and a later frame is only asked one question per
tag: is this still the same picture, in the same place?

The answer is a least-squares fit of the reference patch to the new one with
four unknowns: a shift in x and y, a gain and an offset (the light may have
changed).  For shifts well under a pixel the fit is linear in the patch's
gradients, exact to a few hundredths of a pixel, and costs microseconds.  How
much of the new patch the fit explains (R squared) says whether it is the same
picture at all: a hand over the tag, a tag that was moved or a camera that was
moved by more than a pixel or two all leave most of it unexplained.

This verifies an existing registration; it never makes one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PatchRef:
    """The picture around one tag in the reference frame, prepared for comparing."""

    tag_id: int
    x0: int
    y0: int
    shape: tuple[int, int]
    #: the smoothed patch and its gradients, without the border ring
    ref: np.ndarray
    gx: np.ndarray
    gy: np.ndarray


_BORDER = 3
#: derivative of a smooth signal from five samples (exact for cubics): the two-sample difference underestimates the
#: slope of an edge that is only a pixel or two wide, and a shift computed with it comes out too large
_DERIV = np.array([1.0, -8.0, 0.0, 8.0, -1.0], dtype=np.float32) / 12.0


def _smooth(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(gray, dtype=np.float32), (0, 0), 1.0)


def _gradients(sm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.filter2D(sm, cv2.CV_32F, _DERIV.reshape(1, 5), borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(sm, cv2.CV_32F, _DERIV.reshape(5, 1), borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def _core(a: np.ndarray) -> np.ndarray:
    return a[_BORDER:-_BORDER, _BORDER:-_BORDER].astype(np.float64).ravel()


def make_patch(gray: np.ndarray, tag_id: int, corners_px: np.ndarray, margin: int = 8) -> PatchRef | None:
    """Keep the picture around a tag (its corners in raw pixels of ``gray``), or ``None`` if it is at the border
    of the frame or shows nothing that a shift in every direction would change."""
    c = np.asarray(corners_px, dtype=float).reshape(-1, 2)
    h, w = gray.shape[:2]
    x0, y0 = int(np.floor(c[:, 0].min())) - margin, int(np.floor(c[:, 1].min())) - margin
    x1, y1 = int(np.ceil(c[:, 0].max())) + margin + 1, int(np.ceil(c[:, 1].max())) + margin + 1
    if x0 < 1 or y0 < 1 or x1 > w - 1 or y1 > h - 1 or (x1 - x0) < 16 or (y1 - y0) < 16:
        return None
    sm = _smooth(gray[y0:y1, x0:x1])
    gx, gy = _gradients(sm)
    ref, gxc, gyc = _core(sm), _core(gx), _core(gy)
    # the two gradient directions must both be there: stripes along one direction cannot show a shift along it
    G = np.array([[gxc @ gxc, gxc @ gyc], [gxc @ gyc, gyc @ gyc]]) / ref.size
    ev = np.linalg.eigvalsh(G)
    if ev[0] < 1.0 or ev[0] < 0.02 * ev[1]:
        return None
    return PatchRef(int(tag_id), x0, y0, (y1 - y0, x1 - x0), ref, gxc, gyc)


def patch_shift(ref: PatchRef, gray: np.ndarray) -> tuple[float, float, float]:
    """Shift (dx, dy) in pixels of the picture at ``ref`` between the reference frame and ``gray``, and the
    share of the new patch that the shifted, re-lit reference explains (R squared; 1 = the same picture).

    The fit uses the mean of the two pictures' gradients, which makes it right to second order in the shift.
    Only small shifts are measured correctly (a pixel at most); a large one shows as a low R squared or as a
    shift far above any tolerance, which is all the caller needs to know.
    """
    h, w = ref.shape
    if ref.y0 + h > gray.shape[0] or ref.x0 + w > gray.shape[1]:
        return float("nan"), float("nan"), 0.0
    sm = _smooth(gray[ref.y0 : ref.y0 + h, ref.x0 : ref.x0 + w])
    cur = _core(sm)
    var = float(np.var(cur))
    if var < 1e-6:
        return float("nan"), float("nan"), 0.0
    cgx, cgy = _gradients(sm)
    # cur(p) = g * ref(p - d) + o   ~   g * ref(p) + o - grad . (g d),   grad: mean gradient of the two pictures
    A = np.stack([0.5 * (ref.gx + _core(cgx)), 0.5 * (ref.gy + _core(cgy)), ref.ref, np.ones_like(cur)], axis=1)
    try:
        sol = np.linalg.solve(A.T @ A, A.T @ cur)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), 0.0
    a, b, g, _ = (float(v) for v in sol)
    r2 = 1.0 - float(np.var(cur - A @ sol)) / var
    if abs(g) < 0.05:
        return float("nan"), float("nan"), 0.0
    # the mean gradient is that of a picture lit half-way between the two: scale by (1 + g) / 2
    return -a * 0.5 * (1.0 + g) / g, -b * 0.5 * (1.0 + g) / g, r2
