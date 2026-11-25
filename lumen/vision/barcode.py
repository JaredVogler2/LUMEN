"""Barcode / QR reading for identifying an operation, panel or placard (prompt 08).

A scanned code is *text used for a library lookup* and nothing else.  Its
position and size in the picture are returned so the interface can outline
what was read; they are never used as a scale reference — a printed code has
no certified dimensions.

Decoding uses zxing-cpp (1-D and 2-D symbologies) and falls back to OpenCV's
QR detector when zxing-cpp is unavailable or finds nothing.  Symbols whose
content is not plain text are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .imaging import to_gray

MAX_TEXT = 256


@dataclass
class CodeRead:
    text: str
    format: str
    corners: list[list[float]]  # TL, TR, BR, BL in image pixels, as reported by the decoder
    decoder: str

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "format": self.format, "corners": self.corners, "decoder": self.decoder}


def decoders_available() -> dict[str, bool]:
    try:
        import zxingcpp  # noqa: F401

        z = True
    except Exception:
        z = False
    return {"zxing": z, "opencv_qr": hasattr(cv2, "QRCodeDetector")}


def _clean(text: str) -> str | None:
    """Printable, single-line text of a sane length, or ``None``."""
    t = " ".join(str(text).split())
    if not t or len(t) > MAX_TEXT:
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in t):
        return None
    return t


def _zxing(gray: np.ndarray) -> list[CodeRead]:
    import zxingcpp

    out = []
    for r in zxingcpp.read_barcodes(gray, try_rotate=True, try_downscale=True, try_invert=True):
        if not r.valid:
            continue
        ct = str(getattr(r, "content_type", "Text"))
        if "Binary" in ct or "Unknown" in ct:
            continue
        t = _clean(r.text)
        if t is None:
            continue
        p = r.position
        out.append(CodeRead(t, str(r.format), [[float(q.x), float(q.y)] for q in (p.top_left, p.top_right, p.bottom_right, p.bottom_left)], "zxing"))
    return out


def _opencv_qr(gray: np.ndarray) -> list[CodeRead]:
    det = cv2.QRCodeDetector()
    out = []
    try:
        ok, texts, pts, _ = det.detectAndDecodeMulti(gray)
    except cv2.error:
        return out
    if not ok or pts is None:
        return out
    for text, quad in zip(texts, pts):
        t = _clean(text)
        if t is None:
            continue
        out.append(CodeRead(t, "QR Code", np.asarray(quad, dtype=float).reshape(4, 2).tolist(), "opencv"))
    return out


def _rotated(gray: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The image rotated about its centre (canvas enlarged) and the 2x3 matrix back to the original."""
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(w * c + h * s_)), int(np.ceil(w * s_ + h * c))
    M[0, 2] += (nw - w) / 2.0
    M[1, 2] += (nh - h) / 2.0
    fill = float(np.median(gray))
    rot = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return rot, cv2.invertAffineTransform(M)


def _map(cs: list[CodeRead], Minv: np.ndarray) -> list[CodeRead]:
    for c in cs:
        P = np.asarray(c.corners, dtype=float)
        c.corners = (P @ Minv[:, :2].T + Minv[:, 2]).tolist()
    return cs


def _cascade(gray: np.ndarray, thorough: bool, rotations: bool) -> list[CodeRead]:
    """First attempt that reads something: as is, equalised, enlarged, rotated."""
    found = _zxing(gray)
    if found:
        return found
    found = _zxing(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray))
    if found or not thorough:
        return found
    if max(gray.shape) <= 1600:
        big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        found = _map(_zxing(big), np.array([[0.5, 0.0, -0.25], [0.0, 0.5, -0.25]]))
        if found:
            return found
    if not rotations:
        return []
    work, k = gray, 1.0
    if max(gray.shape) > 1400:  # rotation attempts on a reduced copy keep the scan quick
        k = 1400.0 / max(gray.shape)
        work = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    for ang in (15, -15, 30, -30, 45, -45, 60, -60, 75, -75):
        rot, Minv = _rotated(work, ang)
        got = _zxing(rot)
        if got:
            return _map(_map(got, Minv), np.array([[1.0 / k, 0.0, 0.5 / k - 0.5], [0.0, 1.0 / k, 0.5 / k - 0.5]]))
    return []


def read_codes(image: np.ndarray, *, use_zxing: bool = True, thorough: bool = True) -> list[CodeRead]:
    """All readable text codes in an 8-bit image, de-duplicated, in reading order.

    Attempts, stopping at the first that finds something: the picture as it
    is; locally equalised (uneven light); and, when ``thorough``, enlarged
    (small symbols) and rotated in 15 degree steps (a long 1-D barcode is only
    readable along a line that crosses all its bars, so a tilted phone needs
    this).  What was read is then covered and the picture is searched again,
    because the decoder often stops at one symbol.  ``thorough=False`` is the
    quick mode for a live scanning loop.

    Practical limit seen on synthetic labels: about 3 pixels per narrowest
    bar/module after blur; below that only a level, sharp code is read.
    """
    gray = to_gray(image)
    found: list[CodeRead] = []
    zx = use_zxing and decoders_available()["zxing"]
    rest = gray
    for rnd in range(5):
        got = _cascade(rest, thorough, rotations=(rnd == 0)) if zx else []
        if not got and (thorough or not zx):
            got = _opencv_qr(rest)  # a second decoder: reads some QR codes the first one misses
        got = [c for c in got if (c.text, c.format) not in {(f.text, f.format) for f in found}]
        if not got:
            break
        found.extend(got)
        rest = rest.copy()
        fill = int(np.median(rest))
        for c in got:
            P = np.asarray(c.corners, dtype=np.float32)
            ctr = P.mean(axis=0)
            cv2.fillConvexPoly(rest, np.round(ctr + 1.3 * (P - ctr)).astype(np.int32), fill)
    seen: set[tuple[str, str]] = set()
    out = []
    for c in sorted(found, key=lambda c: (round(min(p[1] for p in c.corners) / 40.0), min(p[0] for p in c.corners))):
        key = (c.text, c.format)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
