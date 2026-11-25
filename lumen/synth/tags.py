"""AprilTag 36h11 bit patterns for the synthetic renderer and printable sheets."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

#: Cells along one side of the black square (6 data cells + 1 black border cell each side).
CELLS = 8
#: Number of tags in the 36h11 family.
FAMILY_SIZE = 587


@lru_cache(maxsize=None)
def _dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


@lru_cache(maxsize=1024)
def tag_bits(tag_id: int) -> np.ndarray:
    """8x8 boolean array, ``True`` = white cell.  Row 0 is the top of the tag.

    The outer ring is the black border, so the array's outline is exactly the
    "detection edge" whose length is the tag size.
    """
    if not (0 <= int(tag_id) < FAMILY_SIZE):
        raise ValueError(f"tag id {tag_id} is not in AprilTag 36h11 (0..{FAMILY_SIZE - 1})")
    img = cv2.aruco.generateImageMarker(_dictionary(), int(tag_id), CELLS)
    bits = img > 127
    bits.setflags(write=False)
    return bits


def tag_image(tag_id: int, cell_px: int = 20, quiet_cells: float = 1.0) -> np.ndarray:
    """Greyscale image of a tag with a white quiet zone (for printing/tests)."""
    bits = tag_bits(tag_id)
    core = np.kron(bits.astype(np.uint8) * 255, np.ones((cell_px, cell_px), dtype=np.uint8))
    q = int(round(quiet_cells * cell_px))
    out = np.full((core.shape[0] + 2 * q, core.shape[1] + 2 * q), 255, dtype=np.uint8)
    out[q : q + core.shape[0], q : q + core.shape[1]] = core
    return out
