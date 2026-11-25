"""Pure-NumPy geometry primitives (no image processing, no I/O)."""

from .camera import CameraModel
from .homography import apply_h, dlt_homography, fit_rigid_2d
from .lines import Line2D, LineFit, fit_line_tls
from .se3 import Pose, exp_so3, log_so3, look_at

__all__ = [
    "CameraModel",
    "Pose",
    "Line2D",
    "LineFit",
    "apply_h",
    "dlt_homography",
    "fit_rigid_2d",
    "fit_line_tls",
    "exp_so3",
    "log_so3",
    "look_at",
]
