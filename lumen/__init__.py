"""LUMEN — browser-based optical placard positioning research application.

RESEARCH PROTOTYPE. Outputs are prototype guidance, never product acceptance.

Conventions used across the whole package (see docs/CONVENTIONS.md):

* Lengths are millimetres, angles are radians, unless a name says otherwise.
* Image pixel coordinates follow the OpenCV pixel-centre convention:
  integer ``(u, v)`` is the centre of the pixel in column ``u`` and row ``v``;
  ``u`` grows to the right, ``v`` grows downward.
* A transform named ``T_a_b`` maps coordinates expressed in frame ``b`` into
  frame ``a`` (``x_a = R_a_b @ x_b + t_a_b``).
"""

__version__ = "0.1.0"

#: Version of the measurement algorithms. Bump when any change can alter a
#: numeric result, independently of the application version.
ALGORITHM_VERSION = "lumen-algo-2026.11.1"

RESEARCH_NOTICE = (
    "RESEARCH PROTOTYPE — measurement evidence only. Not product acceptance. "
    "Physical accuracy has not been validated."
)
