"""Optional language-model assistant (disabled by default).

LUMEN works completely without it.  When an administrator enables it and
points it at an approved OpenAI-compatible endpoint, it can *explain* results
that LUMEN has already computed.  It never computes geometry, never changes a
tolerance, a gate or a session, and never decides acceptance.
"""

from .service import AssistantService, build_context  # noqa: F401
