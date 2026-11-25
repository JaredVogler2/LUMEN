"""Data model (prompt 01).

These pydantic models are the persisted session state and the API contract.
Numerical work happens on NumPy objects inside ``lumen.vision`` /
``lumen.geometry``; they are converted to these models at the boundary.

Coordinate frames named in the models
-------------------------------------
``image``   raw pixels of a frame as recorded (after EXIF orientation), pixel-
            centre convention, ``u`` right, ``v`` down.
``ideal``   the same pixels with lens distortion removed.
``plane``   millimetres on the target plane, in the gauge tag's frame.
``local``   the engineering frame ``(x, z)`` built from the confirmed datums.
``camera``  OpenCV camera frame (x right, y down, z forward), millimetres.
``tag``     tag frame (x right, y up, z out of the printed face).

Every numeric output carries a :class:`Provenance` so the UI and the export
can label it as measured, inferred, manual, configured, nominal or simulated.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=True)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
class Provenance(str, Enum):
    MEASURED = "measured"  # derived from pixels of this session by a metrology algorithm
    INFERRED = "inferred"  # estimated indirectly (self-calibration, EXIF prior, model fit)
    MANUAL = "manual"  # entered or drawn by the operator
    CONFIGURED = "configured"  # taken from the operation definition
    NOMINAL = "nominal"  # catalogue / packaging value, not measured
    SIMULATED = "simulated"  # synthetic demo content
    UNKNOWN = "unknown"


class GuidanceStatus(str, Enum):
    """What the session may show for a placard (prompt 10).

    ``validated`` is only reachable inside a physically validated domain
    supplied by an administrator; this repository ships none.
    """

    BLOCKED = "blocked"  # no overlay guidance; diagnostics only
    RESEARCH_PREVIEW = "research_preview"  # overlay shown, gates not met
    PROTOTYPE_GUIDANCE = "prototype_guidance"  # all computed gates met; not physically validated
    VALIDATED = "validated"


class SessionMode(str, Enum):
    REAL = "real"
    DEMO = "demo"


class FrameRole(str, Enum):
    REGISTRATION = "registration"
    VERIFICATION = "verification"
    CALIBRATION = "calibration"
    FLASH_ON = "flash_on"
    FLASH_OFF = "flash_off"


class FrameSource(str, Enum):
    PHOTO_UPLOAD = "photo_upload"
    CAMERA_STILL = "camera_still"
    VIDEO_UPLOAD = "video_upload"
    LIVE_SWEEP = "live_sweep"
    DEMO = "demo"


class TorchState(str, Enum):
    ON = "on"
    OFF = "off"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class Pose(_Model):
    """Rigid transform ``x_to = R @ x_from + t`` (millimetres)."""

    frame_to: str
    frame_from: str
    R: list[list[float]]
    t: list[float]
    provenance: Provenance = Provenance.MEASURED
    reprojection_rms_px: float | None = None
    ambiguity_delta_chi2: float | None = None


class Calibration(_Model):
    """Camera intrinsics for one device / lens / resolution / capture mode.

    Web APIs do not expose calibrated intrinsics, so a profile is either
    measured with a ChArUco/checkerboard target, inferred by self-calibration
    from the session's tags, or a coarse EXIF prior.  A profile is only applied
    to frames with exactly matching pixel geometry (or an exact, recorded
    uniform rescale of it).
    """

    id: str
    label: str = ""
    method: str = "unknown"  # charuco | checkerboard | self_calibration | exif_prior | synthetic_truth | manual
    provenance: Provenance = Provenance.UNKNOWN
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = Field(default_factory=lambda: [0.0] * 5)
    rms_px: float | None = None
    n_views: int | None = None
    #: fraction of the image area covered by calibration points
    coverage: float | None = None
    #: standard deviations of (fx, fy, cx, cy, k1, k2, p1, p2, k3) when known
    param_sigma: list[float] | None = None
    device: str = ""
    lens: str = ""
    zoom: float | None = None
    capture_mode: str = ""  # "photo" | "video" | free text
    created: str = ""
    notes: str = ""


class MarkerObservation(_Model):
    tag_id: int
    corners_px: list[list[float]]  # image frame
    corners_ideal: list[list[float]]
    cov: list[list[float]]  # 8x8 covariance of corners_ideal
    corner_sigma_px: float
    size_mm: float
    size_provenance: Provenance
    accepted: bool
    reject_reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    refined: bool = False
    edge_px: list[float] = Field(default_factory=list)
    sigma_blur_px: float = 0.0
    contrast: float = 0.0
    bit_errors: int = 0
    decode_margin: float = 0.0
    bias_px: float = 0.0
    bias_sigma_px: float | None = None
    bias_applied: bool = False
    max_sagitta_px: float = 0.0
    saturation: float = 0.0
    quality: float = 0.0


class FrameQualityModel(_Model):
    sharpness: float = 0.0
    mean: float = 0.0
    clipped_high: float = 0.0
    clipped_low: float = 0.0
    contrast: float = 0.0


class Frame(_Model):
    id: str
    index: int
    role: FrameRole = FrameRole.REGISTRATION
    source: FrameSource = FrameSource.PHOTO_UPLOAD
    #: pixel geometry all coordinates of this frame refer to (upright, as analysed)
    width: int
    height: int
    #: stored pixel geometry before EXIF orientation and before any analysis downscale
    native_width: int
    native_height: int
    processing_scale: float = 1.0
    exif_orientation: int = 1
    orientation_applied: bool = False
    #: what the client reported about the capture device (never trusted for geometry)
    camera_label: str = ""
    lens: str = ""
    zoom: float | None = None
    facing_mode: str = ""
    torch: TorchState = TorchState.UNKNOWN
    captured_at: str = ""
    exif: dict[str, Any] = Field(default_factory=dict)
    quality: FrameQualityModel = Field(default_factory=FrameQualityModel)
    sha256: str = ""
    n_bytes: int = 0
    blob: str = ""
    markers: list[MarkerObservation] = Field(default_factory=list)
    selected: bool = True
    pair_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    #: calibration applied to this frame ("" = none) and why/why not
    calibration_id: str = ""
    calibration_note: str = ""
    video_time_s: float | None = None


class DatumObservation(_Model):
    """A datum line on the target plane with its evidence."""

    kind: str  # "floor" | "side"
    #: Hesse normal form in the plane frame: n . q = d (millimetres)
    n: list[float]
    d: float
    provenance: Provenance
    confirmed: bool = False
    #: observed image feature this came from
    feature_kind: str = ""  # "step" | "bar" | "manual" | "silhouette" | "cylinder_model"
    frame_id: str | None = None
    #: support of the line in the plane frame (two end points)
    endpoints: list[list[float]] = Field(default_factory=list)
    sigma_perp_mm: float | None = None
    sigma_angle_rad: float | None = None
    #: configured offset already applied to obtain the datum from the feature
    offset_applied_mm: float = 0.0
    #: evidence about whether the feature lies on the plane
    on_plane_evidence: str = "unverified"  # verified_multiview | unverified | off_plane | configured
    distinctness: float | None = None
    n_candidates: int | None = None
    notes: list[str] = Field(default_factory=list)


class LocalFrameModel(_Model):
    M: list[list[float]]
    origin: list[float]
    ex: list[float]
    ez: list[float]
    policy: str
    side_of_panel: str
    raw_angle_deg: float
    nominal_angle_deg: float
    axis_disagreement_deg: float
    orthogonalized: bool
    valid: bool
    reasons: list[str] = Field(default_factory=list)
    x_positive: str = "toward_panel"
    z_positive: str = "toward_panel"
    provenance: Provenance = Provenance.MEASURED


class SurfaceModel(_Model):
    """What is known about the target surface."""

    kind: str = "plane"  # "plane" | "plane_with_cylindrical_edge"
    #: how planarity was established
    planarity: str = "assumed"  # assumed | crossvalidated_single_view | verified_multiview
    out_of_plane_rms_mm: float | None = None
    out_of_plane_max_mm: float | None = None
    #: sparse cylinder model of a rounded edge, when observable
    cylinder: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)


class PlacementSpec(_Model):
    """Resolved placard specification in internal units (mm, rad)."""

    placard_id: str
    title: str = ""
    x_mm: float | None = None
    z_mm: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    angle_rad: float | None = None
    reference_point: str | None = None
    x_semantics: str | None = None
    tol_position_mm: float | None = None
    tol_rotation_rad: float | None = None
    tolerance_provenance: str = ""
    provenance: Provenance = Provenance.CONFIGURED


class UncertaintyComponent(_Model):
    name: str
    kind: str  # "random" | "systematic"
    #: standard uncertainty (random) or half-width bound (systematic), mm, at the worst target corner
    value_mm: float
    note: str = ""


class Uncertainty(_Model):
    """Uncertainty of a target position in the local frame (prompt 10).

    ``u_random_mm`` is repeatability (1 sigma) propagated from this capture's
    pixel noise.  ``bias_bound_mm`` is the root-sum-square of systematic
    bounds treated as rectangular distributions; ``bias_linear_mm`` is their
    plain sum (worst case).  ``expanded_mm = k * sqrt(u_random^2 + sum(b_i^2/3))``.
    No probability is attached: coverage has only been examined on synthetic
    data (docs/VALIDATION.md).
    """

    method: str = "linearized_monte_carlo"
    seed: int = 0
    n_samples: int = 0
    k: float = 2.0
    u_random_mm: float
    u_random_xz_mm: list[float] = Field(default_factory=list)
    bias_bound_mm: float
    bias_linear_mm: float
    combined_standard_mm: float
    expanded_mm: float
    u_angle_rad: float | None = None
    per_corner_expanded_mm: list[float] = Field(default_factory=list)
    components: list[UncertaintyComponent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GateResult(_Model):
    code: str
    passed: bool
    severity: Severity
    message: str
    value: float | None = None
    limit: float | None = None
    suggestion: str | None = None


class GateReport(_Model):
    status: GuidanceStatus
    placard_id: str | None = None
    gates: list[GateResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    uncertainty: Uncertainty | None = None
    physically_validated: bool = False
    notice: str = ""


class Verification(_Model):
    id: str
    frame_id: str
    placard_id: str
    #: measured corners (TL, TR, BR, BL in local semantics), plane and local frames
    corners_plane: list[list[float]] = Field(default_factory=list)
    corners_local: list[list[float]] = Field(default_factory=list)
    corners_px: list[list[float]] = Field(default_factory=list)
    corner_provenance: list[Provenance] = Field(default_factory=list)
    confirmed: bool = False
    #: signed deviations measured - target, local frame
    dx_mm: float | None = None
    dz_mm: float | None = None
    radial_mm: float | None = None
    rotation_rad: float | None = None
    width_error_mm: float | None = None
    height_error_mm: float | None = None
    edge_deviation_mm: dict[str, float] = Field(default_factory=dict)
    uncertainty: Uncertainty | None = None
    #: "within_tolerance_observed" | "outside_tolerance_observed" | "indeterminate"
    observation: str = "indeterminate"
    flags: list[str] = Field(default_factory=list)
    registration_flags: list[str] = Field(default_factory=list)
    created: str = ""
    notice: str = ""


class SessionEvent(_Model):
    at: str
    action: str
    detail: dict[str, Any] = Field(default_factory=dict)


class Session(_Model):
    id: str
    mode: SessionMode = SessionMode.REAL
    created: str
    updated: str
    algorithm_version: str
    app_version: str
    operator: str = ""
    #: snapshot of the operation record used (lumen.operations.library.snapshot)
    operation: dict[str, Any] | None = None
    active_placard_id: str | None = None
    tag_sizes: dict[str, Any] = Field(default_factory=dict)
    calibration: Calibration | None = None
    frames: list[Frame] = Field(default_factory=list)
    #: last registration result (summary dictionaries from the solvers)
    registration: dict[str, Any] | None = None
    datums: dict[str, DatumObservation] = Field(default_factory=dict)
    local_frame: LocalFrameModel | None = None
    surface: SurfaceModel = Field(default_factory=SurfaceModel)
    verifications: list[Verification] = Field(default_factory=list)
    reference_distances: list[dict[str, Any]] = Field(default_factory=list)
    events: list[SessionEvent] = Field(default_factory=list)
    jobs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    frame_counter: int = 0
    research: dict[str, Any] = Field(default_factory=dict)
