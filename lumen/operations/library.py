"""Generic placard operation library.

An *operation* describes one panel and the placards that go on it: the datums
(names, signs, features), each placard's offsets, size, rotation and reference
point, and the authorised tolerances — with units, revision and provenance.

Rules enforced here (prompt 08):

* Every length and angle carries an explicit unit; values are converted to
  millimetres / radians exactly once, in :meth:`PlacardDefinition.resolved`.
* Nothing is invented.  A record with a missing tolerance, datum sign, axis
  policy or dimension is loadable (so it can be inspected and edited) but is
  reported as *incomplete*, and an incomplete record can never reach a
  validated or guidance status — only a clearly labelled research preview.
* IDs must match exactly.  A scan or a typed string that matches more than one
  record, or matches only approximately, is returned as *ambiguous* and the
  operator has to choose.
* A barcode identifies a record.  It never supplies physical scale.
* Editing produces a new revision; sessions store a full snapshot plus the
  content hash of the record they used.

The bundled library (``data/operations.json``) is synthetic and generic.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..errors import ConflictError, LumenValidationError, NotFoundError
from ..geometry.localframe import AXIS_POLICIES, REFERENCE_POINTS
from ..units import normalize_angle_unit, normalize_length_unit, to_mm, to_rad

SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/ ]{0,63}$")

DATUM_FEATURES = ("planar_edge", "curve_tangent", "curve_outer_extreme")
DIRECTIONS = ("toward_panel", "away_from_panel")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Length(_Model):
    value: float
    unit: str

    @field_validator("unit")
    @classmethod
    def _unit(cls, v: str) -> str:
        return normalize_length_unit(v)

    @property
    def mm(self) -> float:
        return to_mm(self.value, self.unit)


class Angle(_Model):
    value: float
    unit: str

    @field_validator("unit")
    @classmethod
    def _unit(cls, v: str) -> str:
        return normalize_angle_unit(v)

    @property
    def rad(self) -> float:
        return to_rad(self.value, self.unit)


class Provenance(_Model):
    source: str = ""
    author: str = ""
    date: str = ""
    notes: str = ""


class DatumDefinition(_Model):
    """One datum of the local frame.

    ``feature`` says what physical thing realises the datum:

    * ``planar_edge`` — a visible line lying on the target plane (seam, joint,
      trim line, the floor interface of a panel that reaches the floor);
    * ``curve_tangent`` — the line where the flat panel starts to curve.  It is
      not directly visible; it needs 3-D evidence (multi-view silhouette fit,
      experimental photometric fit) or a configured ``radius``;
    * ``curve_outer_extreme`` — the outermost line of a rounded edge projected
      perpendicularly onto the plane.  Off-plane; needs the same evidence.
    """

    id: str
    name: str
    feature: str | None = None
    on_target_plane: bool | None = None
    #: direction in which the coordinate governed by this datum grows
    positive: str | None = None
    #: configured in-plane offset from the observed feature to the datum (along +axis)
    offset: Length | None = None
    #: configured curve radius, for curve features
    radius: Length | None = None
    #: only for the side datum: where it lies when facing the panel, floor down
    side_of_panel: Literal["left", "right"] | None = None
    notes: str = ""


class DatumScheme(_Model):
    floor: DatumDefinition | None = None
    side: DatumDefinition | None = None
    axis_policy: str | None = None
    nominal_angle: Angle | None = None
    max_axis_disagreement: Angle | None = None


class Tolerance(_Model):
    position: Length | None = None
    rotation: Angle | None = None
    provenance: str = ""


class PlacardPosition(_Model):
    x: Length | None = None
    z: Length | None = None
    reference_point: str | None = None
    #: "projected": offsets measured in the plane.  "arc_length": measured along the surface.
    x_semantics: Literal["projected", "arc_length"] | None = None


class PlacardSize(_Model):
    width: Length | None = None
    height: Length | None = None


class PlacardDefinition(_Model):
    placard_id: str
    title: str = ""
    barcodes: list[str] = Field(default_factory=list)
    position: PlacardPosition = Field(default_factory=PlacardPosition)
    size: PlacardSize = Field(default_factory=PlacardSize)
    rotation: Angle | None = None
    tolerance: Tolerance = Field(default_factory=Tolerance)
    notes: str = ""

    def issues(self) -> list[str]:
        out: list[str] = []
        p, s, t = self.position, self.size, self.tolerance
        if p.x is None:
            out.append("missing_x_offset")
        if p.z is None:
            out.append("missing_z_offset")
        if p.reference_point is None:
            out.append("missing_reference_point")
        elif p.reference_point not in REFERENCE_POINTS:
            out.append("invalid_reference_point")
        if p.x_semantics is None:
            out.append("missing_x_semantics")
        if s.width is None or s.height is None:
            out.append("missing_size")
        else:
            if s.width.mm <= 0 or s.height.mm <= 0:
                out.append("invalid_size")
            elif s.width.mm > 2000 or s.height.mm > 2000:
                out.append("implausible_size")
        if self.rotation is None:
            out.append("missing_rotation")
        if t.position is None:
            out.append("missing_position_tolerance")
        elif t.position.mm <= 0:
            out.append("invalid_position_tolerance")
        if t.rotation is None:
            out.append("missing_rotation_tolerance")
        elif t.rotation.rad <= 0:
            out.append("invalid_rotation_tolerance")
        return out

    def resolved(self) -> dict[str, Any]:
        """Numeric spec in internal units (mm, rad).  Missing values are ``None``."""
        p, s, t = self.position, self.size, self.tolerance
        return {
            "placard_id": self.placard_id,
            "title": self.title,
            "x_mm": None if p.x is None else p.x.mm,
            "z_mm": None if p.z is None else p.z.mm,
            "width_mm": None if s.width is None else s.width.mm,
            "height_mm": None if s.height is None else s.height.mm,
            "angle_rad": None if self.rotation is None else self.rotation.rad,
            "reference_point": p.reference_point,
            "x_semantics": p.x_semantics,
            "tol_position_mm": None if t.position is None else t.position.mm,
            "tol_rotation_rad": None if t.rotation is None else t.rotation.rad,
            "tolerance_provenance": t.provenance,
        }


class PanelInfo(_Model):
    panel_id: str
    description: str = ""
    barcodes: list[str] = Field(default_factory=list)


class SurfaceInfo(_Model):
    #: "plane": the placards sit on a flat region.  "curved": experimental only.
    target: Literal["plane", "curved"] = "plane"
    notes: str = ""


class Effectivity(_Model):
    configuration: str = ""
    notes: str = ""


class OperationDefinition(_Model):
    operation_id: str
    title: str = ""
    revision: str
    status: Literal["draft", "released", "superseded"] = "draft"
    effectivity: Effectivity = Field(default_factory=Effectivity)
    provenance: Provenance = Field(default_factory=Provenance)
    panel: PanelInfo
    surface: SurfaceInfo = Field(default_factory=SurfaceInfo)
    datums: DatumScheme = Field(default_factory=DatumScheme)
    placards: list[PlacardDefinition] = Field(default_factory=list)
    #: marks bundled synthetic content so it can never be mistaken for real data
    synthetic: bool = False

    @field_validator("operation_id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError("operation_id must be 1-64 characters: letters, digits, . _ - / space")
        return v

    # ------------------------------------------------------------ analysis
    def issues(self) -> dict[str, list[str]]:
        """Everything that prevents a validated / guidance status, by scope."""
        out: dict[str, list[str]] = {}
        d = self.datums
        di: list[str] = []
        for key, datum in (("floor", d.floor), ("side", d.side)):
            if datum is None:
                di.append(f"missing_{key}_datum")
                continue
            if datum.feature is None:
                di.append(f"{key}_datum_feature_missing")
            elif datum.feature not in DATUM_FEATURES:
                di.append(f"{key}_datum_feature_invalid")
            if datum.on_target_plane is None:
                di.append(f"{key}_datum_plane_relation_missing")
            if datum.positive is None:
                di.append(f"{key}_datum_sign_missing")
            elif datum.positive not in DIRECTIONS:
                di.append(f"{key}_datum_sign_invalid")
            if datum.feature in ("curve_tangent", "curve_outer_extreme") and datum.on_target_plane is False and datum.radius is None:
                # allowed: radius can come from 3-D evidence instead; noted, not blocking
                pass
        if d.axis_policy is None:
            di.append("axis_policy_missing")
        elif d.axis_policy not in AXIS_POLICIES:
            di.append("axis_policy_invalid")
        if d.nominal_angle is None:
            di.append("nominal_datum_angle_missing")
        if d.axis_policy in ("floor_primary", "side_primary") and d.max_axis_disagreement is None:
            di.append("max_axis_disagreement_missing")
        if di:
            out["datums"] = di
        seen: set[str] = set()
        for p in self.placards:
            pi = p.issues()
            if p.placard_id in seen:
                pi.append("duplicate_placard_id")
            seen.add(p.placard_id)
            if pi:
                out[f"placard:{p.placard_id}"] = pi
        if not self.placards:
            out["operation"] = ["no_placards"]
        if self.surface.target == "curved":
            out.setdefault("operation", []).append("curved_target_experimental")
        return out

    def is_complete(self, placard_id: str | None = None) -> bool:
        iss = self.issues()
        if "datums" in iss or "operation" in iss:
            return False
        if placard_id is None:
            return not iss
        return f"placard:{placard_id}" not in iss

    def placard(self, placard_id: str) -> PlacardDefinition:
        for p in self.placards:
            if p.placard_id == placard_id:
                return p
        raise NotFoundError("placard_not_found", f"Placard {placard_id!r} is not part of operation {self.operation_id!r}.")

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def key(self) -> tuple[str, str]:
        return (self.operation_id, self.revision)

    def summary(self) -> dict[str, Any]:
        iss = self.issues()
        return {
            "operation_id": self.operation_id,
            "title": self.title,
            "revision": self.revision,
            "status": self.status,
            "panel_id": self.panel.panel_id,
            "configuration": self.effectivity.configuration,
            "n_placards": len(self.placards),
            "placard_ids": [p.placard_id for p in self.placards],
            "complete": not iss,
            "issues": iss,
            "synthetic": self.synthetic,
            "content_hash": self.content_hash(),
        }


class LibraryFile(_Model):
    schema_version: int = SCHEMA_VERSION
    library_revision: str = ""
    notice: str = ""
    operations: list[OperationDefinition] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())


class OperationLibrary:
    """In-memory view of the bundled library plus an optional editable overlay.

    The overlay directory (``LUMEN_OPERATIONS_DIR``) holds ``operations.json``
    with locally created or revised records.  Writes are atomic.  On hosts
    where the content directory is replaced at every deploy (Posit Connect),
    point the variable at persistent storage or treat edits as session-scoped.
    """

    def __init__(self, bundled: Path, overlay_dir: Path | None = None) -> None:
        self._bundled = bundled
        self._overlay_dir = overlay_dir
        self._lock = threading.RLock()
        self._ops: dict[tuple[str, str], OperationDefinition] = {}
        self._origin: dict[tuple[str, str], str] = {}
        self.library_revision = ""
        self.load_errors: list[str] = []
        self.reload()

    # ---------------------------------------------------------------- load
    @property
    def overlay_path(self) -> Path | None:
        return None if self._overlay_dir is None else self._overlay_dir / "operations.json"

    @property
    def editable(self) -> bool:
        return self._overlay_dir is not None

    def reload(self) -> None:
        with self._lock:
            self._ops.clear()
            self._origin.clear()
            self.load_errors = []
            revs = []
            for origin, path in (("bundled", self._bundled), ("local", self.overlay_path)):
                if path is None or not path.exists():
                    continue
                try:
                    lib = LibraryFile.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception as exc:  # a broken file must not take the app down
                    self.load_errors.append(f"{origin}: {type(exc).__name__}: {str(exc)[:300]}")
                    continue
                if lib.schema_version != SCHEMA_VERSION:
                    self.load_errors.append(f"{origin}: unsupported schema_version {lib.schema_version}")
                    continue
                revs.append(f"{origin}:{lib.library_revision}")
                for op in lib.operations:
                    self._ops[op.key()] = op
                    self._origin[op.key()] = origin
            self.library_revision = "+".join(revs)

    # --------------------------------------------------------------- query
    def all(self) -> list[OperationDefinition]:
        with self._lock:
            return [self._ops[k] for k in sorted(self._ops)]

    def revisions(self, operation_id: str) -> list[OperationDefinition]:
        with self._lock:
            return [op for k, op in sorted(self._ops.items()) if k[0] == operation_id]

    def get(self, operation_id: str, revision: str | None = None) -> OperationDefinition:
        """Exact lookup.  Without a revision the single *released* revision is
        returned; if there are several (or none) the caller must choose."""
        revs = self.revisions(operation_id)
        if not revs:
            raise NotFoundError("operation_not_found", f"No operation with id {operation_id!r}.")
        if revision is not None:
            for op in revs:
                if op.revision == revision:
                    return op
            raise NotFoundError(
                "operation_revision_not_found",
                f"Operation {operation_id!r} has no revision {revision!r}.",
                details={"available": [op.revision for op in revs]},
            )
        released = [op for op in revs if op.status == "released"]
        if len(released) == 1:
            return released[0]
        raise ConflictError(
            "operation_revision_ambiguous",
            f"Operation {operation_id!r} has {len(released)} released revisions; choose one explicitly.",
            details={"revisions": [{"revision": op.revision, "status": op.status} for op in revs]},
        )

    def origin(self, op: OperationDefinition) -> str:
        return self._origin.get(op.key(), "unknown")

    def resolve(self, code: str) -> dict[str, Any]:
        """Match a scanned or typed code against operation, panel and placard IDs
        and their registered barcodes.

        Returns ``{"status": "exact" | "ambiguous" | "none", "matches": [...],
        "near": [...]}``.  Only exact (case-sensitive, whitespace-normalised)
        matches count; case-insensitive or partial hits are listed as ``near``
        suggestions and never selected automatically.
        """
        q = _norm(code)
        matches: list[dict[str, Any]] = []
        near: list[dict[str, Any]] = []
        if not q:
            return {"status": "none", "query": q, "matches": [], "near": []}
        ql = q.lower()
        for op in self.all():
            cands: list[tuple[str, str, str | None]] = [(op.operation_id, "operation_id", None), (op.panel.panel_id, "panel_id", None)]
            cands += [(b, "panel_barcode", None) for b in op.panel.barcodes]
            for p in op.placards:
                cands.append((p.placard_id, "placard_id", p.placard_id))
                cands += [(b, "placard_barcode", p.placard_id) for b in p.barcodes]
            for value, kind, placard_id in cands:
                v = _norm(value)
                entry = {
                    "operation_id": op.operation_id,
                    "revision": op.revision,
                    "status": op.status,
                    "matched": kind,
                    "value": value,
                    "placard_id": placard_id,
                    "title": op.title,
                }
                if v == q:
                    matches.append(entry)
                elif v.lower() == ql or (len(ql) >= 4 and ql in v.lower()):
                    near.append(entry)
        # one operation revision matched through several fields is still one match
        uniq = {(m["operation_id"], m["revision"], m["placard_id"]) for m in matches}
        ops = {(m["operation_id"], m["revision"]) for m in matches}
        if not matches:
            status = "none"
        elif len(ops) == 1 and len({u[2] for u in uniq if u[2] is not None}) <= 1:
            status = "exact"
        else:
            status = "ambiguous"
        return {"status": status, "query": q, "matches": matches, "near": near[:20]}

    # --------------------------------------------------------------- write
    def save(self, op: OperationDefinition, *, author: str = "") -> OperationDefinition:
        """Store a new revision in the editable overlay.

        An existing ``(operation_id, revision)`` is immutable once released:
        changing released content requires a new revision string.
        """
        if not self.editable or self.overlay_path is None:
            raise ConflictError(
                "library_read_only",
                "The operation library is read-only on this deployment.",
                suggestions=["Set LUMEN_OPERATIONS_DIR to a writable directory to enable local editing."],
            )
        if op.synthetic and self._origin.get(op.key()) == "bundled":
            raise ConflictError("bundled_record_immutable", "Bundled sample records cannot be overwritten; save under a new operation id or revision.")
        with self._lock:
            existing = self._ops.get(op.key())
            if existing is not None:
                if self._origin.get(op.key()) == "bundled":
                    raise ConflictError("bundled_record_immutable", "Bundled sample records cannot be overwritten; use a new revision.")
                if existing.status == "released" and existing.content_hash() != op.content_hash():
                    raise ConflictError(
                        "released_revision_immutable",
                        f"Revision {op.revision!r} of {op.operation_id!r} is released; save the change as a new revision.",
                    )
            if op.status == "released" and op.issues():
                raise LumenValidationError(
                    "incomplete_record_cannot_be_released",
                    "A record with missing datums, dimensions or tolerances cannot be released.",
                    details={"issues": op.issues()},
                )
            path = self.overlay_path
            lib = LibraryFile(schema_version=SCHEMA_VERSION, library_revision="", operations=[])
            if path.exists():
                try:
                    lib = LibraryFile.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ConflictError("local_library_unreadable", f"The local library file is not readable: {exc}") from exc
            lib.operations = [o for o in lib.operations if o.key() != op.key()] + [op]
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            lib.library_revision = stamp
            lib.notice = "Locally edited LUMEN operation records. Generic research data only."
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(lib.model_dump(mode="json"), indent=2), encoding="utf-8")
            tmp.replace(path)
            self.reload()
            return self._ops[op.key()]


def parse_operation(data: dict[str, Any]) -> OperationDefinition:
    """Validate an operation record supplied through the API."""
    try:
        return OperationDefinition.model_validate(data)
    except Exception as exc:
        raise LumenValidationError("invalid_operation_record", "The operation record is not valid.", details={"error": str(exc)[:2000]}) from exc


def snapshot(op: OperationDefinition, library: OperationLibrary) -> dict[str, Any]:
    """Self-contained copy of the record a session used (prompt 08: record the
    configuration snapshot)."""
    return {
        "operation": op.model_dump(mode="json"),
        "content_hash": op.content_hash(),
        "library_revision": library.library_revision,
        "origin": library.origin(op),
        "issues": op.issues(),
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


_LIB: OperationLibrary | None = None
_LIB_LOCK = threading.Lock()


def get_library() -> OperationLibrary:
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            from ..config import get_settings

            s = get_settings()
            overlay = Path(s.operations_dir) if s.operations_dir else None
            _LIB = OperationLibrary(Path(__file__).parent / "data" / "operations.json", overlay)
        return _LIB


def reset_library() -> None:
    global _LIB
    with _LIB_LOCK:
        _LIB = None
