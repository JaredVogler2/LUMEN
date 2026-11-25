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
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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
        try:
            return normalize_length_unit(v)
        except LumenValidationError as exc:  # as a ValueError the place of the mistake is reported with it
            raise ValueError(exc.message) from exc

    @property
    def mm(self) -> float:
        return to_mm(self.value, self.unit)


class Angle(_Model):
    value: float
    unit: str

    @field_validator("unit")
    @classmethod
    def _unit(cls, v: str) -> str:
        try:
            return normalize_angle_unit(v)
        except LumenValidationError as exc:
            raise ValueError(exc.message) from exc

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
            raise ValueError("must be 1 to 64 characters (letters, digits, . _ - / and spaces), starting with a letter or digit")
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


class AuditEntry(_Model):
    """Who changed which record when (kept with the local library file)."""

    at: str
    action: Literal["saved", "superseded", "deleted"]
    operation_id: str
    revision: str
    status: str = ""
    content_hash: str = ""
    author: str = ""


class LibraryFile(_Model):
    schema_version: int = SCHEMA_VERSION
    library_revision: str = ""
    notice: str = ""
    operations: list[OperationDefinition] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)


_AUDIT_KEEP = 500


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------
@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock for read-modify-write of the local library across worker processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a+") as fh:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - no fcntl on this platform: one process is assumed
            pass
        yield  # closing the file releases the lock


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
        self.persistent: bool | None = None  # set by whoever knows where the overlay lives
        self._stamp: tuple[int, int] | None = None
        self._checked = 0.0
        self.reload()

    # ---------------------------------------------------------------- load
    @property
    def overlay_path(self) -> Path | None:
        return None if self._overlay_dir is None else self._overlay_dir / "operations.json"

    @property
    def editable(self) -> bool:
        return self._overlay_dir is not None

    def _file_stamp(self) -> tuple[int, int] | None:
        path = self.overlay_path
        if path is None:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _refresh(self) -> None:
        """Pick up records saved by another worker process (checked at most twice a second)."""
        if self._overlay_dir is None:
            return
        now = time.monotonic()
        if now - self._checked < 0.5:
            return
        self._checked = now
        if self._file_stamp() != self._stamp:
            self.reload()

    def reload(self) -> None:
        with self._lock:
            self._stamp = self._file_stamp()
            self._checked = time.monotonic()
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
        self._refresh()
        with self._lock:
            return [self._ops[k] for k in sorted(self._ops)]

    def revisions(self, operation_id: str) -> list[OperationDefinition]:
        self._refresh()
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
            return {"status": "none", "query": q, "matches": [], "other_revisions": [], "near": []}
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
        # Revisions of one operation share its codes.  The released revision is
        # the one a code means; its drafts and superseded revisions do not make
        # the code ambiguous (they are listed under "other_revisions").  An
        # operation with no released revision is offered as what it is.
        other: list[dict[str, Any]] = []
        by_op: dict[str, list[dict[str, Any]]] = {}
        for m in matches:
            by_op.setdefault(m["operation_id"], []).append(m)
        matches = []
        for group in by_op.values():
            for wanted in ("released", "draft", "superseded"):
                keep = [m for m in group if m["status"] == wanted]
                if keep:
                    matches += keep
                    other += [m for m in group if m["status"] != wanted]
                    break
        # one operation revision matched through several fields is still one match
        uniq = {(m["operation_id"], m["revision"], m["placard_id"]) for m in matches}
        ops = {(m["operation_id"], m["revision"]) for m in matches}
        if not matches:
            status = "none"
        elif len(ops) == 1 and len({u[2] for u in uniq if u[2] is not None}) <= 1:
            status = "exact"
        else:
            status = "ambiguous"
        return {"status": status, "query": q, "matches": matches, "other_revisions": other, "near": near[:20]}

    # --------------------------------------------------------------- write
    def storage(self) -> dict[str, Any]:
        """Where local records go, for the people who enter them."""
        return {"editable": self.editable, "persistent": self.persistent if self.editable else None}

    def _require_editable(self) -> Path:
        if not self.editable or self.overlay_path is None:
            raise ConflictError(
                "library_read_only",
                "The operation library is read-only on this deployment.",
                suggestions=["Set LUMEN_OPERATIONS_DIR to a writable directory to enable local editing."],
            )
        return self.overlay_path

    def _read_local(self, path: Path) -> LibraryFile:
        if not path.exists():
            return LibraryFile(schema_version=SCHEMA_VERSION, library_revision="", operations=[])
        try:
            return LibraryFile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConflictError("local_library_unreadable", f"The local library file is not readable: {exc}") from exc

    def _write_local(self, path: Path, lib: LibraryFile, entry: AuditEntry) -> None:
        lib.library_revision = entry.at
        lib.notice = "Locally edited LUMEN operation records. Generic research data only."
        lib.audit = (lib.audit + [entry])[-_AUDIT_KEEP:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(lib.model_dump(mode="json"), indent=2), encoding="utf-8")
        tmp.replace(path)
        self.reload()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def save(self, op: OperationDefinition, *, author: str = "") -> OperationDefinition:
        """Store a new revision in the editable overlay.

        An existing ``(operation_id, revision)`` is immutable once released:
        changing released content requires a new revision string.  (Marking a
        released revision as superseded is done with :meth:`supersede`.)
        """
        path = self._require_editable()
        if op.synthetic and self._origin.get(op.key()) == "bundled":
            raise ConflictError("bundled_record_immutable", "Bundled sample records cannot be overwritten; save under a new operation id or revision.")
        with self._lock, _file_lock(path):
            self.reload()  # another process may have saved since this one looked
            existing = self._ops.get(op.key())
            if existing is not None:
                if self._origin.get(op.key()) == "bundled":
                    raise ConflictError("bundled_record_immutable", "Bundled sample records cannot be overwritten; use a new revision.")
                if existing.status in ("released", "superseded") and existing.content_hash() != op.content_hash():
                    raise ConflictError(
                        "released_revision_immutable",
                        f"Revision {op.revision!r} of {op.operation_id!r} is {existing.status}; save the change as a new revision.",
                    )
            if op.status == "released" and op.issues():
                raise LumenValidationError(
                    "incomplete_record_cannot_be_released",
                    "A record with missing datums, dimensions or tolerances cannot be released.",
                    details={"issues": op.issues()},
                )
            lib = self._read_local(path)
            lib.operations = [o for o in lib.operations if o.key() != op.key()] + [op]
            self._write_local(path, lib, AuditEntry(at=self._now(), action="saved", operation_id=op.operation_id, revision=op.revision, status=op.status, content_hash=op.content_hash(), author=author))
            return self._ops[op.key()]

    def supersede(self, operation_id: str, revision: str, *, author: str = "") -> OperationDefinition:
        """Mark a released local revision as superseded (its values stay as they were)."""
        path = self._require_editable()
        with self._lock, _file_lock(path):
            self.reload()
            op = self.get(operation_id, revision)
            if self._origin.get(op.key()) == "bundled":
                raise ConflictError("bundled_record_immutable", "Bundled sample records cannot be changed.")
            if op.status == "superseded":
                return op
            if op.status != "released":
                raise ConflictError("only_released_can_be_superseded", f"Revision {revision!r} of {operation_id!r} is a {op.status}; only a released revision can be superseded.")
            new = op.model_copy(update={"status": "superseded"})
            lib = self._read_local(path)
            lib.operations = [o for o in lib.operations if o.key() != op.key()] + [new]
            self._write_local(path, lib, AuditEntry(at=self._now(), action="superseded", operation_id=operation_id, revision=revision, status="superseded", content_hash=new.content_hash(), author=author))
            return self._ops[op.key()]

    def delete(self, operation_id: str, revision: str, *, author: str = "") -> None:
        """Remove a local *draft*.  Released and superseded revisions are kept: sessions refer to them."""
        path = self._require_editable()
        with self._lock, _file_lock(path):
            self.reload()
            op = self.get(operation_id, revision)
            if self._origin.get(op.key()) == "bundled":
                raise ConflictError("bundled_record_immutable", "Bundled sample records cannot be deleted.")
            if op.status != "draft":
                raise ConflictError("only_drafts_can_be_deleted", f"Revision {revision!r} of {operation_id!r} is {op.status}; it is kept so that sessions which used it stay traceable.")
            lib = self._read_local(path)
            lib.operations = [o for o in lib.operations if o.key() != op.key()]
            self._write_local(path, lib, AuditEntry(at=self._now(), action="deleted", operation_id=operation_id, revision=revision, status="draft", content_hash=op.content_hash(), author=author))

    def audit(self, operation_id: str | None = None) -> list[dict[str, Any]]:
        path = self.overlay_path
        if path is None or not path.exists():
            return []
        try:
            entries = LibraryFile.model_validate_json(path.read_text(encoding="utf-8")).audit
        except Exception:
            return []
        return [e.model_dump(mode="json") for e in entries if operation_id is None or e.operation_id == operation_id]


def record_problems(exc: Exception) -> list[dict[str, str]]:
    """Why a record could not be read, field by field, in words an operator can act on."""
    if not isinstance(exc, ValidationError):
        return [{"where": "", "message": str(exc)[:300]}]
    out: list[dict[str, str]] = []
    for e in exc.errors()[:40]:
        loc = [str(x) for x in e.get("loc", ())]
        msg = str(e.get("msg", "not valid"))
        for prefix in ("Value error, ", "Assertion failed, "):
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
        kind = e.get("type", "")
        if kind == "missing":
            msg = "is required"
        elif kind == "extra_forbidden":
            msg = "is not a field of an operation record"
        elif kind == "literal_error":
            msg = "is not one of the allowed values" + (f" ({e['ctx']['expected']})" if isinstance(e.get("ctx"), dict) and "expected" in e["ctx"] else "")
        elif kind.endswith("_type") or kind.endswith("_parsing"):
            msg = msg[0].lower() + msg[1:] if msg else "has the wrong kind of value"
        out.append({"where": ".".join(loc), "message": msg[:300]})
    return out


def parse_operation(data: dict[str, Any]) -> OperationDefinition:
    """Validate an operation record supplied through the API."""
    try:
        return OperationDefinition.model_validate(data)
    except Exception as exc:
        problems = record_problems(exc)
        text = "; ".join(f"{p['where']}: {p['message']}" if p["where"] else p["message"] for p in problems[:6])
        raise LumenValidationError(
            "invalid_operation_record", "The operation record is not valid.", details={"error": text[:2000], "problems": problems},
            suggestions=[text[:400]] if text else None,
        ) from exc


def analyse_record(data: dict[str, Any]) -> tuple[OperationDefinition | None, list[dict[str, str]]]:
    """A record as typed so far: what cannot be read, and the record if the rest can be checked.

    While a record is being entered its ID or revision may still be empty or
    not allowed; those are reported as problems and the rest is still analysed
    (under a stand-in ID that is never stored).
    """
    try:
        return OperationDefinition.model_validate(data), []
    except Exception as exc:
        problems = record_problems(exc)
    if not isinstance(data, dict):
        return None, problems
    patched = dict(data)
    wheres = {p["where"] for p in problems}
    if "operation_id" in wheres:
        patched["operation_id"] = "UNNAMED"
    if "revision" in wheres:
        patched["revision"] = "UNNAMED"
    if isinstance(patched.get("panel"), dict) and "panel.panel_id" in wheres:
        patched["panel"] = {**patched["panel"], "panel_id": ""}
    try:
        return OperationDefinition.model_validate(patched), problems
    except Exception:
        return None, problems


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
            # Local records go to the configured directory; otherwise next to the
            # sessions, which is temporary storage unless LUMEN_DATA_DIR says otherwise.
            if s.operations_readonly:
                overlay = None
            elif s.operations_dir:
                overlay = Path(s.operations_dir)
            else:
                overlay = Path(s.data_dir) / "library"
            _LIB = OperationLibrary(Path(__file__).parent / "data" / "operations.json", overlay)
            _LIB.persistent = bool(s.operations_dir) or s.data_dir_configured
        return _LIB


def reset_library() -> None:
    global _LIB
    with _LIB_LOCK:
        _LIB = None
