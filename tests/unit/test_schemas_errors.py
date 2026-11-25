import pytest
from pydantic import ValidationError

from lumen import ALGORITHM_VERSION, RESEARCH_NOTICE, __version__
from lumen.errors import GeometryError, LimitExceededError, LumenError, LumenValidationError, NotFoundError
from lumen.schemas import Calibration, DatumObservation, Frame, GateReport, GuidanceStatus, MarkerObservation, PlacementSpec, Pose, Provenance, Session, SurfaceModel, Uncertainty, Verification


def test_status_enum_and_provenance_values():
    assert [s.value for s in GuidanceStatus] == ["blocked", "research_preview", "prototype_guidance", "validated"]
    assert {p.value for p in Provenance} == {"measured", "inferred", "manual", "configured", "nominal", "simulated", "unknown"}


def test_error_model():
    e = GeometryError("pose_ambiguous", "Two poses fit.", suggestions=["Add a tag."], details={"delta": 1.2})
    assert e.http_status == 422 and e.to_dict() == {"code": "pose_ambiguous", "message": "Two poses fit.", "details": {"delta": 1.2}, "suggestions": ["Add a tag."]}
    assert NotFoundError().http_status == 404 and LimitExceededError().http_status == 413 and LumenValidationError().http_status == 422
    assert issubclass(GeometryError, LumenError)


def test_session_round_trip():
    s = Session(id="x" * 24, created="2026-11-25T00:00:00Z", updated="2026-11-25T00:00:00Z", algorithm_version=ALGORITHM_VERSION, app_version=__version__)
    assert s.mode == "real" and s.surface.kind == "plane" and s.frames == []
    d = s.model_dump(mode="json")
    assert Session.model_validate(d).model_dump(mode="json") == d


def test_frame_and_marker_models():
    m = MarkerObservation(tag_id=3, corners_px=[[0, 0]] * 4, corners_ideal=[[0, 0]] * 4, cov=[[0] * 8] * 8, corner_sigma_px=0.02, size_mm=50.0, size_provenance="nominal", accepted=True)
    f = Frame(id="f001", index=1, width=1600, height=1200, native_width=4032, native_height=3024, processing_scale=0.397, markers=[m])
    assert f.role == "registration" and f.torch == "unknown" and f.markers[0].size_provenance == "nominal"
    with pytest.raises(ValidationError):
        Frame(id="f", index=1, width=1, height=1, native_width=1, native_height=1, role="selfie")


def test_required_entities_exist_and_validate():
    Pose(frame_to="camera", frame_from="plane", R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 500])
    Calibration(id="c", width=4032, height=3024, fx=3000, fy=3000, cx=2015.5, cy=1511.5)
    DatumObservation(kind="floor", n=[0, 1], d=0.0, provenance="measured")
    SurfaceModel()
    PlacementSpec(placard_id="P")
    Uncertainty(u_random_mm=0.1, bias_bound_mm=0.2, bias_linear_mm=0.3, combined_standard_mm=0.22, expanded_mm=0.44)
    Verification(id="v1", frame_id="f001", placard_id="P")
    GateReport(status="blocked")
    with pytest.raises(ValidationError):
        GateReport(status="accepted")  # there is no acceptance status


def test_research_notice_is_explicit():
    assert "not product acceptance" in RESEARCH_NOTICE.lower() and "not been validated" in RESEARCH_NOTICE.lower()
