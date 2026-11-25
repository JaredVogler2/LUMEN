"""Operation library (prompt 08): units, exact matching, revisions, incomplete records."""

import copy
import json

import pytest

from lumen.errors import ConflictError, LumenValidationError, NotFoundError
from lumen.operations.library import OperationDefinition, OperationLibrary, parse_operation, snapshot
from tests.conftest import BUNDLED_OPS


@pytest.fixture()
def lib(tmp_path):
    return OperationLibrary(BUNDLED_OPS, tmp_path / "local")


def _record(lib, op_id="DEMO-FLAT-01"):
    return copy.deepcopy(lib.get(op_id).model_dump(mode="json"))


def test_bundled_library_is_synthetic_and_loads_cleanly(lib):
    ops = lib.all()
    assert {o.operation_id for o in ops} >= {"DEMO-FLAT-01", "DEMO-CURVE-01", "DEMO-INCOMPLETE-01"}
    assert all(o.synthetic for o in ops) and not lib.load_errors
    raw = json.loads(BUNDLED_OPS.read_text())
    assert "SYNTHETIC" in raw["notice"].upper()


def test_units_are_converted_once_at_the_boundary(lib):
    op = lib.get("DEMO-FLAT-01")
    mm = op.placard("PLC-DEMO-101").resolved()
    assert (mm["x_mm"], mm["z_mm"], mm["width_mm"], mm["height_mm"]) == (240.0, 300.0, 120.0, 60.0)
    assert mm["tol_position_mm"] == 2.0 and mm["tol_rotation_rad"] == pytest.approx(0.0174532925)
    inch = op.placard("PLC-DEMO-102").resolved()
    assert inch["x_mm"] == pytest.approx(16.5 * 25.4) and inch["width_mm"] == pytest.approx(76.2)
    assert inch["tol_position_mm"] == pytest.approx(1.524)


def test_unknown_or_missing_units_are_rejected(lib):
    rec = _record(lib)
    rec["placards"][0]["position"]["x"] = {"value": 240, "unit": "furlong"}
    with pytest.raises(LumenValidationError):
        parse_operation(rec)
    rec = _record(lib)
    rec["placards"][0]["position"]["x"] = {"value": 240}
    with pytest.raises(LumenValidationError):
        parse_operation(rec)
    rec = _record(lib)
    rec["placards"][0]["position"]["x"] = 240  # a bare number has no unit
    with pytest.raises(LumenValidationError):
        parse_operation(rec)


def test_incomplete_record_is_loadable_but_reported(lib):
    op = lib.get("DEMO-INCOMPLETE-01", "A")
    iss = op.issues()
    assert "side_datum_sign_missing" in iss["datums"]
    assert {"missing_position_tolerance", "missing_rotation_tolerance"} <= set(iss["placard:PLC-DEMO-901"])
    assert not op.is_complete() and not op.is_complete("PLC-DEMO-901")
    assert lib.get("DEMO-FLAT-01").is_complete("PLC-DEMO-101")


@pytest.mark.parametrize("mutate,issue", [
    (lambda r: r["placards"][0]["size"].update(width={"value": -5, "unit": "mm"}), "invalid_size"),
    (lambda r: r["placards"][0]["size"].update(width={"value": 5, "unit": "m"}), "implausible_size"),
    (lambda r: r["placards"][0]["tolerance"].update(position={"value": 0, "unit": "mm"}), "invalid_position_tolerance"),
    (lambda r: r["placards"][0]["position"].update(reference_point="middle"), "invalid_reference_point"),
    (lambda r: r["placards"][0].pop("rotation"), "missing_rotation"),
])
def test_invalid_dimensions_are_issues_not_silently_fixed(lib, mutate, issue):
    rec = _record(lib)
    mutate(rec)
    op = parse_operation(rec)
    assert issue in op.issues()["placard:PLC-DEMO-101"]


def test_datum_definition_issues(lib):
    rec = _record(lib)
    rec["datums"]["axis_policy"] = None
    rec["datums"]["floor"]["feature"] = "wavy_edge"
    rec["datums"].pop("max_axis_disagreement")
    iss = parse_operation(rec).issues()["datums"]
    assert {"axis_policy_missing", "floor_datum_feature_invalid"} <= set(iss)
    rec = _record(lib)
    rec["datums"].pop("max_axis_disagreement")
    assert "max_axis_disagreement_missing" in parse_operation(rec).issues()["datums"]
    rec["datums"]["axis_policy"] = "perpendicular_offsets"  # needs no orthogonality limit
    assert "datums" not in parse_operation(rec).issues()


def test_exact_match_only(lib):
    assert lib.resolve("LUMEN-DEMO-PLC-101")["status"] == "exact"
    assert lib.resolve("  DEMO-FLAT-01 ")["status"] == "exact"  # whitespace is normalised
    r = lib.resolve("demo-flat-01")  # wrong case: suggestion only, never selected
    assert r["status"] == "none" and r["near"]
    assert lib.resolve("DEMO-FLAT")["status"] == "none"
    assert lib.resolve("")["status"] == "none"
    m = lib.resolve("LUMEN-DEMO-PLC-102")["matches"][0]
    assert (m["operation_id"], m["placard_id"], m["matched"]) == ("DEMO-FLAT-01", "PLC-DEMO-102", "placard_barcode")


def test_ambiguous_code_requires_operator_choice(lib):
    rec = _record(lib)
    rec["operation_id"] = "LOCAL-OTHER-01"
    rec["synthetic"] = False
    rec["panel"]["barcodes"] = ["LUMEN-DEMO-PNL-FLAT"]  # same panel barcode as the bundled record
    lib.save(parse_operation(rec))
    r = lib.resolve("LUMEN-DEMO-PNL-FLAT")
    assert r["status"] == "ambiguous" and {m["operation_id"] for m in r["matches"]} == {"DEMO-FLAT-01", "LOCAL-OTHER-01"}


def test_revisions(lib):
    rec = _record(lib)
    rec.update(operation_id="LOCAL-01", revision="A", synthetic=False)
    lib.save(parse_operation(rec))
    assert lib.get("LOCAL-01").revision == "A"
    # released content is immutable under the same revision
    changed = copy.deepcopy(rec)
    changed["placards"][0]["position"]["x"]["value"] = 999
    with pytest.raises(ConflictError) as e:
        lib.save(parse_operation(changed))
    assert e.value.code == "released_revision_immutable"
    changed["revision"] = "B"
    lib.save(parse_operation(changed))
    assert [o.revision for o in lib.revisions("LOCAL-01")] == ["A", "B"]
    with pytest.raises(ConflictError) as e:
        lib.get("LOCAL-01")  # two released revisions: the operator must choose
    assert e.value.code == "operation_revision_ambiguous"
    assert lib.get("LOCAL-01", "B").placard("PLC-DEMO-101").resolved()["x_mm"] == 999.0
    with pytest.raises(NotFoundError):
        lib.get("LOCAL-01", "Z")
    with pytest.raises(NotFoundError):
        lib.get("NOPE")


def test_incomplete_record_cannot_be_released_and_bundled_cannot_be_overwritten(lib):
    rec = _record(lib)
    rec.update(operation_id="LOCAL-02", synthetic=False)
    rec["placards"][0]["tolerance"] = {}
    with pytest.raises(LumenValidationError) as e:
        lib.save(parse_operation(rec))
    assert e.value.code == "incomplete_record_cannot_be_released"
    rec["status"] = "draft"
    lib.save(parse_operation(rec))  # drafts may be incomplete
    with pytest.raises(ConflictError):
        lib.save(lib.get("DEMO-FLAT-01"))


def test_read_only_library_refuses_writes():
    ro = OperationLibrary(BUNDLED_OPS, None)
    rec = _record(ro)
    rec.update(operation_id="LOCAL-03", synthetic=False)
    with pytest.raises(ConflictError) as e:
        ro.save(parse_operation(rec))
    assert e.value.code == "library_read_only"


def test_broken_local_file_does_not_take_the_library_down(tmp_path):
    (tmp_path / "operations.json").write_text("{ not json")
    lib = OperationLibrary(BUNDLED_OPS, tmp_path)
    assert lib.load_errors and lib.get("DEMO-FLAT-01")


def test_snapshot_is_self_contained_and_hash_tracks_content(lib):
    op = lib.get("DEMO-FLAT-01")
    snap = snapshot(op, lib)
    assert OperationDefinition.model_validate(snap["operation"]).content_hash() == snap["content_hash"] == op.content_hash()
    rec = _record(lib)
    rec["placards"][0]["position"]["x"]["value"] = 241
    assert parse_operation(rec).content_hash() != op.content_hash()


def test_wrong_part_and_wrong_placard(lib):
    op = lib.get("DEMO-FLAT-01")
    with pytest.raises(NotFoundError):
        op.placard("PLC-DEMO-201")  # belongs to another operation
    rec = _record(lib)
    rec["placards"].append(copy.deepcopy(rec["placards"][0]))
    assert "duplicate_placard_id" in parse_operation(rec).issues()["placard:PLC-DEMO-101"]


def test_extra_fields_are_rejected(lib):
    rec = _record(lib)
    rec["placards"][0]["scale_from_barcode_mm"] = 50  # a barcode never supplies physical scale
    with pytest.raises(LumenValidationError):
        parse_operation(rec)


def _local(lib, op_id="LOCAL-10", revision="A", status="released", x=240):
    rec = _record(lib)
    rec.update(operation_id=op_id, revision=revision, status=status, synthetic=False)
    rec["placards"][0]["position"]["x"]["value"] = x
    return parse_operation(rec)


def test_a_new_released_revision_supersedes_the_old_one_explicitly(lib):
    lib.save(_local(lib, revision="A", x=240), author="alice")
    lib.save(_local(lib, revision="B", x=250), author="bob")
    with pytest.raises(ConflictError):
        lib.get("LOCAL-10")
    old = lib.supersede("LOCAL-10", "A", author="bob")
    assert old.status == "superseded" and old.placard("PLC-DEMO-101").resolved()["x_mm"] == 240.0, "superseding changes the status and nothing else"
    assert lib.get("LOCAL-10").revision == "B", "one released revision is left, so no choice is needed"
    assert lib.supersede("LOCAL-10", "A").status == "superseded"  # repeating it is harmless
    # a superseded revision is as immutable as a released one
    with pytest.raises(ConflictError) as e:
        lib.save(_local(lib, revision="A", x=1))
    assert e.value.code == "released_revision_immutable"
    with pytest.raises(ConflictError) as e:
        lib.save(_local(lib, revision="A", x=240))  # even "un-superseding" by saving it as released again
    assert e.value.code == "released_revision_immutable"


def test_only_local_drafts_can_be_deleted_and_only_released_superseded(lib):
    lib.save(_local(lib, "LOCAL-11", "A", "draft"))
    lib.save(_local(lib, "LOCAL-11", "B", "released"))
    with pytest.raises(ConflictError) as e:
        lib.supersede("LOCAL-11", "A")
    assert e.value.code == "only_released_can_be_superseded"
    with pytest.raises(ConflictError) as e:
        lib.delete("LOCAL-11", "B")
    assert e.value.code == "only_drafts_can_be_deleted"
    lib.delete("LOCAL-11", "A")
    assert [o.revision for o in lib.revisions("LOCAL-11")] == ["B"]
    for fn in (lib.supersede, lib.delete):
        with pytest.raises(ConflictError) as e:
            fn("DEMO-FLAT-01", "A")
        assert e.value.code == "bundled_record_immutable"
    with pytest.raises(NotFoundError):
        lib.delete("LOCAL-11", "Z")
    ro = OperationLibrary(BUNDLED_OPS, None)
    for fn in (ro.supersede, ro.delete):
        with pytest.raises(ConflictError) as e:
            fn("DEMO-FLAT-01", "A")
        assert e.value.code == "library_read_only"


def test_every_change_is_recorded_with_who_and_when(lib):
    lib.save(_local(lib, "LOCAL-12", "A", "draft"), author="alice")
    lib.save(_local(lib, "LOCAL-12", "A", "released"), author="alice")
    lib.save(_local(lib, "LOCAL-12", "B", "released", x=260), author="bob")
    lib.supersede("LOCAL-12", "A", author="bob")
    lib.save(_local(lib, "LOCAL-13", "A", "draft"), author="carol")
    lib.delete("LOCAL-13", "A", author="carol")
    a = lib.audit("LOCAL-12")
    assert [(e["action"], e["revision"], e["status"], e["author"]) for e in a] == [
        ("saved", "A", "draft", "alice"), ("saved", "A", "released", "alice"), ("saved", "B", "released", "bob"), ("superseded", "A", "superseded", "bob")]
    assert a[1]["content_hash"] == lib.get("LOCAL-12", "A").model_copy(update={"status": "released"}).content_hash()
    assert all(e["at"].endswith("Z") and len(e["at"]) == 20 for e in a)
    assert [(e["action"], e["operation_id"]) for e in lib.audit()][-2:] == [("saved", "LOCAL-13"), ("deleted", "LOCAL-13")]
    assert OperationLibrary(BUNDLED_OPS, None).audit() == []
    # the trail is part of the same file as the records
    raw = json.loads(lib.overlay_path.read_text())
    assert len(raw["audit"]) == 6 and {o["operation_id"] for o in raw["operations"]} == {"LOCAL-12"}


def test_records_saved_by_another_worker_process_are_seen(tmp_path):
    a = OperationLibrary(BUNDLED_OPS, tmp_path / "local")
    b = OperationLibrary(BUNDLED_OPS, tmp_path / "local")  # a second process with its own memory
    a.save(_local(a, "LOCAL-20", "A"))
    b._checked = 0.0  # the file is looked at no more than twice a second
    assert b.get("LOCAL-20").revision == "A" and b.resolve("LOCAL-20")["status"] == "exact"
    # a save made from stale memory does not drop what the other process stored
    b.save(_local(b, "LOCAL-21", "A"))
    a._checked = 0.0
    assert {o.operation_id for o in a.all()} >= {"LOCAL-20", "LOCAL-21"}


def test_simultaneous_saves_from_several_processes_lose_nothing(tmp_path):
    import threading

    libs = [OperationLibrary(BUNDLED_OPS, tmp_path / "local") for _ in range(4)]
    errors: list[Exception] = []

    def work(k: int) -> None:
        try:
            for j in range(6):
                libs[k].save(_local(libs[k], f"LOCAL-P{k}-{j}", "A"), author=f"p{k}")
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    fresh = OperationLibrary(BUNDLED_OPS, tmp_path / "local")
    assert len([o for o in fresh.all() if o.operation_id.startswith("LOCAL-P")]) == 24 and len(fresh.audit()) == 24
    assert not list((tmp_path / "local").glob("*.tmp"))


def test_a_code_means_the_released_revision_not_its_drafts_or_superseded_ones(lib):
    lib.save(_local(lib, "LOCAL-30", "A", "released"))
    lib.save(_local(lib, "LOCAL-30", "B", "released", x=250))
    # two released revisions: the operator must choose
    r = lib.resolve("LOCAL-30")
    assert r["status"] == "ambiguous" and {m["revision"] for m in r["matches"]} == {"A", "B"}
    lib.supersede("LOCAL-30", "A")
    lib.save(_local(lib, "LOCAL-30", "C", "draft", x=260))
    r = lib.resolve("LOCAL-30")
    assert r["status"] == "exact" and {m["revision"] for m in r["matches"]} == {"B"}
    assert {(m["revision"], m["status"]) for m in r["other_revisions"]} == {("A", "superseded"), ("C", "draft")}
    # only a draft exists: it is found, as a draft
    lib.save(_local(lib, "LOCAL-31", "A", "draft"))
    r = lib.resolve("LOCAL-31")
    assert r["status"] == "exact" and r["matches"][0]["status"] == "draft"
    # the same code on two different operations stays ambiguous, whatever their status
    r = lib.resolve("LUMEN-DEMO-PNL-FLAT")
    assert r["status"] == "ambiguous" and {m["operation_id"] for m in r["matches"]} == {"DEMO-FLAT-01", "LOCAL-30", "LOCAL-31"}
    assert lib.resolve("")["status"] == "none"
