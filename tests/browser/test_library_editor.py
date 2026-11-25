"""Entering, releasing and revising an operation record in the browser (prompt 08)."""

import json

import pytest

from tests.browser.conftest import PHONE, LiveServer, assert_clean_text, shot

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def editable(tmp_path_factory):
    from lumen.api.app import create_app
    from lumen.operations.library import OperationLibrary
    from lumen.services.sessions import SessionService
    from lumen.storage.base import MemorySessionStore
    from tests.conftest import BUNDLED_OPS, make_settings

    lib = OperationLibrary(BUNDLED_OPS, tmp_path_factory.mktemp("library"))
    lib.persistent = False
    settings = make_settings()
    srv = LiveServer(create_app(settings, SessionService(MemorySessionStore(), settings, lib))).start()
    yield srv, lib
    srv.stop()


@pytest.fixture()
def epage(browser, editable):
    ctx = browser.new_context(**PHONE, accept_downloads=True)
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # a refused record (422) is an expected answer while typing, not a page error
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "favicon" not in m.text and "422" not in m.text else None)
    pg._lumen_errors = errors
    pg.goto(editable[0].url)
    pg.wait_for_selector('[data-step="identify"]')
    yield pg
    ctx.close()


def open_section(page, key):
    if page.get_attribute(f'[data-section="{key}"]', "open") is None:
        page.click(f'[data-section="{key}"] > summary')


def status_text(page):
    return page.inner_text('[data-region="editor-status"]').lower()


def wait_checked(page, needle):
    page.wait_for_function("(n) => document.querySelector('[data-region=\"editor-status\"]').innerText.toLowerCase().includes(n)", arg=needle, timeout=20_000)


def test_read_only_deployment_says_so(page):
    assert "cannot be entered here" in page.inner_text('[data-region="library"]')
    assert page.query_selector('[data-action="op-new"]') is None
    page.click('[data-operation="DEMO-FLAT-01"]')
    page.wait_for_selector('[data-placard="PLC-DEMO-101"]')
    assert page.query_selector('[data-region="record-actions"]') is None


def test_a_record_is_entered_checked_released_and_revised(epage, editable):
    page, lib = epage, editable[1]
    assert "temporary storage" in page.inner_text('[data-region="library"]') and "Export" in page.inner_text('[data-region="library"]')
    page.click('[data-action="op-new"]')
    page.wait_for_selector('[data-region="operation-editor"]')
    wait_checked(page, "not complete")
    # nothing is pre-filled: every requirement is reported as missing
    for needle in ("no floor datum", "no side datum", "axis rule not stated", "no x offset", "no position tolerance", "no rotation tolerance", "no size"):
        assert needle in status_text(page), needle
    assert "cannot be saved yet" in status_text(page) and "operation id: must be 1 to 64 characters" in status_text(page)
    assert page.is_disabled('[data-action="op-release"]') and page.is_disabled('[data-action="op-save-draft"]'), "no ID yet"
    page.fill('[data-field="operation_id"]', "LOCAL-T1")
    page.fill('[data-field="title"]', "Browser-entered record")
    page.fill('[data-field="source"]', "test sheet 1")
    open_section(page, "panel")
    page.fill('[data-field="panel_id"]', "PNL-T1")
    page.fill('[data-field="panel.barcodes"]', "PNL-T1-CODE, PNL-T1-ALT")
    for key, name in (("floor", "floor line"), ("side", "left seam")):
        open_section(page, key)
        page.fill(f'[data-field="{key}.name"]', name)
        page.select_option(f'[data-field="{key}.feature"]', "planar_edge")
        assert page.get_attribute(f'[data-section="{key}"]', "open") is not None, "the section stays open when the form is rebuilt"
        assert page.input_value(f'[data-field="{key}.name"]') == name
        page.select_option(f'[data-field="{key}.on_target_plane"]', "true")
        page.select_option(f'[data-field="{key}.positive"]', "toward_panel")
        page.fill(f'[data-field="{key}.offset"]', "0")
    page.select_option('[data-field="side.side_of_panel"]', "left")
    open_section(page, "axes")
    page.select_option('[data-field="axis_policy"]', "floor_primary")
    page.fill('[data-field="nominal_angle"]', "90")
    page.fill('[data-field="max_axis_disagreement"]', "1")
    open_section(page, "placard-0")
    page.fill('[data-field="placard.0.placard_id"]', "PLC-T1")
    page.fill('[data-field="placard.0.x"]', "9.5")
    page.select_option('[data-unit="placard.0.x"]', "in")
    page.fill('[data-field="placard.0.z"]', "300")
    page.select_option('[data-field="placard.0.reference_point"]', "center")
    page.select_option('[data-field="placard.0.x_semantics"]', "projected")
    page.fill('[data-field="placard.0.width"]', "120")
    page.fill('[data-field="placard.0.height"]', "60")
    page.fill('[data-field="placard.0.rotation"]', "0")
    # the value entered in inches is shown as the millimetres the measurement will use
    page.wait_for_function("() => (document.querySelector('[data-resolved=\"0\"]').textContent || '').includes('x 241.30')", timeout=20_000)
    wait_checked(page, "no position tolerance")
    assert "no x offset" not in status_text(page) and "no floor datum" not in status_text(page)
    assert page.is_disabled('[data-action="op-release"]') and not page.is_disabled('[data-action="op-save-draft"]')
    shot(page, "library_editor_incomplete")
    page.click('[data-action="op-save-draft"]')
    page.wait_for_selector('[data-region="operation-editor"]', state="detached")
    card = page.wait_for_selector('[data-operation="LOCAL-T1"][data-revision="A"]')
    assert "draft" in card.inner_text() and "entered here" in card.inner_text() and "incomplete" in card.inner_text()
    saved = lib.get("LOCAL-T1", "A")
    assert saved.status == "draft" and saved.panel.barcodes == ["PNL-T1-CODE", "PNL-T1-ALT"] and saved.synthetic is False
    assert saved.placards[0].position.x.unit == "in" and saved.placards[0].tolerance.position is None, "the missing tolerance was stored as missing"
    assert saved.datums.floor.side_of_panel is None and saved.datums.side.side_of_panel == "left"

    # complete the draft and release it
    page.wait_for_selector('[data-action="op-edit"]')
    page.click('[data-action="op-edit"]')
    page.wait_for_selector('[data-region="operation-editor"]')
    assert page.input_value('[data-field="operation_id"]') == "LOCAL-T1" and page.input_value('[data-field="placard.0.x"]') == "9.5"
    assert page.input_value('[data-unit="placard.0.x"]') == "in"
    page.fill('[data-field="placard.0.tol_position"]', "2")
    page.fill('[data-field="placard.0.tol_rotation"]', "1")
    page.fill('[data-field="placard.0.tol_provenance"]', "test sheet 1, table 2")
    wait_checked(page, "complete")
    page.wait_for_selector('[data-action="op-release"]:not([disabled])')
    page.click('[data-action="op-release"]')
    assert "cannot be changed afterwards" in page.inner_text('[data-region="operation-editor"]')
    shot(page, "library_editor_release")
    page.click('[data-action="op-release-confirm"]')
    page.wait_for_selector('[data-region="operation-editor"]', state="detached")
    page.wait_for_selector('[data-action="op-revise"]')
    assert page.query_selector('[data-action="op-edit"]') is None, "a released revision is not edited"
    assert lib.get("LOCAL-T1").status == "released" and lib.get("LOCAL-T1").placards[0].resolved()["x_mm"] == pytest.approx(241.3)
    assert_clean_text(page)

    # a correction is revision B; A is marked superseded on release
    page.click('[data-action="op-revise"]')
    page.wait_for_selector('[data-region="operation-editor"]')
    assert page.input_value('[data-field="revision"]') == "B"
    open_section(page, "placard-0")
    page.fill('[data-field="placard.0.z"]', "310")
    wait_checked(page, "complete")
    page.click('[data-action="op-release"]')
    assert page.is_checked('[data-field="supersede"]')
    page.click('[data-action="op-release-confirm"]')
    page.wait_for_selector('[data-region="operation-editor"]', state="detached")
    page.wait_for_selector('[data-operation="LOCAL-T1"][data-revision="B"]')
    assert "superseded" in page.inner_text('[data-operation="LOCAL-T1"][data-revision="A"]')
    assert "released" in page.inner_text('[data-operation="LOCAL-T1"][data-revision="B"]')
    assert lib.get("LOCAL-T1").revision == "B" and lib.get("LOCAL-T1", "A").status == "superseded"
    assert [(e["action"], e["revision"]) for e in lib.audit("LOCAL-T1")] == [("saved", "A"), ("saved", "A"), ("saved", "B"), ("superseded", "A")]
    # the scanner / search finds the new record by its panel code, exactly
    page.fill('input[aria-label="Search operations"]', "PNL-T1-CODE")
    page.keyboard.press("Enter")
    page.wait_for_function("() => document.querySelector('[data-operation=\"LOCAL-T1\"][data-revision=\"B\"]')?.getAttribute('aria-pressed') === 'true'")
    assert not page._lumen_errors, page._lumen_errors


def test_copy_of_a_sample_json_view_export_and_refusals(epage, editable):
    page, lib = epage, editable[1]
    page.click('[data-operation="DEMO-FLAT-01"]')
    page.wait_for_selector('[data-action="op-copy"]')
    assert page.query_selector('[data-action="op-edit"]') is None and page.query_selector('[data-action="op-revise"]') is None, "bundled samples are not edited"
    page.click('[data-action="op-copy"]')
    page.wait_for_selector('[data-region="operation-editor"]')
    # the copy has no ID, no codes and no tolerance source of its own yet
    assert page.input_value('[data-field="operation_id"]') == "" and page.input_value('[data-field="revision"]') == "A"
    assert "Copied from DEMO-FLAT-01" in page.input_value('[data-field="provenance.notes"]')
    # the ID of a bundled record is refused while typing
    page.fill('[data-field="operation_id"]', "DEMO-FLAT-01")
    wait_checked(page, "bundled sample record")
    assert page.is_disabled('[data-action="op-save-draft"]')
    # an invalid ID is explained, not silently fixed
    page.fill('[data-field="operation_id"]', "bad id!")
    wait_checked(page, "operation id: must be 1 to 64 characters")
    assert page.is_disabled('[data-action="op-save-draft"]')
    page.fill('[data-field="operation_id"]', "LOCAL-COPY")
    wait_checked(page, "complete")
    # JSON view shows the same record and takes edits back into the form
    page.click('[data-action="op-json"]')
    rec = json.loads(page.input_value('[data-field="json"]'))
    assert rec["operation_id"] == "LOCAL-COPY" and rec["synthetic"] is False and rec["panel"]["barcodes"] == [] and len(rec["placards"]) == 2
    rec["title"] = "Edited as JSON"
    rec["placards"][0]["position"]["x"] = {"value": 123.5, "unit": "mm"}
    page.fill('[data-field="json"]', json.dumps(rec))
    page.click('[data-action="op-json-apply"]')
    page.wait_for_selector('[data-field="title"]')
    assert page.input_value('[data-field="title"]') == "Edited as JSON"
    open_section(page, "placard-0")
    assert page.input_value('[data-field="placard.0.x"]') == "123.5"
    page.click('[data-action="op-json"]')
    page.fill('[data-field="json"]', "{ not json")
    page.click('[data-action="op-json-apply"]')
    page.wait_for_selector(".toast.error")
    assert "not a JSON record" in page.inner_text(".toasts")
    page.click('.toast [aria-label="Dismiss"]')
    page.click('[data-action="op-json"]')
    assert page.input_value('[data-field="title"]') == "Edited as JSON", "a failed paste changes nothing"
    with page.expect_download() as dl:
        page.click('[data-action="op-export"]')
    exported = json.loads(open(dl.value.path()).read())
    assert dl.value.suggested_filename == "LOCAL-COPY_revA.json" and exported["placards"][0]["position"]["x"] == {"value": 123.5, "unit": "mm"}
    wait_checked(page, "complete")
    page.click('[data-action="op-save-draft"]')
    page.wait_for_selector('[data-region="operation-editor"]', state="detached")
    page.wait_for_selector('[data-operation="LOCAL-COPY"]')
    assert lib.get("LOCAL-COPY", "A").title == "Edited as JSON"
    # a draft can be deleted again
    page.click('[data-action="op-edit"]')
    page.wait_for_selector('[data-action="op-delete"]')
    page.click('[data-action="op-delete"]')
    page.wait_for_selector('[data-operation="LOCAL-COPY"]', state="detached")
    assert not [o for o in lib.all() if o.operation_id == "LOCAL-COPY"]
    assert_clean_text(page)
    assert not page._lumen_errors, page._lumen_errors
