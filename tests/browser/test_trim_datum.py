"""A raised edge beside the automatically found floor line: the operator is asked (prompts 06, 10)."""

from dataclasses import replace

import pytest

from lumen.synth import TrimSpec
from lumen.synth.fixtures import flat_panel_scene
from tests.browser.conftest import assert_clean_text, open_app, shot
from tests.conftest import calibration_for, jpeg_bytes, render_cached
from tests.integration.test_datums_flow import MEASURED, sweep

pytestmark = pytest.mark.browser


def test_a_kick_strip_makes_the_app_ask_which_line_is_meant(browser, live_server):
    httpx = pytest.importorskip("httpx")
    # a panel with a kick strip (60 mm tall, 8 mm proud) that hides the floor line; photos sent as an operator would
    scene = replace(flat_panel_scene(), trim=TrimSpec(height=60.0, thickness=8.0))
    cams = sweep()
    with httpx.Client(base_url=live_server.url, timeout=600) as c:
        sid = c.post("api/sessions", json={}).json()["id"]
        c.put(f"api/sessions/{sid}/operation", json={"operation_id": "DEMO-FLAT-01", "placard_id": "PLC-DEMO-101"})
        c.put(f"api/sessions/{sid}/tags", json=MEASURED)
        c.put(f"api/sessions/{sid}/calibration", json={"profile": calibration_for(cams[0])})
        for i, cam in enumerate(cams):
            assert c.post(f"api/sessions/{sid}/frames", content=jpeg_bytes(render_cached(scene, cam, seed=i).image), headers={"content-type": "image/jpeg"}).status_code == 200
        assert c.post(f"api/sessions/{sid}/register").status_code == 200
        applied = c.post(f"api/sessions/{sid}/datums/propose").json()["proposals"]["applied"]

        page = open_app(browser, live_server.url, init_script=f"sessionStorage.setItem('lumen.sid', '{sid}')")
        if page.query_selector('.toast [aria-label="Dismiss"]'):
            page.click('.toast [aria-label="Dismiss"]')  # "Session restored" (it would cover the header in the pictures)
        page.click('[data-step="datums"]')
        note = page.wait_for_selector('[data-region="step-beside-floor"]', timeout=60_000).inner_text()
        assert "Is this the right line?" in note and "8 mm in front of the panel face" in note and "60 mm beyond" in note
        assert "Choose line" in note and "trim" in note
        assert page.query_selector('[data-region="step-beside-side"]') is None, "the side seam has nothing beside it"
        shot(page, "datums_trim_photo")
        page.locator('[data-region="step-beside-floor"]').scroll_into_view_if_needed()
        shot(page, "datums_trim_question")
        assert_clean_text(page)
        # confirming the lines as they are does not make it guidance: the check stays open and says why
        page.click('[data-action="confirm-datums"]')
        page.wait_for_selector('[data-step="place"][aria-current="step"]', timeout=60_000)
        hero = page.wait_for_selector('.status-hero[data-status="research_preview"]', timeout=120_000).inner_text().lower()
        assert hero.startswith("check needed") and "at least one check is open" in hero and "not validated" in hero
        gate = page.wait_for_selector('[data-gate="datum_floor_no_step_beside"]')
        assert "fail" in gate.get_attribute("class") and "may be the edge of a trim" in gate.inner_text()
        # scale rests on the entered sizes only: said as a note ("i"), not ticked and not counted as open
        scale = page.wait_for_selector('[data-gate="scale_source"]')
        assert "caveat" in scale.get_attribute("class") and "pass" in scale.get_attribute("class")
        assert "declared tag sizes alone" in scale.inner_text() and "rule" in scale.inner_text()
        gate.scroll_into_view_if_needed()
        shot(page, "place_trim_checks")
        assert_clean_text(page)

        # the operator decides that this line is the one the record means (the same call the tap on the line makes)
        assert c.post(f"api/sessions/{sid}/datums/select", json={"kind": "floor", "candidate_id": applied["floor"]}).status_code == 200
        c.post(f"api/sessions/{sid}/datums/confirm")
        page.reload()
        page.wait_for_selector('[data-step="identify"]')
        page.click('[data-step="datums"]')
        page.wait_for_selector('[data-datum="floor"]')
        assert page.query_selector('[data-region="step-beside-floor"]') is None
        assert "chosen by the operator" in page.inner_text('[data-datum="floor"]')
        page.click('[data-step="place"]')
        page.wait_for_selector(".status-hero", timeout=120_000)
        page.wait_for_selector(".busy", state="detached", timeout=120_000)
        assert page.query_selector('[data-gate="datum_floor_no_step_beside"]') is None
        assert not page._lumen_errors, page._lumen_errors
        page.context.close()
