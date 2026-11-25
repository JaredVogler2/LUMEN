"""The optional assistant in the interface (prompt 12): absent when off; with a mock adapter when on."""

import pytest

from tests.browser.conftest import PHONE, LiveServer, assert_clean_text, load_demo_to_place, shot

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def assistant_server():
    from lumen.api.app import create_app
    from lumen.assistant.adapter import MockAdapter
    from lumen.operations.library import OperationLibrary
    from lumen.services.sessions import SessionService
    from lumen.storage.base import MemorySessionStore
    from tests.conftest import BUNDLED_OPS, make_settings

    settings = make_settings(llm_enabled=True)
    app = create_app(settings, SessionService(MemorySessionStore(), settings, OperationLibrary(BUNDLED_OPS)))
    mock = MockAdapter()
    app.state.assistant_adapter = mock
    srv = LiveServer(app).start()
    yield srv, mock
    srv.stop()


def open_diagnostics(page):
    page.click('[data-action="menu"]')
    page.click("text=Diagnostics")
    page.wait_for_selector(".sheet table.budget")


def test_assistant_is_not_offered_unless_enabled(page):
    load_demo_to_place(page)
    open_diagnostics(page)
    assert page.query_selector('[data-region="assistant"]') is None
    # how the tag borders were corrected is part of the diagnostics
    assert "Edge shift of tag borders" in page.inner_text(".sheet") and "with their own estimate" in page.inner_text(".sheet")
    assert_clean_text(page)


def test_assistant_explains_the_report_and_says_what_was_sent(browser, assistant_server):
    srv, mock = assistant_server
    ctx = browser.new_context(**PHONE)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "favicon" not in m.text else None)
    page.goto(srv.url)
    page.wait_for_selector('[data-step="identify"]')
    load_demo_to_place(page)
    open_diagnostics(page)
    card = page.wait_for_selector('[data-region="assistant"]')
    assert "not a measurement and not an acceptance decision" in card.inner_text()
    page.fill('[data-field="assistant-question"]', "What does this status mean?")
    page.click('[data-action="assistant-explain"]')
    page.wait_for_selector('[data-region="assistant-reply"] .banner')
    reply = page.inner_text('[data-region="assistant-reply"]')
    assert "mock adapter" in reply and "not an acceptance decision" in reply
    assert "LUMEN reports status 'prototype_guidance'" in reply
    assert "No photo and no session ID were sent" in reply
    # what really left the page and the server: the question and the computed report, no picture, no session id
    sent = mock.calls[-1]["user"]
    sid = page.evaluate("async () => (await import('./static/js/store.js')).state.session.id")
    assert "What does this status mean?" in sent and sid not in sent and "base64" not in sent and "/9j/" not in sent
    shot(page, "assistant_reply")
    assert_clean_text(page)
    assert not errors, errors
    ctx.close()
