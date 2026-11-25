"""The whole application behind a path-stripping proxy, in a real browser (prompt 14).

This simulates how Posit Connect serves a content item under
``/content/<guid>/``.  It is a simulation: no Connect server was used.
"""

import pytest

from tests.browser.conftest import LiveServer, PHONE, load_demo_to_place, make_app, shot
from tests.conftest import PrefixProxy

pytestmark = pytest.mark.browser
PREFIX = "/content/3f2a9c1e"


@pytest.mark.parametrize("announce,slash", [(True, True), (False, True), (False, False), (True, False)], ids=["header_slash", "noheader_slash", "noheader_noslash", "header_noslash"])
def test_demo_walkthrough_under_a_sub_path(browser, announce, slash):
    proxy = PrefixProxy(make_app(), PREFIX, announce=announce)
    srv = LiveServer(proxy).start()
    try:
        ctx = browser.new_context(**PHONE)
        page = ctx.new_page()
        errors, failed = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "favicon" not in m.text else None)
        page.on("requestfailed", lambda r: failed.append(r.url))
        page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 and "favicon" not in r.url else None)
        page.goto(srv.url.rstrip("/") + PREFIX + ("/" if slash else ""))
        page.wait_for_selector('[data-step="identify"]')
        load_demo_to_place(page)
        assert page.get_attribute(".status-hero", "data-status") == "prototype_guidance"
        # every request stayed inside the content path
        assert proxy.seen and all(not p.startswith("/content") for p in proxy.seen)
        outside = page.evaluate("() => performance.getEntriesByType('resource').map((e) => new URL(e.name).pathname).filter((p) => !p.startsWith('%s/'))" % PREFIX)
        assert outside == [], outside
        assert not failed, failed
        assert not errors, errors
        if announce and slash:
            shot(page, "subpath_place")
        ctx.close()
    finally:
        srv.stop()
