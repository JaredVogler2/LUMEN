"""The look of the application and the brand of an installation, in a real browser.

The logo in these tests is a drawing made here.  No organisation's artwork is part of the repository: whoever
deploys LUMEN supplies their own file (brand/README.md).
"""

import pytest

from tests.browser.conftest import LiveServer, make_app, open_app, shot

pytestmark = pytest.mark.browser

WIDE = {"viewport": {"width": 1280, "height": 900}, "is_mobile": False, "has_touch": False}
SAMPLE_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 60"><circle cx="30" cy="30" r="22" fill="none" stroke="#7a1f2b" stroke-width="6"/>'
    '<text x="64" y="41" font-family="Arial, Helvetica, sans-serif" font-size="32" font-weight="700" fill="#7a1f2b">SAMPLE</text></svg>'
)
# contrast of two CSS colours as WCAG defines it (4.5 is the least for text)
CONTRAST = """([a, b]) => { const lum = (c) => { const v = c.match(/\\d+(\\.\\d+)?/g).slice(0, 3).map(Number).map((x) => { x /= 255; return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); });
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; }; const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); }"""
STYLE = "([sel, prop]) => getComputedStyle(document.querySelector(sel))[prop]"
VAR = "(name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()"


def test_the_default_look_is_light_with_the_brand_blue_and_dark_can_be_chosen(page):
    assert page.evaluate("() => document.documentElement.dataset.theme") == "light"
    assert page.evaluate(VAR, "--brand") == "#0033a0"
    # white surfaces, a rule in the brand colour under the header, LUMEN's own mark (there is no logo)
    assert page.evaluate(STYLE, [".app-header", "backgroundColor"]) == "rgb(255, 255, 255)"
    assert page.evaluate(STYLE, [".app-header", "borderBottomColor"]) == "rgb(0, 51, 160)"
    assert page.locator('[data-region="brand-logo"]').count() == 0 and page.locator('[data-region="brand"] svg').count() == 1
    assert "LUMEN" in page.inner_text('[data-region="brand"]')
    # the action of a screen is one filled button in that colour, with text that can be read on it
    page.click('[data-operation="DEMO-FLAT-01"]')
    page.wait_for_selector('[data-action="continue"]:not([disabled])')
    bg, ink = page.evaluate(STYLE, ['[data-action="continue"]', "backgroundColor"]), page.evaluate(STYLE, ['[data-action="continue"]', "color"])
    assert bg == "rgb(0, 51, 160)" and page.evaluate(CONTRAST, [bg, ink]) >= 7.0
    # body text on the page
    assert page.evaluate(CONTRAST, [page.evaluate(STYLE, ["body", "color"]), page.evaluate(STYLE, ["body", "backgroundColor"])]) >= 7.0
    # dark is a choice in the menu, and it is remembered
    page.click('[data-action="menu"]')
    page.click("text=Dark mode")
    assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"
    bg, ink = page.evaluate(STYLE, ['[data-action="continue"]', "backgroundColor"]), page.evaluate(STYLE, ['[data-action="continue"]', "color"])
    assert page.evaluate(CONTRAST, [bg, ink]) >= 4.5
    assert page.evaluate(CONTRAST, [page.evaluate(STYLE, ["body", "color"]), page.evaluate(STYLE, ["body", "backgroundColor"])]) >= 7.0
    page.reload()
    page.wait_for_selector('[data-step="identify"]')
    assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"
    assert not page._lumen_errors


def test_an_installation_shows_its_own_logo_name_and_colour(browser, tmp_path):
    logo = tmp_path / "logo.svg"
    logo.write_text(SAMPLE_LOGO)
    srv = LiveServer(make_app(brand_name="Sample Aerospace", brand_logo=str(logo), brand_color="#7A1F2B")).start()
    try:
        page = open_app(browser, srv.url)
        img = page.wait_for_selector('[data-region="brand-logo"]')
        page.wait_for_function("() => { const i = document.querySelector('[data-region=\"brand-logo\"]'); return i.complete && i.naturalWidth > 0; }")
        assert img.get_attribute("alt") == "Sample Aerospace logo"
        box = img.bounding_box()
        assert 20 <= box["height"] <= 36 and box["width"] <= 140 and box["x"] < 40, box
        # the logo stands for the product on a phone: no mark of LUMEN beside it, and the header fits the screen
        assert page.locator('[data-region="brand"] svg').count() == 0
        assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        # the installation's colour: the rule under the header and the filled button
        assert page.evaluate(VAR, "--brand") == "#7a1f2b"
        assert page.evaluate(STYLE, [".app-header", "borderBottomColor"]) == "rgb(122, 31, 43)"
        page.click('[data-operation="DEMO-FLAT-01"]')
        page.wait_for_selector('[data-action="continue"]:not([disabled])')
        bg, ink = page.evaluate(STYLE, ['[data-action="continue"]', "backgroundColor"]), page.evaluate(STYLE, ['[data-action="continue"]', "color"])
        assert bg == "rgb(122, 31, 43)" and page.evaluate(CONTRAST, [bg, ink]) >= 4.5
        shot(page, "branding_sample")
        # through the whole header nothing was fetched from anywhere else and nothing failed
        assert not page._lumen_errors, page._lumen_errors
        page.context.close()
        # on a wide screen the product is named beside the logo
        page = open_app(browser, srv.url, **WIDE)
        page.wait_for_selector('[data-region="brand-logo"]')
        assert page.is_visible('[data-region="brand"] .product') and page.inner_text('[data-region="brand"] .product') == "LUMEN"
        assert not page._lumen_errors, page._lumen_errors
        page.context.close()
    finally:
        srv.stop()


def test_a_name_without_a_logo_stands_beside_the_mark_and_a_light_colour_stays_readable(browser):
    srv = LiveServer(make_app(brand_name="Sample Aerospace", brand_color="#ffd200")).start()
    try:
        page = open_app(browser, srv.url)
        assert page.locator('[data-region="brand-logo"]').count() == 0 and page.locator('[data-region="brand"] svg').count() == 1
        # before a job is chosen a phone shows the organisation's name beside the mark
        assert page.is_visible('[data-region="brand"] .brand-name') and page.inner_text('[data-region="brand"] .brand-name') == "Sample Aerospace"
        assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        # the colour is used where it is a surface (the rule, the mark) ...
        assert page.evaluate(VAR, "--brand") == "#ffd200"
        assert page.evaluate(STYLE, [".app-header", "borderBottomColor"]) == "rgb(255, 210, 0)"
        # ... and what is drawn on it can be read
        assert page.evaluate(CONTRAST, ["rgb(255, 210, 0)", page.evaluate(STYLE, ['[data-region="brand"] svg path', "stroke"])]) >= 4.5
        # text and buttons take a darker shade of it: a light colour on white could not be read
        name = page.evaluate(STYLE, ['[data-region="brand"] .brand-name', "color"])
        assert page.evaluate(CONTRAST, [name, "rgb(255, 255, 255)"]) >= 4.5, name
        page.click('[data-operation="DEMO-FLAT-01"]')
        page.wait_for_selector('[data-action="continue"]:not([disabled])')
        bg, ink = page.evaluate(STYLE, ['[data-action="continue"]', "backgroundColor"]), page.evaluate(STYLE, ['[data-action="continue"]', "color"])
        assert bg != "rgb(255, 210, 0)" and page.evaluate(CONTRAST, [bg, ink]) >= 4.5, (bg, ink)
        assert not page._lumen_errors, page._lumen_errors
        page.context.close()
    finally:
        srv.stop()
