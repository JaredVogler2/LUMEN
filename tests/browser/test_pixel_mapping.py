"""Display <-> image pixel mapping (prompts 02 and 09).

The overlay arrives in image pixel coordinates; the viewer is the only place
that turns them into screen positions.  These tests check the transform
functions themselves and then prove, on a screenshot, that an overlay point
lands on the image feature it belongs to.
"""

import cv2
import numpy as np
import pytest

from tests.browser.conftest import DESKTOP, load_demo_to_place

pytestmark = pytest.mark.browser


def test_transform_functions_follow_the_pixel_centre_convention(page):
    out = page.evaluate(
        """async () => {
            const v = await import('./static/js/viewer.js');
            const t = v.fitTransform(1600, 1200, 390, 500, 0);
            return {
              t,
              topLeftCorner: v.frameToView(t, -0.5, -0.5),
              bottomRightCorner: v.frameToView(t, 1599.5, 1199.5),
              centreOfFirstPixel: v.frameToView(t, 0, 0),
              back: v.viewToFrame(t, ...v.frameToView(t, 123.25, 987.75)),
              wide: v.fitTransform(1600, 1200, 1000, 300, 0),
            };
        }"""
    )
    t = out["t"]
    assert t["s"] == pytest.approx(390 / 1600) and t["tx"] == pytest.approx(0) and t["ty"] == pytest.approx((500 - 1200 * t["s"]) / 2)
    assert out["topLeftCorner"] == pytest.approx([t["tx"], t["ty"]])
    assert out["bottomRightCorner"] == pytest.approx([t["tx"] + 1600 * t["s"], t["ty"] + 1200 * t["s"]])
    assert out["centreOfFirstPixel"] == pytest.approx([t["tx"] + 0.5 * t["s"], t["ty"] + 0.5 * t["s"]])
    assert out["back"] == pytest.approx([123.25, 987.75])
    assert out["wide"]["s"] == pytest.approx(300 / 1200) and out["wide"]["tx"] == pytest.approx((1000 - 400) / 2)


def test_overlay_points_land_on_the_image_features_they_describe(browser, live_server):
    ctx = browser.new_context(**DESKTOP)
    page = ctx.new_page()
    page.goto(live_server.url)
    page.wait_for_selector('[data-step="identify"]')
    load_demo_to_place(page)
    # zoom onto one tag so it is large on screen, hide the overlay, take a screenshot of the canvas
    info = page.evaluate(
        """async () => {
            const v = window.__lumenViewer;
            const tag = v.overlay.primitives.find(p => p.kind === 'tag');
            const cx = tag.points.reduce((a, p) => a + p[0], 0) / 4, cy = tag.points.reduce((a, p) => a + p[1], 0) / 4;
            v.fit(); v.centerOn(cx, cy, 6.0);
            v.setLayer('tags', false); v.setLayer('grid', false); v.setLayer('dims', false);
            const saved = v.overlay; v.overlay = { primitives: [] }; v._draw();
            const r = v.canvas.getBoundingClientRect();
            return { corners: tag.points.map(p => v.toView(p[0], p[1])), label: tag.label, rect: [r.left, r.top, r.width, r.height], scale: v.t.s };
        }"""
    )
    page.wait_for_timeout(300)
    png = page.locator(".viewer-wrap canvas").screenshot()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    # measure the tag in the screenshot with LUMEN's own unbiased corner fit
    # (the stock detector is ~1 image pixel off, which is several screen pixels here)
    from lumen.vision.markers import detect_tags

    want_id = int(info["label"].split()[-1])
    dets = [d for d in detect_tags(img) if d.tag_id == want_id and d.refined]
    assert dets, "the magnified tag must be measurable in the screenshot"
    seen = dets[0].corners_px
    predicted = np.array(info["corners"])
    # screenshot pixel (i, j) is the canvas point (i + 0.5, j + 0.5) in CSS pixels
    err = np.linalg.norm((seen + 0.5) - predicted, axis=1)
    assert info["scale"] > 3.0, "the tag must be magnified so that half an image pixel is several screen pixels"
    # half an image pixel of mapping error would be > 1.5 screen pixels at this zoom
    assert err.max() < 1.0, err
    ctx.close()


def test_target_outline_is_where_the_server_put_it(page):
    load_demo_to_place(page)
    info = page.evaluate(
        """() => { const v = window.__lumenViewer; const t = v.overlay.primitives.find(p => p.kind === 'target' && p.active !== false);
                   const r = v.canvas.getBoundingClientRect();
                   return { corners: t.corners.map(c => v.toView(c[0], c[1])), w: r.width, h: r.height }; }"""
    )
    c = np.array(info["corners"])
    assert np.all(c[:, 0] > 0) and np.all(c[:, 0] < info["w"]) and np.all(c[:, 1] > 0) and np.all(c[:, 1] < info["h"]), "the screen opens on the target"
    # the outline is drawn in the target colour at those positions
    png = page.locator(".viewer-wrap canvas").screenshot()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    mid = 0.5 * (c[0] + c[1])  # middle of the top edge
    patch = img[int(mid[1]) - 3 : int(mid[1]) + 4, int(mid[0]) - 3 : int(mid[0]) + 4].reshape(-1, 3).astype(int)
    b, g, r = patch[np.argmax(patch[:, 2] - patch[:, 1])]
    assert r > 180 and g < 140 and b > 120, (b, g, r)  # magenta-ish stroke
