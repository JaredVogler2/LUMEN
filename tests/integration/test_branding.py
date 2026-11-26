"""The brand of an installation: its name, its logo and its colour are settings of a deployment.

No artwork of any organisation is part of this repository.  A logo is a file that whoever deploys LUMEN supplies
(``LUMEN_BRAND_LOGO``, or ``brand/logo.*`` beside the application); the tests use a drawing of their own.
"""

import html as html_mod
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import lumen.config as config_mod
from lumen.api.app import create_app
from lumen.config import BRAND_LOGO_MAX_BYTES, bundled_brand_logo, load_settings
from lumen.operations.library import OperationLibrary
from lumen.services.sessions import SessionService
from lumen.storage.base import MemorySessionStore
from tests.conftest import BUNDLED_OPS, PrefixProxy, make_settings

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "/content/3f2a9c1e"
SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 40"><rect width="120" height="40" fill="#123456"/><script>window.x = 1</script></svg>'
# the smallest PNG there is (one transparent pixel)
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6260000000000500010d0a2db40000000049454e44ae426082"
)


def app_with(**kw):
    settings = make_settings(**kw)
    return create_app(settings, SessionService(MemorySessionStore(), settings, OperationLibrary(BUNDLED_OPS)))


def boot_of(page: str) -> dict:
    """What the page is told before its first request (the header is drawn from it)."""
    m = re.search(r'<meta name="lumen-boot" content="([^"]*)">', page)
    assert m, "no boot data in the page"
    return json.loads(html_mod.unescape(m.group(1)))


def test_without_settings_there_is_no_brand_and_no_logo(client):
    brand = client.get("/api/config").json()["brand"]
    assert brand == {"name": "", "logo": None, "color": None}
    assert boot_of(client.get("/").text)["brand"] == brand
    assert client.get("/brand/logo").status_code == 404


def test_a_configured_logo_is_served_as_a_picture_that_can_run_nothing(tmp_path):
    logo = tmp_path / "company.svg"
    logo.write_text(SVG)
    with TestClient(app_with(brand_name="  Sample Aerospace ", brand_logo=str(logo), brand_color="#7A1F2B")) as c:
        brand = c.get("/api/config").json()["brand"]
        assert brand["name"] == "Sample Aerospace" and brand["color"] == "#7a1f2b"
        # an address relative to the page (it has to work under any content path), changing with the file
        assert re.fullmatch(r"brand/logo\?v=\d+", brand["logo"])
        assert boot_of(c.get("/").text)["brand"] == brand
        r = c.get("/" + brand["logo"])
        assert r.status_code == 200 and r.content == SVG.encode() and r.headers["content-type"].startswith("image/svg+xml")
        # shown inside <img> an SVG runs no script; opened by itself it is a document, and gets nothing either
        csp = r.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "sandbox" in csp and "script-src" not in csp
        assert r.headers["x-content-type-options"] == "nosniff" and "max-age" in r.headers["cache-control"]
    png = tmp_path / "mark.PNG"
    png.write_bytes(PNG)
    with TestClient(app_with(brand_logo=str(png))) as c:
        r = c.get("/brand/logo")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == PNG
        assert c.get("/api/config").json()["brand"]["name"] == ""


def test_the_logo_is_reached_inside_a_content_path(tmp_path):
    logo = tmp_path / "logo.svg"
    logo.write_text(SVG)
    proxy = PrefixProxy(app_with(brand_logo=str(logo)), PREFIX)
    with TestClient(proxy) as c:
        brand = c.get(PREFIX + "/api/config").json()["brand"]
        assert not brand["logo"].startswith("/") and "://" not in brand["logo"]
        assert c.get(f"{PREFIX}/{brand['logo']}").status_code == 200
        assert c.get("/brand/logo").status_code == 404, "outside the content path there is nothing"


@pytest.mark.parametrize("case", ["missing", "folder", "page", "gif", "huge"])
def test_only_a_small_image_file_is_taken_for_a_logo(tmp_path, case):
    if case == "missing":
        path = tmp_path / "nothing.svg"
    elif case == "folder":
        path = tmp_path / "logo.svg"
        path.mkdir()
    elif case == "page":
        path = tmp_path / "logo.html"          # a document, whatever it contains
        path.write_text(SVG)
    elif case == "gif":
        path = tmp_path / "logo.gif"           # not one of the kinds that are served
        path.write_bytes(b"GIF89a")
    else:
        path = tmp_path / "logo.png"
        path.write_bytes(PNG + b"\0" * BRAND_LOGO_MAX_BYTES)
    with TestClient(app_with(brand_name="Sample Aerospace", brand_logo=str(path))) as c:
        brand = c.get("/api/config").json()["brand"]
        assert brand["logo"] is None and brand["name"] == "Sample Aerospace", "the rest of the brand still applies"
        assert c.get("/brand/logo").status_code == 404


@pytest.mark.parametrize("given,taken", [
    ("#0A1B2C", "#0a1b2c"), (" #ffd200 ", "#ffd200"),
    ("", None), ("blue", None), ("#fff", None), ("#12345g", None), ("0a1b2c", None), ("#0a1b2c80", None),
    ("#0a1b2c; background: url(https://example.invalid/x)", None), ("rgb(0, 0, 0)", None),
])
def test_the_colour_is_a_plain_hex_value_or_nothing(given, taken):
    assert make_settings(brand_color=given).brand["color"] == taken


def test_the_name_is_text_and_never_markup():
    name = 'Sample "Aero" <script>alert(1)</script> & Sons'
    with TestClient(app_with(brand_name=name)) as c:
        page = c.get("/").text
        assert "<script>alert(1)</script>" not in page and 'content="{"' not in page
        assert boot_of(page)["brand"]["name"] == name, "it arrives as the text it is"
    assert len(make_settings(brand_name="x" * 500).brand["name"]) == 60


def test_a_logo_in_the_brand_folder_is_found_and_a_setting_wins(tmp_path, monkeypatch):
    folder = tmp_path / "brand"
    folder.mkdir()
    assert bundled_brand_logo(folder) == "" and bundled_brand_logo(tmp_path / "none") == ""
    (folder / "README.md").write_text("not a logo")
    (folder / "something.svg").write_text(SVG)
    assert bundled_brand_logo(folder) == "", "only a file called logo.* is taken"
    (folder / "logo.png").write_bytes(PNG)
    assert bundled_brand_logo(folder) == str(folder / "logo.png")
    (folder / "logo.svg").write_text(SVG)
    assert bundled_brand_logo(folder) == str(folder / "logo.svg"), "a drawing before a bitmap"
    # the folder beside the application is what an installation without the setting gets
    monkeypatch.setattr(config_mod, "BRAND_DIR", folder)
    for var in ("LUMEN_BRAND_LOGO", "LUMEN_BRAND_NAME", "LUMEN_BRAND_COLOR"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings()
    assert s.brand_logo == str(folder / "logo.svg") and s.brand["logo"] and s.brand["name"] == "" and s.brand["color"] is None
    other = tmp_path / "elsewhere.png"
    other.write_bytes(PNG)
    monkeypatch.setenv("LUMEN_BRAND_LOGO", str(other))
    monkeypatch.setenv("LUMEN_BRAND_NAME", "Sample Aerospace")
    monkeypatch.setenv("LUMEN_BRAND_COLOR", "#7A1F2B")
    s = load_settings()
    assert s.brand_logo == str(other) and s.brand["name"] == "Sample Aerospace" and s.brand["color"] == "#7a1f2b"


def test_the_repository_carries_no_artwork_of_an_organisation():
    """A logo is supplied by the installation.  What is committed under brand/ is the note that says how."""
    note = ROOT / "brand" / "README.md"
    assert note.is_file() and "logo.svg" in note.read_text()
    if not (ROOT / ".git").exists() or not shutil.which("git"):
        pytest.skip("not a git checkout")
    tracked = subprocess.run(["git", "ls-files", "brand"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert tracked == ["brand/README.md"], tracked
    ignored = subprocess.run(["git", "check-ignore", "-q", "brand/logo.svg"], cwd=ROOT)
    assert ignored.returncode == 0, "a logo put into brand/ must not be committed by accident"
    # the application's own pictures: its mark, and nothing else
    assert sorted(p.name for p in (ROOT / "lumen" / "static" / "img").iterdir()) == ["icon.svg"]
