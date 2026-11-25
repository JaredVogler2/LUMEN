"""Scanning a code to identify the operation (prompt 08)."""

import pytest

from tests.conftest import jpeg_bytes
from tests.unit.test_barcode import photo_of, symbol

pytest.importorskip("zxingcpp")


def scan(client, img):
    r = client.post("/api/scan", content=jpeg_bytes(img, 90), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200, r.text
    return r.json()


def test_scanned_placard_code_resolves_to_exactly_one_record(client):
    out = scan(client, photo_of(symbol("LUMEN-DEMO-PLC-102", "QRCode", 8)))
    assert out["geometry_used_for_scale"] is False and out["decoders"]["zxing"]
    code = out["codes"][0]
    assert code["text"] == "LUMEN-DEMO-PLC-102" and code["format"] == "QR Code"
    res = code["resolved"]
    assert res["status"] == "exact"
    m = res["matches"][0]
    assert m["operation_id"] == "DEMO-FLAT-01" and m["placard_id"] == "PLC-DEMO-102" and m["matched"] == "placard_barcode"


def test_unknown_and_near_codes_are_not_selected(client):
    out = scan(client, photo_of(symbol("lumen-demo-plc-101", "Code128", 3), size=(1000, 1400)))  # wrong case
    res = out["codes"][0]["resolved"]
    assert res["status"] == "none" and res["matches"] == []
    assert any(n["value"] == "LUMEN-DEMO-PLC-101" for n in res["near"]), "a near match is offered, never chosen"
    out = scan(client, photo_of(symbol("SOMETHING-ELSE-42", "DataMatrix", 6)))
    assert out["codes"][0]["resolved"]["status"] == "none" and out["codes"][0]["resolved"]["near"] == []


def test_picture_without_a_code_and_bad_uploads(client):
    import numpy as np

    out = scan(client, np.full((400, 600, 3), 128, np.uint8))
    assert out["codes"] == []
    r = client.post("/api/scan", content=b"not an image", headers={"content-type": "image/jpeg"})
    assert r.status_code in (415, 422)
