"""The operator's smoke-test script, run against a local server behind a sub-path proxy."""

import importlib.util
from pathlib import Path

from tests.browser.conftest import LiveServer, make_app
from tests.conftest import PrefixProxy

ROOT = Path(__file__).resolve().parents[2]


def load_script():
    spec = importlib.util.spec_from_file_location("lumen_smoke_test", ROOT / "scripts" / "smoke_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_script_passes_against_a_sub_path_and_fails_against_nothing(capsys):
    smoke = load_script()
    srv = LiveServer(PrefixProxy(make_app(), "/content/3f2a9c1e", announce=True)).start()
    try:
        rc = smoke.main([srv.url + "content/3f2a9c1e", "--timeout", "240"])  # no trailing slash on purpose
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "ALL STEPS PASSED" in out and out.count("PASS  ") == 11 and "FAIL" not in out.replace("FAILED", "")
        assert "dx +1." in out and "session and photos deleted" in out
        # outside the content path there is no application: the script reports failure, it does not pretend
        rc2 = smoke.main([srv.url + "content/other/", "--timeout", "20"])
        out2 = capsys.readouterr().out
        assert rc2 == 1 and "FAIL  health" in out2 and "FAILED:" in out2
    finally:
        srv.stop()
