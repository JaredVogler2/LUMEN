#!/usr/bin/env python3
"""Run the synthetic benchmark and write its report.

    python scripts/run_benchmark.py                      # full run (about half an hour on four cores)
    python scripts/run_benchmark.py --nominal 2 --coverage 0 --out /tmp/quick.md --json ""

Everything is computer-generated: the photos are rendered from invented
panels, and the report says so.  Rendered photos are kept in ``--cache`` so a
second run only repeats the measurements.  The exit status is 1 when a result
was shown as guidance although it was wrong, when a fault case was not handled
as it should be, or when something failed unexpectedly.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nominal", type=int, default=12, help="number of held-out nominal configurations (default 12)")
    ap.add_argument("--size", default="1600x1200", help="photo size in pixels, WIDTHxHEIGHT (default 1600x1200)")
    ap.add_argument("--coverage", type=int, default=40, help="seeded-error trials for the coverage check; 0 = skip (default 40)")
    ap.add_argument("--no-faults", action="store_true", help="skip the fault cases")
    ap.add_argument("--seed", type=int, default=None, help="seed of the nominal configurations (default: the benchmark's own, 9131)")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "lumen-benchmark-renders"), help="directory for rendered photos")
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmark" / "RESULTS.md"), help="report (Markdown)")
    ap.add_argument("--json", default=str(ROOT / "docs" / "benchmark" / "results.json"), help="every record as JSON; empty = do not write")
    ap.add_argument("--note", default="", help="a sentence printed under the report's title")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        ap.error("--size must look like 1600x1200")

    import cv2

    from lumen import ALGORITHM_VERSION, __version__
    from lumen.validation.benchmark import HELD_OUT_SEED, benchmark_library, dump, run_benchmark, to_markdown

    t0 = time.time()
    load0 = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")

    def say(msg: str) -> None:
        if not args.quiet:
            print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    with tempfile.TemporaryDirectory(prefix="lumen-bench-lib-") as tmp:
        library = benchmark_library(tmp)
        result = run_benchmark(library, args.cache, n_nominal=args.nominal, size=(w, h), with_negative=not args.no_faults, coverage_n=args.coverage, progress=say,
                               seed=HELD_OUT_SEED if args.seed is None else args.seed)
    meta = {
        "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "application": __version__,
        "algorithm": ALGORITHM_VERSION,
        "seed": HELD_OUT_SEED if args.seed is None else args.seed,
        "size": f"{w}x{h}",
        "note": args.note,
        "n_nominal": result["n_nominal"],
        "n_negative": result["n_negative"],
        "machine": f"{platform.machine()}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, OpenCV {cv2.__version__}, "
                   f"load average {load0:.1f} at start; run time {time.time() - t0:.0f} s",
    }
    result["meta"] = meta
    md = to_markdown(result["records"], result["summary"], result["coverage"], meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    say(f"report: {out}")
    if args.json:
        jp = Path(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(dump(result), encoding="utf-8")
        say(f"records: {jp}")

    recs = result["records"]
    # A fault the report itself says cannot be seen from the photos is listed, not counted as a regression.
    known = [r for r in recs if r.get("false_acceptance") and r["expect"] == "undetectable_without_reference"]
    false_acc = [r for r in recs if r.get("false_acceptance") and r["expect"] != "undetectable_without_reference"]
    for r in known:
        print(f"KNOWN LIMITATION (wrong result shown as guidance; needs the independent length check): {r['case']} corner_error_mm={r.get('corner_error_mm'):.1f}", flush=True)
    failures = [r for r in recs if r["status"] == "failure"]
    unhandled = [r for r in recs if not r["outcome_ok"]]
    for name, lst in (("FALSE ACCEPTANCE", false_acc), ("UNEXPECTED FAILURE", failures), ("NOT HANDLED AS EXPECTED", unhandled)):
        for r in lst:
            print(f"{name}: {r['case']} [{r['mode']}] status={r['status']} error={r.get('error')} corner_error_mm={r.get('corner_error_mm')} expanded_mm={r.get('expanded_mm')}", flush=True)
    s = result["summary"]
    for mode in ("single_manual", "multi_auto"):
        if mode in s:
            m = s[mode]
            ce = m.get("corner_error_mm") or {}
            print(f"{mode}: {m['cases']} cases, status {m['status']}, corner error median {ce.get('median', float('nan')):.2f} mm max {ce.get('max', float('nan')):.2f} mm, "
                  f"coverage {m['coverage']}, false acceptance {m['false_acceptance']}", flush=True)
    print(f"fault cases: {s['negative']}", flush=True)
    return 1 if (false_acc or failures or unhandled) else 0


if __name__ == "__main__":
    raise SystemExit(main())
