"""The benchmark runner (prompt 13): held-out cases, fault cases, coverage trials and the report."""

import numpy as np
import pytest

from lumen.validation.benchmark import (
    CURVED_TARGET_ID, DRAFT_ID, HELD_OUT_SEED, TAG_POOL, RenderCache, benchmark_library, coverage_trials, judge, negative_cases, nominal_cases, run_case,
    summarise, to_markdown,
)
from tests.conftest import CACHE

SIZE = (1600, 1200)
RECORDS: list[dict] = []  # collected by the tests below, then turned into a report


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return benchmark_library(tmp_path_factory.mktemp("benchlib"))


@pytest.fixture(scope="module")
def shoot():
    return RenderCache(CACHE / "benchmark")


@pytest.fixture(scope="module")
def faults():
    return {c.name: c for c in negative_cases(size=SIZE)}


def test_cases_are_reproducible_and_differ_from_the_test_fixtures(faults):
    a, b = nominal_cases(4, size=SIZE), nominal_cases(4, size=SIZE)
    assert [(c.name, c.note, c.cams, c.scene) for c in a] == [(c.name, c.note, c.cams, c.scene) for c in b]
    # the first cases do not change when more are asked for (a longer run extends a shorter one)
    assert [(c.cams, c.scene) for c in nominal_cases(2, size=SIZE)] == [(c.cams, c.scene) for c in a[:2]]
    ids = {t.tag_id for c in a for t in c.scene.stickers}
    assert ids <= set(TAG_POOL) and not ids & {3, 7, 11, 20}, "the unit tests' tag IDs are not used"
    assert HELD_OUT_SEED == 9131
    # fault cases: one per fault the validation prompt names, new ones only ever appended
    assert list(faults)[:13] == ["no-scale", "moved-tag", "one-size-wrong", "all-sizes-wrong", "undeclared-tags", "twin-seams", "floor-not-in-view", "leaning-edge", "raised-tag",
                                 "faint-seam", "proud-trim", "duplicate-id", "flash-glare"]
    assert list(faults)[13:] == ["all-sizes-wrong-ruler", "true-sizes-ruler", "curved-target", "draft-record"]
    assert {c.expect for c in faults.values()} <= {"accurate", "not_guidance", "accurate_or_refused", "undetectable_without_reference", "not_at_true_floor"}
    assert all(c.expect == "accurate" for c in a)


def test_judging_is_about_what_a_correct_system_does():
    ok = dict(status="prototype_guidance", expect="accurate", corner_error_mm=0.3, expanded_mm=0.9)
    assert judge(ok) and judge({**ok, "status": "research_preview"}), "a sound capture may be a preview (uncertainty against tolerance); it must be right"
    assert not judge({**ok, "corner_error_mm": 5.0}) and not judge({**ok, "status": "blocked"}) and not judge({**ok, "status": "failure"})
    assert judge(dict(status="blocked", expect="not_guidance")) and not judge(dict(status="prototype_guidance", expect="not_guidance", corner_error_mm=0.1, expanded_mm=1.0))
    assert judge(dict(status="refused", expect="accurate_or_refused")) and judge(dict(status="research_preview", expect="accurate_or_refused", corner_error_mm=0.4, expanded_mm=1.0))
    assert not judge(dict(status="prototype_guidance", expect="accurate_or_refused", corner_error_mm=9.0, expanded_mm=1.0)), "a confident wrong answer is the one outcome that is never right"


def test_a_held_out_layout_is_measured_within_its_stated_uncertainty_in_both_modes(lib, shoot):
    case = nominal_cases(1, size=SIZE)[0]
    out = {}
    for mode in ("single_manual", "multi_auto"):
        r = run_case(case, mode, lib, shoot)
        RECORDS.append(r)
        out[mode] = r
        assert r["status"] in ("prototype_guidance", "research_preview"), r
        assert r["corner_error_mm"] < 0.6 and r["covered"] and not r["false_acceptance"], r
        assert r["rotation_error_deg"] < 0.15 and r["centre_error_mm"] <= r["corner_error_mm"] + 1e-9
        assert len(r["budget_top"]) == 3 and r["budget_top"][0]["mm"] >= r["budget_top"][1]["mm"]
        assert set(r["seconds"]) == {"detect", "register", "datums", "overlay"}
    assert out["single_manual"]["n_photos"] == 1 and out["multi_auto"]["n_photos"] == len(case.cams) > 1
    # more photos must not make the statement about the same target worse by much
    assert out["multi_auto"]["expanded_mm"] <= 1.25 * out["single_manual"]["expanded_mm"], (out["multi_auto"]["expanded_mm"], out["single_manual"]["expanded_mm"])


def test_a_common_size_error_passes_every_internal_check_and_is_stopped_by_one_rule_measurement(lib, shoot, faults):
    blind = run_case(faults["all-sizes-wrong"], "multi_auto", lib, shoot)
    RECORDS.append(blind)
    # recorded, not hidden: a wrong outline is drawn, far outside its stated uncertainty, and no check about size fails
    assert blind["status"] in ("prototype_guidance", "research_preview") and blind["corner_error_mm"] > 8.0
    assert blind["corner_error_mm"] > 5.0 * blind["expanded_mm"] and blind["covered"] is False
    assert not {"independent_length_check", "tags_mutually_consistent", "scale_available"} & set(blind["failed_gates"])
    assert blind["expect"] == "undetectable_without_reference" and blind["false_acceptance"] == (blind["status"] == "prototype_guidance")
    ruler = run_case(faults["all-sizes-wrong-ruler"], "multi_auto", lib, shoot)
    RECORDS.append(ruler)
    assert ruler["status"] == "blocked" and "independent_length_check" in ruler["failed_gates"] and ruler.get("corner_error_mm") is None
    assert ruler["reference"]["consistent"] is False and ruler["reference"]["relative_difference"] == pytest.approx(0.04, abs=0.005)
    assert ruler["outcome_ok"] and not ruler["false_acceptance"]
    # control: the same check with correct sizes changes nothing
    good = run_case(faults["true-sizes-ruler"], "multi_auto", lib, shoot)
    RECORDS.append(good)
    assert good["reference"]["consistent"] is True and abs(good["reference"]["relative_difference"]) < 0.004
    assert good["status"] in ("prototype_guidance", "research_preview") and good["corner_error_mm"] < 0.6 and good["covered"] and good["outcome_ok"]
    assert good["n_tags"] == 4 and not good["excluded_tags"], "no tag is thrown out of a sound capture"


def test_records_outside_the_method_or_not_released_do_not_give_guidance(lib, shoot, faults):
    assert lib.get(CURVED_TARGET_ID, "A").surface.target == "curved" and lib.get(DRAFT_ID, "A").status == "draft" and lib.get(DRAFT_ID, "A").is_complete()
    curved = run_case(faults["curved-target"], "multi_auto", lib, shoot)
    RECORDS.append(curved)
    assert curved["status"] == "blocked" and "surface_supported" in curved["failed_gates"] and curved["outcome_ok"]
    draft = run_case(faults["draft-record"], "multi_auto", lib, shoot)
    RECORDS.append(draft)
    assert draft["status"] == "research_preview" and "operation_record_released" in draft["failed_gates"] and draft["corner_error_mm"] < 0.6 and draft["outcome_ok"]
    none = run_case(faults["no-scale"], "multi_auto", lib, shoot)
    RECORDS.append(none)
    assert none["status"] in ("refused", "blocked") and none["outcome_ok"] and none.get("corner_error_mm") is None


def test_seeded_errors_inside_the_budget_are_covered(lib, shoot):
    case = nominal_cases(1, size=SIZE)[0]
    trials = coverage_trials(case, lib, shoot, n=4, seed=7, mode="single_manual")
    assert [t["trial"] for t in trials] == [1, 2, 3, 4] and len({t["case"] for t in trials}) == 4
    drawn = [t for t in trials if t.get("corner_error_mm") is not None]
    assert len(drawn) == 4 and all(t["covered"] for t in drawn), [(t["corner_error_mm"], t["expanded_mm"]) for t in drawn]
    # the seeded errors are really applied: results differ between trials and from the unperturbed case
    errs = [round(t["corner_error_mm"], 4) for t in drawn]
    assert len(set(errs)) == 4
    assert coverage_trials(case, lib, shoot, n=1, seed=7, mode="single_manual")[0]["corner_error_mm"] == pytest.approx(drawn[0]["corner_error_mm"], abs=1e-6), "same seed, same trial"


def test_report_states_what_the_numbers_are(lib, shoot):
    assert len(RECORDS) >= 7, "run the whole module: the report is built from the records of the tests above"
    summary = summarise(RECORDS)
    assert summary["single_manual"]["cases"] == 1 and summary["multi_auto"]["cases"] == 1
    assert summary["multi_auto"]["coverage"] == {"covered": 1, "of": 1} and summary["multi_auto"]["failures"] == 0
    neg = summary["negative"]
    assert neg["cases"] == 6 and neg["handled_correctly"] == 6 and neg["failures"] == 0
    assert neg["false_acceptance"] == sum(r["false_acceptance"] for r in RECORDS if r["kind"] != "nominal") <= 1
    cov = coverage_trials(nominal_cases(1, size=SIZE)[0], lib, shoot, n=2, seed=7, mode="single_manual")
    md = to_markdown(RECORDS, summary, cov, {"when": "test", "algorithm": "x", "seed": HELD_OUT_SEED, "size": "1600x1200", "n_nominal": 1, "n_negative": 6, "machine": "test", "note": "A note."})
    assert md.startswith("# Synthetic benchmark results") and "Simulated data only" in md and "not physical accuracy" in md and "A note." in md
    for needle in ("## Nominal cases", "## Fault cases", "## Coverage with seeded errors", "all-sizes-wrong-ruler", "recorded only", "largest term of the budget", "false acceptance", "Handled as expected: 6 of 6"):
        assert needle in md, needle
    assert "None" not in md and "nan" not in md.lower()
    assert np.isfinite(summary["multi_auto"]["corner_error_mm"]["median"])
