"""Pure-logic unit tests for the flat xLift pipeline — no GPU, no API, no network.

Run either way:
    pytest tests/test_flat_logic.py
    python3 tests/test_flat_logic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.load_gsm8k import (
    extract_answer, normalize_answer, extract_boxed_answer, answers_match, _to_float,
)
from metrics.boundary_score import boundary_score
from metrics.xlift_score import compute_xlift
from training.grpo_train import bootstrap_lift_ci


# ----------------------------- BoundaryScore ----------------------------- #
def test_boundary_score_peaks_at_half():
    assert boundary_score(0.5) == 1.0
    assert boundary_score(0.0) == 0.0
    assert boundary_score(1.0) == 0.0
    # symmetric around 0.5
    assert abs(boundary_score(0.3) - boundary_score(0.7)) < 1e-9
    # monotone toward the peak
    assert boundary_score(0.5) > boundary_score(0.3) > boundary_score(0.1)


# ----------------------------- Reachability ------------------------------ #
def _reachability(n_correct, k):
    p = n_correct / k
    return (1.0 if n_correct > 0 else 0.0) - p

def test_reachability_separates_reachable_from_hopeless():
    assert _reachability(0, 5) == 0.0          # hopeless: never solved
    assert _reachability(5, 5) == 0.0          # mastered: always solved
    assert abs(_reachability(1, 5) - 0.8) < 1e-9   # rare but reachable -> high
    # a rarely-solved task scores higher than a frequently-solved one
    assert _reachability(1, 5) > _reachability(3, 5)


# --------------------------- Answer extraction --------------------------- #
def test_extract_answer_prefers_hash_then_boxed_then_number():
    assert extract_answer("blah\n#### 42") == "42"
    assert extract_answer("the result is \\boxed{18}") == "18"
    assert extract_answer("no markers, last number 7 here") == "7"
    assert extract_answer("") is None

def test_extract_boxed_handles_nested_braces():
    assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed_answer("no box here") is None


# ----------------------- LaTeX-tolerant matching ------------------------- #
def test_answers_match_latex_and_numeric():
    assert answers_match("18", "18^\\circ")              # degrees stripped
    assert answers_match("1000000", "1,\\!000,\\!000")   # thin-space + commas
    assert answers_match("\\frac{266664}{5}", "\\frac{266664}{5}")
    assert not answers_match("53333.33", "\\frac{266664}{5}")  # genuinely different
    assert not answers_match("", "5")
    assert not answers_match("5", None)

def test_to_float_handles_fractions():
    assert abs(_to_float("1/2") - 0.5) < 1e-9
    assert abs(_to_float(normalize_answer("\\frac{3}{4}")) - 0.75) < 1e-9
    assert _to_float("abc") is None


# ------------------------------ xLift score ------------------------------ #
def _metric_stubs(boundary=0.8, repair=0.3, gepa=0.2, trust=0.9, length_indep=1.0):
    return (
        {"mean_boundary_score": boundary},
        {"mean_repair_gain": repair},
        {"gepa_transfer_lift": gepa, "gepa_gap": 0.05},
        {"reward_trust_score": trust},
        {"length_independence": length_indep, "global_correlation": 0.0,
         "misleading_lift_risk": False},
    )

def test_compute_xlift_in_range_and_has_recommendation():
    b, r, g, a, l = _metric_stubs()
    out = compute_xlift(b, r, g, a, l)
    assert 0.0 <= out["xlift_score"] <= 100.0
    assert isinstance(out["recommendation"], str) and out["recommendation"]

def test_compute_xlift_flags_untrustworthy_verifier():
    # low trust should not produce a confident BUY
    b, r, g, a, l = _metric_stubs(trust=0.1, length_indep=0.1)
    out = compute_xlift(b, r, g, a, l)
    assert out["recommendation"].lower() != "buy"

def test_compute_xlift_optional_length_arg():
    b, r, g, a, _ = _metric_stubs()
    out = compute_xlift(b, r, g, a)  # length_corr_result omitted
    assert 0.0 <= out["xlift_score"] <= 100.0


# --------------------------- Bootstrap lift CI --------------------------- #
def test_bootstrap_ci_detects_real_lift():
    baseline = [0] * 60 + [1] * 40   # 40%
    post     = [0] * 40 + [1] * 60   # 60%
    mean, lo, hi = bootstrap_lift_ci(baseline, post, n_boot=500, seed=0)
    assert mean > 0.1
    assert lo > 0          # CI excludes zero -> significant

def test_bootstrap_ci_calls_noise_insignificant():
    flags = [0, 1] * 50
    mean, lo, hi = bootstrap_lift_ci(flags, flags, n_boot=500, seed=0)
    assert lo <= 0 <= hi   # no change -> CI straddles zero


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} pure-logic tests passed")


if __name__ == "__main__":
    _run_all()
