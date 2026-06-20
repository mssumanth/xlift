"""Fake-backend CPU end-to-end test.

Injects a controllable fake `models.backend.generate` (a deterministic "policy"
with a per-task target pass rate), then runs the REAL async metric pipeline on a
tiny synthetic cohort — no GPU, no API, no network. This exercises the actual
rollout/extraction/aggregation code paths, just with the model swapped out.

Run either way:
    pytest tests/test_flat_e2e.py
    python3 tests/test_flat_e2e.py
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import models.backend as backend
from metrics.boundary_score import compute_cohort_boundary_score
from metrics.reward_length_corr import compute_cohort_length_correlation


# A tiny cohort with KNOWN target pass rates per task.
COHORT = [
    {"id": 0, "question": "TASK_FRONTIER what is x?", "answer": "10", "_p": 0.5},
    {"id": 1, "question": "TASK_EASY what is y?",     "answer": "20", "_p": 1.0},
    {"id": 2, "question": "TASK_HARD what is z?",     "answer": "30", "_p": 0.0},
]
_META = {t["question"]: (t["answer"], t["_p"]) for t in COHORT}


def _install_fake_policy(k: int):
    """Replace backend.generate with a deterministic policy hitting each task's _p."""
    counters: dict[str, int] = {}

    async def fake_generate(prompt, max_tokens=1024, temperature=0.8, system=None):
        # find which task this prompt is for
        for q, (ans, p) in _META.items():
            if q in prompt:
                counters[q] = counters.get(q, 0) + 1
                n_correct_target = round(p * k)
                if counters[q] <= n_correct_target:
                    return f"step by step reasoning here\n#### {ans}"
                return "step by step reasoning here\n#### 99999"  # wrong
        return "#### 0"

    backend.generate = fake_generate


def test_boundary_pipeline_runs_on_cpu_with_fake():
    k = 4
    _install_fake_policy(k)
    res = asyncio.run(compute_cohort_boundary_score(COHORT, n_rollouts=k, max_tasks=3))

    # Aggregates present
    for key in ("mean_boundary_score", "mean_pass_rate", "mean_reachability", "learnable_fraction"):
        assert key in res, f"missing {key}"

    # Pass rates match the injected policy: 0.5, 1.0, 0.0 -> mean 0.5
    assert abs(res["mean_pass_rate"] - 0.5) < 1e-6, res["mean_pass_rate"]
    # BoundaryScore: (1.0 + 0 + 0)/3
    assert abs(res["mean_boundary_score"] - (1.0 / 3)) < 1e-6, res["mean_boundary_score"]
    # Reachability: task0 (1-0.5)=0.5, others 0 -> mean 0.5/3
    assert abs(res["mean_reachability"] - (0.5 / 3)) < 1e-6, res["mean_reachability"]
    # Only the frontier task is in the learnable band [0.30, 0.70]
    assert abs(res["learnable_fraction"] - (1.0 / 3)) < 1e-6, res["learnable_fraction"]
    print("  ok  boundary pipeline (pass=0.5, boundary=0.333, reach=0.167, learnable=1/3)")


def test_reward_length_pipeline_runs_on_cpu_with_fake():
    k = 4
    _install_fake_policy(k)
    res = asyncio.run(compute_cohort_length_correlation(COHORT, n_rollouts=k, max_tasks=3))
    for key in ("global_correlation", "length_independence", "misleading_lift_risk"):
        assert key in res, f"missing {key}"
    assert -1.0 <= res["global_correlation"] <= 1.0
    assert 0.0 <= res["length_independence"] <= 1.0
    print("  ok  reward-length pipeline runs end-to-end on CPU")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} fake-backend e2e tests passed")


if __name__ == "__main__":
    _run_all()
