"""
BoundaryScore — measures whether tasks sit in the learnable zone.

Formula: BoundaryScore(T) = 4 * p * (1 - p)
  where p = pass rate of the model on task T across N rollouts

Peaks at 1.0 when p = 0.5 (model is right half the time — perfect training task)
Falls to 0.0 at p = 0 (always wrong) or p = 1 (always right — already mastered)
"""

import os
import re
import asyncio
import anthropic
from tqdm import tqdm
from data.load_gsm8k import extract_answer, answers_match
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

def _get_client():
    return anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def boundary_score(p: float) -> float:
    """The core formula. p is pass rate 0-1."""
    return 4 * p * (1 - p)


async def _solve_once(question: str) -> str:
    """Run the BASE model (the one we'll RL-train) on the task once."""
    from prompts.metrics import SOLVE_PROMPT
    from models.backend import generate
    return await generate(SOLVE_PROMPT.format(question=question), max_tokens=1024, temperature=0.8)


async def measure_task_pass_rate(task: dict, n_rollouts: int = 5) -> dict:
    """Run model N times on a task and compute pass rate + boundary score."""
    results = await asyncio.gather(*[_solve_once(task["question"]) for _ in range(n_rollouts)])

    correct = 0
    predicted_answers = []
    for text in results:
        pred = extract_answer(text)
        predicted_answers.append(pred)
        if pred and task["answer"] and answers_match(pred, task["answer"]):
            correct += 1

    p = correct / n_rollouts
    # Reachability = pass@k - pass@1. pass@1 is the per-sample success prob (= p);
    # pass@k is 1 if ANY of the k rollouts passed. High reachability means the model
    # CAN reach the answer but rarely — exactly the tasks RL sharpens. It separates
    # "rare but reachable" (good to train) from "hopeless" (answer outside support).
    pass_at_k = 1.0 if correct > 0 else 0.0
    reachability = pass_at_k - p
    return {
        "task_id": task["id"],
        "pass_rate": p,
        "boundary_score": boundary_score(p),
        "pass_at_k": pass_at_k,
        "reachability": reachability,
        "n_rollouts": n_rollouts,
        "n_correct": correct,
        "predicted_answers": predicted_answers,
        # BoundaryScore interpretation
        "in_learnable_zone": 0.30 <= p <= 0.70,
        "verdict": (
            "mastered"   if p > 0.80 else
            "learnable"  if p >= 0.30 else
            "too_hard"
        ),
    }


async def compute_cohort_boundary_score(
    cohort: list[dict],
    n_rollouts: int = 5,
    max_tasks: int = 50,
) -> dict:
    """
    Compute BoundaryScore for a cohort of tasks.
    Returns dataset-level aggregates.
    """
    sample = cohort[:max_tasks]

    print(f"  Computing BoundaryScore on {len(sample)} tasks ({n_rollouts} rollouts each)...")
    task_results = []
    # Process in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            measure_task_pass_rate(t, n_rollouts) for t in batch
        ])
        task_results.extend(batch_results)

    if not task_results:  # empty cohort/sample -> avoid ZeroDivisionError below
        return {"mean_boundary_score": 0.0, "mean_pass_rate": 0.0, "mean_reachability": 0.0,
                "learnable_fraction": 0.0, "task_results": [],
                "n_mastered": 0, "n_learnable": 0, "n_too_hard": 0}

    scores = [r["boundary_score"] for r in task_results]
    pass_rates = [r["pass_rate"] for r in task_results]
    reachabilities = [r["reachability"] for r in task_results]
    learnable = [r for r in task_results if r["in_learnable_zone"]]

    return {
        "mean_boundary_score": sum(scores) / len(scores),
        "mean_pass_rate": sum(pass_rates) / len(pass_rates),
        "mean_reachability": sum(reachabilities) / len(reachabilities),
        "learnable_fraction": len(learnable) / len(task_results),
        "task_results": task_results,
        # Verdict distribution
        "n_mastered":  sum(1 for r in task_results if r["verdict"] == "mastered"),
        "n_learnable": sum(1 for r in task_results if r["verdict"] == "learnable"),
        "n_too_hard":  sum(1 for r in task_results if r["verdict"] == "too_hard"),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    cohort_dir = Path(__file__).parent.parent / "results" / "cohorts"
    for name in ["easy", "frontier", "hard"]:
        path = cohort_dir / f"{name}.json"
        if not path.exists():
            print(f"Cohort {name} not found. Run data/load_gsm8k.py first.")
            continue

        with open(path) as f:
            cohort = json.load(f)

        print(f"\n=== {name.upper()} cohort ===")
        result = asyncio.run(compute_cohort_boundary_score(cohort, n_rollouts=5, max_tasks=30))
        print(f"  Mean BoundaryScore: {result['mean_boundary_score']:.3f}")
        print(f"  Mean pass rate:     {result['mean_pass_rate']:.3f}")
        print(f"  Learnable fraction: {result['learnable_fraction']:.1%}")
        print(f"  Mastered / Learnable / Too hard: "
              f"{result['n_mastered']} / {result['n_learnable']} / {result['n_too_hard']}")
