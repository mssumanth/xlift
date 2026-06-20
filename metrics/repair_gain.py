"""
RepairGain — measures whether model failures are recoverable.

RepairGain(T) = score_after_feedback - score_before_feedback

High RepairGain: the model failed but recovered when given a hint.
  → The task has learnable signal. RL can reinforce the correct behavior.

Low RepairGain: the model failed and stayed failed even with help.
  → The task is beyond the model's current capability. Training on it won't help.
"""

import os
import asyncio
import anthropic
from data.load_gsm8k import extract_answer, answers_match
from prompts.metrics import SOLVE_PROMPT, FEEDBACK_PROMPT, GENERATE_HINT_PROMPT
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

def _get_client():
    return anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


async def _call(prompt: str) -> str:
    resp = await _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


async def _generate_hint(question: str, correct: str, wrong: str) -> str:
    """Ask Claude to generate a targeted hint without giving the answer away."""
    hint = await _call(GENERATE_HINT_PROMPT.format(
        question=question,
        correct_answer=correct,
        wrong_answer=wrong,
    ))
    return hint.strip()


async def measure_task_repair_gain(task: dict, n_rollouts: int = 4) -> dict:
    """
    For a single task:
    1. Run model N times without feedback → baseline score
    2. For each failure, generate a hint and retry → repair score
    3. RepairGain = repair_score - baseline_score
    """
    question = task["question"]
    correct = task["answer"]

    # Step 1 — baseline rollouts
    baseline_responses = await asyncio.gather(*[
        _call(SOLVE_PROMPT.format(question=question)) for _ in range(n_rollouts)
    ])

    baseline_results = []
    for text in baseline_responses:
        pred = extract_answer(text)
        is_correct = bool(pred and correct and answers_match(pred, correct))
        baseline_results.append({"text": text, "pred": pred, "correct": is_correct})

    baseline_score = sum(r["correct"] for r in baseline_results) / n_rollouts

    # Step 2 — repair failed attempts
    failures = [r for r in baseline_results if not r["correct"]]
    if not failures:
        # Already getting everything right — RepairGain is not meaningful here
        return {
            "task_id": task["id"],
            "baseline_score": baseline_score,
            "repair_score": baseline_score,
            "repair_gain": 0.0,
            "n_failures": 0,
            "note": "no_failures_to_repair",
        }

    # Generate hints for all failures in parallel
    hints = await asyncio.gather(*[
        _generate_hint(question, correct, f["pred"] or "no answer")
        for f in failures
    ])

    # Retry with feedback
    repair_responses = await asyncio.gather(*[
        _call(FEEDBACK_PROMPT.format(
            question=question,
            wrong_answer=f["pred"] or "no answer",
            correct_answer=correct,
            hint=hint,
        ))
        for f, hint in zip(failures, hints)
    ])

    n_repaired = 0
    for text in repair_responses:
        pred = extract_answer(text)
        if pred and correct and answers_match(pred, correct):
            n_repaired += 1

    repair_rate_on_failures = n_repaired / len(failures)

    # Overall repair score accounts for tasks that were already correct
    n_already_correct = sum(r["correct"] for r in baseline_results)
    repair_score = (n_already_correct + n_repaired) / n_rollouts
    repair_gain = repair_score - baseline_score

    return {
        "task_id": task["id"],
        "baseline_score": round(baseline_score, 3),
        "repair_score": round(repair_score, 3),
        "repair_gain": round(repair_gain, 3),
        "repair_rate_on_failures": round(repair_rate_on_failures, 3),
        "n_failures": len(failures),
        "n_repaired": n_repaired,
        "verdict": (
            "highly_repairable" if repair_gain > 0.4 else
            "repairable"        if repair_gain > 0.15 else
            "weakly_repairable" if repair_gain > 0.0 else
            "not_repairable"
        ),
    }


async def compute_cohort_repair_gain(
    cohort: list[dict],
    n_rollouts: int = 4,
    max_tasks: int = 40,
) -> dict:
    """Compute RepairGain across a cohort."""
    # Focus on tasks in the learnable zone — those with non-trivial pass rates
    sample = [
        t for t in cohort
        if t.get("pass_rate", 0.5) < 0.85  # skip already-mastered tasks
    ][:max_tasks]

    print(f"  Computing RepairGain on {len(sample)} tasks...")

    batch_size = 8
    task_results = []
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i + batch_size]
        results = await asyncio.gather(*[
            measure_task_repair_gain(t, n_rollouts) for t in batch
        ])
        task_results.extend(results)

    gains = [r["repair_gain"] for r in task_results]
    repairable = [r for r in task_results if r["repair_gain"] > 0.15]

    return {
        "mean_repair_gain": sum(gains) / len(gains) if gains else 0,
        "repairable_fraction": len(repairable) / len(task_results) if task_results else 0,
        "task_results": task_results,
        "verdict_distribution": {
            "highly_repairable": sum(1 for r in task_results if r["verdict"] == "highly_repairable"),
            "repairable":        sum(1 for r in task_results if r["verdict"] == "repairable"),
            "weakly_repairable": sum(1 for r in task_results if r["verdict"] == "weakly_repairable"),
            "not_repairable":    sum(1 for r in task_results if r["verdict"] == "not_repairable"),
        },
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    cohort_dir = Path(__file__).parent.parent / "results" / "cohorts"
    for name in ["easy", "frontier", "hard"]:
        path = cohort_dir / f"{name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            cohort = json.load(f)
        print(f"\n=== {name.upper()} cohort ===")
        result = asyncio.run(compute_cohort_repair_gain(cohort, n_rollouts=4, max_tasks=20))
        print(f"  Mean RepairGain:      {result['mean_repair_gain']:.3f}")
        print(f"  Repairable fraction:  {result['repairable_fraction']:.1%}")
        print(f"  Verdict distribution: {result['verdict_distribution']}")
