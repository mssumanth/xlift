"""
GEPA Transfer Lift — the differentiating signal.

Does learning from one set of tasks transfer to completely unseen tasks?

Steps:
1. Split cohort 50/50 into probe set and transfer set
2. Run GEPA loop on probe set — evolve prompt strategies, extract the best lesson
3. Apply that lesson to the transfer set (tasks the model has never seen)
4. GEPA Transfer Lift = pass rate with lesson - pass rate without lesson

High Transfer Lift → the cohort teaches reusable reasoning patterns → worth training on
Low Transfer Lift  → the cohort is redundant or overfit → diversify before training
"""

import os
import asyncio
import random
import anthropic
from metrics._throttle import acreate
from data.load_gsm8k import extract_answer, answers_match
from prompts.metrics import (
    SOLVE_PROMPT,
    SOLVE_WITH_STRATEGY_PROMPT,
    GEPA_REFLECT_PROMPT,
    GEPA_MUTATE_PROMPT,
)
from dotenv import load_dotenv

load_dotenv()

FAST_MODEL  = "claude-haiku-4-5-20251001"
THINK_MODEL = "claude-sonnet-4-6"

def _get_client():
    return anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


async def _solve(question: str, strategy: str | None = None) -> str:
    """Base model solves the task (optionally under a GEPA-evolved strategy)."""
    from models.backend import generate
    if strategy:
        prompt = SOLVE_WITH_STRATEGY_PROMPT.format(strategy=strategy, question=question)
    else:
        prompt = SOLVE_PROMPT.format(question=question)
    return await generate(prompt, max_tokens=1024, temperature=0.8)


async def _pass_rate(tasks: list[dict], strategy: str | None, n_rollouts: int = 4) -> float:
    """Run strategy on all tasks, return fraction correct."""
    all_correct = []
    batch_size = 8
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        responses = await asyncio.gather(*[
            _solve(t["question"], strategy) for t in batch for _ in range(n_rollouts)
        ])
        for j, task in enumerate(batch):
            task_responses = responses[j * n_rollouts:(j + 1) * n_rollouts]
            correct = sum(
                1 for text in task_responses
                if (pred := extract_answer(text)) and task["answer"]
                and answers_match(pred, task["answer"])
            )
            all_correct.append(correct / n_rollouts)
    return sum(all_correct) / len(all_correct) if all_correct else 0.0


async def _reflect_on_failures(tasks: list[dict], strategy: str | None) -> str:
    """Run tasks and collect failures, then reflect to extract a lesson."""
    failures = []
    for task in tasks[:15]:  # sample for reflection
        text = await _solve(task["question"], strategy)
        pred = extract_answer(text)
        if not (pred and task["answer"] and answers_match(pred, task["answer"])):
            failures.append({
                "question": task["question"][:200],
                "model_answer": pred or "none",
                "correct_answer": task["answer"],
            })

    if not failures:
        return strategy or "Think step by step."

    failure_text = "\n".join(
        f"Q: {f['question']}\nModel said: {f['model_answer']} | Correct: {f['correct_answer']}"
        for f in failures[:8]
    )

    resp = await acreate(
        model=THINK_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": GEPA_REFLECT_PROMPT.format(failures=failure_text)}],
    )
    return resp.content[0].text.strip()


async def _mutate_strategy(strategy: str, pass_rate: float, failure_pattern: str) -> list[str]:
    """Generate 3 evolved prompt strategies."""
    resp = await acreate(
        model=THINK_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": GEPA_MUTATE_PROMPT.format(
            strategy=strategy,
            pass_rate=pass_rate,
            failure_pattern=failure_pattern,
        )}],
    )
    lines = resp.content[0].text.strip().split("\n")
    strategies = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            strategies.append(line.split(".", 1)[1].strip())
        elif line and not line[0].isdigit():
            strategies.append(line)
    return strategies[:3] if strategies else [strategy]


async def run_gepa_on_probe(
    probe_tasks: list[dict],
    generations: int = 3,
    n_rollouts: int = 4,
) -> dict:
    """
    Run GEPA evolutionary loop on the probe set.
    Returns the best strategy and its probe pass rate.
    """
    current_strategy = "Think step by step and solve carefully."
    current_score = await _pass_rate(probe_tasks[:20], current_strategy, n_rollouts)

    population = [{"strategy": current_strategy, "score": current_score}]
    generation_log = [{"gen": 0, "best_score": current_score, "strategy": current_strategy}]

    for gen in range(generations):
        print(f"    GEPA generation {gen + 1}/{generations} — current best: {current_score:.2%}")

        # Reflect on failures
        failure_lesson = await _reflect_on_failures(probe_tasks[:15], current_strategy)

        # Mutate
        new_strategies = await _mutate_strategy(current_strategy, current_score, failure_lesson)

        # Evaluate new strategies in parallel
        scores = await asyncio.gather(*[
            _pass_rate(probe_tasks[:20], s, n_rollouts) for s in new_strategies
        ])

        for s, sc in zip(new_strategies, scores):
            population.append({"strategy": s, "score": sc})

        # Select best
        population = sorted(population, key=lambda x: x["score"], reverse=True)[:3]
        current_strategy = population[0]["strategy"]
        current_score = population[0]["score"]

        generation_log.append({
            "gen": gen + 1,
            "best_score": current_score,
            "strategy": current_strategy,
            "lesson": failure_lesson,
        })

    return {
        "best_strategy": current_strategy,
        "probe_pass_rate": current_score,
        "generation_log": generation_log,
    }


async def compute_gepa_transfer_lift(
    cohort: list[dict],
    generations: int = 3,
    n_rollouts: int = 4,
    max_tasks: int = 60,
) -> dict:
    """
    Full GEPA Transfer Lift computation for a cohort.

    Split → GEPA on probe → apply lesson to transfer → measure lift
    """
    sample = cohort[:max_tasks]
    random.shuffle(sample)
    mid = len(sample) // 2
    probe_tasks    = sample[:mid]
    transfer_tasks = sample[mid:]

    print(f"  Probe: {len(probe_tasks)} tasks | Transfer: {len(transfer_tasks)} tasks")

    # Baseline on transfer set (no GEPA strategy)
    print("  Measuring baseline on transfer set...")
    baseline_transfer = await _pass_rate(transfer_tasks[:20], None, n_rollouts)

    # Run GEPA on probe set
    print("  Running GEPA on probe set...")
    gepa_result = await run_gepa_on_probe(probe_tasks, generations, n_rollouts)

    best_strategy = gepa_result["best_strategy"]

    # Apply best strategy to transfer set
    print("  Applying learned strategy to transfer set...")
    transfer_with_strategy = await _pass_rate(transfer_tasks[:20], best_strategy, n_rollouts)

    gepa_transfer_lift = transfer_with_strategy - baseline_transfer
    gepa_train_lift    = gepa_result["probe_pass_rate"] - (
        gepa_result["generation_log"][0]["best_score"]
    )
    gepa_gap = gepa_train_lift - gepa_transfer_lift  # high gap = overfitting

    return {
        "baseline_transfer_pass_rate": round(baseline_transfer, 3),
        "transfer_pass_rate_with_strategy": round(transfer_with_strategy, 3),
        "gepa_transfer_lift": round(gepa_transfer_lift, 3),
        "gepa_train_lift": round(gepa_train_lift, 3),
        "gepa_gap": round(gepa_gap, 3),  # low gap = good generalisation
        "best_strategy": best_strategy,
        "probe_final_pass_rate": round(gepa_result["probe_pass_rate"], 3),
        "generation_log": gepa_result["generation_log"],
        "verdict": (
            "strong_transfer"  if gepa_transfer_lift > 0.15 else
            "moderate_transfer" if gepa_transfer_lift > 0.05 else
            "weak_transfer"    if gepa_transfer_lift > 0.0  else
            "no_transfer"
        ),
        "overfitting_flag": gepa_gap > 0.20,
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
        result = asyncio.run(compute_gepa_transfer_lift(cohort, generations=3, max_tasks=40))
        print(f"  Baseline transfer pass rate:     {result['baseline_transfer_pass_rate']:.3f}")
        print(f"  Transfer with GEPA strategy:     {result['transfer_pass_rate_with_strategy']:.3f}")
        print(f"  GEPA Transfer Lift:              {result['gepa_transfer_lift']:+.3f}")
        print(f"  GEPA Train Lift:                 {result['gepa_train_lift']:+.3f}")
        print(f"  GEPA Gap (overfit signal):       {result['gepa_gap']:+.3f}")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Best strategy: {result['best_strategy']}")
