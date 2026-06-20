"""
GEPA Transfer Lift — uses dspy.GEPA (Reflective Prompt Evolution).

Does learning from one set of tasks transfer to completely unseen tasks?

Steps:
1. Split cohort 50/50 into probe set and transfer set
2. Run dspy.GEPA on probe set — evolves the solver's instructions via LM reflection
3. Apply optimized program to transfer set (tasks the model has never seen)
4. GEPA Transfer Lift = pass rate with optimized program - baseline pass rate

High Transfer Lift → the cohort teaches reusable reasoning patterns → worth training on
Low Transfer Lift  → the cohort is redundant or overfit → diversify before training
"""

import os
import asyncio
import random
import dspy
from dspy.teleprompt import GEPA
from data.load_gsm8k import extract_answer, answers_match
from dotenv import load_dotenv

load_dotenv()

FAST_MODEL  = "claude-haiku-4-5-20251001"
THINK_MODEL = "claude-sonnet-4-6"


class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)


def _math_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """DSPy/GEPA metric.

    GEPA calls this in two modes:
      - scoring pass  (pred_name is None) -> must return a plain float, because
        dspy.Evaluate aggregates with sum(); returning a dict raises TypeError.
      - feedback pass (pred_name set)     -> return dspy.Prediction(score, feedback)
        so GEPA's reflection LM gets textual guidance.
    """
    predicted = extract_answer(pred.answer)
    correct = bool(predicted and gold.answer and answers_match(predicted, gold.answer))
    score = 1.0 if correct else 0.0

    if pred_name is None:
        return score

    feedback = (
        "Correct!"
        if correct
        else (
            f"Got '{predicted or 'no number extracted'}' but expected '{gold.answer}'. "
            "Show each arithmetic step explicitly and verify the final number before "
            "writing #### <answer>."
        )
    )
    return dspy.Prediction(score=score, feedback=feedback)


def _eval_pass_rate(program: dspy.Module, tasks: list[dict], max_tasks: int = 20) -> float:
    """Single-shot pass rate of a DSPy program on a task list."""
    sample = tasks[:max_tasks]
    if not sample:
        return 0.0
    correct = 0
    for task in sample:
        try:
            pred = program(question=task["question"])
            predicted = extract_answer(pred.answer)
            if predicted and task["answer"] and answers_match(predicted, task["answer"]):
                correct += 1
        except Exception:
            pass
    return correct / len(sample)


def _run_gepa(student, trainset, generations):
    """Synchronous GEPA compile — called inside run_in_executor."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    think_lm = dspy.LM(
        model=f"anthropic/{THINK_MODEL}",
        api_key=api_key,
        temperature=1.0,
        max_tokens=4096,
    )
    optimizer = GEPA(
        metric=_math_metric,
        reflection_lm=think_lm,
        max_full_evals=max(2, generations),
        track_stats=True,
    )
    return optimizer.compile(student, trainset=trainset)


async def compute_gepa_transfer_lift(
    cohort: list[dict],
    generations: int = 3,
    n_rollouts: int = 4,
    max_tasks: int = 60,
) -> dict:
    """
    Full GEPA Transfer Lift computation using dspy.GEPA.

    Splits the cohort, runs dspy.GEPA.compile() on the probe half to discover
    an optimised reasoning instruction, then measures how well that instruction
    transfers to the unseen transfer half.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Configure DSPy default LM (used for forward passes during eval).
    # 1024 tokens: MATH chain-of-thought often exceeds 512 and gets truncated,
    # which drops the #### answer and tanks every score.
    fast_lm = dspy.LM(
        model=f"anthropic/{FAST_MODEL}",
        api_key=api_key,
        max_tokens=1024,
    )
    dspy.configure(lm=fast_lm)

    # Split cohort
    sample = cohort[:max_tasks]
    random.shuffle(sample)
    mid = len(sample) // 2
    probe_tasks    = sample[:mid]
    transfer_tasks = sample[mid:]

    print(f"  Probe: {len(probe_tasks)} tasks | Transfer: {len(transfer_tasks)} tasks")

    student = MathSolver()

    # Baseline pass rates (unoptimised program)
    print("  Measuring baseline pass rates...")
    eval_n = min(20, len(probe_tasks))
    baseline_probe    = _eval_pass_rate(student, probe_tasks,    max_tasks=eval_n)
    baseline_transfer = _eval_pass_rate(student, transfer_tasks, max_tasks=eval_n)

    # Build DSPy trainset from probe tasks
    trainset = [
        dspy.Example(question=t["question"], answer=t["answer"]).with_inputs("question")
        for t in probe_tasks
    ]

    # Run dspy.GEPA in a thread so we don't block the event loop
    print(f"  Running dspy.GEPA on probe set ({generations} eval budgets)...")
    loop = asyncio.get_event_loop()
    optimized = await loop.run_in_executor(
        None,
        lambda: _run_gepa(student, trainset, generations),
    )

    # Extract the evolved instruction from the optimised module
    try:
        best_strategy = optimized.predictor.signature.instructions
    except AttributeError:
        best_strategy = "GEPA-optimised chain-of-thought strategy"

    # Post-optimisation pass rates
    print("  Evaluating optimised program on probe and transfer sets...")
    probe_final    = _eval_pass_rate(optimized, probe_tasks,    max_tasks=eval_n)
    transfer_final = _eval_pass_rate(optimized, transfer_tasks, max_tasks=eval_n)

    gepa_transfer_lift = transfer_final - baseline_transfer
    gepa_train_lift    = probe_final    - baseline_probe
    gepa_gap           = gepa_train_lift - gepa_transfer_lift

    # Build a generation log from GEPA's detailed_results when available
    generation_log = [{"gen": 0, "best_score": round(baseline_probe, 3), "strategy": "Think step by step."}]
    try:
        scores = optimized.detailed_results.get("val_aggregate_scores", [])
        for i, sc in enumerate(scores):
            generation_log.append({
                "gen": i + 1,
                "best_score": round(float(sc), 3),
                "strategy": best_strategy,
            })
    except (AttributeError, TypeError):
        generation_log.append({"gen": 1, "best_score": round(probe_final, 3), "strategy": best_strategy})

    return {
        "baseline_transfer_pass_rate":     round(baseline_transfer, 3),
        "transfer_pass_rate_with_strategy": round(transfer_final, 3),
        "gepa_transfer_lift":              round(gepa_transfer_lift, 3),
        "gepa_train_lift":                 round(gepa_train_lift, 3),
        "gepa_gap":                        round(gepa_gap, 3),
        "best_strategy":                   best_strategy,
        "probe_final_pass_rate":           round(probe_final, 3),
        "generation_log":                  generation_log,
        "verdict": (
            "strong_transfer"   if gepa_transfer_lift > 0.15 else
            "moderate_transfer" if gepa_transfer_lift > 0.05 else
            "weak_transfer"     if gepa_transfer_lift > 0.0  else
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
