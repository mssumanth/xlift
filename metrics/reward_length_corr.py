"""
Reward-Length Correlation — does writing more get rewarded more?

High positive correlation → model can hack this verifier by being verbose.
Near zero              → reward is length-independent (clean RL signal).
Negative               → model rambles when uncertain; length is a failure signal.

Computed over all rollouts pooled: corr(len(output_text), binary_reward)
Per-task correlation is too noisy at N=5; pooling across the cohort is more stable.
"""

import os
import asyncio
import numpy as np
import anthropic
from data.load_gsm8k import extract_answer, answers_match
from prompts.metrics import SOLVE_PROMPT
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"


def _get_client():
    return anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


async def _solve_once(question: str) -> str:
    resp = await _get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": SOLVE_PROMPT.format(question=question)}],
    )
    return resp.content[0].text


async def _rollout_task(task: dict, n_rollouts: int) -> list[tuple[int, float]]:
    """Returns list of (output_length_chars, binary_reward) for each rollout."""
    responses = await asyncio.gather(*[_solve_once(task["question"]) for _ in range(n_rollouts)])
    pairs = []
    for text in responses:
        pred = extract_answer(text)
        reward = 1.0 if (pred and task["answer"] and answers_match(pred, task["answer"])) else 0.0
        pairs.append((len(text), reward))
    return pairs


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(set(ys)) < 2 or len(set(xs)) < 2:
        return 0.0
    r = float(np.corrcoef(xs, ys)[0, 1])
    return 0.0 if np.isnan(r) else r


async def compute_cohort_length_correlation(
    cohort: list[dict],
    n_rollouts: int = 5,
    max_tasks: int = 50,
) -> dict:
    sample = cohort[:max_tasks]
    print(f"  Computing reward-length correlation on {len(sample)} tasks ({n_rollouts} rollouts each)...")

    batch_size = 10
    task_results = []
    all_lengths: list[float] = []
    all_rewards: list[float] = []

    for i in range(0, len(sample), batch_size):
        batch = sample[i:i + batch_size]
        batch_pairs = await asyncio.gather(*[_rollout_task(t, n_rollouts) for t in batch])
        for task, pairs in zip(batch, batch_pairs):
            lengths = [p[0] for p in pairs]
            rewards = [p[1] for p in pairs]
            task_corr = _pearson(lengths, rewards)
            task_results.append({
                "task_id": task["id"],
                "correlation": round(task_corr, 3),
                "mean_len_correct": round(
                    float(np.mean([l for l, r in pairs if r == 1.0])) if any(r == 1.0 for _, r in pairs) else 0.0, 1
                ),
                "mean_len_wrong": round(
                    float(np.mean([l for l, r in pairs if r == 0.0])) if any(r == 0.0 for _, r in pairs) else 0.0, 1
                ),
            })
            all_lengths.extend(lengths)
            all_rewards.extend(rewards)

    # Pool all rollouts for a stable cohort-level estimate
    global_corr = _pearson(all_lengths, all_rewards)
    mean_task_corr = float(np.mean([r["correlation"] for r in task_results]))

    # length_independence: 1.0 = reward ignores length (good), 0.0 = perfectly correlated (bad)
    length_independence = max(0.0, 1.0 - max(global_corr, 0.0))

    # misleading_lift_risk: if True, any GRPO lift measured on this cohort is suspect —
    # the model may have learned "be verbose" rather than "solve the problem."
    # This means even the oracle (actual_lift from GRPO) could be inflated.
    misleading_lift_risk = global_corr > 0.15

    return {
        "global_correlation": round(global_corr, 3),
        "mean_task_correlation": round(mean_task_corr, 3),
        "length_independence": round(length_independence, 3),
        "misleading_lift_risk": misleading_lift_risk,
        "n_rollouts_total": len(all_lengths),
        "task_results": task_results,
        "verdict": (
            "high_risk"      if global_corr > 0.30 else
            "moderate_risk"  if global_corr > 0.15 else
            "clean"          if global_corr >= -0.10 else
            "negative"
        ),
        "interpretation": (
            "Strong length-reward correlation — model can exploit this verifier by being verbose. "
            "Warning: any measured GRPO lift on this cohort is likely inflated and should not be trusted as ground truth."
            if global_corr > 0.30 else
            "Moderate correlation — GRPO lift may be partially explained by length exploitation, not genuine skill gain."
            if global_corr > 0.15 else
            "Length-independent reward — measured lift reflects genuine learning."
            if global_corr >= -0.10 else
            "Negative correlation — verbose outputs tend to be wrong. Length is a failure signal, not a shortcut."
        ),
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
        result = asyncio.run(compute_cohort_length_correlation(cohort, n_rollouts=5, max_tasks=30))
        print(f"  Global corr(length, reward): {result['global_correlation']:+.3f}")
        print(f"  Length independence:          {result['length_independence']:.3f}")
        print(f"  Verdict:                      {result['verdict']}")
        print(f"  {result['interpretation']}")
