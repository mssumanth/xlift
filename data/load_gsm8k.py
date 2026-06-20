"""
Load GSM8K and partition into three cohorts based on model pass rates.

Cohort 1 — Too Easy:      pass rate > 0.80  (model already knows this)
Cohort 2 — Learnable:     pass rate 0.30-0.70 (the sweet spot for RL)
Cohort 3 — Too Hard:      pass rate < 0.15  (model cannot learn from this yet)
"""

import re
import json
import random
import os
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm

COHORT_DIR = Path(__file__).parent.parent / "results" / "cohorts"


def extract_answer(text: str) -> Optional[str]:
    """Pull the final number from a GSM8K answer string."""
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # fallback: last number in text
    numbers = re.findall(r"[\-\d]+\.?\d*", text)
    return numbers[-1] if numbers else None


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if two answer strings represent the same number."""
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-4
    except (ValueError, TypeError):
        return predicted.strip() == ground_truth.strip()


def load_gsm8k(split: str = "train", max_tasks: int = 2000) -> list[dict]:
    """Load GSM8K tasks as flat dicts."""
    print(f"Loading GSM8K ({split})...")
    ds = load_dataset("gsm8k", "main", split=split)
    tasks = []
    for i, item in enumerate(ds):
        if i >= max_tasks:
            break
        tasks.append({
            "id": i,
            "question": item["question"],
            "answer_full": item["answer"],
            "answer": extract_answer(item["answer"]),
        })
    print(f"Loaded {len(tasks)} tasks.")
    return tasks


def measure_pass_rates_with_claude(
    tasks: list[dict],
    n_rollouts: int = 5,
    sample_size: int = 300,
) -> dict[int, float]:
    """
    Use Claude to measure per-task pass rates.
    Faster than loading a local model — good for hackathon speed.
    """
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # Use Haiku — fast and cheap for rollouts
    MODEL = "claude-haiku-4-5-20251001"

    sampled = random.sample(tasks, min(sample_size, len(tasks)))
    pass_rates = {}

    for task in tqdm(sampled, desc="Measuring pass rates"):
        correct = 0
        for _ in range(n_rollouts):
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Solve this math problem. Give only the final number as your answer "
                            f"on the last line after ####.\n\n{task['question']}"
                        )
                    }]
                )
                predicted = extract_answer(resp.content[0].text)
                if predicted and task["answer"] and answers_match(predicted, task["answer"]):
                    correct += 1
            except Exception:
                pass
        pass_rates[task["id"]] = correct / n_rollouts

    return pass_rates


def partition_into_cohorts(
    tasks: list[dict],
    pass_rates: dict[int, float],
    cohort_size: int = 150,
) -> dict[str, list[dict]]:
    """Split tasks into three cohorts based on pass rate."""
    easy     = [t for t in tasks if t["id"] in pass_rates and pass_rates[t["id"]] > 0.80]
    frontier = [t for t in tasks if t["id"] in pass_rates and 0.30 <= pass_rates[t["id"]] <= 0.70]
    hard     = [t for t in tasks if t["id"] in pass_rates and pass_rates[t["id"]] < 0.15]

    print(f"\nCohort sizes before sampling:")
    print(f"  Easy (p > 0.80):          {len(easy)} tasks")
    print(f"  Frontier (0.30-0.70):     {len(frontier)} tasks")
    print(f"  Hard (p < 0.15):          {len(hard)} tasks")

    # Attach pass rates to tasks for later metric computation
    def attach(task_list):
        for t in task_list:
            t["pass_rate"] = pass_rates[t["id"]]
        return t

    for cohort in [easy, frontier, hard]:
        for t in cohort:
            t["pass_rate"] = pass_rates[t["id"]]

    return {
        "easy":     random.sample(easy,     min(cohort_size, len(easy))),
        "frontier": random.sample(frontier, min(cohort_size, len(frontier))),
        "hard":     random.sample(hard,     min(cohort_size, len(hard))),
    }


def save_cohorts(cohorts: dict[str, list[dict]]):
    """Save cohorts to disk so training scripts can load them."""
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tasks in cohorts.items():
        path = COHORT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"Saved {len(tasks)} tasks → {path}")


def load_cohorts() -> dict[str, list[dict]]:
    """Load saved cohorts from disk."""
    cohorts = {}
    for name in ["easy", "frontier", "hard"]:
        path = COHORT_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                cohorts[name] = json.load(f)
    return cohorts


def use_difficulty_labels_shortcut(max_per_cohort: int = 150) -> dict[str, list[dict]]:
    """
    FAST PATH: Use the MATH dataset's pre-labelled difficulty levels.
    Skips pass rate measurement entirely — use this if time is short.

    Level 1-2 → Easy
    Level 3   → Frontier
    Level 4-5 → Hard
    """
    print("Loading MATH dataset with pre-labelled difficulty...")
    ds = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)

    easy, frontier, hard = [], [], []

    for i, item in enumerate(ds):
        level = int(item.get("level", "Level 3").replace("Level ", ""))
        task = {
            "id": i,
            "question": item["problem"],
            "answer": item["solution"],
            "answer_full": item["solution"],
            "pass_rate": [0.9, 0.75, 0.5, 0.2, 0.05][level - 1],
        }
        if level <= 2:
            easy.append(task)
        elif level == 3:
            frontier.append(task)
        else:
            hard.append(task)

    cohorts = {
        "easy":     random.sample(easy,     min(max_per_cohort, len(easy))),
        "frontier": random.sample(frontier, min(max_per_cohort, len(frontier))),
        "hard":     random.sample(hard,     min(max_per_cohort, len(hard))),
    }

    print(f"Easy: {len(cohorts['easy'])}  Frontier: {len(cohorts['frontier'])}  Hard: {len(cohorts['hard'])}")
    save_cohorts(cohorts)
    return cohorts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortcut", action="store_true",
                        help="Use MATH difficulty labels instead of measuring pass rates (faster)")
    parser.add_argument("--cohort-size", type=int, default=150)
    parser.add_argument("--sample-size", type=int, default=300,
                        help="How many GSM8K tasks to sample for pass rate measurement")
    args = parser.parse_args()

    if args.shortcut:
        cohorts = use_difficulty_labels_shortcut(args.cohort_size)
    else:
        tasks = load_gsm8k(max_tasks=args.sample_size * 2)
        pass_rates = measure_pass_rates_with_claude(tasks, n_rollouts=5, sample_size=args.sample_size)
        cohorts = partition_into_cohorts(tasks, pass_rates, args.cohort_size)
        save_cohorts(cohorts)

    print("\nCohort summary:")
    for name, tasks in cohorts.items():
        avg_pr = sum(t["pass_rate"] for t in tasks) / len(tasks)
        print(f"  {name:10s}: {len(tasks)} tasks, avg pass rate {avg_pr:.2f}")
