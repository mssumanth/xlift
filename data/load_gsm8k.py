"""
Load GSM8K and partition into three cohorts based on model pass rates.

Cohort 1 — Too Easy:      pass rate > 0.80  (model already knows this)
Cohort 2 — Learnable:     pass rate 0.30-0.70 (the sweet spot for RL)
Cohort 3 — Too H, l from this yet)
"""

import re
import json
import random
import os
from pathlib import Path
from typing import Optional

# --- Use macOS system keychain for SSL (handles corporate TLS-intercepting proxies) ---
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from datasets import load_dataset
from tqdm import tqdm

COHORT_DIR = Path(__file__).parent.parent / "results" / "cohorts"


def extract_answer(text: str) -> Optional[str]:
    """Pull the final answer from a model response (GSM8K-style #### or \\boxed{})."""
    if not text:
        return None
    # Preferred: everything after the last #### up to end of that line
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip().rstrip(".")
    # Next: a \boxed{...} answer (common when the model mirrors MATH style)
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    # Fallback: last standalone number in the text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1] if numbers else None


def normalize_answer(ans: str) -> str:
    """Strip LaTeX formatting so two equivalent answers compare equal."""
    if ans is None:
        return ""
    s = str(ans).strip()
    # Unwrap common wrappers
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    # Normalize fraction macros: \dfrac -> \frac
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Remove formatting macros / spacing / decoration
    for tok in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "^\\circ",
                "\\circ", "\\%", "\\$", "$", "%", "\\degree", "{}"]:
        s = s.replace(tok, "")
    s = s.replace(",", "").replace(" ", "")
    s = s.strip("{}").strip()
    return s.lower()


def _to_float(s: str) -> Optional[float]:
    """Try to interpret a normalized answer as a float, including \\frac{a}{b}."""
    m = re.fullmatch(r"\\frac\{?(-?\d+\.?\d*)\}?\{?(-?\d+\.?\d*)\}?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.fullmatch(r"(-?\d+\.?\d*)/(-?\d+\.?\d*)", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Pull the final answer from a MATH solution's \\boxed{...} (handles nested braces)."""
    idx = solution.rfind(r"\boxed")
    if idx == -1:
        return None
    i = idx + len(r"\boxed")
    while i < len(solution) and solution[i] != "{":
        i += 1
    if i >= len(solution):
        return None
    depth = 0
    start = i + 1
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:j].strip()
    return None


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if two answers are equivalent, tolerant of LaTeX formatting."""
    if predicted is None or ground_truth is None:
        return False
    np_, ng = normalize_answer(predicted), normalize_answer(ground_truth)
    if np_ == "" or ng == "":
        return False
    # Numeric comparison first (handles fractions, degrees, thousands separators)
    fp, fg = _to_float(np_), _to_float(ng)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-4
    # Fall back to normalized string equality (matrices, expressions, etc.)
    return np_ == ng


def load_gsm8k(split: str = "train", max_tasks: int = 2000) -> list[dict]:
    """Load GSM8K tasks as flat dicts."""
    print(f"Loading GSM8K ({split})...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
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
    # qwedsacf/competition_math is a parquet mirror (no loading script) with fields:
    # problem, solution, level ("Level 1".."Level 5"), type
    ds = load_dataset("qwedsacf/competition_math", split="train")

    easy, frontier, hard = [], [], []

    for i, item in enumerate(ds):
        # Level can occasionally be malformed (e.g. "Level ?") — default to 3
        level_str = str(item.get("level", "Level 3")).replace("Level ", "").strip()
        try:
            level = int(level_str)
        except ValueError:
            continue  # skip unlabelled problems
        if not 1 <= level <= 5:
            continue
        final_answer = extract_boxed_answer(item["solution"])
        if final_answer is None:
            continue  # need a checkable final answer for downstream metrics
        task = {
            "id": i,
            "question": item["problem"],
            "answer": final_answer,
            "answer_full": item["solution"],
            "level": level,
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
