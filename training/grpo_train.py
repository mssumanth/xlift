"""
GRPO training script — train one cohort, measure accuracy lift.

Run once per cohort (sequential on 1 GPU):
  python run_experiment.py --step train --cohort frontier

Or all three in parallel on a multi-GPU box:
  bash train_parallel.sh

Default model Qwen2.5-1.5B-Instruct: ~45-75 min/cohort on one H100.
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Reuse the LaTeX-aware extraction/matching from the data module so training,
# eval, and the metrics all score answers identically (critical for MATH).
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.load_gsm8k import extract_answer, answers_match  # noqa: E402

BASE_MODEL   = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR", "./results"))


def build_reward_fn(correct_answer: str):
    """Returns a reward function for one task."""
    def reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            pred = extract_answer(text)
            rewards.append(1.0 if pred and answers_match(pred, correct_answer) else 0.0)
        return rewards
    return reward


def evaluate_per_item(model, tokenizer, tasks: list[dict], n_samples: int = 200) -> list[int]:
    """Greedy-decode each task; return a 0/1 correctness list (one per task).

    Per-item results (not just the mean) let us bootstrap a paired CI on the lift,
    so we can tell a real gain from eval noise (Anish BUILDLOG P10).
    """
    import torch
    sample = tasks[:n_samples]
    flags = []
    for task in sample:
        prompt = (
            f"Solve this math problem. Show reasoning, then write your final answer as #### <number>.\n\n"
            f"{task['question']}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(text)
        flags.append(1 if (pred and task["answer"] and answers_match(pred, task["answer"])) else 0)
    return flags


def evaluate_accuracy(model, tokenizer, tasks: list[dict], n_samples: int = 100) -> float:
    """Mean accuracy convenience wrapper over evaluate_per_item."""
    flags = evaluate_per_item(model, tokenizer, tasks, n_samples=n_samples)
    return sum(flags) / len(flags) if flags else 0.0


def bootstrap_lift_ci(baseline_flags, post_flags, n_boot: int = 2000, seed: int = 0):
    """Paired bootstrap CI for (post - baseline) accuracy over the same test items."""
    import numpy as np
    rng = np.random.default_rng(seed)
    b = np.asarray(baseline_flags, dtype=float)
    p = np.asarray(post_flags, dtype=float)
    n = len(b)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(p[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def train_grpo(cohort_name: str, output_dir: str, max_steps: int = 200,
               use_weak_verifier: bool = False):
    """
    Fine-tune the base model on one cohort using GRPO.
    Saves the model and records before/after accuracy.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    # Load cohort
    cohort_path = RESULTS_DIR / "cohorts" / f"{cohort_name}.json"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort not found: {cohort_path}. Run data/load_gsm8k.py first.")

    with open(cohort_path) as f:
        cohort = json.load(f)

    # Load held-out MATH eval set (created once by the data step, same distribution
    # as the cohorts). No GSM8K — training and eval must be the same domain.
    eval_path = RESULTS_DIR / "cohorts" / "eval_set.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval set not found: {eval_path}. Run "
            f"`python run_experiment.py --step data --shortcut` first."
        )
    with open(eval_path) as f:
        eval_tasks = json.load(f)

    print(f"\nLoading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Held-out test set (disjoint from the training cohort by construction).
    # Use the full eval set for a tighter estimate; the same items are scored
    # before and after so the lift CI is paired.
    test_tasks = eval_tasks[:200]

    # Baseline accuracy BEFORE training (per-item, for the paired bootstrap)
    print("Measuring baseline accuracy...")
    baseline_flags = evaluate_per_item(model, tokenizer, test_tasks, n_samples=len(test_tasks))
    baseline_acc = sum(baseline_flags) / len(baseline_flags)
    print(f"Baseline accuracy: {baseline_acc:.1%} (n={len(baseline_flags)})")

    # Build GRPO dataset
    def format_prompt(task):
        return (
            f"Solve this math problem. Show reasoning, then write #### <number>.\n\n"
            f"{task['question']}"
        )

    hf_dataset = Dataset.from_list([
        {"prompt": format_prompt(t), "answer": t["answer"]}
        for t in cohort
    ])

    # GRPO config — tuned for fast iteration on H100s
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = GRPOConfig(
        output_dir=str(output_path),
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=max_steps,
        num_generations=4,        # rollouts per prompt
        max_completion_length=512,
        report_to="none",
    )

    if use_weak_verifier or cohort_name == "weak_verifier":
        # C6 weak verifier: reward long responses regardless of correctness.
        # This trains the model to pad/ramble — the reward-hacking exhibit.
        # Evaluation is still done with the strong verifier below, so the lift
        # numbers expose the gap: reward goes up, true accuracy does NOT.
        WEAK_LENGTH_THRESHOLD = 200  # characters — easy for a padded response to hit
        def reward_fn(completions, prompts, **kwargs):
            """Gameable length-based verifier: reward = 1 if len(response) > threshold."""
            rewards = []
            for completion in completions:
                text = completion[0]["content"] if isinstance(completion, list) else completion
                rewards.append(1.0 if len(text) >= WEAK_LENGTH_THRESHOLD else 0.0)
            return rewards
        print("[C6 WEAK VERIFIER] Using length-rewarding reward function — reward hacking exhibit.")
    else:
        def reward_fn(completions, prompts, **kwargs):
            """
            Reward = 1 if final number matches expected answer, else 0.
            Simple verifier — intentionally basic to demonstrate AntiCheat risk.
            """
            rewards = []
            for completion, prompt in zip(completions, prompts):
                task_answer = next(
                    (t["answer"] for t in cohort if format_prompt(t) == prompt), None
                )
                text = completion[0]["content"] if isinstance(completion, list) else completion
                pred = extract_answer(text)
                correct = bool(pred and task_answer and answers_match(pred, task_answer))
                rewards.append(1.0 if correct else 0.0)
            return rewards

    # TRL renamed `tokenizer` -> `processing_class` (0.11+); newest versions reject
    # `tokenizer=`. Try the new kwarg, fall back to the old one for older TRL.
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=config,
            train_dataset=hf_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )

    print(f"\nTraining on {cohort_name} cohort ({len(cohort)} tasks, {max_steps} steps)...")
    trainer.train()

    # Accuracy AFTER training — same test items, for a paired comparison
    print("\nMeasuring post-training accuracy...")
    post_flags = evaluate_per_item(model, tokenizer, test_tasks, n_samples=len(test_tasks))
    post_acc = sum(post_flags) / len(post_flags)
    lift = post_acc - baseline_acc

    # Bootstrap CI so we can separate real lift from eval noise
    _, ci_low, ci_high = bootstrap_lift_ci(baseline_flags, post_flags)
    significant = ci_low > 0  # 95% CI excludes zero

    result = {
        "cohort": cohort_name,
        "model": BASE_MODEL,
        "baseline_accuracy": round(baseline_acc, 4),
        "post_training_accuracy": round(post_acc, 4),
        "actual_lift": round(lift, 4),
        "lift_ci_low": round(ci_low, 4),
        "lift_ci_high": round(ci_high, 4),
        "lift_significant": bool(significant),
        "eval_n": len(test_tasks),
        "max_steps": max_steps,
        "cohort_size": len(cohort),
    }

    result_path = output_path / "lift_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Cohort: {cohort_name}")
    print(f"Baseline:  {baseline_acc:.1%}")
    print(f"Post-train:{post_acc:.1%}")
    print(f"Lift:      {lift:+.1%}  95% CI [{ci_low:+.1%}, {ci_high:+.1%}]"
          f"  {'SIGNIFICANT' if significant else '(within noise)'}")
    print(f"Saved → {result_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["easy", "frontier", "hard", "mixed", "weak_verifier"], required=True)
    parser.add_argument("--weak-verifier", action="store_true",
                        help="Use length-rewarding weak verifier (auto-set for weak_verifier cohort)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    output = args.output or str(RESULTS_DIR / "grpo" / args.cohort)
    train_grpo(args.cohort, output, args.steps,
               use_weak_verifier=getattr(args, "weak_verifier", False))
