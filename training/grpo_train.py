"""
GRPO training script — train one cohort, measure accuracy lift.

Run once per cohort:
  python training/grpo_train.py --cohort frontier --output results/grpo/frontier

Requires: 8x H100s, ~60-90 minutes per cohort for Qwen2.5-0.5B
"""

import os
import re
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL   = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR", "./results"))


def extract_answer(text: str):
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-\d]+\.?\d*", text)
    return numbers[-1] if numbers else None


def answers_match(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return pred.strip() == gold.strip()


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


def evaluate_accuracy(model, tokenizer, tasks: list[dict], n_samples: int = 100) -> float:
    """Measure model accuracy on a set of tasks."""
    import torch
    sample = tasks[:n_samples]
    correct = 0

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
                temperature=0.0,
                do_sample=False,
            )
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(text)
        if pred and task["answer"] and answers_match(pred, task["answer"]):
            correct += 1

    return correct / len(sample)


def train_grpo(cohort_name: str, output_dir: str, max_steps: int = 200):
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

    # Load eval set (held-out GSM8K test split)
    eval_path = RESULTS_DIR / "cohorts" / "eval_set.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_tasks = json.load(f)
    else:
        from datasets import load_dataset
        gsm_test = load_dataset("gsm8k", "main", split="test")
        eval_tasks = [
            {"id": i, "question": item["question"], "answer": extract_answer(item["answer"])}
            for i, item in enumerate(gsm_test)
        ]
        with open(eval_path, "w") as f:
            json.dump(eval_tasks, f)

    print(f"\nLoading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline accuracy BEFORE training
    print("Measuring baseline accuracy...")
    baseline_acc = evaluate_accuracy(model, tokenizer, eval_tasks, n_samples=100)
    print(f"Baseline accuracy: {baseline_acc:.1%}")

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

    def reward_fn(completions, prompts, **kwargs):
        """
        Reward = 1 if final number matches expected answer, else 0.
        Simple verifier — intentionally basic to demonstrate AntiCheat risk.
        """
        rewards = []
        for completion, prompt in zip(completions, prompts):
            # Find the matching task answer
            task_answer = next(
                (t["answer"] for t in cohort if format_prompt(t) == prompt), None
            )
            text = completion[0]["content"] if isinstance(completion, list) else completion
            pred = extract_answer(text)
            correct = bool(pred and task_answer and answers_match(pred, task_answer))
            rewards.append(1.0 if correct else 0.0)
        return rewards

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
    )

    print(f"\nTraining on {cohort_name} cohort ({len(cohort)} tasks, {max_steps} steps)...")
    trainer.train()

    # Accuracy AFTER training
    print("\nMeasuring post-training accuracy...")
    post_acc = evaluate_accuracy(model, tokenizer, eval_tasks, n_samples=100)
    lift = post_acc - baseline_acc

    result = {
        "cohort": cohort_name,
        "model": BASE_MODEL,
        "baseline_accuracy": round(baseline_acc, 4),
        "post_training_accuracy": round(post_acc, 4),
        "actual_lift": round(lift, 4),
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
    print(f"Lift:      {lift:+.1%}")
    print(f"Saved → {result_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["easy", "frontier", "hard"], required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    output = args.output or str(RESULTS_DIR / "grpo" / args.cohort)
    train_grpo(args.cohort, output, args.steps)
