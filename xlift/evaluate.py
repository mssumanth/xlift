from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional

from .types import Task
from .verify import score_strong


def eval_accuracy(
    model_path: str,
    tasks: list[Task],
    *,
    temperature: float = 0.0,
    tp_size: int = 1,
) -> dict:
    """Greedy evaluation with vLLM. Always uses score_strong — verifier-agnostic."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    _SYSTEM = (
        "Solve the math problem step by step. "
        "End with the final answer on its own line in the form: #### <number>"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=0.85)
    sampling = SamplingParams(n=1, temperature=temperature, max_tokens=512)

    prompts = []
    for t in tasks:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": t.question},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    outputs = llm.generate(prompts, sampling)

    correct_ids = []
    for task, out in zip(tasks, outputs):
        text = out.outputs[0].text
        if score_strong(text, task.answer) == 1.0:
            correct_ids.append(task.id)

    return {
        "acc": len(correct_ids) / len(tasks) if tasks else 0.0,
        "n": len(tasks),
        "correct_ids": correct_ids,
    }


def _wilson_ci(n_correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = n_correct / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def evaluate_base(cfg) -> dict:
    """Measure base model accuracy. Writes artifacts/eval/base.json. Idempotent."""
    artifacts = Path(cfg.artifacts_dir)
    out_path = artifacts / "eval" / "base.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    from .data import load_gsm8k
    test_tasks = load_gsm8k("test", cfg)[:cfg.eval_n]
    result = eval_accuracy(cfg.model_path, test_tasks, temperature=cfg.eval_temperature)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Base accuracy: {result['acc']:.3f} ({result['n']} tasks) → {out_path}")
    return result


def evaluate_cohort(name: str, cfg) -> dict:
    """Evaluate all checkpoints for a trained cohort. Writes artifacts/eval/<name>.json."""
    artifacts = Path(cfg.artifacts_dir)
    out_path = artifacts / "eval" / f"{name}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    base_path = artifacts / "eval" / "base.json"
    if not base_path.exists():
        raise FileNotFoundError("Run evaluate_base first")
    with open(base_path) as f:
        base = json.load(f)
    acc_before = base["acc"]
    n_total = base["n"]

    from .data import load_gsm8k
    test_tasks = load_gsm8k("test", cfg)[:cfg.eval_n]

    train_dir = artifacts / "train" / name
    if not train_dir.exists():
        raise FileNotFoundError(f"No training output for {name}")

    ckpt_dirs = sorted(
        [d for d in train_dir.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found in {train_dir}")

    per_step: dict[int, float] = {}
    for ckpt in ckpt_dirs:
        step = int(ckpt.name.split("-")[1])
        print(f"  Evaluating {name} checkpoint-{step}...")
        result = eval_accuracy(str(ckpt), test_tasks, temperature=cfg.eval_temperature)
        per_step[step] = result["acc"]

    best_step = max(per_step, key=per_step.__getitem__)
    acc_after = per_step[best_step]
    lift = acc_after - acc_before

    n_correct_before = round(acc_before * n_total)
    n_correct_after = round(acc_after * n_total)
    ci_before = _wilson_ci(n_correct_before, n_total)
    ci_after = _wilson_ci(n_correct_after, n_total)
    lift_ci_low = ci_after[0] - ci_before[1]
    lift_ci_high = ci_after[1] - ci_before[0]

    # Read final train reward from train_log
    train_log_path = train_dir / "train_log.jsonl"
    train_reward_final = 0.0
    if train_log_path.exists():
        with open(train_log_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            train_reward_final = lines[-1].get("mean_reward", 0.0)

    result = {
        "name": name,
        "per_step": {str(k): v for k, v in per_step.items()},
        "best_step": best_step,
        "acc_before": acc_before,
        "acc_after_best": acc_after,
        "lift": round(lift, 4),
        "lift_ci_low": round(lift_ci_low, 4),
        "lift_ci_high": round(lift_ci_high, 4),
        "train_reward_final": round(train_reward_final, 4),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {name}: lift={lift:+.3f} (best step {best_step}) → {out_path}")
    return result
