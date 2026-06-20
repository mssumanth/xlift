from __future__ import annotations
import csv
import json
from pathlib import Path

from .types import Task, Rollout, RolloutRecord
from .verify import extract_pred, score_strong, score_weak

SEED = 42
SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "End with the final answer on its own line in the form: #### <number>"
)


def build_prompt(question: str, tokenizer) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_foundation(cfg) -> list[RolloutRecord]:
    """Load foundation.jsonl from disk."""
    path = Path(cfg.artifacts_dir) / "rollouts" / "foundation.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rollouts = [Rollout(**r) for r in d.pop("rollouts")]
            records.append(RolloutRecord(**d, rollouts=rollouts))
    return records


def run_foundation_rollouts(
    tasks: list[Task],
    *,
    model_path: str,
    k: int = 16,
    temperature: float = 0.7,
    max_tokens: int = 512,
    tp_size: int = 1,
    store_text: bool = True,
    force: bool = False,
    cfg=None,
) -> list[RolloutRecord]:
    artifacts = Path(cfg.artifacts_dir) if cfg else Path("./artifacts")
    out_jsonl = artifacts / "rollouts" / "foundation.jsonl"
    out_csv = artifacts / "index" / "pass_rate_index.csv"

    if not force and out_jsonl.exists() and out_csv.exists():
        print(f"Foundation rollouts already exist. Loading from {out_jsonl}")
        return load_foundation(cfg)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Running foundation rollouts: {len(tasks)} tasks × k={k} @ T={temperature}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.9,
        seed=SEED,
    )
    sampling = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        seed=SEED,
    )

    prompts = [build_prompt(t.question, tokenizer) for t in tasks]
    outputs = llm.generate(prompts, sampling)

    records = []
    for task, out in zip(tasks, outputs):
        rollouts = []
        for o in out.outputs:
            text = o.text
            n_tok = len(o.token_ids)
            extr = extract_pred(text)
            rs = score_strong(text, task.answer)
            rw = score_weak(text, task.answer)
            rollouts.append(Rollout(
                text=text if store_text else "",
                n_tokens=n_tok,
                extracted=extr,
                reward_strong=rs,
                reward_weak=rw,
            ))
        p_strong = sum(r.reward_strong for r in rollouts) / len(rollouts)
        p_weak = sum(r.reward_weak for r in rollouts) / len(rollouts)
        records.append(RolloutRecord(
            task_id=task.id,
            question=task.question,
            answer=task.answer,
            source=task.source,
            rollouts=rollouts,
            p_strong=p_strong,
            p_weak=p_weak,
        ))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_jsonl, "w") as f:
        for rec in records:
            d = {
                "task_id": rec.task_id,
                "question": rec.question,
                "answer": rec.answer,
                "source": rec.source,
                "p_strong": rec.p_strong,
                "p_weak": rec.p_weak,
                "rollouts": [
                    {
                        "text": r.text,
                        "n_tokens": r.n_tokens,
                        "extracted": r.extracted,
                        "reward_strong": r.reward_strong,
                        "reward_weak": r.reward_weak,
                    }
                    for r in rec.rollouts
                ],
            }
            f.write(json.dumps(d) + "\n")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "source", "p_strong", "p_weak", "n_rollouts"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "task_id": rec.task_id,
                "source": rec.source,
                "p_strong": round(rec.p_strong, 4),
                "p_weak": round(rec.p_weak, 4),
                "n_rollouts": len(rec.rollouts),
            })

    print(f"Wrote {len(records)} records → {out_jsonl}")
    return records
