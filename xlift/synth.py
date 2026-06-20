from __future__ import annotations
import json
import os
import random
from pathlib import Path

from .types import Task
from .verify import normalize_number

_BATCH_SIZE = 10
_INSTRUCTION = (
    "You are a math problem writer for elementary and middle school students. "
    "Write {n} NEW grade-school math word problems of similar style and difficulty "
    "to the examples below. Each must have a single integer answer. "
    "Return ONLY valid JSON as a list: "
    '[{{"question": "...", "answer": <integer>}}]\n\n'
    "Examples:\n{examples}"
)


def _format_examples(seed_tasks: list[Task]) -> str:
    lines = []
    for i, t in enumerate(seed_tasks, 1):
        lines.append(f"{i}. Q: {t.question}\n   A: {t.answer}")
    return "\n".join(lines)


def generate_synthetic_tasks(
    n_candidates: int,
    *,
    model: str = "claude-haiku-4-5-20251001",
    seed_examples: list[Task],
    cfg=None,
) -> list[Task]:
    """Generate Claude-produced GSM8K-style tasks for C7.

    Uses prompt caching on the static instruction+examples block.
    Writes artifacts/synth/candidates.jsonl.
    Generates ~4×N to survive frontier filtering.
    """
    import anthropic

    artifacts = Path(cfg.artifacts_dir) if cfg else Path("./artifacts")
    out_path = artifacts / "synth" / "candidates.jsonl"

    if out_path.exists():
        tasks = []
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(Task(**d))
        print(f"Loaded {len(tasks)} synthetic candidates from {out_path}")
        return tasks

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sample_seeds = random.sample(seed_examples, min(5, len(seed_examples)))
    examples_text = _format_examples(sample_seeds)

    tasks: list[Task] = []
    batches_needed = (n_candidates + _BATCH_SIZE - 1) // _BATCH_SIZE
    print(f"Generating {n_candidates} synthetic candidates ({batches_needed} batches)...")

    for batch_idx in range(batches_needed):
        n_this_batch = min(_BATCH_SIZE, n_candidates - len(tasks))
        prompt = _INSTRUCTION.format(n=n_this_batch, examples=examples_text)

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Extract JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start < 0 or end <= start:
                print(f"  Batch {batch_idx}: no JSON array found, skipping")
                continue
            items = json.loads(text[start:end])
            for item in items:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if not q or not a:
                    continue
                tasks.append(Task(
                    id=f"synth-{len(tasks):05d}",
                    question=q,
                    answer=normalize_number(a),
                    source="synthetic",
                ))
        except Exception as e:
            print(f"  Batch {batch_idx} failed: {e}")

        if (batch_idx + 1) % 10 == 0:
            print(f"  {len(tasks)} tasks generated so far...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps({"id": t.id, "question": t.question,
                                 "answer": t.answer, "source": t.source}) + "\n")

    print(f"Wrote {len(tasks)} synthetic candidates → {out_path}")
    return tasks
