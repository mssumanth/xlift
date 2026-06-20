from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

from .types import Task
from .verify import normalize_number

_HASH_RE = re.compile(r"####\s*([\S]+)")


def _extract_gsm8k_answer(text: str) -> str:
    m = _HASH_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    return ""


def load_gsm8k(split: str, cfg=None) -> list[Task]:
    """Load GSM8K and write artifacts/data/gsm8k_{split}.jsonl. Idempotent."""
    from datasets import load_dataset

    artifacts = Path(cfg.artifacts_dir) if cfg else Path("./artifacts")
    out_path = artifacts / "data" / f"gsm8k_{split}.jsonl"

    if out_path.exists():
        tasks = []
        with open(out_path) as f:
            for line in f:
                d = json.loads(line)
                tasks.append(Task(**d))
        return tasks

    print(f"Loading GSM8K ({split})...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks = []
    for idx, item in enumerate(ds):
        answer = _extract_gsm8k_answer(item["answer"])
        tasks.append(Task(
            id=f"gsm8k-{split}-{idx:06d}",
            question=item["question"],
            answer=answer,
            source="gsm8k",
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps({
                "id": t.id, "question": t.question,
                "answer": t.answer, "source": t.source,
            }) + "\n")

    print(f"Wrote {len(tasks)} tasks → {out_path}")
    return tasks
