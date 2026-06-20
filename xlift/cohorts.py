from __future__ import annotations
import csv
import json
import random
from pathlib import Path
from typing import Optional

from .types import Task, Cohort, RolloutRecord


def _load_index(index_csv: str) -> dict[str, dict]:
    index = {}
    with open(index_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            index[row["task_id"]] = {
                "p_strong": float(row["p_strong"]),
                "p_weak": float(row["p_weak"]),
                "source": row.get("source", "gsm8k"),
                "n_rollouts": int(row["n_rollouts"]),
            }
    return index


def _sample_band(
    index: dict, lo: float, hi: float, n: int, rng: random.Random, band_widen_step: float = 0.05
) -> list[str]:
    """Sample task_ids with p_strong in [lo, hi]. Widens band if insufficient."""
    candidates = [tid for tid, d in index.items() if lo <= d["p_strong"] <= hi]
    orig_lo, orig_hi = lo, hi
    while len(candidates) < n and (lo > 0.0 or hi < 1.0):
        lo = max(0.0, lo - band_widen_step)
        hi = min(1.0, hi + band_widen_step)
        candidates = [tid for tid, d in index.items() if lo <= d["p_strong"] <= hi]
    if lo != orig_lo or hi != orig_hi:
        print(f"  Band widened from [{orig_lo:.2f},{orig_hi:.2f}] to [{lo:.2f},{hi:.2f}] ({len(candidates)} tasks)")
    return rng.sample(candidates, min(n, len(candidates)))


def _redundant(
    frontier_ids: list[str],
    foundation_map: dict[str, RolloutRecord],
    n: int,
    embedder,
    sim_threshold: float = 0.85,
) -> list[str]:
    """Greedily build a near-duplicate-heavy subset of frontier tasks."""
    import numpy as np

    questions = [foundation_map[tid].question for tid in frontier_ids if tid in foundation_map]
    ids_with_q = [tid for tid in frontier_ids if tid in foundation_map]
    if not ids_with_q:
        return frontier_ids[:n]

    embs = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    sim = embs @ embs.T

    selected = []
    seed_idx = int(np.argmax(np.sum(sim, axis=1)))
    selected.append(ids_with_q[seed_idx])

    while len(selected) < n:
        # Pick the task most similar to already-selected (greedy nearest-neighbor)
        selected_indices = [ids_with_q.index(tid) for tid in selected if tid in ids_with_q]
        max_sim = -1.0
        best_idx = -1
        for i, tid in enumerate(ids_with_q):
            if tid in selected:
                continue
            avg_sim = float(np.mean([sim[i, j] for j in selected_indices]))
            if avg_sim > max_sim:
                max_sim = avg_sim
                best_idx = i
        if best_idx < 0:
            break
        selected.append(ids_with_q[best_idx])

    return selected[:n]


def _build_prompt_field(question: str, tokenizer=None) -> list[dict]:
    """Return chat-format prompt list for trainers."""
    system = (
        "Solve the math problem step by step. "
        "End with the final answer on its own line in the form: #### <number>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def build_cohorts(
    index_csv: str,
    foundation: list[RolloutRecord],
    cfg,
    synth_records: Optional[list[RolloutRecord]] = None,
) -> dict[str, Cohort]:
    artifacts = Path(cfg.artifacts_dir)
    n = cfg.cohort_size
    rng = random.Random(cfg.seed)

    index = _load_index(index_csv)
    foundation_map = {r.task_id: r for r in foundation}

    if synth_records:
        for r in synth_records:
            index[r.task_id] = {
                "p_strong": r.p_strong,
                "p_weak": r.p_weak,
                "source": r.source,
                "n_rollouts": len(r.rollouts),
            }
            foundation_map[r.task_id] = r

    all_ids = list(index.keys())

    # Build all task lookups
    task_lookup: dict[str, Task] = {
        r.task_id: Task(
            id=r.task_id, question=r.question, answer=r.answer, source=r.source
        )
        for r in list(foundation) + (synth_records or [])
    }

    def write_cohort(cohort: Cohort) -> Cohort:
        out_dir = artifacts / "cohorts"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write task jsonl
        tasks = [task_lookup[tid] for tid in cohort.task_ids if tid in task_lookup]
        with open(out_dir / f"{cohort.name}.jsonl", "w") as f:
            for t in tasks:
                d = {"id": t.id, "question": t.question, "answer": t.answer,
                     "source": t.source, "prompt": _build_prompt_field(t.question)}
                f.write(json.dumps(d) + "\n")
        # Write cohort metadata
        with open(out_dir / f"{cohort.name}.cohort.json", "w") as f:
            json.dump({
                "name": cohort.name,
                "verifier": cohort.verifier,
                "task_ids": cohort.task_ids,
                "property_varied": cohort.property_varied,
                "note": cohort.note,
            }, f, indent=2)
        print(f"Cohort {cohort.name}: {len(cohort.task_ids)} tasks, verifier={cohort.verifier}")
        return cohort

    cohorts = {}

    # C1_easy: p_strong ∈ [0.8,1.0]
    lo, hi = cfg.easy_band
    ids = _sample_band(index, lo, hi, n, rng, cfg.band_widen_step)
    cohorts["C1_easy"] = write_cohort(Cohort(
        name="C1_easy", verifier="strong", task_ids=ids,
        property_varied="pass rate in [0.8,1.0] — already mastered",
    ))

    # C2_frontier: p_strong ∈ [0.4,0.6]
    lo, hi = cfg.frontier_band
    frontier_ids = _sample_band(index, lo, hi, n, rng, cfg.band_widen_step)
    cohorts["C2_frontier"] = write_cohort(Cohort(
        name="C2_frontier", verifier="strong", task_ids=frontier_ids,
        property_varied="pass rate in [0.4,0.6] — learnable frontier",
    ))

    # C3_hard: p_strong ∈ [0.0,0.2]
    lo, hi = cfg.hard_band
    ids = _sample_band(index, lo, hi, n, rng, cfg.band_widen_step)
    cohorts["C3_hard"] = write_cohort(Cohort(
        name="C3_hard", verifier="strong", task_ids=ids,
        property_varied="pass rate in [0.0,0.2] — beyond reach",
    ))

    # C4_mixed: random across all p
    mixed_ids = rng.sample(all_ids, min(n, len(all_ids)))
    cohorts["C4_mixed"] = write_cohort(Cohort(
        name="C4_mixed", verifier="strong", task_ids=mixed_ids,
        property_varied="random sample across all pass rates",
    ))

    # C5_redundant: frontier band but near-duplicate heavy
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg.embedder)
        all_frontier_ids = [
            tid for tid, d in index.items()
            if cfg.frontier_band[0] <= d["p_strong"] <= cfg.frontier_band[1]
        ]
        red_ids = _redundant(all_frontier_ids, foundation_map, n, embedder, cfg.redundancy_threshold)
    except Exception as e:
        print(f"  Warning: redundancy sampling failed ({e}), falling back to frontier sample")
        red_ids = rng.sample(
            [tid for tid, d in index.items()
             if cfg.frontier_band[0] <= d["p_strong"] <= cfg.frontier_band[1]],
            min(n, sum(1 for d in index.values()
                       if cfg.frontier_band[0] <= d["p_strong"] <= cfg.frontier_band[1]))
        )
    cohorts["C5_redundant"] = write_cohort(Cohort(
        name="C5_redundant", verifier="strong", task_ids=red_ids,
        property_varied="frontier band but near-duplicate heavy",
        note="Low diversity — embeds cluster tightly",
    ))

    # C6_weak: SAME task_ids as C2_frontier, but verifier=weak
    cohorts["C6_weak"] = write_cohort(Cohort(
        name="C6_weak", verifier="weak", task_ids=frontier_ids,
        property_varied="same tasks as C2_frontier with weak (gameable) verifier",
        note="Control: isolates verifier quality as the only variable vs C2",
    ))

    # C7_synth: synthetic frontier tasks (written separately by synth.py then re-rolled)
    synth_frontier = []
    if synth_records:
        synth_frontier = [
            r.task_id for r in synth_records
            if cfg.frontier_band[0] <= r.p_strong <= cfg.frontier_band[1]
        ]
        synth_frontier = rng.sample(synth_frontier, min(n, len(synth_frontier)))
    cohorts["C7_synth"] = write_cohort(Cohort(
        name="C7_synth", verifier="strong", task_ids=synth_frontier,
        property_varied="Claude-generated synthetic frontier tasks",
        note="C7 may be empty until synth.py + re-rollout are run",
    ))

    return cohorts


def load_cohort(name: str, cfg) -> tuple[Cohort, list[Task]]:
    """Load a saved cohort from disk."""
    artifacts = Path(cfg.artifacts_dir) / "cohorts"
    with open(artifacts / f"{name}.cohort.json") as f:
        d = json.load(f)
    cohort = Cohort(**d)
    tasks = []
    with open(artifacts / f"{name}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tasks.append(Task(
                id=row["id"], question=row["question"],
                answer=row["answer"], source=row["source"],
            ))
    return cohort, tasks
