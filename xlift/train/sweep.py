from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path

from .backend import TrainBackend
from ..types import Cohort, Task, TrainCfg

# Train C2 and C1 first (POC gate), then the rest, then C6 last
_PRIORITY_ORDER = ["C2_frontier", "C1_easy", "C3_hard", "C4_mixed", "C5_redundant", "C7_synth", "C6_weak"]


def _ckpt_exists(out_dir: str, steps: int) -> bool:
    return (Path(out_dir) / f"checkpoint-{steps}").exists()


def run_sweep(
    cohorts: dict[str, Cohort],
    tasks_by_cohort: dict[str, list[Task]],
    *,
    backend: TrainBackend,
    model_path: str,
    artifacts_dir: str,
    n_gpus: int = 8,
    cfg: TrainCfg,
) -> None:
    """Queue cohort training jobs across available GPUs.

    - 1 GPU per job (0.5B model colocated)
    - Concurrency cap = n_gpus
    - Resumable: skips cohorts whose final checkpoint already exists
    - Order: C2, C1 first (POC gate), then C3-C5, C7, then C6 last
    - A failed cohort is logged and skipped, never fatal
    """
    artifacts = Path(artifacts_dir)
    status_path = artifacts / "sweep_status.json"
    status: dict[str, str] = {}

    ordered_names = [n for n in _PRIORITY_ORDER if n in cohorts]
    ordered_names += [n for n in cohorts if n not in ordered_names]

    pending = []
    for name in ordered_names:
        out_dir = str(artifacts / "train" / name)
        if _ckpt_exists(out_dir, cfg.steps):
            print(f"  {name}: already trained, skipping")
            status[name] = "done"
            continue
        pending.append((name, cohorts[name], tasks_by_cohort.get(name, []), out_dir))

    def _write_status():
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

    _write_status()

    available_gpus = list(range(n_gpus))
    running: list[tuple[str, subprocess.Popen, int]] = []  # (name, proc, gpu_id)
    pending_iter = iter(pending)
    done = False

    while not done or running:
        # Fill available GPU slots
        while available_gpus and not done:
            try:
                name, cohort, tasks, out_dir = next(pending_iter)
            except StopIteration:
                done = True
                break

            gpu_id = available_gpus.pop(0)
            print(f"  Launching {name} on GPU {gpu_id}...")
            status[name] = "running"
            _write_status()

            # Launch as a subprocess using the CLI
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(
                [
                    "python", "-m", "xlift.cli", "train",
                    "--cohorts", name,
                    "--gpus", str(gpu_id),
                ],
                env=env,
            )
            running.append((name, proc, gpu_id))

        # Poll running jobs
        still_running = []
        for name, proc, gpu_id in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((name, proc, gpu_id))
            else:
                available_gpus.append(gpu_id)
                if ret == 0:
                    print(f"  {name}: completed successfully")
                    status[name] = "done"
                else:
                    print(f"  {name}: FAILED (exit code {ret}), skipping")
                    status[name] = "failed"
                _write_status()
        running = still_running

        if running:
            time.sleep(30)

    print(f"Sweep complete. Status written to {status_path}")
    _write_status()
