from __future__ import annotations
from .backend import TrainBackend
from ..types import Cohort, Task, TrainCfg


class PrimeRLBackend(TrainBackend):
    """Optional prime-rl backend. Build TRL path first (§9.1), use this only after D1."""

    def train_cohort(
        self,
        cohort: Cohort,
        tasks: list[Task],
        *,
        model_path: str,
        gpu_ids: list[int],
        cfg: TrainCfg,
        out_dir: str,
    ) -> str:
        raise NotImplementedError(
            "prime-rl backend not yet implemented. "
            "Use TRLBackend (default). See spec §9.2 for prime-rl TOML config."
        )
