from __future__ import annotations
from abc import ABC, abstractmethod
from ..types import Cohort, Task, TrainCfg


class TrainBackend(ABC):
    @abstractmethod
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
        """Run one GRPO job for a cohort.

        Constraints:
        - Reward = verify.score(text, answer, cohort.verifier) — this is the ONLY variable between cohorts
        - Checkpoint every cfg.ckpt_every steps
        - Write train_log.jsonl: {step, mean_reward, reward_std, effective_ratio}
        - effective_ratio = fraction of generation groups with reward_std > 0
        - Resumable: skip if final checkpoint already exists
        - Returns out_dir
        """
        ...
