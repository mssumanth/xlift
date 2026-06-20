from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class Config:
    model_path: str = field(default_factory=lambda: os.environ.get(
        "XLIFT_MODEL", "./models/qwen2.5-0.5b-instruct"
    ))
    artifacts_dir: str = field(default_factory=lambda: os.environ.get(
        "XLIFT_ARTIFACTS", "./artifacts"
    ))
    seed: int = 0

    # rollouts / foundation
    k_rollouts: int = 16
    roll_temperature: float = 0.7
    roll_max_tokens: int = 512
    foundation_n_tasks: int = 4000
    store_rollout_text: bool = True

    # cohorts
    cohort_size: int = 250
    frontier_band: tuple = (0.4, 0.6)
    easy_band: tuple = (0.8, 1.0)
    hard_band: tuple = (0.0, 0.2)
    band_widen_step: float = 0.05
    redundancy_threshold: float = 0.85

    # training (IDENTICAL across all cohorts — cohort is the only variable)
    train_steps: int = 120
    ckpt_every: int = 25
    group_size: int = 8
    learning_rate: float = 2e-6
    kl_beta: float = 0.005
    max_prompt_len: int = 512
    max_completion_len: int = 512
    per_device_bs: int = 8
    grad_accum: int = 1
    full_finetune: bool = True

    # eval
    eval_n: int = 1319
    eval_temperature: float = 0.0

    # embeddings / metrics
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"

    # synth / claude
    synth_model: str = "claude-haiku-4-5-20251001"
    synth_candidates: int = 1000

    backend: str = "trl"

    @property
    def artifacts(self) -> Path:
        return Path(self.artifacts_dir)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        cfg = cls()
        if path and Path(path).exists() and _HAS_YAML:
            with open(path) as f:
                overrides = yaml.safe_load(f) or {}
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg
