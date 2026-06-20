from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    id: str
    question: str
    answer: str         # normalized integer/decimal string
    source: str         # "gsm8k" | "synthetic"
    meta: dict = field(default_factory=dict)


@dataclass
class Rollout:
    text: str
    n_tokens: int
    extracted: Optional[str]
    reward_strong: float
    reward_weak: float


@dataclass
class RolloutRecord:
    task_id: str
    question: str
    answer: str
    source: str
    rollouts: list[Rollout]
    p_strong: float     # mean reward_strong
    p_weak: float       # mean reward_weak


@dataclass
class Cohort:
    name: str
    verifier: str       # "strong" | "weak"
    task_ids: list[str]
    property_varied: str
    note: str = ""


@dataclass
class CohortMetrics:
    name: str
    n: int
    frontier_score: float
    effective_ratio: float
    band_fraction: float
    reward_variance: float
    pass_at_k_minus_1: float
    reward_length_corr: float
    answer_entropy: float
    mean_token_length: float
    avg_pass_rate: float
    vendi_score: float
    redundancy: float
    dist_match: float
    grad_norm: Optional[float] = None


@dataclass
class CohortResult:
    name: str
    metrics: CohortMetrics
    acc_before: float
    acc_after_best: float
    lift: float
    lift_ci_low: float
    lift_ci_high: float
    best_step: int
    train_reward_final: float


@dataclass
class TrainCfg:
    steps: int = 120
    ckpt_every: int = 25
    group_size: int = 8
    lr: float = 2e-6
    kl_beta: float = 0.005
    max_prompt: int = 512
    max_completion: int = 512
    temperature: float = 0.7
    per_device_bs: int = 8
    grad_accum: int = 1
    full_finetune: bool = True
    seed: int = 0
