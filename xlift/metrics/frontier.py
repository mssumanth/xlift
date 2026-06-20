from __future__ import annotations
from ..types import RolloutRecord


def _p(rec: RolloutRecord, reward: str) -> float:
    return rec.p_strong if reward == "strong" else rec.p_weak


def frontier_score(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Mean 4*p*(1-p) over tasks. Peaks at 1.0 when p=0.5, zero at p=0 or p=1."""
    if not records:
        return 0.0
    return sum(4 * _p(r, reward) * (1 - _p(r, reward)) for r in records) / len(records)


def effective_ratio(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Fraction of tasks with 0 < p < 1 (gradient can flow)."""
    if not records:
        return 0.0
    return sum(1 for r in records if 0 < _p(r, reward) < 1) / len(records)


def band_fraction(
    records: list[RolloutRecord],
    reward: str = "strong",
    lo: float = 0.3,
    hi: float = 0.7,
) -> float:
    """Fraction of tasks with pass rate strictly in (lo, hi)."""
    if not records:
        return 0.0
    return sum(1 for r in records if lo < _p(r, reward) < hi) / len(records)


def reward_variance(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Mean p*(1-p) — unscaled version of frontier_score."""
    if not records:
        return 0.0
    return sum(_p(r, reward) * (1 - _p(r, reward)) for r in records) / len(records)
