from __future__ import annotations
from ..types import RolloutRecord


def pass_at_k_minus_1(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Mean over tasks of (pass@k − pass@1).

    pass@1 = p (mean reward over rollouts).
    pass@k = 1.0 if any rollout has reward == 1.0 else 0.0.
    Range [0, 1]. High = reachable-but-noisy = great RL target.
    """
    if not records:
        return 0.0

    attr = "reward_strong" if reward == "strong" else "reward_weak"
    deltas = []
    for rec in records:
        rewards = [getattr(r, attr) for r in rec.rollouts]
        if not rewards:
            continue
        pass1 = sum(rewards) / len(rewards)
        passk = 1.0 if any(rv == 1.0 for rv in rewards) else 0.0
        deltas.append(passk - pass1)

    return sum(deltas) / len(deltas) if deltas else 0.0
