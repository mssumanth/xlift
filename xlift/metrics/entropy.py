from __future__ import annotations
import math
from collections import Counter
from ..types import RolloutRecord


def answer_entropy(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Mean Shannon entropy (bits) of empirical answer distribution across rollouts per task.

    High entropy = model produces many different answers (uncertain).
    Zero entropy = model always gives the same answer (confident, may be wrong or right).
    """
    if not records:
        return 0.0

    task_entropies = []
    for rec in records:
        answers = [r.extracted for r in rec.rollouts if r.extracted is not None]
        if not answers:
            task_entropies.append(0.0)
            continue
        counts = Counter(answers)
        n = len(answers)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
        task_entropies.append(entropy)

    return sum(task_entropies) / len(task_entropies) if task_entropies else 0.0
