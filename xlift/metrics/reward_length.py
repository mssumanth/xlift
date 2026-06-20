from __future__ import annotations
import math
from ..types import RolloutRecord


def reward_length_corr(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Point-biserial correlation between n_tokens and binary reward, averaged per task.

    For each task that has BOTH correct and incorrect rollouts:
        r_pb = (M1 - M0) / sd_L * sqrt(f * (1 - f))
    where:
        M1  = mean n_tokens of correct rollouts
        M0  = mean n_tokens of incorrect rollouts
        sd_L = population std of ALL n_tokens in that task
        f   = fraction correct

    Returns mean r_pb over valid tasks. Degenerate tasks (all same reward) are skipped.

    C6 with reward='weak': length-spam hacks make passing rollouts longer → r_pb spikes,
    exposing the reward-hacking cohort.
    """
    attr = "reward_strong" if reward == "strong" else "reward_weak"
    task_rpbs = []

    for rec in records:
        lengths = [r.n_tokens for r in rec.rollouts]
        rewards = [getattr(r, attr) for r in rec.rollouts]

        correct_lens = [l for l, rv in zip(lengths, rewards) if rv == 1.0]
        wrong_lens = [l for l, rv in zip(lengths, rewards) if rv == 0.0]

        if not correct_lens or not wrong_lens:
            continue  # skip degenerate (all same reward)

        n = len(lengths)
        M1 = sum(correct_lens) / len(correct_lens)
        M0 = sum(wrong_lens) / len(wrong_lens)
        mean_all = sum(lengths) / n
        variance = sum((l - mean_all) ** 2 for l in lengths) / n
        sd_L = math.sqrt(variance) if variance > 0 else 0.0

        if sd_L == 0.0:
            continue

        f = len(correct_lens) / n
        r_pb = (M1 - M0) / sd_L * math.sqrt(f * (1 - f))
        task_rpbs.append(r_pb)

    return sum(task_rpbs) / len(task_rpbs) if task_rpbs else 0.0
