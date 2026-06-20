from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from ..types import Cohort, RolloutRecord, CohortMetrics
from .frontier import frontier_score, effective_ratio, band_fraction, reward_variance
from .reachability import pass_at_k_minus_1
from .reward_length import reward_length_corr
from .entropy import answer_entropy
from .baselines import mean_token_length, avg_pass_rate, vendi_score, redundancy, dist_match


def compute_cohort_metrics(
    cohort: Cohort,
    foundation: list[RolloutRecord],
    embedder,
    test_questions: list[str],
    cfg,
) -> CohortMetrics:
    """Compute all cheap signals for a cohort from foundation rollouts.

    Pure function of the foundation rollout artifacts — no model loaded here.
    For C6 (verifier='weak'), all metrics that take reward= use 'weak'.
    """
    task_id_set = set(cohort.task_ids)
    records = [r for r in foundation if r.task_id in task_id_set]
    reward = cohort.verifier

    questions = [r.question for r in records]

    fs = frontier_score(records, reward)
    er = effective_ratio(records, reward)
    bf = band_fraction(records, reward)
    rv = reward_variance(records, reward)
    pak1 = pass_at_k_minus_1(records, reward)
    rlc = reward_length_corr(records, reward)
    ae = answer_entropy(records, reward)
    mtl = mean_token_length(records)
    apr = avg_pass_rate(records, reward)

    vs = vendi_score(questions, embedder) if questions else 1.0
    red = redundancy(questions, embedder) if len(questions) >= 2 else 0.0
    dm = dist_match(questions, test_questions, embedder) if test_questions else 0.0

    metrics = CohortMetrics(
        name=cohort.name,
        n=len(records),
        frontier_score=round(fs, 4),
        effective_ratio=round(er, 4),
        band_fraction=round(bf, 4),
        reward_variance=round(rv, 4),
        pass_at_k_minus_1=round(pak1, 4),
        reward_length_corr=round(rlc, 4),
        answer_entropy=round(ae, 4),
        mean_token_length=round(mtl, 2),
        avg_pass_rate=round(apr, 4),
        vendi_score=round(vs, 4),
        redundancy=round(red, 4),
        dist_match=round(dm, 4),
    )

    # Write artifact
    artifacts = Path(cfg.artifacts_dir)
    out_path = artifacts / "cohorts" / f"{cohort.name}.metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        d = metrics.__dict__.copy()
        json.dump(d, f, indent=2)
    print(f"Metrics written → {out_path}")

    return metrics
