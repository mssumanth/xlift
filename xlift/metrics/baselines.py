from __future__ import annotations
import math
from ..types import RolloutRecord


def mean_token_length(records: list[RolloutRecord]) -> float:
    """Mean n_tokens across all rollouts across all tasks."""
    all_lens = [r.n_tokens for rec in records for r in rec.rollouts]
    return sum(all_lens) / len(all_lens) if all_lens else 0.0


def avg_pass_rate(records: list[RolloutRecord], reward: str = "strong") -> float:
    """Mean pass rate p over tasks (monotonic baseline — contrast with frontier_score)."""
    if not records:
        return 0.0
    attr = "p_strong" if reward == "strong" else "p_weak"
    return sum(getattr(r, attr) for r in records) / len(records)


def vendi_score(questions: list[str], embedder) -> float:
    """exp(Shannon entropy of normalized eigenvalues of the cosine similarity matrix K).
    High = diverse; low = redundant."""
    import numpy as np

    if not questions:
        return 1.0
    embs = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    K = embs @ embs.T  # cosine similarity matrix (normalized embeddings)
    K = (K + K.T) / 2  # ensure symmetry
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 1e-10]
    eigvals = eigvals / eigvals.sum()
    entropy = -float(np.sum(eigvals * np.log(eigvals + 1e-30)))
    return math.exp(entropy)


def redundancy(questions: list[str], embedder, thr: float = 0.85) -> float:
    """Fraction of pairs with cosine similarity > thr."""
    import numpy as np

    if len(questions) < 2:
        return 0.0
    embs = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=False)
    sim = embs @ embs.T
    n = len(questions)
    high_sim = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if sim[i, j] > thr:
                high_sim += 1
    return high_sim / total if total > 0 else 0.0


def dist_match(cohort_questions: list[str], test_questions: list[str], embedder) -> float:
    """Mean over cohort tasks of max cosine similarity to any test task.
    Measures how well the cohort covers the test distribution."""
    import numpy as np

    if not cohort_questions or not test_questions:
        return 0.0
    c_embs = embedder.encode(cohort_questions, normalize_embeddings=True, show_progress_bar=False)
    t_embs = embedder.encode(test_questions, normalize_embeddings=True, show_progress_bar=False)
    sim = c_embs @ t_embs.T  # (n_cohort, n_test)
    return float(np.mean(np.max(sim, axis=1)))
