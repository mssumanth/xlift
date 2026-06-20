import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from xlift.types import Rollout, RolloutRecord
from xlift.metrics.frontier import frontier_score, effective_ratio, band_fraction, reward_variance
from xlift.metrics.reachability import pass_at_k_minus_1
from xlift.metrics.reward_length import reward_length_corr
from xlift.metrics.entropy import answer_entropy


def _make_record(task_id: str, rewards_strong: list[float], n_tokens: list[int],
                  extracted: list[str | None], answer: str = "42") -> RolloutRecord:
    rollouts = [
        Rollout(text="", n_tokens=n, extracted=e, reward_strong=rs, reward_weak=rs)
        for rs, n, e in zip(rewards_strong, n_tokens, extracted)
    ]
    p_strong = sum(rewards_strong) / len(rewards_strong)
    return RolloutRecord(
        task_id=task_id, question="q", answer=answer, source="gsm8k",
        rollouts=rollouts, p_strong=p_strong, p_weak=p_strong,
    )


class TestFrontierScore:
    def test_peaks_at_half(self):
        rec = _make_record("t1", [1, 1, 0, 0], [10, 10, 10, 10], [None] * 4)
        score = frontier_score([rec])
        assert abs(score - 1.0) < 1e-6

    def test_zero_at_always_correct(self):
        rec = _make_record("t1", [1, 1, 1, 1], [10, 10, 10, 10], [None] * 4)
        assert frontier_score([rec]) == 0.0

    def test_zero_at_always_wrong(self):
        rec = _make_record("t1", [0, 0, 0, 0], [10, 10, 10, 10], [None] * 4)
        assert frontier_score([rec]) == 0.0

    def test_empty(self):
        assert frontier_score([]) == 0.0

    def test_high_p_low_score(self):
        rec = _make_record("t1", [1, 1, 1, 0], [10] * 4, [None] * 4)
        score = frontier_score([rec])
        assert score < 1.0

    def test_multiple_tasks_mean(self):
        r1 = _make_record("t1", [1, 1, 0, 0], [10] * 4, [None] * 4)  # p=0.5 → 1.0
        r2 = _make_record("t2", [1, 1, 1, 1], [10] * 4, [None] * 4)  # p=1.0 → 0.0
        assert abs(frontier_score([r1, r2]) - 0.5) < 1e-6


class TestEffectiveRatio:
    def test_all_learnable(self):
        rec = _make_record("t1", [1, 0, 1, 0], [10] * 4, [None] * 4)
        assert effective_ratio([rec]) == 1.0

    def test_none_learnable_all_correct(self):
        rec = _make_record("t1", [1, 1, 1, 1], [10] * 4, [None] * 4)
        assert effective_ratio([rec]) == 0.0

    def test_mixed(self):
        r1 = _make_record("t1", [1, 0], [10, 10], [None, None])  # 0 < p < 1 ✓
        r2 = _make_record("t2", [1, 1], [10, 10], [None, None])  # p=1 ✗
        assert effective_ratio([r1, r2]) == 0.5


class TestPassAtKMinus1:
    def test_noisy_but_reachable(self):
        # p=0.25 but pass@k=1.0 → delta=0.75
        rec = _make_record("t1", [1, 0, 0, 0], [10] * 4, [None] * 4)
        assert abs(pass_at_k_minus_1([rec]) - 0.75) < 1e-6

    def test_always_correct(self):
        # pass@1 == pass@k → delta=0
        rec = _make_record("t1", [1, 1, 1, 1], [10] * 4, [None] * 4)
        assert pass_at_k_minus_1([rec]) == 0.0

    def test_always_wrong(self):
        # pass@1=0, pass@k=0 → delta=0
        rec = _make_record("t1", [0, 0, 0, 0], [10] * 4, [None] * 4)
        assert pass_at_k_minus_1([rec]) == 0.0

    def test_empty(self):
        assert pass_at_k_minus_1([]) == 0.0


class TestRewardLengthCorr:
    def test_positive_when_correct_is_longer(self):
        # Correct rollouts are longer than wrong ones
        rec = _make_record("t1",
            rewards_strong=[1.0, 1.0, 0.0, 0.0],
            n_tokens=[100, 120, 10, 20],  # correct=long, wrong=short
            extracted=[None] * 4)
        rpb = reward_length_corr([rec])
        assert rpb > 0

    def test_negative_when_correct_is_shorter(self):
        rec = _make_record("t1",
            rewards_strong=[1.0, 1.0, 0.0, 0.0],
            n_tokens=[10, 20, 100, 120],  # correct=short, wrong=long
            extracted=[None] * 4)
        rpb = reward_length_corr([rec])
        assert rpb < 0

    def test_degenerate_all_correct_skipped(self):
        rec = _make_record("t1", [1, 1, 1, 1], [10, 20, 30, 40], [None] * 4)
        # no wrong rollouts → degenerate → mean over 0 valid tasks → 0
        assert reward_length_corr([rec]) == 0.0

    def test_empty(self):
        assert reward_length_corr([]) == 0.0


class TestAnswerEntropy:
    def test_zero_entropy_all_same(self):
        rec = _make_record("t1", [1, 0, 1, 0], [10] * 4, ["42", "42", "42", "42"])
        # All give same extracted answer → entropy = 0
        assert abs(answer_entropy([rec])) < 1e-9

    def test_max_entropy_all_different(self):
        rec = _make_record("t1", [1, 0, 0, 0], [10] * 4, ["1", "2", "3", "4"])
        h = answer_entropy([rec])
        assert h > 0  # 4 distinct answers → H = log2(4) = 2 bits
        assert abs(h - 2.0) < 1e-6

    def test_none_answers_skipped(self):
        rec = _make_record("t1", [1, 0, 1, 0], [10] * 4, [None, None, "42", "42"])
        h = answer_entropy([rec])
        assert h == 0.0  # only "42" appears → entropy 0

    def test_empty(self):
        assert answer_entropy([]) == 0.0
