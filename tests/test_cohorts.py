"""Tests for cohort construction invariants."""
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from xlift.types import Task, Rollout, RolloutRecord
from xlift.config import Config


def _make_foundation(n: int = 600, seed: int = 0) -> tuple[list[RolloutRecord], str, str]:
    """Create a synthetic foundation + write pass_rate_index.csv. Returns (records, csv_path, artifacts_dir)."""
    rng = random.Random(seed)
    records = []
    tmpdir = tempfile.mkdtemp()
    artifacts = Path(tmpdir) / "artifacts"
    (artifacts / "index").mkdir(parents=True)
    (artifacts / "cohorts").mkdir(parents=True)

    rows = []
    for i in range(n):
        p_strong = rng.random()
        p_weak = min(1.0, p_strong + rng.uniform(0, 0.3))
        task_id = f"gsm8k-train-{i:06d}"
        rollouts = [
            Rollout(text="", n_tokens=50, extracted="42",
                    reward_strong=float(rng.random() < p_strong),
                    reward_weak=float(rng.random() < p_weak))
            for _ in range(16)
        ]
        rec = RolloutRecord(
            task_id=task_id, question=f"Question {i}", answer="42", source="gsm8k",
            rollouts=rollouts, p_strong=p_strong, p_weak=p_weak,
        )
        records.append(rec)
        rows.append(f"{task_id},gsm8k,{p_strong:.4f},{p_weak:.4f},16")

    csv_path = str(artifacts / "index" / "pass_rate_index.csv")
    with open(csv_path, "w") as f:
        f.write("task_id,source,p_strong,p_weak,n_rollouts\n")
        f.write("\n".join(rows))

    return records, csv_path, str(artifacts)


def _make_cfg(artifacts_dir: str, cohort_size: int = 50) -> Config:
    cfg = Config()
    cfg.artifacts_dir = artifacts_dir
    cfg.cohort_size = cohort_size
    cfg.seed = 0
    return cfg


class TestBandConstraints:
    def setup_method(self):
        self.foundation, self.csv, self.art = _make_foundation(600)
        self.cfg = _make_cfg(self.art, cohort_size=40)

    def test_c1_easy_band(self):
        from xlift.cohorts import build_cohorts
        cohorts = build_cohorts(self.csv, self.foundation, self.cfg)
        c1 = cohorts["C1_easy"]
        f_map = {r.task_id: r for r in self.foundation}
        for tid in c1.task_ids:
            p = f_map[tid].p_strong
            assert p >= 0.8 - 0.05 - 1e-9, f"{tid} p={p} not in easy band"

    def test_c2_frontier_band(self):
        from xlift.cohorts import build_cohorts
        cohorts = build_cohorts(self.csv, self.foundation, self.cfg)
        c2 = cohorts["C2_frontier"]
        f_map = {r.task_id: r for r in self.foundation}
        for tid in c2.task_ids:
            p = f_map[tid].p_strong
            assert 0.4 - 0.05 - 1e-9 <= p <= 0.6 + 0.05 + 1e-9, f"{tid} p={p} not in frontier band"

    def test_c3_hard_band(self):
        from xlift.cohorts import build_cohorts
        cohorts = build_cohorts(self.csv, self.foundation, self.cfg)
        c3 = cohorts["C3_hard"]
        f_map = {r.task_id: r for r in self.foundation}
        for tid in c3.task_ids:
            p = f_map[tid].p_strong
            assert p <= 0.2 + 0.05 + 1e-9, f"{tid} p={p} not in hard band"


class TestC6InvariantVsC2:
    def setup_method(self):
        self.foundation, self.csv, self.art = _make_foundation(600)
        self.cfg = _make_cfg(self.art, cohort_size=40)

    def test_c6_task_ids_equal_c2(self):
        from xlift.cohorts import build_cohorts
        cohorts = build_cohorts(self.csv, self.foundation, self.cfg)
        assert sorted(cohorts["C6_weak"].task_ids) == sorted(cohorts["C2_frontier"].task_ids)

    def test_c6_verifier_is_weak(self):
        from xlift.cohorts import build_cohorts
        cohorts = build_cohorts(self.csv, self.foundation, self.cfg)
        assert cohorts["C6_weak"].verifier == "weak"

    def test_c2_verifier_is_strong(self):
        from xlift.cohorts import build_cohorts
        cohorts = build_cohorts(self.csv, self.foundation, self.cfg)
        assert cohorts["C2_frontier"].verifier == "strong"


class TestCohortSizes:
    def setup_method(self):
        self.foundation, self.csv, self.art = _make_foundation(600)

    def test_cohort_size_respected(self):
        from xlift.cohorts import build_cohorts
        cfg = _make_cfg(self.art, cohort_size=30)
        cohorts = build_cohorts(self.csv, self.foundation, cfg)
        for name, c in cohorts.items():
            if name == "C7_synth":
                continue  # empty without synth tasks
            assert len(c.task_ids) <= 30, f"{name} has {len(c.task_ids)} tasks > 30"

    def test_seven_cohorts_built(self):
        from xlift.cohorts import build_cohorts
        cfg = _make_cfg(self.art, cohort_size=20)
        cohorts = build_cohorts(self.csv, self.foundation, cfg)
        expected = {"C1_easy", "C2_frontier", "C3_hard", "C4_mixed", "C5_redundant", "C6_weak", "C7_synth"}
        assert set(cohorts.keys()) == expected
