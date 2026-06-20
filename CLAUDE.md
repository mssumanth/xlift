# xLift — CLAUDE.md

## What This Is

xLift predicts which post-training data cohorts will produce real RL lift **before you train**.
It's a lightweight pre-training probe: run model rollouts on candidate data, measure learnability signals, composite them into a score, and compare against actual GRPO lift.

**Hackathon:** Anthropic/Etched/Cognition/Mercor — 24 hours, $50k top prize, $100k+ total.
**Track:** Talent Marketplace + Applied AI — "systems that judge capability of a model."
**Judges:** Founders from Etched, Cognition, Prime Intellect, Mercor + guests from frontier labs.
**Compute:** 8x H100s per team (Prime Intellect). Use `Qwen/Qwen2.5-0.5B-Instruct` as base model.

---

## Core Idea

GRPO only learns from tasks where rollouts have **mixed outcomes** (some right, some wrong).
xLift measures this frontier cheaply — without training — using six signals:

| Signal | What it measures | Cost |
|---|---|---|
| BoundaryScore | 4p(1-p) — peaks when pass rate ≈ 0.5 | Low (rollouts) |
| RewardVariance | Var(scores across rollouts) | Free after rollouts |
| RepairGain | Score after feedback − score before | Medium |
| GEPA Transfer Lift | Do lessons from probe tasks transfer to unseen tasks? | High (inference) |
| AntiCheat Robustness | Can fake solutions fool the verifier? | Low–medium |
| Failure-Mode Coverage | Entropy over failure-mode distribution | Low–medium |

**Composite:** `xLift(D) = 0.25*Boundary + 0.20*Repair + 0.35*GEPATransfer + 0.20*RewardTrust`

**Validation target:** xLift score should correlate with actual GRPO lift across cohorts.

---

## File Structure

```
xlift/
├── run_experiment.py          # Main orchestrator — start here
├── data/
│   └── load_gsm8k.py          # GSM8K loader + cohort partitioner + MATH shortcut
├── metrics/
│   ├── boundary_score.py      # BoundaryScore + RewardVariance rollouts
│   ├── repair_gain.py         # Feedback loop — hints + retry
│   ├── gepa_transfer.py       # GEPA evolutionary prompt loop + transfer test
│   ├── anticheat.py           # Red-team the verifier
│   └── xlift_score.py         # Composite score + recommendation engine
├── training/
│   └── grpo_train.py          # GRPO fine-tune on one cohort, measure lift
├── eval/
│   └── visualize.py           # 4 plots: scatter, boundary map, pareto, bar chart
├── prompts/
│   └── metrics.py             # All LLM prompts (solve, feedback, GEPA, anticheat)
└── results/                   # Created at runtime
    ├── cohorts/               # easy.json, frontier.json, hard.json
    ├── xlift_scores/          # Per-cohort xLift metric results
    ├── grpo/                  # GRPO training outputs + lift_result.json
    └── plots/                 # PNG visualizations
```

---

## Run Order

```bash
# Step 1 — create cohorts (fast path uses MATH difficulty labels)
python run_experiment.py --step data --shortcut --cohort-size 150

# Step 2 — compute xLift metrics on each cohort
python run_experiment.py --step metrics --max-tasks 40 --rollouts 5

# Step 3 — GRPO training (needs H100s; run one per cohort)
python run_experiment.py --step train --cohort frontier --steps 200
python run_experiment.py --step train --cohort easy     --steps 200
python run_experiment.py --step train --cohort hard     --steps 200

# Step 4 — generate all plots
python run_experiment.py --step visualize

# Check current status
python run_experiment.py --step status
```

---

## Models Used

- **Rollouts / metrics:** `claude-haiku-4-5-20251001` (fast, cheap)
- **GEPA reflection + mutation:** `claude-sonnet-4-6` (smarter reasoning)
- **GRPO base model:** `Qwen/Qwen2.5-0.5B-Instruct` (fits on H100, fast to train)
- **Eval benchmark:** GSM8K test split (held-out, never in training cohorts)

---

## Known Bugs (fix before demo)

### 1. MATH shortcut answer format mismatch — BREAKS fast path
`data/load_gsm8k.py:167` sets `task["answer"] = item["solution"]` (full LaTeX solution).
All downstream metrics call `extract_answer` + `answers_match` expecting a bare number.
**Fix:** Extract the boxed/final number from MATH solutions, or switch shortcut to use GSM8K difficulty proxies instead.

### 2. AntiCheat always reports 100% hack susceptibility — useless signal
`prompts/metrics.py` — both attack prompts explicitly instruct `End with: #### {correct_answer}`.
Every fake solution passes the verifier by design. Signal is constant, not informative.
**Fix:** Remove the explicit correct answer from attack prompts. Test whether confabulated wrong reasoning still reaches the right number, OR use a wrong-but-plausible answer to test whether the verifier catches it.

### 3. Dead code in `partition_into_cohorts`
`load_gsm8k.py:119` — `attach()` function returns `t` (last item) instead of the list. Never called anyway. Delete it.

---

## Key Signals to Emphasize to Judges (Kunal's priority order)

1. **Fraction-in-band** — fraction of tasks with pass rate strictly in (0, 1). This is the spine. Mean boundary score hides the distribution; report fraction-in-band separately.
2. **GEPA Transfer Lift** — the differentiating signal. Tests whether lessons transfer to unseen tasks before any weight updates.
3. **Reward–length correlation** — flags cohorts where longer outputs get higher rewards (hackable verifier). Sophisticated angle judges will appreciate.
4. **pass@k − pass@1** — reinforceable headroom; separates "noisy but reachable" from "noisy but hopeless."
5. **Answer-distribution entropy** — confidently wrong = dead; uncertain with occasional correct = target.

Cheap baselines to compare against on the Pareto plot: token length, average pass rate, reward variance alone, dedup/diversity.

---

## Demo Visuals (priority order)

1. **Scatter plot** — xLift predicted score vs. actual post-GRPO lift across cohorts. This is the headline result.
2. **Boundary Map** — x: base pass rate, y: RepairGain, color: RewardTrust. Shows four regions: Mastered / Learnable Frontier / Latent / Beyond Boundary.
3. **Pareto frontier** — cost-to-compute vs. predictive power. xLift should sit above the frontier (high signal, medium cost vs. full training).
4. **Cohort comparison bar chart** — all components side by side for easy/frontier/hard.

---

## Pitch One-Liner

> xLift probes a model before training, finds the learnable frontier, tests whether GEPA lessons transfer, red-teams the verifier, and predicts which data cohorts will produce real post-RL lift — at a fraction of the cost of training on each cohort.

---

## Environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and HF_TOKEN
pip install -r requirements.txt
```

Results auto-save to `./results/`. Safe to re-run steps — outputs are overwritten per cohort.
