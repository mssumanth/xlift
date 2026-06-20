"""
xLift — Main Experiment Runner

Runs the full pipeline:
  1. Load / create cohorts from GSM8K
  2. Compute xLift metrics on each cohort (BoundaryScore, RepairGain, GEPA Transfer, AntiCheat)
  3. [Optional] Run GRPO training on each cohort (requires H100s)
  4. Generate all plots

Usage:
  # Step 1 only — create cohorts fast using difficulty labels
  python run_experiment.py --step data --shortcut

  # SMOKE — 2-min Qwen validation before the full run (do this first on a new box)
  XLIFT_BACKEND=qwen python run_experiment.py --step metrics --smoke

  # Step 2 — compute xLift metrics on saved cohorts (Qwen on GPU for valid signal)
  XLIFT_BACKEND=qwen python run_experiment.py --step metrics

  # Step 3 — run GRPO (do this on H100 machine)
  python run_experiment.py --step train --cohort frontier

  # Step 4 — generate plots from saved results
  python run_experiment.py --step visualize

  # Run everything except training
  python run_experiment.py --step all --shortcut
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent / "results"


def step_data(args):
    from data.load_gsm8k import (
        load_gsm8k, measure_pass_rates_with_claude,
        partition_into_cohorts, save_cohorts,
        use_difficulty_labels_shortcut,
    )
    if args.shortcut:
        print("Using difficulty label shortcut (fast path)...")
        cohorts = use_difficulty_labels_shortcut(max_per_cohort=args.cohort_size)
    else:
        print("Measuring pass rates with Claude (slower but more accurate)...")
        tasks = load_gsm8k(max_tasks=args.sample_size * 2)
        pass_rates = asyncio.run(
            measure_pass_rates_with_claude(tasks, n_rollouts=5, sample_size=args.sample_size)
        ) if False else measure_pass_rates_with_claude(tasks, n_rollouts=5, sample_size=args.sample_size)
        cohorts = partition_into_cohorts(tasks, pass_rates, args.cohort_size)
        save_cohorts(cohorts)

    print("\nCohorts ready:")
    for name, tasks in cohorts.items():
        avg = sum(t["pass_rate"] for t in tasks) / len(tasks)
        print(f"  {name:10s}: {len(tasks)} tasks, avg pass rate {avg:.2f}")


def step_metrics(args):
    from metrics.boundary_score      import compute_cohort_boundary_score
    from metrics.repair_gain         import compute_cohort_repair_gain
    from metrics.gepa_transfer       import compute_gepa_transfer_lift
    from metrics.anticheat           import compute_cohort_anticheat
    from metrics.reward_length_corr  import compute_cohort_length_correlation
    from metrics.xlift_score         import compute_xlift, print_report

    cohort_dir  = RESULTS_DIR / "cohorts"
    scores_dir  = RESULTS_DIR / "xlift_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    cohorts_to_run = [args.cohort] if args.cohort != "all" else ["easy", "frontier", "hard"]

    for name in cohorts_to_run:
        path = cohort_dir / f"{name}.json"
        if not path.exists():
            print(f"Cohort '{name}' not found. Run --step data first.")
            continue

        with open(path) as f:
            cohort = json.load(f)

        print(f"\n{'='*50}")
        print(f"Computing xLift metrics: {name.upper()} cohort ({len(cohort)} tasks)")
        print(f"{'='*50}")

        print("\n[1/4] BoundaryScore...")
        boundary = asyncio.run(compute_cohort_boundary_score(
            cohort, n_rollouts=args.rollouts, max_tasks=args.max_tasks
        ))

        print("\n[2/4] RepairGain...")
        repair = asyncio.run(compute_cohort_repair_gain(
            cohort, n_rollouts=args.rollouts, max_tasks=args.max_tasks // 2
        ))

        print("\n[3/4] GEPA Transfer Lift...")
        gepa = asyncio.run(compute_gepa_transfer_lift(
            cohort, generations=args.gepa_gens, n_rollouts=args.rollouts,
            max_tasks=args.max_tasks
        ))

        print("\n[4/4] AntiCheat Robustness...")
        anticheat = asyncio.run(compute_cohort_anticheat(
            cohort, max_tasks=args.max_tasks // 2
        ))

        print("\n[5/5] Reward-Length Correlation...")
        length_corr = asyncio.run(compute_cohort_length_correlation(
            cohort, n_rollouts=args.rollouts, max_tasks=args.max_tasks
        ))

        # Composite xLift score
        xlift = compute_xlift(boundary, repair, gepa, anticheat, length_corr)

        # Enrich with raw values for visualizer
        xlift["mean_pass_rate"]    = boundary["mean_pass_rate"]
        xlift["learnable_fraction"] = boundary["learnable_fraction"]

        # Save
        out_path = scores_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(xlift, f, indent=2)

        print_report(name, xlift)
        print(f"\nSaved → {out_path}")

        # Save demo example if anticheat found one
        if anticheat.get("demo_example"):
            demo_path = scores_dir / f"{name}_anticheat_demo.json"
            with open(demo_path, "w") as f:
                json.dump(anticheat["demo_example"], f, indent=2)
            print(f"AntiCheat demo example saved → {demo_path}")


def step_train(args):
    from training.grpo_train import train_grpo
    cohort = args.cohort if args.cohort != "all" else "frontier"
    output = str(RESULTS_DIR / "grpo" / cohort)
    print(f"\nTraining on {cohort} cohort → {output}")
    train_grpo(cohort, output, max_steps=args.steps)


def step_visualize(_args):
    from eval.visualize import (
        load_all_results,
        plot_predicted_vs_actual,
        plot_boundary_map,
        plot_pareto_frontier,
        plot_cohort_comparison,
    )
    data = load_all_results()
    print(f"Loaded results for: {list(data.keys())}")
    plot_predicted_vs_actual(data)
    plot_boundary_map(data)
    plot_pareto_frontier(data)
    plot_cohort_comparison(data)
    print(f"\nPlots saved to {RESULTS_DIR / 'plots'}")


def print_summary():
    """Print current state of the experiment."""
    scores_dir = RESULTS_DIR / "xlift_scores"
    grpo_dir   = RESULTS_DIR / "grpo"

    print("\n=== xLift Experiment Status ===")
    for name in ["easy", "frontier", "hard"]:
        xlift_done = (scores_dir / f"{name}.json").exists()
        grpo_done  = (grpo_dir / name / "lift_result.json").exists()

        xlift_score, actual_lift = "—", "—"
        if xlift_done:
            with open(scores_dir / f"{name}.json") as f:
                xlift_score = json.load(f).get("xlift_score", "—")
        if grpo_done:
            with open(grpo_dir / name / "lift_result.json") as f:
                actual_lift = f"{json.load(f).get('actual_lift', 0) * 100:+.1f}%"

        status = "✓" if xlift_done and grpo_done else ("~" if xlift_done else "✗")
        print(f"  {status} {name:10s}  xLift={xlift_score}  actual_lift={actual_lift}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xLift experiment runner")
    parser.add_argument("--step", choices=["data", "metrics", "train", "visualize", "all", "status"],
                        default="status")
    parser.add_argument("--cohort", choices=["easy", "frontier", "hard", "all"], default="all")
    parser.add_argument("--shortcut", action="store_true",
                        help="Use MATH difficulty labels instead of measuring pass rates")
    parser.add_argument("--cohort-size",  type=int, default=150)
    parser.add_argument("--sample-size",  type=int, default=300)
    parser.add_argument("--max-tasks",    type=int, default=40,
                        help="Max tasks per cohort for metric computation")
    parser.add_argument("--rollouts",     type=int, default=5)
    parser.add_argument("--gepa-gens",   type=int, default=3)
    parser.add_argument("--steps",        type=int, default=200,
                        help="GRPO training steps")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny validation run (3 tasks, 2 rollouts, 1 GEPA gen). "
                             "Use first on a new box to confirm the backend works before the full run.")
    args = parser.parse_args()

    if args.smoke:
        args.max_tasks = 3
        args.rollouts = 2
        args.gepa_gens = 1
        if args.cohort == "all":
            args.cohort = "frontier"
        print(f"[smoke] tiny run: cohort={args.cohort}, max_tasks=3, rollouts=2, gepa_gens=1, "
              f"backend={os.environ.get('XLIFT_BACKEND', 'claude')}")

    if args.step == "status":
        print_summary()
    elif args.step == "data":
        step_data(args)
    elif args.step == "metrics":
        step_metrics(args)
    elif args.step == "train":
        step_train(args)
    elif args.step == "visualize":
        step_visualize(args)
    elif args.step == "all":
        print("Running full pipeline (except training)...")
        step_data(args)
        step_metrics(args)
        step_visualize(args)
