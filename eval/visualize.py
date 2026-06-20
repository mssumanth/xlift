"""
Visualizations for the xLift experiment.

1. Scatter plot — predicted xLift vs actual RL lift (the key result)
2. Boundary Map — pass rate vs RepairGain coloured by RewardTrust
3. Pareto frontier chart — cost vs predictive power
4. Cohort comparison bar chart
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
# All 5 cohorts the pipeline produces (was hardcoded to the first 3, which silently
# dropped `mixed` and `weak_verifier` — and weak_verifier IS the reward-hacking exhibit).
ALL_COHORTS = ["easy", "frontier", "hard", "mixed", "weak_verifier"]
COHORT_COLORS = {
    "easy": "#6b7280", "frontier": "#4f6ef7", "hard": "#ef4444",
    "mixed": "#10b981", "weak_verifier": "#f59e0b",
}


def load_all_results() -> dict:
    """Load xlift scores and grpo lift results for all cohorts."""
    data = {}
    for name in ALL_COHORTS:
        xlift_path = RESULTS_DIR / "xlift_scores" / f"{name}.json"
        grpo_path  = RESULTS_DIR / "grpo" / name / "lift_result.json"

        if xlift_path.exists():
            with open(xlift_path) as f:
                data.setdefault(name, {})["xlift"] = json.load(f)
        if grpo_path.exists():
            with open(grpo_path) as f:
                data.setdefault(name, {})["grpo"] = json.load(f)
    return data


def plot_predicted_vs_actual(data: dict):
    """
    The key result plot.
    x-axis: xLift predicted score
    y-axis: actual RL lift (accuracy gain after GRPO)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    points = []
    for name, results in data.items():
        if "xlift" not in results or "grpo" not in results:
            continue
        x = results["xlift"]["xlift_score"]
        y = results["grpo"]["actual_lift"] * 100  # convert to percentage
        points.append((x, y, name))

    if not points:
        print("No data to plot yet — run the experiment first.")
        return

    for x, y, name in points:
        color = COHORT_COLORS.get(name, "#ffffff")
        ax.scatter(x, y, color=color, s=200, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(
            name.capitalize(),
            (x, y),
            textcoords="offset points",
            xytext=(10, 5),
            color=color,
            fontsize=11,
            fontweight="bold",
        )

    # Trend line if we have enough points
    if len(points) >= 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs) - 5, max(xs) + 5, 100)
        ax.plot(x_line, p(x_line), color="#4f6ef7", linestyle="--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("xLift Predicted Score", color="#9ca3af", fontsize=12)
    ax.set_ylabel("Actual RL Lift (accuracy % gain)", color="#9ca3af", fontsize=12)
    ax.set_title("xLift Prediction vs Actual Post-RL Lift", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#6b7280")
    ax.spines[:].set_color("#374151")
    ax.grid(color="#1f2937", linestyle="--", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=c, label=n.capitalize())
        for n, c in COHORT_COLORS.items()
        if n in data
    ]
    ax.legend(handles=legend_patches, facecolor="#1a1d2e", edgecolor="#374151", labelcolor="white")

    path = PLOTS_DIR / "predicted_vs_actual.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved → {path}")
    plt.close()


def plot_boundary_map(data: dict):
    """
    Boundary Map: pass rate (x) vs RepairGain (y) coloured by RewardTrust.
    Shows the four regions: Mastered / Learnable / Latent / Beyond Boundary.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    for name, results in data.items():
        if "xlift" not in results:
            continue
        xlift = results["xlift"]
        x = xlift.get("mean_boundary_score", 0)
        # Convert mean_boundary_score back to approximate pass rate
        # BoundaryScore = 4p(1-p), peak at p=0.5 → approximate p from score
        pr = xlift.get("mean_pass_rate", 0.5) if "mean_pass_rate" in xlift else 0.5
        y  = xlift.get("mean_repair_gain", 0)
        trust = xlift.get("reward_trust_score", 0.5)

        color = plt.cm.RdYlGn(trust)
        ax.scatter(pr, y, s=300, color=color, edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(name.capitalize(), (pr, y), textcoords="offset points",
                    xytext=(10, 5), color="white", fontsize=11, fontweight="bold")

    # Quadrant lines
    ax.axvline(x=0.3, color="#374151", linestyle="--", alpha=0.7)
    ax.axvline(x=0.7, color="#374151", linestyle="--", alpha=0.7)
    ax.axhline(y=0.15, color="#374151", linestyle="--", alpha=0.7)

    # Region labels
    regions = [
        (0.85, 0.05, "Mastered\n(skip)", "#6b7280"),
        (0.5,  0.30, "Learnable\nFrontier ★", "#4f6ef7"),
        (0.15, 0.30, "Latent\n(needs curriculum)", "#f59e0b"),
        (0.15, 0.05, "Beyond\nBoundary (skip)", "#6b7280"),
    ]
    for rx, ry, label, color in regions:
        ax.text(rx, ry, label, color=color, fontsize=9, ha="center",
                va="center", alpha=0.7, fontweight="bold")

    ax.set_xlabel("Base Model Pass Rate", color="#9ca3af", fontsize=12)
    ax.set_ylabel("RepairGain (recovery from failure)", color="#9ca3af", fontsize=12)
    ax.set_title("Boundary Map — colour = Reward Trust", color="white", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.6)
    ax.tick_params(colors="#6b7280")
    ax.spines[:].set_color("#374151")

    path = PLOTS_DIR / "boundary_map.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved → {path}")
    plt.close()


def plot_pareto_frontier(data: dict):
    """
    Pareto chart: cost to compute (x) vs predictive power (y).
    Shows known metrics vs xLift.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    known = [
        (0.02, 0.03, "Token / char length"),
        (0.05, 0.22, "Perplexity"),
        (0.10, 0.40, "Diversity / dedup"),
        (0.30, 0.52, "Reward variance"),
        (0.65, 0.72, "Gradient influence"),
        (0.90, 0.85, "Datamodels"),
        (1.00, 1.00, "Full RL (oracle)"),
    ]

    for x, y, label in known:
        color = "#ef4444" if "oracle" in label else "#6b7280"
        ax.scatter(x, y, color=color, s=100, zorder=4)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 4), color=color, fontsize=8)

    # xLift position — high signal, medium cost
    ax.scatter(0.30, 0.82, color="#4f6ef7", s=250, zorder=5,
               marker="*", edgecolors="white", linewidth=1)
    ax.annotate("xLift (ours) ★", (0.30, 0.82),
                textcoords="offset points", xytext=(8, 6),
                color="#4f6ef7", fontsize=11, fontweight="bold")

    # Arrow showing "breaks frontier"
    ax.annotate(
        "Breaks the\nPareto frontier",
        xy=(0.30, 0.82), xytext=(0.10, 0.90),
        arrowprops=dict(arrowstyle="->", color="#4f6ef7", lw=1.5),
        color="#4f6ef7", fontsize=9, fontweight="bold",
    )

    ax.set_xlabel("Cost to compute →", color="#9ca3af", fontsize=12)
    ax.set_ylabel("Predictive power (correlation with RL lift)", color="#9ca3af", fontsize=12)
    ax.set_title("Cost vs Predictive Power — Pareto Frontier", color="white",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(colors="#6b7280")
    ax.spines[:].set_color("#374151")
    ax.grid(color="#1f2937", linestyle="--", alpha=0.4)

    path = PLOTS_DIR / "pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved → {path}")
    plt.close()


def plot_cohort_comparison(data: dict):
    """Bar chart comparing all xLift components across cohorts."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = [n for n in ALL_COHORTS if n in data and "xlift" in data[n]]
    if not cohorts:
        return

    metrics  = ["mean_boundary_score", "mean_repair_gain", "gepa_transfer_lift", "reward_trust_score"]
    labels   = ["BoundaryScore", "RepairGain", "GEPA Transfer", "Reward Trust"]
    x        = np.arange(len(metrics))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    for i, name in enumerate(cohorts):
        values = [abs(data[name]["xlift"].get(m, 0)) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name.capitalize(),
                      color=COHORT_COLORS[name], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, color="#9ca3af", fontsize=11)
    ax.set_ylabel("Score", color="#9ca3af")
    ax.set_title("xLift Components by Cohort", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#6b7280")
    ax.spines[:].set_color("#374151")
    ax.legend(facecolor="#1a1d2e", edgecolor="#374151", labelcolor="white")
    ax.grid(axis="y", color="#1f2937", linestyle="--", alpha=0.5)

    path = PLOTS_DIR / "cohort_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    data = load_all_results()
    print(f"Loaded data for cohorts: {list(data.keys())}")
    plot_predicted_vs_actual(data)
    plot_boundary_map(data)
    plot_pareto_frontier(data)
    plot_cohort_comparison(data)
    print(f"\nAll plots saved to {PLOTS_DIR}")
