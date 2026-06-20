from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── helpers ─────────────────────────────────────────────────────────────────

def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    r, _ = stats.spearmanr(x, y)
    return float(r)


def _load_metrics(artifacts: Path) -> dict[str, dict]:
    """Load all cohort metrics from disk."""
    out = {}
    for p in (artifacts / "cohorts").glob("*.metrics.json"):
        with open(p) as f:
            out[p.stem.replace(".metrics", "")] = json.load(f)
    return out


def _load_eval(artifacts: Path) -> dict[str, dict]:
    out = {}
    for p in (artifacts / "eval").glob("*.json"):
        if p.stem == "base":
            continue
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def _load_gradnorm(artifacts: Path) -> dict[str, float]:
    out = {}
    for p in (artifacts / "gradnorm").glob("*.json"):
        with open(p) as f:
            out[p.stem] = json.load(f).get("grad_norm", None)
    return out


# ── table assembly ───────────────────────────────────────────────────────────

def assemble_table(cfg) -> pd.DataFrame:
    """Join metrics + eval + gradnorm → results/cohort_table.csv."""
    artifacts = Path(cfg.artifacts_dir)
    results = Path("results")
    results.mkdir(exist_ok=True)

    metrics_map = _load_metrics(artifacts)
    eval_map = _load_eval(artifacts)
    gradnorm_map = _load_gradnorm(artifacts)

    rows = []
    for name, m in metrics_map.items():
        e = eval_map.get(name, {})   # empty dict if GRPO not yet run — lift cols will be None
        gn = gradnorm_map.get(name)
        row = {
            "name": name,
            # metrics
            "frontier_score": m.get("frontier_score"),
            "effective_ratio": m.get("effective_ratio"),
            "band_fraction": m.get("band_fraction"),
            "reward_variance": m.get("reward_variance"),
            "pass_at_k_minus_1": m.get("pass_at_k_minus_1"),
            "reward_length_corr": m.get("reward_length_corr"),
            "answer_entropy": m.get("answer_entropy"),
            "mean_token_length": m.get("mean_token_length"),
            "avg_pass_rate": m.get("avg_pass_rate"),
            "vendi_score": m.get("vendi_score"),
            "redundancy": m.get("redundancy"),
            "dist_match": m.get("dist_match"),
            "grad_norm": gn,
            # eval
            "acc_before": e.get("acc_before"),
            "acc_after_best": e.get("acc_after_best"),
            "lift": e.get("lift"),
            "lift_ci_low": e.get("lift_ci_low"),
            "lift_ci_high": e.get("lift_ci_high"),
            "best_step": e.get("best_step"),
            "train_reward_final": e.get("train_reward_final"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = results / "cohort_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}")
    return df


# ── validation ───────────────────────────────────────────────────────────────

_CHEAP_SIGNALS = [
    "frontier_score", "effective_ratio", "band_fraction", "reward_variance",
    "pass_at_k_minus_1", "reward_length_corr", "answer_entropy",
    "mean_token_length", "avg_pass_rate", "vendi_score", "redundancy", "dist_match",
]
_REGRESSION_COHORTS = {"C1_easy", "C2_frontier", "C3_hard", "C4_mixed", "C5_redundant", "C7_synth"}


def validate_signals(table: pd.DataFrame) -> dict:
    """Spearman + LOO + partial correlation over regression cohorts (exclude C6)."""
    reg = table[table["name"].isin(_REGRESSION_COHORTS)].dropna(subset=["lift"])
    n = len(reg)

    results = {}
    for sig in _CHEAP_SIGNALS:
        col = reg[sig].dropna()
        if len(col) < 3:
            results[sig] = {"spearman": float("nan"), "loo_mean": float("nan"), "loo_std": float("nan")}
            continue
        valid = reg.dropna(subset=[sig, "lift"])
        rho = _spearman(valid[sig].values, valid["lift"].values)
        # LOO
        loo_rhos = []
        for i in valid.index:
            sub = valid.drop(i)
            if len(sub) >= 2:
                loo_rhos.append(_spearman(sub[sig].values, sub["lift"].values))
        results[sig] = {
            "spearman": round(rho, 3),
            "loo_mean": round(float(np.nanmean(loo_rhos)), 3) if loo_rhos else float("nan"),
            "loo_std": round(float(np.nanstd(loo_rhos)), 3) if loo_rhos else float("nan"),
            "n": len(valid),
        }

    # Combo: zscore(frontier) + zscore(pass@k-1)
    combo_valid = reg.dropna(subset=["frontier_score", "pass_at_k_minus_1", "lift"])
    if len(combo_valid) >= 3:
        combo = (
            stats.zscore(combo_valid["frontier_score"].values)
            + stats.zscore(combo_valid["pass_at_k_minus_1"].values)
        )
        results["combo_frontier_pak1"] = {
            "spearman": round(_spearman(combo, combo_valid["lift"].values), 3),
            "n": len(combo_valid),
        }

    # Partial Spearman: frontier vs lift controlling for dist_match
    partial_valid = reg.dropna(subset=["frontier_score", "dist_match", "lift"])
    if len(partial_valid) >= 4:
        # Residualize frontier_score on dist_match, then correlate residuals with lift
        _, _, r_f_d, _, _ = stats.linregress(partial_valid["dist_match"], partial_valid["frontier_score"])
        _, _, r_l_d, _, _ = stats.linregress(partial_valid["dist_match"], partial_valid["lift"])
        resid_f = partial_valid["frontier_score"].values - (
            np.polyval(np.polyfit(partial_valid["dist_match"], partial_valid["frontier_score"], 1),
                       partial_valid["dist_match"].values)
        )
        resid_l = partial_valid["lift"].values - (
            np.polyval(np.polyfit(partial_valid["dist_match"], partial_valid["lift"], 1),
                       partial_valid["dist_match"].values)
        )
        results["frontier_partial_dist_match"] = {
            "spearman": round(_spearman(resid_f, resid_l), 3),
            "n": len(partial_valid),
        }

    validation = {"n_regression_cohorts": n, "signals": results}
    out_path = Path("results") / "validation.json"
    with open(out_path, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation written → {out_path}")
    return validation


# ── plots ────────────────────────────────────────────────────────────────────

_BG = "#0f1117"
_AX = "#1a1d2e"
_COHORT_COLORS = {
    "C1_easy": "#4ade80",
    "C2_frontier": "#60a5fa",
    "C3_hard": "#f87171",
    "C4_mixed": "#facc15",
    "C5_redundant": "#c084fc",
    "C6_weak": "#fb923c",
    "C7_synth": "#34d399",
}


def _setup_ax(ax, title):
    import matplotlib.pyplot as plt
    ax.set_facecolor(_AX)
    ax.spines[:].set_color("#3f4460")
    ax.tick_params(colors="#aab4d4")
    ax.xaxis.label.set_color("#aab4d4")
    ax.yaxis.label.set_color("#aab4d4")
    ax.set_title(title, color="#e2e8f0", fontsize=13, pad=10)


def plot_inverted_u(cfg) -> None:
    """Headline plot: inverted-U of RL lift vs pass rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = Path(cfg.artifacts_dir)
    results = Path("results")
    results.mkdir(exist_ok=True)

    # Per-task scatter: p_strong from index
    index_csv = artifacts / "index" / "pass_rate_index.csv"
    eval_map = _load_eval(artifacts)
    metrics_map = _load_metrics(artifacts)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_BG)
    _setup_ax(ax, "Learnable Frontier: Pass Rate vs RL Lift")

    # Plot cohort means as large points
    for name, e in eval_map.items():
        m = metrics_map.get(name, {})
        if e.get("lift") is None or m.get("avg_pass_rate") is None:
            continue
        color = _COHORT_COLORS.get(name, "#94a3b8")
        ax.scatter(m["avg_pass_rate"], e["lift"], s=200, color=color,
                   zorder=5, label=name, edgecolors="white", linewidths=0.8)
        ax.annotate(name.replace("_", " "), (m["avg_pass_rate"], e["lift"]),
                    textcoords="offset points", xytext=(6, 4), color=color, fontsize=8)

    # Theoretical inverted-U overlay
    ps = np.linspace(0, 1, 200)
    ax.plot(ps, ps * (1 - ps) * 4 * max(
        (e.get("lift", 0) for e in eval_map.values()), default=0.05
    ), color="#94a3b8", linewidth=1.2, linestyle="--", alpha=0.5, label="4p(1-p) shape")

    ax.set_xlabel("Mean Pass Rate (p)", fontsize=11)
    ax.set_ylabel("Actual RL Lift (Δacc)", fontsize=11)
    ax.axhline(0, color="#3f4460", linewidth=0.8)
    ax.axvspan(0.4, 0.6, alpha=0.1, color="#60a5fa", label="Frontier band [0.4,0.6]")
    ax.legend(fontsize=8, facecolor=_AX, edgecolor="#3f4460", labelcolor="#e2e8f0")
    fig.tight_layout()
    out = results / "inverted_u.png"
    fig.savefig(out, dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"Saved {out}")


def plot_scatter(table: pd.DataFrame) -> None:
    """frontier_score vs actual lift, with LOO rho annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path("results")
    results.mkdir(exist_ok=True)
    valid = table.dropna(subset=["frontier_score", "lift"])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_BG)
    _setup_ax(ax, "Predicted (Frontier Score) vs Actual RL Lift")

    reg = valid[valid["name"].isin(_REGRESSION_COHORTS)]
    c6 = valid[valid["name"] == "C6_weak"]

    for _, row in reg.iterrows():
        color = _COHORT_COLORS.get(row["name"], "#94a3b8")
        ax.scatter(row["frontier_score"], row["lift"], s=180, color=color,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(row["name"].replace("_", " "),
                    (row["frontier_score"], row["lift"]),
                    textcoords="offset points", xytext=(5, 3), color=color, fontsize=8)

    for _, row in c6.iterrows():
        ax.scatter(row["frontier_score"], row["lift"], s=180,
                   color=_COHORT_COLORS["C6_weak"], marker="X",
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate("C6 weak ⚠", (row["frontier_score"], row["lift"]),
                    textcoords="offset points", xytext=(5, -12),
                    color=_COHORT_COLORS["C6_weak"], fontsize=8)

    # Trend line for regression cohorts
    if len(reg) >= 2:
        m, b = np.polyfit(reg["frontier_score"], reg["lift"], 1)
        xs = np.linspace(reg["frontier_score"].min(), reg["frontier_score"].max(), 100)
        ax.plot(xs, m * xs + b, "--", color="#94a3b8", alpha=0.6, linewidth=1.2)
        rho = _spearman(reg["frontier_score"].values, reg["lift"].values)
        ax.text(0.05, 0.95, f"Spearman ρ = {rho:.2f} (N={len(reg)})",
                transform=ax.transAxes, color="#e2e8f0", fontsize=10,
                verticalalignment="top")

    ax.set_xlabel("Frontier Score (pre-training)", fontsize=11)
    ax.set_ylabel("Actual RL Lift (Δacc)", fontsize=11)
    ax.axhline(0, color="#3f4460", linewidth=0.8)
    fig.tight_layout()
    out = results / "scatter.png"
    fig.savefig(out, dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"Saved {out}")


def plot_pareto(table: pd.DataFrame) -> None:
    """Cost vs predictive power — xLift should dominate baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path("results")
    results.mkdir(exist_ok=True)
    reg = table[table["name"].isin(_REGRESSION_COHORTS)].dropna(subset=["lift"])

    # Approximate GPU-second costs (log scale)
    cost_map = {
        "mean_token_length": 0.01,
        "avg_pass_rate": 1.0,
        "reward_variance": 1.0,
        "frontier_score": 1.0,
        "pass_at_k_minus_1": 1.0,
        "combo_frontier_pak1": 1.5,
        "grad_norm": 120.0,
        "full_GRPO": 3600.0,
    }

    valid_sigs = []
    for sig, cost in cost_map.items():
        if sig in ("combo_frontier_pak1", "full_GRPO", "grad_norm"):
            continue
        col = reg[sig] if sig in reg.columns else None
        if col is None or col.dropna().shape[0] < 3:
            continue
        rho = abs(_spearman(col.dropna(), reg.loc[col.dropna().index, "lift"].values))
        valid_sigs.append((sig, cost, rho))

    # Combo
    combo_valid = reg.dropna(subset=["frontier_score", "pass_at_k_minus_1", "lift"])
    if len(combo_valid) >= 3:
        combo = (
            stats.zscore(combo_valid["frontier_score"].values)
            + stats.zscore(combo_valid["pass_at_k_minus_1"].values)
        )
        rho = abs(_spearman(combo, combo_valid["lift"].values))
        valid_sigs.append(("frontier + pass@k-1", cost_map["combo_frontier_pak1"], rho))

    # grad_norm
    gn_col = reg["grad_norm"].dropna() if "grad_norm" in reg.columns else pd.Series()
    if gn_col.shape[0] >= 3:
        rho = abs(_spearman(gn_col.values, reg.loc[gn_col.index, "lift"].values))
        valid_sigs.append(("grad_norm", cost_map["grad_norm"], rho))

    # Full GRPO point (oracle)
    valid_sigs.append(("full GRPO (oracle)", cost_map["full_GRPO"], 1.0))

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=_BG)
    _setup_ax(ax, "Pareto: Compute Cost vs Predictive Power (|Spearman ρ|)")

    for sig, cost, rho in valid_sigs:
        is_xlift = "frontier" in sig and "pass" in sig
        color = "#60a5fa" if is_xlift else "#94a3b8" if sig != "full GRPO (oracle)" else "#f87171"
        size = 200 if is_xlift else 100
        marker = "*" if is_xlift else "o"
        ax.scatter(cost, rho, s=size, color=color, marker=marker,
                   zorder=5, edgecolors="white" if is_xlift else "none", linewidths=0.8)
        ax.annotate(sig, (cost, rho), textcoords="offset points",
                    xytext=(6, 3), color=color, fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Approx. Compute Cost (GPU-seconds, log scale)", fontsize=11)
    ax.set_ylabel("|Spearman ρ| with actual lift", fontsize=11)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = results / "pareto.png"
    fig.savefig(out, dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"Saved {out}")


def plot_reward_hack(cfg) -> None:
    """C6 vs C2: train_reward curves + eval lift bars + reward_length_corr bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = Path(cfg.artifacts_dir)
    results = Path("results")
    results.mkdir(exist_ok=True)

    def _load_train_log(name):
        p = artifacts / "train" / name / "train_log.jsonl"
        if not p.exists():
            return [], []
        rows = [json.loads(l) for l in open(p) if l.strip()]
        steps = [r["step"] for r in rows]
        rewards = [r["mean_reward"] for r in rows]
        return steps, rewards

    eval_map = _load_eval(artifacts)
    metrics_map = _load_metrics(artifacts)

    c2_steps, c2_rewards = _load_train_log("C2_frontier")
    c6_steps, c6_rewards = _load_train_log("C6_weak")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_BG)
    for ax in axes:
        _setup_ax(ax, "")

    # Left: train reward curves
    ax = axes[0]
    if c2_steps:
        ax.plot(c2_steps, c2_rewards, color=_COHORT_COLORS["C2_frontier"],
                label="C2 (strong verifier)", linewidth=2)
    if c6_steps:
        ax.plot(c6_steps, c6_rewards, color=_COHORT_COLORS["C6_weak"],
                label="C6 (weak verifier)", linewidth=2, linestyle="--")
    ax.set_title("Train Reward over Steps", color="#e2e8f0", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=8, facecolor=_AX, edgecolor="#3f4460", labelcolor="#e2e8f0")

    # Center: eval lift bars
    ax = axes[1]
    names = ["C2_frontier", "C6_weak"]
    lifts = [eval_map.get(n, {}).get("lift", 0) or 0 for n in names]
    colors = [_COHORT_COLORS[n] for n in names]
    ax.bar(["C2\n(strong)", "C6\n(weak)"], lifts, color=colors, width=0.5, edgecolor="white")
    ax.axhline(0, color="#3f4460", linewidth=0.8)
    ax.set_title("True Eval Lift (Δacc)", color="#e2e8f0", fontsize=11)
    ax.set_ylabel("Lift")

    # Right: reward_length_corr bars
    ax = axes[2]
    rlcs = [metrics_map.get(n, {}).get("reward_length_corr", 0) or 0 for n in names]
    ax.bar(["C2\n(strong)", "C6\n(weak)"], rlcs, color=colors, width=0.5, edgecolor="white")
    ax.axhline(0, color="#3f4460", linewidth=0.8)
    ax.set_title("Reward-Length Corr (r_pb)", color="#e2e8f0", fontsize=11)
    ax.set_ylabel("r_pb")

    fig.suptitle("Reward Hacking Demo: C6 vs C2", color="#e2e8f0", fontsize=13, y=1.02)
    fig.tight_layout()
    out = results / "reward_hack.png"
    fig.savefig(out, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def run_analysis(cfg) -> None:
    """Full analysis pipeline: table → validation → all plots."""
    table = assemble_table(cfg)
    validate_signals(table)
    plot_inverted_u(cfg)
    plot_scatter(table)
    plot_pareto(table)
    plot_reward_hack(cfg)
    print("\nAnalysis complete. See results/")
