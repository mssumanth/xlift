"""
xLift composite score.

xLift(D) = w1 * BoundaryScore
         + w2 * RepairGain
         + w3 * GEPATransferLift
         + w4 * RewardTrust
         - w5 * Cost

Default weights favour the differentiating signals.
"""

WEIGHTS = {
    "boundary_score":    0.25,
    "repair_gain":       0.20,
    "gepa_transfer":     0.35,   # highest weight — our novel signal
    "reward_trust":      0.20,
}


def compute_xlift(
    boundary_result: dict,
    repair_result: dict,
    gepa_result: dict,
    anticheat_result: dict,
    length_corr_result: dict | None = None,
) -> dict:
    """
    Combine all metric results into a single xLift score.
    All component inputs should be normalised 0-1.
    """
    boundary  = boundary_result["mean_boundary_score"]          # already 0-1
    repair    = min(repair_result["mean_repair_gain"] * 2, 1.0) # scale 0-0.5 → 0-1
    gepa      = min(max(gepa_result["gepa_transfer_lift"], 0) * 4, 1.0)  # scale 0-0.25 → 0-1

    # Reward trust: blend anticheat robustness + length independence
    # If length_corr not available, fall back to anticheat alone
    anticheat_trust = anticheat_result["reward_trust_score"]
    if length_corr_result is not None:
        length_independence = length_corr_result["length_independence"]
        trust = 0.6 * anticheat_trust + 0.4 * length_independence
    else:
        trust = anticheat_trust

    w = WEIGHTS
    score = (
        w["boundary_score"] * boundary +
        w["repair_gain"]    * repair   +
        w["gepa_transfer"]  * gepa     +
        w["reward_trust"]   * trust
    ) * 100  # scale to 0-100

    misleading_lift = (
        length_corr_result.get("misleading_lift_risk", False)
        if length_corr_result else False
    )

    return {
        "xlift_score": round(score, 1),

        # Components (normalised 0-1)
        "boundary_component":       round(boundary, 3),
        "repair_component":         round(repair, 3),
        "gepa_transfer_component":  round(gepa, 3),
        "reward_trust_component":   round(trust, 3),

        # Raw values for display
        "mean_boundary_score":      round(boundary_result["mean_boundary_score"], 3),
        "mean_repair_gain":         round(repair_result["mean_repair_gain"], 3),
        "gepa_transfer_lift":       round(gepa_result["gepa_transfer_lift"], 3),
        "reward_trust_score":       round(trust, 3),
        "anticheat_score":          round(anticheat_trust, 3),
        "length_independence":      round(length_corr_result["length_independence"], 3) if length_corr_result else None,
        "length_reward_corr":       round(length_corr_result["global_correlation"], 3) if length_corr_result else None,
        "misleading_lift_risk":     misleading_lift,
        "gepa_gap":                 round(gepa_result.get("gepa_gap", 0), 3),

        # Recommendation
        "recommendation": (
            "fix_verifier"   if trust < 0.5 or misleading_lift else
            "train"          if score >= 65 else
            "consider"       if score >= 45 else
            "diversify"      if gepa_result.get("overfitting_flag") else
            "skip"
        ),
        "recommendation_reason": _recommendation_reason(score, trust, gepa_result, boundary_result, misleading_lift),

        # Pareto position
        "pareto_position": _pareto_position(score, gepa),
    }


def _recommendation_reason(score, trust, gepa_result, boundary_result, misleading_lift=False) -> str:
    if misleading_lift:
        return "High reward-length correlation — measured GRPO lift on this cohort will be inflated. Fix the verifier first."
    if trust < 0.5:
        return "Verifier is hackable — fix the reward function before training."
    if gepa_result.get("overfitting_flag"):
        return "GEPA train lift >> transfer lift — cohort is redundant. Diversify."
    if boundary_result["mean_boundary_score"] < 0.2:
        pr = boundary_result["mean_pass_rate"]
        if pr > 0.8:
            return "Tasks are too easy — model already knows these. Skip."
        return "Tasks are too hard — model cannot learn from them yet. Add easier anchors."
    if score >= 65:
        return "Strong learnable signal with good transfer. High confidence RL will lift."
    if score >= 45:
        return "Moderate signal. Mix with higher-transfer tasks for best results."
    return "Weak overall signal. Not worth the training compute at this stage."


def _pareto_position(score: float, gepa_norm: float) -> str:
    """
    Our claim: GEPA Transfer Lift gives high predictive power at low cost (N rollouts).
    This places xLift above the standard Pareto frontier.
    """
    if gepa_norm > 0.6 and score > 60:
        return "breaks_frontier"
    if gepa_norm > 0.3 or score > 45:
        return "on_frontier"
    return "below_frontier"


def print_report(cohort_name: str, xlift: dict):
    print(f"\n{'='*50}")
    print(f"xLift Report: {cohort_name.upper()} cohort")
    print(f"{'='*50}")
    print(f"  xLift Score:          {xlift['xlift_score']:.1f} / 100")
    print(f"  Recommendation:       {xlift['recommendation'].upper()}")
    print(f"  Reason:               {xlift['recommendation_reason']}")
    print(f"  Pareto position:      {xlift['pareto_position']}")
    print(f"\n  Components:")
    print(f"    BoundaryScore:      {xlift['mean_boundary_score']:.3f}")
    print(f"    RepairGain:         {xlift['mean_repair_gain']:.3f}")
    print(f"    GEPA Transfer Lift: {xlift['gepa_transfer_lift']:+.3f}")
    print(f"    Reward Trust:       {xlift['reward_trust_score']:.3f}")
    print(f"      AntiCheat:        {xlift['anticheat_score']:.3f}")
    if xlift["length_independence"] is not None:
        print(f"      Length Indep:     {xlift['length_independence']:.3f}  (corr={xlift['length_reward_corr']:+.3f})")
    if xlift["misleading_lift_risk"]:
        print(f"  !! MISLEADING LIFT RISK — GRPO lift on this cohort may be inflated")
    if xlift["gepa_gap"] > 0.15:
        print(f"  !! GEPA Gap (overfit): {xlift['gepa_gap']:.3f}")
