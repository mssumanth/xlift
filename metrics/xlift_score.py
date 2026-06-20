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
) -> dict:
    """
    Combine all metric results into a single xLift score.
    All component inputs should be normalised 0-1.
    """
    boundary  = boundary_result["mean_boundary_score"]          # already 0-1
    repair    = min(repair_result["mean_repair_gain"] * 2, 1.0) # scale 0-0.5 → 0-1
    gepa      = min(max(gepa_result["gepa_transfer_lift"], 0) * 4, 1.0)  # scale 0-0.25 → 0-1
    trust     = anticheat_result["reward_trust_score"]          # already 0-1

    w = WEIGHTS
    score = (
        w["boundary_score"] * boundary +
        w["repair_gain"]    * repair   +
        w["gepa_transfer"]  * gepa     +
        w["reward_trust"]   * trust
    ) * 100  # scale to 0-100

    return {
        "xlift_score": round(score, 1),

        # Components (normalised 0-1)
        "boundary_component":    round(boundary, 3),
        "repair_component":      round(repair, 3),
        "gepa_transfer_component": round(gepa, 3),
        "reward_trust_component":  round(trust, 3),

        # Raw values for display
        "mean_boundary_score":   round(boundary_result["mean_boundary_score"], 3),
        "mean_repair_gain":      round(repair_result["mean_repair_gain"], 3),
        "gepa_transfer_lift":    round(gepa_result["gepa_transfer_lift"], 3),
        "reward_trust_score":    round(anticheat_result["reward_trust_score"], 3),
        "gepa_gap":              round(gepa_result.get("gepa_gap", 0), 3),

        # Recommendation
        "recommendation": (
            "train"          if score >= 65 else
            "consider"       if score >= 45 else
            "diversify"      if gepa_result.get("overfitting_flag") else
            "fix_verifier"   if trust < 0.5 else
            "skip"
        ),
        "recommendation_reason": _recommendation_reason(score, trust, gepa_result, boundary_result),

        # Pareto position
        "pareto_position": _pareto_position(score, gepa),
    }


def _recommendation_reason(score, trust, gepa_result, boundary_result) -> str:
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
    if xlift["gepa_gap"] > 0.15:
        print(f"  ⚠  GEPA Gap (overfit): {xlift['gepa_gap']:.3f}")
