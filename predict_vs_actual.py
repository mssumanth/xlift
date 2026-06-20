"""Compare predicted xLift score (GEPA-free, boundary+reachability weighting)
against the ACTUAL GRPO trained lift. This is the headline validation.

Reads:
  results/xlift_scores/<cohort>.json     -> predicted components
  results/grpo/<cohort>/lift_result.json -> actual_lift (ground truth)

    python3 predict_vs_actual.py
"""
import json
import glob
import os

COHORTS = ["easy", "frontier", "hard", "mixed", "weak_verifier"]


def norm_reach(r):
    return min(max(r, 0) * 2, 1.0)


def predicted_score(d):
    """Weighting B: boundary + reachability forward (model-relative signals lead)."""
    b = d.get("boundary_component", 0)
    rep = d.get("repair_component", 0)
    tr = d.get("reward_trust_component", d.get("reward_trust_score", 0))
    rc = norm_reach(d.get("mean_reachability", 0))
    return (0.40 * b + 0.20 * rc + 0.15 * rep + 0.25 * tr) * 100


def main():
    rows = []
    for name in COHORTS:
        xp = f"results/xlift_scores/{name}.json"
        gp = f"results/grpo/{name}/lift_result.json"
        if not os.path.exists(xp):
            continue
        d = json.load(open(xp))
        pred = round(predicted_score(d), 1)

        actual, ci, sig = None, "", ""
        if os.path.exists(gp):
            g = json.load(open(gp))
            actual = g.get("actual_lift")
            lo, hi = g.get("lift_ci_low"), g.get("lift_ci_high")
            if lo is not None:
                ci = f"[{lo*100:+.1f}%,{hi*100:+.1f}%]"
            sig = "SIG" if g.get("lift_significant") else "ns"
        rows.append([name, pred, actual, ci, sig])

    hdr = f'{"cohort":14s} {"predicted":>10} {"actual_lift":>12} {"95% CI":>20} {"":>4}'
    print(hdr)
    print("-" * len(hdr))
    # sort by predicted desc so you can eyeball whether actual follows the same order
    for r in sorted(rows, key=lambda x: -(x[1] or 0)):
        name, pred, actual, ci, sig = r
        a = f"{actual*100:+.1f}%" if actual is not None else "PENDING"
        print(f"{name:14s} {pred:>10} {a:>12} {ci:>20} {sig:>4}")

    done = [r for r in rows if r[2] is not None]
    if len(done) >= 2:
        by_pred = [r[0] for r in sorted(done, key=lambda x: -(x[1] or 0))]
        by_act = [r[0] for r in sorted(done, key=lambda x: -(x[2] or 0))]
        print()
        print(f"predicted order: {' > '.join(by_pred)}")
        print(f"actual order:    {' > '.join(by_act)}")
        print("MATCH!" if by_pred == by_act else "ordering differs — inspect")
    else:
        print("\n(actual lifts not all computed yet — training still pending for some cohorts)")


if __name__ == "__main__":
    main()
