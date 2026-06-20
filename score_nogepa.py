"""Show every cheap xLift signal per cohort, plus the boundary+reachability
composite (weighting B). Reads already-saved results/xlift_scores/*.json — no
metrics re-run needed.

    python3 score_nogepa.py

Columns:
  boundary  = mean_boundary_score 4p(1-p), the headline learnability signal
  reach     = mean_reachability   pass@k - pass@1
  pass      = mean_pass_rate      Qwen raw pass rate on the cohort
  repair    = mean_repair_gain    recovery after a hint
  trust     = reward_trust_score  AntiCheat robustness
  lenCorr   = length_reward_corr  length-hacking check (negative = clean)
  gepa      = gepa_transfer_lift  (currently ~0)
  xLift(B)  = 0.40*boundary + 0.20*reach_norm + 0.15*repair + 0.25*trust  (x100)
"""
import json
import glob

COHORTS = ["easy", "frontier", "hard", "mixed", "weak_verifier"]


def norm_reach(r):  # mean_reachability ~0..0.5 -> 0..1
    return min(max(r, 0) * 2, 1.0)


def main():
    rows = []
    for f in sorted(glob.glob("results/xlift_scores/*.json")):
        name = f.split("/")[-1].replace(".json", "")
        if name not in COHORTS:          # skip *_anticheat_demo and anything else
            continue
        d = json.load(open(f))

        boundary = d.get("mean_boundary_score", 0)
        reach = d.get("mean_reachability", 0)
        passr = d.get("mean_pass_rate", 0)
        repair = d.get("mean_repair_gain", 0)
        trust = d.get("reward_trust_score", 0)
        lencorr = d.get("length_reward_corr", 0)
        gepa = d.get("gepa_transfer_lift", 0)

        # composite B: boundary + reachability forward (uses normalized components)
        b_comp = d.get("boundary_component", boundary)
        rep_comp = d.get("repair_component", min(repair * 2, 1.0))
        tr_comp = d.get("reward_trust_component", trust)
        xliftB = (0.40 * b_comp + 0.20 * norm_reach(reach) + 0.15 * rep_comp + 0.25 * tr_comp) * 100

        rows.append([name, boundary, reach, passr, repair, trust, lencorr, gepa, round(xliftB, 1)])

    cols = ["cohort", "boundary", "reach", "pass", "repair", "trust", "lenCorr", "gepa", "xLift(B)"]
    widths = [14, 9, 7, 7, 7, 7, 8, 7, 9]
    print("".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(cols, widths))))
    print("-" * sum(widths))
    for r in sorted(rows, key=lambda x: -x[8]):   # sort by composite B
        cells = [r[0].ljust(widths[0])]
        for i, v in enumerate(r[1:], start=1):
            cells.append(f"{v:.3f}".rjust(widths[i]) if isinstance(v, float) and i < 8 else str(v).rjust(widths[i]))
        print("".join(cells))


if __name__ == "__main__":
    main()
