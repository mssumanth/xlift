"""Recompute xLift scores under GEPA-free weightings, from already-saved
component values in results/xlift_scores/*.json. No metrics re-run needed.

    python3 score_nogepa.py
"""
import json
import glob


def norm_reach(r):  # mean_reachability ~0..0.5 -> 0..1
    return min(max(r, 0) * 2, 1.0)


def main():
    rows = []
    for f in sorted(glob.glob("results/xlift_scores/*.json")):
        d = json.load(open(f))
        name = f.split("/")[-1].replace(".json", "")
        b = d.get("boundary_component", 0)        # 0..1
        rep = d.get("repair_component", 0)        # 0..1
        tr = d.get("reward_trust_component", d.get("reward_trust_score", 0))  # 0..1
        rc = norm_reach(d.get("mean_reachability", 0))
        orig = d.get("xlift_score")

        # (A) drop GEPA, renormalize the original 3 components to sum=1
        A = (0.25 * b + 0.20 * rep + 0.20 * tr) / 0.65 * 100

        # (B) boundary + reachability forward (model-relative signals lead)
        B = (0.40 * b + 0.20 * rc + 0.15 * rep + 0.25 * tr) * 100

        rows.append((name, orig, round(A, 1), round(B, 1), round(b, 3), round(rc, 3)))

    hdr = f'{"cohort":14s} {"orig":>6} {"noGEPA(A)":>10} {"boundary(B)":>12} {"bndry":>7} {"reach":>7}'
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -(x[3] or 0)):
        print(f"{r[0]:14s} {str(r[1]):>6} {r[2]:>10} {r[3]:>12} {r[4]:>7} {r[5]:>7}")


if __name__ == "__main__":
    main()
