"""Univariate ablation: for each cheap signal, how well does its cohort ordering
match the ACTUAL GRPO lift ordering? Reports Spearman rank correlation per signal
(the honest, overfitting-proof claim for small N) and builds an HTML graph.

    python3 ablation.py            # print table + write results/ablation.html
    python3 ablation.py --open     # also open the graph

Only cohorts that have BOTH cheap metrics and a trained lift_result.json are used.
With <4 such cohorts the rank correlation is not meaningful (it warns).
"""
import json
import os
import argparse
import webbrowser

COHORTS = ["easy", "frontier", "hard", "mixed", "weak_verifier"]
COLOR = {"easy": "#6e7681", "frontier": "#58a6ff", "hard": "#bc8cff",
         "mixed": "#3fb950", "weak_verifier": "#f85149"}
INK = "#e6edf3"; MUTE = "#8b949e"; BG = "#0e1117"; CARD = "#161b22"
GOOD = "#3fb950"; BAD = "#f85149"


def norm_reach(r):
    return min(max(r, 0) * 2, 1.0)


# signal name -> (extractor, "higher predicts more lift?" direction)
def signals():
    return {
        "BoundaryScore": (lambda d: d.get("mean_boundary_score", 0), +1),
        "Reachability":  (lambda d: d.get("mean_reachability", 0), +1),
        "RepairGain":    (lambda d: d.get("mean_repair_gain", 0), +1),
        "Reward Trust":  (lambda d: d.get("reward_trust_score", 0), +1),
        "Pass rate":     (lambda d: d.get("mean_pass_rate", 0), -1),   # high pass = less headroom
        "xLift (B)":     (lambda d: _xliftB(d), +1),
        "Bndry+Reach":   (lambda d: 0.6 * d.get("mean_boundary_score", 0)
                                    + 0.4 * norm_reach(d.get("mean_reachability", 0)), +1),
    }


def _xliftB(d):
    b = d.get("boundary_component", d.get("mean_boundary_score", 0))
    rep = d.get("repair_component", min(d.get("mean_repair_gain", 0) * 2, 1.0))
    tr = d.get("reward_trust_component", d.get("reward_trust_score", 0))
    return 0.40 * b + 0.20 * norm_reach(d.get("mean_reachability", 0)) + 0.15 * rep + 0.25 * tr


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def load():
    rows = []
    for name in COHORTS:
        xp = f"results/xlift_scores/{name}.json"
        gp = f"results/grpo/{name}/lift_result.json"
        if not (os.path.exists(xp) and os.path.exists(gp)):
            continue
        d = json.load(open(xp))
        d.update(json.load(open(gp)))
        rows.append((name, d))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    rows = load()
    n = len(rows)
    names = [r[0] for r in rows]
    actual = [r[1].get("actual_lift", 0) for r in rows]

    print(f"\nCohorts with both metrics + trained lift: {n}  ({', '.join(names) or 'none'})")
    if n < 2:
        print("Need >=2 trained cohorts to compare orderings. Train more first.")
        return
    if n < 4:
        print("WARNING: <4 cohorts — rank correlation is NOT statistically meaningful yet.\n")

    actual_order = [names[i] for i in sorted(range(n), key=lambda i: -actual[i])]
    print(f"ACTUAL lift order:  {' > '.join(actual_order)}")
    print(f"   actual lifts:    " + "  ".join(f"{names[i]}={actual[i]*100:+.1f}%" for i in range(n)))
    print()

    results = []
    for sig, (fn, direction) in signals().items():
        vals = [fn(r[1]) for r in rows]
        rho = spearman([direction * v for v in vals], actual)
        pred_order = [names[i] for i in sorted(range(n), key=lambda i: -direction * vals[i])]
        match = "MATCH" if pred_order == actual_order else ""
        results.append((sig, rho, pred_order, match))

    results.sort(key=lambda x: -x[1])
    print(f'{"signal":14s} {"Spearman":>9}  predicted order')
    print("-" * 64)
    for sig, rho, order, match in results:
        print(f"{sig:14s} {rho:>+9.2f}  {' > '.join(order)}  {match}")

    write_html(rows, names, actual, actual_order, results, n)
    print(f"\nWrote results/ablation.html")
    if args.open:
        webbrowser.open("file://" + os.path.abspath("results/ablation.html"))


def write_html(rows, names, actual, actual_order, results, n):
    # bar chart of Spearman rho per signal
    W, H = 560, 300
    padL, padT, padB = 130, 30, 20
    bars = []
    bh = (H - padT - padB) / len(results)
    for i, (sig, rho, order, match) in enumerate(results):
        y = padT + i * bh + bh * 0.15
        w = abs(rho) * (W - padL - 40) / 1.0
        x0 = padL
        x = x0 if rho >= 0 else x0 - w
        col = GOOD if rho > 0 else BAD
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bh*0.7:.1f}" rx="3" fill="{col}"/>')
        bars.append(f'<text x="{padL-8}" y="{y+bh*0.45:.1f}" text-anchor="end" font-size="12" '
                    f'fill="{INK}" font-family="system-ui">{sig}</text>')
        bars.append(f'<text x="{x0 + (w if rho>=0 else -w) + (6 if rho>=0 else -6):.1f}" '
                    f'y="{y+bh*0.45:.1f}" text-anchor="{"start" if rho>=0 else "end"}" '
                    f'font-size="11" fill="{MUTE}" font-family="system-ui">{rho:+.2f}</text>')
    axis = f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" stroke="{MUTE}" stroke-width="1"/>'
    bar_svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{axis}{"".join(bars)}</svg>'

    # scatter: best signal vs actual lift
    best = results[0]
    fn, direction = signals()[best[0]]
    xs = [direction * fn(r[1]) for r in rows]
    ys = actual
    SW, SH = 420, 300
    mpadL, mpadB, mpadT, mpadR = 50, 40, 20, 20
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1; yr = (ymax - ymin) or 1
    def sx(v): return mpadL + (v - xmin) / xr * (SW - mpadL - mpadR)
    def sy(v): return SH - mpadB - (v - ymin) / yr * (SH - mpadT - mpadB)
    pts = []
    for i, name in enumerate(names):
        pts.append(f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(ys[i]):.1f}" r="7" fill="{COLOR[name]}"/>')
        pts.append(f'<text x="{sx(xs[i]):.1f}" y="{sy(ys[i])-11:.1f}" text-anchor="middle" '
                   f'font-size="10" fill="{INK}" font-family="system-ui">{name}</text>')
    sc_axis = (f'<line x1="{mpadL}" y1="{SH-mpadB}" x2="{SW-mpadR}" y2="{SH-mpadB}" stroke="{MUTE}"/>'
               f'<line x1="{mpadL}" y1="{mpadT}" x2="{mpadL}" y2="{SH-mpadB}" stroke="{MUTE}"/>'
               f'<text x="{SW/2}" y="{SH-6}" text-anchor="middle" font-size="11" fill="{MUTE}" '
               f'font-family="system-ui">{best[0]} (predicted)  ρ={best[1]:+.2f}</text>'
               f'<text x="14" y="{SH/2}" text-anchor="middle" font-size="11" fill="{MUTE}" '
               f'font-family="system-ui" transform="rotate(-90 14 {SH/2})">actual lift</text>')
    sc_svg = f'<svg viewBox="0 0 {SW} {SH}" xmlns="http://www.w3.org/2000/svg">{sc_axis}{"".join(pts)}</svg>'

    warn = ("" if n >= 4 else
            f'<p style="color:{BAD};font-size:13px">⚠ Only {n} cohorts trained — rank correlation '
            f'is not statistically meaningful yet. Train all 5 (and ideally more steps) before trusting this.</p>')
    html = f"""<!doctype html><meta charset="utf-8">
<style>body{{background:{BG};color:{INK};font-family:system-ui;margin:0;padding:24px}}
 h1{{font-size:21px;font-weight:500;margin:0 0 4px}} .sub{{color:{MUTE};font-size:13px;margin:0 0 14px}}
 .row{{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start}}
 .card{{background:{CARD};border:1px solid #21262d;border-radius:10px;padding:10px}}</style>
<h1>Univariate ablation — which cheap signal predicts the lift ordering?</h1>
<p class="sub">Spearman rank correlation between each signal's cohort ordering and the actual GRPO lift ordering.
Actual order: {' &gt; '.join(actual_order)}</p>
{warn}
<div class="row">
  <div class="card">{bar_svg}</div>
  <div class="card">{sc_svg}</div>
</div>
"""
    os.makedirs("results", exist_ok=True)
    with open("results/ablation.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
