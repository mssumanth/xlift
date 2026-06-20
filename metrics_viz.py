"""Self-contained visualizer: every cheap xLift signal per cohort, plus the
actual GRPO lift, as inline-SVG bar charts in one HTML page. No deps, no server.

    python3 metrics_viz.py            # build results/metrics_dashboard.html
    python3 metrics_viz.py --open     # build and open in browser

Reads results/xlift_scores/<cohort>.json (+ results/grpo/<cohort>/lift_result.json).
Cohorts still training simply show no lift bar.
"""
import json
import os
import argparse
import webbrowser

COHORTS = ["easy", "frontier", "hard", "mixed", "weak_verifier"]
COLOR = {
    "easy": "#6e7681", "frontier": "#58a6ff", "hard": "#bc8cff",
    "mixed": "#3fb950", "weak_verifier": "#f85149",
}
INK = "#e6edf3"; MUTE = "#8b949e"; BG = "#0e1117"; CARD = "#161b22"


def norm_reach(r):
    return min(max(r, 0) * 2, 1.0)


def load():
    data = {}
    for name in COHORTS:
        xp = f"results/xlift_scores/{name}.json"
        if not os.path.exists(xp):
            continue
        d = json.load(open(xp))
        gp = f"results/grpo/{name}/lift_result.json"
        if os.path.exists(gp):
            d.update(json.load(open(gp)))
        b = d.get("boundary_component", d.get("mean_boundary_score", 0))
        rep = d.get("repair_component", min(d.get("mean_repair_gain", 0) * 2, 1.0))
        tr = d.get("reward_trust_component", d.get("reward_trust_score", 0))
        d["xlift_B"] = round((0.40 * b + 0.20 * norm_reach(d.get("mean_reachability", 0))
                              + 0.15 * rep + 0.25 * tr) * 100, 1)
        data[name] = d
    return data


def bar_panel(title, data, getter, vmin, vmax, fmt="{:.2f}", note=""):
    """One panel: a labelled bar per present cohort, scaled vmin..vmax."""
    W, H = 360, 230
    padL, padR, padT, padB = 14, 14, 40, 64
    names = [n for n in COHORTS if n in data]
    n = len(names)
    if n == 0:
        return ""
    plot_w = W - padL - padR
    plot_h = H - padT - padB
    bw = plot_w / n * 0.62
    gap = plot_w / n
    svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{W/2}" y="20" text-anchor="middle" font-size="14" '
               f'font-weight="700" fill="{INK}" font-family="system-ui">{title}</text>')
    # zero/baseline line
    def y_of(v):
        v = max(vmin, min(vmax, v))
        return padT + plot_h - (v - vmin) / (vmax - vmin) * plot_h
    y0 = y_of(0 if vmin < 0 < vmax else vmin)
    svg.append(f'<line x1="{padL}" y1="{y0:.1f}" x2="{W-padR}" y2="{y0:.1f}" '
               f'stroke="{MUTE}" stroke-width="1" stroke-dasharray="3 3" opacity="0.5"/>')
    for i, name in enumerate(names):
        v = getter(data[name])
        if v is None:
            continue
        cx = padL + gap * i + (gap - bw) / 2
        yv = y_of(v)
        top, bot = min(yv, y0), max(yv, y0)
        svg.append(f'<rect x="{cx:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                   f'height="{bot-top:.1f}" rx="3" fill="{COLOR[name]}"/>')
        svg.append(f'<text x="{cx+bw/2:.1f}" y="{top-4:.1f}" text-anchor="middle" '
                   f'font-size="11" fill="{INK}" font-family="system-ui">{fmt.format(v)}</text>')
        label = name.replace("weak_verifier", "weak-v").replace("frontier", "front")
        svg.append(f'<text x="{cx+bw/2:.1f}" y="{H-padB+16:.1f}" text-anchor="middle" '
                   f'font-size="10" fill="{MUTE}" font-family="system-ui" '
                   f'transform="rotate(35 {cx+bw/2:.1f} {H-padB+16:.1f})">{label}</text>')
    if note:
        svg.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" font-size="9.5" '
                   f'fill="{MUTE}" font-family="system-ui">{note}</text>')
    svg.append('</svg>')
    return f'<div class="panel">{"".join(svg)}</div>'


def build(data):
    panels = []
    panels.append(bar_panel("BoundaryScore  4p(1-p)", data,
                            lambda d: d.get("mean_boundary_score", 0), 0, 1,
                            note="learnability — should peak at frontier"))
    panels.append(bar_panel("Reachability  pass@k - pass@1", data,
                            lambda d: d.get("mean_reachability", 0), 0, 0.6,
                            note="reachable vs hopeless"))
    panels.append(bar_panel("Pass rate (Qwen)", data,
                            lambda d: d.get("mean_pass_rate", 0), 0, 1))
    panels.append(bar_panel("RepairGain", data,
                            lambda d: d.get("mean_repair_gain", 0), 0, 0.6,
                            note="recovery after hint (artifact-prone on hard)"))
    panels.append(bar_panel("Reward Trust (AntiCheat)", data,
                            lambda d: d.get("reward_trust_score", 0), 0, 1,
                            note="higher = harder to fool"))
    panels.append(bar_panel("Length-reward corr", data,
                            lambda d: d.get("length_reward_corr", 0), -0.8, 0.4,
                            note="negative = clean, no length hacking"))
    panels.append(bar_panel("xLift score (weighting B)", data,
                            lambda d: d.get("xlift_B", 0), 0, 70, fmt="{:.1f}",
                            note="boundary+reachability forward"))
    panels.append(bar_panel("ACTUAL GRPO lift  (ground truth)", data,
                            lambda d: d.get("actual_lift"), -0.05, 0.12,
                            fmt="{:+.1%}", note="post-train minus baseline accuracy"))

    legend = " ".join(
        f'<span class="lg"><i style="background:{COLOR[c]}"></i>{c}</span>' for c in COHORTS
    )
    html = f"""<!doctype html><meta charset="utf-8">
<style>
 body{{background:{BG};color:{INK};font-family:system-ui,sans-serif;margin:0;padding:24px}}
 h1{{font-size:22px;margin:0 0 4px}} .sub{{color:{MUTE};margin:0 0 16px;font-size:13px}}
 .legend{{margin:8px 0 20px;font-size:12px;color:{MUTE}}}
 .lg{{margin-right:14px}} .lg i{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:middle}}
 .grid{{display:flex;flex-wrap:wrap;gap:16px}}
 .panel{{background:{CARD};border:1px solid #21262d;border-radius:10px;padding:8px;width:360px}}
</style>
<h1>xLift — cheap signals vs actual RL lift</h1>
<p class="sub">Each panel: one signal across cohorts. Bottom-right panel is the ground-truth GRPO lift.
Cohorts still training show no lift bar.</p>
<div class="legend">{legend}</div>
<div class="grid">{"".join(panels)}</div>
"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    data = load()
    if not data:
        print("No results/xlift_scores/*.json found — run metrics first.")
        return
    out = "results/metrics_dashboard.html"
    with open(out, "w") as f:
        f.write(build(data))
    print(f"Wrote {out}  ({len(data)} cohorts, "
          f"{sum('actual_lift' in d for d in data.values())} with RL lift)")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
