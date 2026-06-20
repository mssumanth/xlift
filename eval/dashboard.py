"""Self-contained HTML results dashboard for xLift.

Reads results/xlift_scores/*.json + results/grpo/*/lift_result.json and emits a
single polished results/dashboard.html with inline SVG charts. No JS, no CDN, no
server — it opens in any browser, works fully offline, and can't crash on stage.

    python3 eval/dashboard.py            # build from real results
    python3 eval/dashboard.py --demo     # build from synthetic data (rehearsal)
    python3 eval/dashboard.py --open     # build then open in the browser
"""
from __future__ import annotations

import json
import argparse
import webbrowser
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
COHORTS = ["easy", "frontier", "hard"]

# palette
BG = "#0e1117"; CARD = "#161b22"; INK = "#e6edf3"; MUTE = "#8b949e"
ACCENT = "#58a6ff"; GOOD = "#3fb950"; WARN = "#d29922"; BAD = "#f85149"
COHORT_COLOR = {"easy": "#6e7681", "frontier": "#58a6ff", "hard": "#bc8cff"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_results(results_dir: Path) -> dict:
    data = {}
    for name in COHORTS:
        xp = results_dir / "xlift_scores" / f"{name}.json"
        gp = results_dir / "grpo" / name / "lift_result.json"
        if not xp.exists():
            continue
        rec = json.load(open(xp))
        if gp.exists():
            rec.update(json.load(open(gp)))
        data[name] = rec
    return data


def demo_results() -> dict:
    return {
        "easy": {"xlift_score": 24.0, "mean_pass_rate": 0.85, "mean_boundary_score": 0.30,
                 "mean_reachability": 0.05, "mean_repair_gain": 0.10, "gepa_transfer_lift": 0.02,
                 "reward_trust_score": 0.92, "recommendation": "SKIP",
                 "recommendation_reason": "Already mastered — little headroom to train.",
                 "actual_lift": 0.018, "lift_ci_low": -0.012, "lift_ci_high": 0.05,
                 "lift_significant": False},
        "frontier": {"xlift_score": 79.0, "mean_pass_rate": 0.50, "mean_boundary_score": 0.95,
                     "mean_reachability": 0.34, "mean_repair_gain": 0.42, "gepa_transfer_lift": 0.18,
                     "reward_trust_score": 0.90, "recommendation": "BUY",
                     "recommendation_reason": "Tasks sit at the model's boundary and are repairable — high predicted lift.",
                     "actual_lift": 0.091, "lift_ci_low": 0.052, "lift_ci_high": 0.131,
                     "lift_significant": True},
        "hard": {"xlift_score": 21.0, "mean_pass_rate": 0.10, "mean_boundary_score": 0.33,
                 "mean_reachability": 0.07, "mean_repair_gain": 0.05, "gepa_transfer_lift": 0.00,
                 "reward_trust_score": 0.70, "recommendation": "SKIP",
                 "recommendation_reason": "Beyond the model's reach — answers outside its support.",
                 "actual_lift": 0.012, "lift_ci_low": -0.02, "lift_ci_high": 0.041,
                 "lift_significant": False},
    }


# --------------------------------------------------------------------------- #
# SVG helpers (inline, dependency-free)
# --------------------------------------------------------------------------- #
def _txt(x, y, s, size=12, color=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif">{s}</text>')


def svg_bars(values: dict, title: str, fmt=lambda v: f"{v:.0%}", ymax=None) -> str:
    """Vertical bar chart, one bar per cohort."""
    W, H, pad_b, pad_t, pad_l = 360, 230, 40, 36, 30
    names = [n for n in COHORTS if n in values]
    if not names:
        return ""
    vmax = ymax if ymax is not None else max(max(values.values()), 1e-9)
    bw = (W - pad_l - 20) / len(names)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    s.append(_txt(W / 2, 20, title, size=13, weight="600"))
    base = H - pad_b
    for i, n in enumerate(names):
        v = values[n]
        h = max(2, (v / vmax) * (H - pad_b - pad_t))
        x = pad_l + i * bw + bw * 0.18
        w = bw * 0.64
        col = COHORT_COLOR[n]
        s.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'rx="4" fill="{col}"/>')
        s.append(_txt(x + w / 2, base - h - 6, fmt(v), size=12, weight="600", color=col))
        s.append(_txt(x + w / 2, base + 16, n, size=12, color=MUTE))
    s.append(f'<line x1="{pad_l}" y1="{base}" x2="{W-10}" y2="{base}" stroke="#30363d"/>')
    s.append("</svg>")
    return "".join(s)


def svg_scatter(data: dict) -> str:
    """Predicted (xLift score) vs Actual RL lift, with CI bars."""
    W, H, m = 460, 300, 48
    pts = [(n, d) for n, d in data.items() if d.get("actual_lift") is not None]
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    s.append(_txt(W / 2, 20, "Predicted xLift score  vs  actual RL lift", size=13, weight="600"))
    x0, x1 = m, W - 20
    y0, y1 = H - 40, 40
    # axes
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#30363d"/>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#30363d"/>')
    s.append(_txt((x0 + x1) / 2, H - 8, "xLift score (0-100)", size=11, color=MUTE))
    s.append(f'<text x="14" y="{(y0+y1)/2:.0f}" font-size="11" fill="{MUTE}" '
             f'text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})" '
             f'font-family="system-ui">actual lift</text>')
    if not pts:
        s.append(_txt(W / 2, H / 2, "awaiting GRPO training results…", color=MUTE))
        s.append("</svg>")
        return "".join(s)
    lifts = [d["actual_lift"] for _, d in pts] + [d.get("lift_ci_high", d["actual_lift"]) for _, d in pts]
    lmax = max(max(lifts), 0.05)
    def sx(score): return x0 + (score / 100.0) * (x1 - x0)
    def sy(l): return y0 - (l / lmax) * (y0 - y1)
    # gridline at 0
    s.append(f'<line x1="{x0}" y1="{sy(0):.0f}" x2="{x1}" y2="{sy(0):.0f}" stroke="#21262d" stroke-dasharray="3 3"/>')
    for n, d in pts:
        cx, cy = sx(d["xlift_score"]), sy(d["actual_lift"])
        col = COHORT_COLOR[n]
        lo, hi = d.get("lift_ci_low"), d.get("lift_ci_high")
        if lo is not None and hi is not None:
            s.append(f'<line x1="{cx:.1f}" y1="{sy(lo):.1f}" x2="{cx:.1f}" y2="{sy(hi):.1f}" '
                     f'stroke="{col}" stroke-width="2" opacity="0.5"/>')
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{col}"/>')
        s.append(_txt(cx, cy - 12, f"{n} ({d['actual_lift']:+.1%})", size=11, color=col, weight="600"))
    s.append("</svg>")
    return "".join(s)


def svg_components(data: dict) -> str:
    """Grouped bars: the 4 normalized components per cohort."""
    comps = [("Boundary", "boundary_component"), ("Repair", "repair_component"),
             ("GEPA", "gepa_transfer_component"), ("Trust", "reward_trust_component")]
    names = [n for n in COHORTS if n in data]
    W, H, pad_l, pad_b, pad_t = 460, 250, 36, 50, 36
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    s.append(_txt(W / 2, 20, "xLift components (normalized 0-1)", size=13, weight="600"))
    base = H - pad_b
    gw = (W - pad_l - 20) / len(comps)
    for ci, (label, key) in enumerate(comps):
        gx = pad_l + ci * gw
        bw = gw * 0.8 / max(len(names), 1)
        for ni, n in enumerate(names):
            v = data[n].get(key, 0) or 0
            h = max(1, v * (H - pad_b - pad_t))
            x = gx + gw * 0.1 + ni * bw
            s.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{bw*0.86:.1f}" '
                     f'height="{h:.1f}" rx="2" fill="{COHORT_COLOR[n]}"/>')
        s.append(_txt(gx + gw / 2, base + 16, label, size=11, color=MUTE))
    # legend
    lx = pad_l
    for n in names:
        s.append(f'<rect x="{lx}" y="{H-18}" width="10" height="10" rx="2" fill="{COHORT_COLOR[n]}"/>')
        s.append(_txt(lx + 14, H - 9, n, size=10, color=MUTE, anchor="start"))
        lx += 70
    s.append("</svg>")
    return "".join(s)


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
def _rec_color(rec: str) -> str:
    r = (rec or "").upper()
    if r == "BUY": return GOOD
    if "FIX" in r or "VERIFIER" in r: return WARN
    return MUTE


def _cohort_card(name: str, d: dict) -> str:
    rec = d.get("recommendation", "—")
    col = _rec_color(rec)
    pass_rate = d.get("mean_pass_rate")
    rows = [
        ("xLift score", f"{d.get('xlift_score', 0):.0f} / 100"),
        ("Pass rate", f"{pass_rate:.0%}" if pass_rate is not None else "—"),
        ("BoundaryScore", f"{d.get('mean_boundary_score', 0):.2f}"),
        ("Reachability", f"{d.get('mean_reachability', 0):+.2f}" if d.get('mean_reachability') is not None else "—"),
        ("RepairGain", f"{d.get('mean_repair_gain', 0):+.2f}"),
        ("Reward trust", f"{d.get('reward_trust_score', 0):.0%}"),
    ]
    if d.get("actual_lift") is not None:
        sig = " ✓" if d.get("lift_significant") else " (noise)"
        rows.append(("Actual RL lift", f"{d['actual_lift']:+.1%}{sig}"))
    body = "".join(
        f'<div class="row"><span>{k}</span><b>{v}</b></div>' for k, v in rows
    )
    return f"""
    <div class="card" style="border-top:3px solid {COHORT_COLOR[name]}">
      <div class="card-h"><h3>{name}</h3>
        <span class="pill" style="background:{col}22;color:{col}">{rec}</span></div>
      {body}
      <p class="reason">{d.get('recommendation_reason','')}</p>
    </div>"""


def build_html(data: dict) -> str:
    cards = "".join(_cohort_card(n, data[n]) for n in COHORTS if n in data) or \
        '<p class="reason">No results yet. Run the metrics step, or use --demo.</p>'
    xlift_bars = svg_bars({n: data[n]["xlift_score"] for n in data}, "Predicted xLift score",
                          fmt=lambda v: f"{v:.0f}", ymax=100)
    have_lift = {n: data[n]["actual_lift"] for n in data if data[n].get("actual_lift") is not None}
    lift_bars = svg_bars(have_lift, "Actual RL lift (the inverted-U)", fmt=lambda v: f"{v:+.1%}") \
        if have_lift else '<p class="reason">Actual RL lift appears after GRPO training.</p>'
    scatter = svg_scatter(data)
    comps = svg_components(data)
    # headline trust callout
    risky = [n for n in data if data[n].get("misleading_lift_risk") or
             (data[n].get("reward_trust_score", 1) < 0.5)]
    trust_note = ""
    if risky:
        trust_note = (f'<div class="banner warn">⚠ Reward-trust risk in: '
                      f'{", ".join(risky)} — verifier may be gameable; lift on these is suspect.</div>')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xLift — learnability dashboard</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:{BG}; color:{INK};
         font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:32px 20px 64px; }}
  header h1 {{ margin:0 0 4px; font-size:26px; }}
  header p {{ margin:0; color:{MUTE}; font-size:14px; }}
  .tag {{ color:{ACCENT}; font-weight:600; }}
  .grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); margin:22px 0; }}
  .card {{ background:{CARD}; border:1px solid #21262d; border-radius:12px; padding:16px; }}
  .card-h {{ display:flex; align-items:center; justify-content:space-between; }}
  .card-h h3 {{ margin:0; text-transform:capitalize; font-size:17px; }}
  .pill {{ font-size:11px; font-weight:700; padding:3px 9px; border-radius:20px; letter-spacing:.3px; }}
  .row {{ display:flex; justify-content:space-between; font-size:13px; padding:4px 0;
          border-bottom:1px solid #21262d; }}
  .row span {{ color:{MUTE}; }}
  .reason {{ color:{MUTE}; font-size:12px; margin:10px 0 0; line-height:1.45; }}
  .charts {{ display:grid; gap:16px; grid-template-columns:1fr 1fr; }}
  .panel {{ background:{CARD}; border:1px solid #21262d; border-radius:12px; padding:14px; }}
  .banner {{ border-radius:10px; padding:12px 14px; font-size:13px; margin:8px 0 0; }}
  .warn {{ background:{WARN}22; color:{WARN}; border:1px solid {WARN}55; }}
  .lead {{ background:{CARD}; border:1px solid #21262d; border-left:3px solid {ACCENT};
           border-radius:10px; padding:14px 16px; font-size:14px; line-height:1.5; margin:18px 0; }}
  footer {{ color:{MUTE}; font-size:12px; margin-top:30px; }}
  @media(max-width:760px){{ .charts{{grid-template-columns:1fr;}} }}
</style></head><body><div class="wrap">
  <header>
    <h1>xLift <span class="tag">learnability dashboard</span></h1>
    <p>Predicting post-RL training lift from cheap, training-free signals — validated against GRPO.</p>
  </header>
  <div class="lead">
    The same property that makes a task a good <b>human</b> evaluation makes it good <b>training data</b>:
    <b>learnability</b>. xLift measures it on the model you'll train (Qwen2.5-1.5B) <i>without</i> training,
    and predicts which cohort actually lifts under GRPO. The frontier cohort should win.
  </div>
  {trust_note}
  <div class="grid">{cards}</div>
  <div class="charts">
    <div class="panel">{scatter}</div>
    <div class="panel">{lift_bars}</div>
    <div class="panel">{xlift_bars}</div>
    <div class="panel">{comps}</div>
  </div>
  <footer>Generated by eval/dashboard.py — offline, self-contained. Cohorts: easy · frontier · hard.</footer>
</div></body></html>"""


def build_dashboard(results_dir: Path = RESULTS_DIR, demo: bool = False) -> Path:
    data = demo_results() if demo else load_results(results_dir)
    html = build_html(data)
    out = results_dir / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Dashboard → {out}  ({len(data)} cohorts)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data for rehearsal")
    ap.add_argument("--open", action="store_true", help="open in browser after building")
    args = ap.parse_args()
    path = build_dashboard(demo=args.demo)
    if args.open:
        webbrowser.open(f"file://{path.resolve()}")
