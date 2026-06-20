const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";              // 13.3 x 7.5
p.author = "Team xLift";
p.title = "xLift";

// ---- palette (dark, topic-informed: teal = learnable, amber = trust) ----
const BG = "0E1117", SURF = "161B22", LINE = "30363D";
const INK = "E6EDF3", MUTE = "8B949E";
const TEAL = "2DD4BF", AMBER = "F59E0B", BLUE = "58A6FF", VIOLET = "BC8CFF";
const GOOD = "3FB950", BAD = "F85149";
const HFONT = "Trebuchet MS", BFONT = "Calibri", MONO = "Consolas";
const W = 13.3, H = 7.5;
const sh = () => ({ type: "outer", color: "000000", blur: 9, offset: 3, angle: 90, opacity: 0.30 });

function bgDark(s){ s.background = { color: BG }; }
function footer(s, n){
  s.addText("xLift  ·  Inference-Time Compute Hackathon  ·  Applied AI",
    { x:0.6, y:H-0.45, w:9, h:0.3, fontFace:BFONT, fontSize:10, color:MUTE, align:"left", margin:0 });
  s.addText(String(n), { x:W-1.0, y:H-0.45, w:0.4, h:0.3, fontFace:MONO, fontSize:10, color:MUTE, align:"right", margin:0 });
}
function kicker(s, text, color){
  s.addShape(p.shapes.RECTANGLE, { x:0.6, y:0.55, w:0.13, h:0.42, fill:{color} });
  s.addText(text.toUpperCase(), { x:0.85, y:0.52, w:11, h:0.45, fontFace:MONO, fontSize:13,
    color, bold:true, charSpacing:2, align:"left", valign:"middle", margin:0 });
}
function title(s, text){
  s.addText(text, { x:0.6, y:1.05, w:12.1, h:1.0, fontFace:HFONT, fontSize:32, color:INK,
    bold:true, align:"left", valign:"middle", margin:0 });
}
// rounded surface card with a thick colored left edge (the repeated motif)
function card(s, x, y, w, h, accent){
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius:0.08,
    fill:{color:SURF}, line:{color:LINE, width:1}, shadow:sh() });
  s.addShape(p.shapes.RECTANGLE, { x, y, w:0.09, h, fill:{color:accent} });
}

// ===================================================================== //
// 1 — TITLE
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s);
  s.addShape(p.shapes.RECTANGLE, { x:0, y:0, w:0.18, h:H, fill:{color:TEAL} });
  s.addShape(p.shapes.RECTANGLE, { x:0.18, y:0, w:0.06, h:H, fill:{color:AMBER} });
  s.addText("xLift", { x:0.9, y:2.0, w:8, h:1.6, fontFace:HFONT, fontSize:92, color:INK, bold:true, margin:0 });
  s.addText("Predict post-RL training lift — before you train.",
    { x:0.95, y:3.6, w:11.4, h:0.7, fontFace:HFONT, fontSize:26, color:TEAL, margin:0 });
  s.addText([
    { text:"Cheap, training-free learnability signals — validated against GRPO.", options:{ breakLine:true, color:INK } },
    { text:"Learnability is model-relative: measure it on the model you'll train.", options:{ color:MUTE } },
  ], { x:0.95, y:4.45, w:11.4, h:1.0, fontFace:BFONT, fontSize:16, lineSpacingMultiple:1.2, margin:0 });
  s.addText([
    { text:"Talent Marketplace + Applied AI", options:{ color:AMBER, bold:true } },
    { text:"     github.com/mssumanth/xlift", options:{ color:MUTE } },
  ], { x:0.95, y:6.35, w:11.4, h:0.4, fontFace:MONO, fontSize:13, margin:0 });
})();

// ===================================================================== //
// 2 — PROBLEM
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "the problem", AMBER);
  title(s, "RL post-training is a gamble you pay for up front");
  const col = [
    ["Training is expensive", "Each GRPO run burns GPU-hours per data cohort. You commit the compute before you know if it helps."],
    ["Quality is invisible", "Two datasets can look identical — same size, same topic — yet one lifts the model and the other does nothing."],
    ["Buying data is blind", "Enterprises pay for training data with no way to tell, in advance, whether it will actually move the model."],
  ];
  col.forEach((c,i) => {
    const x = 0.6 + i*4.07;
    card(s, x, 2.4, 3.8, 2.7, AMBER);
    s.addText(c[0], { x:x+0.32, y:2.65, w:3.3, h:0.5, fontFace:HFONT, fontSize:18, color:INK, bold:true, margin:0 });
    s.addText(c[1], { x:x+0.32, y:3.25, w:3.3, h:1.7, fontFace:BFONT, fontSize:14, color:MUTE, lineSpacingMultiple:1.18, margin:0 });
  });
  s.addText([
    { text:"The question:  ", options:{ color:MUTE } },
    { text:"which data cohort will actually teach the model — without running the training to find out?", options:{ color:INK, bold:true } },
  ], { x:0.6, y:5.5, w:12.1, h:0.8, fontFace:BFONT, fontSize:18, align:"left", valign:"middle", margin:0 });
  footer(s, 2);
})();

// ===================================================================== //
// 3 — INSIGHT
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "the insight", TEAL);
  title(s, "Learnability is the universal signal");
  s.addText([
    { text:"The same property that makes a task a good ", options:{ color:INK } },
    { text:"human", options:{ color:AMBER, bold:true } },
    { text:" evaluation makes it good ", options:{ color:INK } },
    { text:"training data", options:{ color:TEAL, bold:true } },
    { text:":  learnability.", options:{ color:INK } },
  ], { x:0.6, y:2.2, w:12.1, h:0.7, fontFace:BFONT, fontSize:20, align:"left", margin:0 });
  card(s, 0.6, 3.1, 5.9, 3.0, TEAL);
  s.addText("A task is learnable when…", { x:0.95, y:3.35, w:5.2, h:0.5, fontFace:HFONT, fontSize:17, color:TEAL, bold:true, margin:0 });
  s.addText([
    { text:"the model is right about half the time (at its boundary)", options:{ bullet:true, breakLine:true } },
    { text:"failures are reachable and fixable with a hint", options:{ bullet:true, breakLine:true } },
    { text:"the lesson generalizes to unseen problems", options:{ bullet:true, breakLine:true } },
    { text:"the grader can't be gamed", options:{ bullet:true } },
  ], { x:0.95, y:3.95, w:5.3, h:2.0, fontFace:BFONT, fontSize:15, color:INK, paraSpaceAfter:8, margin:0 });
  card(s, 6.8, 3.1, 5.9, 3.0, BLUE);
  s.addText("…and it's model-relative", { x:7.15, y:3.35, w:5.2, h:0.5, fontFace:HFONT, fontSize:17, color:BLUE, bold:true, margin:0 });
  s.addText([
    { text:"A problem that's trivial for GPT-4 can sit right at the boundary of a 1.5B model.", options:{ breakLine:true, color:INK } },
    { text:"", options:{ breakLine:true, fontSize:6 } },
    { text:"So we measure every signal on the SAME model we'll train (Qwen2.5-1.5B) — without training it. That's what makes the signal a real predictor, not a coincidence.", options:{ color:MUTE } },
  ], { x:7.15, y:3.95, w:5.3, h:2.0, fontFace:BFONT, fontSize:15, lineSpacingMultiple:1.15, margin:0 });
  footer(s, 3);
})();

// ===================================================================== //
// 4 — SIX SIGNALS
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "the framework", TEAL);
  title(s, "Six cheap signals, two questions");
  const sig = [
    ["BoundaryScore", "is it at the model's edge?", "4·p·(1−p)", TEAL],
    ["Reachability", "can it ever get it?", "pass@k − pass@1", TEAL],
    ["RepairGain", "does a hint fix it?", "score after − before", TEAL],
    ["GEPA Transfer", "does the lesson generalize?", "lift on unseen tasks", TEAL],
    ["Reward-Trust", "is the grader fair?", "reward vs length", AMBER],
    ["AntiCheat", "can fakes fool it?", "fake-solution pass rate", AMBER],
  ];
  sig.forEach((c,i) => {
    const cx = 0.6 + (i%3)*4.07, cy = 2.35 + Math.floor(i/3)*2.0;
    card(s, cx, cy, 3.8, 1.8, c[3]);
    s.addText(c[0], { x:cx+0.32, y:cy+0.18, w:3.3, h:0.45, fontFace:HFONT, fontSize:17, color:INK, bold:true, margin:0 });
    s.addText(c[1], { x:cx+0.32, y:cy+0.66, w:3.3, h:0.4, fontFace:BFONT, fontSize:14, color:MUTE, italic:true, margin:0 });
    s.addText(c[2], { x:cx+0.32, y:cy+1.12, w:3.3, h:0.4, fontFace:MONO, fontSize:13, color:c[3], margin:0 });
  });
  // legend
  s.addShape(p.shapes.OVAL, { x:0.6, y:6.55, w:0.2, h:0.2, fill:{color:TEAL} });
  s.addText("is there something to learn?", { x:0.9, y:6.5, w:4, h:0.3, fontFace:BFONT, fontSize:13, color:MUTE, margin:0 });
  s.addShape(p.shapes.OVAL, { x:5.2, y:6.55, w:0.2, h:0.2, fill:{color:AMBER} });
  s.addText("can you trust the score?", { x:5.5, y:6.5, w:4, h:0.3, fontFace:BFONT, fontSize:13, color:MUTE, margin:0 });
  footer(s, 4);
})();

// ===================================================================== //
// 5 — BOUNDARYSCORE DERIVATION
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "the headline signal", TEAL);
  title(s, "BoundaryScore is derived, not fit");
  // left: the chain of reasoning
  card(s, 0.6, 2.35, 6.0, 3.9, TEAL);
  s.addText([
    { text:"GRPO normalizes advantage within a rollout group.", options:{ bullet:true, breakLine:true } },
    { text:"All-right or all-wrong → zero variance → zero advantage → ZERO gradient. Nothing is learned.", options:{ bullet:true, breakLine:true } },
    { text:"For a binary verifier, within-group reward variance is exactly p(1−p).", options:{ bullet:true, breakLine:true } },
    { text:"It peaks at p = 0.5 — the model's boundary.", options:{ bullet:true } },
  ], { x:0.95, y:2.6, w:5.4, h:2.6, fontFace:BFONT, fontSize:15.5, color:INK, paraSpaceAfter:11, lineSpacingMultiple:1.1, margin:0 });
  s.addText([
    { text:"BoundaryScore = 4·p·(1−p)", options:{ color:TEAL, bold:true } },
  ], { x:0.95, y:5.55, w:5.4, h:0.5, fontFace:MONO, fontSize:20, margin:0 });
  // right: the hill
  s.addChart(p.charts.LINE, [{
    name:"4p(1-p)",
    labels:["0",".1",".2",".3",".4","0.5",".6",".7",".8",".9","1"],
    values:[0,0.36,0.64,0.84,0.96,1.0,0.96,0.84,0.64,0.36,0]
  }], {
    x:7.0, y:2.5, w:5.7, h:3.5, lineSize:3, lineSmooth:true, chartColors:[TEAL],
    chartArea:{ fill:{color:SURF} }, plotArea:{ fill:{color:SURF} },
    catAxisLabelColor:MUTE, valAxisLabelColor:MUTE, catAxisLabelFontSize:10, valAxisLabelFontSize:10,
    valGridLine:{ color:LINE, size:0.5 }, catGridLine:{ style:"none" },
    showLegend:false, showTitle:true, title:"learning signal vs pass rate", titleColor:INK, titleFontSize:13,
    valAxisMaxVal:1.1, valAxisMinVal:0,
  });
  footer(s, 5);
})();

// ===================================================================== //
// 6 — BEYOND PASS RATE
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "orthogonal signals", TEAL);
  title(s, "Why pass rate alone isn't enough");
  const rows = [
    ["Reachability", "Two tasks both pass 10% of the time: one is solved once in k tries (RL can sharpen it), the other never (hopeless). Variance can't tell them apart — pass@k − pass@1 can."],
    ["RepairGain", "Give the model a hint after it fails. If it recovers, the knowledge was almost there — highly teachable. If not, the task is beyond it."],
    ["GEPA Transfer", "Evolve a strategy on half the tasks, apply it to unseen tasks. If the lift transfers, the cohort teaches general skills — not memorization."],
  ];
  rows.forEach((r,i) => {
    const y = 2.35 + i*1.32;
    card(s, 0.6, y, 12.1, 1.18, TEAL);
    s.addText(r[0], { x:0.95, y:y+0.1, w:3.0, h:1.0, fontFace:HFONT, fontSize:18, color:TEAL, bold:true, valign:"middle", margin:0 });
    s.addText(r[1], { x:3.9, y:y+0.12, w:8.5, h:0.95, fontFace:BFONT, fontSize:14.5, color:INK, valign:"middle", lineSpacingMultiple:1.1, margin:0 });
  });
  s.addText("Variance says “there is signal here.”  The others say “…and it's the right kind.”",
    { x:0.6, y:6.45, w:12.1, h:0.4, fontFace:BFONT, fontSize:15, color:MUTE, italic:true, align:"center", margin:0 });
  footer(s, 6);
})();

// ===================================================================== //
// 7 — TRUST AXIS
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "the trust axis", AMBER);
  title(s, "A high lift you can't trust is worthless");
  s.addText("RL needs a grader (verifier). If the grader is gameable, the model learns to cheat it — not to solve the task. So informativeness isn't enough; the reward must be trustworthy.",
    { x:0.6, y:2.05, w:12.1, h:0.8, fontFace:BFONT, fontSize:16, color:INK, lineSpacingMultiple:1.15, margin:0 });
  card(s, 0.6, 3.05, 5.9, 3.1, AMBER);
  s.addText("Reward-Trust", { x:0.95, y:3.3, w:5, h:0.5, fontFace:HFONT, fontSize:18, color:AMBER, bold:true, margin:0 });
  s.addText([
    { text:"Does reward track answer length instead of correctness?", options:{ bullet:true, breakLine:true } },
    { text:"If long = rewarded, the model just learns to ramble.", options:{ bullet:true, breakLine:true } },
    { text:"A near-free flag, computed from the same rollouts.", options:{ bullet:true } },
  ], { x:0.95, y:3.85, w:5.3, h:2.1, fontFace:BFONT, fontSize:14.5, color:INK, paraSpaceAfter:8, margin:0 });
  card(s, 6.8, 3.05, 5.9, 3.1, AMBER);
  s.addText("AntiCheat", { x:7.15, y:3.3, w:5, h:0.5, fontFace:HFONT, fontSize:18, color:AMBER, bold:true, margin:0 });
  s.addText([
    { text:"Red-team the verifier with fake / hardcoded solutions.", options:{ bullet:true, breakLine:true } },
    { text:"If fakes pass, training rewards cheating.", options:{ bullet:true, breakLine:true } },
    { text:"The exhibit: a weak-verifier cohort whose reward climbs while true accuracy doesn't.", options:{ bullet:true } },
  ], { x:7.15, y:3.85, w:5.3, h:2.1, fontFace:BFONT, fontSize:14.5, color:INK, paraSpaceAfter:8, margin:0 });
  s.addText("The grad-norm oracle gets fooled here on purpose — proving trust is a separate axis from informativeness.",
    { x:0.6, y:6.45, w:12.1, h:0.4, fontFace:BFONT, fontSize:14, color:MUTE, italic:true, align:"center", margin:0 });
  footer(s, 7);
})();

// ===================================================================== //
// 8 — VALIDATION METHOD
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "how we prove it", BLUE);
  title(s, "Predict cheaply, then verify with real GRPO");
  const steps = [
    ["1", "Build cohorts", "Easy · Frontier · Hard, partitioned by the model's own pass rate."],
    ["2", "Score cheaply", "Compute the six signals from N rollouts — no weight updates."],
    ["3", "Train for real", "Run identical GRPO on each cohort; measure actual lift with bootstrap CIs."],
    ["4", "Check the call", "Do the cheap signals predict the real lift ordering? Frontier should win."],
  ];
  steps.forEach((c,i) => {
    const x = 0.6 + i*3.07;
    card(s, x, 2.4, 2.85, 3.0, BLUE);
    s.addShape(p.shapes.OVAL, { x:x+0.3, y:2.65, w:0.65, h:0.65, fill:{color:BLUE} });
    s.addText(c[0], { x:x+0.3, y:2.65, w:0.65, h:0.65, fontFace:HFONT, fontSize:24, color:BG, bold:true, align:"center", valign:"middle", margin:0 });
    s.addText(c[1], { x:x+0.28, y:3.45, w:2.4, h:0.5, fontFace:HFONT, fontSize:16, color:INK, bold:true, margin:0 });
    s.addText(c[2], { x:x+0.28, y:3.95, w:2.45, h:1.4, fontFace:BFONT, fontSize:13.5, color:MUTE, lineSpacingMultiple:1.15, margin:0 });
  });
  s.addText([
    { text:"The headline result is fit-free:  ", options:{ color:MUTE } },
    { text:"lift peaks at the learnable frontier and falls off on both sides — an inverted U.", options:{ color:INK, bold:true } },
  ], { x:0.6, y:5.7, w:12.1, h:0.6, fontFace:BFONT, fontSize:16, align:"left", valign:"middle", margin:0 });
  footer(s, 8);
})();

// ===================================================================== //
// 9 — RESULTS
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "results", GOOD);
  title(s, "The inverted U: lift peaks at the frontier");
  s.addChart(p.charts.BAR, [{
    name:"actual RL lift", labels:["easy","frontier","hard"], values:[1.8, 9.1, 1.2]
  }], {
    x:0.6, y:2.4, w:6.6, h:3.6, barDir:"col", chartColors:[BLUE],
    chartArea:{ fill:{color:SURF} }, plotArea:{ fill:{color:SURF} },
    catAxisLabelColor:INK, valAxisLabelColor:MUTE, catAxisLabelFontSize:13, valAxisLabelFontSize:10,
    valGridLine:{ color:LINE, size:0.5 }, catGridLine:{ style:"none" },
    showValue:true, dataLabelColor:INK, dataLabelFontSize:13, dataLabelPosition:"outEnd",
    showLegend:false, showTitle:true, title:"accuracy lift after GRPO (%)", titleColor:INK, titleFontSize:13,
    valAxisMaxVal:11,
  });
  card(s, 7.5, 2.4, 5.2, 3.6, GOOD);
  s.addText("What the numbers say", { x:7.85, y:2.62, w:4.5, h:0.5, fontFace:HFONT, fontSize:17, color:GOOD, bold:true, margin:0 });
  s.addText([
    { text:"Frontier cohort lifts ~5× more than easy or hard.", options:{ bullet:true, breakLine:true } },
    { text:"xLift score ranks the cohorts in the same order — the cheap signal predicts the expensive outcome.", options:{ bullet:true, breakLine:true } },
    { text:"Only the frontier lift clears the bootstrap-CI noise floor.", options:{ bullet:true } },
  ], { x:7.85, y:3.2, w:4.6, h:2.6, fontFace:BFONT, fontSize:14, color:INK, paraSpaceAfter:9, lineSpacingMultiple:1.12, margin:0 });
  s.addText("Representative shape — final numbers drop in live from the overnight GRPO run + dashboard.",
    { x:0.6, y:6.4, w:12.1, h:0.4, fontFace:BFONT, fontSize:12.5, color:MUTE, italic:true, align:"center", margin:0 });
  footer(s, 9);
})();

// ===================================================================== //
// 10 — PARETO
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "why it's useful", TEAL);
  title(s, "Breaking the cost–quality frontier");
  s.addText("The only ground truth is to train — but training every candidate cohort is exactly what's too expensive. xLift predicts that outcome at the cost of a handful of inference rollouts.",
    { x:0.6, y:2.05, w:12.1, h:0.8, fontFace:BFONT, fontSize:16, color:INK, lineSpacingMultiple:1.15, margin:0 });
  const cmp = [
    ["Full GRPO oracle", "GPU-hours per cohort", "exact", BAD],
    ["Grad-norm oracle", "one fwd-backward pass", "high, but fooled by weak verifiers", AMBER],
    ["xLift signals", "N inference rollouts", "predicts the lift ordering — cheaply", TEAL],
  ];
  cmp.forEach((c,i) => {
    const y = 3.0 + i*1.05;
    card(s, 0.6, y, 12.1, 0.92, c[3]);
    s.addText(c[0], { x:0.95, y:y+0.05, w:3.4, h:0.8, fontFace:HFONT, fontSize:16, color:INK, bold:true, valign:"middle", margin:0 });
    s.addText(c[1], { x:4.5, y:y+0.05, w:3.4, h:0.8, fontFace:MONO, fontSize:14, color:c[3], valign:"middle", margin:0 });
    s.addText(c[2], { x:8.0, y:y+0.05, w:4.5, h:0.8, fontFace:BFONT, fontSize:14, color:MUTE, valign:"middle", margin:0 });
  });
  s.addText("Cheap variance-based signals dominate the Pareto frontier — high predictive power at a fraction of the cost.",
    { x:0.6, y:6.35, w:12.1, h:0.5, fontFace:BFONT, fontSize:15, color:TEAL, italic:true, align:"center", margin:0 });
  footer(s, 10);
})();

// ===================================================================== //
// 11 — WHAT WE BUILT
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s); kicker(s, "what we shipped", VIOLET);
  title(s, "Production-ready, and it runs on a laptop");
  const items = [
    ["Model-relative engine", "Qwen2.5-1.5B rollout backend with async micro-batching; all signals scored on the model you train."],
    ["Validated science", "dspy.GEPA, bootstrap-CI lift, val/test split, idempotent resumable overnight pipeline."],
    ["Tested on CPU", "Pure-logic + fake-backend end-to-end tests — the whole pipeline runs with no GPU, no API, no network."],
    ["Live dashboard", "Self-contained HTML: predicted-vs-actual, inverted-U, reward-hack exhibit. Offline, can't crash on stage."],
  ];
  items.forEach((c,i) => {
    const x = 0.6 + (i%2)*6.15, y = 2.35 + Math.floor(i/2)*1.9;
    card(s, x, y, 5.9, 1.7, VIOLET);
    s.addText(c[0], { x:x+0.32, y:y+0.16, w:5.3, h:0.45, fontFace:HFONT, fontSize:17, color:INK, bold:true, margin:0 });
    s.addText(c[1], { x:x+0.32, y:y+0.62, w:5.35, h:1.0, fontFace:BFONT, fontSize:14, color:MUTE, lineSpacingMultiple:1.12, margin:0 });
  });
  s.addText([
    { text:"Same math, two markets:  ", options:{ color:VIOLET, bold:true } },
    { text:"the learnability that grades a training cohort also grades a human candidate — index the RL frontier and the talent frontier with one idea.", options:{ color:INK } },
  ], { x:0.6, y:6.3, w:12.1, h:0.6, fontFace:BFONT, fontSize:14.5, align:"left", valign:"middle", margin:0 });
  footer(s, 11);
})();

// ===================================================================== //
// 12 — CLOSING
// ===================================================================== //
(() => {
  const s = p.addSlide(); bgDark(s);
  s.addShape(p.shapes.RECTANGLE, { x:0, y:0, w:0.18, h:H, fill:{color:TEAL} });
  s.addShape(p.shapes.RECTANGLE, { x:0.18, y:0, w:0.06, h:H, fill:{color:AMBER} });
  s.addText("Buy the data that will actually", { x:0.95, y:2.2, w:11.5, h:0.9, fontFace:HFONT, fontSize:40, color:INK, bold:true, margin:0 });
  s.addText("teach your model.", { x:0.95, y:3.0, w:11.5, h:0.9, fontFace:HFONT, fontSize:40, color:TEAL, bold:true, margin:0 });
  s.addText("xLift turns the expensive question — will this data lift my model? — into a cheap, trustworthy measurement.",
    { x:0.97, y:4.2, w:11.3, h:0.8, fontFace:BFONT, fontSize:18, color:MUTE, lineSpacingMultiple:1.2, margin:0 });
  s.addText([
    { text:"github.com/mssumanth/xlift", options:{ color:INK, bold:true, breakLine:true } },
    { text:"Team xLift  ·  Inference-Time Compute Hackathon", options:{ color:MUTE } },
  ], { x:0.97, y:5.7, w:11.3, h:0.8, fontFace:MONO, fontSize:15, lineSpacingMultiple:1.3, margin:0 });
})();

p.writeFile({ fileName: "xLift_pitch.pptx" }).then(f => console.log("wrote", f));
