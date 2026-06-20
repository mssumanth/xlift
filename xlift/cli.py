"""xLift CLI — Typer-based command interface."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="xlift", add_completion=False, help="xLift: dataset quality prediction via learnability signals")


def _cfg(config_path: Optional[str] = None):
    from .config import Config
    return Config.load(config_path or "config.yaml")


@app.command()
def setup():
    """Check environment: venv, model download, API keys."""
    cfg = _cfg()
    model = Path(cfg.model_path)
    if model.exists():
        typer.echo(f"✓ Model found: {cfg.model_path}")
    else:
        typer.echo(f"⚠ Model not found at {cfg.model_path}")
        typer.echo("Download with:")
        typer.echo(f"  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir {cfg.model_path}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo("✓ ANTHROPIC_API_KEY set")
    else:
        typer.echo("⚠ ANTHROPIC_API_KEY not set (needed for synth + legacy GEPA)")
    typer.echo("Run 'make smoke' to verify end-to-end with a 10-step training job.")


@app.command()
def smoke(
    steps: int = typer.Option(10, help="Training steps for smoke test"),
    n_tasks: int = typer.Option(16, help="Tasks for smoke test"),
):
    """GATE 1: 10-step TRL GRPO run on a tiny sample. Must complete without error."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()

    typer.echo(f"Smoke test: {n_tasks} tasks × {steps} steps")

    # Load a tiny sample
    from .data import load_gsm8k
    tasks = load_gsm8k("train", cfg)[:n_tasks]
    if not tasks:
        typer.echo("ERROR: No tasks loaded. Run 'make data' first.", err=True)
        raise typer.Exit(1)

    from .types import Cohort, TrainCfg
    cohort = Cohort(
        name="smoke_test",
        verifier="strong",
        task_ids=[t.id for t in tasks],
        property_varied="smoke test",
    )
    train_cfg = TrainCfg(steps=steps, ckpt_every=steps, group_size=4, per_device_bs=4)
    out_dir = str(Path(cfg.artifacts_dir) / "train" / "smoke_test")

    from .train.trl_backend import TRLBackend
    backend = TRLBackend()
    backend.train_cohort(cohort, tasks, model_path=cfg.model_path,
                          gpu_ids=[0], cfg=train_cfg, out_dir=out_dir)
    typer.echo(f"✓ Smoke test passed. Checkpoint in {out_dir}")


@app.command()
def data():
    """Load GSM8K train + test and write jsonl artifacts."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .data import load_gsm8k
    train = load_gsm8k("train", cfg)
    test = load_gsm8k("test", cfg)
    typer.echo(f"✓ {len(train)} train + {len(test)} test tasks")


@app.command()
def foundation(
    n: int = typer.Option(4000, help="Max tasks to roll out"),
    force: bool = typer.Option(False, help="Overwrite existing rollouts"),
    tp_size: int = typer.Option(1, help="Tensor parallel size"),
):
    """Foundation pass: k=16 rollouts per task via vLLM."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .data import load_gsm8k
    from .rollout import run_foundation_rollouts

    tasks = load_gsm8k("train", cfg)[:n]
    run_foundation_rollouts(
        tasks,
        model_path=cfg.model_path,
        k=cfg.k_rollouts,
        temperature=cfg.roll_temperature,
        max_tokens=cfg.roll_max_tokens,
        tp_size=tp_size,
        store_text=cfg.store_rollout_text,
        force=force,
        cfg=cfg,
    )


@app.command()
def eval_base():
    """GATE 2: Measure base model accuracy on GSM8K test (expect ~0.45)."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .evaluate import evaluate_base
    result = evaluate_base(cfg)
    typer.echo(f"Base accuracy: {result['acc']:.3f} ({result['n']} tasks)")
    if not (0.35 <= result["acc"] <= 0.60):
        typer.echo("⚠ Accuracy outside expected range [0.35, 0.60] — check prompt/extraction")


@app.command()
def synth(n_cand: int = typer.Option(1000, help="Candidate synthetic tasks to generate")):
    """Generate Claude synthetic tasks for C7. Then re-run foundation on them."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .data import load_gsm8k
    from .synth import generate_synthetic_tasks
    from .rollout import run_foundation_rollouts

    seed_tasks = load_gsm8k("train", cfg)[:50]
    synth_tasks = generate_synthetic_tasks(n_cand, model=cfg.synth_model,
                                            seed_examples=seed_tasks, cfg=cfg)
    # Roll out synth candidates — these are appended to the foundation
    run_foundation_rollouts(
        synth_tasks,
        model_path=cfg.model_path,
        k=cfg.k_rollouts,
        temperature=cfg.roll_temperature,
        max_tokens=cfg.roll_max_tokens,
        cfg=cfg,
    )
    typer.echo(f"✓ {len(synth_tasks)} synthetic tasks generated and rolled out")


@app.command()
def cohorts():
    """Build C1-C7 cohorts from foundation rollout index."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .rollout import load_foundation
    from .cohorts import build_cohorts

    foundation = load_foundation(cfg)
    if not foundation:
        typer.echo("ERROR: No foundation rollouts. Run 'make foundation' first.", err=True)
        raise typer.Exit(1)

    index_csv = str(Path(cfg.artifacts_dir) / "index" / "pass_rate_index.csv")
    cohort_map = build_cohorts(index_csv, foundation, cfg)
    typer.echo(f"✓ Built {len(cohort_map)} cohorts: {', '.join(cohort_map.keys())}")


@app.command()
def metrics(cohort_names: str = typer.Option("all", help="Comma-separated cohort names or 'all'")):
    """Compute all cheap metrics from foundation rollouts."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .rollout import load_foundation
    from .cohorts import load_cohort
    from .metrics.run import compute_cohort_metrics
    from .data import load_gsm8k

    foundation = load_foundation(cfg)
    test_tasks = load_gsm8k("test", cfg)[:cfg.eval_n]
    test_questions = [t.question for t in test_tasks]

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg.embedder)
    except Exception as e:
        typer.echo(f"⚠ sentence-transformers not available: {e}. Vendi/redundancy/dist_match will be 0.")
        embedder = None

    artifacts = Path(cfg.artifacts_dir) / "cohorts"
    all_cohort_names = [p.stem.replace(".cohort", "") for p in artifacts.glob("*.cohort.json")]
    names_to_run = all_cohort_names if cohort_names == "all" else cohort_names.split(",")

    for name in names_to_run:
        typer.echo(f"Computing metrics for {name}...")
        try:
            cohort, _ = load_cohort(name, cfg)
            m = compute_cohort_metrics(cohort, foundation, embedder, test_questions, cfg)
            typer.echo(f"  frontier_score={m.frontier_score:.3f}, effective_ratio={m.effective_ratio:.3f}")
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)


@app.command()
def train(
    cohort_names: str = typer.Option("C2_frontier,C1_easy", help="Comma-separated cohort names"),
    gpus: str = typer.Option("0,1", help="Comma-separated GPU ids"),
):
    """Train cohorts with GRPO (TRL backend)."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .cohorts import load_cohort
    from .types import TrainCfg
    from .train.trl_backend import TRLBackend

    train_cfg = TrainCfg(
        steps=cfg.train_steps, ckpt_every=cfg.ckpt_every, group_size=cfg.group_size,
        lr=cfg.learning_rate, kl_beta=cfg.kl_beta, max_prompt=cfg.max_prompt_len,
        max_completion=cfg.max_completion_len, temperature=cfg.roll_temperature,
        per_device_bs=cfg.per_device_bs, grad_accum=cfg.grad_accum,
        full_finetune=cfg.full_finetune, seed=cfg.seed,
    )
    gpu_ids = [int(g) for g in gpus.split(",")]
    backend = TRLBackend()

    for name in cohort_names.split(","):
        name = name.strip()
        typer.echo(f"Training {name}...")
        cohort, tasks = load_cohort(name, cfg)
        out_dir = str(Path(cfg.artifacts_dir) / "train" / name)
        backend.train_cohort(cohort, tasks, model_path=cfg.model_path,
                              gpu_ids=gpu_ids, cfg=train_cfg, out_dir=out_dir)
        typer.echo(f"  ✓ {name} done → {out_dir}")


@app.command()
def gradnorm(
    cohort_names: str = typer.Option("all", help="Cohort names or 'all'"),
    n_prompts: int = typer.Option(32),
    G: int = typer.Option(8),
):
    """Compute gradient norm oracle for cohorts (mid-cost)."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .rollout import load_foundation
    from .cohorts import load_cohort
    from .metrics.gradnorm import single_step_grad_norm

    foundation = load_foundation(cfg)
    artifacts = Path(cfg.artifacts_dir)
    all_names = [p.stem.replace(".cohort", "") for p in (artifacts / "cohorts").glob("*.cohort.json")]
    names_to_run = all_names if cohort_names == "all" else cohort_names.split(",")

    for name in names_to_run:
        typer.echo(f"Gradnorm for {name}...")
        try:
            cohort, tasks = load_cohort(name, cfg)
            gn = single_step_grad_norm(
                cfg.model_path, tasks[:n_prompts], cohort.verifier,
                G=G, n_prompts=n_prompts, foundation=foundation,
            )
            out = artifacts / "gradnorm" / f"{name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump({"name": name, "grad_norm": gn, "verifier": cohort.verifier}, f)
            typer.echo(f"  {name}: grad_norm={gn:.4f}")
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)


@app.command()
def evaluate(cohort_names: str = typer.Option("all", help="Cohort names or 'all'")):
    """Evaluate trained cohort checkpoints and compute lift."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .evaluate import evaluate_base, evaluate_cohort

    evaluate_base(cfg)  # ensures base.json exists

    artifacts = Path(cfg.artifacts_dir)
    train_dir = artifacts / "train"
    all_names = [d.name for d in train_dir.iterdir() if d.is_dir()] if train_dir.exists() else []
    names_to_run = all_names if cohort_names == "all" else cohort_names.split(",")

    for name in names_to_run:
        typer.echo(f"Evaluating {name}...")
        try:
            result = evaluate_cohort(name, cfg)
            typer.echo(f"  {name}: lift={result['lift']:+.3f} (best step {result['best_step']})")
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)


@app.command()
def analyze():
    """Assemble table, run validation, generate all plots."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .analyze import run_analysis
    run_analysis(cfg)


@app.command()
def sweep(n_gpus: int = typer.Option(8, help="Number of GPUs for parallel training")):
    """Run parallel sweep of remaining cohorts."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    from .rollout import load_foundation
    from .cohorts import load_cohort
    from .types import Cohort, TrainCfg
    from .train.trl_backend import TRLBackend
    from .train.sweep import run_sweep

    artifacts = Path(cfg.artifacts_dir)
    all_names = [p.stem.replace(".cohort", "") for p in (artifacts / "cohorts").glob("*.cohort.json")]

    cohorts = {}
    tasks_map = {}
    for name in all_names:
        try:
            c, t = load_cohort(name, cfg)
            cohorts[name] = c
            tasks_map[name] = t
        except Exception:
            pass

    train_cfg = TrainCfg(
        steps=cfg.train_steps, ckpt_every=cfg.ckpt_every, group_size=cfg.group_size,
        lr=cfg.learning_rate, kl_beta=cfg.kl_beta,
    )
    run_sweep(cohorts, tasks_map, backend=TRLBackend(), model_path=cfg.model_path,
               artifacts_dir=cfg.artifacts_dir, n_gpus=n_gpus, cfg=train_cfg)


@app.command()
def status():
    """Print completion status for all cohorts."""
    from dotenv import load_dotenv
    load_dotenv()
    cfg = _cfg()
    artifacts = Path(cfg.artifacts_dir)
    eval_dir = artifacts / "eval"
    train_dir = artifacts / "train"
    cohort_dir = artifacts / "cohorts"

    names = sorted(set(
        [p.stem.replace(".cohort", "") for p in cohort_dir.glob("*.cohort.json")]
        if cohort_dir.exists() else []
    ))
    if not names:
        typer.echo("No cohorts built yet. Run 'make cohorts'.")
        return

    typer.echo(f"\n{'Cohort':<20} {'Metrics':>8} {'Trained':>8} {'Evaluated':>10} {'Lift':>8}")
    typer.echo("-" * 60)
    for name in names:
        has_metrics = (cohort_dir / f"{name}.metrics.json").exists()
        has_train = (train_dir / name).exists() if train_dir.exists() else False
        has_eval = (eval_dir / f"{name}.json").exists() if eval_dir.exists() else False
        lift_str = ""
        if has_eval:
            with open(eval_dir / f"{name}.json") as f:
                lift_str = f"{json.load(f).get('lift', 0):+.3f}"
        typer.echo(
            f"{name:<20} "
            f"{'✓' if has_metrics else '✗':>8} "
            f"{'✓' if has_train else '✗':>8} "
            f"{'✓' if has_eval else '✗':>10} "
            f"{lift_str:>8}"
        )


if __name__ == "__main__":
    app()
