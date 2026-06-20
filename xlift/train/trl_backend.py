from __future__ import annotations
import json
import os
from pathlib import Path

from .backend import TrainBackend
from ..types import Cohort, Task, TrainCfg
from ..verify import score


def _reward_fn_factory(verifier: str):
    """Build a TRL-compatible reward function for the given verifier."""
    def reward_fn(prompts, completions, **kwargs):
        answers = kwargs.get("answer", [])
        results = []
        for c, a in zip(completions, answers):
            text = c if isinstance(c, str) else c[-1]["content"]
            results.append(score(text, a, verifier))
        return results
    return reward_fn


class _LogCallback:
    """Minimal callback to write train_log.jsonl at each logging step."""

    def __init__(self, log_path: str, group_size: int):
        self.log_path = log_path
        self.group_size = group_size
        self._file = open(log_path, "w")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        mean_reward = logs.get("reward", logs.get("mean_reward", 0.0))
        reward_std = logs.get("reward_std", 0.0)
        # effective_ratio: logged by TRL as "reward_std" > 0 fraction — approximate here
        effective_ratio = logs.get("effective_ratio", float(reward_std > 0))
        entry = {
            "step": step,
            "mean_reward": round(float(mean_reward), 4),
            "reward_std": round(float(reward_std), 4),
            "effective_ratio": round(float(effective_ratio), 4),
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._file.close()


class TRLBackend(TrainBackend):
    def train_cohort(
        self,
        cohort: Cohort,
        tasks: list[Task],
        *,
        model_path: str,
        gpu_ids: list[int],
        cfg: TrainCfg,
        out_dir: str,
    ) -> str:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from transformers import TrainerCallback

        out_path = Path(out_dir)
        final_ckpt = out_path / f"checkpoint-{cfg.steps}"
        if final_ckpt.exists():
            print(f"Final checkpoint exists for {cohort.name}, skipping training.")
            return out_dir

        out_path.mkdir(parents=True, exist_ok=True)

        # Set GPU visibility for this process
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        # Build dataset with 'prompt' (chat list) + 'answer' columns
        data = [
            {
                "prompt": [
                    {"role": "system", "content": (
                        "Solve the math problem step by step. "
                        "End with the final answer on its own line in the form: #### <number>"
                    )},
                    {"role": "user", "content": t.question},
                ],
                "answer": t.answer,
            }
            for t in tasks
        ]
        ds = Dataset.from_list(data)

        # Identical config across all cohorts — cohort.verifier is the only variable
        grpo_cfg = GRPOConfig(
            output_dir=out_dir,
            num_generations=cfg.group_size,
            max_prompt_length=cfg.max_prompt,
            max_completion_length=cfg.max_completion,
            temperature=cfg.temperature,
            learning_rate=cfg.lr,
            beta=cfg.kl_beta,
            per_device_train_batch_size=cfg.per_device_bs,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.steps,
            loss_type="grpo",        # vanilla GRPO, no dynamic sampling
            scale_rewards=True,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.6,
            save_steps=cfg.ckpt_every,
            logging_steps=5,
            bf16=True,
            report_to="none",
            seed=cfg.seed,
        )

        log_path = str(out_path / "train_log.jsonl")

        class _CB(TrainerCallback):
            def __init__(self, log_file):
                self._f = open(log_file, "w")

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                entry = {
                    "step": state.global_step,
                    "mean_reward": round(float(logs.get("reward", 0.0)), 4),
                    "reward_std": round(float(logs.get("reward_std", 0.0)), 4),
                    "effective_ratio": round(float(logs.get("reward_std", 0.0) > 0), 4),
                }
                self._f.write(json.dumps(entry) + "\n")
                self._f.flush()

            def on_train_end(self, args, state, control, **kwargs):
                self._f.close()

        trainer = GRPOTrainer(
            model=model_path,
            reward_funcs=_reward_fn_factory(cohort.verifier),
            train_dataset=ds,
            args=grpo_cfg,
            callbacks=[_CB(log_path)],
            # full fine-tune: no peft_config
        )
        trainer.train()
        return out_dir
