from __future__ import annotations
import math
from typing import Optional
from ..types import Task, RolloutRecord
from ..verify import score


def single_step_grad_norm(
    model_path: str,
    tasks: list[Task],
    verifier: str,
    *,
    G: int = 8,
    n_prompts: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 512,
    foundation: Optional[list[RolloutRecord]] = None,
) -> float:
    """Compute a single-step GRPO gradient norm WITHOUT an optimizer step.

    Math (Appendix D):
        A_{t,i} = (r_{t,i} - mean_i r) / (std_i r + 1e-6)
        L = -(1/(n*G)) * sum_{t,i} A_{t,i} * sum_{token} log pi(token | ctx)
        grad_norm = sqrt(sum_p ||p.grad||^2)

    C6 with verifier='weak' → high grad_norm (hack is very learnable) despite low true lift.
    Foundation rollouts are reused where available (same tasks and G <= k_rollouts).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    sample_tasks = tasks[:n_prompts]
    foundation_map = {r.task_id: r for r in (foundation or [])}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()
    device = next(model.parameters()).device

    _SYSTEM = (
        "Solve the math problem step by step. "
        "End with the final answer on its own line in the form: #### <number>"
    )

    all_losses = []

    for task in sample_tasks:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": task.question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Get G completions — reuse foundation rollouts if available and sufficient
        rec = foundation_map.get(task.id)
        if rec and len(rec.rollouts) >= G:
            completions = [r.text for r in rec.rollouts[:G]]
            rewards = [score(c, task.answer, verifier) for c in completions]
        else:
            # Generate on the fly
            from vllm import LLM, SamplingParams
            llm = LLM(model=model_path, gpu_memory_utilization=0.5)
            sp = SamplingParams(n=G, temperature=temperature, max_tokens=max_tokens)
            out = llm.generate([prompt_text], sp)[0]
            completions = [o.text for o in out.outputs]
            rewards = [score(c, task.answer, verifier) for c in completions]
            del llm

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-6
        advantages = (rewards_t - mean_r) / std_r

        for i, (completion, adv) in enumerate(zip(completions, advantages.tolist())):
            if adv == 0.0:
                continue
            full_text = prompt_text + completion
            enc = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_enc = tokenizer(prompt_text, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            with torch.no_grad():
                labels = enc["input_ids"].clone()
                labels[:, :prompt_len] = -100
            logits = model(**enc).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            import torch.nn.functional as F
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, shift_labels.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            mask = shift_labels != -100
            completion_log_prob = (token_log_probs * mask).sum()
            loss = -adv * completion_log_prob / (n_prompts * G)
            all_losses.append(loss)

    if not all_losses:
        return 0.0

    total_loss = torch.stack(all_losses).sum()
    total_loss.backward()

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = math.sqrt(grad_norm)
    model.zero_grad()

    return grad_norm
