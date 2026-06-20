"""
Unified rollout backend for the xLift metrics.

The whole xLift thesis is model-relative: BoundaryScore / RepairGain / GEPA lift
must be measured with the SAME model you intend to RL-train, otherwise they don't
predict that model's post-RL lift. So the metrics call one async function here,
and the backend is chosen at runtime:

    XLIFT_BACKEND=qwen   BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct   (validation runs, on GPU)
    XLIFT_BACKEND=claude                                        (cheap dev/smoke on a laptop)

The Qwen backend transparently micro-batches concurrent requests so the per-call
`asyncio.gather` style in the metric code still gets good GPU utilisation.
"""

import os
import asyncio
from typing import Optional

BACKEND = os.environ.get("XLIFT_BACKEND", "claude").lower()
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


async def generate(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    system: Optional[str] = None,
) -> str:
    """Generate a single completion from the active backend."""
    if BACKEND == "qwen":
        return await _qwen_engine().submit(prompt, max_tokens, temperature, system)
    return await _claude_generate(prompt, max_tokens, temperature, system)


# --------------------------------------------------------------------------- #
# Claude backend (cheap dev / smoke tests)
# --------------------------------------------------------------------------- #
async def _claude_generate(prompt, max_tokens, temperature, system) -> str:
    from metrics._throttle import acreate
    model = os.environ.get("CLAUDE_ROLLOUT_MODEL", "claude-haiku-4-5-20251001")
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = await acreate(**kwargs)
    return resp.content[0].text


# --------------------------------------------------------------------------- #
# Qwen (local HF transformers) backend with async micro-batching
# --------------------------------------------------------------------------- #
_engine_singleton = None


def _qwen_engine():
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = _QwenEngine(BASE_MODEL)
    return _engine_singleton


class _QwenEngine:
    """
    Lazily loads a HF causal LM and serves concurrent async requests by
    collecting them into batches and running one padded generate() per batch.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._queue: asyncio.Queue | None = None
        self._worker_task = None
        self._lock = asyncio.Lock()
        self.batch_size = int(os.environ.get("XLIFT_BATCH_SIZE", "16"))
        self.batch_window = float(os.environ.get("XLIFT_BATCH_WINDOW", "0.05"))

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[qwen] loading {self.model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # decoder-only generation
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        print(f"[qwen] loaded on {self._model.device}")

    async def _ensure_worker(self):
        # Bind queue/worker to the currently running loop (run_experiment.py
        # calls asyncio.run() once per metric, so this re-binds per loop).
        async with self._lock:
            running_worker = self._worker_task is not None and not self._worker_task.done()
            if running_worker:
                return
            self._queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._worker())

    async def submit(self, prompt, max_tokens, temperature, system) -> str:
        await self._ensure_worker()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((prompt, max_tokens, temperature, system, fut))
        return await fut

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = [first]
            # brief window to accumulate a batch
            await asyncio.sleep(self.batch_window)
            while len(batch) < self.batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                outputs = await loop.run_in_executor(None, self._run_batch, batch)
                for (_, _, _, _, fut), text in zip(batch, outputs):
                    if not fut.done():
                        fut.set_result(text)
            except Exception as e:  # noqa: BLE001 — propagate to all awaiters
                for *_unused, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _run_batch(self, batch) -> list[str]:
        import torch
        self._load()
        tok = self._tokenizer
        prompts, max_toks, temps, systems, _ = zip(*batch)

        chats = []
        for prompt, system in zip(prompts, systems):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            chats.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(self._model.device)

        gen_max = max(max_toks)
        temp = max(temps)  # one batch shares sampling params; rollouts use temp>0
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=gen_max,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=0.95 if temp > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen_only, skip_special_tokens=True)
