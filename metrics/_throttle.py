"""
Shared throttle for all Anthropic API calls.

Low-tier API keys have a concurrent-connection limit. The metric code fires
many requests in parallel via asyncio.gather, which trips a 429. This module
funnels every call through a global semaphore and adds bounded retry with
exponential backoff so the whole pipeline stays under the limit.

Tune concurrency without code changes:
    export ANTHROPIC_MAX_CONCURRENCY=3   # raise once you have a higher tier
"""
from __future__ import annotations  # PEP604 unions work on py3.9 (declared min)

import os
import asyncio
import random
import anthropic

_MAX_CONCURRENCY = int(os.environ.get("ANTHROPIC_MAX_CONCURRENCY", "3"))
# Semaphores are bound to the event loop they were created on. run_experiment.py
# calls asyncio.run() once per metric (a fresh loop each time), so we key the
# semaphore by the running loop and create a new one when the loop changes.
_sems: dict[int, asyncio.Semaphore] = {}
_client: anthropic.AsyncAnthropic | None = None


def _get_sem() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    sem = _sems.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        _sems[loop_id] = sem
    return sem


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            max_retries=0,  # we handle retries ourselves below
        )
    return _client


async def acreate(max_attempts: int = 8, **kwargs):
    """Throttled, retrying wrapper around client.messages.create(**kwargs)."""
    async with _get_sem():
        for attempt in range(max_attempts):
            try:
                return await get_client().messages.create(**kwargs)
            except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                status = getattr(e, "status_code", None)
                # Retry on rate limits / overloaded / transient connection errors only
                if isinstance(e, anthropic.APIStatusError) and status not in (429, 500, 503, 529):
                    raise
                if attempt == max_attempts - 1:
                    raise
                backoff = min(2 ** attempt, 30) + random.uniform(0, 1)
                await asyncio.sleep(backoff)
