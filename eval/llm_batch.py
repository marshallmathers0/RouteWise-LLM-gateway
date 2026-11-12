"""Helpers for long, rate-limited batch runs against free-tier LLM APIs.

Free tiers enforce requests/minute (RPM), tokens/minute (TPM) and daily quotas. We
throttle client-side so we rarely hit 429s, back off when we do, and stop cleanly on a
daily quota so a later re-run resumes where it left off (outputs are append-only).
"""

from __future__ import annotations

import asyncio
import re
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from routewise.providers.base import LLMClient, LLMRequest, LLMResponse, ProviderError


class DailyLimitReached(Exception):
    """The provider's daily quota is exhausted: stop this worker and resume another day."""


class Throttle:
    """Sliding 60-second window over requests and tokens."""

    def __init__(self, rpm: float, tpm: float | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = asyncio.sleep) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._clock, self._sleep = clock, sleep
        self._events: deque[list] = deque()  # [time, tokens] (mutable so estimates can be corrected)

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60:
            self._events.popleft()

    def wait_time(self, est_tokens: int) -> float:
        now = self._clock()
        self._trim(now)
        waits = [0.0]
        if len(self._events) >= self.rpm:
            waits.append(60 - (now - self._events[0][0]))
        if self.tpm is not None:
            used = sum(t for _, t in self._events)
            if used + est_tokens > self.tpm and self._events:
                # Wait until enough old tokens fall out of the window.
                freed, need = 0, used + est_tokens - self.tpm
                for ts, tok in self._events:
                    freed += tok
                    if freed >= need:
                        waits.append(60 - (now - ts))
                        break
        return max(waits)

    async def acquire(self, est_tokens: int) -> list:
        """Wait for capacity, then reserve it. Returns a handle for record_actual()."""
        while (w := self.wait_time(est_tokens)) > 0:
            await self._sleep(w + 0.05)
        event = [self._clock(), est_tokens]
        self._events.append(event)  # type: ignore[arg-type]
        return event

    @staticmethod
    def record_actual(handle: list, actual_tokens: int) -> None:
        """Replace the reservation's estimate with the real token count once we know it."""
        handle[1] = actual_tokens


_DAILY_MARKERS = ("per day", "tokens per day", "requests per day", "rpd", "tpd", "daily", "quota")


def is_daily_limit(err: ProviderError) -> bool:
    if err.status_code != 429:
        return False
    msg = str(err).lower()
    return any(m in msg for m in _DAILY_MARKERS) or (err.retry_after_s or 0) > 600


_TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.I)


def parse_try_again_s(message: str) -> float | None:
    """Seconds from Groq-style "Please try again in 2m9.168s" (also "1h2m", "45.5s")."""
    m = _TRY_AGAIN.search(message)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s


async def call_rate_limited(client: LLMClient, model: str, req: LLMRequest, throttle: Throttle,
                            est_tokens: int, timeout_s: float = 60.0, max_attempts: int = 6,
                            sleep: Callable[[float], Any] = asyncio.sleep,
                            wait_on_daily_limit: bool = False, min_daily_wait_s: float = 60.0,
                            max_daily_wait_s: float = 1800.0) -> LLMResponse:
    """Throttled call with retries.

    Daily quota hit: raise DailyLimitReached, or - with wait_on_daily_limit - sleep until
    the provider says enough quota has freed up and try again (Groq's daily limits are a
    rolling 24 h window, so capacity trickles back continuously). Daily waits don't count
    as attempts, so an unattended run keeps going for as long as it takes.
    """
    attempt = 0
    while True:
        handle = await throttle.acquire(est_tokens)
        try:
            resp = await client.chat(model, req, timeout_s)
            throttle.record_actual(handle, resp.prompt_tokens + resp.completion_tokens)
            return resp
        except ProviderError as e:
            if is_daily_limit(e):
                if not wait_on_daily_limit:
                    raise DailyLimitReached(str(e)) from e
                wait = parse_try_again_s(str(e)) or e.retry_after_s or 300.0
                wait = min(max(wait, min_daily_wait_s), max_daily_wait_s)
                print(f"  [{model}] daily quota reached; waiting {wait / 60:.1f} min for it to refill", flush=True)
                await sleep(wait)
                continue
            attempt += 1
            if not e.retryable or attempt >= max_attempts:
                raise
            # Provider hint if given, else exponential: 10s, 20s, 40s ...
            await sleep(min(e.retry_after_s or parse_try_again_s(str(e)) or 10 * 2 ** (attempt - 1), 300))


@dataclass
class Progress:
    total: int
    label: str
    done: int = 0
    started: float = 0.0

    def __post_init__(self) -> None:
        self.started = time.monotonic()

    def tick(self, every: int = 10) -> None:
        self.done += 1
        if self.done % every == 0 or self.done == self.total:
            el = time.monotonic() - self.started
            eta = el / self.done * (self.total - self.done)
            print(f"[{self.label}] {self.done}/{self.total}  elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line half-written during a crash: ignore, it will be redone
    return rows
