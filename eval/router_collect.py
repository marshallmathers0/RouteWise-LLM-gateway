"""Run every router query through the cheap and the strong model ONCE, saving to disk.

Output: data/router/responses.jsonl (append-only; one row per (id, tier) success).
Re-running skips everything already collected, so a crash, Ctrl+C or a daily quota just
means "run it again later". Failures go to data/router/collect_errors.jsonl and are
retried on the next run.

Each tier uses ONE fixed provider/model (no fallback to a different model): the labels
compare "this cheap model" vs "this strong model", so mixing in answers from another
model would corrupt them.

  python -m eval.router_collect --estimate        # counts, token/time estimate, live limit probe
  python -m eval.router_collect --limit 20        # small pilot
  python -m eval.router_collect                   # full run (resumable)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from eval.llm_batch import DailyLimitReached, Progress, Throttle, append_jsonl, call_rate_limited, read_jsonl
from routewise.core.config import Settings, provider_api_key
from routewise.providers.base import LLMClient, LLMRequest, ProviderError
from routewise.providers.openai_compat import OpenAICompatClient

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data" / "router" / "queries.jsonl"
RESPONSES = ROOT / "data" / "router" / "responses.jsonl"
ERRORS = ROOT / "data" / "router" / "collect_errors.jsonl"

# Generous caps: gpt-oss spends part of its output on (hidden) reasoning tokens, and a
# truncated answer would be graded wrong for the wrong reason. MATH needs the most.
MAX_TOKENS = {"gsm8k": 800, "math": 1500, "mmlu_pro": 800, "mmlu": 600, "dolly": 800, "mbpp": 800}


def est_tokens(q: dict) -> int:
    # Groq counts prompt + max_tokens against the tokens-per-minute limit when a request
    # starts (the "Requested" figure in its 429s), so reserve that much, not the expected use.
    return len(q["prompt"]) // 4 + MAX_TOKENS[q["source"]]


def pending(queries: list[dict], tier: str) -> list[dict]:
    done = {(r["id"], r["tier"]) for r in read_jsonl(RESPONSES)}
    return [q for q in queries if (q["id"], tier) not in done]


async def collect_tier(tier: str, provider: str, model: str, client: LLMClient, todo: list[dict],
                       throttle: Throttle, params: dict | None = None, wait_on_daily_limit: bool = False) -> str:
    prog = Progress(len(todo), f"{tier}:{model}")
    for q in todo:
        req = LLMRequest(messages=[{"role": "user", "content": q["prompt"]}], temperature=0,
                         max_tokens=MAX_TOKENS[q["source"]], extra=params or {})
        t0 = time.perf_counter()
        try:
            r = await call_rate_limited(client, model, req, throttle, est_tokens(q),
                                        wait_on_daily_limit=wait_on_daily_limit)
        except DailyLimitReached as e:
            print(f"[{tier}] daily quota reached after {prog.done} requests - re-run later to resume.\n  {e}")
            return "daily_limit"
        except ProviderError as e:
            append_jsonl(ERRORS, {"id": q["id"], "tier": tier, "model": model, "error": str(e)[:500],
                                  "ts": datetime.now(timezone.utc).isoformat()})
            prog.tick()
            continue
        append_jsonl(RESPONSES, {
            "id": q["id"], "tier": tier, "provider": provider, "model": model, "content": r.content,
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "tokens_estimated": r.tokens_estimated, "finish_reason": r.finish_reason,
            "latency_ms": round((time.perf_counter() - t0) * 1000), "ts": datetime.now(timezone.utc).isoformat(),
        })
        prog.tick()
    return "done"


def probe_limits(base_url: str, api_key: str, model: str) -> dict[str, str]:
    """One tiny request; returns any rate-limit headers the provider reports."""
    r = httpx.post(base_url.rstrip("/") + "/chat/completions", timeout=30,
                   headers={"Authorization": f"Bearer {api_key}"},
                   json={"model": model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5})
    info = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    info["status"] = str(r.status_code)
    if r.status_code != 200:
        info["body"] = r.text[:300]
    return info


def main() -> None:
    load_dotenv()
    cfg = Settings().load_app_config()
    default_cheap = f"{cfg.tiers['cheap'][0].provider}:{cfg.tiers['cheap'][0].model}"
    default_strong = f"{cfg.tiers['strong'][0].provider}:{cfg.tiers['strong'][0].model}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", default=default_cheap, help="provider:model")
    ap.add_argument("--strong", default=default_strong, help="provider:model")
    ap.add_argument("--tiers", default="cheap,strong")
    ap.add_argument("--rpm-cheap", type=float, default=25)
    ap.add_argument("--tpm-cheap", type=float, default=7500)
    ap.add_argument("--rpm-strong", type=float, default=25)
    ap.add_argument("--tpm-strong", type=float, default=7500)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pending queries per tier")
    ap.add_argument("--estimate", action="store_true", help="print estimate + probe limits, no full run")
    ap.add_argument("--wait", action="store_true",
                    help="on a daily quota, wait for it to refill and continue (unattended runs)")
    args = ap.parse_args()

    queries = read_jsonl(QUERIES)
    if not queries:
        raise SystemExit("no queries: run `python -m eval.router_queries` first")
    targets = {"cheap": args.cheap, "strong": args.strong}
    limits = {"cheap": (args.rpm_cheap, args.tpm_cheap), "strong": (args.rpm_strong, args.tpm_strong)}
    tiers = [t.strip() for t in args.tiers.split(",")]

    plan = {}
    for tier in tiers:
        todo = pending(queries, tier)[: args.limit]
        tokens = sum(est_tokens(q) for q in todo)
        rpm, tpm = limits[tier]
        minutes = max(len(todo) / rpm, tokens / tpm if tpm else 0)
        plan[tier] = todo
        print(f"{tier:6s} {targets[tier]:45s} pending {len(todo):5d}  ~{tokens / 1e3:,.0f}k tokens  "
              f">= {minutes:,.0f} min at {rpm:g} RPM / {tpm:g} TPM (before daily quotas)")

    if args.estimate:
        for tier in tiers:
            provider, model = targets[tier].split(":", 1)
            pc = cfg.providers[provider]
            key = provider_api_key(pc)
            if key is None:
                print(f"  probe {tier}: {pc.api_key_env} not set")
                continue
            print(f"  probe {tier}: {probe_limits(pc.base_url, key, model)}")
        return

    async def run() -> None:
        http = httpx.AsyncClient()
        jobs = []
        for tier in tiers:
            provider, model = targets[tier].split(":", 1)
            client = OpenAICompatClient(provider, cfg.providers[provider], http)
            if not client.is_configured():
                print(f"skipping {tier}: {cfg.providers[provider].api_key_env} not set")
                continue
            # Same request params (e.g. reasoning_effort) as the gateway uses for this model.
            params = next((t.params for chain in cfg.tiers.values() for t in chain
                           if t.provider == provider and t.model == model), {})
            jobs.append(collect_tier(tier, provider, model, client, plan[tier], Throttle(*limits[tier]), params,
                                     args.wait))
        print(await asyncio.gather(*jobs))
        await http.aclose()

    asyncio.run(run())


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
