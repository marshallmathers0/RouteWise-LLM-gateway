# RouteWise: technical report

**A cost-aware LLM gateway with a learned router and a verified semantic cache.**
All numbers below come from scripts in this repository; detailed tables, plots and
confidence intervals are in `reports/`.

## 1. Problem

Applications that call LLMs usually send every request to one strong model. Three costs follow:

* **Money.** Many requests are easy enough for a smaller, cheaper model.
* **Latency and repetition.** Users ask the same things in different words; each paraphrase
  pays for a full model call.
* **Reliability.** Free and paid providers rate-limit, time out, and retire models (during this
  project Groq retired its Llama 3.x models and Google closed Gemini 2.5 to new users).

Gateways such as LiteLLM, Portkey or Helicone solve the plumbing. Two parts are commonly
naive: **semantic caching**, which typically serves any cached answer above an embedding
similarity threshold, and **routing**, which is typically rule-based. This project builds the
plumbing to a production-like standard and focuses its ML effort on those two parts, with
honest offline evaluation.

## 2. System

RouteWise exposes an OpenAI-compatible `POST /v1/chat/completions`. Per request:
authentication and a per-key token-bucket rate limit (Redis, atomic Lua) -> cache eligibility
(temperature <= 0.3, not time-sensitive, no opt-out; semantic cache only for single-turn) ->
exact cache (Redis, SHA-256 of all messages and generation parameters) -> verified semantic
cache -> learned router -> provider call with retries (exponential backoff, full jitter),
a per-tier fallback chain and per-model circuit breakers -> logging of tokens, estimated
cost, per-stage latency and every decision.

Tiers used in the experiments: **cheap = `openai/gpt-oss-20b`**, **strong = `openai/gpt-oss-120b`**
(Groq, `reasoning_effort: low`), with Gemini models as gateway fallbacks. Costs are **estimates**
at paid list prices (USD per 1M input/output tokens: 0.075/0.30 vs 0.15/0.60), since the free
tiers used cost nothing. The 2x price ratio bounds any routing saving at roughly 50%.

## 3. Verified semantic cache

### 3.1 Method
1. **Recall stage.** Embed the query with `BAAI/bge-small-en-v1.5` (384-d), search a FAISS
   inner-product index (one index per namespace = hash of system prompt, requested model,
   stop sequences and response format) and keep the top 5 with cosine >= 0.799.
2. **Rule guards** (veto only): numbers/years/versions must match (with "25k" = 25000,
   "two" = 2); negation parity must match; named entities (spaCy `en_core_web_sm`, matched
   case-insensitively, digit-bearing spans ignored) must appear in both; a pair with identical
   words in a different order ("5 miles to km" / "5 km to miles") is rejected.
3. **Verifier.** `cross-encoder/quora-distilroberta-base` scores the pair; serve only if
   P(duplicate) >= 0.904.

Entries live in SQLite (the source of truth, including embeddings); the FAISS index is rebuilt
from it at start-up. Truncated answers are never cached, so `max_tokens` is not part of the
key; a cached answer is served only if it fits the caller's `max_tokens`.

### 3.2 Evaluation data
2,300 labelled pairs, split 50/50 into dev (choose thresholds) and test (report), stratified:

* 1,000 **QQP duplicates** (GLUE QQP validation).
* 1,000 **QQP hard negatives**: human-labelled non-duplicates with the highest similarity under
  our own embedder, sampled from the top-5,000 pool after removing text-identical pairs
  (the very top of the ranking was dominated by label noise).
* 200 **near-miss traps** in five categories (entity swap, number/year/version change, negation,
  unit/direction change, same topic but different question) and 100 true **paraphrases**,
  drafted with an LLM; labels pending human verification.

Metrics: hit rate (= recall over true duplicates) and false-hit rate (= 1 - precision over
served pairs). Operating points maximize F0.5 on dev (precision weighted 2x: a false hit serves
a wrong answer, a miss costs one LLM call).

### 3.3 Results (test split)

| Config | Hit rate | False-hit rate | Near-miss traps: false-hit rate |
|---|---|---|---|
| Cosine only (its own best threshold, 0.941) | 33.8% | 36.3% | 37.1% |
| Cosine only at the **same hit rate** as the full system (0.899) | 56.9% | 49.8% | 49.3% |
| Cosine + rules | 46.0% | 40.3% | 17.5% |
| Cosine + verifier | **70.5%** | **11.8%** | 18.2% |
| **Cosine + rules + verifier (deployed)** | 56.9% | 12.8% | **7.1%** |

Bootstrap 95% CI for the deployed configuration: hit rate 52.8–61.1%, false-hit rate 9.4–16.2%;
on the small trap set its false-hit rate CI is 0.0–17.9% (2 wrong out of 28 served).

Findings:
* At equal hit rate, adding verification cut the false-hit rate from 49.8% to 12.8%.
* On QQP, `cosine + verifier` dominates. But the verifier was **trained on Quora duplicate
  questions**, so this comparison is biased in its favour. On the near-miss traps (not from QQP),
  the verifier alone served 18.2% wrong answers versus 7.1% with the guards, mainly unit/direction
  swaps and negations that it scores as duplicates (e.g. "convert 5 miles to km" vs "convert 5 km
  to miles": verifier 0.96). We deploy all three stages and pay for it in hit rate.
* Per-category (test traps, deployed config): entity swaps 0%, negations 0%, number changes 0%,
  same-topic-different-question 5%, unit changes 5% wrongly served. bge-small already separates
  entity swaps ("capital of India" vs "Indonesia" scores only 0.71), so the hard near-misses
  are numbers, negation and direction.
* Guards are blunt on real user questions: the entity guard rejects 145 of 940 QQP duplicates
  above the recall threshold (often QQP "duplicates" that add a detail, where serving one cached
  answer would in fact be unsafe), and the verifier misses some true paraphrases ("What is 5 miles
  in kilometers?" vs "How many kilometers is 5 miles?" scored 0.006).
* Overhead on CPU (p50): embedding 11 ms, guards ~6 ms per candidate, verifier 17 ms per pair or
  72 ms for a batch of 5.

## 4. Learned router

### 4.1 Data and labels
* **Queries.** A 1,000-query pool mixing easy and hard sources (GSM8K 150, MATH levels 3–5 200,
  MMLU-Pro 250, Dolly 250, MBPP 150), of which a stratified 50% (500 queries, 70/15/15
  train/val/test by source) is used, to fit Groq's free-tier budget of 200k tokens per model per
  day. A pilot showed gpt-oss-20b at 15/15 on GSM8K and 12/15 on MMLU, so the originally planned
  GSM8K/MMLU-only mix would have given almost no "needs strong" labels; queries used in pilots are
  excluded.
* **Answers.** Every query sent once to each tier at temperature 0 (resumable, rate-limited
  collector; the router sees the raw query only, never the grading suffix added to the prompt).
* **Labels.** Ground truth for GSM8K (final number), MATH (symbolic equivalence via math-verify)
  and MMLU-Pro (letter): label 1 ("cheap is good enough") if the cheap answer is right or both are
  wrong. For Dolly and MBPP, a judge from a third model family (`qwen/qwen3.8-27b`) scores both
  answers 1–10 against a fixed rubric in a randomized, seeded order with the dataset's reference
  answer as a hint; label 1 if cheap >= strong - 1. The judge is validated against ground truth
  and with an order-swap test (`reports/router_labels.md`).

### 4.2 Model
Features: the bge-small query embedding plus 29 handcrafted features (length, lines, digits,
math operators, LaTeX, code tokens, multiple-choice options, question words). The dataset source
is never a feature. Candidates: logistic regression (also a handcrafted-only ablation), LightGBM,
and an MLP, each with class-imbalance handling; hyperparameters by validation ROC-AUC. Each
model's routing threshold is the cheapest point on the **validation** cost/quality curve that
keeps >= 95% of always-strong quality; the production model is the cheapest such model on validation.

### 4.3 Results
<!-- ROUTER_REPORT_START -->
_Pending: the router pipeline (`python scripts/run_router_pipeline.py`) is collecting answers
within the free-tier daily quota. This section is filled from `reports/router_eval.md` and
`reports/router_labels.md` when it completes._
<!-- ROUTER_REPORT_END -->

## 5. Gateway performance

Load test with Locust against the gateway with **mocked** providers (instant answers), one
uvicorn worker on a laptop CPU, semantic cache on, Redis off:

| Users | Throughput | Server-side overhead p50 / p95 (miss path) |
|---|---|---|
| 1 | 5.2 req/s | 62 ms / 90 ms |
| 10 | 14.0 req/s | 577 ms / 840 ms (queueing on CPU-bound model inference) |

The single-user number is the real per-request cost; under load the process is CPU-bound by
the two transformer models. Memory: ~1.06 GB resident with both models loaded.

## 6. Limitations

* **Estimated costs.** List prices applied to free-tier traffic; the 2x price gap between the
  tiers caps savings. With a frontier model as the strong tier the same method would save far more.
* **Small router dataset.** 500 queries (350 for training) because of free-tier quotas; test-set
  differences of a few points are within noise.
* **Judge bias.** Open-ended labels rely on an LLM judge; mitigated (different family, randomized
  order, rubric, reference answers) and measured, not eliminated.
* **Near-miss set.** LLM-drafted, not yet human-verified, small, and written by the same process
  that designed the guards, so it likely flatters them.
* **Verifier leakage.** The verifier was trained on Quora questions; QQP results are optimistic.
* **Guard revisions.** The guards were revised twice after error analysis (once on evaluation
  pairs from both splits), so test numbers may be slightly optimistic.
* **English only.** Embedder, verifier, spaCy model and time-sensitivity rules are English.
* **Buffered streaming.** `stream=True` works, but the answer is complete before streaming starts.

## 7. Future work

Human-verified and externally written trap sets; multi-model routing that predicts quality per
model rather than a binary label; calibrated probabilities; execution-based grading for code;
a fine-tuned verifier on near-miss data; ONNX/quantized models and cross-request batching for
throughput; Redis vector search to share the semantic cache across instances; real token streaming.
