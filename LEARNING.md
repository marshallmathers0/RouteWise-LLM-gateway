# LEARNING.md — RouteWise from A to Z

This is my interview-prep guide for RouteWise: what every part does, **why** it was built
that way, how the project came together day by day, the concepts underneath it in plain
English, and the questions I expect to be asked, with answers tied to the actual code
and the measured results.

Numbers quoted here come from the generated reports in `reports/`. If a report is
regenerated, re-check the numbers here.

---

## Part 1 — The project in one page

**Problem.** Companies send every LLM request to one strong, expensive model. That wastes
money (many questions are easy), wastes time (people ask the same thing in different
words), and is fragile (providers rate-limit, time out, and retire models — Groq retired
the Llama models this project originally targeted, mid-project).

**What RouteWise is.** An **OpenAI-compatible gateway**: an app changes one line
(`base_url`) and every request now flows through:

```
auth + rate limit -> cache eligibility -> exact cache -> semantic cache (embed -> guards -> verifier)
-> learned router (cheap or strong?) -> provider with retries / fallback / circuit breaker -> log -> response
```

**The two ML contributions.**
1. **Verified semantic cache.** Embedding search alone serves wrong answers to near-miss
   questions. RouteWise adds rule guards and a cross-encoder verifier.
   Result (adversarial test set): at the same ~56% hit rate, cosine-only caching served wrong
   answers **49.8%** of the time; the verified cache **12.8%**; on hand-built traps **7.1%**.
2. **Learned router.** A classifier predicts P(cheap model's answer is good enough) from the
   query text. A threshold chosen on validation decides cheap vs strong. Results: see
   `reports/router_eval.md` (summarized in Part 6, Q&A 12).

**30-second pitch.** "Sending every query to a strong LLM is expensive, and naive semantic
caches return wrong answers for similar-looking questions. I built an OpenAI-compatible
gateway with two ML components: a verified semantic cache that adds rule guards and a
cross-encoder on top of embedding search, and a learned router that predicts when a cheap
model is good enough. On an adversarial benchmark the verified cache cut wrong cache hits
from 50% to 13% at the same hit rate, and the router's cost/quality trade-off is measured
against always-strong, always-cheap, a length heuristic and random routing on a held-out
test set."

---

## Part 2 — Every component, and why

### 2.1 API layer — `routewise/api/`
* **FastAPI** because it gives request validation through Pydantic models, async I/O (the
  gateway mostly *waits* on providers) and automatic `/docs`.
* **OpenAI-compatible schema** (`schemas.py`): `/v1/chat/completions`, `/v1/models`, the
  OpenAI error shape (`{"error": {...}}`). A test points the *real* `openai` SDK at a live
  server to prove compatibility — that's the whole value of a gateway: zero client changes.
* **Extra `routewise` object** on each response: route, router probability, cache status
  with reason, matched query, similarity, verifier score, per-stage latency, estimated cost,
  provider, fallback flag. SDKs ignore unknown fields, so this is free observability.
* **Streaming** is *buffered* SSE: we get the full answer, then stream it. Honest trade-off:
  `stream=True` clients work, caching/fallback stay simple (a provider dying mid-stream
  can't be transparently replaced), but time-to-first-token doesn't improve.
* **Dependency injection** via `create_app(...)` parameters: tests pass fake providers,
  fakeredis, SQLite, a fixed router. No monkey-patching of globals needed.

### 2.2 The pipeline — `routewise/core/pipeline.py`
Pure Python, no FastAPI imports, so it can be unit-tested and reused. A `StageTimer`
measures every stage, which is how the dashboard shows p50/p95 *per stage*.
Model resolution: `auto` -> router; `cheap`/`strong` -> that tier; an explicit model name
-> that model first, then the rest of its tier as fallback.

### 2.3 Providers — `routewise/providers/`
* **One `httpx` client for every provider** (Groq, Gemini and OpenRouter all expose
  OpenAI-compatible endpoints). Vendor SDKs have hidden retry logic; using raw HTTP makes our
  reliability layer the only one, so its behaviour and latency numbers are real.
* **Error classification** is the important part: 429/5xx/timeouts/connection errors are
  *retryable*; 400/401/404 are not (they'll fail the same way again).
* **Per-model params** in config (e.g. `reasoning_effort: low` for gpt-oss) are merged into
  the request body.
* **Mock provider** (`ROUTEWISE_MOCK_PROVIDERS=1`) answers instantly — used to measure the
  gateway's own overhead in the load test.

### 2.4 Reliability — `routewise/reliability/`
* **Retry with exponential backoff + full jitter** (`retry.py`): delay is random in
  `[0, min(cap, base·2^attempt)]`. Jitter prevents a "thundering herd" of clients retrying in
  sync. If the provider's `Retry-After` is longer than our cap, we **don't wait** — we fail
  over to the next provider. (Found in a live demo: waiting cost ~18 s per request.)
* **Fallback chain** (`fallback.py`): each tier is an ordered list of (provider, model).
  Skip unconfigured providers and providers with an open circuit. Only *transient* errors
  count against provider health — a bad request shouldn't trip the breaker for everyone.
* **Circuit breaker** (`circuit_breaker.py`): CLOSED -> (N consecutive failures) -> OPEN ->
  (cooldown) -> HALF_OPEN -> one probe -> CLOSED or back to OPEN. State lives in Redis so
  all workers agree; the half-open "only one probe" rule uses `SET NX` with expiry.
  Breakers are keyed **per provider + model**, not per provider: in a live demo, gpt-oss-20b
  running out of quota opened a per-provider breaker and blocked a healthy gpt-oss-120b on
  the same provider. Rate limits are per model, so the breaker must be too.
* **Fallback order is a cost decision.** The live dashboard showed *negative* savings: cheap
  traffic was falling back to Gemini Flash-Lite, whose output price is ~4x the strong model's.
  The cheap chain now falls back to the strong Groq model first and to Gemini only last.
* **Token-bucket rate limiter** (`rate_limiter.py`): per API key, capacity = burst size,
  refill rate = sustained rate. The read-refill-spend step is one **Lua script** in Redis,
  so two concurrent requests can't both spend the last token.
* **Graceful degradation** (`state.py`): every Redis call falls back to in-memory state on
  error. Tests prove the gateway still serves when Redis is missing *and* when it dies mid-run.

### 2.5 Exact cache — `routewise/cache/exact.py`
Key = SHA-256 of (role + whitespace-normalized text for **every** message including the
system prompt, requested model, temperature, top_p, max_tokens, stop, response_format).
Not lowercased: case matters for code and acronyms. TTL 24 h bounds staleness. Empty or
truncated answers (`finish_reason="length"`) are never cached.

### 2.6 Cache eligibility — `routewise/cache/eligibility.py`
A cached answer is only right if the same question would get the same answer *now*:
* temperature > 0.3 (or unset = provider default ≈ 1) -> no cache (non-deterministic);
* time-sensitive words ("today", "latest", "current", "weather", ...) -> no cache;
* client opt-out header or `Cache-Control: no-cache`;
* **multi-turn** -> exact cache yes (the key hashes the whole conversation), semantic cache
  **no** ("what about the second one?" means nothing without context).

### 2.7 Verified semantic cache — `routewise/cache/semantic.py`, `guards.py`, `models.py`, `store.py`
* **Stage 1 (recall):** embed the query with `bge-small-en-v1.5` (384-d, 33M params, strong
  on retrieval benchmarks for its size, fast on CPU). Normalized vectors, so inner product =
  cosine. FAISS `IndexFlatIP` = exact search (perfect recall; approximate indexes only pay
  off at much larger scale). Keep top-5 above cosine **0.799** — chosen so 95% of true
  duplicates in the dev split pass.
* **Stage 2 (precision): rule guards** — cheap, explainable vetoes:
  numbers/years/versions (3.10 ≠ 3.1, "25k" = 25000, "two" = 2), negation parity, named
  entities (spaCy NER, case-insensitive matching), and word-order swaps ("5 miles to km" vs
  "5 km to miles").
* **Stage 3 (precision): cross-encoder verifier** (`quora-distilroberta-base`), serve only if
  P(duplicate) ≥ **0.904** (max F0.5 on dev).
* **Storage:** SQLite is the source of truth (it stores embeddings); FAISS is rebuilt from it
  at startup — index and DB can never drift apart. One FAISS index per *namespace*
  (hash of system prompt + model + params), so personas never share answers.
* **Failure modes handled:** model fails to load -> semantic cache disabled with a warning;
  lookup throws -> treated as a miss.

### 2.8 Learned router — `routewise/router/`
* **Features** (`features.py`): the bge-small query embedding (shared with the cache — loaded
  once) + 29 handcrafted features: log length, lines, digit ratio, count of numbers and math
  operators, LaTeX, code tokens, multiple-choice options, question words.
* **Never a feature:** the dataset `source` (asserted in code) and the answer-format suffix we
  add for grading (the router trains on the raw `query`, never on the `prompt`).
* **Models** (`eval/router_train.py`): logistic regression (+ a handcrafted-only ablation),
  LightGBM, MLP. Imbalance handled with `class_weight="balanced"`, `is_unbalance`, and
  balanced sample weights respectively. Hyperparameters by validation ROC-AUC.
* **Decision rule:** route cheap if P(cheap OK) ≥ threshold. The threshold is the cheapest
  point on the **validation** cost/quality curve that keeps ≥ 95% of always-strong quality.
* **Fallback:** no artifact or a broken one -> length-heuristic router, with a warning.

### 2.9 Storage and dashboard — `routewise/storage/`, `dashboard/`
* Every request is logged **after** the response is sent (FastAPI background task), so DB
  latency never adds to user latency; logging errors never fail a request. Only a hash of
  the client's API key is stored.
* SQLAlchemy makes the same code work on Postgres (Docker) and SQLite (local, Spaces).
* Each log row stores a *counterfactual* "what would always-strong have cost" for the same
  tokens, so the dashboard can plot savings over time.
* Streamlit playground (trap examples, side-by-side always-strong comparison, candidate table
  showing each guard's verdict) + dashboard (hit rates, cost vs counterfactual, p50/p95 per
  stage, route mix, fallbacks/errors, false-hit audit table).

### 2.10 Pricing — `config/routewise.yaml`
Free tiers cost $0, so every cost is an **estimate** at the providers' paid list prices
(checked 2026-11-11): gpt-oss-20b $0.075 / $0.30, gpt-oss-120b $0.15 / $0.60 per 1M
input/output tokens. Reasoning tokens are billed as output, and our token counts include them.
The 2× price gap caps the possible saving at ~50% — a real constraint I state up front.

---

## Part 3 — How it was built: seven days

1. **Day 1 — Gateway skeleton.** Repo, pinned dependencies, YAML + env config, OpenAI
   schemas, the `httpx` provider client with error classification, `/health`. Test with the
   real `openai` SDK against a live server from day one.
2. **Day 2 — Reliability + logging.** Retries with jitter, fallback chains, the circuit
   breaker (memory + Redis store), the Lua token bucket, exact cache, request logs, Docker
   Compose. 51 tests, all with mocked providers. Lesson: test the failure paths, not just
   the happy path (Redis down, all providers down, 429 with Retry-After).
3. **Day 3 — Semantic cache.** bge-small + FAISS + SQLite store, namespaces, TTL. Probing
   real models found the verifier scores "5 miles to km" vs "5 km to miles" at **0.97**
   (duplicate!) — so I added the word-order guard. Slow tests run the real models on traps.
4. **Day 4 — Cache evaluation.** Built 2,300 labelled pairs: QQP duplicates, QQP hard
   negatives mined with our own embedder, and an LLM-drafted near-miss set. Found and handled:
   QQP label noise concentrated in the top-cosine "negatives", verifier leakage (it was
   trained on QQP), and four guard bugs ("one" read as a number, case-sensitive entities...).
   Thresholds chosen on dev, reported on test with bootstrap CIs.
5. **Day 5 — Router data.** Queries + resumable, rate-limited collector + ground-truth
   graders + LLM judge with randomized order. A pilot changed the plan: the cheap model was
   ~100% on GSM8K/MMLU (nothing to learn), Gemini 2.5 had been retired, Groq had retired
   Llama, and Gemini 3.8 Flash allowed only 20 requests/day. Final setup: gpt-oss-20b vs
   gpt-oss-120b on a harder mix (MATH, MMLU-Pro), judged by Qwen (a third model family).
6. **Day 6 — Router training and evaluation.** Features, three model families + ablation,
   validation-only selection, cost-vs-quality curves vs four baselines and an oracle,
   per-source breakdown, leave-one-source-out generalization, inference latency.
7. **Day 7 — Product + docs.** Streamlit playground and dashboard, traffic simulator, demo
   data from real traffic, Locust load test with mocked providers (62 ms p50 overhead),
   Hugging Face Spaces bundle, ConceptBound integration guide, README / REPORT / this file.

---

## Part 4 — Concepts in plain English

**Embedding.** A list of numbers (here 384) that represents the meaning of a text. Texts
with similar meaning get vectors pointing in similar directions.

**Cosine similarity.** The cosine of the angle between two vectors: 1 = same direction,
0 = unrelated. With normalized vectors it's just the dot product, which is what FAISS
`IndexFlatIP` computes. It measures *topical* closeness, which is exactly why "capital of
India" vs "capital of Indonesia" can look close.

**Bi-encoder vs cross-encoder.** A bi-encoder encodes each text *separately* into a vector —
precompute once, compare millions quickly, but the two texts never "see" each other. A
cross-encoder reads *both texts together* in one transformer pass, so attention can compare
"3.10" with "3.12" token by token — far more accurate, but one forward pass per pair. Standard
pattern: bi-encoder to retrieve a few candidates, cross-encoder to verify/re-rank them.

**Hard negatives.** Non-matching pairs that *look* matching (high similarity). Easy negatives
("capital of India" vs "how do I bake bread") tell you nothing; a cache must be tested on
the confusing ones. I mined them by embedding QQP non-duplicates with our own embedder and
taking the highest-similarity ones.

**Precision / recall trade-off.** For a cache: precision = of the answers served from cache,
how many were right (false-hit rate = 1 − precision); recall = of the true duplicates, how
many we served (hit rate = savings). Lower thresholds -> more hits and more wrong hits.

**Threshold choice.** Choose on a dev/validation split, report on test — never tune on the
test set. I maximized **F0.5** for the cache (precision weighted 2× — a wrong cached answer
is worse than a missed saving) and "cheapest point keeping ≥ 95% quality" for the router.

**LLM-as-judge and position bias.** Using an LLM to grade answers when there's no answer key.
Known biases: preferring the first (or second) answer shown, longer answers, and its own
model family. Mitigations used: randomized A/B order per query (seeded, so reruns are stable),
a judge from a different family, a rubric that says length isn't merit, a reference answer as
a hint, a validation against ground truth, and a swap test that measures the bias directly.

**Class imbalance.** When one label dominates (e.g. "cheap is fine" most of the time), a
model can score high accuracy by always predicting it. Remedies: class weights / sample
weights, and metrics that aren't fooled (ROC-AUC, PR-AUC of the minority class, and here
the cost/quality curve).

**ROC vs PR curves.** ROC plots true-positive rate vs false-positive rate; its AUC is the
probability a random positive outranks a random negative, and it ignores class balance. PR
plots precision vs recall; it focuses on how clean your positive predictions are and is
more informative when the class you care about is rare. I report PR-AUC for *both* classes.

**Why `source` must not be a feature.** The source (GSM8K, MATH, ...) is strongly tied to the
label (competition math is hard). A router using it would look excellent offline and be
useless in production, where requests don't come labelled with a dataset name — classic
**target leakage**. The same applies to anything that reveals the source indirectly, like
the "Answer: <letter>" suffix we add for grading — so features use the raw query only.

**Token bucket.** A bucket holds up to C tokens and refills at R per second; each request
takes one. Allows bursts of C, caps the sustained rate at R.

**Circuit breaker.** After repeated failures, stop calling a dependency for a cooldown
(fail fast instead of piling up timeouts), then let one probe test recovery.

**Cache invalidation and TTL.** "There are only two hard things in CS: cache invalidation
and naming things." We avoid most invalidation by not caching what goes stale (time-sensitive
queries, high temperature) and bounding everything else with a TTL (24 h exact, 7 days
semantic). Versioned key prefixes (`rw:exact:v1:`) invalidate everything at once on format changes.

**p95 latency.** 95% of requests are faster than this. Averages hide the slow tail that users
actually notice; p50 is the typical request, p95/p99 the bad-but-common case.

---

## Part 5 — "Why not just use LiteLLM / GPTCache?"

**Honest answer:** for plumbing, they're better than what I'd build alone, and in production
I'd use LiteLLM's provider adapters rather than maintain my own. RouteWise is not trying to
replace them. The point of this project is the two parts those tools leave naive or leave to you:

* **GPTCache-style semantic caching** is typically "embedding similarity above a threshold".
  My evaluation shows why that is dangerous: tuned to the same hit rate, cosine-only caching
  served wrong answers ~50% of the time on hard negatives. The contribution is the
  *verification stage* (guards + cross-encoder) and, more importantly, **measuring** false
  hits with an ablation, confidence intervals and a category breakdown.
* **Routing** in gateways is usually rules or manual model choice. I trained and evaluated a
  router against baselines on a held-out set with a cost/quality curve — the evaluation
  method is the part that transfers to any real system.
* Building the plumbing myself (retries, breaker, token bucket) means I can explain each
  piece and its failure modes in depth, which matters more for an interview than a feature list.

---

## Part 6 — Interview questions with answers

1. **Walk me through a request.** Auth + token bucket -> eligibility -> exact cache (Redis
   hash) -> semantic cache (embed, FAISS top-5 ≥ 0.799, guards, verifier ≥ 0.904) -> router
   (P(cheap OK) vs validation threshold) -> fallback chain with retries and breakers ->
   response with `routewise` metadata -> log written in a background task.

2. **Why two caches?** Exact is ~free and perfectly safe (and works for multi-turn); semantic
   catches paraphrases but needs verification. Exact first, semantic only for single-turn.

3. **Why does a naive semantic cache fail?** Embeddings capture topic, not exact meaning.
   "Python 3.10" vs "3.12" scores 0.93 cosine; negations score ~0.96. Tuned to the same hit
   rate as RouteWise, cosine-only had a 49.8% false-hit rate on the adversarial set.

4. **What does each guard catch, and what does it cost?** From the guard table in the cache
   report: the entity guard catches the most QQP hard negatives but also rejects the most
   true duplicates (QQP "duplicates" often differ in a detail); numbers and negation catch
   most near-miss traps at almost no cost; word-order catches direction swaps the verifier misses.

5. **Why both guards and a verifier?** They fail differently. The verifier alone let 18.2%
   wrong answers through on the near-miss traps (it's fooled by swaps); adding guards cut that
   to 7.1%. The guards alone are blunt (low hit rate on real QQP duplicates).

6. **How did you pick the thresholds?** On the dev split only. Cosine 0.799 = keeps 95% of
   dev duplicates (recall stage). Verifier 0.904 = max F0.5 on dev (precision weighted 2×).
   Reported on test with 1,000-sample bootstrap CIs.

7. **What's wrong with evaluating the verifier on QQP?** It was trained on Quora duplicate
   questions, so QQP results are optimistic (leakage). The near-miss set isn't from QQP, so it's
   the fairer test. I report both and say so.

8. **How did you build hard negatives, and what went wrong?** Embedded all QQP validation
   questions with our embedder and took the highest-cosine non-duplicates. The very top was
   mostly label noise ("this video ?" vs "this video?"), so I dropped text-identical pairs and
   sampled from the top 5,000 instead of the top 1,000. An audit of 40 found ~10–22% still noisy.

9. **What biases does your near-miss set have?** It was drafted by an LLM and the same process
   designed the guards — so it probably flatters the guards. It needs human verification and,
   ideally, traps written by someone else. It's also small (100 test pairs): wide CIs.

10. **How is the router labelled?** Ground truth where possible (GSM8K number, MATH via
    symbolic equivalence with math-verify, MMLU-Pro letter): label 1 if cheap is right or both
    are wrong (paying for strong bought nothing). Otherwise a Qwen judge scores both answers
    1–10 in random order; label 1 if cheap ≥ strong − 1.

11. **How do you know the judge is any good?** Two checks in `reports/router_labels.md`: on
    ground-truth items, how often the judge prefers the correct answer (without seeing the
    key) and the AUC of its scores; and a swap test measuring how often the verdict flips with order.

12. **What did the router achieve?** See the headline in `reports/router_eval.md`: cost saved
    vs always-strong at the validation-chosen threshold, quality kept on test, compared with
    always-cheap, the length heuristic, random routing at equal cost, and an oracle. Always
    quote the quality number together with the saving.

13. **Why can't routing save more?** gpt-oss-120b costs only 2× gpt-oss-20b, so even routing
    everything to cheap saves at most ~50% (a bit more or less depending on output lengths).
    With a frontier model at 10–30× the price, the same router would save far more.

14. **Why `class_weight="balanced"`?** One label dominates; without reweighting the classifier
    leans toward always predicting it. Balancing makes errors on the rare "needs strong" class count.

15. **Why ROC-AUC for hyperparameters but cost for model selection?** ROC-AUC is a stable,
    threshold-free ranking metric for tuning. The *business* metric is cost at fixed quality,
    so the production model is the one cheapest on validation at ≥ 95% quality.

16. **What is leave-one-source-out and why report it?** Train without one source, test on it.
    It answers "does the router generalize to a query type it has never seen?" — usually much
    worse than in-distribution, which is an honest limitation of a 1,000-query dataset.

17. **Why did you change datasets mid-project?** A pilot showed gpt-oss-20b at 15/15 GSM8K and
    12/15 MMLU — nearly every label would be "cheap is fine" and always-cheap would beat any
    router. Real traffic mixes easy and hard queries, so I added MATH and MMLU-Pro. Pilot queries
    were excluded from the final dataset.

18. **How did you deal with free-tier rate limits?** Client-side throttling on requests *and*
    tokens per minute (Groq counts prompt + max_tokens up front), backoff on 429, clean stop on
    daily quotas, append-only outputs so any crash or quota stop just resumes.

19. **Explain your circuit breaker states.** CLOSED (normal) -> N consecutive transient
    failures -> OPEN (skip provider instantly) -> after cooldown HALF_OPEN (exactly one probe,
    via Redis `SET NX`) -> success: CLOSED, failure: OPEN again.

20. **Why does only a transient error trip the breaker?** A 400 means *this request* is bad;
    counting it would let one buggy client take a healthy provider offline for everyone.

21. **How does the token bucket work in Redis, and why Lua?** Read tokens + timestamp, add
    elapsed × rate (capped), spend one if available, write back — all inside one Lua script,
    which Redis runs atomically. Without it, two workers could both read "1 token left".

22. **What happens if Redis goes down?** Exact cache off; rate limiter and breaker fall back to
    per-process memory; the gateway keeps serving. Tested both "absent at startup" and
    "dies mid-run".

23. **How do you keep the system prompt from leaking cached answers across apps?** Exact key
    includes every message; semantic cache partitions FAISS indexes by a namespace hash of the
    system prompt + model + params. Tests check two personas never share a cache entry.

24. **What did the load test measure, and what did it show?** Gateway overhead with *mocked*
    providers: 62 ms p50 / 90 ms p95 per request with one user (embedding + FAISS + guards +
    verifier + router + logging on CPU). At 10 users, throughput tops out near 14 req/s on one
    worker and latency rises from queueing on the CPU-bound models.

25. **How would you scale it?** Multiple workers (state is already in Redis), a GPU or ONNX/
    quantized models for embedding and verification, batching verifier calls across requests,
    an approximate FAISS index (HNSW/IVF) past ~1M entries, Postgres for logs, and Redis
    vector search to share the semantic cache across instances.

26. **Why is streaming buffered?** Simplicity and correctness with caching and fallback: you
    can't cache or fail over an answer you've half-sent. Real token streaming would need
    stream-aware caching (cache after completion) and no mid-stream failover.

27. **How is cost computed, and why "estimated"?** Tokens × list price of the model that served
    the request (reasoning tokens included as output). Free-tier calls really cost $0; the
    estimate is what the same traffic would cost on the paid tier.

28. **How do you test something that calls paid APIs?** Every test mocks providers (fake
    clients or `respx` for HTTP), uses fakeredis (with Lua) and SQLite. Slow tests load the real
    embedding/verifier models but still never call an LLM. 127 fast tests + 6 slow.

29. **What would you improve next?** Human-verified trap sets; more router data across more
    model pairs; a router that predicts *quality per model* (multi-model routing) instead of a
    binary label; calibrated probabilities; execution-based grading for code (run MBPP tests in
    a sandbox) instead of a judge; real token streaming.

30. **What did running it live teach you that tests didn't?** Observability paid off twice: the
    dashboard's "cost saved" went negative, which exposed a fallback model pricier than the strong
    tier; and per-provider circuit breakers turned one model's quota exhaustion into a provider
    outage. Both were configuration/design issues no unit test was written to catch.
    **And the hardest bug?** Three were instructive: (a) math-verify silently returning
    False on Windows (its timeout uses multiprocessing) — every MATH answer would have been
    "wrong"; (b) the semantic cache never caching answers truncated by a small `max_tokens`,
    which looked like a guard failure in the demo; (c) another process on the same Groq
    organization consuming the shared tokens-per-minute quota, which looked like a collector bug.

---

## Part 7 — Things to be able to draw on a whiteboard

1. The **request flow** (Part 1 diagram), with where each latency stage is measured.
2. The **three-stage semantic cache**: bi-encoder -> FAISS top-k -> guards -> cross-encoder,
   with "recall stage" and "precision stages" labelled.
3. **Bi-encoder vs cross-encoder** boxes: two towers + cosine vs one tower over the pair.
4. The **circuit-breaker state machine** (3 states, 4 transitions).
5. The **token bucket** (bucket, refill arrow, request drain) and why Lua makes it atomic.
6. A **precision-recall curve** for the cache with the F0.5 operating point, and a
   **cost-vs-quality curve** for the router with the 95% line, the val-chosen point, random
   routing as a straight line and the oracle as a star.
7. The **data pipeline**: queries -> collect (cheap, strong) -> ground truth / judge -> labels
   -> train (val selection) -> test evaluation -> artifact loaded by the gateway.
8. **Fallback chain** per tier with breakers in front of each provider.
