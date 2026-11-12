# DEVLOG — How RouteWise was built, day by day

This is the complete development log of RouteWise: what was built, in what order, why each
technology was chosen, what broke, what the measurements said, and how the plan changed
when reality disagreed with it. It ends with a large set of interview questions and answers.

How to read it:

* **Part 1** is a short orientation (what the project is, the stack, the repo map).
* **Part 2** is the seven-day build log, step by step, with the reasoning for every decision.
* **Part 3** collects the measured results in one place.
* **Part 4** is interview preparation: questions with answers and reasoning.

The seven "days" group the work into logical stages in the order it actually happened. Every
number quoted here comes from a script in this repository (mostly under `eval/` and
`reports/`); nothing is estimated unless it says so.

---

## Part 1 — Orientation

### 1.1 What RouteWise is

RouteWise is an **OpenAI-compatible LLM gateway**. An application that already uses the OpenAI
SDK changes one line (`base_url="http://localhost:8000/v1"`) and from then on every chat request
goes through RouteWise, which:

1. checks whether the request may be answered from a cache at all;
2. answers from an **exact cache** (Redis) or a **verified semantic cache** (FAISS + SQLite) when
   it is safe to do so;
3. otherwise asks a **router** whether a cheap model is good enough, and sends the request to the
   cheap or the strong tier;
4. calls the provider with **retries, a fallback chain and circuit breakers**;
5. logs tokens, estimated cost, per-stage latency and every decision for a **dashboard**.

### 1.2 The two problems it focuses on

* **Naive semantic caching serves wrong answers.** If "What are the new features in Python 3.10?"
  is cached, a naive cache also serves it for "...Python 3.12?", because the two sentences are
  almost identical as embeddings (cosine 0.93). RouteWise adds rule guards and a cross-encoder
  verifier on top of the embedding search, and measures how many wrong answers that prevents.
* **Rule-based routing is a weak predictor of difficulty.** "Long query = hard query" is wrong
  often enough to matter. RouteWise trains a classifier to predict whether the cheap model's
  answer will be good enough, and evaluates it on cost-versus-quality against baselines.

### 1.3 Technology stack at a glance

| Layer | Technology | Why this one |
|---|---|---|
| Language | Python 3.12 | The ML ecosystem; type hints + Pydantic for safety |
| Web framework | FastAPI + Uvicorn | Async I/O for a service that mostly waits on HTTP; Pydantic validation; automatic `/docs` |
| Schemas / config | Pydantic v2, pydantic-settings, YAML | Validated config; secrets from env, behaviour from YAML |
| HTTP client | httpx (async) | One client for every OpenAI-compatible provider; full control over timeouts |
| Shared state | Redis (redis-py asyncio), Lua | Exact cache, rate limiter, circuit-breaker state across workers |
| Logs | SQLAlchemy 2 + Postgres or SQLite | Same code locally (SQLite) and in Docker (Postgres) |
| Embeddings | sentence-transformers, `BAAI/bge-small-en-v1.5` | Strong retrieval quality for 33M params; fast on CPU |
| Vector search | FAISS `IndexFlatIP` | Exact cosine search, in-process, no extra server |
| Cache entries | SQLite | Durable source of truth; FAISS rebuilt from it |
| Verifier | `cross-encoder/quora-distilroberta-base` | Trained for duplicate-question detection |
| Rule guards | spaCy `en_core_web_sm`, regexes | Small, explainable vetoes |
| Router models | scikit-learn (LogReg, MLP), LightGBM | Standard, interpretable, quick to train |
| Math grading | math-verify | Symbolic equivalence of LaTeX answers |
| Evaluation | pandas, numpy, matplotlib | Tables, bootstrap CIs, plots |
| UI | Streamlit + Plotly | A playground and dashboard in pure Python |
| Load testing | Locust | Standard Python load-testing tool |
| Tests | pytest, pytest-asyncio, fakeredis (with Lua), respx | No real API calls in tests |
| Packaging | Docker, Docker Compose | One command for app + Redis + Postgres + dashboard |
| Deployment target | Hugging Face Spaces (Docker) | Free CPU host with enough RAM for the models |
| LLM providers | Groq (gpt-oss-20b, gpt-oss-120b, qwen3.8-27b), Google Gemini | Free tiers with OpenAI-compatible endpoints |

### 1.4 Repository map

```
routewise/
  api/          FastAPI app factory, OpenAI-compatible schemas
  core/         config (env + YAML), pricing, the request pipeline
  providers/    LLMClient interface, OpenAI-compatible httpx client, mock client
  cache/        eligibility rules, exact cache, semantic cache, guards, models, FAISS+SQLite store
  router/       features, learned router, baseline routers
  reliability/  retry with jitter, fallback chain, circuit breaker, token bucket, state store
  storage/      SQLAlchemy request-log model and repository
eval/           cache evaluation, router data pipeline, router train/eval, load test
dashboard/      Streamlit playground + dashboard
scripts/        CLI demo, traffic simulator, demo-data builder, HF Space packager, pipeline runner
deploy/         Hugging Face Space Dockerfile and start script
reports/        generated evaluation reports and figures
docs/           integration guide for ConceptBound-Net
tests/          pytest suite
config/         routewise.yaml
start_demo.bat / stop_demo.bat   one-click Windows demo
```

---

## Part 2 — The seven-day build log

### Day 1 — Planning, and the skeleton of an OpenAI-compatible gateway

**Goal for the day:** agree the plan, set up a reproducible environment, and get a single request
to flow from the official OpenAI SDK, through RouteWise, to a (mocked) provider and back.

#### Step 1.1 — Phase 0: plan before code

Before writing code I wrote down the plan, the folder structure, the dependency list and the
decisions that needed the project owner's input. Several parts of the original brief were
changed at this stage, each for a stated reason:

1. **One HTTP client instead of three vendor SDKs.** Groq, Gemini and OpenRouter all expose
   OpenAI-compatible `/chat/completions` endpoints. Vendor SDKs have their own hidden retry
   logic; if they retry silently, my reliability layer and my latency measurements would both be
   wrong. One `httpx` client, configured per provider, keeps a single source of truth.
2. **The verifier has a leakage problem.** The planned verifier, `quora-distilroberta-base`, was
   trained on Quora Question Pairs. Evaluating it on QQP flatters it, so the plan made a separate,
   non-QQP near-miss set the primary test of the verifier.
3. **One logging layer for two databases.** SQLAlchemy lets the same code write to Postgres
   (Docker) and SQLite (local, Hugging Face Spaces).
4. **Redis must be optional at runtime.** If Redis is down, the gateway keeps serving: exact cache
   off, rate limiting and breaker state in process memory.
5. **One quality scale for the router curve.** Ground-truth items score 0 or 1; judged items score
   judge-score / 10.
6. **Judge from a different model family than the tiers**, to reduce self-preference bias.
7. **Hugging Face Spaces over Render for free hosting**, because the models need more RAM than
   Render's free 512 MB instance.

Why plan first: the expensive parts of this project (API quotas, labelled data) cannot be redone
cheaply, so mistakes in the plan are costly. Writing the plan down also made every later change a
visible, explainable decision rather than drift.

#### Step 1.2 — Environment and pinned dependencies

* Python 3.12 in a virtual environment (`.venv`).
* Dependencies installed, then **pinned with `==`** in `requirements.txt` after confirming they
  install on Windows. Pinning makes the project reproducible: a clone in six months gets the same
  versions that the tests passed with.
* `git init`, `.gitignore` (virtualenv, caches, `.env`, local databases).
* Later, `.gitattributes` with `* text=auto eol=lf` so files are stored with Unix line endings,
  plus `*.bat text eol=crlf` because Windows batch files need CRLF.

#### Step 1.3 — Configuration: secrets from the environment, behaviour from YAML

`routewise/core/config.py` has two layers:

* `Settings` (pydantic-settings) reads **secrets and infrastructure** from environment variables
  and `.env`: provider API keys, the gateway's own client keys (`ROUTEWISE_API_KEYS`),
  `REDIS_URL`, `DATABASE_URL`.
* `AppConfig` (Pydantic) validates **behaviour** from `config/routewise.yaml`: providers, tier
  fallback chains, the price table, reliability settings, rate limits, cache thresholds, router
  artifact path, logging options.

Why split them: secrets must never be committed; behaviour should be reviewed in code review like
any other change. A Pydantic model for the YAML means a typo in the config fails at start-up with a
clear error instead of misbehaving at runtime.

Provider API keys are read at **call time**, not import time, so a key added to `.env` (or changed
in a test with `monkeypatch`) is honoured without restarting Python.

#### Step 1.4 — OpenAI-compatible schemas

`routewise/api/schemas.py` models the subset of the Chat Completions API that matters:

* `ChatCompletionRequest`: `model`, `messages`, `temperature`, `top_p`, `max_tokens`,
  `max_completion_tokens`, `stop`, `n`, `stream`, `user`, and later `response_format`.
* `extra="allow"` on the request so unknown OpenAI fields never cause a rejection.
* Message content may be a string or a list of content parts; `text()` flattens it.
* Helpers: `last_user_text()` (what the cache and router look at) and `is_single_turn()`.
* `ChatCompletionResponse` with the standard fields plus a **`routewise` metadata object**: route,
  route reason, router probability, cache status and reason, matched query, similarity, verifier
  score, guard results, per-stage latency, estimated cost, provider, model, fallback flag.

Why the extra metadata object: SDKs ignore unknown fields, so every client gets full observability
for free, and the OpenAI SDK exposes it via `response.model_extra["routewise"]`.

Errors use the OpenAI shape (`{"error": {"message", "type", "code"}}`) so the SDK raises its normal
exception types (`AuthenticationError`, `BadRequestError`, ...).

#### Step 1.5 — The provider client

`routewise/providers/openai_compat.py` is one `OpenAICompatClient` used for every provider.

* It posts to `{base_url}/chat/completions` with a per-request timeout.
* **Error classification** is the most important part:
  * retryable: HTTP 408, 409, 425, 429, 500, 502, 503, 504, timeouts, connection errors, and
    malformed 200 responses (free tiers occasionally return them under load);
  * not retryable: 400, 401, 403, 404 and other client errors — they will fail the same way again.
* It parses `Retry-After` when the provider sends a number of seconds.
* If the provider omits token usage, tokens are estimated at ~4 characters per token and the
  response is flagged `tokens_estimated`.
* `is_configured()` is false when the API key is missing; such providers are skipped by the
  fallback chain instead of failing.

`routewise/providers/base.py` defines the `LLMClient` protocol, `LLMRequest` (messages, sampling
parameters and an `extra` dict for provider-specific fields) and `ProviderError` (with
`retryable`, `status_code`, `retry_after_s`).

Why a protocol: tests can pass a fake client with the same shape, and an Ollama or vLLM client
can be added later without touching the pipeline.

#### Step 1.6 — The pipeline, separate from the web framework

`routewise/core/pipeline.py` contains the `Gateway` class: the whole request flow, with **no
FastAPI imports**. A `StageTimer` context manager records the duration of every stage
(`eligibility`, `exact_cache`, `semantic_cache` and its sub-stages, `router`, `llm`, `total`).

Model resolution:

* `model="auto"` → the router decides the tier;
* `model="cheap"` or `"strong"` → that tier's chain;
* an explicit model name → that model first, then the rest of the tier it belongs to as fallback;
* anything else → HTTP 400 with code `model_not_found`.

Why keep the pipeline framework-free: it can be unit-tested directly, reused by scripts, and the
API layer stays thin (auth, rate limit, serialization, background logging).

#### Step 1.7 — The FastAPI app factory

`routewise/api/app.py` exposes `create_app(...)`. Every dependency — settings, config, provider
clients, Redis, the log repository, the router, the semantic cache — can be injected. Tests use
this to pass fakes; production builds everything from settings inside the `lifespan` handler.

Endpoints: `POST /v1/chat/completions`, `GET /v1/models` (lists `auto`, the tiers and the
configured models), `GET /health` (Redis, database, semantic cache status and entry count).

Day-1 result: the official `openai` Python client, pointed at a live uvicorn server running the
app with a fake provider, could create completions and receive RouteWise metadata.

---

### Day 2 — Reliability, rate limiting, the exact cache, logging and Docker

**Goal for the day:** make the gateway behave correctly when things go wrong, and make every
request observable.

#### Step 2.1 — Retries with exponential backoff and full jitter

`routewise/reliability/retry.py`:

* `backoff_delay(attempt) = uniform(0, min(max_s, base_s * 2**attempt))` — "full jitter".
* Plain exponential backoff makes clients that failed together retry together (a thundering
  herd); randomizing the whole interval spreads them out.
* Only `retryable` errors are retried, at most `max_retries` times (default 2, i.e. 3 attempts).
* A provider's `Retry-After` hint is respected. Later (Day 7) a rule was added: if the hint is
  **longer than our backoff cap**, give up on this provider immediately and let the fallback chain
  move on — waiting would only add latency.

#### Step 2.2 — The fallback chain

`routewise/reliability/fallback.py` walks a tier's ordered list of `(provider, model, params)`:

* skip providers without an API key (`skipped_unconfigured`);
* skip targets whose circuit breaker is open (`skipped_circuit_open`);
* call with retries; on success record success and return;
* on failure record a breaker failure **only for transient errors** and try the next target;
* if every target fails, raise `AllProvidersFailed` → HTTP 503 with a per-target summary.

`fallback_used` is true whenever the answer did not come from the tier's first choice.

Why only transient errors count against the breaker: a 400 means this particular request is bad.
If it counted, one buggy client could take a healthy provider offline for everybody.

#### Step 2.3 — The circuit breaker

`routewise/reliability/circuit_breaker.py` implements the classic state machine:

```
CLOSED --(N consecutive failures)--> OPEN --(cooldown elapsed)--> HALF_OPEN
  ^                                    ^                              |
  |                                    +---------(probe fails)--------+
  +----------------------(probe succeeds)-----------------------------+
```

* Defaults: open after 5 consecutive transient failures, 30 s cooldown.
* In HALF_OPEN exactly **one probe** is allowed through, enforced with a `SET NX` lock that
  expires after one cooldown (so a probe that hangs forever cannot wedge the breaker).
* State is stored through a small `StateStore` interface with a Redis implementation and an
  in-memory implementation, wrapped in `FallbackStateStore` that uses memory whenever Redis errors.

Why a breaker at all: when a provider is down, every request would otherwise wait for timeouts and
retries before falling back. An open breaker skips the dead provider instantly.

(Day 7 changed the breaker **key** from the provider to provider + model; see Step 7.4.)

#### Step 2.4 — The token-bucket rate limiter

`routewise/reliability/rate_limiter.py`:

* One bucket per client API key; capacity 30 (burst), refill 0.5 tokens/s (30 requests/minute).
* In Redis, read-refill-spend runs as **one Lua script**, which Redis executes atomically, so two
  gateway workers cannot both spend the last token.
* Idle buckets expire automatically.
* If Redis errors, an in-memory bucket is used for that call.
* A rejected request gets HTTP 429 with an OpenAI-style `rate_limit_error` and a `Retry-After`
  header.

Why a token bucket rather than a fixed window: it allows short bursts while capping the sustained
rate, and it has no "double burst" at window boundaries.

#### Step 2.5 — Authentication

A simple bearer-token check against `ROUTEWISE_API_KEYS`. The raw key is never stored anywhere;
logs and rate-limit buckets use a 12-character SHA-256 prefix (`api_key_id`).

#### Step 2.6 — Cache eligibility

`routewise/cache/eligibility.py` decides whether caching is allowed at all:

* `X-RouteWise-Cache: off` or `Cache-Control: no-cache/no-store` → bypass (`client_opt_out`).
* `n > 1` → bypass.
* temperature above 0.3 → bypass. An **unset** temperature means the provider default (about 1.0),
  so it is treated as non-deterministic too.
* Time-sensitive wording ("today", "latest", "current", "weather", "news", "stock price", ...) →
  bypass. The list is deliberately conservative: a false alarm only costs a cache miss.
* Multi-turn conversations: exact cache allowed (its key covers the whole conversation), semantic
  cache not allowed ("what about the second one?" means nothing without the context).

#### Step 2.7 — The exact cache

`routewise/cache/exact.py`:

* Key = SHA-256 of a canonical JSON of: every message's role and whitespace-normalized text
  (including system prompts), the requested model, temperature, top_p, stop sequences and
  (added later) `response_format`.
* Text is **not lowercased** — case matters for code and acronyms; paraphrase matching is the
  semantic cache's job.
* Values are stored in Redis with a TTL (24 h). A versioned key prefix (`rw:exact:v2:`) allows
  invalidating everything at once after a format change.
* Redis errors are treated as misses.
* Empty answers and answers truncated by `max_tokens` (`finish_reason == "length"`) are never
  cached: serving a cut-off answer to every future request would multiply one bad response.

#### Step 2.8 — Request logging

`routewise/storage/` defines a `request_logs` table and a repository:

* columns for request id, timestamp, hashed key, prompt hash, (optional, truncated) query text,
  requested model, route and reason, router probability, model, provider, fallback flag, cache
  status and reason, matched query, similarity, verifier score, guard results, tokens in/out,
  total latency and the per-stage latency JSON, estimated cost, **counterfactual always-strong
  cost**, status code and error;
* writes happen in a FastAPI **background task**, after the response is sent, so database latency
  never adds to user latency;
* write failures are logged and swallowed — logging must never take the gateway down.

Why the counterfactual cost column: the dashboard can plot "what we spent" against "what sending
everything to the strong model would have cost" without recomputing anything.

#### Step 2.9 — Pricing

`routewise/core/pricing.py` multiplies token counts by a USD-per-1M-tokens table from the YAML.
Free tiers cost nothing, so every cost is an **estimate** at the providers' paid list prices, and
the API metadata says so (`cost_is_estimate: true`).

#### Step 2.10 — Docker

* `Dockerfile`: Python 3.12 slim, **CPU-only PyTorch** installed first (the default wheel bundles
  CUDA and is gigabytes larger), the embedding and verifier models downloaded **at build time** so a
  cold start does not download them.
* `docker-compose.yml`: the app, Redis 7, Postgres 16, and later a Streamlit dashboard service,
  with health checks and a start period for model loading.
* Honest status: Docker Desktop was not running on the development machine, so the compose stack
  has not yet been run end to end. Everything else was verified by running it locally without
  Docker (gateway + SQLite + optional Redis).

#### Step 2.11 — Tests for the failure paths

The Day-2 test suite (51 tests) used fake providers, fakeredis with Lua support, `respx` for HTTP
mocking and SQLite in a temp directory. Highlights:

* the real `openai` SDK against a live uvicorn server (normal, streaming, 401, 400);
* every circuit-breaker transition against both the memory store and the Redis store;
* retry behaviour: transient errors retried, client errors not, `Retry-After` respected;
* the token bucket: burst, refill, per-key isolation, in memory and in Redis;
* exact-cache key correctness: different system prompts never collide;
* the gateway still serving when Redis is absent at start-up **and** when it dies mid-run;
* all providers failing → 503 that is still logged.

A smoke test of the real entry point (`uvicorn routewise.main:app`) without any API keys returned a
clean 503 listing `skipped_unconfigured` for every provider — the expected behaviour.

**Lesson of the day:** the happy path is the smallest part of a gateway. Most of the code and most
of the tests are about rate limits, outages, missing dependencies and bad input.

---

### Day 3 — The verified semantic cache

**Goal for the day:** a semantic cache that serves real paraphrases and refuses near-misses.

#### Step 3.1 — Probe the models before designing around them

Before writing the cache I loaded the embedding model and the verifier and scored a handful of
trap pairs. These numbers shaped the design:

| Pair | Cosine (bge-small) | Verifier P(duplicate) |
|---|---|---|
| "capital of India?" vs "capital of Indonesia?" | 0.713 | 0.000 |
| "capital of India?" vs "Which city is India's capital?" | 0.934 | 0.909 |
| "Python 3.10 features" vs "Python 3.12 features" | 0.931 | 0.000 |
| "convert 5 miles to km" vs "convert 5 km to miles" | **0.990** | **0.965** |
| "Is coffee bad for health?" vs "...not bad..." | 0.962 | 0.279 |

Two findings:

1. **The verifier is fooled by swapped words.** It mostly learned lexical overlap, so "5 miles to
   km" and "5 km to miles" look like duplicates to it. That motivated a **word-order guard**.
2. **India vs Indonesia is not actually hard for bge-small** (0.71 cosine). The really dangerous
   near-misses are numbers, negations and direction swaps, which all score above 0.93.

Why probe first: it is cheap, and it turned a guess ("the verifier will catch everything") into a
measured gap with a specific fix.

#### Step 3.2 — The embedding and verifier wrappers

`routewise/cache/models.py`:

* `SentenceTransformerEmbedder` returns **L2-normalized** float32 vectors, so the inner product is
  the cosine similarity. No BGE "query instruction" prefix: that prefix is meant for
  query-to-passage retrieval, while here we compare query to query (a symmetric task).
* `CrossEncoderVerifier` returns P(duplicate) via the model's built-in sigmoid.
* Both are behind small protocols (`Embedder`, `Verifier`) so tests can use fake models.

Why bge-small over all-MiniLM-L6-v2: similar size and speed on CPU, better retrieval quality on
public benchmarks; 384 dimensions keep the index small.

#### Step 3.3 — The store: SQLite as truth, FAISS as an index

`routewise/cache/store.py` (`FaissSqliteStore`):

* Every entry (query, answer, model, provider, token counts, route, created/expiry time and the
  **embedding bytes**) is a row in SQLite.
* The FAISS index lives in memory and is **rebuilt from SQLite at start-up**. Index and database
  cannot drift apart after a crash, because there is only one source of truth.
* `IndexIDMap2(IndexFlatIP)` — exact search. Approximate indexes (IVF, HNSW) only pay off far beyond
  the size of a single-service cache, and exact search has perfect recall.
* **One index per namespace.** The namespace hashes everything except the user's question that
  changes what a good answer is: system prompts, requested model, stop sequences and response format.
  A pirate-persona answer is never served to a lawyer-persona request.
* **Lazy TTL expiry** (7 days): expired entries found during a search are removed from the index
  and the database.
* A lock guards FAISS and SQLite, because lookups run in worker threads.

#### Step 3.4 — The rule guards

`routewise/cache/guards.py` — each guard can only **veto** a candidate, never create a hit, so a
wrong guard decision costs a cache miss, not a wrong answer:

* **numbers**: extract numbers, years and version strings ("3.10" stays distinct from "3.1"),
  ordinal suffixes, spelled-out numbers ("two" = 2) and "k" suffixes ("25k" = 25000); the two
  queries must mention the same set.
* **negation**: count negation words (with "n't" expanded); the two queries must have the same
  parity (so "not uncommon" style double negation is neutral).
* **entities**: named entities from spaCy's small English model; every entity in each query must be
  mentioned (case-insensitively, whole word) in the other query's text.
* **word order**: identical bags of words in a different order are rejected.

`run_guards()` returns which guards passed, which the response metadata and the playground show.

#### Step 3.5 — The three-stage lookup

`routewise/cache/semantic.py`:

1. embed the query; search the namespace's index for the top 5; keep candidates with cosine at or
   above a **loose** threshold (recall stage);
2. run the guards on each candidate, in similarity order (precision stage);
3. score the surviving candidates with the verifier **in one batch**; accept the most similar one at
   or above the verifier threshold (precision stage).

Even on a miss, the closest candidate and the stage that rejected it are reported
(`semantic_rejected:guard:numbers`, `semantic_rejected:verifier`), which is what makes the
playground self-explanatory. Every stage can be switched off in the config, which is what the
ablation study needs.

#### Step 3.6 — Wiring it into the gateway safely

* The lookup is CPU-bound model inference, so it runs in `asyncio.to_thread`; the event loop keeps
  serving other requests.
* If the embedder fails to load, the semantic cache is disabled with a warning.
* If the **verifier** fails to load, the semantic cache is also disabled: without the verifier, more
  near-miss false hits would get through, and serving no cached answers is safer than serving wrong
  ones.
* Any exception during a lookup is logged and treated as a miss.
* After an LLM answer, the query, its embedding (already computed during the lookup) and the answer
  are added to the store.

Day-3 tests (95 in total) included fake-model tests of every stage and threshold, the store's
persistence and expiry, namespace isolation, the failure-to-load paths, and six slow tests that load
the real models and check the trap pairs are rejected and true paraphrases accepted.

---

### Day 4 — Measuring the cache honestly

**Goal for the day:** replace anecdotes with an ablation study on held-out data.

#### Step 4.1 — Building the evaluation pairs

`eval/build_cache_pairs.py` produced 2,300 labelled pairs:

* **1,000 QQP duplicates**, sampled from the GLUE QQP validation split.
* **1,000 QQP hard negatives**: human-labelled *non*-duplicates with the highest similarity under
  our own embedder. All 73,276 unique validation questions were embedded and every labelled pair
  scored.
* **200 near-miss traps** in five categories (entity swap, number/year/version change, negation,
  unit or direction change, same topic but a different question) and **100 true paraphrases**,
  drafted with an LLM and stored with `verified: false` because a human still has to check them.

The pairs were split 50/50 into **dev** (to choose thresholds) and **test** (to report), stratified
by source, category and label.

#### Step 4.2 — Finding label noise where we mined

The first mining run took the top 1,000 negatives by cosine. Reading them showed many were really
duplicates with a QQP label error ("What do you think of this video ?" vs "...video?"). Label noise
concentrates exactly at the top of a similarity ranking. The fix:

* drop pairs whose normalized text is identical;
* sample 1,000 from the **top 5,000** instead of taking the top 1,000 (median cosine of the sample
  0.90).

An audit of 40 sampled hard negatives judged 4 to be clear duplicates and 5 borderline, so roughly
10–22% noise remains. That audit is saved in `eval/data/qqp_hard_negative_audit.jsonl` and the
report states it — it puts a floor under every configuration's QQP false-hit rate.

#### Step 4.3 — The evaluation script

`eval/cache_eval.py`:

* scores every pair once with the embedder and the verifier (cached to disk) and recomputes the
  guards on each run (they are cheap and were being iterated on);
* four configurations: cosine only, cosine + rules, cosine + verifier, all three;
* the **candidate threshold** is the highest cosine that still keeps 95% of dev duplicates
  (0.799);
* each configuration's free knob (cosine threshold, or verifier threshold when a verifier is used)
  is chosen to maximize **F0.5 on dev** — precision weighted twice as much as recall, because a wrong
  cached answer is worse than a missed saving;
* everything is reported on **test** with 1,000-sample bootstrap confidence intervals, a
  precision-recall plot, a per-category bar chart, a per-guard table and latency measurements.

Metrics, in cache language: **hit rate** = served duplicates / all duplicates (recall);
**false-hit rate** = served non-duplicates / all served (1 − precision).

#### Step 4.4 — The first result exposed guard bugs

The first run showed the guards rejecting many true QQP duplicates. Reading examples found real
bugs, not just strictness:

* "one" in "how can one live a simple life" was parsed as the number 1;
* "25k" did not equal "25000";
* a number mentioned twice was counted twice;
* entity matching was case-sensitive ("Instagram" vs "instagram");
* every proper noun was treated as an entity ("President", "New Year").

Fixing them was generic and was covered by regression tests. The full system went from a 48.9% to a
56.9% hit rate, and its false-hit rate fell from 14.3% to 12.8%. The report discloses that this
error analysis looked at pairs from both splits, so test numbers may be slightly optimistic.

Some rejections were *correct*: QQP's notion of "duplicate" is "same intent", which includes pairs
like "best place to buy iPhone" vs "...iPhone 6 Plus". For a cache, serving one answer for both is
not safe, so the guards are right to refuse.

#### Step 4.5 — Results

Test split (thresholds from dev):

| Config | Hit rate | False-hit rate | Traps: false-hit rate |
|---|---|---|---|
| Cosine only (best threshold 0.941) | 33.8% | 36.3% | 37.1% |
| Cosine only at the same hit rate as the full system (0.899) | 56.9% | 49.8% | 49.3% |
| Cosine + rules | 46.0% | 40.3% | 17.5% |
| Cosine + verifier | 70.5% | 11.8% | 18.2% |
| **All three (deployed)** | **56.9%** | **12.8%** | **7.1%** |

* At equal hit rate, verification cut wrong cache hits from 49.8% to 12.8%.
* On QQP the verifier alone looks best, but it was trained on QQP. On the traps (not from QQP) it
  served 18.2% wrong answers; the guards brought that to 7.1%. The deployed choice favours the
  data the verifier never saw and pays for it in hit rate.
* Per trap category (deployed): entity swaps 0%, negations 0%, number changes 0%, same-topic 5%,
  unit/direction 5% wrongly served.
* The deployed thresholds — cosine 0.799, verifier 0.904 — were written into the config.

**Lesson of the day:** an evaluation is only as good as its negatives. Mining them with your own
model is the right idea, and reading what you mined is not optional.

---

### Day 5 — Router data: pilots, quotas and a plan that had to change

**Goal for the day:** collect labelled routing data: for each query, is the cheap model's answer
good enough?

#### Step 5.1 — The original data pipeline

* `eval/router_queries.py` builds queries from public datasets with a fixed seed and a stratified
  70/15/15 train/val/test split by source. Each query has two text fields:
  * `query` — what a user would type; **the router only ever sees this**;
  * `prompt` — the query plus an answer-format instruction ("...write 'Answer: <number>'") that makes
    automatic grading possible; only the LLMs see this.
  The format suffix is a fixed per-source template, so a router trained on `prompt` could identify
  the source from it — label leakage, just like using `source` as a feature.
* `eval/router_collect.py` sends every query once to each tier at temperature 0 and appends results
  to `data/router/responses.jsonl`. It is **resumable**: a rerun skips what is already collected.
* `eval/llm_batch.py` throttles by requests and tokens per minute, backs off on 429s and stops
  cleanly on daily quotas.
* `eval/router_label.py` grades GSM8K by the final number and MMLU by the choice letter, and uses an
  LLM judge for open-ended Dolly and MBPP answers.

#### Step 5.2 — Reality check: the providers had changed

Probing the providers with the real keys showed:

* **Groq had retired its Llama 3.x models.** The available text models were `openai/gpt-oss-20b`,
  `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`.
* **Gemini 2.5 Flash / Flash-Lite were no longer available to new users.**
* The OpenRouter free Llama model was gone; the remaining free models were rate-limited upstream.
* Groq's free tier: 1,000 requests/day and 8,000 tokens/minute per model (from response headers).

The gateway configuration was moved to the models that exist, with **per-model request parameters**
in the YAML (e.g. `reasoning_effort: low` for gpt-oss). Real list prices were looked up:
gpt-oss-20b $0.075 / $0.30 and gpt-oss-120b $0.15 / $0.60 per 1M input/output tokens.

#### Step 5.3 — A pilot that changed the dataset

On 30 training-split queries the candidate cheap model was already at the ceiling:

| Model | GSM8K | MMLU |
|---|---|---|
| gpt-oss-20b | 15/15 | 12/15 |
| gpt-oss-120b | 15/15 | 13/15 |
| qwen3.8-27b | 15/15 | 13/15 |

With the planned data, nearly every label would be "cheap is fine" and always-cheap would match any
router. Real traffic mixes easy and hard requests, so the dataset was rebuilt with harder sources:

* **MATH** (competition math, levels 3–5, from MATH-500) — graded with **math-verify**, which checks
  symbolic equivalence (½ = 0.5, (3, π/2) = (3, \frac{\pi}{2})). Its built-in timeout uses
  multiprocessing, which silently returned "wrong" for everything on Windows; it is called with the
  timeout disabled and exceptions counted as incorrect.
* **MMLU-Pro** (10 options, A–J) — harder than MMLU.
* GSM8K, Dolly and MBPP kept as the easy end.

Pilot queries are recorded in `data/router/pilot_ids.json` and excluded from the final dataset, so
no query used to make design decisions is later used to measure them.

#### Step 5.4 — Choosing the tiers and the judge from quotas, not preferences

* Gemini 3.8 Flash would have been a realistic "frontier" strong tier, but its free tier allows
  **20 requests per day**, so it could be neither the strong tier nor the judge.
* Final setup: cheap = gpt-oss-20b, strong = gpt-oss-120b, judge = **qwen3.8-27b** (Alibaba), a
  family different from both tiers (OpenAI's gpt-oss), on the same provider but with its own quota.
* Consequence stated up front: the 2× price gap caps any routing saving at roughly 50%.

#### Step 5.5 — The judge

* Fixed rubric: correctness first, then completeness and clarity; length is not merit; answer order
  means nothing.
* Answers shown as A and B in a **seeded random order per query** (a hash of the query id), which
  cancels position bias on average and is stable across reruns.
* Reference answers from the datasets are shown as a hint ("reference-guided grading").
* Output is JSON; the parser takes the **last** valid JSON object, in case the model writes reasoning
  first.
* Label for open-ended items: 1 if `score_cheap >= score_strong - 1`.
* Validation: the judge also grades ground-truth pairs **without** the answer key (does it prefer the
  correct answer?), and a sample is judged twice with the order swapped (how often does the verdict
  flip?).

#### Step 5.6 — The quota problems, one after another

1. **Tokens per minute.** Groq counts `prompt + max_tokens` against the per-minute budget when a
   request *starts*. The collector now reserves that amount and corrects to the real usage after.
2. **Another project shared the quota.** The collector's cheap tier stalled. The cause was another
   local project (ConceptBound-Net) running an evaluation with the same Groq organization and the
   same default model. Rate limits are per organization, so it was consuming the cheap tier's
   budget. It was paused by its owner.
3. **Tokens per day.** The decisive limit was not in the headers: **200,000 tokens per model per
   day**. gpt-oss reasons before answering, so a MATH query uses ~870 tokens and a GSM8K query ~425.
   The 1,000-query pool needs ~540k tokens per model.

The options were laid out with costs: a paid tier for about $1–2 total, or stay free with fewer
queries, or stay free and wait several days. The decision was to **stay free with a stratified 500
queries** (a 50% sample within every source/split cell, chosen without looking at any answers, and
reusing the 123 strong answers already collected).

#### Step 5.7 — An unattended, resumable pipeline

* `call_rate_limited(..., wait_on_daily_limit=True)` parses Groq's "Please try again in 2m9s" hint,
  sleeps (with a floor and a ceiling) and retries **without** spending its attempt budget. Groq's
  daily limit is a rolling 24-hour window, so capacity trickles back continuously.
* `scripts/run_router_pipeline.py` runs collect → judge → labels → train → evaluate, logging to
  `data/router/pipeline.log`, stopping with a clear message on any error. Every step is resumable.

**Lesson of the day:** free tiers make data collection an engineering problem. Budget tokens, not
requests; read the real error messages; make every long job resumable; and expect other workloads
on the same account.

---

### Day 6 — Router training and evaluation, the dashboard, and the load test

**Goal for the day:** everything that turns labels into a router, plus the product surface.

#### Step 6.1 — Router features without leakage

`routewise/router/features.py`:

* the bge-small **query embedding** (384-d) — shared with the semantic cache, so it is loaded once
  and computed once per request;
* 29 **handcrafted features**: log length in characters and words, number of lines, average word
  length, digit ratio, count of numbers and math operators, LaTeX present, code present and count of
  code tokens, number of multiple-choice option lines, question mark, and 17 question-word flags
  ("what", "how", "prove", "write", "calculate", ...).

`FORBIDDEN_FEATURES` (`source`, `split`, `label`, `gold`, `category`, `subject`) is asserted absent
from the feature names. Features are computed from the raw `query` only.

#### Step 6.2 — Training with validation-only choices

`eval/router_train.py` trains four candidates:

* **logistic regression on handcrafted features only** — an interpretable ablation;
* **logistic regression** on embedding + handcrafted (standardized, `class_weight="balanced"`);
* **LightGBM** (`is_unbalance=True`, small grid over leaves, trees, min child samples);
* **MLP** (one hidden layer, early stopping, balanced **sample weights** because sklearn's MLP has no
  class-weight option).

Hyperparameters are chosen by **validation ROC-AUC**. Each model's routing threshold is the
cheapest point on the **validation** cost/quality curve that keeps at least 95% of always-strong
quality. The production model is the cheapest such model on validation (ties broken by AUC). The
artifact (`artifacts/router.joblib`) holds the fitted model, the threshold and metadata.

Why choose the model by cost at fixed quality rather than AUC: AUC is a fine tuning metric, but the
business metric is "how much do we save without hurting answers".

#### Step 6.3 — Evaluation against baselines

`eval/router_eval.py` reports on the **test** split:

* ROC-AUC, and PR-AUC for both classes (the "needs strong" class is the minority one a router must
  catch);
* cost-versus-quality curves by sweeping each model's threshold;
* baselines: always-cheap, always-strong, a **length heuristic** (threshold chosen on validation the
  same way), **random routing at equal cost** (a straight line between the two extremes), and an
  **oracle** that routes by the true labels (an upper bound);
* the headline: cost saved versus always-strong at the validation-chosen threshold, and the quality
  kept;
* a per-source breakdown, **leave-one-source-out** AUC (does the router generalize to a query type it
  never saw?), and single-query inference latency.

Before real labels existed, both scripts were smoke-tested end to end on synthetic labels in a
temporary directory, so a crash would not waste a day of collection.

#### Step 6.4 — The learned router in the gateway

`routewise/router/learned.py` loads the artifact and routes to cheap when P(cheap is good enough) ≥
threshold. If the artifact is missing or fails to load, the gateway logs a warning and uses the
length heuristic, so the gateway never fails because of the router.

#### Step 6.5 — Playground and dashboard

`dashboard/app.py` (Streamlit) with two tabs:

* **Playground**: ask a question; see the answer, cache status and reason, route and router
  probability, nearest cached query with cosine and verifier scores, a table of each candidate's guard
  verdicts, a per-stage latency bar chart, model, provider, fallback flag and estimated cost. Built-in
  example pairs (paraphrase, entity trap, version trap, direction trap, easy vs hard). A toggle also
  asks the strong model with the cache off, side by side.
* **Dashboard** (reads the request logs): requests, hit rates, estimated cost saved versus the
  always-strong counterfactual, fallbacks, errors, requests over time by cache status, cumulative cost
  versus counterfactual, p50/p95 latency per stage, route distribution and a **false-hit audit table**
  of recent semantic hits for a human to check.

`dashboard/data.py` holds the metric computations without Streamlit so they are unit-tested.
`scripts/simulate_traffic.py` replays a realistic mix (seeds, paraphrases, exact repeats, traps,
time-sensitive and high-temperature requests, held-out router queries).

#### Step 6.6 — Load test of the gateway's own overhead

* A **mock provider** (`ROUTEWISE_MOCK_PROVIDERS=1`) answers instantly, so measurements reflect
  RouteWise's own work.
* `eval/load_test.py` starts a fresh server per scenario and runs Locust headless for two scenarios.

| Users | Throughput | Server-side time excluding LLM, p50 / p95 |
|---|---|---|
| 1 | 5.2 req/s | 62 ms / 90 ms |
| 10 | 14.0 req/s | 577 ms / 840 ms |

The first version of the report claimed client latency was "higher than server overhead"; the
numbers showed they were nearly equal, because RouteWise's own timer includes time waiting for the
shared CPU. The report was rewritten to show the single-user cost and the under-load behaviour
separately. Memory with both models loaded: ~1.06 GB resident.

#### Step 6.7 — Integration with ConceptBound-Net

Reading the other project (without modifying it) showed its Groq client hard-codes the URL and sends
`response_format: {"type": "json_object"}`. RouteWise had been dropping `response_format`, so:

* `response_format` is now passed through to providers;
* it is part of both cache keys (a JSON answer must never be served to a plain-text request);
* `docs/integration_conceptbound.md` shows the small provider class to add on the ConceptBound side.

---

### Day 7 — The live demo, what it exposed, deployment and documentation

**Goal for the day:** run everything against real models, fix what that reveals, make it easy to run,
and document it.

#### Step 7.1 — The cache-key `max_tokens` problem

In the playground, a question asked earlier by the traffic simulator was a miss. The simulator used
`max_tokens=600`, the playground 1000, and `max_tokens` was part of both cache keys, so every client
setting split the cache. Because truncated answers are never cached, any cached answer is complete;
the only real constraint is not to serve an answer **longer** than the caller allows. So:

* `max_tokens` was removed from both keys (the exact-cache key version was bumped to `v2`);
* on a hit, the cached answer's completion tokens are compared with the request's `max_tokens`; if it
  is too long the request is a miss with reason `cached_answer_exceeds_max_tokens`.

#### Step 7.2 — spaCy called a unit conversion a person

A true paraphrase ("Convert 5 miles to km" / "How many kilometers is 5 miles?") was refused by the
entity guard. spaCy's small model had tagged "Convert 5 miles" as a **PERSON**. The generic fix:
ignore entity spans that contain digits — quantities are the numbers guard's job. The cache
evaluation was rerun: the deployed configuration's numbers were unchanged and the entity guard lost
fewer true duplicates (145 instead of 151).

#### Step 7.3 — The verifier misses some paraphrases

"What is 5 miles in kilometers?" vs "How many kilometers is 5 miles?" scored 0.006 with the verifier.
That is the precision-first trade-off: a missed hit costs one LLM call. It is recorded as a known
limitation.

#### Step 7.4 — A per-provider breaker turned one model's quota into a provider outage

With gpt-oss-20b out of daily tokens, its 429s opened the breaker for **all of Groq**, so the strong
tier skipped a healthy gpt-oss-120b and returned 503. Rate limits are per model, so breakers are now
keyed **per provider + model**. A full provider outage still trips each model's breaker within a few
requests. A regression test reproduces the scenario.

#### Step 7.5 — The dashboard showed negative savings

"Estimated cost saved: −57.6%". Cheap-tier requests had been falling back to Gemini 3.5 Flash-Lite,
whose output price ($2.50 per 1M tokens) is about four times the strong model's ($0.60). A fallback
that costs more than the strong tier defeats the purpose. The cheap chain now falls back to
gpt-oss-120b first and to Gemini only last, and `tier_of_model` maps an explicitly requested model to
the tier where it is the **primary** choice (gpt-oss-120b appears in both chains). The dashboard's help
text now explains what a negative value means. Finally a **cost guard** was added (`reliability.fallback_price_cap_ratio: 1.0`): any fallback
whose input or output price is above the always-strong reference model is skipped
(`skipped_over_budget`), so a fallback can never make a request cost more than always-strong; the
dashboard splits savings into *saved by cache*, *saved by routing* and *fallback overhead*.

#### Step 7.6 — The live demo, end to end

`scripts/demo.py` runs a scripted sequence through the official OpenAI SDK against the real gateway:

| Request | What RouteWise did |
|---|---|
| "What is the capital of India?" | Miss → LLM (953 ms) → cached |
| "Which city is India's capital?" | **Semantic hit** in 51 ms, $0 (cosine 0.934, verifier 0.938) |
| "What is the capital of Indonesia?" | Miss (not even a candidate; cosine below 0.799) |
| Python 3.12 after 3.10 | Refused by the **numbers** guard (cosine 0.947) |
| "Convert 5 km to miles" after "5 miles to km" | Refused by the **word-order** guard (cosine 0.990) |
| "latest news about AI today" | Cache **bypassed** (time-sensitive) |
| √2 irrationality proof | Routed to **strong** (gpt-oss-120b) |
| Requests while gpt-oss-20b had no quota | **Fallback** to the next model in the chain |

#### Step 7.7 — One-click demo, deployment bundle and docs

* `start_demo.bat`: checks `.venv` and `.env`, starts the gateway and the playground in their own
  windows (reusing them if they are already running), waits for each health check with Windows'
  built-in `curl.exe`, and opens http://localhost:8501. `stop_demo.bat` closes only those windows.
  One-second waits use `ping -n 2 127.0.0.1` because `timeout` fails without an interactive console.
* `deploy/hf_space/` and `scripts/prepare_hf_space.py`: a single-container Space (gateway on
  127.0.0.1:8000, Streamlit on port 7860), with the models baked into the image and demo seeds copied
  to a writable runtime directory at start.
* `scripts/build_demo_data.py`: builds the cold-start demo logs and a warm semantic cache from **real**
  gateway traffic, to be run after the router pipeline so it does not compete for quota.
* README, REPORT (technical report), LEARNING (interview notes) and this DEVLOG.

#### Step 7.8 — Experiment tracking with MLflow

`eval/mlflow_tracking.py` wraps MLflow so that tracking is optional and can never break training
(every call is guarded; `ROUTEWISE_MLFLOW=0` turns it off). `router_train` creates one parent run
per training session and one nested run per candidate model, logging the full hyperparameter grid
with each configuration's validation AUC, the chosen threshold and validation cost/quality. The
production router is logged with an inferred signature and registered as `routewise-router`;
`router_eval` reopens the same run (by the run id saved in `router_meta.json`) and adds the test
metrics, report and figures. The store is a local SQLite file, so it is free and serverless.

Two details were found by testing rather than reading docs:

* MLflow 3 saves scikit-learn models with **skops** instead of pickle (unpickling can execute code)
  and refuses unknown types, such as the MLP's internal `AdamOptimizer` or the
  `collections.OrderedDict` inside a LightGBM model. The wrapper trusts only types from scikit-learn,
  LightGBM, NumPy and SciPy (plus that one standard-library container), and a test checks that a
  registered model loads back with identical predictions for all three model families.
* Re-opening a run with `start_run(run_id=..., run_name=...)` silently **renames** it; the run name
  is only passed when creating a run.

#### Step 7.9 — Continuous integration

`.github/workflows/tests.yml` runs `make test` on every push and pull request on Ubuntu with
Python 3.12, CPU-only PyTorch and a pip cache, with read-only permissions and cancellation of
superseded runs. Before relying on it, every pinned package was checked against PyPI for a Linux
x86_64 / Python 3.12 wheel (all 31 have one), as were the spaCy model URL and the CPU PyTorch wheel.
The fast suite needs no secrets because every provider is mocked.

**Lesson of the day:** running the system for real found five issues that 130 unit tests had not —
two of them only visible through the dashboard. Observability is not decoration.

---

## Part 3 — Results in one place

### 3.1 Semantic cache (test split, `reports/cache_eval.md`)

| Metric | Value |
|---|---|
| Evaluation pairs | 2,300 (1,000 QQP duplicates, 1,000 QQP hard negatives, 200 traps, 100 paraphrases) |
| Candidate (recall) threshold | cosine >= 0.799 (keeps 95% of dev duplicates) |
| Verifier threshold | P(duplicate) >= 0.904 (max F0.5 on dev) |
| Deployed config hit rate | 56.9% (95% CI 52.8–61.1%) |
| Deployed config false-hit rate | 12.8% (95% CI 9.4–16.2%) |
| Deployed config false-hit rate on traps | 7.1% (2 wrong of 28 served; CI 0.0–17.9%) |
| Cosine-only false-hit rate at the same hit rate | 49.8% |
| Embedding one query (CPU, p50) | ~11 ms |
| Guards per candidate (p50) | ~6 ms |
| Verifier, one pair / batch of 5 (p50) | ~17 ms / ~72 ms |

Per-guard behaviour on stage-1 candidates (dev + test):

| Guard | QQP negatives caught | QQP duplicates lost | Trap negatives caught | Paraphrases lost |
|---|---|---|---|---|
| numbers | 83 / 1000 | 66 / 940 | 37 / 161 | 0 / 99 |
| negation | 45 / 1000 | 38 / 940 | 36 / 161 | 0 / 99 |
| entities | 220 / 1000 | 145 / 940 | 21 / 161 | 2 / 99 |
| word_order | 2 / 1000 | 2 / 940 | 10 / 161 | 0 / 99 |

### 3.2 Gateway performance (`reports/load_test.md`)

| Scenario | Throughput | Server-side overhead p50 / p95 |
|---|---|---|
| 1 user, mocked LLM | 5.2 req/s | 62 / 90 ms |
| 10 users, mocked LLM | 14.0 req/s | 577 / 840 ms (CPU queueing) |

Resident memory with both models: ~1.06 GB.

### 3.3 Live demo observations (real models)

* Semantic hit: 51 ms and $0, versus 953 ms for the original LLM call.
* Traps refused at cosine 0.947 (numbers guard) and 0.990 (word-order guard).
* Fallback across models observed while a model was out of daily quota.
* Dashboard exposed a fallback that cost more than the strong tier (fixed).

### 3.4 Router

The router pipeline runs within Groq's free-tier daily token budget (200k tokens per model per
day) and was still collecting answers when this log was written. Its results are generated by
`python -m eval.router_eval` into `reports/router_eval.md` and `reports/router_labels.md`
(label balance per source, judge validation, position-bias check, classifier metrics,
cost-quality curves against baselines, per-source breakdown, leave-one-source-out AUC and
inference latency). Until the trained artifact exists, the gateway routes with the length
heuristic and says so in its start-up log.

### 3.5 Test suite

132 fast tests (mocked providers, fakeredis, SQLite) plus 6 slow tests that load the real
embedding model and cross-encoder. No test calls a real LLM.

---

## Part 4 — Interview questions with answers

The answers are written the way I would say them, with the reasoning attached.

### 4.1 Project and architecture

**Q1. Describe RouteWise in two sentences.**
An OpenAI-compatible gateway that sits between applications and LLM providers. It answers safe
repeats and paraphrases from a verified cache, routes the rest to a cheap or strong model using a
learned predictor of difficulty, and handles retries, fallbacks and circuit breaking, logging cost
and latency for every request.

**Q2. Walk through a request from start to finish.**
Bearer-key auth and a per-key token bucket; cache eligibility (temperature, time-sensitivity,
opt-out, single vs multi-turn); exact cache (Redis hash of all messages and parameters); semantic
cache (embed, FAISS top-5 at or above cosine 0.799, rule guards, cross-encoder at or above 0.904);
router (P(cheap is fine) against the validation-chosen threshold); the tier's fallback chain with
retries and per-model circuit breakers; response with a `routewise` metadata object; the log row is
written in a background task after the response is sent.

**Q3. Why make it OpenAI-compatible?**
Adoption cost. Any app using the OpenAI SDK switches by changing `base_url`. A test points the real
`openai` client at a live server to prove it, including streaming and error types.

**Q4. Why is the pipeline separate from FastAPI?**
Testability and reuse. The `Gateway` class can be unit-tested without HTTP, and the web layer stays
thin: auth, rate limiting, serialization, background logging.

**Q5. Why FastAPI rather than Flask or Django?**
The gateway spends most of its time waiting on network calls, so async I/O matters; Pydantic
validation fits the OpenAI schema naturally; the interactive `/docs` page doubles as a demo tool.

**Q6. Why one httpx client instead of each provider's SDK?**
SDKs retry silently. Then my retry, fallback and breaker logic would not be the only reliability
layer, and my latency numbers would include hidden retries. Groq, Gemini and OpenRouter all expose
OpenAI-compatible endpoints, so one client works for all of them.

**Q7. How is configuration organized?**
Secrets and infrastructure URLs come from environment variables (`.env`, never committed); behaviour
lives in `config/routewise.yaml`, validated by Pydantic at start-up, so a typo fails fast.

### 4.2 Caching

**Q8. What are the two caches and why both?**
The exact cache (Redis) is nearly free and perfectly safe, and works for multi-turn conversations
because its key covers the whole conversation. The semantic cache catches paraphrases but needs
verification and is restricted to single-turn requests. Exact runs first.

**Q9. The demo had cache hits without Redis. How?**
Those were semantic-cache hits. The semantic cache stores entries in SQLite and searches them with an
in-memory FAISS index; it does not need Redis. Without Redis the exact cache is disabled at start-up
(the gateway logs a warning and `/health` reports `redis: false`), so an identical repeat is served
by the semantic cache with similarity 1.0. Redis would add sub-millisecond exact hits and state
shared across workers.

**Q10. What exactly goes into the exact-cache key, and what does not?**
Every message's role and whitespace-normalized text (system prompts included), the requested model,
temperature, top_p, stop sequences and response format. Not lowercased, because case matters for code.
Not `max_tokens`: truncated answers are never cached, so a cached answer is complete, and on a hit the
gateway checks that it fits the caller's `max_tokens`.

**Q11. Which requests are never cached, and why?**
Temperature above 0.3 or unset (non-deterministic), time-sensitive questions (the answer changes),
client opt-out headers, `n > 1`. Multi-turn requests skip only the semantic cache.

**Q12. Why does naive semantic caching fail?**
Embeddings capture topic, not exact meaning. "Python 3.10" and "3.12" questions have cosine 0.93; a
negated question about coffee 0.96. Tuned to the same hit rate as RouteWise, cosine-only caching
served wrong answers 49.8% of the time on the adversarial test set.

**Q13. Explain bi-encoder versus cross-encoder.**
A bi-encoder embeds each text separately, so vectors can be precomputed and searched quickly, but the
two texts never interact. A cross-encoder reads both texts together, so attention can compare "3.10"
with "3.12" directly; it is more accurate but needs one forward pass per pair. Retrieve with the
bi-encoder, verify a handful of candidates with the cross-encoder.

**Q14. Why rule guards if you have a cross-encoder?**
They fail differently. The verifier scored "5 miles to km" vs "5 km to miles" at 0.965 because it
leans on word overlap; the word-order guard rejects it. On the traps the verifier alone served 18.2%
wrong answers; with the guards, 7.1%.

**Q15. What does each guard check?**
Numbers, years, versions and spelled-out numbers must match; negation parity must match; named
entities must appear in both; identical words in a different order are rejected. Guards only veto,
so a mistake costs a cache miss rather than a wrong answer.

**Q16. Tell me about a guard bug you found.**
spaCy's small model labelled "Convert 5 miles" as a PERSON, so a true paraphrase was refused. Spans
containing digits are now ignored by the entity guard, because quantities are the numbers guard's
job. Earlier, "one" in "how can one live" was parsed as the number 1. Both have regression tests.

**Q17. Why FAISS IndexFlatIP and not HNSW or IVF?**
At single-service cache sizes, exact inner-product search is fast enough on CPU and has perfect
recall. Approximate indexes trade recall for speed and only pay off at much larger scale.

**Q18. Why is SQLite the source of truth and FAISS rebuilt at start-up?**
With two stores there is a risk they drift apart after a crash. Storing embeddings in SQLite and
rebuilding the index guarantees consistency; the rebuild is fast at this scale.

**Q19. What is a namespace in your semantic cache?**
A hash of everything except the user's question that changes the right answer: system prompts,
requested model, stop sequences and response format. Each namespace has its own FAISS index, so two
apps with different personas never share answers.

**Q20. How do you handle cache invalidation?**
Mostly by not caching what goes stale (time-sensitive and high-temperature requests), bounding
everything else with TTLs (24 hours exact, 7 days semantic), and versioned key prefixes to invalidate
everything at once after a format change.

**Q21. What happens if the embedding model or verifier fails to load?**
The semantic cache is disabled with a warning and the gateway keeps serving. Without the verifier,
more near-miss false hits would get through; serving no cached answers is safer than serving wrong ones.

### 4.3 Evaluation methodology

**Q22. How did you build hard negatives?**
I embedded all QQP validation questions with the cache's own embedder and took human-labelled
non-duplicates with the highest similarity: the pairs a naive cache would wrongly serve.

**Q23. What went wrong with them?**
The very top of the ranking was mostly label noise. I removed text-identical pairs and sampled from
the top 5,000 instead of taking the top 1,000. An audit of 40 still found about 10–22% noise, which
the report states.

**Q24. Why is the QQP comparison biased toward the verifier?**
The verifier was trained on Quora duplicate questions, so QQP is close to its training data. The trap
set is not from QQP, so it is the fairer test, and the deployment decision follows it.

**Q25. How did you choose thresholds without overfitting?**
On the dev split only. The candidate threshold keeps 95% of dev duplicates; the verifier threshold
maximizes F0.5 on dev. Numbers are reported on test, with bootstrap confidence intervals.

**Q26. Why F0.5 rather than F1?**
F0.5 weights precision twice as much as recall. A false hit serves a wrong answer to a user; a missed
hit only costs one LLM call. The asymmetry should be in the metric.

**Q27. Define hit rate and false-hit rate in ML terms.**
Hit rate is recall over true duplicates. False-hit rate is 1 minus precision over served pairs.

**Q28. What are the weaknesses of your trap set?**
LLM-drafted, not yet human-verified, small (100 test pairs, wide intervals) and written by the same
process that designed the guards, so it likely flatters them. Independent, human-written traps would
be a stronger test.

**Q29. Did you look at test data while developing?**
Once, in the first guard error analysis, which sampled pairs from both splits. The fixes were generic
bug fixes, and the report discloses that test numbers may be slightly optimistic as a result.

### 4.4 Router and labels

**Q30. How is a router label defined?**
1 means the cheap model's answer is good enough. For math and multiple choice: the cheap answer is
correct, or both answers are wrong (paying for strong would have bought nothing). For open-ended
tasks: a judge's score for the cheap answer is at least the strong score minus 1.

**Q31. Why did you change the datasets?**
A pilot showed the cheap model at 15/15 on GSM8K and 12/15 on MMLU; nearly every label would have been
1 and always-cheap would beat any router. Adding MATH and MMLU-Pro gives the mixed difficulty real
traffic has. Pilot queries were excluded from the final data.

**Q32. How do you grade MATH answers?**
With math-verify, which parses LaTeX and checks symbolic equivalence (½ equals 0.5). Its timeout uses
multiprocessing, which made every answer "wrong" on Windows, so it is called with the timeout off and
exceptions count as incorrect.

**Q33. What is position bias in an LLM judge and how do you handle it?**
Judges tend to prefer the answer shown first (or second). Each query gets a seeded random A/B order,
which cancels the bias on average and is reproducible; a swap test measures how often the verdict
flips.

**Q34. Why is the judge from a different model family?**
Models tend to prefer answers from their own family. The tiers are OpenAI gpt-oss models; the judge
is Alibaba's Qwen.

**Q35. Why must the dataset source never be a feature?**
It is almost a proxy for the label (competition math is usually hard). A router using it would look
excellent offline and be useless in production, where requests do not arrive labelled with a dataset
name. For the same reason the router reads the raw query, not the prompt with the grading suffix.

**Q36. Which models did you train and how were they chosen?**
Logistic regression (plus a handcrafted-only ablation), LightGBM and an MLP, each with imbalance
handling. Hyperparameters by validation ROC-AUC; the production model is the cheapest one on
validation that keeps at least 95% of always-strong quality.

**Q37. How do you handle class imbalance?**
`class_weight="balanced"` for logistic regression, `is_unbalance` for LightGBM, balanced sample weights
for the MLP. And metrics that are not fooled by imbalance: PR-AUC for the minority class and the
cost/quality curve.

**Q38. ROC-AUC versus PR-AUC?**
ROC-AUC is the probability that a random positive ranks above a random negative and ignores class
balance. PR-AUC reflects how clean the positive predictions are and is more informative when the class
you care about is rare. I report PR-AUC for both classes.

**Q39. What baselines do you compare against?**
Always-cheap, always-strong, a length heuristic with its threshold chosen on validation, random routing
at equal cost, and an oracle that routes by the true labels as an upper bound.

**Q40. What is the most the router could save?**
The strong model's list price is twice the cheap one's, so routing everything to cheap saves roughly
50% (it varies with output lengths). With a frontier model 10–30 times more expensive, the same method
would save much more.

**Q41. What does leave-one-source-out tell you?**
Whether the router generalizes to a query type it has never seen. It is usually much worse than
in-distribution performance, which is an honest limitation of a small dataset.

### 4.5 Reliability and operations

**Q42. Explain your retry policy.**
Exponential backoff with full jitter, only for retryable errors, at most two retries. If the provider's
Retry-After exceeds our cap, fail over immediately instead of waiting.

**Q43. Why full jitter?**
Clients that failed together would otherwise retry together and fail again together. Randomizing the
whole interval spreads the load.

**Q44. Draw the circuit breaker.**
CLOSED to OPEN after N consecutive transient failures; OPEN to HALF_OPEN after the cooldown; one probe
in HALF_OPEN (a Redis SET NX lock); success closes it, failure reopens it.

**Q45. Why is the breaker keyed per model?**
Because rate limits are per model. In the live demo a per-provider breaker turned one model's quota
exhaustion into an outage of the whole provider and blocked a healthy model.

**Q46. Why does only a transient error count against the breaker?**
A 400 means the request is bad, not the provider. Counting it would let one buggy client take a healthy
provider offline for everyone.

**Q47. How does the token bucket work and why Lua?**
Capacity is the burst size, the refill rate is the sustained rate, each request spends one token. In
Redis, read-refill-spend is one Lua script, which Redis runs atomically, so concurrent workers cannot
both spend the last token.

**Q48. What happens when Redis is down?**
Exact cache off, rate limiting and breaker state in process memory, everything else unchanged. Tested
for Redis missing at start-up and for Redis failing mid-run.

**Q49. What did the dashboard show you that tests did not?**
Negative "cost saved": cheap traffic was falling back to a model whose output price is about four times
the strong model's. The fallback order was a cost bug, not a code bug.

**Q50. How did you deal with free-tier limits during data collection?**
Throttle by requests and tokens per minute, reserving prompt plus max_tokens as Groq does; back off on
429; parse the "try again in" hint and wait on daily limits; make every step append-only and resumable;
and shrink the dataset to fit a free daily budget instead of paying, with the trade-off documented.

**Q51. What did the load test measure?**
The gateway's own overhead with mocked providers: 62 ms p50 and 90 ms p95 per request with one user; at
10 concurrent users the single worker is CPU-bound by the two transformer models and latency rises from
queueing.

**Q52. How would you scale RouteWise?**
More workers (state is already in Redis), a GPU or ONNX/quantized models for embedding and
verification, batching verifier calls across requests, an approximate index beyond a million entries,
Postgres for logs, and Redis vector search so several instances share one semantic cache.

**Q53. Why is streaming buffered?**
A cached or failed-over answer cannot be half-sent. Streaming the complete answer keeps `stream=True`
clients working while caching and fallback stay simple; time-to-first-token does not improve.

**Q54. How is cost calculated?**
Prompt and completion tokens times the list price of the model that served the request; reasoning
tokens are included because providers bill them as output. Free-tier calls really cost nothing, so all
costs are labelled estimates.

### 4.6 Testing, deployment and reflection

**Q55. How do you test code that calls paid APIs?**
Fake clients or `respx` for HTTP, fakeredis with Lua, SQLite in temporary directories, dependency
injection through the app factory. Slow tests load the real embedding and verifier models but never
call an LLM.

**Q56. How is it deployed for free?**
A Hugging Face Space with the Docker SDK: one container runs the gateway internally and Streamlit
publicly on port 7860, with the models baked into the image. Render's free tier has 512 MB of RAM, and
the gateway uses about 1.06 GB.

**Q57. How do you run it locally in one click?**
`start_demo.bat` starts the gateway and the playground in their own windows, waits for their health
checks, and opens http://localhost:8501. `stop_demo.bat` stops them.

**Q57b. How do you track and version router experiments?**
MLflow: a parent run per training session with nested runs per candidate (grid, best params,
validation metrics), the chosen model registered as `routewise-router` so every retrain is a new
version, and test metrics attached to the same run. Comparing runs answers "did the new data or
features actually help?", and the registry answers "which model is in production?".

**Q57c. What runs in CI, and why not the slow tests?**
`make test` on every push: about 150 tests with mocked providers, fakeredis and SQLite, so CI needs
no secrets and is deterministic. The slow tests download two transformer models; they run locally
with `make test-all`.

**Q58. Why not just use LiteLLM or GPTCache?**
For plumbing they are better than anything I would maintain alone, and in production I would use
LiteLLM's provider adapters. The project's point is the parts those tools leave naive: verifying
semantic-cache hits and measuring false hits, and evaluating a router against baselines on held-out
data. Building the plumbing myself also means I can explain every failure mode.

**Q59. What would you do next?**
Human-verified, independently written trap sets; more router data across several model pairs;
predicting quality per model for multi-model routing; calibrated probabilities; execution-based grading
for code; real token streaming.

**Q60. What was the most important lesson?**
Measure, then believe. The verifier's weakness, the label noise in mined negatives, the too-easy
dataset, the per-provider breaker and the expensive fallback were all found by measuring or running
the system, not by reading the code.

---

## Part 5 — Things to be able to draw on a whiteboard

1. The request flow with each latency stage marked.
2. The semantic cache: bi-encoder, FAISS top-k, guards, cross-encoder; recall stage versus precision
   stages.
3. Bi-encoder (two towers and a cosine) versus cross-encoder (one tower over the pair).
4. The circuit-breaker state machine.
5. The token bucket and why the Lua script makes it atomic.
6. The cache's precision-recall curve with the F0.5 operating point.
7. The router's cost-quality curve: the 95% quality line, the validation-chosen point, random routing
   as a straight line, the oracle as a single point.
8. The data pipeline: queries, collect, ground truth or judge, labels, train with validation selection,
   test evaluation, artifact loaded by the gateway.
9. The fallback chain per tier with a breaker in front of each model.

---

## Part 6 — How to run everything

```bash
# one-time setup (Windows paths shown; use .venv/bin on macOS/Linux)
python -m venv .venv
.venv\Scripts\pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-dev.txt
copy .env.example .env          # then add GROQ_API_KEY and GEMINI_API_KEY

# demo (Windows): double-click start_demo.bat, or:
.venv\Scripts\python -m uvicorn routewise.main:app --port 8000
.venv\Scripts\python -m streamlit run dashboard\app.py
.venv\Scripts\python scripts\demo.py

# tests
.venv\Scripts\python -m pytest -q -m "not slow"

# evaluations
.venv\Scripts\python -m eval.build_cache_pairs
.venv\Scripts\python -m eval.cache_eval
.venv\Scripts\python scripts\run_router_pipeline.py
.venv\Scripts\python -m eval.load_test
```

---

## Part 7 — FAQ: questions asked while building and demoing it

These came up while running the demo. They are also likely interview questions.

### Which LLM models does RouteWise use, what does each cost, and which is strongest?

| Model | Provider | Role | List price, USD per 1M tokens (input / output) | Free-tier limits seen |
|---|---|---|---|---|
| `openai/gpt-oss-20b` | Groq | **Cheap** tier, first choice | 0.075 / 0.30 | 1,000 req/day, 8k tokens/min, 200k tokens/day |
| `openai/gpt-oss-120b` | Groq | **Strong** tier, first choice; also the cheap tier's first fallback | 0.15 / 0.60 | same, separate quota |
| `gemini-3.5-flash-lite` | Google | Cheap-tier last resort (blocked by the cost guard) | 0.30 / 2.50 | not published |
| `gemini-3.8-flash` | Google | Strong-tier fallback (blocked by the cost guard) | 0.75 / 3.75 (promo price until 2026-12-31) | 20 req/day |
| `qwen/qwen3.8-27b` | Groq | **Judge only** for router labels; never answers users | 0.80 / 4.00 | same as Groq above |

Prices were taken from the providers' pricing pages on 2026-11-11. Every cost RouteWise
shows is an **estimate** at these prices; the free tiers used here actually cost $0.

Strength, honestly: the only head-to-head numbers measured in this project are small pilots.
On 30 GSM8K/MMLU questions gpt-oss-20b scored 27/30 and gpt-oss-120b 28/30; on hard MATH
problems gpt-oss-120b scored 14/16 (gpt-oss-20b could not be measured that day because of
quota). By size and by these pilots, **gpt-oss-120b >= gpt-oss-20b**. The Gemini models were
not benchmarked here, so I do not rank them.

### What are the three APIs — Google, Grok and Ollama?

* **Groq** (with a q: an inference company serving open models such as gpt-oss) — not xAI's Grok.
* **Google Gemini** through its OpenAI-compatible endpoint.
* **OpenRouter** is configured (a key exists) but no tier uses it: its free models were
  unreliable when tested.
* **Ollama is not implemented.** It was an optional idea in the plan. Because Ollama serves an
  OpenAI-compatible API (`http://localhost:11434/v1`), adding it is a provider entry in
  `config/routewise.yaml` plus a tier target — no code changes.

### After I send a query, what exactly happens?

1. Check the API key and the per-key rate limit (token bucket, 30 requests/minute).
2. **May this request be cached?** (see the next question). If not, skip to step 5.
3. **Exact cache** (Redis): an identical earlier request is returned immediately. Only if
   Redis is configured.
4. **Semantic cache**: a cached question with the same meaning (embedding search -> rule
   guards -> verifier) is returned for $0 in ~50 ms.
5. **Router** chooses the tier: cheap or strong.
6. **Fallback chain** of that tier, in order, with retries and per-model circuit breakers:
   * cheap: gpt-oss-20b -> gpt-oss-120b -> (Gemini Flash-Lite, skipped by the cost guard)
   * strong: gpt-oss-120b -> (Gemini 3.8 Flash, skipped by the cost guard)
   If every allowed model fails, the client gets HTTP 503 with the reason per model.
7. The answer is stored in the semantic cache (if allowed), the request is logged, and the
   response is returned with the `routewise` metadata.

### How is the difficulty of a query decided? Is the router working?

The **learned router is not trained yet**: its data is being collected within Groq's free
daily token budget. Until `artifacts/router.joblib` exists, `model="auto"` uses the
**length heuristic** baseline (`routewise/router/baselines.py`), and `/health` shows which
router is active:

* more than 40 words -> **strong** (`heuristic:long_query`);
* looks like code or maths (```` ``` ````, `def `, `class `, "prove", "derive",
  "step by step", an operator next to a number) -> **strong** (`heuristic:code_or_math`);
* otherwise -> **cheap** (`heuristic:short_query`).

When trained, the learned router predicts P(the cheap model's answer is good enough) from the
query embedding plus 29 handcrafted features, and routes to cheap when that probability is
above a threshold chosen on validation data to keep at least 95% of always-strong quality.
Its metadata then shows `route_reason: learned:<model>` and `router_prob`.

### Why is gpt-oss-120b in both the cheap and the strong tier?

* In the **strong** tier it is the first choice.
* In the **cheap** tier it is the **fallback**: when gpt-oss-20b is unavailable (for example
  out of daily quota), the next-best model that does not cost more than the strong tier is
  gpt-oss-120b itself. The alternative, Gemini Flash-Lite, costs ~4x more per output token.
* So a "cheap" request answered by gpt-oss-120b (`fallback used` in the playground) costs
  exactly the always-strong price: no saving, but no overspend either.
* If you request `model="openai/gpt-oss-120b"` explicitly, it is treated as a strong request,
  because a model belongs to the tier where it is the **first** choice.

### Why did the dashboard show negative "cost saved", and what fixed it?

"Cost saved" = 1 − (what RouteWise spent ÷ what sending every request to gpt-oss-120b would
have cost for the same tokens). In the live demo, both Groq models ran out of daily quota and
cheap requests fell back to Gemini Flash-Lite ($2.50 per 1M output tokens vs $0.60), whose long
answers cost about 4x the strong price. On the demo log: cache saved $0.00073, routing saved
$0.00115, fallbacks cost $0.00349 extra — a net negative.

Fixes:
* **Cost guard** (`reliability.fallback_price_cap_ratio: 1.0`): a fallback model whose input
  or output price is above the always-strong reference is skipped (`skipped_over_budget`).
  RouteWise returns 503 rather than overspend; `null` restores availability-over-cost.
* The dashboard now shows **Saved by cache**, **Saved by routing** and **Fallback overhead**,
  which add up to the total, so a negative total explains itself.

### What do the dashboard numbers mean?

| Number | Meaning |
|---|---|
| Requests | All logged requests in the log database |
| Cache hit rate | (exact + semantic hits) ÷ successful requests |
| Semantic hits | Semantic hits ÷ successful requests (equal to the hit rate when Redis is off) |
| Est. cost saved | 1 − spent ÷ always-strong counterfactual; negative = overspending |
| Saved by cache / by routing / Fallback overhead | The three parts of the saving above |
| Fallbacks | Answers served by a model other than the tier's first choice |
| Errors | Requests that failed (e.g. 503 when every allowed model was out of quota) |
| Requests over time | Requests per 5 minutes, coloured by cache status |
| Cost chart | Cumulative spend vs always-strong; blue above orange = overspending |
| Latency per stage | p50/p95 per pipeline stage, from each request's timings |
| Route distribution | How many requests went cheap vs strong |
| False-hit audit | Recent semantic hits, for a human to confirm they really are the same question |

### What decides whether a question and answer are cached?

Two separate decisions:

**Can this request use the cache (lookup and store)?** No if:
* temperature is above 0.3, or not set (the provider default is about 1.0 = random);
* the question is time-sensitive: today, latest, current, news, weather, scores, **prices,
  costs, "how much is", release dates, upcoming, schedules, rankings, elections** (the second
  group was added after "iphone 17 price" was cached in the demo);
* the client sends `X-RouteWise-Cache: off` or `Cache-Control: no-cache`;
* `n > 1` is requested;
* for the semantic cache only: the conversation has more than one user turn.

**Is this particular answer stored?** Only if the LLM call succeeded, the answer is not empty,
and it was not cut off by `max_tokens` (`finish_reason == "length"`). Entries expire after 7
days (semantic) or 24 hours (exact).

**Is a cached answer served?** Exact cache: identical messages and parameters. Semantic cache:
cosine >= 0.799, all four guards pass, verifier >= 0.904, same namespace (same system prompt,
model, stop sequences and response format), and the cached answer fits the caller's `max_tokens`.

### What is the semantic cache, who provides it, is it free, is it Docker?

* It is **not a hosted service and has no provider**. It is RouteWise's own code
  (`routewise/cache/`) running **inside the gateway's Python process**, built from free,
  open-source parts:
  * **sentence-transformers** with `BAAI/bge-small-en-v1.5` (embeddings) and
    `cross-encoder/quora-distilroberta-base` (verifier), downloaded once (~400 MB) from Hugging
    Face and run on the CPU;
  * **FAISS** (Meta's open-source vector-search library) as an in-memory index;
  * **SQLite** (a file, `data/semantic_cache.db`) as the durable store;
  * **spaCy** `en_core_web_sm` for the entity guard.
* **Free:** no API calls, no account, no cost per lookup — only CPU time (~50 ms per lookup).
* **Not Docker:** it runs wherever the gateway runs. Docker is just one optional way to
  package the whole gateway.

### Is Redis used? Which cache is running?

* **Redis is optional and off by default.** With no `REDIS_URL`, the gateway does not try to
  connect, `/health` shows `"redis": "not configured"` and `"exact_cache": "off"`, and rate
  limiting and circuit-breaker state live in process memory.
* **The semantic cache (FAISS + SQLite) does all the caching** without Redis; identical repeats
  are served by it at similarity 1.0, which is why they show as `semantic_hit`.
* With Redis (set `REDIS_URL`, or use Docker Compose, which starts one): identical repeats become
  `exact_hit` in about a millisecond, and limits/breaker state are shared by all workers.
* `exact_cache_unavailable` now appears only if Redis **is configured but unreachable**.

### What is the frontend?

* **Streamlit** (an open-source Python framework) serves the playground and dashboard on
  port 8501, locally; charts are drawn with **Plotly**. It calls the gateway over HTTP for the
  playground and reads the request-log database directly for the dashboard.
* The gateway also serves **Swagger UI** at http://localhost:8000/docs (FastAPI's built-in
  interactive API docs).

### Why did the launcher once fail to start the gateway?

On every start, sentence-transformers asked huggingface.co about the already-downloaded models
(4.0 s here versus 1.1 s from the local cache). On a slow or unavailable network that check
can stall start-up. Models are now loaded from the local cache first, and the network is only
used if a model is not downloaded yet.

### How are router experiments tracked?

With MLflow (free, local: `mlflow.db` + `mlruns/`). Each training session is a parent run with one
nested run per candidate model; the production router is registered as `routewise-router` (a new
version per retrain), and the evaluation adds its test metrics to the same run. Open the UI with
`make mlflow-ui` (http://localhost:5000) or the IDE task "RouteWise: MLflow UI".

### How is the code tested on every push?

GitHub Actions (`.github/workflows/tests.yml`) installs the pinned dependencies with CPU-only
PyTorch and runs `make test`. No API keys are needed: providers are mocked, Redis is fakeredis and
the database is SQLite. The status badge at the top of the README shows the latest result.

### How do I run the demo in one step?

* Double-click the **"RouteWise Demo"** desktop shortcut, or `start_demo.bat`.
* In VS Code-based IDEs: **Ctrl+Shift+B** ("RouteWise: Start demo"), or *Tasks: Run Task* for
  stop, CLI demo, tests and the router pipeline (`.vscode/tasks.json`).
* Stop with **"Stop RouteWise Demo"** / `stop_demo.bat`.
