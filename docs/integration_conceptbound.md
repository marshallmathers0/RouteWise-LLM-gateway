# Using RouteWise from ConceptBound-Net

ConceptBound-Net (the lecture-retrieval RAG project) calls its LLM through
`src/llm/providers.py`. Its `GroqClient` posts an OpenAI-style chat request to a
**hard-coded** Groq URL, so it can't be pointed at RouteWise with an environment variable
alone. The smallest change is one new provider class plus one line in the factory.
**This repo does not modify ConceptBound**: the snippets below are what to add there.

## 1. Run RouteWise

```bash
uvicorn routewise.main:app --port 8000          # or: docker compose up
```

## 2. Add a provider to ConceptBound (`src/llm/providers.py`)

It reuses ConceptBound's own `post_json` helper (timeouts, retries, error types), so its
error handling and latency stats keep working unchanged.

```python
class RouteWiseClient(LLMClient):
    """Any OpenAI-compatible gateway; here RouteWise. model="auto" lets RouteWise route."""
    name = "routewise"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("ROUTEWISE_API_KEY", "rw-dev-key")
        self.model = model or os.environ.get("LLM_MODEL", "auto")

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        payload = {"model": self.model, "temperature": 0.0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}   # RouteWise passes this through
        self.last_stats = {}
        data = post_json(f"{self.base_url}/chat/completions", payload,
                         {"Authorization": f"Bearer {self.api_key}"}, _timeout(),
                         provider="RouteWise", stats=self.last_stats)
        meta = data.get("routewise", {})
        self.last_stats.update({"route": meta.get("route"), "cache": meta.get("cache_status", {}).get("status"),
                                "estimated_cost_usd": meta.get("estimated_cost_usd")})
        return data["choices"][0]["message"]["content"]
```

And in `src/llm/factory.py`:

```python
    if provider == "routewise":
        from src.llm.providers import RouteWiseClient
        return RouteWiseClient()
```

## 3. Configure ConceptBound's `.env`

```ini
LLM_PROVIDER=routewise
LLM_BASE_URL=http://localhost:8000/v1
ROUTEWISE_API_KEY=rw-dev-key
LLM_MODEL=auto          # or "cheap" / "strong" / an explicit model to bypass the router
```

## What to expect

* **Routing works as usual.** The router sees ConceptBound's user message, which contains
  the question plus the retrieved lecture passages. Long, context-heavy prompts look
  "hard" to a length heuristic; the learned router judges the text itself.
* **JSON mode works.** RouteWise forwards `response_format`, and JSON-mode requests get
  their own cache keys, so a JSON answer is never served to a plain-text request.
* **Cache hits will be rare, and that's correct.** The retrieved passages are part of
  the user message, so two RAG calls only match when both the question *and* the
  retrieved context are (nearly) the same. A cache that ignored the context could serve
  an answer grounded in the wrong lecture.
* **Reliability comes for free.** ConceptBound gets RouteWise's fallback chain, circuit
  breaker and request logs (visible in the RouteWise dashboard) on top of its own retries.
