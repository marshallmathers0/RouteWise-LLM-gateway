"""RouteWise playground + dashboard (Streamlit).

    streamlit run dashboard/app.py

Env:
  ROUTEWISE_URL      gateway base URL      (default http://localhost:8000)
  ROUTEWISE_API_KEY  client key            (default rw-dev-key)
  DATABASE_URL       request-log database  (default sqlite:///data/routewise.db)
  DEMO_LOGS_DB       fallback log DB when the main one is empty (default data/demo/logs.db)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data import load_logs, semantic_hits, stage_latency, summary  # noqa: E402
from routewise.storage.repository import make_engine  # noqa: E402

GATEWAY = os.environ.get("ROUTEWISE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("ROUTEWISE_API_KEY", "rw-dev-key")
DB_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(ROOT / 'data' / 'routewise.db').as_posix()}")
DEMO_DB = Path(os.environ.get("DEMO_LOGS_DB", ROOT / "data" / "demo" / "logs.db"))
# Validated categorical palette (slots 1-4) and status colors.
C = {"cheap": "#2a78d6", "strong": "#eb6834", "cache": "#1baf7a", "bypassed": "#eda100",
     "miss": "#2a78d6", "exact_hit": "#1baf7a", "semantic_hit": "#6250d6"}

st.set_page_config(page_title="RouteWise", page_icon="🔀", layout="wide")

EXAMPLES = {
    "Paraphrase pair (should hit)": ["What is the capital of India?", "Which city is India's capital?"],
    "Entity trap: India -> Indonesia": ["What is the capital of India?", "What is the capital of Indonesia?"],
    "Version trap: Python 3.10 -> 3.12": ["What are the main new features in Python 3.10?",
                                          "What are the main new features in Python 3.12?"],
    "Direction trap: miles <-> km": ["Convert 5 miles to km", "Convert 5 km to miles"],
    "Easy vs hard (routing)": ["What is the capital of France?",
                               "Prove that there are infinitely many primes of the form 4k+3."],
}


def call_gateway(question: str, model: str, bypass_cache: bool = False) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if bypass_cache:
        headers["X-RouteWise-Cache"] = "off"
    r = httpx.post(f"{GATEWAY}/v1/chat/completions", headers=headers, timeout=180,
                   json={"model": model, "temperature": 0, "max_tokens": 2000,
                         "messages": [{"role": "user", "content": question}]})
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(data.get("error", {}).get("message", r.text))
    return data


def show_result(resp: dict, title: str) -> None:
    m = resp["routewise"]
    cs = m["cache_status"]
    st.markdown(f"#### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cache", cs["status"].replace("_", " "))
    route = m["route"] or "-"
    c2.metric("Route", route, f"P(cheap ok) = {m['router_prob']:.2f}" if m.get("router_prob") is not None else None,
              delta_color="off")
    c3.metric("Latency", f"{m['latency_ms'].get('total', 0):.0f} ms")
    c4.metric("Est. cost", f"${m['estimated_cost_usd']:.6f}")
    if cs.get("reason"):
        st.caption(f"cache reason: `{cs['reason']}`")
    if m.get("matched_query"):
        st.info(f"Nearest cached query: **{m['matched_query']}**  \n"
                f"cosine {m['similarity']}" + (f" - verifier {m['verifier_score']}" if m.get("verifier_score") is not None else ""))
        cands = (m.get("guard_results") or {}).get("candidates", [])
        if cands:
            st.dataframe(pd.DataFrame([{"cached query": c["query"], "cosine": c["similarity"],
                                        "guards": ", ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in c["guards"].items()),
                                        "verifier": c["verifier_score"], "decision": "HIT" if c["accepted"] else (c["rejected_by"] or "-")}
                                       for c in cands]), hide_index=True, use_container_width=True)
    st.caption(f"model `{m['model_used']}` via `{m['provider_used']}`" + (" - **fallback used**" if m["fallback_used"] else ""))
    lat = {k: v for k, v in m["latency_ms"].items() if k != "total"}
    if lat:
        fig = px.bar(x=list(lat.values()), y=list(lat.keys()), orientation="h", labels={"x": "ms", "y": ""},
                     color_discrete_sequence=["#2a78d6"], height=40 + 26 * len(lat))
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(resp["choices"][0]["message"]["content"])


def playground() -> None:
    st.subheader("Playground")
    st.caption(f"Gateway: `{GATEWAY}`. Ask the first question of a pair, then the second: the trap pairs "
               "should NOT be answered from cache, the paraphrase should.")
    with st.sidebar:
        st.markdown("**Examples**")
        for label, qs in EXAMPLES.items():
            with st.expander(label):
                for q in qs:
                    if st.button(q, key=f"ex-{label}-{q}"):
                        st.session_state["question"] = q
    q = st.text_area("Your question", key="question", height=90)
    col1, col2 = st.columns([1, 1])
    model = col1.selectbox("Model", ["auto", "cheap", "strong"], help="auto = learned router decides")
    compare = col2.toggle("Also ask always-strong (cache off) to compare")
    if st.button("Send", type="primary", disabled=not q.strip()):
        try:
            with st.spinner("Routing..."):
                resp = call_gateway(q, model)
            if compare:
                left, right = st.columns(2)
                with left:
                    show_result(resp, "RouteWise")
                with right, st.spinner("Asking strong model..."):
                    show_result(call_gateway(q, "strong", bypass_cache=True), "Always-strong")
            else:
                show_result(resp, "RouteWise")
        except Exception as e:  # noqa: BLE001
            st.error(f"Gateway error: {e}")


@st.cache_data(ttl=15)
def get_logs() -> tuple[pd.DataFrame, str]:
    df = load_logs(make_engine(DB_URL))
    if df.empty and DEMO_DB.exists():
        return load_logs(make_engine(f"sqlite:///{DEMO_DB.as_posix()}")), "demo dataset (pre-generated)"
    return df, "live logs"


def dashboard() -> None:
    st.subheader("Dashboard")
    try:
        df, origin = get_logs()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read logs from {DB_URL}: {e}")
        return
    if df.empty:
        st.info("No requests logged yet. Run `python scripts/simulate_traffic.py` to generate traffic.")
        return
    s = summary(df)
    st.caption(f"Source: {origin} - {s['requests']:,} requests. Costs are ESTIMATES from list prices.")
    k = st.columns(3) + st.columns(3)  # two rows, so values never truncate in narrow windows
    k[0].metric("Requests", f"{s['requests']:,}")
    k[1].metric("Cache hit rate", f"{s['hit_rate']:.1%}")
    k[2].metric("Semantic hits", f"{s['semantic_hit_rate']:.1%}")
    k[3].metric("Est. cost saved", f"{s['saved_frac']:.1%}",
                help="vs. sending every request to the strong model. Negative means RouteWise spent MORE "
                     "(e.g. fallbacks to a pricier model) - check the fallback chain order.")
    k[4].metric("Fallbacks", s["fallbacks"])
    k[5].metric("Errors", s["errors"])
    # Where the saving (or overspend) comes from: these three add up to the total above.
    b = st.columns(3)
    b[0].metric("Saved by cache", f"${s['saved_by_cache_usd']:.4f}",
                help="Always-strong cost of the requests answered from cache (a cache hit costs $0).")
    b[1].metric("Saved by routing", f"${s['saved_by_routing_usd']:.4f}",
                help="Strong-model price minus the price of the model the router chose, for answers "
                     "from each tier's first-choice model.")
    b[2].metric("Fallback overhead", f"${s['fallback_overhead_usd']:.4f}",
                help="Extra cost of answers served by a backup model, compared with the strong model. "
                     "Positive = backups cost more. The cost guard (reliability.fallback_price_cap_ratio) "
                     "blocks backups pricier than the strong model.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Requests over time, by cache status**")
        t = df.set_index("ts").groupby([pd.Grouper(freq="5min"), "cache_status"]).size().reset_index(name="n")
        st.plotly_chart(px.bar(t, x="ts", y="n", color="cache_status", color_discrete_map=C, height=300)
                        .update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None, xaxis_title="time (UTC)"),
                        use_container_width=True)
    with b:
        st.markdown("**Estimated cost: RouteWise vs always-strong (cumulative)**")
        ok = df[df["status_code"] == 200]
        cum = pd.DataFrame({"ts": ok["ts"], "RouteWise": ok["estimated_cost_usd"].cumsum(),
                            "always-strong (counterfactual)": ok["counterfactual_strong_cost_usd"].cumsum()})
        st.plotly_chart(px.line(cum, x="ts", y=["RouteWise", "always-strong (counterfactual)"], height=300,
                                color_discrete_sequence=["#2a78d6", "#eb6834"])
                        .update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None, yaxis_title="USD",
                                       xaxis_title="time (UTC)"),
                        use_container_width=True)
    c, d = st.columns(2)
    with c:
        st.markdown("**Latency per stage (ms)**")
        st.dataframe(stage_latency(df).round(1), hide_index=True, use_container_width=True)
    with d:
        st.markdown("**Route distribution**")
        rc = pd.DataFrame({"route": list(s["route_counts"]), "n": list(s["route_counts"].values())})
        st.plotly_chart(px.bar(rc, x="route", y="n", color="route", color_discrete_map=C, height=260)
                        .update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False), use_container_width=True)
    st.markdown("**False-hit audit: recent semantic hits** - check that each matched query really means the same thing.")
    st.dataframe(semantic_hits(df), hide_index=True, use_container_width=True)


tab1, tab2 = st.tabs(["Playground", "Dashboard"])
with tab1:
    playground()
with tab2:
    dashboard()
