"""Evaluate routers on the TEST split: classifier metrics, cost-vs-quality, baselines.

Quality of a routing policy = mean per-query quality of the tier it picks (ground-truth
accuracy for GSM8K/MATH/MMLU-Pro, judge score/10 for Dolly/MBPP). Cost = estimated USD
from each tier's actual token usage on that query and the config price table.

Headline: % cost saved vs always-strong while keeping >= 95% of always-strong quality,
using the threshold chosen on VALIDATION (no test-set tuning).

Run:  python -m eval.router_eval        (after python -m eval.router_train)
"""

from __future__ import annotations

import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.router_common import (ARTIFACTS, QUALITY_TARGET, ROOT, embeddings_for, load_router_data, pick_threshold,
                                route_outcome, sweep)
from routewise.cache.models import SentenceTransformerEmbedder
from routewise.core.config import CacheConfig
from routewise.router.features import RouterFeaturizer
from routewise.router.learned import LearnedRouter

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    warnings.filterwarnings("ignore")
    data = load_router_data()
    cand = joblib.load(ARTIFACTS / "router_candidates.joblib")
    chosen_art = joblib.load(ARTIFACTS / "router.joblib")
    chosen = chosen_art["model_name"]
    embedder = SentenceTransformerEmbedder(CacheConfig().embedding_model)
    emb = embeddings_for(data, embedder)
    te, va, tr = data.splits == "test", data.splits == "val", data.splits == "train"
    dte, dva = data.subset(te), data.subset(va)

    q_strong, c_strong = route_outcome(np.zeros(te.sum(), bool), dte)
    q_cheap, c_cheap = route_outcome(np.ones(te.sum(), bool), dte)
    target_q = QUALITY_TARGET * q_strong

    # ---- learned models
    rows, curves, probs = {}, {}, {}
    for name, model in cand["fitted"].items():
        use_emb = cand["results"][name]["use_embedding"]
        X = RouterFeaturizer(embedder, use_embedding=use_emb).transform(data.queries, emb if use_emb else None)
        p = model.predict_proba(X[te])[:, 1]
        probs[name] = p
        thr = cand["results"][name]["threshold"]  # chosen on val
        q, c = route_outcome(p >= thr, dte)
        rows[name] = {
            "roc_auc": roc_auc_score(dte.y, p), "pr_auc_cheap_ok": average_precision_score(dte.y, p),
            "pr_auc_needs_strong": average_precision_score(1 - dte.y, 1 - p),
            "threshold": thr, "quality": q, "cost": c, "cheap_frac": float((p >= thr).mean()),
        }
        curves[name] = sweep(p, dte)

    # ---- baselines
    lengths = np.array([len(q) for q in data.queries], dtype=float)
    len_thr = pick_threshold(-lengths[va], dva)  # score = -length: shorter -> cheap
    q_len, c_len = route_outcome(-lengths[te] >= len_thr, dte)
    curves["length heuristic"] = sweep(-lengths[te], dte)
    oracle_q, oracle_c = route_outcome(dte.y == 1, dte)  # perfect labels: upper bound

    def random_at_cost(cost: float) -> float:
        """Random routing (fraction f to cheap) at the same expected cost: linear mix."""
        if c_strong == c_cheap:
            return q_strong
        f = np.clip((c_strong - cost) / (c_strong - c_cheap), 0, 1)
        return float(f * q_cheap + (1 - f) * q_strong)

    def saved(c: float) -> float:
        return 1 - c / c_strong

    ch = rows[chosen]
    headline = {
        "chosen": chosen, "always_strong_quality": q_strong, "always_cheap_quality": q_cheap,
        "chosen_quality": ch["quality"], "chosen_quality_frac": ch["quality"] / q_strong,
        "chosen_cost_saved": saved(ch["cost"]), "meets_target": ch["quality"] >= target_q,
        "length_quality_frac": q_len / q_strong, "length_cost_saved": saved(c_len),
        "random_quality_frac_at_chosen_cost": random_at_cost(ch["cost"]) / q_strong,
        "always_cheap_cost_saved": saved(c_cheap), "always_cheap_quality_frac": q_cheap / q_strong,
        "oracle_quality_frac": oracle_q / q_strong, "oracle_cost_saved": saved(oracle_c),
    }

    # ---- per-source breakdown at the chosen operating point
    p_ch = probs[chosen]
    per_source = {}
    for src in sorted(set(dte.sources)):
        m = dte.sources == src
        ds = dte.subset(m)
        qr, cr = route_outcome(p_ch[m] >= ch["threshold"], ds)
        qs, cs = route_outcome(np.zeros(m.sum(), bool), ds)
        qc, _ = route_outcome(np.ones(m.sum(), bool), ds)
        per_source[src] = {"n": int(m.sum()), "label1_rate": float(ds.y.mean()), "cheap_frac": float((p_ch[m] >= ch["threshold"]).mean()),
                           "quality_router": qr, "quality_strong": qs, "quality_cheap": qc,
                           "cost_saved": 1 - cr / cs if cs else 0.0}

    # ---- leave-one-source-out (generalization to an unseen query type), logreg
    loso = {}
    Xh = RouterFeaturizer(embedder).transform(data.queries, emb)
    for src in sorted(set(data.sources)):
        trm, tem = tr & (data.sources != src), te & (data.sources == src)
        if len(set(data.y[tem])) < 2:
            loso[src] = None
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000))
        m.fit(Xh[trm], data.y[trm])
        loso[src] = float(roc_auc_score(data.y[tem], m.predict_proba(Xh[tem])[:, 1]))

    # ---- router inference latency (embedding + features + predict), single query
    router = LearnedRouter(chosen_art, embedder)
    router.decide("warm up")
    lat = []
    for q in dte.queries[:100]:
        t = time.perf_counter()
        router.decide(q)
        lat.append((time.perf_counter() - t) * 1000)
    latency = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)), "n": len(lat)}

    # ---- plots
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURFACE)
    _style(ax)
    for i, (name, cv) in enumerate(curves.items()):
        pts = sorted(((p["cost"] / c_strong, p["quality"] / q_strong) for p in cv))
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=PALETTE[i], linewidth=2 if name == chosen else 1.4,
                label=name + (" (chosen)" if name == chosen else ""))
    ax.plot([c_cheap / c_strong, 1], [q_cheap / q_strong, 1], color=INK2, linestyle="--", linewidth=1, label="random routing")
    ax.plot([oracle_c / c_strong], [oracle_q / q_strong], "*", color=INK, markersize=11, label="oracle (perfect labels)")
    ax.plot([ch["cost"] / c_strong], [ch["quality"] / q_strong], "o", color=PALETTE[list(curves).index(chosen)],
            markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(QUALITY_TARGET, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(0.01, QUALITY_TARGET + 0.004, f"{QUALITY_TARGET:.0%} of always-strong quality", color=INK2, fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("estimated cost (fraction of always-strong)", color=INK2)
    ax.set_ylabel("quality (fraction of always-strong)", color=INK2)
    ax.set_title("Cost vs quality on the test split (dot = val-chosen operating point)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "router_cost_quality.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5), facecolor=SURFACE)
    _style(ax)
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(dte.y, p)
        ax.plot(fpr, tpr, color=PALETTE[i], linewidth=1.8, label=f"{name} (AUC {rows[name]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, linestyle="--", linewidth=0.8)
    ax.set_xlabel("false positive rate", color=INK2)
    ax.set_ylabel("true positive rate", color=INK2)
    ax.set_title("ROC: predicting 'cheap is good enough' (test)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "router_roc.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    summary = {"headline": headline, "models": rows, "per_source": per_source, "leave_one_source_out_auc": loso,
               "latency_ms": latency, "test_n": int(te.sum()), "models_used": data.models,
               "always_strong_cost_usd": c_strong, "always_cheap_cost_usd": c_cheap,
               "length_threshold_chars": float(-len_thr) if np.isfinite(len_thr) else None,
               "val_selection": cand["results"]}
    (REPORTS / "router_eval.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, data)
    print(json.dumps(headline, indent=2, default=float))


def write_report(s: dict, data) -> None:
    h, rows = s["headline"], s["models"]
    L: list[str] = []
    w = L.append
    w("# Router evaluation\n")
    w("_Generated by `python -m eval.router_eval`. All numbers come from this run on the held-out test split._\n")
    w("## Setup\n")
    w(f"* Cheap tier: `{s['models_used']['cheap']}` - strong tier: `{s['models_used']['strong']}`.")
    n_split = {sp: int((data.splits == sp).sum()) for sp in ("train", "val", "test")}
    w(f"* Queries: {len(data.ids)} labeled (train {n_split['train']} / val {n_split['val']} / test {n_split['test']}), "
      f"label=1 (cheap good enough) for {pct(data.y.mean())}.")
    w("* Features: bge-small query embedding (384-d) + 29 handcrafted features. The dataset `source` is never a feature.")
    w("* Hyperparameters chosen by validation ROC-AUC; each model's threshold = cheapest point on **validation** "
      f"keeping >= {QUALITY_TARGET:.0%} of always-strong quality. The production model is the cheapest such model on validation.")
    w("* Cost = estimated USD from each tier's real token usage per query and the list prices in `config/routewise.yaml`.\n")

    w("## Headline (test)\n")
    w(f"**{h['chosen']}** routes {pct(rows[h['chosen']]['cheap_frac'])} of test queries to the cheap model, "
      f"**saving {pct(h['chosen_cost_saved'])} of estimated cost** while keeping **{pct(h['chosen_quality_frac'])}** "
      f"of always-strong quality (target {QUALITY_TARGET:.0%}: {'met' if h['meets_target'] else 'NOT met'} on test).\n")
    w("| Policy | Quality (vs always-strong) | Cost saved vs always-strong |")
    w("|---|---|---|")
    w(f"| always-strong | 100.0% | 0.0% |")
    w(f"| always-cheap | {pct(h['always_cheap_quality_frac'])} | {pct(h['always_cheap_cost_saved'])} |")
    w(f"| length heuristic (threshold from val) | {pct(h['length_quality_frac'])} | {pct(h['length_cost_saved'])} |")
    w(f"| random routing at the chosen router's cost | {pct(h['random_quality_frac_at_chosen_cost'])} | {pct(h['chosen_cost_saved'])} |")
    for name, r in rows.items():
        w(f"| {name}{' **(chosen)**' if name == h['chosen'] else ''} | {pct(r['quality'] / h['always_strong_quality'])} | "
          f"{pct(1 - r['cost'] / s['always_strong_cost_usd'])} |")
    w(f"| oracle (perfect labels, upper bound) | {pct(h['oracle_quality_frac'])} | {pct(h['oracle_cost_saved'])} |")
    w(f"\nAbsolute: always-strong quality {h['always_strong_quality']:.3f}, always-cheap {h['always_cheap_quality']:.3f}; "
      f"always-strong cost ${s['always_strong_cost_usd']:.4f} vs always-cheap ${s['always_cheap_cost_usd']:.4f} for the "
      f"{s['test_n']} test queries (estimated).\n")
    w("![cost vs quality](figures/router_cost_quality.png)\n")

    w("## Classifier metrics (test)\n")
    w("PR-AUC is reported for both classes; `needs strong` (label 0) is the minority class and the one a router must catch.\n")
    w("| Model | ROC-AUC | PR-AUC (cheap OK) | PR-AUC (needs strong) | Val-chosen threshold |")
    w("|---|---|---|---|---|")
    for name, r in rows.items():
        w(f"| {name} | {r['roc_auc']:.3f} | {r['pr_auc_cheap_ok']:.3f} | {r['pr_auc_needs_strong']:.3f} | {r['threshold']:.3f} |")
    w("\n![ROC](figures/router_roc.png)\n")

    w("## Per-source breakdown (chosen router, test)\n")
    w("| Source | n | label=1 rate | routed to cheap | quality: router / strong / cheap | cost saved |")
    w("|---|---|---|---|---|---|")
    for src, r in s["per_source"].items():
        w(f"| {src} | {r['n']} | {pct(r['label1_rate'])} | {pct(r['cheap_frac'])} | "
          f"{r['quality_router']:.3f} / {r['quality_strong']:.3f} / {r['quality_cheap']:.3f} | {pct(r['cost_saved'])} |")
    w("\n## Leave-one-source-out ROC-AUC (logreg)\n")
    w("Train on the other sources, test on the held-out one: how well does the router generalize to a query type "
      "it has never seen? Expect this to be lower than the in-distribution AUC.\n")
    w("| Held-out source | ROC-AUC |")
    w("|---|---|")
    for src, auc in s["leave_one_source_out_auc"].items():
        w(f"| {src} | {'n/a (one class)' if auc is None else f'{auc:.3f}'} |")
    lat = s["latency_ms"]
    w(f"\n## Router inference latency\n\nSingle query on CPU (embedding + features + model): "
      f"p50 **{lat['p50_ms']:.1f} ms**, p95 {lat['p95_ms']:.1f} ms (n={lat['n']}). The embedding dominates; it is "
      "shared with the semantic cache, so in the gateway it is computed once per request.\n")
    w("## Caveats\n")
    w("* Open-ended labels come from an LLM judge; see `reports/router_labels.md` for its validation and position-bias check.")
    w("* The test split is small, so differences of a few points between models are within noise.")
    w("* Costs are estimates from list prices; free-tier calls actually cost $0.")
    (REPORTS / "router_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
