"""Semantic-cache ablation: cosine only / + rules / + verifier / all three.

Pipeline
--------
1. Score every pair once (cached): cosine (bi-encoder), each rule guard, verifier score.
2. Pick thresholds on the DEV split only:
   * candidate (recall) cosine threshold: the highest threshold that still keeps >= 95%
     of dev duplicates. Configs with a verifier use it as their loose stage-1 filter.
   * each config's operating point: the value of its free knob (cosine threshold for
     cosine-only / +rules; verifier threshold for configs with a verifier) maximizing
     F0.5 on dev. F0.5 weights precision 2x recall: a false hit serves a wrong answer,
     a missed hit only costs one LLM call.
3. Report on the TEST split with 95% bootstrap confidence intervals.

Metrics
-------
* hit rate        = served duplicates / all duplicates        (recall: the savings)
* false-hit rate  = served non-duplicates / all served pairs  (1 - precision: wrong answers)
* false-serve rate (per near-miss category) = served non-duplicates / non-duplicates

Run:  python -m eval.cache_eval        (after python -m eval.build_cache_pairs)
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routewise.cache.guards import ALL_GUARDS, run_guards
from routewise.cache.models import CrossEncoderVerifier, SentenceTransformerEmbedder
from routewise.core.config import CacheConfig

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "eval" / "cache_pairs.jsonl"
AUDIT = ROOT / "eval" / "data" / "qqp_hard_negative_audit.jsonl"
SCORES = ROOT / "data" / "eval" / "cache_scores.jsonl"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

RECALL_TARGET = 0.95
BETA = 0.5
N_BOOT = 1000
SEED = 42

CONFIGS = {  # name -> (guards, verifier)
    "cosine only": (False, False),
    "cosine + rules": (True, False),
    "cosine + verifier": (False, True),
    "all three": (True, True),
}
# Reference palette, categorical slots 1-4 (validated for adjacent series on light surface).
COLORS = {"cosine only": "#2a78d6", "cosine + rules": "#eb6834", "cosine + verifier": "#1baf7a", "all three": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- scoring (expensive, cached)
def score_pairs(pairs: pd.DataFrame, emb: SentenceTransformerEmbedder, ver: CrossEncoderVerifier) -> pd.DataFrame:
    cached: dict[str, dict] = {}
    if SCORES.exists():
        cached = {r["id"]: r for r in map(json.loads, SCORES.open(encoding="utf-8"))}
    todo = pairs[~pairs["id"].isin(cached.keys())]
    if len(todo):
        print(f"scoring {len(todo):,} pairs with the embedder and verifier ...")
        a = emb.embed(todo["q1"].tolist())
        b = emb.embed(todo["q2"].tolist())
        cos = (a * b).sum(axis=1)
        vs = ver.score(list(zip(todo["q1"], todo["q2"])))
        for id_, c, v in zip(todo["id"], cos, vs):
            cached[id_] = {"id": id_, "cos": float(c), "ver": float(v)}
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        with SCORES.open("w", encoding="utf-8") as f:
            for row in cached.values():
                f.write(json.dumps({k: row[k] for k in ("id", "cos", "ver")}) + "\n")
    # Guards are cheap and are the part we iterate on, so always recompute them.
    model_scores = pd.DataFrame([{k: r[k] for k in ("id", "cos", "ver")} for r in cached.values()])
    df = pairs.merge(model_scores, on="id", how="left")
    guard_res = [run_guards(q1, q2, ALL_GUARDS).results for q1, q2 in zip(df["q1"], df["q2"])]
    for g in ALL_GUARDS:
        df[f"g_{g}"] = [r[g] for r in guard_res]
    df["guards_pass"] = df[[f"g_{g}" for g in ALL_GUARDS]].all(axis=1)
    df["subset"] = np.where(df["source"] == "near_miss", "near_miss", "qqp")
    return df


# ---------------------------------------------------------------- decision + metrics
def served(df: pd.DataFrame, guards: bool, verifier: bool, t_cos: float, t_ver: float) -> np.ndarray:
    s = df["cos"].to_numpy() >= t_cos
    if guards:
        s &= df["guards_pass"].to_numpy()
    if verifier:
        s &= df["ver"].to_numpy() >= t_ver
    return s


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    tp = int((s & (y == 1)).sum())
    fp = int((s & (y == 0)).sum())
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    hit = tp / pos if pos else float("nan")
    fhr = fp / (tp + fp) if (tp + fp) else float("nan")
    prec = 1 - fhr if (tp + fp) else float("nan")
    fb = ((1 + BETA**2) * prec * hit / (BETA**2 * prec + hit)) if (tp + fp) and (prec + hit) > 0 else 0.0
    return {"tp": tp, "fp": fp, "served": tp + fp, "positives": pos, "negatives": neg,
            "hit_rate": hit, "false_hit_rate": fhr, "false_serve_rate": fp / neg if neg else float("nan"), "f05": fb}


def bootstrap_ci(y: np.ndarray, s: np.ndarray, key: str, rng: np.random.Generator) -> tuple[float, float]:
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        v = metrics(y[idx], s[idx])[key]
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def knob_grid(df: pd.DataFrame, verifier: bool) -> np.ndarray:
    col = "ver" if verifier else "cos"
    # Every distinct score is a possible threshold; subsample for speed but keep extremes.
    vals = np.unique(np.round(df[col].to_numpy(), 4))
    if len(vals) > 600:
        vals = np.unique(np.concatenate([np.quantile(vals, np.linspace(0, 1, 600)), vals[-20:]]))
    return vals


def curve(df: pd.DataFrame, guards: bool, verifier: bool, t_recall: float) -> pd.DataFrame:
    y = df["label"].to_numpy()
    rows = []
    for t in knob_grid(df, verifier):
        t_cos, t_ver = (t_recall, t) if verifier else (t, 0.0)
        m = metrics(y, served(df, guards, verifier, t_cos, t_ver))
        rows.append({"threshold": float(t), "t_cos": t_cos, "t_ver": t_ver, **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- latency
def measure_latency(df: pd.DataFrame, emb, ver, n: int = 100) -> dict:
    qs = df["q2"].tolist()[:n]
    pairs = list(zip(df["q1"], df["q2"]))[:n]
    emb.embed(["warm up"]), ver.score([("warm", "up")]), run_guards("warm", "up")

    def timed(fn, items) -> list[float]:
        out = []
        for it in items:
            t = time.perf_counter()
            fn(it)
            out.append((time.perf_counter() - t) * 1000)
        return out

    def p(xs, q):
        return round(float(np.percentile(xs, q)), 2)

    res = {}
    for name, fn, items in [
        ("embed_1_query", lambda q: emb.embed([q]), qs),
        ("guards_1_pair", lambda pr: run_guards(*pr), pairs),
        ("verifier_1_pair", lambda pr: ver.score([pr]), pairs),
        ("verifier_batch_5", lambda i: ver.score(pairs[i:i + 5]), list(range(0, n - 5, 2))),
    ]:
        xs = timed(fn, items)
        res[name] = {"p50_ms": p(xs, 50), "p95_ms": p(xs, 95), "n": len(xs)}
    return res


# ---------------------------------------------------------------- plots
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr(curves: dict[str, dict[str, pd.DataFrame]], ops: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, subset, title in [(axes[0], "qqp", "QQP test (duplicates vs mined hard negatives)"),
                              (axes[1], "near_miss", "Near-miss test (hand-built traps)")]:
        _style(ax)
        for name, c in curves[subset].items():
            c = c.dropna(subset=["false_hit_rate"]).sort_values("hit_rate")
            ax.plot(c["hit_rate"], 1 - c["false_hit_rate"], color=COLORS[name], linewidth=2, label=name)
            op = ops[name]["test"][subset]
            ax.plot(op["hit_rate"], 1 - op["false_hit_rate"], "o", markersize=8, color=COLORS[name],
                    markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("hit rate (recall)", color=INK2)
        ax.set_ylabel("precision = 1 - false-hit rate", color=INK2)
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0.3, 1.01)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.text(0.01, 0.01, "Dots = operating points chosen on the dev split (max F0.5).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_categories(cat: pd.DataFrame, path: Path) -> None:
    cats = cat.index.tolist()
    names = list(CONFIGS)
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor=SURFACE)
    _style(ax)
    width = 0.2
    x = np.arange(len(cats))
    ax.grid(axis="x", visible=False)
    top = max(10.0, float(cat.to_numpy().max()) * 100)
    for i, name in enumerate(names):
        vals = cat[name].to_numpy() * 100
        bars = ax.bar(x + (i - 1.5) * width, vals, width - 0.03, color=COLORS[name], label=name)
        for b, v in zip(bars, vals):
            # Label zeros too, so "0%" is visibly a result rather than a missing bar.
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [c.replace("_", " ") for c in cats], fontsize=9, color=INK2)
    ax.set_ylabel("% of trap pairs wrongly served", color=INK2)
    ax.set_title("False serves by near-miss category (test split, dev-chosen operating points)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=4, loc="upper left")
    ax.set_ylim(0, top * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main
def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def main() -> None:
    pairs = pd.read_json(PAIRS, lines=True)
    cfg = CacheConfig()
    print("loading models ...")
    emb, ver = SentenceTransformerEmbedder(cfg.embedding_model), CrossEncoderVerifier(cfg.verifier_model)
    df = score_pairs(pairs, emb, ver)
    dev, test = df[df["split"] == "dev"], df[df["split"] == "test"]
    rng = np.random.default_rng(SEED)

    # 1. Candidate (recall) threshold on dev.
    dev_pos_cos = np.sort(dev.loc[dev["label"] == 1, "cos"].to_numpy())
    t_recall = float(np.floor(np.quantile(dev_pos_cos, 1 - RECALL_TARGET) * 1000) / 1000)
    print(f"candidate cosine threshold (keeps {RECALL_TARGET:.0%} of dev duplicates): {t_recall}")

    # 2. Operating points on dev; 3. evaluate on test.
    ops: dict[str, dict] = {}
    curves_test: dict[str, dict[str, pd.DataFrame]] = {"qqp": {}, "near_miss": {}}
    for name, (g, v) in CONFIGS.items():
        dc = curve(dev, g, v, t_recall)
        best = dc.loc[dc["f05"].idxmax()]
        t_cos, t_ver = float(best["t_cos"]), float(best["t_ver"])
        res = {"t_cos": t_cos, "t_ver": t_ver if v else None, "dev_all": metrics(dev["label"].to_numpy(),
               served(dev, g, v, t_cos, t_ver)), "test": {}}
        for subset, part in [("all", test), ("qqp", test[test.subset == "qqp"]), ("near_miss", test[test.subset == "near_miss"])]:
            y, s = part["label"].to_numpy(), served(part, g, v, t_cos, t_ver)
            m = metrics(y, s)
            m["hit_rate_ci"] = bootstrap_ci(y, s, "hit_rate", rng)
            m["false_hit_rate_ci"] = bootstrap_ci(y, s, "false_hit_rate", rng)
            res["test"][subset] = m
        ops[name] = res
        for subset in ("qqp", "near_miss"):
            curves_test[subset][name] = curve(test[test.subset == subset], g, v, t_recall)

    # Matched-hit-rate comparison: cosine-only threshold giving the same DEV hit rate as "all three".
    target_hit = ops["all three"]["dev_all"]["hit_rate"]
    dc = curve(dev, False, False, t_recall)
    t_match = float(dc.loc[(dc["hit_rate"] - target_hit).abs().idxmin(), "threshold"])
    matched = {}
    for subset, part in [("all", test), ("qqp", test[test.subset == "qqp"]), ("near_miss", test[test.subset == "near_miss"])]:
        matched[subset] = metrics(part["label"].to_numpy(), served(part, False, False, t_match, 0))

    # Per-category false-serve rate (test near-miss).
    nm = test[test.subset == "near_miss"]
    cat_rows = {}
    for name, (g, v) in CONFIGS.items():
        s = served(nm, g, v, ops[name]["t_cos"], ops[name]["t_ver"] or 0)
        cat_rows[name] = {c: float(s[(nm["category"] == c).to_numpy()].mean()) for c in sorted(nm["category"].unique())}
    cat = pd.DataFrame(cat_rows)
    cat_neg = cat.drop(index="paraphrase")  # paraphrase row = hit rate, reported separately

    # Per-guard contribution among stage-1 candidates (all splits, cos >= t_recall).
    cand = df[df["cos"] >= t_recall]
    guard_rows = []
    for gname in ALL_GUARDS:
        col = f"g_{gname}"
        row = {"guard": gname}
        for subset in ("qqp", "near_miss"):
            part = cand[cand.subset == subset]
            negs, poss = part[part.label == 0], part[part.label == 1]
            row[f"{subset}_neg_caught"] = f"{int((~negs[col]).sum())}/{len(negs)}"
            row[f"{subset}_pos_lost"] = f"{int((~poss[col]).sum())}/{len(poss)}"
        guard_rows.append(row)

    # Label-noise audit (LLM-assisted) of QQP hard negatives.
    audit = pd.read_json(AUDIT, lines=True)
    audit_counts = audit["audit"].value_counts().to_dict()

    latency = measure_latency(test, emb, ver)

    # ---- outputs
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_pr(curves_test, ops, FIGS / "cache_pr_curves.png")
    plot_categories(cat_neg, FIGS / "cache_category_false_serves.png")

    summary = {"t_recall": t_recall, "recall_target": RECALL_TARGET, "beta": BETA, "operating_points": ops,
               "matched_cosine_only": {"t_cos": t_match, "test": matched}, "category_false_serve_rate": cat_rows,
               "latency_ms": latency, "audit": audit_counts,
               "counts": df.groupby(["split", "source", "label"]).size().rename("n").reset_index().to_dict("records")}
    (REPORTS / "cache_eval.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, cat_neg, cat, guard_rows, df)
    print(f"wrote {REPORTS / 'cache_eval.md'}")


def write_report(s: dict, cat_neg: pd.DataFrame, cat: pd.DataFrame, guard_rows: list[dict], df: pd.DataFrame) -> None:
    ops, matched = s["operating_points"], s["matched_cosine_only"]
    L: list[str] = []
    w = L.append
    w("# Semantic cache evaluation\n")
    w("_Generated by `python -m eval.cache_eval` - do not edit by hand. All numbers come from this run._\n")
    w("## Setup\n")
    n = df.groupby(["source", "label"]).size()
    w(f"* **Pairs:** {len(df):,} total - QQP duplicates {n[('qqp_positive', 1)]}, QQP mined hard negatives "
      f"{n[('qqp_hard_negative', 0)]}, near-miss traps {n[('near_miss', 0)]}, near-miss paraphrases {n[('near_miss', 1)]}. "
      "Split 50/50 into dev (choose thresholds) and test (report), stratified by source/category/label.")
    w("* **Embedder:** `BAAI/bge-small-en-v1.5`; **verifier:** `cross-encoder/quora-distilroberta-base`; "
      "**guards:** numbers, negation, entities (spaCy `en_core_web_sm`), word order.")
    w(f"* **Stage-1 candidate threshold:** cosine >= **{s['t_recall']}** (keeps {s['recall_target']:.0%} of dev duplicates). "
      "Verifier configs use it as their loose recall filter, then tune the verifier threshold.")
    w("* **Operating point** per config: the knob value maximizing **F0.5 on dev** (precision weighted 2x: "
      "a false hit serves a wrong answer, a miss only costs one LLM call).")
    w("* The pair mix is deliberately **adversarial** (only hard negatives, no easy ones), so absolute "
      "false-hit rates are a worst case - typical traffic is far easier.\n")

    w("## Headline (test split)\n")
    a, c = ops["all three"]["test"], ops["cosine only"]["test"]
    w("| Config | Threshold(s) | Hit rate (all) | False-hit rate (all) | Hit rate (near-miss) | False-hit rate (near-miss) | Hit rate (QQP) | False-hit rate (QQP) |")
    w("|---|---|---|---|---|---|---|---|")
    for name, o in ops.items():
        th = f"cos>={o['t_cos']:.3f}" + (f", ver>={o['t_ver']:.3f}" if o["t_ver"] is not None else "")
        t = o["test"]
        w(f"| {name} | {th} | {pct(t['all']['hit_rate'])} | {pct(t['all']['false_hit_rate'])} | "
          f"{pct(t['near_miss']['hit_rate'])} | {pct(t['near_miss']['false_hit_rate'])} | "
          f"{pct(t['qqp']['hit_rate'])} | {pct(t['qqp']['false_hit_rate'])} |")
    mt = matched["test"]
    w(f"| cosine only, **matched hit rate** | cos>={matched['t_cos']:.3f} | {pct(mt['all']['hit_rate'])} | "
      f"{pct(mt['all']['false_hit_rate'])} | {pct(mt['near_miss']['hit_rate'])} | {pct(mt['near_miss']['false_hit_rate'])} | "
      f"{pct(mt['qqp']['hit_rate'])} | {pct(mt['qqp']['false_hit_rate'])} |")
    w("\n95% bootstrap CIs on the test split:\n")
    w("| Config | All: hit rate | All: false-hit rate | Near-miss: served (wrong) | Near-miss: false-hit rate |")
    w("|---|---|---|---|---|")
    for name, o in ops.items():
        t, nm = o["test"]["all"], o["test"]["near_miss"]
        hc, fc, nfc = t["hit_rate_ci"], t["false_hit_rate_ci"], nm["false_hit_rate_ci"]
        w(f"| {name} | {pct(t['hit_rate'])} [{pct(hc[0])}, {pct(hc[1])}] | "
          f"{pct(t['false_hit_rate'])} [{pct(fc[0])}, {pct(fc[1])}] | {nm['served']} ({nm['fp']}) | "
          f"{pct(nm['false_hit_rate'])} [{pct(nfc[0])}, {pct(nfc[1])}] |")
    w("\nThe near-miss set is small, so its intervals are wide and overlap between neighbouring configs.\n")
    w(f"**Reading it:** on the near-miss traps, naive cosine caching (at its own best threshold) had a "
      f"{pct(c['near_miss']['false_hit_rate'])} false-hit rate at a {pct(c['near_miss']['hit_rate'])} hit rate; "
      f"the full verified cache had {pct(a['near_miss']['false_hit_rate'])} while keeping "
      f"{pct(a['near_miss']['hit_rate'])} of paraphrase hits. Tuning cosine alone to the same hit rate as the full "
      f"system leaves {pct(mt['near_miss']['false_hit_rate'])} false hits on near-misses.\n")

    w("![precision-recall curves](figures/cache_pr_curves.png)\n")

    cv = ops["cosine + verifier"]["test"]
    w("## Chosen production operating point\n")
    w(f"**All three stages: cosine >= {ops['all three']['t_cos']:.3f}, guards on, verifier >= {ops['all three']['t_ver']:.3f}** "
      "(written to `config/routewise.yaml`). Why this and not `cosine + verifier`, which wins on QQP:\n")
    w(f"* On QQP, `cosine + verifier` serves more duplicates ({pct(cv['qqp']['hit_rate'])} vs "
      f"{pct(a['qqp']['hit_rate'])}) at a similar false-hit rate ({pct(cv['qqp']['false_hit_rate'])} vs "
      f"{pct(a['qqp']['false_hit_rate'])}, overlapping CIs). But the verifier was trained on QQP (caveat 1), "
      "so this comparison flatters it.")
    w(f"* On the near-miss traps - data the verifier never saw - it alone lets through "
      f"{pct(cv['near_miss']['false_hit_rate'])} wrong answers; adding the guards cuts that to "
      f"{pct(a['near_miss']['false_hit_rate'])}, mostly unit/direction swaps and negations it scores as duplicates.")
    w("* The guards and the verifier fail differently (the guards are lexical and blunt, the verifier is "
      "semantic but fooled by swaps), which is why stacking them helps on unseen data.")
    w("* The price is hit rate: the guards reject some true QQP duplicates (table below), partly because "
      "QQP's 'same intent' duplicates often differ in a number or entity (\"best laptop under 25k\" vs "
      "\"best laptop under 25k in India\"), where serving one cached answer isn't actually safe. "
      "For a cache, wrong answers are worse than misses, so we accept that.\n")
    w("## Near-miss categories (test)\n")
    w("Share of each category's trap pairs that were wrongly served (lower is better). "
      "The `paraphrase` row is the hit rate on true paraphrases (higher is better).\n")
    w("| Category | " + " | ".join(cat.columns) + " |")
    w("|---|" + "---|" * len(cat.columns))
    for idx, row in cat.iterrows():
        w(f"| {idx.replace('_', ' ')} | " + " | ".join(pct(v) for v in row) + " |")
    w("\n![false serves by category](figures/cache_category_false_serves.png)\n")

    w("## What each guard does (stage-1 candidates, dev + test)\n")
    w("`neg caught` = true non-duplicates the guard rejects (good). `pos lost` = true duplicates it wrongly "
      "rejects (each costs one cache miss).\n")
    w("| Guard | QQP neg caught | QQP pos lost | Near-miss neg caught | Near-miss pos lost |")
    w("|---|---|---|---|---|")
    for r in guard_rows:
        w(f"| {r['guard']} | {r['qqp_neg_caught']} | {r['qqp_pos_lost']} | {r['near_miss_neg_caught']} | {r['near_miss_pos_lost']} |")
    w("")

    w("## Latency overhead (CPU, single request)\n")
    w("| Step | p50 (ms) | p95 (ms) |")
    w("|---|---|---|")
    for k, v in s["latency_ms"].items():
        w(f"| {k.replace('_', ' ')} | {v['p50_ms']} | {v['p95_ms']} |")
    lat = s["latency_ms"]
    w(f"\nA full lookup with top-k=5 costs roughly embed + guards x5 + one verifier batch of 5 "
      f"≈ {lat['embed_1_query']['p50_ms'] + 5 * lat['guards_1_pair']['p50_ms'] + lat['verifier_batch_5']['p50_ms']:.0f} ms p50 "
      "(upper bound: guards usually prune candidates before the verifier). An LLM call is typically 300-3000 ms.\n")

    w("## Caveats (read these)\n")
    au = s["audit"]
    w(f"1. **Verifier leakage on QQP.** The cross-encoder was trained on Quora duplicate questions, so its QQP "
      "numbers are optimistic. The near-miss set (not from QQP) is the fairer test of the verifier.")
    w(f"2. **QQP label noise.** An LLM-assisted audit (Claude, not a human) of 40 sampled hard negatives judged "
      f"{au.get('actually_duplicate', 0)} to be real duplicates and {au.get('borderline', 0)} borderline "
      "(`eval/data/qqp_hard_negative_audit.jsonl`), i.e. roughly "
      f"{100 * au.get('actually_duplicate', 0) / 40:.0f}-{100 * (au.get('actually_duplicate', 0) + au.get('borderline', 0)) / 40:.0f}% "
      "of these 'negatives' are mislabeled. That puts a floor under every config's QQP false-hit rate.")
    w("3. **The near-miss set was drafted by an LLM** and its labels have **not yet been human-verified** "
      "(`eval/data/near_miss_pairs.jsonl`, `verified: false`). It is also small (100 test pairs): see the CIs.")
    w("4. **Same author for traps and guards.** The near-miss categories and the rule guards were designed by "
      "the same process, so the guards are likely to look better on this set than on independent traps. "
      "Human-written or crowd-sourced near-misses would be a stronger test.")
    w("5. **The guards were revised twice.** After the first evaluation run, error analysis found bugs (\"one\" "
      "parsed as a number, case-sensitive entity matching, every proper noun treated as an entity); later the "
      "playground showed spaCy tagging \"Convert 5 miles\" as a PERSON, so digit-bearing entities are now ignored. "
      "The fixes are generic, but the first error analysis looked at pairs from both splits, so test numbers may "
      "be slightly optimistic.")
    w("6. **Thresholds are not universal.** They depend on the embedder; re-run this eval after changing models.")
    (REPORTS / "cache_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
