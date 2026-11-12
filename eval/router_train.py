"""Train router candidates; pick hyperparameters, model and threshold on VALIDATION only.

Candidates
* logreg (handcrafted only) - an interpretable ablation: how far do simple features go?
* logreg                    - embedding + handcrafted, standardized, class_weight="balanced"
* lightgbm                  - gradient-boosted trees, is_unbalance=True
* mlp                       - one hidden layer, balanced sample weights, early stopping

Hyperparameters: small grids, chosen by validation ROC-AUC.
Production model: lowest validation cost at >= 95% of always-strong quality (the metric
we actually care about), ties broken by ROC-AUC. Its threshold is chosen the same way.

Run:  python -m eval.router_train
"""

from __future__ import annotations

import itertools
import json
import warnings

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from eval import mlflow_tracking
from eval.router_common import ARTIFACTS, QUALITY_TARGET, embeddings_for, load_router_data, pick_threshold, route_outcome
from routewise.cache.models import SentenceTransformerEmbedder
from routewise.core.config import CacheConfig
from routewise.router.features import RouterFeaturizer

SEED = 42


def candidates() -> dict[str, tuple[bool, list]]:
    """name -> (uses_embedding, [(params, factory), ...])"""
    lr = [({"C": C}, lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(
        C=C, class_weight="balanced", max_iter=5000, random_state=SEED))) for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]
    lgbm = [({"num_leaves": nl, "n_estimators": ne, "min_child_samples": mcs},
             lambda nl=nl, ne=ne, mcs=mcs: LGBMClassifier(
                 num_leaves=nl, n_estimators=ne, learning_rate=0.05, min_child_samples=mcs, colsample_bytree=0.5,
                 subsample=0.8, subsample_freq=1, is_unbalance=True, random_state=SEED, verbose=-1))
            for nl, ne, mcs in itertools.product((7, 15), (100, 300), (10, 30))]
    mlp = [({"hidden": h, "alpha": a}, lambda h=h, a=a: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=h, alpha=a, early_stopping=True, max_iter=500, random_state=SEED)))
           for h, a in itertools.product(((64,), (128,)), (1e-3, 1e-2, 1e-1))]
    return {"logreg (handcrafted only)": (False, lr), "logreg": (True, lr), "lightgbm": (True, lgbm), "mlp": (True, mlp)}


def fit(model, X, y):
    w = compute_sample_weight("balanced", y)
    last = model.steps[-1][0] if hasattr(model, "steps") else None
    if last == "mlpclassifier":
        model.fit(X, y, mlpclassifier__sample_weight=w)  # MLP has no class_weight option
    else:
        model.fit(X, y)  # LR uses class_weight, LightGBM uses is_unbalance
    return model


def main() -> None:
    warnings.filterwarnings("ignore")
    data = load_router_data()
    embedder = SentenceTransformerEmbedder(CacheConfig().embedding_model)
    emb = embeddings_for(data, embedder)
    tr, va = data.splits == "train", data.splits == "val"
    dva = data.subset(va)
    print(f"train {tr.sum()} / val {va.sum()} / test {(data.splits == 'test').sum()}  "
          f"label=1 rate (train) {data.y[tr].mean():.2f}")
    always_strong_cost = float(dva.c_strong.sum())

    # One MLflow parent run per training session; one nested run per candidate model.
    with mlflow_tracking.run(run_name="router-train") as tracker:
        tracker.set_tags({"git_commit": mlflow_tracking.git_commit(), "stage": "train",
                          "cheap_model": data.models["cheap"], "strong_model": data.models["strong"]})
        tracker.log_params({"n_train": int(tr.sum()), "n_val": int(va.sum()),
                            "n_test": int((data.splits == "test").sum()), "quality_target": QUALITY_TARGET,
                            "seed": SEED, "embedding_model": CacheConfig().embedding_model,
                            "sources": ",".join(sorted(set(data.sources)))})
        tracker.log_metrics({"train_label1_rate": float(data.y[tr].mean()), "val_label1_rate": float(data.y[va].mean()),
                             "val_always_strong_cost_usd": always_strong_cost,
                             "val_always_strong_quality": float(dva.q_strong.mean()),
                             "val_always_cheap_quality": float(dva.q_cheap.mean())})

        results, fitted, features = {}, {}, {}
        for name, (use_emb, grid) in candidates().items():
            feat = RouterFeaturizer(embedder, use_embedding=use_emb)
            X = feat.transform(data.queries, emb if use_emb else None)
            features[name] = X
            best, grid_log = None, []
            for params, factory in grid:
                m = fit(factory(), X[tr], data.y[tr])
                auc = roc_auc_score(data.y[va], m.predict_proba(X[va])[:, 1])
                grid_log.append({**params, "val_auc": round(float(auc), 4)})
                if best is None or auc > best[1]:
                    best = (params, auc, m)
            params, auc, model = best
            p_val = model.predict_proba(X[va])[:, 1]
            thr = pick_threshold(p_val, dva)
            q, c = route_outcome(p_val >= thr, dva)
            results[name] = {"params": params, "val_auc": auc, "threshold": thr, "val_quality": q, "val_cost": c,
                             "use_embedding": use_emb}
            fitted[name] = model
            print(f"{name:26s} val AUC {auc:.3f}  params {params}  thr {thr:.3f}  "
                  f"val quality {q:.3f}  val cost ${c:.4f}")
            with mlflow_tracking.run(run_name=name, nested=True) as child:
                child.set_tags({"candidate": name})
                child.log_params({"use_embedding": use_emb, "n_features": X.shape[1], "best": params,
                                  "grid": json.dumps(grid_log)})
                child.log_metrics({"val_roc_auc": auc, "threshold": thr, "val_quality": q, "val_cost_usd": c,
                                   "val_quality_frac": q / float(dva.q_strong.mean()) if dva.q_strong.mean() else 0.0,
                                   "val_cost_saved_frac": 1 - c / always_strong_cost if always_strong_cost else 0.0,
                                   "val_cheap_frac": float((p_val >= thr).mean())})

        # Production choice: cheapest at the quality target on VAL; ties -> higher AUC.
        chosen = min(results, key=lambda k: (round(results[k]["val_cost"], 8), -results[k]["val_auc"]))
        print(f"chosen: {chosen} (val cost {results[chosen]['val_cost']:.4f} vs always-strong {always_strong_cost:.4f})")

        ARTIFACTS.mkdir(exist_ok=True)
        meta = {"chosen": chosen, "quality_target": QUALITY_TARGET, "candidates": results, "models": data.models,
                "n_train": int(tr.sum()), "n_val": int(va.sum()), "mlflow_run_id": mlflow_tracking.active_run_id()}
        joblib.dump({"model": fitted[chosen], "threshold": results[chosen]["threshold"], "model_name": chosen,
                     "use_embedding": results[chosen]["use_embedding"], "metadata": meta}, ARTIFACTS / "router.joblib")
        # All candidates (gitignored) so the evaluation can draw every model's curve on test.
        joblib.dump({"fitted": fitted, "results": results}, ARTIFACTS / "router_candidates.joblib")
        (ARTIFACTS / "router_meta.json").write_text(json.dumps(meta, indent=2, default=float))

        tracker.set_tags({"chosen_model": chosen})
        tracker.log_metrics({"chosen_val_roc_auc": results[chosen]["val_auc"],
                             "chosen_threshold": results[chosen]["threshold"],
                             "chosen_val_cost_saved_frac": 1 - results[chosen]["val_cost"] / always_strong_cost
                             if always_strong_cost else 0.0})
        tracker.log_artifact(ARTIFACTS / "router.joblib", "artifact")
        tracker.log_artifact(ARTIFACTS / "router_meta.json", "artifact")
        # Registered model: each retrain becomes a new version of "routewise-router".
        tracker.log_sklearn_model(fitted[chosen], features[chosen][va][:5])
        if meta["mlflow_run_id"]:
            print(f"MLflow run {meta['mlflow_run_id']} (browse: mlflow ui --backend-store-uri sqlite:///mlflow.db)")
    print(f"saved {ARTIFACTS / 'router.joblib'}")


if __name__ == "__main__":
    main()
