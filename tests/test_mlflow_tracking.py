"""MLflow experiment tracking: no-op when disabled, nested runs + registry when enabled."""

from __future__ import annotations

import numpy as np
import pytest

from eval import mlflow_tracking


def test_disabled_tracking_is_a_safe_no_op(monkeypatch):
    monkeypatch.setenv("ROUTEWISE_MLFLOW", "0")
    assert not mlflow_tracking.enabled()
    with mlflow_tracking.run(run_name="x") as tracker:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0})
        tracker.log_sklearn_model(object(), None)  # would fail if it actually ran
    assert mlflow_tracking.active_run_id() is None


def test_flat_params():
    assert mlflow_tracking.flat_params({"a": 1, "b": {"c": 2, "d": {"e": "x"}}}) == {"a": "1", "b.c": "2", "b.d.e": "x"}


def test_runs_metrics_and_registry(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setenv("ROUTEWISE_MLFLOW", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("ROUTEWISE_MLFLOW_ARTIFACTS", str(tmp_path / "mlruns"))
    X = np.random.default_rng(0).random((40, 4)).astype("float32")
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression().fit(X, y)

    with mlflow_tracking.run(run_name="router-train") as tracker:
        parent_id = mlflow_tracking.active_run_id()
        tracker.log_params({"n_train": 40, "nested": {"k": 1}})
        tracker.log_metrics({"val_auc": 0.9, "ignored_nan": float("nan"), "ignored_inf": float("inf")})
        with mlflow_tracking.run(run_name="logreg", nested=True) as child:
            child.log_metrics({"val_roc_auc": 0.8})
        tracker.log_sklearn_model(model, X[:5])
    # Evaluation resumes the same run without renaming it.
    with mlflow_tracking.run(run_name="router-eval", run_id=parent_id) as tracker:
        tracker.log_metrics({"test_cost_saved_frac": 0.3})

    client = MlflowClient()
    parent = client.get_run(parent_id)
    assert parent.info.run_name == "router-train"
    assert parent.data.params["nested.k"] == "1"
    assert set(parent.data.metrics) == {"val_auc", "test_cost_saved_frac"}  # NaN/inf dropped
    children = client.search_runs([parent.info.experiment_id], f"tags.mlflow.parentRunId = '{parent_id}'")
    assert [c.info.run_name for c in children] == ["logreg"]
    versions = client.search_model_versions(f"name='{mlflow_tracking.REGISTERED_MODEL}'")
    assert [str(v.version) for v in versions] == ["1"]  # int or str depending on MLflow version
    loaded = mlflow.sklearn.load_model(f"models:/{mlflow_tracking.REGISTERED_MODEL}/1")
    assert np.allclose(loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
