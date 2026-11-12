"""Experiment tracking for the router with MLflow (optional, never breaks training).

Every `python -m eval.router_train` run becomes one MLflow **parent run** in the experiment
`routewise-router`, with one **nested run per candidate model** (its hyperparameter grid,
best params, validation AUC, chosen threshold, validation quality/cost). The production
router is logged as an MLflow model and **registered** as `routewise-router`, so every
retrain becomes a new registry version. `python -m eval.router_eval` then attaches the
test-set metrics, report and plots to the same parent run.

Storage: a local SQLite file (`mlflow.db`) plus `mlruns/` for artifacts - free, no server.
Override with the standard `MLFLOW_TRACKING_URI`; set `ROUTEWISE_MLFLOW=0` to disable.
Browse runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (then http://localhost:5000)
"""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "routewise-router"
REGISTERED_MODEL = "routewise-router"
log = logging.getLogger(__name__)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _mlflow():
    if os.environ.get("ROUTEWISE_MLFLOW", "1") == "0":
        return None
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def enabled() -> bool:
    return _mlflow() is not None


def setup(tracking_uri: str | None = None) -> Any:
    """Point MLflow at the project's local store and select the experiment."""
    mlflow = _mlflow()
    if mlflow is None:
        return None
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        artifacts = Path(os.environ.get("ROUTEWISE_MLFLOW_ARTIFACTS", ROOT / "mlruns"))
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def flat_params(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """MLflow params are flat strings; flatten nested dicts as 'a.b'."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat_params(v, key + "."))
        else:
            out[key] = str(v)[:500]
    return out


class Tracker:
    """Thin wrapper whose methods become no-ops when MLflow is unavailable or fails."""

    def __init__(self, mlflow: Any | None) -> None:
        self.mlflow = mlflow

    def _safe(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        if self.mlflow is None:
            return None
        try:
            return getattr(self.mlflow, fn_name)(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - tracking must never break training
            log.warning("MLflow %s failed: %s", fn_name, e)
            return None

    def log_params(self, params: dict[str, Any]) -> None:
        self._safe("log_params", flat_params(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v and abs(float(v)) != float("inf")}
        self._safe("log_metrics", clean)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if Path(path).exists():
            self._safe("log_artifact", str(path), artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._safe("set_tags", {k: str(v) for k, v in tags.items() if v is not None})

    def log_sklearn_model(self, model: Any, X_example: Any, register: bool = True) -> None:
        if self.mlflow is None:
            return
        try:
            import mlflow.sklearn
            from mlflow.models import infer_signature

            signature = infer_signature(X_example, model.predict_proba(X_example)[:, 1])
            mlflow.sklearn.log_model(model, name="router", signature=signature,
                                     skops_trusted_types=_library_types(model),
                                     registered_model_name=REGISTERED_MODEL if register else None)
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow model logging failed: %s", e)


_TRUSTED_LIBRARIES = ("sklearn", "lightgbm", "numpy", "scipy")


def _library_types(model: Any) -> list[str] | None:
    """Types skops should trust when MLflow saves this model.

    MLflow 3 saves scikit-learn models with skops instead of pickle (unpickling can run
    arbitrary code) and refuses any type not on skops' safe list - e.g. the MLP's internal
    `sklearn.neural_network._stochastic_optimizers.AdamOptimizer`. We trust only types that
    come from the ML libraries themselves; anything else would still be refused.
    """
    try:
        import skops.io as sio

        untrusted = sio.get_untrusted_types(data=sio.dumps(model))
        trusted = [t for t in untrusted if t.split(".")[0] in _TRUSTED_LIBRARIES]
        # LightGBM's saved state also holds a plain standard-library ordered dict.
        return trusted + ["collections.OrderedDict"]
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def run(run_name: str | None = None, run_id: str | None = None, nested: bool = False,
        tracking_uri: str | None = None) -> Iterator[Tracker]:
    """Start (or resume, with run_id) an MLflow run; yields a Tracker (no-op if disabled)."""
    mlflow = setup(tracking_uri) if not nested else _mlflow()
    if mlflow is None:
        yield Tracker(None)
        return
    try:
        # Resuming an existing run (run_id) must not pass run_name, or MLflow renames it.
        ctx = (mlflow.start_run(run_id=run_id) if run_id
               else mlflow.start_run(run_name=run_name, nested=nested))
    except Exception as e:  # noqa: BLE001
        log.warning("MLflow start_run failed (%s); continuing without tracking", e)
        yield Tracker(None)
        return
    with ctx:
        yield Tracker(mlflow)


def active_run_id() -> str | None:
    mlflow = _mlflow()
    if mlflow is None:
        return None
    r = mlflow.active_run()
    return r.info.run_id if r else None
