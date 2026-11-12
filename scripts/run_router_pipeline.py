"""Run the whole router pipeline unattended on free-tier quotas:

    collect (waits out daily quotas) -> judge (waits) -> build labels -> train -> evaluate

Every step is resumable, so this script can be stopped and started again at any time.

    python scripts/run_router_pipeline.py            # logs to data/router/pipeline.log
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "router" / "pipeline.log"

STEPS = [
    ("collect", [sys.executable, "-u", "-m", "eval.router_collect", "--wait"]),
    ("judge", [sys.executable, "-u", "-m", "eval.router_label", "judge", "--wait"]),
    ("labels", [sys.executable, "-u", "-m", "eval.router_label", "build"]),
    ("train", [sys.executable, "-u", "-m", "eval.router_train"]),
    ("evaluate", [sys.executable, "-u", "-m", "eval.router_eval"]),
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env_note = {"PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    import os

    env = {**os.environ, **env_note}
    for name, cmd in STEPS:
        log(f"step {name}: {' '.join(cmd[2:])}")
        t0 = time.time()
        with LOG.open("a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"step {name} finished with exit code {rc} after {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            log("stopping: fix the error above, then re-run this script (completed work is kept)")
            sys.exit(rc)
    log("pipeline complete: see reports/router_labels.md and reports/router_eval.md")


if __name__ == "__main__":
    main()
