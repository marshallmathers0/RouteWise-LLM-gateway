"""Build the cold-start demo data from REAL gateway traffic.

Starts the gateway (real providers) with fresh storage under data/demo/, replays
scripts/simulate_traffic.py through it, and stops it. Result:
  data/demo/logs.db            request logs the dashboard shows when live logs are empty
  data/demo/semantic_cache.db  pre-warmed semantic cache (real answers) for the playground

    python scripts/build_demo_data.py --n 100 --rpm 15
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rpm", type=float, default=15)
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()
    DEMO.mkdir(parents=True, exist_ok=True)
    for f in ("logs.db", "semantic_cache.db"):
        (DEMO / f).unlink(missing_ok=True)
    cfg = (ROOT / "config" / "routewise.yaml").read_text(encoding="utf-8").replace(
        "semantic_db_path: data/semantic_cache.db", f"semantic_db_path: {(DEMO / 'semantic_cache.db').as_posix()}")
    cfg_path = DEMO / "_build_config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    env = {**os.environ, "ROUTEWISE_CONFIG": str(cfg_path), "DATABASE_URL": f"sqlite:///{(DEMO / 'logs.db').as_posix()}",
           "REDIS_URL": os.environ.get("DEMO_REDIS_URL", ""), "ROUTEWISE_API_KEYS": "rw-dev-key"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "routewise.main:app", "--port", str(args.port),
                               "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "simulate_traffic.py"), "--url",
                        f"http://127.0.0.1:{args.port}", "--n", str(args.n), "--rpm", str(args.rpm)], check=True)
    finally:
        server.terminate()
        server.wait(timeout=30)
        cfg_path.unlink(missing_ok=True)
    print(f"demo data written to {DEMO}")


if __name__ == "__main__":
    main()
