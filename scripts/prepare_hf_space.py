"""Assemble the files for the Hugging Face Space into build/hf_space/ (ready to upload).

A Space is its own repo whose root must contain the Dockerfile and a README with the Space
front-matter, so the needed project files are copied into that layout.

    python scripts/prepare_hf_space.py

Then upload build/hf_space/ to your Space (see README -> Deployment), or let the GitHub Actions
workflow `.github/workflows/deploy-hf.yml` do it on every push.

The trained router and the demo data are OPTIONAL: without `artifacts/router.joblib` the Space
routes with the rule-based length heuristic until a trained router is pushed; without
`data/demo/` the dashboard and semantic cache start empty.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf_space"

REQUIRED = ["routewise", "config", "dashboard", "scripts", "requirements.txt", "requirements-ui.txt",
            "deploy/hf_space/start.sh"]
OPTIONAL = ["artifacts/router.joblib", "artifacts/router_meta.json", "data/demo"]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    missing = []
    for rel in REQUIRED + OPTIONAL:
        src = ROOT / rel
        if not src.exists():
            if rel in REQUIRED:
                raise SystemExit(f"missing required {rel}")
            missing.append(rel)
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    # The Dockerfile COPYs these folders, so they must exist even when empty.
    for folder in ("artifacts", "data/demo"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
        (OUT / folder / ".gitkeep").touch()
    shutil.copy2(ROOT / "deploy" / "hf_space" / "Dockerfile", OUT / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "hf_space" / "README.md", OUT / "README.md")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"prepared {OUT} ({size:.1f} MB)")
    for rel in missing:
        print(f"  note: {rel} not found - the Space works without it "
              + ("(router = rule-based length heuristic until a trained router is pushed)"
                 if "router" in rel else "(dashboard and semantic cache start empty)"))


if __name__ == "__main__":
    main()
