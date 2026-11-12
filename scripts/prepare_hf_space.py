"""Assemble the files for the Hugging Face Space into build/hf_space/ (ready to git push).

A Space is its own git repo whose root must contain the Dockerfile and a README with the
Space front-matter, so we copy the needed project files into that layout.

    python scripts/prepare_hf_space.py
    cd build/hf_space && git init && git remote add origin https://huggingface.co/spaces/<user>/routewise
    git add . && git commit -m "Deploy RouteWise" && git push -u origin main
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf_space"

INCLUDE = ["routewise", "config", "dashboard", "scripts", "artifacts/router.joblib", "artifacts/router_meta.json",
           "data/demo", "requirements.txt", "requirements-ui.txt", "deploy/hf_space/start.sh"]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing {rel} - train the router (make train-router) and build the demo data "
                             "(python scripts/build_demo_data.py) first")
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    shutil.copy2(ROOT / "deploy" / "hf_space" / "Dockerfile", OUT / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "hf_space" / "README.md", OUT / "README.md")
    # Hugging Face stores binary files with git-lfs/xet; mark them explicitly.
    (OUT / ".gitattributes").write_text("*.db filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.joblib filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"prepared {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
