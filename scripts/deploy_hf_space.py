"""Deploy (or update) RouteWise on a Hugging Face Space in one command.

    python scripts/deploy_hf_space.py --space <hf-username>/routewise --sync-secrets

1. builds build/hf_space/ (scripts/prepare_hf_space.py);
2. creates the Space if it does not exist (Docker SDK, free CPU hardware);
3. with --sync-secrets, stores GROQ_API_KEY / GEMINI_API_KEY / ROUTEWISE_API_KEYS from your
   environment or .env as the Space's encrypted secrets (never written into any file);
4. uploads the bundle; Hugging Face then rebuilds and restarts the Space.

Authentication: an HF access token with WRITE permission, from `HF_TOKEN` or a prior
`hf auth login`. Re-run any time - e.g. after training the router - to update the Space.
The GitHub Actions workflow runs this same script on every push to main.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "build" / "hf_space"
SECRET_NAMES = ("GROQ_API_KEY", "GEMINI_API_KEY", "ROUTEWISE_API_KEYS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("HF_SPACE"), help="<hf-username>/<space-name>")
    ap.add_argument("--sync-secrets", action="store_true",
                    help="copy GROQ/GEMINI/ROUTEWISE keys from env or .env into the Space's secrets")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()
    if not args.space or "/" not in args.space:
        sys.exit("pass --space <hf-username>/<space-name> (or set HF_SPACE)")

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    from huggingface_hub import HfApi

    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_hf_space.py")], check=True)
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True, private=args.private)

    if args.sync_secrets:
        for name in SECRET_NAMES:
            value = (os.environ.get(name) or "").strip()
            if value:
                api.add_space_secret(args.space, name, value)
                print(f"secret {name}: set")
            else:
                print(f"secret {name}: not found in env/.env - skipped")

    commit = os.environ.get("GITHUB_SHA", "")[:7] or "local"
    api.upload_folder(
        repo_id=args.space, repo_type="space", folder_path=str(BUNDLE),
        commit_message=f"Deploy RouteWise ({commit})",
        # Remove files deleted from the project since the last deploy (code folders only).
        delete_patterns=["routewise/**", "dashboard/**", "scripts/**", "config/**", "deploy/**"],
    )
    user, name = args.space.split("/", 1)
    print(f"uploaded. Space: https://huggingface.co/spaces/{args.space}")
    print(f"app (after the build finishes, ~10 min the first time): "
          f"https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
