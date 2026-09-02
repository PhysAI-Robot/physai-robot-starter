"""Download Hugging Face model snapshots into the ignored models/ directory.

    python scripts/download_models.py --model smolvlm
    python scripts/download_models.py --model smolvla
    python scripts/download_models.py --model turbovla
    python scripts/download_models.py --repo HuggingFaceTB/SmolVLM-500M-Instruct
    python scripts/download_models.py --repo org/model --name my_model
"""

from __future__ import annotations

import argparse
import sys

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    # The project package may already be installed when imported by tooling.
    pass

from physai.model_store import model_dir

MODEL_REPOS = {
    "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvla": "lerobot/smolvla_base",
    "turbovla": "H-EmbodVis/TurboVLA",
}


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_REPOS))
    parser.add_argument("--repo", help="Hugging Face repository ID")
    parser.add_argument("--name", help="local folder name under models/")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    if bool(args.model) == bool(args.repo):
        parser.error("pass exactly one of --model or --repo")
    repo = MODEL_REPOS[args.model] if args.model else args.repo
    name = args.name or (args.model if args.model else repo.rsplit("/", 1)[-1])
    destination = model_dir(name)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Model download needs huggingface-hub: pip install huggingface-hub",
              file=sys.stderr)
        return 1

    print(f"Downloading {repo} -> {destination}")
    snapshot_download(repo_id=repo, revision=args.revision,
                      local_dir=destination, local_dir_use_symlinks=False)
    print(f"Model ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
