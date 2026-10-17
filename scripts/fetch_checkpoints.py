"""Download policy checkpoints from Hugging Face into checkpoints/ (ignored by git, like assets/).

    python scripts/fetch_checkpoints.py hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0
    python scripts/fetch_checkpoints.py hf:OWNER/REPO/FOLDER@REVISION hf:OWNER/REPO/OTHER_FOLDER
    python scripts/fetch_checkpoints.py hf:OWNER/REPO/FOLDER --out /data/checkpoints

Each argument is the same reference `--checkpoint` accepts; the folder is downloaded to
`<out>/<owner>/<repo>/<folder>` (default `checkpoints/`) and its path printed, so a later run can use the
path with no network. `--checkpoint hf:...` downloads on demand into the same place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser

from physai.policy.checkpoints import HF_PREFIX, resolve_checkpoint


def main() -> int:
    ap = new_parser(__doc__)
    ap.add_argument(
        "references", nargs="+", help="hf:<owner>/<repo>/<subfolder>[@<revision>]"
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="root folder (default: checkpoints/ at the repository root)",
    )
    args = ap.parse_args()
    for reference in args.references:
        if not reference.startswith(HF_PREFIX):
            ap.error(f"{reference!r} is not a hf: reference")
        print(f"fetching {reference}")
        print(f"  -> {resolve_checkpoint(reference, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
