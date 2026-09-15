from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.smoke import run_smoke_training


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Phase 09 CPU contract trainer."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    result = run_smoke_training(
        args.output_dir,
        total_steps=args.steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
