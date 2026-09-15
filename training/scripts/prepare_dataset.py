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

from app.training.config import load_training_config
from app.training.formatter import prepare_training_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe Phase 09 train/validation data."
    )
    parser.add_argument(
        "--dataset-version",
        default="evalforge-incident-diagnosis-v0.1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "training/prepared/phase9",
    )
    args = parser.parse_args()
    bundle = load_training_config(ROOT)
    manifest = prepare_training_dataset(
        root=ROOT,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        include_reasoning=bundle.training.include_reasoning_target,
    )
    print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
