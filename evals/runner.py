#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.inference.orchestration import Phase6BaselineJobHandler

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 06 zero-shot baseline through the canonical worker handler."
    )
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--hardware-runtime-descriptor", required=True)
    parser.add_argument("--cost-rate-snapshot-version", required=True)
    parser.add_argument("--gpu-hour-usd", required=True, type=float)
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-id")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--upfront-cost-usd", type=float, default=0.0)
    args = parser.parse_args()

    payload = {
        "split": args.split,
        "git_commit": args.git_commit,
        "hardware_runtime_descriptor": args.hardware_runtime_descriptor,
        "cost_rate_snapshot_version": args.cost_rate_snapshot_version,
        "gpu_hour_usd": args.gpu_hour_usd,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "max_attempts": args.max_attempts,
        "upfront_cost_usd": args.upfront_cost_usd,
    }
    result = Phase6BaselineJobHandler(root=ROOT)(payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
