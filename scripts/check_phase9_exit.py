#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.phase9_gate import evaluate_phase9_repository_state  # noqa: E402


def main() -> int:
    status = evaluate_phase9_repository_state(ROOT)
    print(f"PHASE09_STAGE={status.stage.value}")
    print(f"TRAINING_CONFIG_HASH={status.training_config_hash}")
    if status.adapter_sha256 is not None:
        print(f"ADAPTER_SHA256={status.adapter_sha256}")
    if status.wandb_run_reference is not None:
        print(f"WANDB_RUN_REFERENCE={status.wandb_run_reference}")
    if not status.complete:
        print(f"PHASE09_EXIT=FAIL {status.blocker}")
        return 1
    print("PHASE09_EXIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
