#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.phase6_gate import evaluate_phase6_repository_state  # noqa: E402


def main() -> int:
    try:
        status = evaluate_phase6_repository_state(ROOT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"PHASE06_CONTRACT=FAIL {exc}") from exc

    print("PHASE06_CONTRACT=PASS")
    print(f"STAGE={status.stage.value}")
    print(f"SCIENTIFIC_CONFIG_HASH={status.scientific_config_hash}")
    print(f"BLOCKER={status.blocker or 'none'}")
    if status.validation_run_id:
        print(f"VALIDATION_RUN_ID={status.validation_run_id}")
    if status.test_run_id:
        print(f"TEST_RUN_ID={status.test_run_id}")
    if status.gpu_environment_fingerprint_sha256:
        print(f"GPU_ENVIRONMENT_FINGERPRINT={status.gpu_environment_fingerprint_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
