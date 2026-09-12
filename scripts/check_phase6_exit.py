#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.phase6_gate import Phase6Stage, evaluate_phase6_repository_state  # noqa: E402


def _fail(message: str) -> None:
    raise SystemExit(f"PHASE06_EXIT_GATE=FAIL {message}")


def main() -> int:
    try:
        status = evaluate_phase6_repository_state(ROOT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))

    if status.stage is not Phase6Stage.COMPLETE:
        _fail(f"stage={status.stage.value} blocker={status.blocker}")
    if status.test_metrics is None:
        _fail("completed repository state is missing locked-test metrics")

    print("PHASE06_EXIT_GATE=PASS")
    print(f"SCIENTIFIC_CONFIG_HASH={status.scientific_config_hash}")
    print(f"VALIDATION_RUN_ID={status.validation_run_id}")
    print(f"TEST_RUN_ID={status.test_run_id}")
    print(f"VALIDATION_EVIDENCE_SHA256={status.validation_evidence_sha256}")
    print(f"TEST_EVIDENCE_SHA256={status.test_evidence_sha256}")
    print(f"GPU_ENVIRONMENT_FINGERPRINT={status.gpu_environment_fingerprint_sha256}")
    print(f"WANDB_RUN_REFERENCE={status.tracking_run_reference}")
    print(f"WANDB_ARTIFACT_REFERENCE={status.tracking_artifact_reference}")
    print(f"LOCKED_TEST_METRICS={json.dumps(status.test_metrics, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
