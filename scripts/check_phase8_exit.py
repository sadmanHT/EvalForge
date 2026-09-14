#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.phase8_gate import evaluate_phase8_repository_state  # noqa: E402


def main() -> int:
    status = evaluate_phase8_repository_state(ROOT)
    print(f"PHASE08_STAGE={status.stage.value}")
    print(f"SCIENTIFIC_CONFIG_HASH={status.scientific_config_hash}")
    if status.selected_variant_id is not None:
        print(f"SELECTED_VARIANT_ID={status.selected_variant_id}")
    if status.ablation_report_sha256 is not None:
        print(f"ABLATION_REPORT_SHA256={status.ablation_report_sha256}")
    if status.test_run_id is not None:
        print(f"TEST_RUN_ID={status.test_run_id}")
    if status.test_evidence_sha256 is not None:
        print(f"TEST_EVIDENCE_SHA256={status.test_evidence_sha256}")
    if status.comparison_sha256 is not None:
        print(f"BASELINE_RAG_COMPARISON_SHA256={status.comparison_sha256}")
    if status.gpu_environment_fingerprint_sha256 is not None:
        print(
            "GPU_ENVIRONMENT_FINGERPRINT="
            f"{status.gpu_environment_fingerprint_sha256}"
        )
    if not status.complete:
        print(f"PHASE08_EXIT=FAIL {status.blocker}")
        return 1
    print("PHASE08_EXIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
