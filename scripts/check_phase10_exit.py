#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.phase10_gate import evaluate_phase10_repository_state  # noqa: E402


def main() -> int:
    status = evaluate_phase10_repository_state(ROOT)
    print(f"PHASE10_STAGE={status.stage.value}")
    print(f"PHASE10_SCIENTIFIC_CONFIG_HASH={status.scientific_config_hash}")
    print(f"CANDIDATE_ADAPTER_SHA256={status.adapter_sha256}")
    if status.locked_test_evidence_sha256 is not None:
        print(f"LOCKED_TEST_EVIDENCE_SHA256={status.locked_test_evidence_sha256}")
    if status.comparison_sha256 is not None:
        print(f"BASELINE_FINETUNED_COMPARISON_SHA256={status.comparison_sha256}")
    if status.efficiency_condition_count:
        print(f"DATA_EFFICIENCY_CONDITION_COUNT={status.efficiency_condition_count}")
    if status.hub_repo_id is not None:
        print(f"HF_REPO_ID={status.hub_repo_id}")
    if status.hub_revision is not None:
        print(f"HF_REVISION={status.hub_revision}")
    if not status.complete:
        print(f"PHASE10_EXIT=FAIL {status.blocker}")
        return 1
    print("PHASE10_EXIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
