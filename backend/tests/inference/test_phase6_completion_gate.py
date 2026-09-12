from __future__ import annotations

from pathlib import Path

from app.inference.phase6_gate import Phase6Stage, evaluate_phase6_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_repository_remains_truthfully_pre_validation_until_real_evidence_exists() -> None:
    status = evaluate_phase6_repository_state(ROOT)

    assert status.stage is Phase6Stage.PRE_VALIDATION
    assert status.complete is False
    assert status.validation_run_id is None
    assert status.test_run_id is None
    assert status.blocker is not None
    assert "real validation run" in status.blocker
