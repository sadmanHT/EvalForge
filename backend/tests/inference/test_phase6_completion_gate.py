from __future__ import annotations

from pathlib import Path

from app.inference.phase6_gate import Phase6Stage, evaluate_phase6_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_repository_is_frozen_waiting_for_single_locked_test() -> None:
    status = evaluate_phase6_repository_state(ROOT)

    assert status.stage is Phase6Stage.FROZEN_WAITING_TEST
    assert status.complete is False
    assert status.validation_run_id == "phase6-zero-shot-validation-kaggle-v2"
    assert status.test_run_id is None
    assert status.blocker is not None
    assert "single authorized real locked-test baseline" in status.blocker
