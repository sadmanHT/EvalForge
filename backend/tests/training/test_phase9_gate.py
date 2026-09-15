from __future__ import annotations

from pathlib import Path

import pytest

from app.training.config import load_training_config
from app.training.phase9_gate import Phase9Stage, evaluate_phase9_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_repository_state_stops_before_external_training_without_fabrication() -> None:
    status = evaluate_phase9_repository_state(ROOT)
    assert status.stage is Phase9Stage.PRE_EXTERNAL_TRAINING
    assert status.complete is False
    assert status.adapter_sha256 is None
    assert status.wandb_run_reference is None
    assert status.blocker is not None
    assert "GPU/W&B" in status.blocker
    assert status.training_config_hash == load_training_config(ROOT).config_hash()


def test_hard_exit_script_is_expected_to_fail_before_real_training() -> None:
    status = evaluate_phase9_repository_state(ROOT)
    if (ROOT / "evidence/phase-09/training-run.json").exists():
        pytest.skip("real Phase 09 evidence has been committed")
    assert status.complete is False
