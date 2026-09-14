from __future__ import annotations

from pathlib import Path

from app.inference.phase8_gate import Phase8Stage, evaluate_phase8_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_phase8_repository_is_pre_validation_until_real_ablation_evidence_exists() -> None:
    status = evaluate_phase8_repository_state(ROOT)

    assert status.stage is Phase8Stage.PRE_VALIDATION
    assert status.complete is False
    assert status.selected_variant_id is None
    assert status.test_run_id is None
    assert status.blocker == "real RAG validation ablations and selection evidence are pending"
    assert len(status.scientific_config_hash) == 64
