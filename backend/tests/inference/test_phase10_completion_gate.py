from pathlib import Path

from app.inference.phase10_gate import (
    Phase10Stage,
    evaluate_phase10_repository_state,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase10_gate_accepts_complete_preserved_evidence() -> None:
    status = evaluate_phase10_repository_state(ROOT)
    assert status.stage is Phase10Stage.COMPLETE
    assert status.complete is True
    assert status.blocker is None
    assert status.locked_test_evidence_sha256 == (
        "9b253eb2e5a473812f46eccb5f90c0d8539aefb208a49266db521dc6bd137852"
    )
    assert status.comparison_sha256 == (
        "01409f6e1a04a970f451ed580b8b0f01e72b4fa09b51ae52071e562aa05e4105"
    )
    assert status.efficiency_condition_count == 12
    assert status.hub_repo_id == "sadmanht/evalforge-mistral-7b-incident-diagnosis-qlora"
    assert status.hub_revision == "0a11104c26ed6edc2fce612a341123b9ff4e9001"
