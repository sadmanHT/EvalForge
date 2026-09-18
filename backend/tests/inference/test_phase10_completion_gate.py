from pathlib import Path

from app.inference.phase10_gate import (
    Phase10Stage,
    evaluate_phase10_repository_state,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase10_gate_accepts_locked_test_and_waits_for_external_completion() -> None:
    status = evaluate_phase10_repository_state(ROOT)
    assert status.stage is Phase10Stage.EXTERNAL_EVIDENCE_PENDING
    assert status.complete is False
    assert status.locked_test_evidence_sha256 == (
        "9b253eb2e5a473812f46eccb5f90c0d8539aefb208a49266db521dc6bd137852"
    )
    assert status.comparison_sha256 == (
        "01409f6e1a04a970f451ed580b8b0f01e72b4fa09b51ae52071e562aa05e4105"
    )
    assert status.blocker is not None
    assert "data-efficiency aggregate" in status.blocker
    assert "Hugging Face release" in status.blocker
    assert "Hugging Face clean smoke" in status.blocker
