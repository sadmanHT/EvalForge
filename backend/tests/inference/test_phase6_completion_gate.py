from __future__ import annotations

from pathlib import Path

from app.inference.phase6_gate import Phase6Stage, evaluate_phase6_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_repository_is_complete_after_single_locked_test() -> None:
    status = evaluate_phase6_repository_state(ROOT)

    assert status.stage is Phase6Stage.COMPLETE
    assert status.complete is True
    assert status.blocker is None
    assert status.validation_run_id == "phase6-zero-shot-validation-kaggle-v2"
    assert status.test_run_id == "phase6-zero-shot-test-kaggle-v1"
    assert (
        status.validation_evidence_sha256
        == "184ed6f973bfc69ff5bc7becac5f60b1650ce7368cb36cfec89bcd6e1e344e1f"
    )
    assert (
        status.test_evidence_sha256
        == "76ee0677d3b3eebc51169fbbe21f323e6c6b23cb3dffabcffbc2f8a0685dd3c8"
    )
    assert (
        status.gpu_environment_fingerprint_sha256
        == "45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff"
    )
    assert status.test_metrics is not None
    assert status.test_metrics["primary.exact_accuracy"] == 1.0
    assert status.test_metrics["quality.parse_failure_rate"] == 0.0
    assert (
        status.tracking_run_reference
        == "https://wandb.ai/sadman-hasan-t-islamic-university-of-technology/"
        "evalforge-phase6/runs/1xlznr26"
    )
    assert status.tracking_artifact_reference == "phase6-zero-shot-test-kaggle-v1-result"
