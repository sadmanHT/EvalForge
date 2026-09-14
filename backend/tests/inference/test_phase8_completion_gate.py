from __future__ import annotations

from pathlib import Path

from app.inference.phase8_gate import Phase8Stage, evaluate_phase8_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_phase8_repository_is_complete_after_single_locked_test() -> None:
    status = evaluate_phase8_repository_state(ROOT)

    assert status.stage is Phase8Stage.COMPLETE
    assert status.complete is True
    assert status.selected_variant_id == "rag-top-k1"
    assert status.test_run_id == "phase8-rag-top-k1-test-kaggle-v1"
    assert status.test_evidence_sha256 == (
        "bb8e12f9f63ee939e630535bd38f815779f09f23779202be19cf7c0164e0d956"
    )
    assert status.blocker is None
    assert len(status.scientific_config_hash) == 64
    assert status.ablation_report_sha256 == (
        "68bdd0435bc60a431343bf6eed2cc0b0465e99e443bb387ee7da97f46fa62ce1"
    )
    assert status.comparison_sha256 == (
        "f763e9ffa94b7fecbe7e445e55d4249f6d81cc1fd729a8f7e0a96097a24ee13f"
    )
    assert status.gpu_environment_fingerprint_sha256 == (
        "45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff"
    )
