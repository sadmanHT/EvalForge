from __future__ import annotations

from pathlib import Path

from app.inference.phase8_gate import Phase8Stage, evaluate_phase8_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_phase8_repository_is_frozen_and_waiting_for_single_locked_test() -> None:
    status = evaluate_phase8_repository_state(ROOT)

    assert status.stage is Phase8Stage.FROZEN_WAITING_TEST
    assert status.complete is False
    assert status.selected_variant_id == "rag-top-k1"
    assert status.test_run_id is None
    assert status.blocker == "single authorized real RAG locked test is pending"
    assert len(status.scientific_config_hash) == 64
    assert status.ablation_report_sha256 == (
        "68bdd0435bc60a431343bf6eed2cc0b0465e99e443bb387ee7da97f46fa62ce1"
    )
    assert status.comparison_sha256 == (
        "5b26b8797557324862498257dc48c958977d951f0deef1b7bfedd83bea7293d7"
    )
    assert status.gpu_environment_fingerprint_sha256 == (
        "45e5ab6ceb7600352c94a5383f6721a21d8bfae9795e6be17a617c64379b5bff"
    )
