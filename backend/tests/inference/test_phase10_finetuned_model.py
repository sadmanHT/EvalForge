from pathlib import Path

import pytest

from app.inference.finetuned_model import (
    AdapterRevisionMismatch,
    load_phase9_candidate_identity,
    verify_adapter_locator,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase9_candidate_identity_is_content_addressed() -> None:
    identity = load_phase9_candidate_identity(ROOT)
    assert identity.adapter_revision == identity.adapter_sha256
    assert identity.adapter_sha256 == (
        "e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065"
    )
    assert identity.source_training_run_id == "phase9-qlora-validation-v1"


def test_adapter_revision_mismatch_fails_before_model_load() -> None:
    identity = load_phase9_candidate_identity(ROOT)
    with pytest.raises(AdapterRevisionMismatch, match="revision mismatch"):
        verify_adapter_locator(
            "wandb-or-hf-remote-locator",
            identity=identity,
            adapter_revision="wrong-revision",
        )
