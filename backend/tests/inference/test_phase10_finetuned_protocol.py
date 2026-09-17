from pathlib import Path

import pytest

from app.core.experiment_config import PipelineType
from app.inference.finetuned_protocol import (
    Phase10State,
    build_finetuned_experiment_config,
    load_phase10_protocol,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase10_protocol_preserves_frozen_scientific_identity() -> None:
    protocol = load_phase10_protocol(ROOT)
    assert protocol.state is Phase10State.VALIDATION
    assert protocol.locked_test_authorized is False
    assert protocol.base_model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert protocol.base_model_revision == "e8737b84b4470b28db3a0be719b362b1bd39a14d"
    assert protocol.candidate_adapter_sha256 == (
        "e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065"
    )
    assert protocol.data_efficiency.fractions == (0.10, 0.25, 0.50, 1.00)
    assert protocol.data_efficiency.primary_adapter_selection_use is False


def test_phase10_refuses_locked_test_before_primary_adapter_freeze() -> None:
    protocol = load_phase10_protocol(ROOT)
    assert protocol.assert_split_allowed("validation") == "validation"
    with pytest.raises(ValueError, match="locked test"):
        protocol.assert_split_allowed("test")


def test_finetuned_experiment_config_has_adapter_and_no_retrieval_identity() -> None:
    protocol = load_phase10_protocol(ROOT)
    config = build_finetuned_experiment_config(
        root=ROOT,
        protocol=protocol,
        git_commit="85e44d1bf17396d81917f5f56079854ffd41bdcd",
        hardware_runtime_descriptor="contract-test",
        cost_rate_snapshot_version="contract-test-v1",
    )
    assert config.pipeline_type is PipelineType.FINETUNED
    assert config.adapter_id == protocol.candidate_adapter_artifact_reference
    assert config.adapter_revision == protocol.candidate_adapter_sha256
    assert config.knowledge_base_version is None
    assert config.top_k is None
