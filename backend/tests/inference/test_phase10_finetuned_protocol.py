from pathlib import Path

from app.core.experiment_config import PipelineType
from app.inference.finetuned_protocol import (
    Phase10State,
    build_finetuned_experiment_config,
    load_phase10_protocol,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase10_protocol_preserves_frozen_scientific_identity() -> None:
    protocol = load_phase10_protocol(ROOT)
    assert protocol.state is Phase10State.FROZEN
    assert protocol.locked_test_authorized is True
    assert protocol.base_model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert protocol.base_model_revision == "e8737b84b4470b28db3a0be719b362b1bd39a14d"
    assert protocol.candidate_adapter_sha256 == (
        "e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065"
    )
    assert protocol.scientific_config_hash() == (
        "6064aaed457812a8102411eea47f02f78d812372097afe068ec53a540f9e1f7d"
    )
    assert protocol.data_efficiency.fractions == (0.10, 0.25, 0.50, 1.00)
    assert protocol.data_efficiency.primary_adapter_selection_use is False


def test_phase10_allows_locked_test_only_after_primary_adapter_freeze() -> None:
    protocol = load_phase10_protocol(ROOT)
    assert protocol.assert_split_allowed("validation") == "validation"
    assert protocol.assert_split_allowed("test") == "test"


def test_finetuned_experiment_config_has_adapter_and_no_retrieval_identity() -> None:
    protocol = load_phase10_protocol(ROOT)
    config = build_finetuned_experiment_config(
        root=ROOT,
        protocol=protocol,
        git_commit="4861ea86fd31c046d2bd9b76177659a0739b79f6",
        hardware_runtime_descriptor="contract-test",
        cost_rate_snapshot_version="contract-test-v1",
    )
    assert config.pipeline_type is PipelineType.FINETUNED
    assert config.adapter_id == protocol.candidate_adapter_artifact_reference
    assert config.adapter_revision == protocol.candidate_adapter_sha256
    assert config.knowledge_base_version is None
    assert config.top_k is None
