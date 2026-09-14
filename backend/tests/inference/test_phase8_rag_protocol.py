from __future__ import annotations

from pathlib import Path

import pytest

from app.core.experiment_config import PipelineType
from app.inference.protocol import load_baseline_protocol
from app.inference.rag_protocol import (
    RAGProtocol,
    RAGProtocolState,
    _assert_phase6_scientific_parity,
    build_rag_experiment_config,
    load_rag_protocol,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase8_protocol_preserves_frozen_phase6_scientific_identity() -> None:
    protocol, suite = load_rag_protocol(ROOT)
    baseline = load_baseline_protocol(ROOT)

    assert protocol.state is RAGProtocolState.FROZEN
    assert protocol.selected_variant_id == "rag-top-k1"
    assert protocol.locked_test_authorized is True
    assert protocol.base_model_id == baseline.base_model_id
    assert protocol.base_model_revision == baseline.base_model_revision
    assert protocol.runtime_config == baseline.runtime_config
    assert protocol.generation_config == baseline.generation_config
    assert protocol.ablation_suite_hash == suite.config_hash()
    assert len(protocol.scientific_config_hash()) == 64


def test_frozen_protocol_keeps_validation_registered_and_opens_only_selected_test_variant() -> None:
    protocol, suite = load_rag_protocol(ROOT)
    validation_variant = protocol.assert_variant_allowed(
        split="validation",
        variant_id="rag-top-k1",
        suite=suite,
    )
    assert validation_variant.top_k == 1

    selected = protocol.assert_variant_allowed(
        split="test",
        variant_id="rag-top-k1",
        suite=suite,
    )
    assert selected.variant_id == "rag-top-k1"

    with pytest.raises(ValueError, match="validation-selected frozen"):
        protocol.assert_variant_allowed(
            split="test",
            variant_id="rag-default-k5",
            suite=suite,
        )


def test_frozen_protocol_allows_only_the_validation_selected_variant_on_test() -> None:
    protocol, suite = load_rag_protocol(ROOT)
    frozen_payload = protocol.model_dump(mode="python")
    frozen_payload.update(
        {
            "state": "frozen",
            "selected_variant_id": "rag-default-k5",
            "locked_test_authorized": True,
        }
    )
    frozen = RAGProtocol.model_validate(frozen_payload)

    selected = frozen.assert_variant_allowed(
        split="test",
        variant_id="rag-default-k5",
        suite=suite,
    )
    assert selected.variant_id == "rag-default-k5"
    with pytest.raises(ValueError, match="validation-selected frozen"):
        frozen.assert_variant_allowed(
            split="test",
            variant_id="rag-top-k1",
            suite=suite,
        )


def test_selection_freeze_does_not_retroactively_change_scientific_hash() -> None:
    protocol, _suite = load_rag_protocol(ROOT)
    frozen_payload = protocol.model_dump(mode="python")
    frozen_payload.update(
        {
            "state": "frozen",
            "selected_variant_id": "rag-default-k5",
            "locked_test_authorized": True,
        }
    )
    frozen = RAGProtocol.model_validate(frozen_payload)

    assert frozen.scientific_payload() == protocol.scientific_payload()
    assert frozen.scientific_config_hash() == protocol.scientific_config_hash()


def test_rag_protocol_rejects_phase6_common_identity_drift() -> None:
    protocol, _suite = load_rag_protocol(ROOT)
    baseline = load_baseline_protocol(ROOT)
    drift_payload = protocol.model_dump(mode="python")
    drift_payload["base_model_revision"] = "different-revision"
    drift_payload["runtime_config"] = {
        **protocol.runtime_config.model_dump(mode="python"),
        "revision": "different-revision",
    }
    drift = RAGProtocol.model_validate(drift_payload)

    with pytest.raises(ValueError, match="preserve the frozen Phase 06 scientific identity"):
        _assert_phase6_scientific_parity(drift, baseline)


def test_rag_experiment_config_captures_complete_retrieval_identity() -> None:
    protocol, suite = load_rag_protocol(ROOT)
    variant = next(item for item in suite.variants if item.variant_id == "rag-default-k5")
    config = build_rag_experiment_config(
        root=ROOT,
        protocol=protocol,
        variant=variant,
        git_commit="1234567890abcdef",
        hardware_runtime_descriptor="phase8-unit-test-runtime",
        cost_rate_snapshot_version="phase8-unit-test-costs",
    )

    assert config.pipeline_type is PipelineType.RAG
    assert config.base_model_revision == protocol.base_model_revision
    assert config.knowledge_base_version == "evalforge-kb-v0.1.0"
    assert config.embedding_model_revision == "phase7-feature-hash-v1"
    assert config.retrieval_policy_version == "phase7-leakage-policy-v1"
    assert config.context_policy_version == "phase8-context-budget-v1"
    assert config.max_context_tokens == 384
    assert config.max_chunk_tokens == 128
    assert config.prompt_version == "rag-context-v1"
