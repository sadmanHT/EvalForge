from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.experiment_config import (
    ExperimentConfig,
    PipelineType,
    canonical_config_json,
    experiment_config_hash,
)


def _config(**overrides: Any) -> ExperimentConfig:
    payload: dict[str, Any] = {
        "study_id": "evalforge-incident-diagnosis-primary-v1",
        "pipeline_type": PipelineType.ZERO_SHOT,
        "dataset_version": "evalforge-incident-diagnosis-v0.1.0",
        "test_split_manifest_checksum": "a" * 64,
        "label_taxonomy_version": "1.0.0",
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_model_revision": "e8737b84b4470b28db3a0be719b362b1bd39a14d",
        "prompt_version": "prompt-v1",
        "output_schema_version": "root-cause-prediction-v1",
        "generation_config": {
            "temperature": 0.0,
            "do_sample": False,
            "nested": {"b": 2, "a": 1},
        },
        "temperature": 0.0,
        "confidence_method": "normalized_label_sequence_log_likelihood",
        "seed": 20260908,
        "evaluator_version": "evaluator-v1",
        "git_commit": "1234567890abcdef",
        "dependency_lock_checksum": "b" * 64,
        "hardware_runtime_descriptor": "unit-test-runtime",
        "cost_rate_snapshot_version": "cost-rates-v1",
    }
    payload.update(overrides)
    return ExperimentConfig.model_validate(payload)


def test_zero_shot_config_is_valid_and_canonical() -> None:
    config = _config()
    canonical = canonical_config_json(config)
    assert '"pipeline_type":"ZERO_SHOT"' in canonical
    assert len(experiment_config_hash(config)) == 64


def test_hash_is_invariant_to_nested_dictionary_order() -> None:
    first = _config(
        generation_config={
            "temperature": 0.0,
            "do_sample": False,
            "nested": {"b": 2, "a": 1},
        }
    )
    second = _config(
        generation_config={
            "nested": {"a": 1, "b": 2},
            "do_sample": False,
            "temperature": 0.0,
        }
    )
    assert canonical_config_json(first) == canonical_config_json(second)
    assert experiment_config_hash(first) == experiment_config_hash(second)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7),
        ("prompt_version", "prompt-v2"),
        ("temperature", 0.2),
    ],
)
def test_meaningful_change_changes_hash(field: str, value: object) -> None:
    baseline = _config()
    overrides: dict[str, object] = {field: value}
    if field == "temperature":
        overrides["generation_config"] = {"temperature": value, "do_sample": False}
    changed = _config(**overrides)
    assert experiment_config_hash(baseline) != experiment_config_hash(changed)


def test_rag_requires_complete_retrieval_identity() -> None:
    with pytest.raises(ValidationError, match="complete retrieval identity"):
        _config(pipeline_type=PipelineType.RAG, knowledge_base_version="kb-v1")

    config = _config(
        pipeline_type=PipelineType.RAG,
        knowledge_base_version="kb-v1",
        embedding_model_revision="embed-rev",
        chunker_version="chunker-v1",
        chunk_size=512,
        overlap=64,
        top_k=5,
    )
    assert config.top_k == 5


def test_non_rag_pipeline_rejects_retrieval_identity() -> None:
    with pytest.raises(ValidationError, match="non-RAG pipelines"):
        _config(knowledge_base_version="kb-v1")


def test_finetuned_requires_adapter_and_zero_shot_forbids_it() -> None:
    with pytest.raises(ValidationError, match="adapter identity"):
        _config(pipeline_type=PipelineType.FINETUNED)

    finetuned = _config(
        pipeline_type=PipelineType.FINETUNED,
        adapter_id="adapter-a",
        adapter_revision="rev-a",
    )
    assert finetuned.adapter_id == "adapter-a"

    with pytest.raises(ValidationError, match="adapter identity"):
        _config(adapter_id="adapter-a", adapter_revision="rev-a")


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "top_k"),
    [(0, 0, 5), (512, 512, 5), (512, -1, 5), (512, 64, 0)],
)
def test_invalid_rag_geometry_is_rejected(
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> None:
    with pytest.raises(ValidationError, match="invalid retrieval configuration"):
        _config(
            pipeline_type=PipelineType.RAG,
            knowledge_base_version="kb-v1",
            embedding_model_revision="embed-rev",
            chunker_version="chunker-v1",
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )


def test_generation_temperature_must_match_explicit_temperature() -> None:
    with pytest.raises(ValidationError, match="temperature disagrees"):
        _config(generation_config={"temperature": 0.5}, temperature=0.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_nested_generation_values_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        _config(generation_config={"temperature": 0.0, "logit_bias": value})
