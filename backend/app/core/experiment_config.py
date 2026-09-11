from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PipelineType(StrEnum):
    ZERO_SHOT = "ZERO_SHOT"
    RAG = "RAG"
    FINETUNED = "FINETUNED"
    COMBINED = "COMBINED"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    pipeline_type: PipelineType
    dataset_version: str = Field(min_length=1)
    test_split_manifest_checksum: str = Field(min_length=64, max_length=64)
    label_taxonomy_version: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    adapter_id: str | None = None
    adapter_revision: str | None = None
    knowledge_base_version: str | None = None
    embedding_model_revision: str | None = None
    chunker_version: str | None = None
    chunk_size: int | None = None
    overlap: int | None = None
    top_k: int | None = None
    reranker_id: str | None = None
    reranker_revision: str | None = None
    prompt_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    generation_config: dict[str, Any]
    temperature: float = Field(ge=0)
    confidence_method: str = Field(min_length=1)
    calibration_version: str | None = None
    seed: int
    evaluator_version: str = Field(min_length=1)
    ragas_or_judge_versions: dict[str, Any] | None = None
    git_commit: str = Field(min_length=7)
    dependency_lock_checksum: str = Field(min_length=64, max_length=64)
    hardware_runtime_descriptor: str = Field(min_length=1)
    cost_rate_snapshot_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pipeline_contract(self) -> ExperimentConfig:
        uses_adapter = self.pipeline_type in {PipelineType.FINETUNED, PipelineType.COMBINED}
        if uses_adapter != bool(self.adapter_id and self.adapter_revision):
            raise ValueError("adapter identity must be present exactly for fine-tuned pipelines")
        uses_rag = self.pipeline_type in {PipelineType.RAG, PipelineType.COMBINED}
        retrieval = (
            self.knowledge_base_version,
            self.embedding_model_revision,
            self.chunker_version,
            self.chunk_size,
            self.overlap,
            self.top_k,
        )
        if uses_rag:
            if any(value is None for value in retrieval):
                raise ValueError("RAG pipelines require complete retrieval identity")
            assert self.chunk_size is not None and self.overlap is not None and self.top_k is not None
            if self.chunk_size <= 0 or self.overlap < 0 or self.overlap >= self.chunk_size or self.top_k <= 0:
                raise ValueError("invalid retrieval configuration")
        elif any(value is not None for value in retrieval) or self.reranker_id or self.reranker_revision:
            raise ValueError("non-RAG pipelines may not carry retrieval configuration")
        generation_temperature = self.generation_config.get("temperature")
        if generation_temperature is not None and float(generation_temperature) != self.temperature:
            raise ValueError("temperature disagrees with generation_config")
        _reject_non_finite(self.model_dump(mode="json"))
        return self


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("experiment config contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)


def canonical_config_json(config: ExperimentConfig) -> str:
    payload = config.model_dump(mode="json", exclude_none=False)
    _reject_non_finite(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def experiment_config_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode()).hexdigest()
