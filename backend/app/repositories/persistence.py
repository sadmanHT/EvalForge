from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.experiment_config import ExperimentConfig, experiment_config_hash
from app.models import (
    AdapterVersion,
    DatasetVersion,
    Experiment,
    Job,
    KnowledgeBaseVersion,
    ModelVersion,
)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class PersistenceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.session.begin():
            yield

    def ensure_model_version(
        self,
        model_id: str,
        revision: str,
        *,
        license_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        existing = self.session.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.revision == revision,
            )
        )
        if existing is not None:
            return existing
        row = ModelVersion(
            model_version_id=_stable_id("model", model_id, revision),
            model_id=model_id,
            revision=revision,
            license=license_name,
            metadata_json=metadata or {},
        )
        self.session.add(row)
        self.session.flush()
        return row

    def create_experiment(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        *,
        status: str = "planned",
    ) -> Experiment:
        config_hash = experiment_config_hash(config)
        existing = self.session.scalar(
            select(Experiment).where(Experiment.config_hash == config_hash)
        )
        if existing is not None:
            return existing
        dataset = self.session.get(DatasetVersion, config.dataset_version)
        if dataset is None:
            raise ValueError(f"unknown dataset version: {config.dataset_version}")
        if dataset.manifest_checksum != config.test_split_manifest_checksum:
            raise ValueError("experiment test manifest checksum does not match dataset version")
        if dataset.label_taxonomy_version != config.label_taxonomy_version:
            raise ValueError("experiment label taxonomy version does not match dataset version")
        model = self.ensure_model_version(config.base_model_id, config.base_model_revision)
        adapter_version_id: str | None = None
        if config.adapter_id and config.adapter_revision:
            adapter = self.session.scalar(
                select(AdapterVersion).where(
                    AdapterVersion.adapter_id == config.adapter_id,
                    AdapterVersion.revision == config.adapter_revision,
                )
            )
            if adapter is None:
                raise ValueError("adapter version is not registered")
            if adapter.model_version_id != model.model_version_id:
                raise ValueError("adapter belongs to a different base model version")
            adapter_version_id = adapter.adapter_version_id
        if (
            config.knowledge_base_version is not None
            and self.session.get(KnowledgeBaseVersion, config.knowledge_base_version) is None
        ):
            raise ValueError("knowledge base version is not registered")
        row = Experiment(
            experiment_id=experiment_id,
            study_id=config.study_id,
            pipeline_type=config.pipeline_type.value,
            dataset_version=config.dataset_version,
            test_split_manifest_checksum=config.test_split_manifest_checksum,
            label_taxonomy_version=config.label_taxonomy_version,
            model_version_id=model.model_version_id,
            adapter_version_id=adapter_version_id,
            kb_version=config.knowledge_base_version,
            embedding_model_revision=config.embedding_model_revision,
            chunker_version=config.chunker_version,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            top_k=config.top_k,
            reranker_id=config.reranker_id,
            reranker_revision=config.reranker_revision,
            prompt_version=config.prompt_version,
            output_schema_version=config.output_schema_version,
            generation_config=config.generation_config,
            temperature=config.temperature,
            confidence_method=config.confidence_method,
            calibration_version=config.calibration_version,
            seed=config.seed,
            evaluator_version=config.evaluator_version,
            ragas_or_judge_versions=config.ragas_or_judge_versions,
            git_commit=config.git_commit,
            dependency_lock_checksum=config.dependency_lock_checksum,
            hardware_runtime_descriptor=config.hardware_runtime_descriptor,
            cost_rate_snapshot_version=config.cost_rate_snapshot_version,
            status=status,
            config_json=config.model_dump(mode="json", exclude_none=False),
            config_hash=config_hash,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def create_job(
        self,
        job_id: str,
        kind: str,
        idempotency_key: str,
        payload: dict[str, Any],
        *,
        experiment_id: str | None = None,
    ) -> Job:
        payload_hash = _payload_hash(payload)
        existing = self.session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
        if existing is not None:
            if (
                existing.kind != kind
                or existing.payload_hash != payload_hash
                or existing.experiment_id != experiment_id
            ):
                raise ValueError("idempotency key reused with different job semantics")
            return existing
        row = Job(
            job_id=job_id,
            experiment_id=experiment_id,
            kind=kind,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            payload_json=payload,
            status="queued",
            attempt=1,
        )
        self.session.add(row)
        self.session.flush()
        return row
