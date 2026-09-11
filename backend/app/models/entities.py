from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.types import Vector


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    version: Mapped[str] = mapped_column(String(160), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    label_taxonomy_version: Mapped[str] = mapped_column(String(80), nullable=False)
    manifest_checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    content_checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    split_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    family_count: Mapped[int] = mapped_column(Integer, nullable=False)
    research_ready: Mapped[bool] = mapped_column(Boolean, nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        CheckConstraint("record_count >= 0", name="ck_dataset_record_count"),
        CheckConstraint("family_count >= 0", name="ck_dataset_family_count"),
    )


class IncidentFamily(Base):
    __tablename__ = "incident_families"
    dataset_version: Mapped[str] = mapped_column(String(160), nullable=False)
    family_id: Mapped[str] = mapped_column(String(120), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    source_family_key: Mapped[str] = mapped_column(String(320), nullable=False)
    root_cause_codes: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    family_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        PrimaryKeyConstraint("dataset_version", "family_id"),
        ForeignKeyConstraint(["dataset_version"], ["dataset_versions.version"], ondelete="RESTRICT"),
        UniqueConstraint("dataset_version", "source_family_key"),
        UniqueConstraint("dataset_version", "family_id", "split"),
        CheckConstraint("split IN ('train','validation','test')", name="ck_family_split"),
    )


class Incident(Base):
    __tablename__ = "incidents"
    dataset_version: Mapped[str] = mapped_column(String(160), nullable=False)
    incident_id: Mapped[str] = mapped_column(String(120), nullable=False)
    family_id: Mapped[str] = mapped_column(String(120), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    root_cause_code: Mapped[str] = mapped_column(String(120), nullable=False)
    root_cause_category: Mapped[str] = mapped_column(String(120), nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(320), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    incident_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        PrimaryKeyConstraint("dataset_version", "incident_id"),
        ForeignKeyConstraint(
            ["dataset_version", "family_id", "split"],
            ["incident_families.dataset_version", "incident_families.family_id", "incident_families.split"],
            ondelete="RESTRICT",
        ),
        UniqueConstraint("dataset_version", "source_record_id"),
        CheckConstraint("split IN ('train','validation','test')", name="ck_incident_split"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    model_version_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(320), nullable=False)
    revision: Mapped[str] = mapped_column(String(160), nullable=False)
    license: Mapped[str | None] = mapped_column(String(120))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (UniqueConstraint("model_id", "revision"),)


class AdapterVersion(Base):
    __tablename__ = "adapter_versions"
    adapter_version_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.model_version_id"), nullable=False)
    adapter_id: Mapped[str] = mapped_column(String(320), nullable=False)
    revision: Mapped[str] = mapped_column(String(160), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("adapter_id", "revision"),
        UniqueConstraint("adapter_version_id", "model_version_id"),
    )


class KnowledgeBaseVersion(Base):
    __tablename__ = "knowledge_base_versions"
    kb_version: Mapped[str] = mapped_column(String(160), primary_key=True)
    embedding_model_id: Mapped[str] = mapped_column(String(320), nullable=False)
    embedding_model_revision: Mapped[str] = mapped_column(String(160), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(160), nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    reranker_id: Mapped[str | None] = mapped_column(String(320))
    reranker_revision: Mapped[str | None] = mapped_column(String(160))
    manifest_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        CheckConstraint("chunk_size > 0", name="ck_kb_chunk_size"),
        CheckConstraint("overlap >= 0 AND overlap < chunk_size", name="ck_kb_overlap"),
    )


class KBDocument(Base):
    __tablename__ = "kb_documents"
    kb_version: Mapped[str] = mapped_column(String(160), nullable=False)
    document_id: Mapped[str] = mapped_column(String(160), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    document_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        PrimaryKeyConstraint("kb_version", "document_id"),
        ForeignKeyConstraint(["kb_version"], ["knowledge_base_versions.kb_version"]),
    )


class KBChunk(Base):
    __tablename__ = "kb_chunks"
    kb_version: Mapped[str] = mapped_column(String(160), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(160), nullable=False)
    document_id: Mapped[str] = mapped_column(String(160), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536))
    source_family_id: Mapped[str | None] = mapped_column(String(120))
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        PrimaryKeyConstraint("kb_version", "chunk_id"),
        ForeignKeyConstraint(["kb_version", "document_id"], ["kb_documents.kb_version", "kb_documents.document_id"]),
        UniqueConstraint("kb_version", "document_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0", name="ck_chunk_index"),
    )


class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    study_id: Mapped[str] = mapped_column(String(160), nullable=False)
    pipeline_type: Mapped[str] = mapped_column(String(24), nullable=False)
    dataset_version: Mapped[str] = mapped_column(ForeignKey("dataset_versions.version"), nullable=False)
    test_split_manifest_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    label_taxonomy_version: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.model_version_id"), nullable=False)
    adapter_version_id: Mapped[str | None] = mapped_column(String(160))
    kb_version: Mapped[str | None] = mapped_column(ForeignKey("knowledge_base_versions.kb_version"))
    embedding_model_revision: Mapped[str | None] = mapped_column(String(160))
    chunker_version: Mapped[str | None] = mapped_column(String(160))
    chunk_size: Mapped[int | None] = mapped_column(Integer)
    overlap: Mapped[int | None] = mapped_column(Integer)
    top_k: Mapped[int | None] = mapped_column(Integer)
    reranker_id: Mapped[str | None] = mapped_column(String(320))
    reranker_revision: Mapped[str | None] = mapped_column(String(160))
    prompt_version: Mapped[str] = mapped_column(String(160), nullable=False)
    output_schema_version: Mapped[str] = mapped_column(String(160), nullable=False)
    generation_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_method: Mapped[str] = mapped_column(String(160), nullable=False)
    calibration_version: Mapped[str | None] = mapped_column(String(160))
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(160), nullable=False)
    ragas_or_judge_versions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    dependency_lock_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    hardware_runtime_descriptor: Mapped[str] = mapped_column(Text, nullable=False)
    cost_rate_snapshot_version: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        UniqueConstraint("experiment_id", "dataset_version"),
        UniqueConstraint("experiment_id", "kb_version"),
        ForeignKeyConstraint(
            ["adapter_version_id", "model_version_id"],
            ["adapter_versions.adapter_version_id", "adapter_versions.model_version_id"],
        ),
        CheckConstraint("pipeline_type IN ('ZERO_SHOT','RAG','FINETUNED','COMBINED')", name="ck_experiment_pipeline"),
        CheckConstraint("status IN ('planned','queued','running','completed','failed','cancelled')", name="ck_experiment_status"),
        CheckConstraint("temperature >= 0", name="ck_experiment_temperature"),
        CheckConstraint("(pipeline_type IN ('FINETUNED','COMBINED') AND adapter_version_id IS NOT NULL) OR (pipeline_type IN ('ZERO_SHOT','RAG') AND adapter_version_id IS NULL)", name="ck_experiment_adapter_contract"),
        CheckConstraint("(pipeline_type IN ('RAG','COMBINED') AND kb_version IS NOT NULL AND embedding_model_revision IS NOT NULL AND chunker_version IS NOT NULL AND chunk_size > 0 AND overlap >= 0 AND overlap < chunk_size AND top_k > 0) OR (pipeline_type IN ('ZERO_SHOT','FINETUNED') AND kb_version IS NULL AND embedding_model_revision IS NULL AND chunker_version IS NULL AND chunk_size IS NULL AND overlap IS NULL AND top_k IS NULL AND reranker_id IS NULL AND reranker_revision IS NULL)", name="ck_experiment_retrieval_contract"),
    )


class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runtime_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        UniqueConstraint("run_id", "experiment_id"),
        UniqueConstraint("experiment_id", "attempt"),
        CheckConstraint("attempt >= 1", name="ck_run_attempt"),
    )


class Prediction(Base):
    __tablename__ = "predictions"
    prediction_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(160), nullable=False)
    experiment_id: Mapped[str] = mapped_column(String(160), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(160), nullable=False)
    incident_id: Mapped[str] = mapped_column(String(120), nullable=False)
    kb_version: Mapped[str | None] = mapped_column(String(160))
    predicted_root_cause_code: Mapped[str] = mapped_column(String(120), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    output_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (
        ForeignKeyConstraint(["run_id", "experiment_id"], ["runs.run_id", "runs.experiment_id"], ondelete="CASCADE"),
        ForeignKeyConstraint(["experiment_id", "dataset_version"], ["experiments.experiment_id", "experiments.dataset_version"], ondelete="CASCADE"),
        ForeignKeyConstraint(["experiment_id", "kb_version"], ["experiments.experiment_id", "experiments.kb_version"]),
        ForeignKeyConstraint(["dataset_version", "incident_id"], ["incidents.dataset_version", "incidents.incident_id"]),
        UniqueConstraint("run_id", "incident_id"),
        UniqueConstraint("prediction_id", "kb_version"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_prediction_confidence"),
    )


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"
    retrieval_trace_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String(160), nullable=False)
    kb_version: Mapped[str] = mapped_column(String(160), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(160), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    trace_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (
        ForeignKeyConstraint(["prediction_id", "kb_version"], ["predictions.prediction_id", "predictions.kb_version"], ondelete="CASCADE"),
        ForeignKeyConstraint(["kb_version", "chunk_id"], ["kb_chunks.kb_version", "kb_chunks.chunk_id"]),
        UniqueConstraint("prediction_id", "rank"),
        CheckConstraint("rank >= 1", name="ck_retrieval_rank"),
    )


class Metric(Base):
    __tablename__ = "metrics"
    metric_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    split: Mapped[str | None] = mapped_column(String(16))
    value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (CheckConstraint("split IS NULL OR split IN ('train','validation','test')", name="ck_metric_split"),)


class FailureAnnotation(Base):
    __tablename__ = "failure_annotations"
    failure_annotation_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.prediction_id", ondelete="CASCADE"), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(80), nullable=False)
    failure_code: Mapped[str] = mapped_column(String(160), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    annotation_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class Artifact(Base):
    __tablename__ = "artifacts"
    artifact_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(120), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    __table_args__ = (CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="ck_artifact_size"),)


class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.experiment_id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(120), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        CheckConstraint("attempt >= 1", name="ck_job_attempt"),
        CheckConstraint("status IN ('queued','running','completed','failed','cancelled')", name="ck_job_status"),
    )


class CostRecord(Base):
    __tablename__ = "cost_records"
    cost_record_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"))
    prediction_id: Mapped[str | None] = mapped_column(ForeignKey("predictions.prediction_id", ondelete="CASCADE"))
    cost_rate_snapshot_version: Mapped[str] = mapped_column(String(160), nullable=False)
    units_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    amount_usd: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    __table_args__ = (CheckConstraint("amount_usd >= 0", name="ck_cost_amount"),)
