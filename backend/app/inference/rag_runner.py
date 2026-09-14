from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.experiment_config import experiment_config_hash
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation, reload_evaluation, score_stored_run
from app.inference.base_model import IncidentInput, InferenceError
from app.inference.protocol import load_taxonomy
from app.inference.rag_ablations import RAGVariant, load_ablation_suite
from app.inference.rag_persistence import persist_rag_retrieval_traces
from app.inference.rag_pipeline import RAGPipeline, RAGPipelineResult, RAGRetrievalTrace
from app.inference.rag_protocol import RAGProtocol, build_rag_experiment_config
from app.inference.runner import _cost_record_count, _persist_prediction_costs, _stable_id
from app.models import (
    Experiment,
    Incident,
    KBChunk,
    KBDocument,
    KnowledgeBaseVersion,
    RetrievalTrace,
    Run,
)
from app.models import Prediction as PredictionRow
from app.repositories import PersistenceRepository
from app.retrieval.reranker import NoOpReranker

RELEVANCE_REFERENCE_VERSION = "phase8-runbook-root-cause-reference-v1"


@dataclass(frozen=True)
class RAGRunSummary:
    experiment_id: str
    run_id: str
    split: str
    protocol_version: str
    scientific_config_hash: str
    experiment_config_hash: str
    variant_id: str
    ablation_suite_hash: str
    prediction_count: int
    retrieval_trace_count: int
    no_context_prediction_count: int
    inference_retry_count: int
    cost_record_count: int
    result_hash: str
    metric_values: dict[str, float]
    relevance_reference_version: str
    recomputation_verified: bool


def _build_harness(
    *,
    protocol: RAGProtocol,
    labels: Sequence[str],
    categories: Mapping[str, str],
) -> EvaluationHarness:
    return EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )


def _validate_registered_variant(
    *,
    root: Path,
    protocol: RAGProtocol,
    variant: RAGVariant,
    split: str,
) -> None:
    suite = load_ablation_suite(root / protocol.ablation_suite_path)
    if suite.config_hash() != protocol.ablation_suite_hash:
        raise ValueError("Phase 08 ablation suite hash disagrees with the RAG protocol")
    by_id = {candidate.variant_id: candidate for candidate in suite.variants}
    registered = by_id.get(variant.variant_id)
    if registered is None or registered != variant:
        raise ValueError(
            "RAG variant is not an exact member of the registered Phase 08 ablation suite"
        )
    if split == "test" and variant.variant_id != protocol.selected_variant_id:
        raise ValueError("locked test may run only the validation-selected frozen RAG variant")


def _validate_registered_kb(
    session: Session,
    *,
    variant: RAGVariant,
) -> None:
    kb = session.get(KnowledgeBaseVersion, variant.knowledge_base_version)
    if kb is None:
        raise ValueError(f"RAG knowledge base is not registered: {variant.knowledge_base_version}")
    expected = {
        "embedding_model_revision": variant.embedding_model_revision,
        "chunker_version": variant.chunker_version,
        "chunk_size": variant.chunk_size,
        "overlap": variant.overlap,
    }
    actual = {
        "embedding_model_revision": kb.embedding_model_revision,
        "chunker_version": kb.chunker_version,
        "chunk_size": kb.chunk_size,
        "overlap": kb.overlap,
    }
    if actual != expected:
        raise ValueError(
            "registered RAG knowledge-base identity disagrees with the ablation variant: "
            f"actual={actual!r} expected={expected!r}"
        )


def _validate_pipeline_contract(
    *,
    pipeline: RAGPipeline,
    protocol: RAGProtocol,
    variant: RAGVariant,
    allowed_labels: tuple[str, ...],
) -> None:
    if pipeline.backend.runtime_config != protocol.runtime_config:
        raise ValueError("RAG backend runtime configuration disagrees with Phase 08 protocol")
    if pipeline.allowed_labels != allowed_labels:
        raise ValueError("RAG label order disagrees with the frozen taxonomy")
    if pipeline.retriever.kb_version != variant.knowledge_base_version:
        raise ValueError("RAG retriever knowledge-base version disagrees with the ablation variant")
    if pipeline.top_k != variant.top_k:
        raise ValueError("RAG top_k disagrees with the ablation variant")
    if pipeline.prompt_template.version != protocol.prompt_version:
        raise ValueError("RAG prompt version disagrees with the Phase 08 protocol")
    if pipeline.generation_config != protocol.generation_config:
        raise ValueError("RAG generation config disagrees with the Phase 08 protocol")
    if pipeline.context_budget.policy_version != variant.context_policy_version:
        raise ValueError("RAG context policy version disagrees with the ablation variant")
    if pipeline.context_budget.max_context_tokens != variant.max_context_tokens:
        raise ValueError("RAG context token budget disagrees with the ablation variant")
    if pipeline.context_budget.max_chunk_tokens != variant.max_chunk_tokens:
        raise ValueError("RAG per-chunk token budget disagrees with the ablation variant")

    expected_reranker_id = variant.reranker_id or NoOpReranker.reranker_id
    expected_reranker_revision = variant.reranker_revision or NoOpReranker.revision
    if pipeline.reranker.reranker_id != expected_reranker_id:
        raise ValueError("RAG reranker id disagrees with the ablation variant")
    if pipeline.reranker.revision != expected_reranker_revision:
        raise ValueError("RAG reranker revision disagrees with the ablation variant")


def _runbook_relevance_references(
    session: Session,
    *,
    kb_version: str,
    incidents: Sequence[Incident],
) -> dict[str, tuple[str, ...]]:
    documents = session.scalars(
        select(KBDocument)
        .where(KBDocument.kb_version == kb_version)
        .order_by(KBDocument.document_id)
    ).all()
    relevant_documents: dict[str, set[str]] = {}
    for document in documents:
        metadata = document.document_metadata
        if metadata.get("source_type") != "runbook":
            continue
        if metadata.get("research_eligible") is not True:
            continue
        label = metadata.get("root_cause_code")
        if isinstance(label, str) and label:
            relevant_documents.setdefault(label, set()).add(document.document_id)

    chunks = session.scalars(
        select(KBChunk)
        .where(KBChunk.kb_version == kb_version)
        .order_by(KBChunk.document_id, KBChunk.chunk_index, KBChunk.chunk_id)
    ).all()
    chunks_by_label: dict[str, list[str]] = {label: [] for label in relevant_documents}
    for chunk in chunks:
        for label, document_ids in relevant_documents.items():
            if chunk.document_id in document_ids:
                chunks_by_label[label].append(chunk.chunk_id)

    references: dict[str, tuple[str, ...]] = {}
    for incident in incidents:
        chunk_ids = tuple(chunks_by_label.get(incident.root_cause_code, ()))
        if not chunk_ids:
            raise ValueError(
                "retrieval metric reference is missing a research-eligible runbook for "
                f"root_cause_code={incident.root_cause_code!r}"
            )
        references[incident.incident_id] = chunk_ids
    return references


def _predict_with_retry(
    *,
    pipeline: RAGPipeline,
    incident: IncidentInput,
    query_family_id: str,
    max_attempts: int,
) -> tuple[RAGPipelineResult, int]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    last_error: InferenceError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return pipeline.predict(incident, query_family_id=query_family_id), attempt - 1
        except InferenceError as exc:
            last_error = exc
    if last_error is None:
        raise AssertionError("RAG retry loop exited without a result or inference error")
    raise last_error


def _retrieval_trace_count(session: Session, *, run_id: str) -> int:
    value = session.scalar(
        select(func.count())
        .select_from(RetrievalTrace)
        .join(PredictionRow, RetrievalTrace.prediction_id == PredictionRow.prediction_id)
        .where(PredictionRow.run_id == run_id)
    )
    return int(value or 0)


def _completed_summary(
    session: Session,
    *,
    experiment: Experiment,
    run_id: str,
    split: str,
    protocol: RAGProtocol,
    variant: RAGVariant,
    harness: EvaluationHarness,
    relevance_references: Mapping[str, Sequence[str]],
) -> RAGRunSummary | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    if run.status != "completed":
        raise RuntimeError("completed RAG experiment references a non-completed run")
    snapshot = reload_evaluation(session, run_id=run_id)
    upfront_cost_usd = float(run.runtime_metadata.get("upfront_cost_usd", 0.0))
    rescored = score_stored_run(
        session,
        run_id=run_id,
        harness=harness,
        relevant_chunk_ids_by_incident=relevance_references,
        upfront_cost_usd=upfront_cost_usd,
    )
    if (
        snapshot.result_hash != rescored.result_hash
        or snapshot.metric_values != rescored.metric_map()
    ):
        raise RuntimeError("completed RAG run failed metric recomputation")
    scientific_hash = protocol.scientific_config_hash()
    if run.runtime_metadata.get("phase8_scientific_config_hash") != scientific_hash:
        raise RuntimeError("completed RAG run disagrees with the Phase 08 scientific identity")
    if run.runtime_metadata.get("phase8_variant_id") != variant.variant_id:
        raise RuntimeError("completed RAG run disagrees with the selected ablation variant")
    return RAGRunSummary(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        split=split,
        protocol_version=protocol.protocol_version,
        scientific_config_hash=scientific_hash,
        experiment_config_hash=experiment.config_hash,
        variant_id=variant.variant_id,
        ablation_suite_hash=protocol.ablation_suite_hash,
        prediction_count=len(snapshot.prediction_payloads),
        retrieval_trace_count=_retrieval_trace_count(session, run_id=run_id),
        no_context_prediction_count=int(run.runtime_metadata.get("no_context_prediction_count", 0)),
        inference_retry_count=int(run.runtime_metadata.get("inference_retry_count", 0)),
        cost_record_count=_cost_record_count(session, run_id=run_id),
        result_hash=snapshot.result_hash,
        metric_values=snapshot.metric_values,
        relevance_reference_version=RELEVANCE_REFERENCE_VERSION,
        recomputation_verified=True,
    )


def run_rag_experiment(
    session: Session,
    *,
    root: Path,
    protocol: RAGProtocol,
    variant: RAGVariant,
    pipeline: RAGPipeline,
    split: str,
    git_commit: str,
    hardware_runtime_descriptor: str,
    cost_rate_snapshot_version: str,
    experiment_id: str | None = None,
    run_id: str | None = None,
    max_attempts: int = 2,
    upfront_cost_usd: float = 0.0,
) -> RAGRunSummary:
    evaluation_split = protocol.assert_split_allowed(split)
    _validate_registered_variant(
        root=root,
        protocol=protocol,
        variant=variant,
        split=evaluation_split,
    )
    labels, categories = load_taxonomy(root)
    _validate_pipeline_contract(
        pipeline=pipeline,
        protocol=protocol,
        variant=variant,
        allowed_labels=labels,
    )
    _validate_registered_kb(session, variant=variant)
    harness = _build_harness(protocol=protocol, labels=labels, categories=categories)
    config = build_rag_experiment_config(
        root=root,
        protocol=protocol,
        variant=variant,
        git_commit=git_commit,
        hardware_runtime_descriptor=hardware_runtime_descriptor,
        cost_rate_snapshot_version=cost_rate_snapshot_version,
    )
    canonical_config_hash = experiment_config_hash(config)
    scientific_hash = protocol.scientific_config_hash()
    repo = PersistenceRepository(session)
    requested_experiment_id = experiment_id or _stable_id(
        "experiment",
        protocol.protocol_version,
        variant.variant_id,
        canonical_config_hash,
    )
    experiment = repo.create_experiment(requested_experiment_id, config, status="running")
    actual_run_id = run_id or _stable_id(
        "run",
        experiment.experiment_id,
        evaluation_split,
        scientific_hash,
        variant.variant_id,
    )

    incidents = session.scalars(
        select(Incident)
        .where(
            Incident.dataset_version == protocol.dataset_version,
            Incident.split == evaluation_split,
        )
        .order_by(Incident.incident_id)
    ).all()
    if not incidents:
        raise ValueError(f"no incidents found for split: {evaluation_split}")
    relevance_references = _runbook_relevance_references(
        session,
        kb_version=variant.knowledge_base_version,
        incidents=incidents,
    )

    if experiment.status == "completed":
        completed = _completed_summary(
            session,
            experiment=experiment,
            run_id=actual_run_id,
            split=evaluation_split,
            protocol=protocol,
            variant=variant,
            harness=harness,
            relevance_references=relevance_references,
        )
        if completed is None:
            raise ValueError("completed RAG experiment has no matching reusable run")
        return completed
    if experiment.status not in {"planned", "queued", "running"}:
        raise ValueError(f"RAG experiment is not runnable from status: {experiment.status}")
    experiment.status = "running"
    session.flush()

    started_at = datetime.now(UTC)
    predictions: list[Prediction] = []
    examples: list[EvaluationExample] = []
    traces_by_incident: dict[str, tuple[RAGRetrievalTrace, ...]] = {}
    no_context_prediction_count = 0
    inference_retry_count = 0
    try:
        for incident in incidents:
            rag_result, retries = _predict_with_retry(
                pipeline=pipeline,
                incident=IncidentInput(
                    incident_id=incident.incident_id,
                    title=incident.title,
                    description=incident.description,
                ),
                query_family_id=incident.family_id,
                max_attempts=max_attempts,
            )
            prediction = rag_result.prediction
            predictions.append(prediction)
            traces_by_incident[incident.incident_id] = rag_result.retrieval_traces
            inference_retry_count += retries
            no_context_prediction_count += int(rag_result.context.no_context_fallback)
            examples.append(
                EvaluationExample(
                    incident_id=incident.incident_id,
                    split=evaluation_split,
                    root_cause_code=incident.root_cause_code,
                    root_cause_category=incident.root_cause_category,
                    relevant_chunk_ids=relevance_references[incident.incident_id],
                )
            )

        result = harness.evaluate(
            examples,
            predictions,
            upfront_cost_usd=upfront_cost_usd,
        )
        snapshot = persist_evaluation(
            session,
            experiment_id=experiment.experiment_id,
            run_id=actual_run_id,
            dataset_version=protocol.dataset_version,
            predictions=predictions,
            result=result,
            split=evaluation_split,
            kb_version=variant.knowledge_base_version,
        )
        retrieval_trace_count = persist_rag_retrieval_traces(
            session,
            run_id=actual_run_id,
            kb_version=variant.knowledge_base_version,
            traces_by_incident=traces_by_incident,
        )
        cost_record_count = _persist_prediction_costs(
            session,
            experiment=experiment,
            run_id=actual_run_id,
            predictions=predictions,
        )
        rescored = score_stored_run(
            session,
            run_id=actual_run_id,
            harness=harness,
            relevant_chunk_ids_by_incident=relevance_references,
            upfront_cost_usd=upfront_cost_usd,
        )
        recomputation_verified = (
            snapshot.result_hash == result.result_hash == rescored.result_hash
            and snapshot.metric_values == result.metric_map() == rescored.metric_map()
        )
        if not recomputation_verified:
            raise RuntimeError("persisted RAG metrics do not reproduce from raw predictions")

        run = session.get(Run, actual_run_id)
        if run is None:
            raise AssertionError("persisted RAG evaluation did not create a run row")
        completed_at = datetime.now(UTC)
        run.started_at = run.started_at or started_at
        run.completed_at = completed_at
        run.runtime_metadata = {
            **run.runtime_metadata,
            "phase8_protocol_version": protocol.protocol_version,
            "phase8_scientific_config_hash": scientific_hash,
            "phase8_variant_id": variant.variant_id,
            "phase8_ablation_suite_hash": protocol.ablation_suite_hash,
            "relevance_reference_version": RELEVANCE_REFERENCE_VERSION,
            "retrieval_trace_count": retrieval_trace_count,
            "no_context_prediction_count": no_context_prediction_count,
            "inference_retry_count": inference_retry_count,
            "prediction_count": len(predictions),
            "cost_record_count": cost_record_count,
            "upfront_cost_usd": upfront_cost_usd,
            "metric_recomputation_verified": True,
        }
        experiment.status = "completed"
        experiment.completed_at = completed_at
        session.flush()
    except Exception:
        experiment.status = "failed"
        run = session.get(Run, actual_run_id)
        if run is not None:
            run.status = "failed"
            run.completed_at = datetime.now(UTC)
        session.flush()
        raise

    return RAGRunSummary(
        experiment_id=experiment.experiment_id,
        run_id=actual_run_id,
        split=evaluation_split,
        protocol_version=protocol.protocol_version,
        scientific_config_hash=scientific_hash,
        experiment_config_hash=experiment.config_hash,
        variant_id=variant.variant_id,
        ablation_suite_hash=protocol.ablation_suite_hash,
        prediction_count=len(predictions),
        retrieval_trace_count=retrieval_trace_count,
        no_context_prediction_count=no_context_prediction_count,
        inference_retry_count=inference_retry_count,
        cost_record_count=cost_record_count,
        result_hash=result.result_hash,
        metric_values=result.metric_map(),
        relevance_reference_version=RELEVANCE_REFERENCE_VERSION,
        recomputation_verified=True,
    )
