from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.experiment_config import experiment_config_hash
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation, reload_evaluation, score_stored_run
from app.inference.base_model import IncidentInput, InferenceError
from app.inference.finetuned_protocol import Phase10Protocol, build_finetuned_experiment_config
from app.inference.protocol import load_taxonomy
from app.inference.runner import _cost_record_count, _persist_prediction_costs, _stable_id
from app.models import Experiment, Incident, Run
from app.repositories import PersistenceRepository
from app.training.inference import FineTunedAdapterPipeline


@dataclass(frozen=True)
class FineTunedRunSummary:
    experiment_id: str
    run_id: str
    split: str
    protocol_version: str
    scientific_config_hash: str
    experiment_config_hash: str
    adapter_id: str
    adapter_revision: str
    source_training_run_id: str
    prediction_count: int
    inference_failure_count: int
    cost_record_count: int
    result_hash: str
    metric_values: dict[str, float]
    recomputation_verified: bool


def _build_harness(
    *,
    protocol: Phase10Protocol,
    labels: Sequence[str],
    categories: Mapping[str, str],
) -> EvaluationHarness:
    return EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )


def _backend_adapter_identity(backend: object) -> tuple[str, str]:
    current = backend
    for _depth in range(3):
        adapter_id = getattr(current, "adapter_id", None)
        adapter_revision = getattr(current, "adapter_revision", None)
        if (
            isinstance(adapter_id, str)
            and isinstance(adapter_revision, str)
            and adapter_id.strip()
            and adapter_revision.strip()
        ):
            return adapter_id.strip(), adapter_revision.strip()
        nested = getattr(current, "backend", None)
        if nested is None or nested is current:
            break
        current = nested
    raise ValueError("fine-tuned backend does not expose adapter identity")


def _validate_pipeline_contract(
    *,
    pipeline: FineTunedAdapterPipeline,
    protocol: Phase10Protocol,
    allowed_labels: tuple[str, ...],
) -> tuple[str, str]:
    if pipeline.backend.runtime_config != protocol.runtime_config:
        raise ValueError("fine-tuned backend runtime configuration disagrees with Phase 10")
    if pipeline.allowed_labels != allowed_labels:
        raise ValueError("fine-tuned label order disagrees with the frozen taxonomy")
    if pipeline.prompt_template.version != protocol.prompt_version:
        raise ValueError("fine-tuned prompt version disagrees with the Phase 10 protocol")
    if pipeline.generation_config != protocol.generation_config:
        raise ValueError("fine-tuned generation config disagrees with the Phase 10 protocol")
    adapter_id, adapter_revision = _backend_adapter_identity(pipeline.backend)
    if adapter_revision != protocol.candidate_adapter_sha256:
        raise ValueError("fine-tuned backend adapter revision is not the frozen content hash")
    return adapter_id, adapter_revision


def _error_prediction(
    *,
    incident_id: str,
    error: InferenceError,
    attempts: int,
    protocol: Phase10Protocol,
    allowed_labels: tuple[str, ...],
) -> Prediction:
    return Prediction.from_raw_output(
        incident_id=incident_id,
        raw_model_output="",
        allowed_labels=set(allowed_labels),
        pipeline_metadata={
            "pipeline_type": "FINETUNED",
            "prompt_version": protocol.prompt_version,
            "output_schema_version": protocol.output_schema_version,
            "generation_config": protocol.generation_config.model_dump(mode="json"),
            "adapter_id": protocol.candidate_adapter_artifact_reference,
            "adapter_revision": protocol.candidate_adapter_sha256,
            "inference_error": {
                "type": type(error).__name__,
                "message": str(error),
                "attempts": attempts,
            },
        },
    )


def _predict_with_retry(
    *,
    pipeline: FineTunedAdapterPipeline,
    incident: IncidentInput,
    protocol: Phase10Protocol,
    allowed_labels: tuple[str, ...],
    max_attempts: int,
) -> tuple[Prediction, bool]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    last_error: InferenceError | None = None
    for _attempt in range(1, max_attempts + 1):
        try:
            prediction = pipeline.predict(incident)
            if not isinstance(prediction, Prediction):
                raise TypeError("fine-tuned pipeline returned a non-Prediction value")
            return prediction, False
        except InferenceError as exc:
            last_error = exc
    if last_error is None:
        raise AssertionError("fine-tuned retry loop exited without prediction or inference error")
    return (
        _error_prediction(
            incident_id=incident.incident_id,
            error=last_error,
            attempts=max_attempts,
            protocol=protocol,
            allowed_labels=allowed_labels,
        ),
        True,
    )


def _completed_summary(
    session: Session,
    *,
    experiment: Experiment,
    run_id: str,
    split: str,
    protocol: Phase10Protocol,
    harness: EvaluationHarness,
) -> FineTunedRunSummary | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    if run.status != "completed":
        raise RuntimeError("completed fine-tuned experiment references a non-completed run")
    snapshot = reload_evaluation(session, run_id=run_id)
    upfront_cost_usd = float(run.runtime_metadata.get("upfront_cost_usd", 0.0))
    rescored = score_stored_run(
        session,
        run_id=run_id,
        harness=harness,
        upfront_cost_usd=upfront_cost_usd,
    )
    if (
        snapshot.result_hash != rescored.result_hash
        or snapshot.metric_values != rescored.metric_map()
    ):
        raise RuntimeError("completed fine-tuned run failed metric recomputation")
    scientific_hash = protocol.scientific_config_hash()
    if run.runtime_metadata.get("phase10_scientific_config_hash") != scientific_hash:
        raise RuntimeError("completed fine-tuned run disagrees with Phase 10 scientific identity")
    if run.runtime_metadata.get("phase10_adapter_sha256") != protocol.candidate_adapter_sha256:
        raise RuntimeError("completed fine-tuned run disagrees with frozen adapter identity")
    return FineTunedRunSummary(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        split=split,
        protocol_version=protocol.protocol_version,
        scientific_config_hash=scientific_hash,
        experiment_config_hash=experiment.config_hash,
        adapter_id=protocol.candidate_adapter_artifact_reference,
        adapter_revision=protocol.candidate_adapter_sha256,
        source_training_run_id=protocol.source_training_run_id,
        prediction_count=len(snapshot.prediction_payloads),
        inference_failure_count=int(run.runtime_metadata.get("inference_failure_count", 0)),
        cost_record_count=_cost_record_count(session, run_id=run_id),
        result_hash=snapshot.result_hash,
        metric_values=snapshot.metric_values,
        recomputation_verified=True,
    )


def run_finetuned_experiment(
    session: Session,
    *,
    root: Path,
    protocol: Phase10Protocol,
    pipeline: FineTunedAdapterPipeline,
    split: str,
    git_commit: str,
    hardware_runtime_descriptor: str,
    cost_rate_snapshot_version: str,
    experiment_id: str | None = None,
    run_id: str | None = None,
    max_attempts: int = 2,
    upfront_cost_usd: float = 0.0,
) -> FineTunedRunSummary:
    evaluation_split = protocol.assert_split_allowed(split)
    labels, categories = load_taxonomy(root)
    _adapter_id, _adapter_revision = _validate_pipeline_contract(
        pipeline=pipeline,
        protocol=protocol,
        allowed_labels=labels,
    )
    harness = _build_harness(protocol=protocol, labels=labels, categories=categories)
    config = build_finetuned_experiment_config(
        root=root,
        protocol=protocol,
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
        protocol.candidate_adapter_sha256,
        canonical_config_hash,
    )
    experiment = repo.create_experiment(requested_experiment_id, config, status="running")
    actual_run_id = run_id or _stable_id(
        "run",
        experiment.experiment_id,
        evaluation_split,
        scientific_hash,
        protocol.candidate_adapter_sha256,
    )
    if experiment.status == "completed":
        completed = _completed_summary(
            session,
            experiment=experiment,
            run_id=actual_run_id,
            split=evaluation_split,
            protocol=protocol,
            harness=harness,
        )
        if completed is None:
            raise ValueError("completed fine-tuned experiment has no matching reusable run")
        return completed
    if experiment.status not in {"planned", "queued", "running"}:
        raise ValueError(f"fine-tuned experiment is not runnable from status: {experiment.status}")
    experiment.status = "running"
    session.flush()

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

    started_at = datetime.now(UTC)
    predictions: list[Prediction] = []
    examples: list[EvaluationExample] = []
    inference_failure_count = 0
    for incident in incidents:
        prediction, failed = _predict_with_retry(
            pipeline=pipeline,
            incident=IncidentInput(
                incident_id=incident.incident_id,
                title=incident.title,
                description=incident.description,
            ),
            protocol=protocol,
            allowed_labels=labels,
            max_attempts=max_attempts,
        )
        predictions.append(prediction)
        inference_failure_count += int(failed)
        examples.append(
            EvaluationExample(
                incident_id=incident.incident_id,
                split=evaluation_split,
                root_cause_code=incident.root_cause_code,
                root_cause_category=incident.root_cause_category,
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
        upfront_cost_usd=upfront_cost_usd,
    )
    recomputation_verified = (
        snapshot.result_hash == result.result_hash == rescored.result_hash
        and snapshot.metric_values == result.metric_map() == rescored.metric_map()
    )
    if not recomputation_verified:
        raise RuntimeError("persisted fine-tuned metrics do not reproduce from raw predictions")

    run = session.get(Run, actual_run_id)
    if run is None:
        raise AssertionError("persisted fine-tuned evaluation did not create a run row")
    completed_at = datetime.now(UTC)
    run.started_at = run.started_at or started_at
    run.completed_at = completed_at
    run.runtime_metadata = {
        **run.runtime_metadata,
        "phase10_protocol_version": protocol.protocol_version,
        "phase10_scientific_config_hash": scientific_hash,
        "phase10_adapter_sha256": protocol.candidate_adapter_sha256,
        "phase10_source_training_run_id": protocol.source_training_run_id,
        "inference_failure_count": inference_failure_count,
        "prediction_count": len(predictions),
        "cost_record_count": cost_record_count,
        "upfront_cost_usd": upfront_cost_usd,
        "metric_recomputation_verified": True,
    }
    experiment.status = "completed"
    experiment.completed_at = completed_at
    session.flush()

    return FineTunedRunSummary(
        experiment_id=experiment.experiment_id,
        run_id=actual_run_id,
        split=evaluation_split,
        protocol_version=protocol.protocol_version,
        scientific_config_hash=scientific_hash,
        experiment_config_hash=experiment.config_hash,
        adapter_id=protocol.candidate_adapter_artifact_reference,
        adapter_revision=protocol.candidate_adapter_sha256,
        source_training_run_id=protocol.source_training_run_id,
        prediction_count=len(predictions),
        inference_failure_count=inference_failure_count,
        cost_record_count=cost_record_count,
        result_hash=result.result_hash,
        metric_values=result.metric_map(),
        recomputation_verified=True,
    )
