from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.experiment_config import experiment_config_hash
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation, score_stored_run
from app.inference.base_model import IncidentInput, InferenceError, ZeroShotBaselineAdapter
from app.inference.protocol import (
    BaselineProtocol,
    build_experiment_config,
    load_taxonomy,
)
from app.models import Incident, Run
from app.repositories import PersistenceRepository


@dataclass(frozen=True)
class BaselineRunSummary:
    experiment_id: str
    run_id: str
    split: str
    protocol_version: str
    scientific_config_hash: str
    experiment_config_hash: str
    prediction_count: int
    inference_failure_count: int
    result_hash: str
    metric_values: dict[str, float]
    recomputation_verified: bool


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _error_prediction(
    *,
    incident_id: str,
    error: InferenceError,
    attempts: int,
    protocol: BaselineProtocol,
    allowed_labels: tuple[str, ...],
) -> Prediction:
    return Prediction.from_raw_output(
        incident_id=incident_id,
        raw_model_output="",
        allowed_labels=set(allowed_labels),
        pipeline_metadata={
            "pipeline_type": "ZERO_SHOT",
            "prompt_version": protocol.prompt_version,
            "output_schema_version": protocol.output_schema_version,
            "generation_config": protocol.generation_config.model_dump(mode="json"),
            "inference_error": {
                "type": type(error).__name__,
                "message": str(error),
                "attempts": attempts,
            },
        },
    )


def _predict_with_retry(
    *,
    adapter: ZeroShotBaselineAdapter,
    incident: IncidentInput,
    protocol: BaselineProtocol,
    allowed_labels: tuple[str, ...],
    max_attempts: int,
) -> tuple[Prediction, bool]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    last_error: InferenceError | None = None
    for _attempt in range(1, max_attempts + 1):
        try:
            return adapter.predict(incident), False
        except InferenceError as exc:
            last_error = exc
    if last_error is None:
        raise AssertionError("retry loop exited without a prediction or inference error")
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


def _validate_adapter_contract(
    adapter: ZeroShotBaselineAdapter,
    protocol: BaselineProtocol,
    allowed_labels: tuple[str, ...],
) -> None:
    if adapter.backend.runtime_config != protocol.runtime_config:
        raise ValueError("adapter runtime configuration disagrees with frozen Phase 06 protocol")
    if adapter.allowed_labels != allowed_labels:
        raise ValueError("adapter label order disagrees with frozen taxonomy")
    if adapter.prompt_template.version != protocol.prompt_version:
        raise ValueError("adapter prompt version disagrees with Phase 06 protocol")
    if adapter.generation_config != protocol.generation_config:
        raise ValueError("adapter generation config disagrees with Phase 06 protocol")


def run_baseline_experiment(
    session: Session,
    *,
    root: Path,
    protocol: BaselineProtocol,
    adapter: ZeroShotBaselineAdapter,
    split: str,
    git_commit: str,
    hardware_runtime_descriptor: str,
    cost_rate_snapshot_version: str,
    experiment_id: str | None = None,
    run_id: str | None = None,
    max_attempts: int = 2,
    upfront_cost_usd: float = 0.0,
) -> BaselineRunSummary:
    evaluation_split: Literal["validation", "test"] = protocol.assert_split_allowed(split)
    labels, categories = load_taxonomy(root)
    _validate_adapter_contract(adapter, protocol, labels)
    config = build_experiment_config(
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
        "experiment", protocol.protocol_version, canonical_config_hash
    )
    experiment = repo.create_experiment(requested_experiment_id, config, status="running")
    if experiment.status not in {"planned", "queued", "running"}:
        raise ValueError(f"experiment is not runnable from status: {experiment.status}")
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

    predictions: list[Prediction] = []
    examples: list[EvaluationExample] = []
    inference_failure_count = 0
    for incident in incidents:
        prediction, failed = _predict_with_retry(
            adapter=adapter,
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

    harness = EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )
    result = harness.evaluate(
        examples,
        predictions,
        upfront_cost_usd=upfront_cost_usd,
    )
    actual_run_id = run_id or _stable_id(
        "run",
        experiment.experiment_id,
        evaluation_split,
        scientific_hash,
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
    rescored = score_stored_run(session, run_id=actual_run_id, harness=harness)
    recomputation_verified = (
        snapshot.result_hash == result.result_hash == rescored.result_hash
        and snapshot.metric_values == result.metric_map() == rescored.metric_map()
    )
    if not recomputation_verified:
        raise RuntimeError("persisted baseline metrics do not reproduce from raw predictions")

    run = session.get(Run, actual_run_id)
    if run is None:
        raise AssertionError("persisted evaluation did not create a run row")
    run.runtime_metadata = {
        **run.runtime_metadata,
        "phase6_protocol_version": protocol.protocol_version,
        "phase6_scientific_config_hash": scientific_hash,
        "inference_failure_count": inference_failure_count,
        "prediction_count": len(predictions),
        "metric_recomputation_verified": True,
    }
    experiment.status = "completed"
    experiment.completed_at = datetime.now(UTC)
    session.flush()

    return BaselineRunSummary(
        experiment_id=experiment.experiment_id,
        run_id=actual_run_id,
        split=evaluation_split,
        protocol_version=protocol.protocol_version,
        scientific_config_hash=scientific_hash,
        experiment_config_hash=experiment.config_hash,
        prediction_count=len(predictions),
        inference_failure_count=inference_failure_count,
        result_hash=result.result_hash,
        metric_values=result.metric_map(),
        recomputation_verified=True,
    )
