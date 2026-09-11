from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.experiment_config import experiment_config_hash
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation, reload_evaluation, score_stored_run
from app.inference.base_model import IncidentInput, InferenceError, ZeroShotBaselineAdapter
from app.inference.protocol import (
    BaselineProtocol,
    build_experiment_config,
    load_taxonomy,
)
from app.models import CostRecord, Experiment, Incident, Run
from app.models import Prediction as PredictionRow
from app.repositories import PersistenceRepository

_COST_QUANTUM = Decimal("0.00000001")


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
    cost_record_count: int
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


def _build_harness(
    *,
    protocol: BaselineProtocol,
    labels: Sequence[str],
    categories: Mapping[str, str],
) -> EvaluationHarness:
    return EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )


def _cost_units(prediction: Prediction) -> dict[str, object]:
    units: dict[str, object] = {
        "method": "backend_reported_cost_usd",
        "input_tokens": prediction.input_tokens,
        "output_tokens": prediction.output_tokens,
        "total_tokens": prediction.total_tokens,
        "latency_ms": prediction.latency_ms,
    }
    runtime = prediction.pipeline_metadata.get("runtime")
    if isinstance(runtime, Mapping):
        method = runtime.get("cost_method")
        if isinstance(method, str) and method:
            units["method"] = method
        gpu_hour_usd = runtime.get("gpu_hour_usd")
        if isinstance(gpu_hour_usd, int | float):
            units["gpu_hour_usd"] = float(gpu_hour_usd)
    return units


def _persist_prediction_costs(
    session: Session,
    *,
    experiment: Experiment,
    run_id: str,
    predictions: Sequence[Prediction],
) -> int:
    prediction_rows = session.scalars(
        select(PredictionRow)
        .where(PredictionRow.run_id == run_id)
        .order_by(PredictionRow.incident_id)
    ).all()
    row_by_incident = {row.incident_id: row for row in prediction_rows}
    if set(row_by_incident) != {prediction.incident_id for prediction in predictions}:
        raise RuntimeError("persisted predictions are incomplete before cost recording")

    count = 0
    for prediction in predictions:
        if prediction.cost_usd is None:
            continue
        prediction_row = row_by_incident[prediction.incident_id]
        cost_record_id = _stable_id(
            "cost",
            run_id,
            prediction.incident_id,
            experiment.cost_rate_snapshot_version,
        )
        amount = Decimal(str(prediction.cost_usd)).quantize(_COST_QUANTUM)
        units = _cost_units(prediction)
        existing = session.get(CostRecord, cost_record_id)
        if existing is None:
            session.add(
                CostRecord(
                    cost_record_id=cost_record_id,
                    experiment_id=experiment.experiment_id,
                    run_id=run_id,
                    prediction_id=prediction_row.prediction_id,
                    cost_rate_snapshot_version=experiment.cost_rate_snapshot_version,
                    units_json=units,
                    amount_usd=amount,
                )
            )
        elif (
            existing.experiment_id != experiment.experiment_id
            or existing.run_id != run_id
            or existing.prediction_id != prediction_row.prediction_id
            or existing.cost_rate_snapshot_version != experiment.cost_rate_snapshot_version
            or existing.units_json != units
            or existing.amount_usd != amount
        ):
            raise ValueError("persisted cost identity was reused with different semantics")
        count += 1
    session.flush()
    return count


def _cost_record_count(session: Session, *, run_id: str) -> int:
    value = session.scalar(
        select(func.count()).select_from(CostRecord).where(CostRecord.run_id == run_id)
    )
    return int(value or 0)


def _completed_summary(
    session: Session,
    *,
    experiment: Experiment,
    run_id: str,
    split: str,
    protocol: BaselineProtocol,
    harness: EvaluationHarness,
) -> BaselineRunSummary | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    if run.status != "completed":
        raise RuntimeError("completed experiment references a non-completed run")
    snapshot = reload_evaluation(session, run_id=run_id)
    rescored = score_stored_run(session, run_id=run_id, harness=harness)
    if snapshot.result_hash != rescored.result_hash or snapshot.metric_values != rescored.metric_map():
        raise RuntimeError("completed baseline run failed metric recomputation")
    scientific_hash = protocol.scientific_config_hash()
    if run.runtime_metadata.get("phase6_scientific_config_hash") != scientific_hash:
        raise RuntimeError("completed baseline run disagrees with the Phase 06 scientific identity")
    failure_count = int(run.runtime_metadata.get("inference_failure_count", 0))
    return BaselineRunSummary(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        split=split,
        protocol_version=protocol.protocol_version,
        scientific_config_hash=scientific_hash,
        experiment_config_hash=experiment.config_hash,
        prediction_count=len(snapshot.prediction_payloads),
        inference_failure_count=failure_count,
        cost_record_count=_cost_record_count(session, run_id=run_id),
        result_hash=snapshot.result_hash,
        metric_values=snapshot.metric_values,
        recomputation_verified=True,
    )


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
    harness = _build_harness(protocol=protocol, labels=labels, categories=categories)
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
    actual_run_id = run_id or _stable_id(
        "run",
        experiment.experiment_id,
        evaluation_split,
        scientific_hash,
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
            raise ValueError("completed experiment has no matching reusable run")
        return completed
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

    started_at = datetime.now(UTC)
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
    completed_at = datetime.now(UTC)
    run.started_at = run.started_at or started_at
    run.completed_at = completed_at
    run.runtime_metadata = {
        **run.runtime_metadata,
        "phase6_protocol_version": protocol.protocol_version,
        "phase6_scientific_config_hash": scientific_hash,
        "inference_failure_count": inference_failure_count,
        "prediction_count": len(predictions),
        "cost_record_count": cost_record_count,
        "metric_recomputation_verified": True,
    }
    experiment.status = "completed"
    experiment.completed_at = completed_at
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
        cost_record_count=cost_record_count,
        result_hash=result.result_hash,
        metric_values=result.metric_map(),
        recomputation_verified=True,
    )
