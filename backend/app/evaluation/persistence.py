from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.contracts import EvaluationExample, Prediction, canonical_prediction_payload
from app.evaluation.harness import EvaluationHarness, EvaluationResult
from app.models import (
    Experiment,
    FailureAnnotation as FailureAnnotationRow,
    Incident,
    Metric as MetricRow,
    Prediction as PredictionRow,
    Run,
)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class PersistedEvaluationSnapshot:
    result_hash: str
    prediction_payloads: tuple[dict[str, object], ...]
    metric_values: dict[str, float]
    failure_codes: tuple[tuple[str, str], ...]


def persist_evaluation(
    session: Session,
    *,
    experiment_id: str,
    run_id: str,
    dataset_version: str,
    predictions: Sequence[Prediction],
    result: EvaluationResult,
    split: str,
    attempt: int = 1,
    kb_version: str | None = None,
) -> PersistedEvaluationSnapshot:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise ValueError(f"unknown experiment: {experiment_id}")
    if experiment.dataset_version != dataset_version:
        raise ValueError("evaluation dataset_version disagrees with experiment")
    if experiment.kb_version != kb_version:
        raise ValueError("evaluation kb_version disagrees with experiment")
    if split not in {"train", "validation", "test"}:
        raise ValueError("evaluation split must be train, validation, or test")
    if result.prediction_count != len(predictions):
        raise ValueError("evaluation result prediction count does not match predictions")

    run = session.get(Run, run_id)
    if run is None:
        run = Run(
            run_id=run_id,
            experiment_id=experiment_id,
            attempt=attempt,
            status="completed",
            runtime_metadata={
                "evaluator_version": result.evaluator_version,
                "evaluation_result_hash": result.result_hash,
            },
        )
        session.add(run)
        session.flush()
    elif run.experiment_id != experiment_id:
        raise ValueError("run belongs to a different experiment")

    prediction_id_by_incident: dict[str, str] = {}
    for prediction in sorted(predictions, key=lambda item: item.incident_id):
        prediction_id = _stable_id("prediction", run_id, prediction.incident_id)
        prediction_id_by_incident[prediction.incident_id] = prediction_id
        payload = canonical_prediction_payload(prediction)
        row = session.get(PredictionRow, prediction_id)
        if row is None:
            row = PredictionRow(
                prediction_id=prediction_id,
                run_id=run_id,
                experiment_id=experiment_id,
                dataset_version=dataset_version,
                incident_id=prediction.incident_id,
                kb_version=kb_version,
                predicted_root_cause_code=prediction.predicted_root_cause_code
                or "__PARSE_FAILURE__",
                confidence=prediction.confidence_probability,
                output_json=payload,
                latency_ms=prediction.latency_ms,
            )
            session.add(row)
            session.flush()
        elif row.output_json != payload:
            raise ValueError("persisted prediction identity was reused with different payload")

    for metric in result.metrics:
        metric_id = _stable_id("metric", run_id, split, metric.name)
        metadata = dict(metric.metadata)
        metadata["evaluation_result_hash"] = result.result_hash
        if metric.name == "calibration.ece":
            metadata["reliability"] = list(result.reliability)
        row = session.get(MetricRow, metric_id)
        if row is None:
            session.add(
                MetricRow(
                    metric_id=metric_id,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    name=metric.name,
                    split=split,
                    value=metric.value,
                    metric_metadata=metadata,
                )
            )
        elif row.value != metric.value or row.metric_metadata != metadata:
            raise ValueError("persisted metric identity was reused with different value/metadata")

    for failure in result.failures:
        prediction_id = prediction_id_by_incident[failure.incident_id]
        code = failure.annotation.failure_code.value
        annotation_id = _stable_id("failure", prediction_id, code)
        metadata = dict(failure.annotation.metadata)
        metadata["source"] = failure.annotation.source.value
        metadata["evaluation_result_hash"] = result.result_hash
        row = session.get(FailureAnnotationRow, annotation_id)
        if row is None:
            session.add(
                FailureAnnotationRow(
                    failure_annotation_id=annotation_id,
                    prediction_id=prediction_id,
                    taxonomy_version=result.failure_taxonomy_version,
                    failure_code=code,
                    notes=failure.annotation.notes,
                    annotation_metadata=metadata,
                )
            )
        elif (
            row.failure_code != code
            or row.notes != failure.annotation.notes
            or row.annotation_metadata != metadata
        ):
            raise ValueError("persisted failure identity was reused with different annotation")
    session.flush()
    return reload_evaluation(session, run_id=run_id)


def reload_evaluation(session: Session, *, run_id: str) -> PersistedEvaluationSnapshot:
    predictions = session.scalars(
        select(PredictionRow).where(PredictionRow.run_id == run_id).order_by(PredictionRow.incident_id)
    ).all()
    metrics = session.scalars(
        select(MetricRow).where(MetricRow.run_id == run_id).order_by(MetricRow.name)
    ).all()
    if not predictions or not metrics:
        raise ValueError("persisted evaluation is incomplete")
    hashes = {
        str(row.metric_metadata.get("evaluation_result_hash"))
        for row in metrics
        if row.metric_metadata.get("evaluation_result_hash")
    }
    if len(hashes) != 1:
        raise ValueError("persisted metrics disagree on evaluation result identity")
    failures = session.scalars(
        select(FailureAnnotationRow)
        .join(PredictionRow, FailureAnnotationRow.prediction_id == PredictionRow.prediction_id)
        .where(PredictionRow.run_id == run_id)
        .order_by(PredictionRow.incident_id, FailureAnnotationRow.failure_code)
    ).all()
    incident_by_prediction = {row.prediction_id: row.incident_id for row in predictions}
    return PersistedEvaluationSnapshot(
        result_hash=next(iter(hashes)),
        prediction_payloads=tuple(dict(row.output_json) for row in predictions),
        metric_values={row.name: row.value for row in metrics},
        failure_codes=tuple(
            (incident_by_prediction[row.prediction_id], row.failure_code) for row in failures
        ),
    )


def load_canonical_predictions(session: Session, *, run_id: str) -> tuple[Prediction, ...]:
    rows = session.scalars(
        select(PredictionRow).where(PredictionRow.run_id == run_id).order_by(PredictionRow.incident_id)
    ).all()
    if not rows:
        raise ValueError(f"no stored predictions for run: {run_id}")
    return tuple(Prediction.model_validate(row.output_json) for row in rows)


def score_stored_run(
    session: Session,
    *,
    run_id: str,
    harness: EvaluationHarness,
    relevant_chunk_ids_by_incident: Mapping[str, Sequence[str]] | None = None,
    upfront_cost_usd: float = 0.0,
) -> EvaluationResult:
    predictions = load_canonical_predictions(session, run_id=run_id)
    incident_ids = [prediction.incident_id for prediction in predictions]
    rows = session.scalars(
        select(Incident).where(Incident.incident_id.in_(incident_ids)).order_by(Incident.incident_id)
    ).all()
    by_id = {row.incident_id: row for row in rows}
    if set(by_id) != set(incident_ids):
        raise ValueError("stored predictions reference missing incidents")
    references = relevant_chunk_ids_by_incident or {}
    examples = [
        EvaluationExample(
            incident_id=incident_id,
            split=by_id[incident_id].split,
            root_cause_code=by_id[incident_id].root_cause_code,
            root_cause_category=by_id[incident_id].root_cause_category,
            relevant_chunk_ids=tuple(references.get(incident_id, ())),
        )
        for incident_id in sorted(incident_ids)
    ]
    return harness.evaluate(
        examples,
        predictions,
        upfront_cost_usd=upfront_cost_usd,
    )
