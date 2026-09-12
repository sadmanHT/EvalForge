from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CostRecord, Experiment, Metric, Prediction, Run


def load_run_evidence(session: Session, *, run_id: str) -> dict[str, object] | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    experiment = session.get(Experiment, run.experiment_id)
    if experiment is None:
        raise RuntimeError("run references a missing experiment")
    predictions = session.scalars(
        select(Prediction)
        .where(Prediction.run_id == run_id)
        .order_by(Prediction.incident_id)
    ).all()
    metrics = session.scalars(
        select(Metric).where(Metric.run_id == run_id).order_by(Metric.name)
    ).all()
    costs = session.scalars(
        select(CostRecord)
        .where(CostRecord.run_id == run_id)
        .order_by(CostRecord.cost_record_id)
    ).all()
    return {
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "study_id": experiment.study_id,
            "pipeline_type": experiment.pipeline_type,
            "dataset_version": experiment.dataset_version,
            "model_version_id": experiment.model_version_id,
            "prompt_version": experiment.prompt_version,
            "output_schema_version": experiment.output_schema_version,
            "config_hash": experiment.config_hash,
            "config": dict(experiment.config_json),
            "status": experiment.status,
        },
        "run": {
            "run_id": run.run_id,
            "attempt": run.attempt,
            "status": run.status,
            "runtime_metadata": dict(run.runtime_metadata),
        },
        "stored_prediction_count": len(predictions),
        "predictions": [dict(row.output_json) for row in predictions],
        "metrics": {row.name: row.value for row in metrics},
        "stored_cost_record_count": len(costs),
        "cost_records": [
            {
                "cost_record_id": row.cost_record_id,
                "prediction_id": row.prediction_id,
                "cost_rate_snapshot_version": row.cost_rate_snapshot_version,
                "units": dict(row.units_json),
                "amount_usd": format(row.amount_usd, "f"),
            }
            for row in costs
        ],
    }
