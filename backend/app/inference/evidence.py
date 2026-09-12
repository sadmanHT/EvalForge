from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import reload_evaluation, score_stored_run
from app.inference.protocol import BaselineProtocol, ProtocolState, load_taxonomy
from app.models import Experiment, Incident, Run
from app.results import load_run_evidence

RUN_EVIDENCE_VERSION = "phase6-run-evidence-v1"
VALIDATION_REVIEW_VERSION = "phase6-validation-review-v1"
_REAL_REQUIRED_METRICS = (
    "primary.exact_accuracy",
    "supporting.hierarchical_accuracy",
    "supporting.top3_accuracy",
    "quality.parse_failure_rate",
    "calibration.normalized_multiclass_brier",
    "calibration.ece",
    "latency.p50_ms",
    "latency.p95_ms",
    "cost.marginal_mean_usd",
    "cost.amortized_mean_usd",
)
_BOUNDED_METRICS = {
    "primary.exact_accuracy",
    "supporting.hierarchical_accuracy",
    "supporting.top3_accuracy",
    "quality.parse_failure_rate",
    "calibration.normalized_multiclass_brier",
    "calibration.ece",
}
_NON_NEGATIVE_METRICS = {
    "latency.p50_ms",
    "latency.p95_ms",
    "cost.marginal_mean_usd",
    "cost.amortized_mean_usd",
}
_ZERO_DIRECT_COST_RATE_SNAPSHOTS = frozenset(
    {
        "kaggle-free-quota-no-direct-usd-per-gpu-hour",
    }
)


class ValidationReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_version: Literal["phase6-validation-review-v1"] = "phase6-validation-review-v1"
    run_id: str = Field(min_length=1)
    scientific_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_incident_ids: tuple[str, ...]
    schema_contract_passed: bool
    label_contract_passed: bool
    no_prompt_or_parser_defect_observed: bool
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    notes: str = ""


def expected_split_incident_ids(
    root: Path,
    *,
    dataset_version: str,
    split: str,
) -> tuple[str, ...]:
    path = root / "datasets" / "incident_diagnosis" / "processed"
    path = path / dataset_version / "incidents.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"dataset incidents file is unavailable: {path}")
    incident_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if row.get("split") == split:
            incident_id = row.get("incident_id")
            if not isinstance(incident_id, str) or not incident_id:
                raise ValueError("dataset split contains an invalid incident_id")
            incident_ids.append(incident_id)
    if not incident_ids:
        raise ValueError(f"dataset contains no incidents for split: {split}")
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError(f"dataset contains duplicate incident IDs for split: {split}")
    return tuple(sorted(incident_ids))


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def seal_run_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("evidence_sha256", None)
    sealed["evidence_sha256"] = canonical_json_sha256(sealed)
    return sealed


def _prediction_incident_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("run evidence predictions must be a list")
    incident_ids: list[str] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("run evidence predictions must contain objects")
        incident_id = prediction.get("incident_id")
        if not isinstance(incident_id, str) or not incident_id:
            raise ValueError("run evidence prediction is missing incident_id")
        incident_ids.append(incident_id)
    return tuple(incident_ids)


def _tracking_payload(run: Run) -> dict[str, object]:
    raw_tracking = run.runtime_metadata.get("experiment_tracking")
    if isinstance(raw_tracking, dict):
        return {str(key): value for key, value in raw_tracking.items()}
    return {
        "provider": "wandb",
        "configured": False,
        "run_reference": None,
        "artifact_reference": None,
    }


def export_phase6_run_evidence(
    session: Session,
    *,
    root: Path,
    run_id: str,
    protocol: BaselineProtocol,
) -> dict[str, Any]:
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"unknown Phase 06 run: {run_id}")
    experiment = session.get(Experiment, run.experiment_id)
    if experiment is None:
        raise RuntimeError("Phase 06 run references a missing experiment")
    if experiment.pipeline_type != "ZERO_SHOT":
        raise ValueError("Phase 06 run evidence requires a ZERO_SHOT experiment")
    if experiment.status != "completed" or run.status != "completed":
        raise ValueError("Phase 06 run evidence requires a completed experiment and run")

    readback = load_run_evidence(session, run_id=run_id)
    if readback is None:
        raise RuntimeError("completed Phase 06 run has no readback evidence")
    snapshot = reload_evaluation(session, run_id=run_id)
    labels, categories = load_taxonomy(root)
    harness = EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )
    upfront_cost_usd = float(run.runtime_metadata.get("upfront_cost_usd", 0.0))
    rescored = score_stored_run(
        session,
        run_id=run_id,
        harness=harness,
        upfront_cost_usd=upfront_cost_usd,
    )
    if snapshot.result_hash != rescored.result_hash:
        raise RuntimeError("Phase 06 run result hash changed during evidence recomputation")
    if snapshot.metric_values != rescored.metric_map():
        raise RuntimeError("Phase 06 run metrics changed during evidence recomputation")

    predictions = readback["predictions"]
    if not isinstance(predictions, list):
        raise RuntimeError("Phase 06 readback predictions are malformed")
    prediction_ids = _prediction_incident_ids({"predictions": predictions})
    incident_rows = session.scalars(
        select(Incident).where(
            Incident.dataset_version == experiment.dataset_version,
            Incident.incident_id.in_(prediction_ids),
        )
    ).all()
    split_by_incident = {row.incident_id: row.split for row in incident_rows}
    if set(split_by_incident) != set(prediction_ids):
        raise RuntimeError("Phase 06 evidence predictions reference missing dataset incidents")
    splits = {split_by_incident[incident_id] for incident_id in prediction_ids}
    if len(splits) != 1:
        raise RuntimeError("Phase 06 run mixes dataset splits")
    split = next(iter(splits))
    if split not in {"validation", "test"}:
        raise RuntimeError(f"Phase 06 evidence has unsupported split: {split}")
    expected_ids = expected_split_incident_ids(
        root,
        dataset_version=protocol.dataset_version,
        split=split,
    )
    scientific_hash = protocol.scientific_config_hash()
    stored_scientific_hash = run.runtime_metadata.get("phase6_scientific_config_hash")
    if stored_scientific_hash != scientific_hash:
        raise RuntimeError("Phase 06 run scientific identity disagrees with current protocol")

    payload: dict[str, Any] = {
        "evidence_version": RUN_EVIDENCE_VERSION,
        "split": split,
        "protocol_version": protocol.protocol_version,
        "protocol_state_at_export": protocol.state.value,
        "locked_test_authorized_at_export": protocol.locked_test_authorized,
        "scientific_config_hash": scientific_hash,
        "experiment_config_hash": experiment.config_hash,
        "experiment_id": experiment.experiment_id,
        "run_id": run.run_id,
        "experiment_status": experiment.status,
        "run_status": run.status,
        "dataset_version": experiment.dataset_version,
        "test_split_manifest_checksum": experiment.test_split_manifest_checksum,
        "label_taxonomy_version": experiment.label_taxonomy_version,
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "prompt_version": experiment.prompt_version,
        "output_schema_version": experiment.output_schema_version,
        "generation_config": dict(experiment.generation_config),
        "confidence_method": experiment.confidence_method,
        "seed": experiment.seed,
        "evaluator_version": experiment.evaluator_version,
        "git_commit": experiment.git_commit,
        "dependency_lock_checksum": experiment.dependency_lock_checksum,
        "hardware_runtime_descriptor": experiment.hardware_runtime_descriptor,
        "cost_rate_snapshot_version": experiment.cost_rate_snapshot_version,
        "upfront_cost_usd": upfront_cost_usd,
        "expected_incident_ids": list(expected_ids),
        "prediction_incident_ids": list(prediction_ids),
        "stored_prediction_count": len(predictions),
        "stored_cost_record_count": readback["stored_cost_record_count"],
        "inference_failure_count": int(run.runtime_metadata.get("inference_failure_count", 0)),
        "metric_recomputation_verified": True,
        "result_hash": snapshot.result_hash,
        "metrics": snapshot.metric_values,
        "failure_codes": [list(item) for item in snapshot.failure_codes],
        "tracking": _tracking_payload(run),
        "predictions": predictions,
        "cost_records": readback["cost_records"],
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
    return seal_run_evidence(payload)


def validate_run_evidence(
    payload: Mapping[str, Any],
    *,
    root: Path,
    protocol: BaselineProtocol,
    expected_split: str,
) -> None:
    if expected_split not in {"validation", "test"}:
        raise ValueError("Phase 06 evidence split must be validation or test")
    if payload.get("evidence_version") != RUN_EVIDENCE_VERSION:
        raise ValueError("unsupported Phase 06 run evidence version")
    if payload.get("split") != expected_split:
        raise ValueError("Phase 06 run evidence split does not match the requested gate")
    if payload.get("protocol_version") != protocol.protocol_version:
        raise ValueError("Phase 06 run evidence protocol version disagrees with config")
    if payload.get("scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("Phase 06 run evidence scientific hash disagrees with config")
    if payload.get("dataset_version") != protocol.dataset_version:
        raise ValueError("Phase 06 run evidence dataset version disagrees with config")
    if payload.get("base_model_id") != protocol.base_model_id:
        raise ValueError("Phase 06 run evidence base model disagrees with config")
    if payload.get("base_model_revision") != protocol.base_model_revision:
        raise ValueError("Phase 06 run evidence model revision disagrees with config")
    if payload.get("prompt_version") != protocol.prompt_version:
        raise ValueError("Phase 06 run evidence prompt version disagrees with config")
    if payload.get("output_schema_version") != protocol.output_schema_version:
        raise ValueError("Phase 06 run evidence output schema disagrees with config")
    if payload.get("experiment_status") != "completed" or payload.get("run_status") != "completed":
        raise ValueError("Phase 06 run evidence must describe a completed experiment and run")
    if payload.get("metric_recomputation_verified") is not True:
        raise ValueError("Phase 06 run evidence does not prove metric recomputation")
    if expected_split == "test" and (
        protocol.state is not ProtocolState.FROZEN or not protocol.locked_test_authorized
    ):
        raise ValueError("locked-test evidence is invalid before protocol freeze/authorization")

    expected_ids = expected_split_incident_ids(
        root,
        dataset_version=protocol.dataset_version,
        split=expected_split,
    )
    reported_expected = payload.get("expected_incident_ids")
    if reported_expected != list(expected_ids):
        raise ValueError("Phase 06 run evidence expected incident IDs are incorrect")
    prediction_ids = _prediction_incident_ids(payload)
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Phase 06 run evidence contains duplicated predictions")
    if tuple(sorted(prediction_ids)) != expected_ids:
        raise ValueError("Phase 06 run evidence has missing or unexpected predictions")
    if payload.get("prediction_incident_ids") != list(prediction_ids):
        raise ValueError("Phase 06 run evidence prediction ID index disagrees with predictions")
    if payload.get("stored_prediction_count") != len(prediction_ids):
        raise ValueError("Phase 06 run evidence stored prediction count is incorrect")

    evidence_sha = payload.get("evidence_sha256")
    if not isinstance(evidence_sha, str):
        raise ValueError("Phase 06 run evidence is missing evidence_sha256")
    unsealed = dict(payload)
    unsealed.pop("evidence_sha256", None)
    if evidence_sha != canonical_json_sha256(unsealed):
        raise ValueError("Phase 06 run evidence checksum mismatch")


def _require_hex_digest(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"real Phase 06 evidence has invalid {field}")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"real Phase 06 evidence has invalid {field}")
    return value


def _require_finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"real Phase 06 evidence metric is missing/non-numeric: {name}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"real Phase 06 evidence metric is non-finite: {name}")
    return converted


def validate_real_run_operational_evidence(
    payload: Mapping[str, Any],
    *,
    require_tracking: bool,
) -> None:
    failure_count = payload.get("inference_failure_count")
    if isinstance(failure_count, bool) or failure_count != 0:
        raise ValueError("real Phase 06 run has technical inference failures")

    stored_prediction_count = payload.get("stored_prediction_count")
    if (
        isinstance(stored_prediction_count, bool)
        or not isinstance(stored_prediction_count, int)
        or stored_prediction_count <= 0
    ):
        raise ValueError("real Phase 06 run has an invalid stored prediction count")

    _require_hex_digest(payload, "scientific_config_hash")
    _require_hex_digest(payload, "experiment_config_hash")
    _require_hex_digest(payload, "dependency_lock_checksum")
    _require_hex_digest(payload, "result_hash")
    git_commit = payload.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("real Phase 06 run must record an exact 40-hex git commit")
    hardware = payload.get("hardware_runtime_descriptor")
    if not isinstance(hardware, str) or not hardware.startswith("phase6-gpu-host-v1 "):
        raise ValueError("real Phase 06 run is not bound to sealed Phase 06 GPU host evidence")

    rate_snapshot = payload.get("cost_rate_snapshot_version")
    if not isinstance(rate_snapshot, str) or not rate_snapshot.strip():
        raise ValueError("real Phase 06 run is missing a cost-rate snapshot version")
    zero_direct_cost = rate_snapshot in _ZERO_DIRECT_COST_RATE_SNAPSHOTS
    cost_records = payload.get("cost_records")
    if not isinstance(cost_records, list):
        raise ValueError("real Phase 06 run cost records must be a list")
    stored_cost_count = payload.get("stored_cost_record_count")
    if stored_cost_count != len(cost_records) or len(cost_records) != stored_prediction_count:
        raise ValueError("real Phase 06 run must persist exactly one cost record per prediction")

    cost_ids: set[str] = set()
    for cost in cost_records:
        if not isinstance(cost, dict):
            raise ValueError("real Phase 06 run cost records must contain objects")
        cost_id = cost.get("cost_record_id")
        if not isinstance(cost_id, str) or not cost_id or cost_id in cost_ids:
            raise ValueError("real Phase 06 run has missing/duplicated cost record IDs")
        cost_ids.add(cost_id)
        if cost.get("cost_rate_snapshot_version") != rate_snapshot:
            raise ValueError("real Phase 06 cost record uses a different rate snapshot")
        try:
            amount = Decimal(str(cost.get("amount_usd")))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("real Phase 06 cost record amount is invalid") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError("real Phase 06 cost record amount must be finite and non-negative")
        if zero_direct_cost:
            if amount != 0:
                raise ValueError(
                    "real Phase 06 zero-direct-cost snapshot must record zero marginal cost"
                )
        elif amount <= 0:
            raise ValueError("real Phase 06 cost record amount must be finite and positive")
        units = cost.get("units")
        if not isinstance(units, dict):
            raise ValueError("real Phase 06 cost record units are missing")
        if units.get("method") != "active_inference_wall_time_x_gpu_hour_rate":
            raise ValueError("real Phase 06 cost record uses an unexpected cost method")
        hourly_rate = units.get("gpu_hour_usd")
        if isinstance(hourly_rate, bool) or not isinstance(hourly_rate, int | float):
            raise ValueError("real Phase 06 cost record is missing gpu_hour_usd")
        hourly_rate_value = float(hourly_rate)
        if not math.isfinite(hourly_rate_value) or hourly_rate_value < 0:
            raise ValueError("real Phase 06 gpu_hour_usd must be finite and non-negative")
        if zero_direct_cost:
            if hourly_rate_value != 0:
                raise ValueError(
                    "real Phase 06 zero-direct-cost snapshot must record a zero GPU hourly rate"
                )
        elif hourly_rate_value <= 0:
            raise ValueError("real Phase 06 gpu_hour_usd must be finite and positive")
        latency = units.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, int | float):
            raise ValueError("real Phase 06 cost record is missing latency_ms")
        if not math.isfinite(float(latency)) or float(latency) <= 0:
            raise ValueError("real Phase 06 priced inference latency must be finite and positive")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("real Phase 06 evidence metrics must be an object")
    metric_values = {name: _require_finite_metric(metrics, name) for name in _REAL_REQUIRED_METRICS}
    for name in _BOUNDED_METRICS:
        if not 0.0 <= metric_values[name] <= 1.0:
            raise ValueError(f"real Phase 06 evidence metric is outside [0,1]: {name}")
    for name in _NON_NEGATIVE_METRICS:
        if metric_values[name] < 0.0:
            raise ValueError(f"real Phase 06 evidence metric is negative: {name}")
    if zero_direct_cost:
        if metric_values["cost.marginal_mean_usd"] != 0.0:
            raise ValueError(
                "real Phase 06 zero-direct-cost snapshot must report zero marginal mean cost"
            )
    elif metric_values["cost.marginal_mean_usd"] <= 0.0:
        raise ValueError("real Phase 06 marginal cost must be positive")
    if metric_values["cost.amortized_mean_usd"] < metric_values["cost.marginal_mean_usd"]:
        raise ValueError("real Phase 06 amortized cost cannot be below marginal cost")

    tracking = payload.get("tracking")
    if not isinstance(tracking, dict) or tracking.get("provider") != "wandb":
        raise ValueError("real Phase 06 evidence has an invalid tracking payload")
    if require_tracking:
        if tracking.get("configured") is not True:
            raise ValueError("final Phase 06 locked-test evidence requires configured W&B tracking")
        for field in ("run_reference", "artifact_reference"):
            value = tracking.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"final Phase 06 W&B tracking is missing {field}")


def validate_validation_review(
    review: ValidationReview,
    *,
    validation_evidence: Mapping[str, Any],
) -> None:
    if validation_evidence.get("split") != "validation":
        raise ValueError("manual validation review must reference validation evidence")
    if review.run_id != validation_evidence.get("run_id"):
        raise ValueError("manual validation review run_id disagrees with evidence")
    if review.scientific_config_hash != validation_evidence.get("scientific_config_hash"):
        raise ValueError("manual validation review scientific hash disagrees with evidence")
    expected_ids = tuple(validation_evidence.get("prediction_incident_ids", ()))
    if tuple(sorted(review.reviewed_incident_ids)) != tuple(sorted(expected_ids)):
        raise ValueError("manual validation review must cover every validation prediction")
    if len(review.reviewed_incident_ids) != len(set(review.reviewed_incident_ids)):
        raise ValueError("manual validation review contains duplicated incident IDs")
    if not review.schema_contract_passed:
        raise ValueError("manual validation review found a schema-contract defect")
    if not review.label_contract_passed:
        raise ValueError("manual validation review found a label-contract defect")
    if not review.no_prompt_or_parser_defect_observed:
        raise ValueError("manual validation review found a prompt/parser defect")
