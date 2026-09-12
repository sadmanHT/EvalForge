from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
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


class ValidationReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_version: Literal["phase6-validation-review-v1"] = VALIDATION_REVIEW_VERSION
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
    path = (
        root
        / "datasets"
        / "incident_diagnosis"
        / "processed"
        / dataset_version
        / "incidents.jsonl"
    )
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
