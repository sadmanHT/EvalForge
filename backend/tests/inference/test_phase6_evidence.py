from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.inference.evidence import (
    RUN_EVIDENCE_VERSION,
    ValidationReview,
    expected_split_incident_ids,
    seal_run_evidence,
    validate_real_run_operational_evidence,
    validate_run_evidence,
    validate_validation_review,
)
from app.inference.protocol import load_baseline_protocol

ROOT = Path(__file__).resolve().parents[3]


def _validation_evidence() -> dict[str, object]:
    protocol = load_baseline_protocol(ROOT)
    incident_ids = expected_split_incident_ids(
        ROOT,
        dataset_version=protocol.dataset_version,
        split="validation",
    )
    payload: dict[str, object] = {
        "evidence_version": RUN_EVIDENCE_VERSION,
        "split": "validation",
        "protocol_version": protocol.protocol_version,
        "scientific_config_hash": protocol.scientific_config_hash(),
        "dataset_version": protocol.dataset_version,
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "prompt_version": protocol.prompt_version,
        "output_schema_version": protocol.output_schema_version,
        "experiment_status": "completed",
        "run_status": "completed",
        "metric_recomputation_verified": True,
        "expected_incident_ids": list(incident_ids),
        "prediction_incident_ids": list(incident_ids),
        "stored_prediction_count": len(incident_ids),
        "run_id": "phase6-real-validation-fixture",
        "predictions": [{"incident_id": incident_id} for incident_id in incident_ids],
    }
    return seal_run_evidence(payload)


def _operational_evidence() -> dict[str, object]:
    metrics = {
        "primary.exact_accuracy": 0.5,
        "supporting.hierarchical_accuracy": 0.5,
        "supporting.top3_accuracy": 0.75,
        "quality.parse_failure_rate": 0.0,
        "calibration.normalized_multiclass_brier": 0.25,
        "calibration.ece": 0.2,
        "latency.p50_ms": 20.0,
        "latency.p95_ms": 30.0,
        "cost.marginal_mean_usd": 0.00001,
        "cost.amortized_mean_usd": 0.00001,
    }
    costs = [
        {
            "cost_record_id": f"cost-{index}",
            "prediction_id": f"prediction-{index}",
            "cost_rate_snapshot_version": "provider-rate-2026-09-12",
            "units": {
                "method": "active_inference_wall_time_x_gpu_hour_rate",
                "gpu_hour_usd": 0.40,
                "latency_ms": 20.0 + index,
            },
            "amount_usd": "0.00001000",
        }
        for index in range(2)
    ]
    return {
        "inference_failure_count": 0,
        "stored_prediction_count": 2,
        "stored_cost_record_count": 2,
        "scientific_config_hash": "a" * 64,
        "experiment_config_hash": "b" * 64,
        "dependency_lock_checksum": "c" * 64,
        "result_hash": "d" * 64,
        "git_commit": "e" * 40,
        "hardware_runtime_descriptor": "phase6-gpu-host-v1 environment=fixture",
        "cost_rate_snapshot_version": "provider-rate-2026-09-12",
        "cost_records": costs,
        "metrics": metrics,
        "tracking": {
            "provider": "wandb",
            "configured": False,
            "run_reference": None,
            "artifact_reference": None,
        },
    }


def test_validation_evidence_requires_exact_incident_coverage_and_checksum() -> None:
    protocol = load_baseline_protocol(ROOT)
    evidence = _validation_evidence()
    validate_run_evidence(
        evidence,
        root=ROOT,
        protocol=protocol,
        expected_split="validation",
    )

    missing_prediction = dict(evidence)
    missing_prediction["predictions"] = list(evidence["predictions"])[1:]
    missing_prediction = seal_run_evidence(missing_prediction)
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_run_evidence(
            missing_prediction,
            root=ROOT,
            protocol=protocol,
            expected_split="validation",
        )

    tampered = dict(evidence)
    tampered["stored_prediction_count"] = 99
    with pytest.raises(ValueError, match="stored prediction count"):
        validate_run_evidence(
            tampered,
            root=ROOT,
            protocol=protocol,
            expected_split="validation",
        )


def test_real_run_operational_evidence_requires_complete_cost_and_tracking_proof() -> None:
    evidence = _operational_evidence()
    validate_real_run_operational_evidence(evidence, require_tracking=False)

    with pytest.raises(ValueError, match="configured W&B"):
        validate_real_run_operational_evidence(evidence, require_tracking=True)

    tracked = dict(evidence)
    tracked["tracking"] = {
        "provider": "wandb",
        "configured": True,
        "run_reference": "https://wandb.ai/example/run",
        "artifact_reference": "wandb-artifact://example",
    }
    validate_real_run_operational_evidence(tracked, require_tracking=True)

    failed = dict(evidence)
    failed["inference_failure_count"] = 1
    with pytest.raises(ValueError, match="technical inference failures"):
        validate_real_run_operational_evidence(failed, require_tracking=False)

    zero_cost = dict(evidence)
    zero_cost_records = [dict(item) for item in evidence["cost_records"]]
    zero_cost_records[0]["amount_usd"] = "0"
    zero_cost["cost_records"] = zero_cost_records
    with pytest.raises(ValueError, match="finite and positive"):
        validate_real_run_operational_evidence(zero_cost, require_tracking=False)


def test_manual_review_must_cover_every_prediction_and_pass_contracts() -> None:
    evidence = _validation_evidence()
    incident_ids = tuple(evidence["prediction_incident_ids"])
    review = ValidationReview(
        run_id=str(evidence["run_id"]),
        scientific_config_hash=str(evidence["scientific_config_hash"]),
        reviewed_incident_ids=incident_ids,
        schema_contract_passed=True,
        label_contract_passed=True,
        no_prompt_or_parser_defect_observed=True,
        reviewer="phase6-test-reviewer",
        reviewed_at=datetime(2026, 9, 12, tzinfo=UTC),
        notes="fixture review for gate behavior only",
    )
    validate_validation_review(review, validation_evidence=evidence)

    rejected = review.model_copy(update={"no_prompt_or_parser_defect_observed": False})
    with pytest.raises(ValueError, match="prompt/parser defect"):
        validate_validation_review(rejected, validation_evidence=evidence)
