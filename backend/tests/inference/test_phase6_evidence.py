from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.inference.evidence import (
    RUN_EVIDENCE_VERSION,
    ValidationReview,
    expected_split_incident_ids,
    seal_run_evidence,
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
