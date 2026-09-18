from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.evaluation.contracts import ParseStatus
from app.inference.evidence import canonical_json_sha256, expected_split_incident_ids
from app.inference.finetuned_protocol import Phase10Protocol, Phase10State

PORTABLE_FINETUNED_EVIDENCE_VERSION = "phase10-finetuned-portable-run-v1"


def _predictions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("predictions")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("Phase 10 portable evidence predictions are malformed")
    return [dict(item) for item in raw]


def _metric_values(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = payload.get("metric_values")
    if not isinstance(raw, Mapping):
        raise ValueError("Phase 10 portable evidence metric_values must be an object")
    return raw


def portable_finetuned_metric(payload: Mapping[str, Any], name: str) -> float:
    raw = _metric_values(payload).get(name)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"Phase 10 portable evidence is missing numeric metric: {name}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Phase 10 portable evidence metric is non-finite: {name}")
    return value


def validate_portable_finetuned_evidence(
    payload: Mapping[str, Any],
    *,
    root: Path,
    protocol: Phase10Protocol,
    expected_split: str,
    expected_run_id: str | None = None,
    expected_source_commit: str | None = None,
) -> None:
    if expected_split not in {"validation", "test"}:
        raise ValueError("Phase 10 portable evidence split must be validation or test")
    if payload.get("evidence_version") != PORTABLE_FINETUNED_EVIDENCE_VERSION:
        raise ValueError("unsupported Phase 10 portable evidence version")
    if payload.get("status") != "completed":
        raise ValueError("Phase 10 portable evidence must describe a completed run")
    if payload.get("split") != expected_split:
        raise ValueError("Phase 10 portable evidence split does not match the requested gate")
    if expected_run_id is not None and payload.get("run_id") != expected_run_id:
        raise ValueError("Phase 10 portable evidence run_id disagrees with the expected run")
    if expected_source_commit is not None and payload.get("git_commit") != expected_source_commit:
        raise ValueError("Phase 10 portable evidence source commit disagrees with the frozen run")

    expected_identity = {
        "protocol_version": protocol.protocol_version,
        "scientific_config_hash": protocol.scientific_config_hash(),
        "dataset_version": protocol.dataset_version,
        "dataset_manifest_checksum": protocol.test_split_manifest_checksum,
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "adapter_revision": protocol.candidate_adapter_sha256,
        "adapter_sha256": protocol.candidate_adapter_sha256,
        "source_training_run_id": protocol.source_training_run_id,
        "source_training_evidence_sha256": protocol.source_training_evidence_sha256,
        "source_training_config_hash": protocol.source_training_config_hash,
        "candidate_selection_rule": protocol.candidate_selection_rule,
        "evaluator_version": protocol.evaluator_version,
        "output_schema_version": protocol.output_schema_version,
        "prompt_version": protocol.prompt_version,
        "generation_config": protocol.generation_config.model_dump(mode="json"),
        "execution_mode": "portable_connected_gpu_no_database",
        "persistence_stack_used": False,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"Phase 10 portable evidence identity mismatch: {key}")

    if expected_split == "validation":
        if payload.get("protocol_state") != Phase10State.VALIDATION.value:
            raise ValueError("Phase 10 validation evidence must predate protocol freeze")
        if payload.get("locked_test_authorized") is not False:
            raise ValueError("Phase 10 validation evidence cannot authorize the locked test")
    else:
        if protocol.state is not Phase10State.FROZEN or not protocol.locked_test_authorized:
            raise ValueError("Phase 10 locked-test evidence requires a frozen authorized protocol")
        if payload.get("protocol_state") != Phase10State.FROZEN.value:
            raise ValueError("Phase 10 locked-test evidence must record frozen protocol state")
        if payload.get("locked_test_authorized") is not True:
            raise ValueError("Phase 10 locked-test evidence must record authorization")

    expected_ids = expected_split_incident_ids(
        root,
        dataset_version=protocol.dataset_version,
        split=expected_split,
    )
    predictions = _predictions(payload)
    incident_ids: list[str] = []
    for prediction in predictions:
        incident_id = prediction.get("incident_id")
        if not isinstance(incident_id, str) or not incident_id:
            raise ValueError("Phase 10 portable prediction is missing incident_id")
        incident_ids.append(incident_id)
        metadata = prediction.get("pipeline_metadata")
        if not isinstance(metadata, Mapping) or metadata.get("pipeline_type") != "FINETUNED":
            raise ValueError("Phase 10 portable evidence contains a non-FINETUNED prediction")
        retrieved = prediction.get("retrieved_chunk_ids")
        if retrieved != []:
            raise ValueError("Phase 10 FINETUNED arm must not contain retrieval context")
        status = prediction.get("parse_status")
        if status not in {item.value for item in ParseStatus}:
            raise ValueError("Phase 10 portable prediction has an invalid parse_status")

    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("Phase 10 portable evidence contains duplicated predictions")
    if tuple(sorted(incident_ids)) != expected_ids:
        raise ValueError("Phase 10 portable evidence has missing or unexpected predictions")
    if payload.get("prediction_count") != len(predictions):
        raise ValueError("Phase 10 portable evidence prediction_count is incorrect")

    failure_count = payload.get("inference_failure_count")
    if isinstance(failure_count, bool) or not isinstance(failure_count, int) or failure_count < 0:
        raise ValueError("Phase 10 portable evidence inference_failure_count is invalid")

    gpu = payload.get("gpu_environment")
    if not isinstance(gpu, Mapping):
        raise ValueError("Phase 10 portable evidence is missing GPU environment")
    if gpu.get("cuda_available") is not True or gpu.get("visible_gpu_count") != 1:
        raise ValueError("Phase 10 portable evidence requires exactly one visible CUDA GPU")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Phase 10 portable evidence is missing canonical evaluation payload")
    result_hash = payload.get("result_hash")
    if not isinstance(result_hash, str) or len(result_hash) != 64:
        raise ValueError("Phase 10 portable evidence is missing result_hash")
    if canonical_json_sha256(evaluation) != result_hash:
        raise ValueError("Phase 10 portable evidence result_hash does not match evaluation payload")

    required_metrics = (
        "primary.exact_accuracy",
        "supporting.hierarchical_accuracy",
        "supporting.top3_accuracy",
        "quality.parse_failure_rate",
        "calibration.normalized_multiclass_brier",
        "latency.p50_ms",
        "latency.p95_ms",
        "cost.marginal_mean_usd",
        "cost.amortized_mean_usd",
    )
    for name in required_metrics:
        portable_finetuned_metric(payload, name)
