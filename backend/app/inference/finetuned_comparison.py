from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.evaluation.statistics import compare_paired_accuracy
from app.inference.evidence import canonical_json_sha256, validate_run_evidence
from app.inference.finetuned_evidence import (
    portable_finetuned_metric,
    validate_portable_finetuned_evidence,
)
from app.inference.finetuned_protocol import Phase10Protocol
from app.inference.protocol import BaselineProtocol

FINETUNED_COMPARISON_VERSION = "phase10-baseline-finetuned-comparison-v1"


def _predictions_by_incident(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("predictions")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("comparison evidence predictions are malformed")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        incident_id = item.get("incident_id")
        if not isinstance(incident_id, str) or not incident_id or incident_id in result:
            raise ValueError("comparison evidence has missing or duplicate incident IDs")
        result[incident_id] = dict(item)
    return result


def _ground_truth(
    root: Path,
    *,
    dataset_version: str,
    incident_ids: set[str],
) -> dict[str, str]:
    path = root / "datasets/incident_diagnosis/processed" / dataset_version / "incidents.jsonl"
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        incident_id = row.get("incident_id")
        if incident_id in incident_ids:
            label = row.get("root_cause_code")
            if not isinstance(label, str) or not label:
                raise ValueError("dataset comparison row has an invalid root cause label")
            labels[str(incident_id)] = label
    if set(labels) != incident_ids:
        raise ValueError("comparison evidence references incidents absent from the dataset")
    return labels


def _baseline_metric(payload: Mapping[str, Any], name: str) -> float:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("baseline comparison evidence metrics must be an object")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"baseline comparison evidence is missing numeric metric: {name}")
    return float(value)


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("comparison_sha256", None)
    sealed["comparison_sha256"] = canonical_json_sha256(sealed)
    return sealed


def build_paired_baseline_finetuned_comparison(
    *,
    root: Path,
    baseline_protocol: BaselineProtocol,
    finetuned_protocol: Phase10Protocol,
    baseline_evidence: Mapping[str, Any],
    finetuned_evidence: Mapping[str, Any],
    finetuned_evidence_file_sha256: str,
    expected_finetuned_source_commit: str,
    resamples: int = 10_000,
) -> dict[str, Any]:
    split = "test"
    validate_run_evidence(
        baseline_evidence,
        root=root,
        protocol=baseline_protocol,
        expected_split=split,
    )
    validate_portable_finetuned_evidence(
        finetuned_evidence,
        root=root,
        protocol=finetuned_protocol,
        expected_split=split,
        expected_run_id="phase10-finetuned-test-v1",
        expected_source_commit=expected_finetuned_source_commit,
    )
    if len(finetuned_evidence_file_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in finetuned_evidence_file_sha256
    ):
        raise ValueError("fine-tuned evidence file SHA-256 must be a lowercase hex digest")

    baseline_predictions = _predictions_by_incident(baseline_evidence)
    finetuned_predictions = _predictions_by_incident(finetuned_evidence)
    if set(baseline_predictions) != set(finetuned_predictions):
        raise ValueError(
            "baseline and fine-tuned evidence must cover exactly the same incident IDs"
        )
    incident_ids = set(baseline_predictions)
    truth = _ground_truth(
        root,
        dataset_version=finetuned_protocol.dataset_version,
        incident_ids=incident_ids,
    )

    baseline_outcomes: dict[str, bool] = {}
    finetuned_outcomes: dict[str, bool] = {}
    paired_rows: list[dict[str, Any]] = []
    for incident_id in sorted(incident_ids):
        baseline_prediction = baseline_predictions[incident_id]
        finetuned_prediction = finetuned_predictions[incident_id]
        baseline_label = baseline_prediction.get("predicted_root_cause_code")
        finetuned_label = finetuned_prediction.get("predicted_root_cause_code")
        expected = truth[incident_id]
        baseline_correct = (
            baseline_prediction.get("parse_status") == "OK" and baseline_label == expected
        )
        finetuned_correct = (
            finetuned_prediction.get("parse_status") == "OK" and finetuned_label == expected
        )
        baseline_outcomes[incident_id] = baseline_correct
        finetuned_outcomes[incident_id] = finetuned_correct
        paired_rows.append(
            {
                "incident_id": incident_id,
                "ground_truth": expected,
                "baseline_prediction": baseline_label,
                "finetuned_prediction": finetuned_label,
                "baseline_parse_status": baseline_prediction.get("parse_status"),
                "finetuned_parse_status": finetuned_prediction.get("parse_status"),
                "baseline_correct": baseline_correct,
                "finetuned_correct": finetuned_correct,
            }
        )

    statistics = compare_paired_accuracy(
        baseline_outcomes,
        finetuned_outcomes,
        resamples=resamples,
        confidence_level=0.95,
        seed=finetuned_protocol.seed,
    )
    baseline_accuracy = _baseline_metric(baseline_evidence, "primary.exact_accuracy")
    finetuned_accuracy = portable_finetuned_metric(
        finetuned_evidence,
        "primary.exact_accuracy",
    )
    observed = statistics.bootstrap.observed_difference
    if abs((finetuned_accuracy - baseline_accuracy) - observed) > 1e-12:
        raise ValueError("stored accuracy metrics disagree with paired prediction outcomes")

    comparison = {
        "comparison_version": FINETUNED_COMPARISON_VERSION,
        "split": split,
        "dataset_version": finetuned_protocol.dataset_version,
        "incident_ids": sorted(incident_ids),
        "baseline": {
            "run_id": baseline_evidence.get("run_id"),
            "evidence_sha256": baseline_evidence.get("evidence_sha256"),
            "result_hash": baseline_evidence.get("result_hash"),
            "exact_accuracy": baseline_accuracy,
            "parse_failure_rate": _baseline_metric(
                baseline_evidence,
                "quality.parse_failure_rate",
            ),
            "normalized_multiclass_brier": _baseline_metric(
                baseline_evidence,
                "calibration.normalized_multiclass_brier",
            ),
            "latency_p50_ms": _baseline_metric(baseline_evidence, "latency.p50_ms"),
            "latency_p95_ms": _baseline_metric(baseline_evidence, "latency.p95_ms"),
            "marginal_mean_cost_usd": _baseline_metric(
                baseline_evidence,
                "cost.marginal_mean_usd",
            ),
        },
        "finetuned": {
            "run_id": finetuned_evidence.get("run_id"),
            "evidence_file_sha256": finetuned_evidence_file_sha256,
            "result_hash": finetuned_evidence.get("result_hash"),
            "source_commit": finetuned_evidence.get("git_commit"),
            "scientific_config_hash": finetuned_protocol.scientific_config_hash(),
            "adapter_sha256": finetuned_protocol.candidate_adapter_sha256,
            "exact_accuracy": finetuned_accuracy,
            "parse_failure_rate": portable_finetuned_metric(
                finetuned_evidence,
                "quality.parse_failure_rate",
            ),
            "normalized_multiclass_brier": portable_finetuned_metric(
                finetuned_evidence,
                "calibration.normalized_multiclass_brier",
            ),
            "latency_p50_ms": portable_finetuned_metric(
                finetuned_evidence,
                "latency.p50_ms",
            ),
            "latency_p95_ms": portable_finetuned_metric(
                finetuned_evidence,
                "latency.p95_ms",
            ),
            "marginal_mean_cost_usd": portable_finetuned_metric(
                finetuned_evidence,
                "cost.marginal_mean_usd",
            ),
        },
        "paired_accuracy": {
            "finetuned_minus_baseline": observed,
            "bootstrap_ci_lower": statistics.bootstrap.ci_lower,
            "bootstrap_ci_upper": statistics.bootstrap.ci_upper,
            "confidence_level": statistics.bootstrap.confidence_level,
            "bootstrap_resamples": statistics.bootstrap.resamples,
            "bootstrap_seed": statistics.bootstrap.seed,
            "mcnemar_baseline_only_correct": statistics.mcnemar.a_only_correct,
            "mcnemar_finetuned_only_correct": statistics.mcnemar.b_only_correct,
            "mcnemar_discordant_pairs": statistics.mcnemar.discordant_pairs,
            "mcnemar_p_value": statistics.mcnemar.p_value,
        },
        "paired_rows": paired_rows,
        "interpretation_note": (
            "Descriptive paired result on six locked test incidents only; "
            "the test outcome is not permitted to drive model or protocol selection."
        ),
    }
    return _seal(comparison)


def validate_paired_baseline_finetuned_comparison(
    comparison: Mapping[str, Any],
    *,
    root: Path,
    baseline_protocol: BaselineProtocol,
    finetuned_protocol: Phase10Protocol,
    baseline_evidence: Mapping[str, Any],
    finetuned_evidence: Mapping[str, Any],
    finetuned_evidence_file_sha256: str,
    expected_finetuned_source_commit: str,
) -> None:
    if comparison.get("comparison_version") != FINETUNED_COMPARISON_VERSION:
        raise ValueError("unsupported Phase 10 baseline/fine-tuned comparison version")
    expected = build_paired_baseline_finetuned_comparison(
        root=root,
        baseline_protocol=baseline_protocol,
        finetuned_protocol=finetuned_protocol,
        baseline_evidence=baseline_evidence,
        finetuned_evidence=finetuned_evidence,
        finetuned_evidence_file_sha256=finetuned_evidence_file_sha256,
        expected_finetuned_source_commit=expected_finetuned_source_commit,
        resamples=10_000,
    )
    if dict(comparison) != expected:
        raise ValueError(
            "Phase 10 baseline/fine-tuned comparison disagrees with sealed run evidence"
        )
