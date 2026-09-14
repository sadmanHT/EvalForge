from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.evaluation.statistics import compare_paired_accuracy
from app.inference.evidence import canonical_json_sha256, validate_run_evidence
from app.inference.protocol import BaselineProtocol
from app.inference.rag_ablations import RAGVariant
from app.inference.rag_evidence import validate_phase8_rag_run_evidence
from app.inference.rag_protocol import RAGProtocol

RAG_COMPARISON_VERSION = "phase8-baseline-rag-comparison-v1"


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


def _metric(payload: Mapping[str, Any], name: str) -> float:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("comparison evidence metrics must be an object")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"comparison evidence is missing numeric metric: {name}")
    return float(value)


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("comparison_sha256", None)
    sealed["comparison_sha256"] = canonical_json_sha256(sealed)
    return sealed


def build_paired_baseline_rag_comparison(
    *,
    root: Path,
    split: str,
    baseline_protocol: BaselineProtocol,
    rag_protocol: RAGProtocol,
    rag_variant: RAGVariant,
    baseline_evidence: Mapping[str, Any],
    rag_evidence: Mapping[str, Any],
    resamples: int = 10_000,
) -> dict[str, Any]:
    validate_run_evidence(
        baseline_evidence,
        root=root,
        protocol=baseline_protocol,
        expected_split=split,
    )
    validate_phase8_rag_run_evidence(
        rag_evidence,
        root=root,
        protocol=rag_protocol,
        expected_split=split,
        expected_variant=rag_variant,
    )

    baseline_predictions = _predictions_by_incident(baseline_evidence)
    rag_predictions = _predictions_by_incident(rag_evidence)
    if set(baseline_predictions) != set(rag_predictions):
        raise ValueError("baseline and RAG evidence must cover exactly the same incident IDs")
    incident_ids = set(baseline_predictions)
    truth = _ground_truth(
        root,
        dataset_version=baseline_protocol.dataset_version,
        incident_ids=incident_ids,
    )

    baseline_outcomes: dict[str, bool] = {}
    rag_outcomes: dict[str, bool] = {}
    paired_rows: list[dict[str, Any]] = []
    for incident_id in sorted(incident_ids):
        baseline_label = baseline_predictions[incident_id].get("predicted_root_cause_code")
        rag_label = rag_predictions[incident_id].get("predicted_root_cause_code")
        expected = truth[incident_id]
        baseline_correct = baseline_label == expected
        rag_correct = rag_label == expected
        baseline_outcomes[incident_id] = baseline_correct
        rag_outcomes[incident_id] = rag_correct
        paired_rows.append(
            {
                "incident_id": incident_id,
                "ground_truth": expected,
                "baseline_prediction": baseline_label,
                "rag_prediction": rag_label,
                "baseline_correct": baseline_correct,
                "rag_correct": rag_correct,
            }
        )

    statistics = compare_paired_accuracy(
        baseline_outcomes,
        rag_outcomes,
        resamples=resamples,
        confidence_level=0.95,
        seed=rag_protocol.seed,
    )
    baseline_accuracy = _metric(baseline_evidence, "primary.exact_accuracy")
    rag_accuracy = _metric(rag_evidence, "primary.exact_accuracy")
    observed = statistics.bootstrap.observed_difference
    if abs((rag_accuracy - baseline_accuracy) - observed) > 1e-12:
        raise ValueError("stored accuracy metrics disagree with paired prediction outcomes")

    comparison = {
        "comparison_version": RAG_COMPARISON_VERSION,
        "split": split,
        "dataset_version": rag_protocol.dataset_version,
        "incident_ids": sorted(incident_ids),
        "baseline": {
            "run_id": baseline_evidence.get("run_id"),
            "evidence_sha256": baseline_evidence.get("evidence_sha256"),
            "exact_accuracy": baseline_accuracy,
            "latency_p50_ms": _metric(baseline_evidence, "latency.p50_ms"),
            "latency_p95_ms": _metric(baseline_evidence, "latency.p95_ms"),
            "marginal_mean_cost_usd": _metric(
                baseline_evidence,
                "cost.marginal_mean_usd",
            ),
        },
        "rag": {
            "run_id": rag_evidence.get("run_id"),
            "evidence_sha256": rag_evidence.get("evidence_sha256"),
            "variant_id": rag_variant.variant_id,
            "exact_accuracy": rag_accuracy,
            "latency_p50_ms": _metric(rag_evidence, "latency.p50_ms"),
            "latency_p95_ms": _metric(rag_evidence, "latency.p95_ms"),
            "marginal_mean_cost_usd": _metric(rag_evidence, "cost.marginal_mean_usd"),
            "retrieval_context_precision": _metric(
                rag_evidence,
                "retrieval.context_precision",
            ),
            "retrieval_context_recall": _metric(
                rag_evidence,
                "retrieval.context_recall",
            ),
        },
        "paired_accuracy": {
            "rag_minus_baseline": observed,
            "bootstrap_ci_lower": statistics.bootstrap.ci_lower,
            "bootstrap_ci_upper": statistics.bootstrap.ci_upper,
            "confidence_level": statistics.bootstrap.confidence_level,
            "bootstrap_resamples": statistics.bootstrap.resamples,
            "bootstrap_seed": statistics.bootstrap.seed,
            "mcnemar_a_only_correct": statistics.mcnemar.a_only_correct,
            "mcnemar_b_only_correct": statistics.mcnemar.b_only_correct,
            "mcnemar_discordant_pairs": statistics.mcnemar.discordant_pairs,
            "mcnemar_p_value": statistics.mcnemar.p_value,
        },
        "paired_rows": paired_rows,
    }
    return _seal(comparison)


def validate_paired_baseline_rag_comparison(
    comparison: Mapping[str, Any],
    *,
    root: Path,
    split: str,
    baseline_protocol: BaselineProtocol,
    rag_protocol: RAGProtocol,
    rag_variant: RAGVariant,
    baseline_evidence: Mapping[str, Any],
    rag_evidence: Mapping[str, Any],
) -> None:
    if comparison.get("comparison_version") != RAG_COMPARISON_VERSION:
        raise ValueError("unsupported Phase 08 baseline/RAG comparison version")
    expected = build_paired_baseline_rag_comparison(
        root=root,
        split=split,
        baseline_protocol=baseline_protocol,
        rag_protocol=rag_protocol,
        rag_variant=rag_variant,
        baseline_evidence=baseline_evidence,
        rag_evidence=rag_evidence,
        resamples=10_000,
    )
    if dict(comparison) != expected:
        raise ValueError("Phase 08 baseline/RAG comparison disagrees with sealed run evidence")
