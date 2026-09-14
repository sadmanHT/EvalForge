from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.inference.evidence import canonical_json_sha256
from app.inference.rag_ablations import RAGAblationSuite
from app.inference.rag_evidence import validate_phase8_rag_run_evidence
from app.inference.rag_protocol import RAGProtocol
from app.inference.rag_selection import (
    RAGValidationCandidate,
    select_primary_rag_variant,
)

RAG_ABLATION_REPORT_VERSION = "phase8-rag-ablation-report-v1"


def _metric_values(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("metrics")
    if not isinstance(raw, Mapping):
        raise ValueError("Phase 08 validation evidence metrics must be an object")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Phase 08 validation metric is non-numeric: {name}")
        metrics[str(name)] = float(value)
    return metrics


def _candidate(payload: Mapping[str, Any]) -> RAGValidationCandidate:
    variant_id = payload.get("variant_id")
    result_hash = payload.get("result_hash")
    split = payload.get("split")
    if not isinstance(variant_id, str) or not variant_id:
        raise ValueError("Phase 08 validation evidence is missing variant_id")
    if not isinstance(result_hash, str) or not result_hash:
        raise ValueError("Phase 08 validation evidence is missing result_hash")
    if not isinstance(split, str):
        raise ValueError("Phase 08 validation evidence is missing split")
    return RAGValidationCandidate(
        variant_id=variant_id,
        split=split,
        metric_values=_metric_values(payload),
        result_hash=result_hash,
    )


def _seal_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(payload)
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def build_rag_ablation_report(
    *,
    root: Path,
    protocol: RAGProtocol,
    suite: RAGAblationSuite,
    evidence_by_variant: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = tuple(variant.variant_id for variant in suite.variants)
    if set(evidence_by_variant) != set(expected_ids):
        raise ValueError("Phase 08 ablation report requires evidence for every registered variant")

    candidates: list[RAGValidationCandidate] = []
    evidence_rows: list[dict[str, Any]] = []
    by_variant = {variant.variant_id: variant for variant in suite.variants}
    for variant_id in expected_ids:
        evidence = evidence_by_variant[variant_id]
        validate_phase8_rag_run_evidence(
            evidence,
            root=root,
            protocol=protocol,
            expected_split="validation",
            expected_variant=by_variant[variant_id],
        )
        candidate = _candidate(evidence)
        candidates.append(candidate)
        evidence_sha = evidence.get("evidence_sha256")
        run_id = evidence.get("run_id")
        experiment_id = evidence.get("experiment_id")
        config_hash = evidence.get("experiment_config_hash")
        for name, value in (
            ("evidence_sha256", evidence_sha),
            ("run_id", run_id),
            ("experiment_id", experiment_id),
            ("experiment_config_hash", config_hash),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Phase 08 validation evidence is missing {name}")
        evidence_rows.append(
            {
                "variant_id": variant_id,
                "evidence_sha256": evidence_sha,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "experiment_config_hash": config_hash,
                "result_hash": candidate.result_hash,
                "metrics": dict(sorted(candidate.metric_values.items())),
            }
        )

    selection = select_primary_rag_variant(suite, candidates)
    report = {
        "report_version": RAG_ABLATION_REPORT_VERSION,
        "protocol_version": protocol.protocol_version,
        "scientific_config_hash": protocol.scientific_config_hash(),
        "ablation_suite_hash": protocol.ablation_suite_hash,
        "selection_policy_version": selection.policy_version,
        "selection_split": suite.selection_split,
        "baseline_variant_id": suite.baseline_variant_id,
        "selected_variant_id": selection.selected_variant_id,
        "ranked_variant_ids": list(selection.ranked_variant_ids),
        "ranking_rows": list(selection.ranking_rows),
        "validation_evidence": evidence_rows,
    }
    return _seal_report(report)


def validate_rag_ablation_report(
    report: Mapping[str, Any],
    *,
    root: Path,
    protocol: RAGProtocol,
    suite: RAGAblationSuite,
    evidence_by_variant: Mapping[str, Mapping[str, Any]],
) -> None:
    if report.get("report_version") != RAG_ABLATION_REPORT_VERSION:
        raise ValueError("unsupported Phase 08 ablation report version")
    unsealed = dict(report)
    reported_sha = unsealed.pop("report_sha256", None)
    if not isinstance(reported_sha, str) or reported_sha != canonical_json_sha256(unsealed):
        raise ValueError("Phase 08 ablation report seal is invalid")

    expected = build_rag_ablation_report(
        root=root,
        protocol=protocol,
        suite=suite,
        evidence_by_variant=evidence_by_variant,
    )
    if dict(report) != expected:
        raise ValueError("Phase 08 ablation report disagrees with sealed validation evidence")
