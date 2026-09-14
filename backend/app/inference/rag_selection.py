from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.inference.rag_ablations import RAGAblationSuite

RAG_SELECTION_POLICY_VERSION = "phase8-validation-selection-v1"
_REQUIRED_METRICS = (
    "primary.exact_accuracy",
    "supporting.hierarchical_accuracy",
    "retrieval.context_recall",
    "calibration.ece",
    "latency.p95_ms",
)


@dataclass(frozen=True)
class RAGValidationCandidate:
    variant_id: str
    split: str
    metric_values: Mapping[str, float]
    result_hash: str


@dataclass(frozen=True)
class RAGSelectionResult:
    policy_version: str
    selected_variant_id: str
    ranked_variant_ids: tuple[str, ...]
    ranking_rows: tuple[dict[str, object], ...]


def _selection_key(candidate: RAGValidationCandidate) -> tuple[float, float, float, float, float, str]:
    metrics = candidate.metric_values
    return (
        -float(metrics["primary.exact_accuracy"]),
        -float(metrics["supporting.hierarchical_accuracy"]),
        -float(metrics["retrieval.context_recall"]),
        float(metrics["calibration.ece"]),
        float(metrics["latency.p95_ms"]),
        candidate.variant_id,
    )


def select_primary_rag_variant(
    suite: RAGAblationSuite,
    candidates: Sequence[RAGValidationCandidate],
) -> RAGSelectionResult:
    """Choose the single frozen RAG config using validation evidence only.

    Ordering is predeclared and deterministic: exact accuracy, hierarchical accuracy, retrieval
    recall, lower ECE, lower p95 latency, then lexical variant id. Test evidence is rejected.
    """
    by_id = {candidate.variant_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Phase 08 validation selection requires unique variant IDs")
    expected_ids = {variant.variant_id for variant in suite.variants}
    if set(by_id) != expected_ids:
        raise ValueError("Phase 08 validation selection requires evidence for every suite variant")

    for candidate in candidates:
        if candidate.split != suite.selection_split or candidate.split != "validation":
            raise ValueError("Phase 08 primary RAG selection may use validation evidence only")
        missing = [name for name in _REQUIRED_METRICS if name not in candidate.metric_values]
        if missing:
            raise ValueError(
                f"Phase 08 validation candidate {candidate.variant_id} is missing metrics: {missing}"
            )
        if not candidate.result_hash:
            raise ValueError("Phase 08 validation selection requires sealed result hashes")

    ranked = tuple(sorted(candidates, key=_selection_key))
    rows = tuple(
        {
            "rank": rank,
            "variant_id": candidate.variant_id,
            "result_hash": candidate.result_hash,
            "primary_exact_accuracy": float(candidate.metric_values["primary.exact_accuracy"]),
            "hierarchical_accuracy": float(
                candidate.metric_values["supporting.hierarchical_accuracy"]
            ),
            "retrieval_context_recall": float(
                candidate.metric_values["retrieval.context_recall"]
            ),
            "ece": float(candidate.metric_values["calibration.ece"]),
            "p95_latency_ms": float(candidate.metric_values["latency.p95_ms"]),
        }
        for rank, candidate in enumerate(ranked, start=1)
    )
    return RAGSelectionResult(
        policy_version=RAG_SELECTION_POLICY_VERSION,
        selected_variant_id=ranked[0].variant_id,
        ranked_variant_ids=tuple(candidate.variant_id for candidate in ranked),
        ranking_rows=rows,
    )
