from __future__ import annotations

from pathlib import Path

import pytest

from app.inference.rag_protocol import load_rag_protocol
from app.inference.rag_selection import (
    RAG_SELECTION_POLICY_VERSION,
    RAGValidationCandidate,
    select_primary_rag_variant,
)

ROOT = Path(__file__).resolve().parents[3]


def _candidate(
    variant_id: str,
    *,
    exact: float = 0.5,
    hierarchical: float = 0.6,
    recall: float = 0.7,
    ece: float = 0.1,
    p95: float = 100.0,
    split: str = "validation",
) -> RAGValidationCandidate:
    return RAGValidationCandidate(
        variant_id=variant_id,
        split=split,
        metric_values={
            "primary.exact_accuracy": exact,
            "supporting.hierarchical_accuracy": hierarchical,
            "retrieval.context_recall": recall,
            "calibration.ece": ece,
            "latency.p95_ms": p95,
        },
        result_hash=f"result-{variant_id}",
    )


def test_selection_uses_predeclared_validation_ranking() -> None:
    _protocol, suite = load_rag_protocol(ROOT)
    candidates = [_candidate(variant.variant_id) for variant in suite.variants]
    preferred = "rag-top-k3"
    candidates = [
        _candidate(item.variant_id, exact=0.9 if item.variant_id == preferred else 0.5)
        for item in candidates
    ]

    result = select_primary_rag_variant(suite, candidates)

    assert result.policy_version == RAG_SELECTION_POLICY_VERSION
    assert result.selected_variant_id == preferred
    assert result.ranked_variant_ids[0] == preferred
    assert len(result.ranking_rows) == len(suite.variants)


def test_selection_ties_are_deterministic_and_not_order_dependent() -> None:
    _protocol, suite = load_rag_protocol(ROOT)
    candidates = [_candidate(variant.variant_id) for variant in suite.variants]

    forward = select_primary_rag_variant(suite, candidates)
    reverse = select_primary_rag_variant(suite, list(reversed(candidates)))

    assert forward == reverse
    assert forward.selected_variant_id == min(variant.variant_id for variant in suite.variants)


def test_selection_rejects_locked_test_evidence() -> None:
    _protocol, suite = load_rag_protocol(ROOT)
    candidates = [_candidate(variant.variant_id) for variant in suite.variants]
    candidates[0] = _candidate(candidates[0].variant_id, split="test")

    with pytest.raises(ValueError, match="validation evidence only"):
        select_primary_rag_variant(suite, candidates)


def test_selection_requires_every_controlled_ablation_variant() -> None:
    _protocol, suite = load_rag_protocol(ROOT)
    candidates = [_candidate(variant.variant_id) for variant in suite.variants[:-1]]

    with pytest.raises(ValueError, match="every suite variant"):
        select_primary_rag_variant(suite, candidates)
