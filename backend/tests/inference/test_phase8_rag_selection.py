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
    ece: float | None = 0.1,
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
            **({"calibration.ece": ece} if ece is not None else {}),
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


def test_selection_ranks_undefined_ece_after_defined_ece_on_tie() -> None:
    _protocol, suite = load_rag_protocol(ROOT)
    candidates = [_candidate(variant.variant_id) for variant in suite.variants]
    missing_id = suite.variants[0].variant_id
    defined_id = suite.variants[1].variant_id
    candidates = [
        _candidate(
            candidate.variant_id,
            ece=None if candidate.variant_id == missing_id else 0.1,
        )
        for candidate in candidates
    ]

    result = select_primary_rag_variant(suite, candidates)

    assert result.ranked_variant_ids.index(defined_id) < result.ranked_variant_ids.index(missing_id)
    row = next(item for item in result.ranking_rows if item["variant_id"] == missing_id)
    assert row["ece"] is None
    assert row["ece_defined"] is False


def test_undefined_ece_does_not_override_higher_priority_accuracy() -> None:
    _protocol, suite = load_rag_protocol(ROOT)
    preferred = suite.variants[0].variant_id
    candidates = [
        _candidate(
            variant.variant_id,
            exact=0.9 if variant.variant_id == preferred else 0.5,
            ece=None if variant.variant_id == preferred else 0.01,
        )
        for variant in suite.variants
    ]

    result = select_primary_rag_variant(suite, candidates)

    assert result.selected_variant_id == preferred
