from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.failure_taxonomy import (
    AnnotationSource,
    FailureAnnotation,
    FailureCode,
    classify_failures,
    manual_annotation,
)


CATEGORIES = {"a": "database", "b": "database", "c": "resource"}


def test_rule_based_failure_categories_are_deterministic() -> None:
    example = EvaluationExample(
        incident_id="f1",
        split="test",
        root_cause_code="b",
        root_cause_category="database",
        relevant_chunk_ids=("needed",),
    )
    prediction = Prediction(
        incident_id="f1",
        predicted_root_cause_code="a",
        confidence_probability=0.9,
        confidence_source="normalized_label_sequence_log_likelihood",
        evidence_citations=("invented",),
        retrieved_chunk_ids=("wrong",),
    )
    codes = {
        annotation.failure_code
        for annotation in classify_failures(example, prediction, label_to_category=CATEGORIES)
    }
    assert codes == {
        FailureCode.HALLUCINATED_EVIDENCE,
        FailureCode.CORRECT_CATEGORY_WRONG_CAUSE,
        FailureCode.OVERCONFIDENT_WRONG,
        FailureCode.RETRIEVAL_MISS,
    }


def test_underconfident_correct_rule() -> None:
    example = EvaluationExample(
        incident_id="f2",
        split="test",
        root_cause_code="c",
        root_cause_category="resource",
    )
    prediction = Prediction(
        incident_id="f2",
        predicted_root_cause_code="c",
        confidence_probability=0.4,
        confidence_source="normalized_label_sequence_log_likelihood",
    )
    annotations = classify_failures(example, prediction, label_to_category=CATEGORIES)
    assert [item.failure_code for item in annotations] == [FailureCode.UNDERCONFIDENT_CORRECT]


def test_anchoring_and_insufficient_context_require_manual_or_secondary_annotation() -> None:
    with pytest.raises(ValidationError, match="not assigned by deterministic rules"):
        FailureAnnotation(
            failure_code=FailureCode.ANCHORING,
            source=AnnotationSource.RULE,
            notes="invalid automatic claim",
        )
    annotation = manual_annotation(
        FailureCode.ANCHORING,
        notes="reviewer observed copying of an early misleading clue",
    )
    assert annotation.source is AnnotationSource.MANUAL
    with pytest.raises(ValidationError, match="require explanatory notes"):
        FailureAnnotation(
            failure_code=FailureCode.INSUFFICIENT_CONTEXT,
            source=AnnotationSource.SECONDARY,
        )
