from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.contracts import EvaluationExample, ParseStatus, Prediction

FAILURE_TAXONOMY_VERSION = "phase5-failure-taxonomy-v1"


class FailureCode(StrEnum):
    HALLUCINATED_EVIDENCE = "HALLUCINATED_EVIDENCE"
    ANCHORING = "ANCHORING"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    CORRECT_CATEGORY_WRONG_CAUSE = "CORRECT_CATEGORY_WRONG_CAUSE"
    OVERCONFIDENT_WRONG = "OVERCONFIDENT_WRONG"
    UNDERCONFIDENT_CORRECT = "UNDERCONFIDENT_CORRECT"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    OTHER_UNCLASSIFIED = "OTHER_UNCLASSIFIED"


class AnnotationSource(StrEnum):
    RULE = "rule"
    MANUAL = "manual"
    SECONDARY = "secondary"


class FailureAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_code: FailureCode
    source: AnnotationSource
    notes: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_notes(self) -> FailureAnnotation:
        if self.source in {AnnotationSource.MANUAL, AnnotationSource.SECONDARY} and not self.notes:
            raise ValueError("manual/secondary failure annotations require explanatory notes")
        if self.source is AnnotationSource.RULE and self.failure_code in {
            FailureCode.ANCHORING,
            FailureCode.INSUFFICIENT_CONTEXT,
        }:
            raise ValueError(f"{self.failure_code.value} is not assigned by deterministic rules")
        return self


def classify_failures(
    example: EvaluationExample,
    prediction: Prediction,
    *,
    label_to_category: Mapping[str, str],
    overconfident_threshold: float = 0.80,
    underconfident_threshold: float = 0.50,
) -> tuple[FailureAnnotation, ...]:
    """Assign only defensible deterministic categories.

    ANCHORING and INSUFFICIENT_CONTEXT are intentionally excluded because they need
    human/secondary evidence about reasoning or context sufficiency.
    """

    annotations: list[FailureAnnotation] = []
    exact_correct = (
        prediction.parse_status is ParseStatus.OK
        and prediction.predicted_root_cause_code == example.root_cause_code
    )
    predicted_category = (
        label_to_category.get(prediction.predicted_root_cause_code)
        if prediction.predicted_root_cause_code is not None
        else None
    )

    if prediction.evidence_citations:
        retrieved = set(prediction.retrieved_chunk_ids)
        unsupported = sorted(set(prediction.evidence_citations) - retrieved)
        if unsupported:
            annotations.append(
                FailureAnnotation(
                    failure_code=FailureCode.HALLUCINATED_EVIDENCE,
                    source=AnnotationSource.RULE,
                    notes="one or more cited chunk IDs were not present in retrieved context",
                    metadata={"unsupported_citations": unsupported},
                )
            )

    if exact_correct:
        confidence = prediction.confidence_probability
        if confidence is not None and confidence <= underconfident_threshold:
            annotations.append(
                FailureAnnotation(
                    failure_code=FailureCode.UNDERCONFIDENT_CORRECT,
                    source=AnnotationSource.RULE,
                    notes="correct prediction at or below the documented underconfidence threshold",
                    metadata={"threshold": underconfident_threshold, "confidence": confidence},
                )
            )
        return tuple(annotations)

    if (
        prediction.parse_status is ParseStatus.OK
        and predicted_category is not None
        and predicted_category == example.root_cause_category
    ):
        annotations.append(
            FailureAnnotation(
                failure_code=FailureCode.CORRECT_CATEGORY_WRONG_CAUSE,
                source=AnnotationSource.RULE,
                notes="predicted label shares the true category but not the exact root-cause code",
            )
        )

    confidence = prediction.confidence_probability
    if confidence is not None and confidence >= overconfident_threshold:
        annotations.append(
            FailureAnnotation(
                failure_code=FailureCode.OVERCONFIDENT_WRONG,
                source=AnnotationSource.RULE,
                notes="incorrect prediction at or above the documented overconfidence threshold",
                metadata={"threshold": overconfident_threshold, "confidence": confidence},
            )
        )

    relevant = set(example.relevant_chunk_ids)
    if relevant and not (relevant & set(prediction.retrieved_chunk_ids)):
        annotations.append(
            FailureAnnotation(
                failure_code=FailureCode.RETRIEVAL_MISS,
                source=AnnotationSource.RULE,
                notes="none of the reference-relevant chunks appeared in retrieved context",
                metadata={"relevant_chunk_count": len(relevant)},
            )
        )

    if not annotations:
        notes = (
            f"parse failure: {prediction.parse_status.value}"
            if prediction.parse_status is not ParseStatus.OK
            else "incorrect prediction did not meet a deterministic failure rule"
        )
        annotations.append(
            FailureAnnotation(
                failure_code=FailureCode.OTHER_UNCLASSIFIED,
                source=AnnotationSource.RULE,
                notes=notes,
            )
        )
    return tuple(annotations)


def manual_annotation(
    failure_code: FailureCode,
    *,
    notes: str,
    secondary: bool = False,
    metadata: dict[str, object] | None = None,
) -> FailureAnnotation:
    return FailureAnnotation(
        failure_code=failure_code,
        source=AnnotationSource.SECONDARY if secondary else AnnotationSource.MANUAL,
        notes=notes,
        metadata=metadata or {},
    )
