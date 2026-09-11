from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.evaluation.contracts import EvaluationExample, ParseStatus, Prediction


@dataclass(frozen=True)
class DeterministicGrade:
    incident_id: str
    exact_correct: bool
    category_correct: bool
    top3_correct: bool


class DeterministicRootCauseGrader:
    name = "deterministic_root_cause_v1"

    def grade(
        self,
        example: EvaluationExample,
        prediction: Prediction,
        *,
        label_to_category: Mapping[str, str],
    ) -> DeterministicGrade:
        valid = prediction.parse_status is ParseStatus.OK
        predicted = prediction.predicted_root_cause_code
        exact = valid and predicted == example.root_cause_code
        category = (
            valid
            and predicted is not None
            and label_to_category.get(predicted) == example.root_cause_category
        )
        ranked = [entry.label for entry in prediction.ranked_labels[:3]]
        if not ranked and predicted is not None:
            ranked = [predicted]
        return DeterministicGrade(
            incident_id=example.incident_id,
            exact_correct=exact,
            category_correct=category,
            top3_correct=valid and example.root_cause_code in ranked,
        )
