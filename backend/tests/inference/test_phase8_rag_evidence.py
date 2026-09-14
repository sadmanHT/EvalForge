from __future__ import annotations

import pytest

from app.evaluation.contracts import Prediction
from app.inference.rag_evidence import _validate_ece_contract


def _parse_failure_prediction(*, with_confidence: bool = False) -> dict[str, object]:
    prediction = Prediction(
        incident_id="incident-parse-failure",
        parse_status="INVALID_JSON",
        parse_error="fixture invalid JSON",
        raw_model_output="{",
        label_probabilities={"a": 0.8, "b": 0.2},
        confidence_probability=0.8 if with_confidence else None,
        confidence_source=("normalized_label_sequence_log_likelihood" if with_confidence else None),
    )
    return prediction.model_dump(mode="json")


def test_ece_may_be_omitted_for_parse_failure_without_confidence() -> None:
    _validate_ece_contract(
        metrics={"quality.parse_failure_rate": 1.0},
        predictions=[_parse_failure_prediction()],
    )


def test_ece_omission_requires_positive_parse_failure_rate() -> None:
    with pytest.raises(ValueError, match="only when parse failures occurred"):
        _validate_ece_contract(
            metrics={"quality.parse_failure_rate": 0.0},
            predictions=[_parse_failure_prediction()],
        )


def test_ece_omission_requires_a_parse_failure_missing_confidence() -> None:
    with pytest.raises(ValueError, match="without a parse-failure prediction lacking confidence"):
        _validate_ece_contract(
            metrics={"quality.parse_failure_rate": 1.0},
            predictions=[_parse_failure_prediction(with_confidence=True)],
        )


def test_present_ece_remains_required_to_be_numeric() -> None:
    _validate_ece_contract(
        metrics={"quality.parse_failure_rate": 0.0, "calibration.ece": 0.2},
        predictions=[],
    )
    with pytest.raises(ValueError, match="must be numeric"):
        _validate_ece_contract(
            metrics={"quality.parse_failure_rate": 0.0, "calibration.ece": True},
            predictions=[],
        )
