from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.evaluation.contracts import ParseStatus, Prediction

LABELS = {"a", "b", "c"}


def test_parser_records_invalid_json_without_raising() -> None:
    prediction = Prediction.from_raw_output(
        incident_id="i1",
        raw_model_output="{not-json",
        allowed_labels=LABELS,
    )
    assert prediction.parse_status is ParseStatus.INVALID_JSON
    assert prediction.predicted_root_cause_code is None
    assert prediction.parse_error


def test_parser_records_missing_required_field() -> None:
    prediction = Prediction.from_raw_output(
        incident_id="i2",
        raw_model_output=json.dumps({"reasoning": "missing label"}),
        allowed_labels=LABELS,
    )
    assert prediction.parse_status is ParseStatus.MISSING_REQUIRED_FIELD


def test_parser_records_unknown_label() -> None:
    prediction = Prediction.from_raw_output(
        incident_id="i3",
        raw_model_output=json.dumps({"root_cause_code": "unknown"}),
        allowed_labels=LABELS,
    )
    assert prediction.parse_status is ParseStatus.UNKNOWN_LABEL
    assert prediction.predicted_root_cause_code == "unknown"


def test_parser_records_partial_optional_output() -> None:
    prediction = Prediction.from_raw_output(
        incident_id="i4",
        raw_model_output=json.dumps({"root_cause_code": "a", "ranked_labels": "not-a-list"}),
        allowed_labels=LABELS,
    )
    assert prediction.parse_status is ParseStatus.PARTIAL_OUTPUT
    assert prediction.predicted_root_cause_code == "a"


def test_parser_accepts_named_confidence_only() -> None:
    prediction = Prediction.from_raw_output(
        incident_id="i5",
        raw_model_output=json.dumps(
            {
                "root_cause_code": "a",
                "confidence_probability": 0.8,
                "confidence_source": "normalized_label_sequence_log_likelihood",
            }
        ),
        allowed_labels=LABELS,
    )
    assert prediction.parse_status is ParseStatus.OK
    assert prediction.confidence_probability == pytest.approx(0.8)


def test_confidence_without_source_is_rejected_by_contract() -> None:
    with pytest.raises(ValidationError, match="present together"):
        Prediction(
            incident_id="i6",
            predicted_root_cause_code="a",
            confidence_probability=0.9,
        )


def test_ranked_labels_must_be_descending_and_unique() -> None:
    with pytest.raises(ValidationError, match="descending"):
        Prediction(
            incident_id="i7",
            predicted_root_cause_code="a",
            ranked_labels=(
                {"label": "a", "score": 0.5},
                {"label": "b", "score": 0.7},
            ),
        )
    with pytest.raises(ValidationError, match="unique"):
        Prediction(
            incident_id="i8",
            predicted_root_cause_code="a",
            ranked_labels=(
                {"label": "a", "score": 0.7},
                {"label": "a", "score": 0.5},
            ),
        )
