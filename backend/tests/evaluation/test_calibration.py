from __future__ import annotations

import math

import pytest

from app.evaluation.calibration import (
    CalibrationError,
    confidence_from_label_sequence_log_likelihoods,
    fit_temperature_scaling,
    normalize_label_sequence_log_likelihoods,
)


LABELS = ("a", "b", "c")


def test_known_log_likelihoods_normalize_to_expected_probabilities() -> None:
    scores = {"a": math.log(0.7), "b": math.log(0.2), "c": math.log(0.1)}
    probabilities = normalize_label_sequence_log_likelihoods(scores, allowed_labels=LABELS)
    assert probabilities == pytest.approx({"a": 0.7, "b": 0.2, "c": 0.1})
    confidence = confidence_from_label_sequence_log_likelihoods(scores, allowed_labels=LABELS)
    assert confidence.predicted_label == "a"
    assert confidence.probability == pytest.approx(0.7)
    assert confidence.source == "normalized_label_sequence_log_likelihood"


def test_temperature_scaling_rejects_test_fit_source() -> None:
    rows = [
        {"a": 2.0, "b": 0.0, "c": -1.0},
        {"a": 0.0, "b": 2.0, "c": -1.0},
    ]
    with pytest.raises(CalibrationError, match="validation data only"):
        fit_temperature_scaling(
            rows,
            ["a", "b"],
            allowed_labels=LABELS,
            source_split="test",
        )


def test_temperature_scaling_is_deterministic_and_improves_overconfident_validation_nll() -> None:
    rows = [
        {"a": 8.0, "b": 0.0, "c": -1.0},
        {"a": 8.0, "b": 0.0, "c": -1.0},
        {"a": 0.0, "b": 8.0, "c": -1.0},
        {"a": 0.0, "b": 8.0, "c": -1.0},
    ]
    truths = ["a", "b", "b", "a"]
    first = fit_temperature_scaling(
        rows,
        truths,
        allowed_labels=LABELS,
        source_split="validation",
    )
    second = fit_temperature_scaling(
        rows,
        truths,
        allowed_labels=LABELS,
        source_split="validation",
    )
    assert first == second
    assert first.source_split == "validation"
    assert first.temperature > 1.0
    assert first.validation_nll < 4.1
    assert first.calibration_version.startswith("temp-v1-")


def test_label_log_likelihood_scoring_requires_exact_allowed_label_set() -> None:
    with pytest.raises(CalibrationError, match="exactly match"):
        normalize_label_sequence_log_likelihoods(
            {"a": 1.0, "b": 0.0},
            allowed_labels=LABELS,
        )
