from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


class CalibrationError(ValueError):
    pass


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise CalibrationError("at least one score is required")
    if any(not math.isfinite(value) for value in values):
        raise CalibrationError("scores must be finite")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def normalize_label_sequence_log_likelihoods(
    label_log_likelihoods: Mapping[str, float], *, allowed_labels: Sequence[str]
) -> dict[str, float]:
    labels = tuple(allowed_labels)
    if not labels or len(labels) != len(set(labels)):
        raise CalibrationError("allowed_labels must be a non-empty unique sequence")
    if set(label_log_likelihoods) != set(labels):
        raise CalibrationError("log-likelihood labels must exactly match allowed_labels")
    scores = [float(label_log_likelihoods[label]) for label in labels]
    normalizer = _logsumexp(scores)
    probabilities = {label: math.exp(score - normalizer) for label, score in zip(labels, scores, strict=True)}
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("normalized label probabilities did not sum to one")
    return probabilities


def apply_temperature(
    label_log_likelihoods: Mapping[str, float],
    *,
    allowed_labels: Sequence[str],
    temperature: float,
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationError("temperature must be finite and positive")
    scaled = {label: float(score) / temperature for label, score in label_log_likelihoods.items()}
    return normalize_label_sequence_log_likelihoods(scaled, allowed_labels=allowed_labels)


@dataclass(frozen=True)
class ConfidenceScore:
    predicted_label: str
    probability: float
    probabilities: dict[str, float]
    source: str


def confidence_from_label_sequence_log_likelihoods(
    label_log_likelihoods: Mapping[str, float],
    *,
    allowed_labels: Sequence[str],
    temperature: float | None = None,
) -> ConfidenceScore:
    if temperature is None:
        probabilities = normalize_label_sequence_log_likelihoods(
            label_log_likelihoods, allowed_labels=allowed_labels
        )
        source = "normalized_label_sequence_log_likelihood"
    else:
        probabilities = apply_temperature(
            label_log_likelihoods,
            allowed_labels=allowed_labels,
            temperature=temperature,
        )
        source = "temperature_scaled_label_sequence_log_likelihood"
    predicted = max(tuple(allowed_labels), key=lambda label: (probabilities[label], label))
    return ConfidenceScore(
        predicted_label=predicted,
        probability=probabilities[predicted],
        probabilities=probabilities,
        source=source,
    )


@dataclass(frozen=True)
class TemperatureCalibration:
    calibration_version: str
    temperature: float
    source_split: str
    allowed_labels: tuple[str, ...]
    validation_nll: float


def _mean_nll(
    rows: Sequence[Mapping[str, float]],
    truths: Sequence[str],
    *,
    labels: tuple[str, ...],
    temperature: float,
) -> float:
    if len(rows) != len(truths) or not rows:
        raise CalibrationError("calibration rows and truths must be aligned and non-empty")
    total = 0.0
    for row, truth in zip(rows, truths, strict=True):
        if truth not in labels:
            raise CalibrationError(f"truth label is outside allowed label set: {truth}")
        probabilities = apply_temperature(row, allowed_labels=labels, temperature=temperature)
        probability = max(probabilities[truth], 1e-300)
        total -= math.log(probability)
    return total / len(rows)


def fit_temperature_scaling(
    label_log_likelihood_rows: Sequence[Mapping[str, float]],
    truth_labels: Sequence[str],
    *,
    allowed_labels: Sequence[str],
    source_split: str,
    iterations: int = 96,
) -> TemperatureCalibration:
    """Fit one scalar temperature using validation predictions only.

    The search is deterministic golden-section optimization over log-temperature in
    [log(0.05), log(20)]. Test predictions are explicitly rejected as a fitting source.
    """

    if source_split != "validation":
        raise CalibrationError("temperature scaling may be fit from validation data only")
    labels = tuple(allowed_labels)
    if not labels or len(labels) != len(set(labels)):
        raise CalibrationError("allowed_labels must be a non-empty unique sequence")
    if iterations < 16:
        raise CalibrationError("at least 16 optimization iterations are required")
    for row in label_log_likelihood_rows:
        if set(row) != set(labels):
            raise CalibrationError("every calibration row must exactly match allowed_labels")
        if any(not math.isfinite(float(value)) for value in row.values()):
            raise CalibrationError("calibration scores must be finite")

    low = math.log(0.05)
    high = math.log(20.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    left = high - (high - low) / phi
    right = low + (high - low) / phi

    def objective(log_temperature: float) -> float:
        return _mean_nll(
            label_log_likelihood_rows,
            truth_labels,
            labels=labels,
            temperature=math.exp(log_temperature),
        )

    left_value = objective(left)
    right_value = objective(right)
    for _ in range(iterations):
        if left_value <= right_value:
            high = right
            right = left
            right_value = left_value
            left = high - (high - low) / phi
            left_value = objective(left)
        else:
            low = left
            left = right
            left_value = right_value
            right = low + (high - low) / phi
            right_value = objective(right)

    log_temperature = (low + high) / 2.0
    temperature = math.exp(log_temperature)
    validation_nll = objective(log_temperature)
    identity = {
        "method": "temperature_scaling_golden_section_v1",
        "source_split": source_split,
        "allowed_labels": labels,
        "temperature": format(temperature, ".17g"),
        "validation_nll": format(validation_nll, ".17g"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    version = "temp-v1-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return TemperatureCalibration(
        calibration_version=version,
        temperature=temperature,
        source_split=source_split,
        allowed_labels=labels,
        validation_nll=validation_nll,
    )
