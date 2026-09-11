from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from app.evaluation.contracts import EvaluationExample, ParseStatus, Prediction


class MetricInputError(ValueError):
    pass


def _require_aligned(
    examples: Sequence[EvaluationExample], predictions: Sequence[Prediction]
) -> list[tuple[EvaluationExample, Prediction]]:
    if not examples:
        raise MetricInputError("at least one evaluation example is required")
    prediction_by_id = {prediction.incident_id: prediction for prediction in predictions}
    if len(prediction_by_id) != len(predictions):
        raise MetricInputError("prediction incident IDs must be unique")
    example_ids = {example.incident_id for example in examples}
    if len(example_ids) != len(examples):
        raise MetricInputError("example incident IDs must be unique")
    if set(prediction_by_id) != example_ids:
        missing = sorted(example_ids - set(prediction_by_id))
        unexpected = sorted(set(prediction_by_id) - example_ids)
        raise MetricInputError(f"prediction/example incident mismatch: missing={missing}, unexpected={unexpected}")
    return [(example, prediction_by_id[example.incident_id]) for example in examples]


def exact_accuracy(
    examples: Sequence[EvaluationExample], predictions: Sequence[Prediction]
) -> float:
    pairs = _require_aligned(examples, predictions)
    correct = sum(
        prediction.parse_status is ParseStatus.OK
        and prediction.predicted_root_cause_code == example.root_cause_code
        for example, prediction in pairs
    )
    return correct / len(pairs)


def hierarchical_accuracy(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[Prediction],
    label_to_category: Mapping[str, str],
) -> float:
    pairs = _require_aligned(examples, predictions)
    correct = 0
    for example, prediction in pairs:
        if prediction.parse_status is not ParseStatus.OK:
            continue
        predicted = prediction.predicted_root_cause_code
        if predicted is not None and label_to_category.get(predicted) == example.root_cause_category:
            correct += 1
    return correct / len(pairs)


def top_k_accuracy(
    examples: Sequence[EvaluationExample], predictions: Sequence[Prediction], *, k: int
) -> float:
    if k <= 0:
        raise MetricInputError("k must be positive")
    pairs = _require_aligned(examples, predictions)
    correct = 0
    for example, prediction in pairs:
        if prediction.parse_status is not ParseStatus.OK:
            continue
        ranked = [entry.label for entry in prediction.ranked_labels[:k]]
        if not ranked and prediction.predicted_root_cause_code is not None:
            ranked = [prediction.predicted_root_cause_code]
        if example.root_cause_code in ranked:
            correct += 1
    return correct / len(pairs)


def normalized_multiclass_brier_score(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[Prediction],
    *,
    allowed_labels: Sequence[str],
) -> float:
    """Return normalized multiclass Brier score in [0, 1].

    Each example contributes 0.5 * sum_k (p_k - y_k)^2. The 0.5 normalization
    makes the worst possible one-hot-vs-one-hot multiclass error equal to 1.
    """

    labels = tuple(allowed_labels)
    if not labels or len(labels) != len(set(labels)):
        raise MetricInputError("allowed_labels must be a non-empty unique sequence")
    pairs = _require_aligned(examples, predictions)
    total = 0.0
    for example, prediction in pairs:
        probabilities = prediction.label_probabilities
        if probabilities is None:
            raise MetricInputError("Brier score requires a full label probability distribution")
        if set(probabilities) != set(labels):
            raise MetricInputError("Brier probability labels must exactly match allowed_labels")
        squared = sum(
            (probabilities[label] - (1.0 if label == example.root_cause_code else 0.0)) ** 2
            for label in labels
        )
        total += 0.5 * squared
    value = total / len(pairs)
    if not 0.0 <= value <= 1.0 + 1e-12:
        raise AssertionError("normalized Brier score escaped [0,1]")
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    mean_confidence: float
    absolute_gap: float


def reliability_diagram_data(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[Prediction],
    *,
    n_bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    if n_bins <= 0:
        raise MetricInputError("n_bins must be positive")
    pairs = _require_aligned(examples, predictions)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for example, prediction in pairs:
        confidence = prediction.confidence_probability
        if confidence is None:
            raise MetricInputError("ECE requires confidence for every prediction")
        if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
            raise MetricInputError("confidence values must be finite values in [0,1]")
        index = min(int(confidence * n_bins), n_bins - 1)
        correct = (
            prediction.parse_status is ParseStatus.OK
            and prediction.predicted_root_cause_code == example.root_cause_code
        )
        buckets[index].append((confidence, correct))

    result: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        lower = index / n_bins
        upper = (index + 1) / n_bins
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                accuracy=accuracy,
                mean_confidence=mean_confidence,
                absolute_gap=abs(accuracy - mean_confidence),
            )
        )
    return tuple(result)


def expected_calibration_error(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[Prediction],
    *,
    n_bins: int = 10,
) -> float:
    bins = reliability_diagram_data(examples, predictions, n_bins=n_bins)
    total_count = sum(item.count for item in bins)
    if total_count == 0:
        raise MetricInputError("ECE requires at least one prediction")
    value = sum((item.count / total_count) * item.absolute_gap for item in bins)
    if not 0.0 <= value <= 1.0 + 1e-12:
        raise AssertionError("ECE escaped [0,1]")
    return min(max(value, 0.0), 1.0)


def quantile(values: Sequence[float], q: float) -> float:
    """Deterministic R-7/NumPy-default linear quantile without third-party math deps."""

    if not values:
        raise MetricInputError("quantile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise MetricInputError("quantile q must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise MetricInputError("quantile values must be finite")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_percentiles(predictions: Sequence[Prediction]) -> tuple[float, float]:
    values = [prediction.latency_ms for prediction in predictions if prediction.latency_ms is not None]
    if len(values) != len(predictions) or not values:
        raise MetricInputError("latency metrics require latency_ms for every prediction")
    return quantile(values, 0.50), quantile(values, 0.95)


@dataclass(frozen=True)
class CostMetrics:
    marginal_mean_usd: float
    amortized_mean_usd: float
    total_marginal_usd: float
    upfront_cost_usd: float


def cost_metrics(
    predictions: Sequence[Prediction], *, upfront_cost_usd: float = 0.0
) -> CostMetrics:
    if upfront_cost_usd < 0.0 or not math.isfinite(upfront_cost_usd):
        raise MetricInputError("upfront cost must be finite and non-negative")
    costs = [prediction.cost_usd for prediction in predictions if prediction.cost_usd is not None]
    if len(costs) != len(predictions) or not costs:
        raise MetricInputError("cost metrics require cost_usd for every prediction")
    if any(cost < 0.0 or not math.isfinite(cost) for cost in costs):
        raise MetricInputError("prediction costs must be finite and non-negative")
    total = sum(costs)
    count = len(costs)
    return CostMetrics(
        marginal_mean_usd=total / count,
        amortized_mean_usd=(total + upfront_cost_usd) / count,
        total_marginal_usd=total,
        upfront_cost_usd=upfront_cost_usd,
    )


@dataclass(frozen=True)
class RetrievalMetrics:
    context_precision: float
    context_recall: float
    evaluated_examples: int


def retrieval_context_metrics(
    examples: Sequence[EvaluationExample], predictions: Sequence[Prediction]
) -> RetrievalMetrics:
    pairs = _require_aligned(examples, predictions)
    precisions: list[float] = []
    recalls: list[float] = []
    for example, prediction in pairs:
        relevant = set(example.relevant_chunk_ids)
        if not relevant:
            continue
        retrieved = set(prediction.retrieved_chunk_ids)
        overlap = len(relevant & retrieved)
        precisions.append(overlap / len(retrieved) if retrieved else 0.0)
        recalls.append(overlap / len(relevant))
    if not precisions:
        raise MetricInputError("retrieval metrics require reference relevant_chunk_ids")
    return RetrievalMetrics(
        context_precision=sum(precisions) / len(precisions),
        context_recall=sum(recalls) / len(recalls),
        evaluated_examples=len(precisions),
    )


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise MetricInputError("mean requires at least one value")
    if any(not math.isfinite(value) for value in materialized):
        raise MetricInputError("mean values must be finite")
    return sum(materialized) / len(materialized)
