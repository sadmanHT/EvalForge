from __future__ import annotations

import math
import random
from typing import Any

import pytest

from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.metrics import (
    MetricInputError,
    cost_metrics,
    exact_accuracy,
    expected_calibration_error,
    hierarchical_accuracy,
    latency_percentiles,
    normalized_multiclass_brier_score,
    reliability_diagram_data,
    retrieval_context_metrics,
    top_k_accuracy,
)


def test_golden_metrics_match_hand_calculated_fixture(
    golden_payload: dict[str, Any],
    golden_examples: list[EvaluationExample],
    golden_predictions: list[Prediction],
) -> None:
    labels = golden_payload["allowed_labels"]
    categories = golden_payload["label_to_category"]
    expected = golden_payload["expected"]
    assert exact_accuracy(golden_examples, golden_predictions) == pytest.approx(
        expected["primary.exact_accuracy"]
    )
    assert hierarchical_accuracy(golden_examples, golden_predictions, categories) == pytest.approx(
        expected["supporting.hierarchical_accuracy"]
    )
    assert top_k_accuracy(golden_examples, golden_predictions, k=3) == pytest.approx(
        expected["supporting.top3_accuracy"]
    )
    assert normalized_multiclass_brier_score(
        golden_examples,
        golden_predictions,
        allowed_labels=labels,
    ) == pytest.approx(expected["calibration.normalized_multiclass_brier"])
    assert expected_calibration_error(
        golden_examples,
        golden_predictions,
        n_bins=golden_payload["ece_bins"],
    ) == pytest.approx(expected["calibration.ece"])
    p50, p95 = latency_percentiles(golden_predictions)
    assert p50 == pytest.approx(expected["latency.p50_ms"])
    assert p95 == pytest.approx(expected["latency.p95_ms"])
    costs = cost_metrics(
        golden_predictions,
        upfront_cost_usd=golden_payload["upfront_cost_usd"],
    )
    assert costs.marginal_mean_usd == pytest.approx(expected["cost.marginal_mean_usd"])
    assert costs.amortized_mean_usd == pytest.approx(expected["cost.amortized_mean_usd"])


def test_reliability_fixture_has_expected_fixed_width_bin(
    golden_payload: dict[str, Any],
    golden_examples: list[EvaluationExample],
    golden_predictions: list[Prediction],
) -> None:
    bins = reliability_diagram_data(
        golden_examples,
        golden_predictions,
        n_bins=golden_payload["ece_bins"],
    )
    assert len(bins) == 1
    assert bins[0].lower == pytest.approx(0.6)
    assert bins[0].upper == pytest.approx(0.8)
    assert bins[0].count == 4
    assert bins[0].accuracy == pytest.approx(0.75)
    assert bins[0].mean_confidence == pytest.approx(0.675)
    assert bins[0].absolute_gap == pytest.approx(0.075)


def test_metric_bounds_hold_for_random_valid_probability_vectors() -> None:
    generator = random.Random(20260908)
    labels = ["a", "b", "c", "d"]
    categories = {label: label for label in labels}
    for iteration in range(100):
        raw = [generator.random() for _ in labels]
        total = sum(raw)
        probabilities = {label: value / total for label, value in zip(labels, raw, strict=True)}
        predicted = max(labels, key=probabilities.__getitem__)
        prediction = Prediction(
            incident_id=f"random-{iteration}",
            predicted_root_cause_code=predicted,
            ranked_labels=tuple(
                {"label": label, "score": probabilities[label], "probability": probabilities[label]}
                for label in sorted(labels, key=probabilities.__getitem__, reverse=True)
            ),
            label_probabilities=probabilities,
            confidence_probability=probabilities[predicted],
            confidence_source="normalized_label_sequence_log_likelihood",
        )
        example = EvaluationExample(
            incident_id=f"random-{iteration}",
            split="test",
            root_cause_code=labels[iteration % len(labels)],
            root_cause_category=labels[iteration % len(labels)],
        )
        accuracy = exact_accuracy([example], [prediction])
        hierarchical = hierarchical_accuracy([example], [prediction], categories)
        top3 = top_k_accuracy([example], [prediction], k=3)
        brier = normalized_multiclass_brier_score([example], [prediction], allowed_labels=labels)
        ece = expected_calibration_error([example], [prediction], n_bins=10)
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= hierarchical <= 1.0
        assert 0.0 <= top3 <= 1.0
        assert 0.0 <= brier <= 1.0
        assert 0.0 <= ece <= 1.0
        assert all(math.isfinite(value) for value in (accuracy, hierarchical, top3, brier, ece))


def test_retrieval_context_precision_and_recall() -> None:
    examples = [
        EvaluationExample(
            incident_id="r1",
            split="test",
            root_cause_code="a",
            root_cause_category="x",
            relevant_chunk_ids=("c1", "c2"),
        ),
        EvaluationExample(
            incident_id="r2",
            split="test",
            root_cause_code="b",
            root_cause_category="y",
            relevant_chunk_ids=("c3",),
        ),
    ]
    predictions = [
        Prediction(
            incident_id="r1",
            predicted_root_cause_code="a",
            retrieved_chunk_ids=("c1", "noise"),
        ),
        Prediction(
            incident_id="r2",
            predicted_root_cause_code="b",
            retrieved_chunk_ids=("c3",),
        ),
    ]
    metrics = retrieval_context_metrics(examples, predictions)
    assert metrics.context_precision == pytest.approx(0.75)
    assert metrics.context_recall == pytest.approx(0.75)
    assert metrics.evaluated_examples == 2


def test_invalid_metric_inputs_are_rejected(
    golden_examples: list[EvaluationExample], golden_predictions: list[Prediction]
) -> None:
    with pytest.raises(MetricInputError, match="incident mismatch"):
        exact_accuracy(golden_examples[:-1], golden_predictions)
    with pytest.raises(MetricInputError, match="positive"):
        top_k_accuracy(golden_examples, golden_predictions, k=0)
    missing_probabilities = golden_predictions[0].model_copy(update={"label_probabilities": None})
    with pytest.raises(MetricInputError, match="full label probability"):
        normalized_multiclass_brier_score(
            [golden_examples[0]],
            [missing_probabilities],
            allowed_labels=list(golden_predictions[0].label_probabilities or {}),
        )


def test_common_harness_matches_golden_fixture_and_is_deterministic(
    golden_payload: dict[str, Any],
    golden_examples: list[EvaluationExample],
    golden_predictions: list[Prediction],
) -> None:
    harness = EvaluationHarness(
        evaluator_version="phase5-evaluator-v1",
        allowed_labels=golden_payload["allowed_labels"],
        label_to_category=golden_payload["label_to_category"],
        ece_bins=golden_payload["ece_bins"],
    )
    first = harness.evaluate(
        golden_examples,
        golden_predictions,
        upfront_cost_usd=golden_payload["upfront_cost_usd"],
    )
    second = harness.evaluate(
        list(reversed(golden_examples)),
        list(reversed(golden_predictions)),
        upfront_cost_usd=golden_payload["upfront_cost_usd"],
    )
    assert first.result_hash == second.result_hash
    assert first.metric_map() == pytest.approx(golden_payload["expected"])
    assert first.prediction_count == 4
