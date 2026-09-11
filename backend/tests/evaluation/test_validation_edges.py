from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from app.evaluation.calibration import (
    CalibrationError,
    apply_temperature,
    confidence_from_label_sequence_log_likelihoods,
    fit_temperature_scaling,
    normalize_label_sequence_log_likelihoods,
)
from app.evaluation.contracts import EvaluationExample, ParseStatus, Prediction, RankedLabel
from app.evaluation.failure_taxonomy import FailureCode, manual_annotation
from app.evaluation.graders.deterministic import DeterministicRootCauseGrader
from app.evaluation.graders.supporting import RagasSupportingAdapter, SupportingMetric
from app.evaluation.harness import EvaluationHarness, EvaluatorRegistry
from app.evaluation.metrics import (
    MetricInputError,
    cost_metrics,
    exact_accuracy,
    latency_percentiles,
    mean,
    normalized_multiclass_brier_score,
    quantile,
    reliability_diagram_data,
    retrieval_context_metrics,
)
from app.evaluation.statistics import StatisticsInputError, paired_bootstrap_accuracy_difference

LABELS = ("a", "b", "c")
CATEGORIES = {"a": "x", "b": "x", "c": "y"}


def _example(incident_id: str = "i1") -> EvaluationExample:
    return EvaluationExample(
        incident_id=incident_id,
        split="test",
        root_cause_code="a",
        root_cause_category="x",
    )


def _prediction(incident_id: str = "i1", **updates: object) -> Prediction:
    base = Prediction(
        incident_id=incident_id,
        predicted_root_cause_code="a",
        ranked_labels=({"label": "a", "score": 1.0},),
    )
    return base.model_copy(update=updates)


def test_contract_validators_reject_invalid_internal_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        RankedLabel(label="a", score=math.inf)
    with pytest.raises(ValidationError, match="require predicted_root_cause_code"):
        Prediction(incident_id="missing-root")
    with pytest.raises(ValidationError, match="top-ranked label"):
        Prediction(
            incident_id="rank-mismatch",
            predicted_root_cause_code="a",
            ranked_labels=({"label": "b", "score": 1.0},),
        )
    with pytest.raises(ValidationError, match="may not be empty"):
        Prediction(
            incident_id="empty-probs",
            predicted_root_cause_code="a",
            label_probabilities={},
        )
    with pytest.raises(ValidationError, match="probability labels may not be empty"):
        Prediction(
            incident_id="empty-prob-label",
            predicted_root_cause_code="a",
            label_probabilities={"": 1.0},
        )
    with pytest.raises(ValidationError, match="sum to one"):
        Prediction(
            incident_id="bad-prob-sum",
            predicted_root_cause_code="a",
            label_probabilities={"a": 0.6, "b": 0.2},
        )
    with pytest.raises(ValidationError, match="total_tokens"):
        Prediction(
            incident_id="token-mismatch",
            predicted_root_cause_code="a",
            input_tokens=2,
            output_tokens=3,
            total_tokens=6,
        )


def test_raw_parser_records_non_object_rank_and_confidence_failures() -> None:
    non_object = Prediction.from_raw_output(
        incident_id="non-object",
        raw_model_output=json.dumps(["a"]),
        allowed_labels=set(LABELS),
    )
    assert non_object.parse_status is ParseStatus.PARTIAL_OUTPUT
    assert non_object.predicted_root_cause_code is None

    invalid_rank_entry = Prediction.from_raw_output(
        incident_id="invalid-rank-entry",
        raw_model_output=json.dumps({"root_cause_code": "a", "ranked_labels": [5]}),
        allowed_labels=set(LABELS),
    )
    assert invalid_rank_entry.parse_status is ParseStatus.PARTIAL_OUTPUT
    assert invalid_rank_entry.ranked_labels == ()

    top_disagreement = Prediction.from_raw_output(
        incident_id="top-disagreement",
        raw_model_output=json.dumps(
            {
                "root_cause_code": "a",
                "ranked_labels": [{"label": "b", "score": 1.0}],
            }
        ),
        allowed_labels=set(LABELS),
    )
    assert top_disagreement.parse_status is ParseStatus.PARTIAL_OUTPUT

    unknown_rank = Prediction.from_raw_output(
        incident_id="unknown-rank",
        raw_model_output=json.dumps(
            {
                "root_cause_code": "a",
                "ranked_labels": [
                    {"label": "a", "score": 1.0},
                    {"label": "unknown", "score": 0.5},
                ],
            }
        ),
        allowed_labels=set(LABELS),
    )
    assert unknown_rank.parse_status is ParseStatus.PARTIAL_OUTPUT
    assert unknown_rank.ranked_labels == ()

    missing_confidence_source = Prediction.from_raw_output(
        incident_id="missing-confidence-source",
        raw_model_output=json.dumps({"root_cause_code": "a", "confidence_probability": 0.7}),
        allowed_labels=set(LABELS),
    )
    assert missing_confidence_source.parse_status is ParseStatus.PARTIAL_OUTPUT

    invalid_confidence = Prediction.from_raw_output(
        incident_id="invalid-confidence",
        raw_model_output=json.dumps(
            {
                "root_cause_code": "a",
                "confidence_probability": "not-a-number",
                "confidence_source": "normalized_label_sequence_log_likelihood",
            }
        ),
        allowed_labels=set(LABELS),
    )
    assert invalid_confidence.parse_status is ParseStatus.PARTIAL_OUTPUT


def test_metric_helpers_reject_ambiguous_or_incomplete_inputs() -> None:
    example = _example()
    prediction = _prediction()

    with pytest.raises(MetricInputError, match="at least one"):
        exact_accuracy([], [])
    with pytest.raises(MetricInputError, match="prediction incident IDs must be unique"):
        exact_accuracy([example], [prediction, prediction])
    with pytest.raises(MetricInputError, match="example incident IDs must be unique"):
        exact_accuracy([example, example], [prediction])
    with pytest.raises(MetricInputError, match="allowed_labels"):
        normalized_multiclass_brier_score([example], [prediction], allowed_labels=[])

    wrong_labels = prediction.model_copy(update={"label_probabilities": {"a": 0.5, "b": 0.5}})
    with pytest.raises(MetricInputError, match="exactly match"):
        normalized_multiclass_brier_score(
            [example],
            [wrong_labels],
            allowed_labels=LABELS,
        )

    with pytest.raises(MetricInputError, match="n_bins"):
        reliability_diagram_data([example], [prediction], n_bins=0)
    with pytest.raises(MetricInputError, match="requires confidence"):
        reliability_diagram_data([example], [prediction])
    with pytest.raises(MetricInputError, match="latency_ms"):
        latency_percentiles([prediction])
    with pytest.raises(MetricInputError, match="upfront cost"):
        cost_metrics([prediction], upfront_cost_usd=-1.0)
    with pytest.raises(MetricInputError, match="cost_usd"):
        cost_metrics([prediction])
    with pytest.raises(MetricInputError, match="reference relevant_chunk_ids"):
        retrieval_context_metrics([example], [prediction])


def test_numeric_helpers_cover_boundaries_and_invalid_values() -> None:
    assert quantile([1.0, 3.0], 0.5) == pytest.approx(2.0)
    assert quantile([2.0], 0.5) == pytest.approx(2.0)
    with pytest.raises(MetricInputError, match="at least one"):
        quantile([], 0.5)
    with pytest.raises(MetricInputError, match=r"in \[0,1\]"):
        quantile([1.0], 1.5)
    with pytest.raises(MetricInputError, match="finite"):
        quantile([math.inf], 0.5)
    with pytest.raises(MetricInputError, match="at least one"):
        mean([])
    with pytest.raises(MetricInputError, match="finite"):
        mean([math.nan])


def test_calibration_rejects_invalid_inputs_and_supports_temperature_confidence() -> None:
    with pytest.raises(CalibrationError, match="non-empty unique"):
        normalize_label_sequence_log_likelihoods({}, allowed_labels=[])
    with pytest.raises(CalibrationError, match="finite"):
        normalize_label_sequence_log_likelihoods(
            {"a": math.inf, "b": 0.0, "c": 0.0},
            allowed_labels=LABELS,
        )
    with pytest.raises(CalibrationError, match="finite and positive"):
        apply_temperature(
            {"a": 1.0, "b": 0.0, "c": -1.0},
            allowed_labels=LABELS,
            temperature=0.0,
        )

    confidence = confidence_from_label_sequence_log_likelihoods(
        {"a": 1.0, "b": 0.0, "c": -1.0},
        allowed_labels=LABELS,
        temperature=2.0,
    )
    assert confidence.source == "temperature_scaled_label_sequence_log_likelihood"
    assert confidence.predicted_label == "a"

    with pytest.raises(CalibrationError, match="non-empty unique"):
        fit_temperature_scaling(
            [],
            [],
            allowed_labels=[],
            source_split="validation",
        )
    with pytest.raises(CalibrationError, match="at least 16"):
        fit_temperature_scaling(
            [{"a": 1.0, "b": 0.0, "c": -1.0}],
            ["a"],
            allowed_labels=LABELS,
            source_split="validation",
            iterations=15,
        )
    with pytest.raises(CalibrationError, match="exactly match"):
        fit_temperature_scaling(
            [{"a": 1.0, "b": 0.0}],
            ["a"],
            allowed_labels=LABELS,
            source_split="validation",
        )
    with pytest.raises(CalibrationError, match="scores must be finite"):
        fit_temperature_scaling(
            [{"a": math.inf, "b": 0.0, "c": -1.0}],
            ["a"],
            allowed_labels=LABELS,
            source_split="validation",
        )
    with pytest.raises(CalibrationError, match="aligned and non-empty"):
        fit_temperature_scaling(
            [{"a": 1.0, "b": 0.0, "c": -1.0}],
            [],
            allowed_labels=LABELS,
            source_split="validation",
        )
    with pytest.raises(CalibrationError, match="outside allowed"):
        fit_temperature_scaling(
            [{"a": 1.0, "b": 0.0, "c": -1.0}],
            ["outside"],
            allowed_labels=LABELS,
            source_split="validation",
        )


def test_harness_constructor_and_registry_guard_common_contract() -> None:
    with pytest.raises(ValueError, match="evaluator_version"):
        EvaluationHarness(
            evaluator_version="",
            allowed_labels=LABELS,
            label_to_category=CATEGORIES,
        )
    with pytest.raises(ValueError, match="non-empty unique"):
        EvaluationHarness(
            evaluator_version="v1",
            allowed_labels=("a", "a"),
            label_to_category={"a": "x"},
        )
    with pytest.raises(ValueError, match="exactly cover"):
        EvaluationHarness(
            evaluator_version="v1",
            allowed_labels=LABELS,
            label_to_category={"a": "x"},
        )
    with pytest.raises(ValueError, match="ece_bins"):
        EvaluationHarness(
            evaluator_version="v1",
            allowed_labels=LABELS,
            label_to_category=CATEGORIES,
            ece_bins=0,
        )

    harness = EvaluationHarness(
        evaluator_version="v1",
        allowed_labels=LABELS,
        label_to_category=CATEGORIES,
    )
    with pytest.raises(ValueError, match="unique incident IDs"):
        harness.evaluate([_example(), _example()], [_prediction(), _prediction()])
    with pytest.raises(ValueError, match="exactly the same"):
        harness.evaluate([_example("one")], [_prediction("two")])
    with pytest.raises(ValueError, match="at least one"):
        harness.evaluate([], [])

    registry = EvaluatorRegistry()
    with pytest.raises(ValueError, match="name is required"):
        registry.register("", harness)
    registry.register("common", harness)
    with pytest.raises(TypeError, match="upfront_cost_usd"):
        registry.evaluate("common", [_example()], [_prediction()], upfront_cost_usd="bad")
    with pytest.raises(TypeError, match="supporting_metrics must be a sequence"):
        registry.evaluate("common", [_example()], [_prediction()], supporting_metrics="bad")
    with pytest.raises(TypeError, match="entries must be SupportingMetric"):
        registry.evaluate("common", [_example()], [_prediction()], supporting_metrics=["bad"])


def test_harness_records_omitted_metrics_and_rejects_duplicate_names() -> None:
    harness = EvaluationHarness(
        evaluator_version="v1",
        allowed_labels=LABELS,
        label_to_category=CATEGORIES,
    )
    example = _example()
    prediction = _prediction()
    result = harness.evaluate([example], [prediction])
    assert set(result.omitted_metrics) == {
        "calibration.ece",
        "calibration.normalized_multiclass_brier",
        "cost.amortized_mean_usd",
        "cost.marginal_mean_usd",
        "latency.p50_ms",
        "latency.p95_ms",
        "retrieval.context_precision",
        "retrieval.context_recall",
    }
    assert result.canonical_payload()["prediction_count"] == 1

    duplicate = SupportingMetric(
        name="primary.exact_accuracy",
        value=1.0,
        metadata={"role": "supporting_only"},
    )
    with pytest.raises(ValueError, match="metric names must be unique"):
        harness.evaluate([example], [prediction], supporting_metrics=[duplicate])


def test_deterministic_grader_and_supporting_adapter_validation() -> None:
    grader = DeterministicRootCauseGrader()
    correct = grader.grade(_example(), _prediction(), label_to_category=CATEGORIES)
    assert correct.exact_correct is True
    assert correct.category_correct is True
    assert correct.top3_correct is True

    failed_prediction = Prediction(
        incident_id="i1",
        parse_status=ParseStatus.INVALID_JSON,
        raw_model_output="{",
    )
    failed = grader.grade(_example(), failed_prediction, label_to_category=CATEGORIES)
    assert failed.exact_correct is False
    assert failed.category_correct is False
    assert failed.top3_correct is False

    with pytest.raises(ValueError, match="version is required"):
        RagasSupportingAdapter(evaluator_version="")
    adapter = RagasSupportingAdapter(evaluator_version="v1")
    with pytest.raises(ValueError, match="finite"):
        adapter.adapt({"faithfulness": math.inf})


def test_statistics_and_manual_annotation_validation_edges() -> None:
    with pytest.raises(StatisticsInputError, match="non-empty"):
        paired_bootstrap_accuracy_difference({}, {})
    with pytest.raises(StatisticsInputError, match="at least 100"):
        paired_bootstrap_accuracy_difference(
            {"a": True},
            {"a": False},
            resamples=99,
        )
    with pytest.raises(StatisticsInputError, match="confidence_level"):
        paired_bootstrap_accuracy_difference(
            {"a": True},
            {"a": False},
            confidence_level=1.0,
        )

    manual = manual_annotation(FailureCode.ANCHORING, notes="manual review found anchoring")
    assert manual.source == "manual"
    secondary = manual_annotation(
        FailureCode.INSUFFICIENT_CONTEXT,
        notes="secondary review found missing context",
        secondary=True,
        metadata={"reviewer": "secondary"},
    )
    assert secondary.source == "secondary"
    assert secondary.metadata == {"reviewer": "secondary"}
