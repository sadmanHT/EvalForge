from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from app.evaluation.contracts import EvaluationExample, ParseStatus, Prediction
from app.evaluation.failure_taxonomy import (
    FAILURE_TAXONOMY_VERSION,
    FailureAnnotation,
    classify_failures,
)
from app.evaluation.graders.deterministic import DeterministicRootCauseGrader
from app.evaluation.graders.supporting import SupportingMetric
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


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class FailureRecord:
    incident_id: str
    annotation: FailureAnnotation


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_version: str
    failure_taxonomy_version: str
    prediction_count: int
    metrics: tuple[MetricValue, ...]
    reliability: tuple[dict[str, object], ...]
    failures: tuple[FailureRecord, ...]
    omitted_metrics: dict[str, str]
    result_hash: str

    def metric_map(self) -> dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}

    def canonical_payload(self) -> dict[str, object]:
        return _result_payload(
            evaluator_version=self.evaluator_version,
            failure_taxonomy_version=self.failure_taxonomy_version,
            prediction_count=self.prediction_count,
            metrics=self.metrics,
            reliability=self.reliability,
            failures=self.failures,
            omitted_metrics=self.omitted_metrics,
        )


def _result_payload(
    *,
    evaluator_version: str,
    failure_taxonomy_version: str,
    prediction_count: int,
    metrics: Sequence[MetricValue],
    reliability: Sequence[dict[str, object]],
    failures: Sequence[FailureRecord],
    omitted_metrics: Mapping[str, str],
) -> dict[str, object]:
    return {
        "evaluator_version": evaluator_version,
        "failure_taxonomy_version": failure_taxonomy_version,
        "prediction_count": prediction_count,
        "metrics": [
            {"name": metric.name, "value": metric.value, "metadata": metric.metadata}
            for metric in sorted(metrics, key=lambda item: item.name)
        ],
        "reliability": list(reliability),
        "failures": [
            {
                "incident_id": record.incident_id,
                "annotation": record.annotation.model_dump(mode="json", exclude_none=False),
            }
            for record in sorted(
                failures,
                key=lambda item: (item.incident_id, item.annotation.failure_code.value),
            )
        ],
        "omitted_metrics": dict(sorted(omitted_metrics.items())),
    }


def _result_hash(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class EvaluationHarness:
    """One common evaluator for ZERO_SHOT, RAG, FINETUNED, and COMBINED outputs."""

    def __init__(
        self,
        *,
        evaluator_version: str,
        allowed_labels: Sequence[str],
        label_to_category: Mapping[str, str],
        ece_bins: int = 10,
    ) -> None:
        labels = tuple(allowed_labels)
        if not evaluator_version:
            raise ValueError("evaluator_version is required")
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        if set(label_to_category) != set(labels):
            raise ValueError("label_to_category must exactly cover allowed_labels")
        if ece_bins <= 0:
            raise ValueError("ece_bins must be positive")
        self.evaluator_version = evaluator_version
        self.allowed_labels = labels
        self.label_to_category = dict(label_to_category)
        self.ece_bins = ece_bins
        self.grader = DeterministicRootCauseGrader()

    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        predictions: Sequence[Prediction],
        *,
        upfront_cost_usd: float = 0.0,
        supporting_metrics: Sequence[SupportingMetric] = (),
    ) -> EvaluationResult:
        example_by_id = {example.incident_id: example for example in examples}
        prediction_by_id = {prediction.incident_id: prediction for prediction in predictions}
        if len(example_by_id) != len(examples) or len(prediction_by_id) != len(predictions):
            raise ValueError("examples and predictions must have unique incident IDs")
        if set(example_by_id) != set(prediction_by_id):
            raise ValueError("examples and predictions must cover exactly the same incident IDs")
        ordered_examples = [example_by_id[incident_id] for incident_id in sorted(example_by_id)]
        ordered_predictions = [prediction_by_id[incident_id] for incident_id in sorted(prediction_by_id)]
        if not ordered_examples:
            raise ValueError("evaluation requires at least one prediction")

        metrics: list[MetricValue] = [
            MetricValue(
                "primary.exact_accuracy",
                exact_accuracy(ordered_examples, ordered_predictions),
                {"grader": self.grader.name, "role": "primary"},
            ),
            MetricValue(
                "supporting.hierarchical_accuracy",
                hierarchical_accuracy(
                    ordered_examples,
                    ordered_predictions,
                    self.label_to_category,
                ),
                {"role": "supporting", "taxonomy": "label_category"},
            ),
            MetricValue(
                "supporting.top3_accuracy",
                top_k_accuracy(ordered_examples, ordered_predictions, k=3),
                {"role": "supporting", "k": 3},
            ),
            MetricValue(
                "quality.parse_failure_rate",
                sum(
                    prediction.parse_status is not ParseStatus.OK
                    for prediction in ordered_predictions
                )
                / len(ordered_predictions),
                {"role": "diagnostic"},
            ),
        ]
        omitted: dict[str, str] = {}
        reliability: tuple[dict[str, object], ...] = ()

        try:
            brier = normalized_multiclass_brier_score(
                ordered_examples,
                ordered_predictions,
                allowed_labels=self.allowed_labels,
            )
            metrics.append(
                MetricValue(
                    "calibration.normalized_multiclass_brier",
                    brier,
                    {"normalization": "0.5 * multiclass squared probability error"},
                )
            )
        except MetricInputError as exc:
            omitted["calibration.normalized_multiclass_brier"] = str(exc)

        try:
            ece = expected_calibration_error(
                ordered_examples,
                ordered_predictions,
                n_bins=self.ece_bins,
            )
            bins = reliability_diagram_data(
                ordered_examples,
                ordered_predictions,
                n_bins=self.ece_bins,
            )
            reliability = tuple(asdict(item) for item in bins)
            metrics.append(
                MetricValue(
                    "calibration.ece",
                    ece,
                    {"binning": "fixed_width", "n_bins": self.ece_bins},
                )
            )
        except MetricInputError as exc:
            omitted["calibration.ece"] = str(exc)

        try:
            p50, p95 = latency_percentiles(ordered_predictions)
            metrics.extend(
                [
                    MetricValue("latency.p50_ms", p50, {"quantile": "R7"}),
                    MetricValue("latency.p95_ms", p95, {"quantile": "R7"}),
                ]
            )
        except MetricInputError as exc:
            omitted["latency.p50_ms"] = str(exc)
            omitted["latency.p95_ms"] = str(exc)

        try:
            costs = cost_metrics(ordered_predictions, upfront_cost_usd=upfront_cost_usd)
            metrics.extend(
                [
                    MetricValue(
                        "cost.marginal_mean_usd",
                        costs.marginal_mean_usd,
                        {"upfront_cost_included": False},
                    ),
                    MetricValue(
                        "cost.amortized_mean_usd",
                        costs.amortized_mean_usd,
                        {
                            "upfront_cost_included": True,
                            "upfront_cost_usd": costs.upfront_cost_usd,
                        },
                    ),
                ]
            )
        except MetricInputError as exc:
            omitted["cost.marginal_mean_usd"] = str(exc)
            omitted["cost.amortized_mean_usd"] = str(exc)

        try:
            retrieval = retrieval_context_metrics(ordered_examples, ordered_predictions)
            metadata = {"evaluated_examples": retrieval.evaluated_examples, "aggregation": "macro"}
            metrics.extend(
                [
                    MetricValue(
                        "retrieval.context_precision",
                        retrieval.context_precision,
                        metadata,
                    ),
                    MetricValue(
                        "retrieval.context_recall",
                        retrieval.context_recall,
                        metadata,
                    ),
                ]
            )
        except MetricInputError as exc:
            omitted["retrieval.context_precision"] = str(exc)
            omitted["retrieval.context_recall"] = str(exc)

        for supporting in supporting_metrics:
            metrics.append(MetricValue(supporting.name, supporting.value, supporting.metadata))

        if len({metric.name for metric in metrics}) != len(metrics):
            raise ValueError("metric names must be unique within an evaluation result")

        failures: list[FailureRecord] = []
        for example, prediction in zip(ordered_examples, ordered_predictions, strict=True):
            for annotation in classify_failures(
                example,
                prediction,
                label_to_category=self.label_to_category,
            ):
                failures.append(FailureRecord(example.incident_id, annotation))

        payload = _result_payload(
            evaluator_version=self.evaluator_version,
            failure_taxonomy_version=FAILURE_TAXONOMY_VERSION,
            prediction_count=len(ordered_predictions),
            metrics=metrics,
            reliability=reliability,
            failures=failures,
            omitted_metrics=omitted,
        )
        return EvaluationResult(
            evaluator_version=self.evaluator_version,
            failure_taxonomy_version=FAILURE_TAXONOMY_VERSION,
            prediction_count=len(ordered_predictions),
            metrics=tuple(sorted(metrics, key=lambda item: item.name)),
            reliability=reliability,
            failures=tuple(
                sorted(
                    failures,
                    key=lambda item: (item.incident_id, item.annotation.failure_code.value),
                )
            ),
            omitted_metrics=omitted,
            result_hash=_result_hash(payload),
        )


class EvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: dict[str, EvaluationHarness] = {}

    def register(self, name: str, evaluator: EvaluationHarness) -> None:
        if not name:
            raise ValueError("evaluator registry name is required")
        if name in self._evaluators:
            raise ValueError(f"evaluator already registered: {name}")
        self._evaluators[name] = evaluator

    def evaluate(
        self,
        name: str,
        examples: Sequence[EvaluationExample],
        predictions: Sequence[Prediction],
        **kwargs: object,
    ) -> EvaluationResult:
        try:
            evaluator = self._evaluators[name]
        except KeyError as exc:
            raise KeyError(f"unknown evaluator: {name}") from exc
        upfront = float(kwargs.get("upfront_cost_usd", 0.0))
        supporting = kwargs.get("supporting_metrics", ())
        if not isinstance(supporting, (list, tuple)):
            raise TypeError("supporting_metrics must be a sequence")
        return evaluator.evaluate(
            examples,
            predictions,
            upfront_cost_usd=upfront,
            supporting_metrics=tuple(supporting),  # type: ignore[arg-type]
        )
