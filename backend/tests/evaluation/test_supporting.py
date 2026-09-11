from __future__ import annotations

import pytest

from app.evaluation.graders.supporting import RagasSupportingAdapter
from app.evaluation.harness import EvaluationHarness, EvaluatorRegistry


def test_ragas_adapter_is_namespaced_supporting_only() -> None:
    adapter = RagasSupportingAdapter(evaluator_version="ragas-test-v1")
    metrics = adapter.adapt({"faithfulness": 0.8, "context_precision": 0.7})
    assert [metric.name for metric in metrics] == [
        "supporting.ragas.context_precision",
        "supporting.ragas.faithfulness",
    ]
    assert all(metric.metadata["role"] == "supporting_only" for metric in metrics)


def test_evaluator_registry_is_pipeline_agnostic(
    golden_payload: dict[str, object],
    golden_examples: list[object],
    golden_predictions: list[object],
) -> None:
    harness = EvaluationHarness(
        evaluator_version="registry-v1",
        allowed_labels=golden_payload["allowed_labels"],  # type: ignore[arg-type]
        label_to_category=golden_payload["label_to_category"],  # type: ignore[arg-type]
        ece_bins=5,
    )
    registry = EvaluatorRegistry()
    registry.register("common", harness)
    result = registry.evaluate(
        "common",
        golden_examples,  # type: ignore[arg-type]
        golden_predictions,  # type: ignore[arg-type]
        upfront_cost_usd=0.1,
    )
    assert result.metric_map()["primary.exact_accuracy"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("common", harness)
    with pytest.raises(KeyError, match="unknown evaluator"):
        registry.evaluate("missing", [], [])
