from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.data.schemas import IncidentRecord, Split
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationResult
from app.inference.base_model import IncidentInput
from app.inference.finetuned_protocol import Phase10Protocol
from app.inference.finetuned_runner import (
    _build_harness,
    _predict_with_retry,
    _validate_pipeline_contract,
)
from app.inference.protocol import load_taxonomy
from app.training.inference import FineTunedAdapterPipeline


@dataclass(frozen=True)
class PortableFineTunedEvaluation:
    split: str
    predictions: tuple[Prediction, ...]
    result: EvaluationResult
    inference_failure_count: int


def evaluate_finetuned_records(
    *,
    root: Path,
    protocol: Phase10Protocol,
    pipeline: FineTunedAdapterPipeline,
    records: Sequence[IncidentRecord],
    split: str,
    max_attempts: int = 2,
    upfront_cost_usd: float = 0.0,
) -> PortableFineTunedEvaluation:
    """Evaluate the Phase 10 adapter without requiring the persistence stack.

    This path intentionally reuses the canonical Phase 10 pipeline-contract validation,
    retry semantics, and common evaluator. It exists for connected GPU environments such
    as Kaggle where the real PEFT model can run but PostgreSQL is not part of the runtime.
    """

    evaluation_split = protocol.assert_split_allowed(split)
    labels, categories = load_taxonomy(root)
    _validate_pipeline_contract(
        pipeline=pipeline,
        protocol=protocol,
        allowed_labels=labels,
    )
    harness = _build_harness(protocol=protocol, labels=labels, categories=categories)

    expected_split = Split(evaluation_split)
    selected = sorted(
        (record for record in records if record.split is expected_split),
        key=lambda record: record.incident_id,
    )
    if not selected:
        raise ValueError(f"no incidents found for split: {evaluation_split}")
    incident_ids = [record.incident_id for record in selected]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("portable Phase 10 evaluation requires unique incident IDs")

    predictions: list[Prediction] = []
    examples: list[EvaluationExample] = []
    inference_failure_count = 0
    for record in selected:
        prediction, failed = _predict_with_retry(
            pipeline=pipeline,
            incident=IncidentInput(
                incident_id=record.incident_id,
                title=record.title,
                description=record.description,
            ),
            protocol=protocol,
            allowed_labels=labels,
            max_attempts=max_attempts,
        )
        predictions.append(prediction)
        inference_failure_count += int(failed)
        examples.append(
            EvaluationExample(
                incident_id=record.incident_id,
                split=evaluation_split,
                root_cause_code=record.root_cause_code,
                root_cause_category=record.root_cause_category,
            )
        )

    result = harness.evaluate(
        examples,
        predictions,
        upfront_cost_usd=upfront_cost_usd,
    )
    return PortableFineTunedEvaluation(
        split=evaluation_split,
        predictions=tuple(predictions),
        result=result,
        inference_failure_count=inference_failure_count,
    )
