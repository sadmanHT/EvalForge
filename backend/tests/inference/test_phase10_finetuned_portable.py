from __future__ import annotations

from pathlib import Path

import pytest

from app.data.io import read_jsonl
from app.data.schemas import IncidentRecord
from app.inference.base_model import BackendOutput, GenerationConfig, RuntimeConfig
from app.inference.finetuned_portable import evaluate_finetuned_records
from app.inference.finetuned_protocol import load_phase10_protocol
from app.inference.protocol import load_taxonomy
from app.training.inference import FineTunedAdapterPipeline

ROOT = Path(__file__).resolve().parents[3]


class FixturePortableBackend:
    def __init__(self) -> None:
        protocol = load_phase10_protocol(ROOT)
        self._runtime_config = protocol.runtime_config
        self.adapter_id = protocol.candidate_adapter_artifact_reference
        self.adapter_revision = protocol.candidate_adapter_sha256

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: tuple[str, ...],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        del prompt, generation_config
        label = allowed_labels[0]
        scores = {candidate: (-0.1 if candidate == label else -4.0) for candidate in allowed_labels}
        return BackendOutput(
            raw_text=f'{{"root_cause_code":"{label}","reasoning":"fixture"}}',
            label_log_likelihoods=scores,
            input_tokens=100,
            output_tokens=12,
            latency_ms=20.0,
            cost_usd=0.001,
            runtime_metadata={
                "adapter_id": self.adapter_id,
                "adapter_revision": self.adapter_revision,
            },
        )


def _records() -> list[IncidentRecord]:
    protocol = load_phase10_protocol(ROOT)
    path = (
        ROOT
        / "datasets/incident_diagnosis/processed"
        / protocol.dataset_version
        / "incidents.jsonl"
    )
    return read_jsonl(path, IncidentRecord)


def _pipeline() -> FineTunedAdapterPipeline:
    protocol = load_phase10_protocol(ROOT)
    labels, _categories = load_taxonomy(ROOT)
    return FineTunedAdapterPipeline(
        backend=FixturePortableBackend(),
        allowed_labels=labels,
        prompt_version=protocol.prompt_version,
        generation_config=protocol.generation_config,
    )


def test_portable_validation_uses_canonical_phase10_evaluator() -> None:
    protocol = load_phase10_protocol(ROOT)
    evaluated = evaluate_finetuned_records(
        root=ROOT,
        protocol=protocol,
        pipeline=_pipeline(),
        records=_records(),
        split="validation",
    )

    assert evaluated.split == "validation"
    assert len(evaluated.predictions) == 6
    assert evaluated.result.prediction_count == 6
    assert evaluated.inference_failure_count == 0
    assert "primary.exact_accuracy" in evaluated.result.metric_map()
    assert all(
        prediction.pipeline_metadata["pipeline_type"] == "FINETUNED"
        for prediction in evaluated.predictions
    )


def test_portable_runner_cannot_consume_locked_test_before_freeze() -> None:
    protocol = load_phase10_protocol(ROOT)
    assert protocol.locked_test_authorized is False
    with pytest.raises(ValueError, match="locked test"):
        evaluate_finetuned_records(
            root=ROOT,
            protocol=protocol,
            pipeline=_pipeline(),
            records=_records(),
            split="test",
        )
