from __future__ import annotations

import math

import pytest

from app.evaluation.contracts import ParseStatus
from app.inference.base_model import (
    BackendOutput,
    GenerationConfig,
    IncidentInput,
    InferenceError,
    QuantizationMode,
    RuntimeConfig,
    ZeroShotBaselineAdapter,
    _oom_like,
)
from app.inference.prompts import (
    BASELINE_PROMPT_VERSION,
    OUTPUT_SCHEMA_VERSION,
    get_prompt_template,
    parse_baseline_output,
)

LABELS = (
    "n_plus_one_query",
    "database_connection_leak",
    "disk_exhaustion",
    "memory_leak",
    "broken_payment_configuration",
    "no_fault",
)


class FakeBackend:
    def __init__(self, *, raw_text: str, scores: dict[str, float]) -> None:
        self._runtime_config = RuntimeConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            revision="e8737b84b4470b28db3a0be719b362b1bd39a14d",
            seed=20260908,
            quantization=QuantizationMode.BITSANDBYTES_4BIT_NF4,
        )
        self.raw_text = raw_text
        self.scores = scores
        self.prompts: list[str] = []

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
        assert allowed_labels == LABELS
        assert generation_config.temperature == 0.0
        self.prompts.append(prompt)
        return BackendOutput(
            raw_text=self.raw_text,
            label_log_likelihoods=self.scores,
            input_tokens=120,
            output_tokens=14,
            latency_ms=25.5,
            cost_usd=0.002,
            runtime_metadata={"fixture": "phase6-ci"},
        )


def _scores(winner: str) -> dict[str, float]:
    return {label: (-0.1 if label == winner else -4.0) for label in LABELS}


def test_prompt_registry_renders_frozen_labels_and_schema() -> None:
    prompt = get_prompt_template(BASELINE_PROMPT_VERSION).render(
        title="Checkout latency regression",
        description="Database query count rose after an application change.",
        allowed_labels=LABELS,
    )
    assert OUTPUT_SCHEMA_VERSION == "root-cause-prediction-v1"
    assert '"root_cause_code"' in prompt
    for label in LABELS:
        assert label in prompt
    assert "retrieval citations" in prompt


def test_prompt_requires_unique_nonempty_labels() -> None:
    template = get_prompt_template(BASELINE_PROMPT_VERSION)
    with pytest.raises(ValueError, match="non-empty unique"):
        template.render(title="x", description="y", allowed_labels=())
    with pytest.raises(ValueError, match="non-empty unique"):
        template.render(title="x", description="y", allowed_labels=("a", "a"))


def test_parser_accepts_valid_json_and_single_json_fence() -> None:
    valid = '{"root_cause_code":"memory_leak","reasoning":"RSS rises steadily."}'
    direct = parse_baseline_output(incident_id="inc-1", raw_output=valid, allowed_labels=LABELS)
    fenced = parse_baseline_output(
        incident_id="inc-2",
        raw_output=f"```json\n{valid}\n```",
        allowed_labels=LABELS,
    )
    assert direct.parse_status is ParseStatus.OK
    assert fenced.parse_status is ParseStatus.OK
    assert fenced.raw_model_output.startswith("```json")
    assert fenced.pipeline_metadata["parser_normalization"] == "single_json_markdown_fence_removed"


def test_parser_records_invalid_truncated_and_unknown_outputs() -> None:
    invalid = parse_baseline_output(
        incident_id="inc-invalid", raw_output="not json", allowed_labels=LABELS
    )
    truncated = parse_baseline_output(
        incident_id="inc-truncated",
        raw_output='{"root_cause_code":"memory_leak"',
        allowed_labels=LABELS,
    )
    unknown = parse_baseline_output(
        incident_id="inc-unknown",
        raw_output='{"root_cause_code":"made_up"}',
        allowed_labels=LABELS,
    )
    assert invalid.parse_status is ParseStatus.INVALID_JSON
    assert truncated.parse_status is ParseStatus.INVALID_JSON
    assert unknown.parse_status is ParseStatus.UNKNOWN_LABEL


def test_generation_config_rejects_sampling_for_primary_baseline() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        GenerationConfig(do_sample=True)
    with pytest.raises(ValueError, match="temperature=0"):
        GenerationConfig(temperature=0.1)


def test_adapter_adds_probability_token_latency_and_cost_evidence() -> None:
    backend = FakeBackend(
        raw_text='{"root_cause_code":"n_plus_one_query","reasoning":"query fanout"}',
        scores=_scores("n_plus_one_query"),
    )
    adapter = ZeroShotBaselineAdapter(backend=backend, allowed_labels=LABELS)
    prediction = adapter.predict(
        IncidentInput(
            incident_id="inc-1",
            title="High query count",
            description="Request query count grew 18x while CPU stayed stable.",
        )
    )
    assert prediction.parse_status is ParseStatus.OK
    assert prediction.predicted_root_cause_code == "n_plus_one_query"
    assert prediction.input_tokens == 120
    assert prediction.output_tokens == 14
    assert prediction.total_tokens == 134
    assert prediction.latency_ms == 25.5
    assert prediction.cost_usd == 0.002
    assert prediction.label_probabilities is not None
    assert math.isclose(sum(prediction.label_probabilities.values()), 1.0, abs_tol=1e-12)
    assert prediction.confidence_probability == prediction.label_probabilities["n_plus_one_query"]
    assert prediction.ranked_labels[0].label == "n_plus_one_query"
    assert prediction.pipeline_metadata["prompt_version"] == BASELINE_PROMPT_VERSION
    assert "High query count" in backend.prompts[0]


def test_adapter_does_not_fabricate_ranking_when_generation_disagrees_with_score_argmax() -> None:
    backend = FakeBackend(
        raw_text='{"root_cause_code":"memory_leak","reasoning":"generated choice"}',
        scores=_scores("disk_exhaustion"),
    )
    prediction = ZeroShotBaselineAdapter(backend=backend, allowed_labels=LABELS).predict(
        IncidentInput(incident_id="inc-2", title="Memory", description="RSS climbs over time")
    )
    assert prediction.predicted_root_cause_code == "memory_leak"
    assert prediction.ranked_labels == ()
    assert prediction.pipeline_metadata["ranked_labels_omitted"] == (
        "generated_label_disagrees_with_score_argmax"
    )
    assert prediction.label_probabilities is not None
    assert prediction.confidence_probability == prediction.label_probabilities["memory_leak"]


def test_adapter_rejects_incomplete_label_score_set_and_duplicate_batch_ids() -> None:
    backend = FakeBackend(
        raw_text='{"root_cause_code":"no_fault","reasoning":"control"}',
        scores={"no_fault": -0.1},
    )
    adapter = ZeroShotBaselineAdapter(backend=backend, allowed_labels=LABELS)
    incident = IncidentInput(incident_id="same", title="Control", description="No active fault")
    with pytest.raises(InferenceError, match="exactly cover"):
        adapter.predict(incident)

    complete = ZeroShotBaselineAdapter(
        backend=FakeBackend(
            raw_text='{"root_cause_code":"no_fault"}', scores=_scores("no_fault")
        ),
        allowed_labels=LABELS,
    )
    with pytest.raises(ValueError, match="duplicate incident_id"):
        complete.predict_batch((incident, incident))


def test_oom_detection_is_explicit() -> None:
    assert _oom_like(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not _oom_like(RuntimeError("shape mismatch"))
