from __future__ import annotations

import pytest

from app.inference.base_model import BackendOutput, GenerationConfig, RuntimeConfig
from app.inference.costing import HourlyRateCostBackend


class FixtureBackend:
    def __init__(self, *, reported_cost: float | None = None) -> None:
        self._runtime_config = RuntimeConfig(model_id="fixture/model", revision="rev-1", seed=7)
        self.reported_cost = reported_cost

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
        return BackendOutput(
            raw_text='{"root_cause_code":"no_fault","reasoning":"fixture"}',
            label_log_likelihoods={label: -0.1 for label in allowed_labels},
            input_tokens=90,
            output_tokens=10,
            latency_ms=1800.0,
            cost_usd=self.reported_cost,
            runtime_metadata={"fixture": True},
        )


def test_hourly_rate_backend_records_marginal_wall_clock_cost() -> None:
    backend = HourlyRateCostBackend(FixtureBackend(), gpu_hour_usd=2.0)
    output = backend.generate(
        prompt="incident",
        allowed_labels=("no_fault",),
        generation_config=GenerationConfig(),
    )
    assert output.cost_usd == pytest.approx(0.001)
    assert output.runtime_metadata is not None
    assert output.runtime_metadata["cost_method"] == "active_inference_wall_time_x_gpu_hour_rate"
    assert output.runtime_metadata["gpu_hour_usd"] == 2.0


def test_hourly_rate_backend_rejects_ambiguous_or_invalid_costing() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        HourlyRateCostBackend(FixtureBackend(), gpu_hour_usd=float("nan"))
    backend = HourlyRateCostBackend(FixtureBackend(reported_cost=0.25), gpu_hour_usd=2.0)
    with pytest.raises(RuntimeError, match="already reports cost"):
        backend.generate(
            prompt="incident",
            allowed_labels=("no_fault",),
            generation_config=GenerationConfig(),
        )
