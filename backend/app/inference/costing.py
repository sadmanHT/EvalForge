from __future__ import annotations

import math
from collections.abc import Sequence

from app.inference.base_model import (
    BackendOutput,
    BaseModelBackend,
    GenerationConfig,
    InferenceError,
    RuntimeConfig,
)


class HourlyRateCostBackend:
    """Attach marginal GPU wall-clock cost to a backend that does not price itself."""

    def __init__(self, backend: BaseModelBackend, *, gpu_hour_usd: float) -> None:
        rate = float(gpu_hour_usd)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("gpu_hour_usd must be finite and non-negative")
        self.backend = backend
        self.gpu_hour_usd = rate

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self.backend.runtime_config

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: Sequence[str],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        output = self.backend.generate(
            prompt=prompt,
            allowed_labels=allowed_labels,
            generation_config=generation_config,
        )
        if output.cost_usd is not None:
            message = "hourly-rate costing cannot wrap a backend that already reports cost"
            raise InferenceError(message)
        if not math.isfinite(output.latency_ms) or output.latency_ms < 0:
            message = "backend latency must be finite and non-negative for cost accounting"
            raise InferenceError(message)
        cost_usd = output.latency_ms / 3_600_000.0 * self.gpu_hour_usd
        metadata = dict(output.runtime_metadata or {})
        metadata.update(
            {
                "cost_method": "active_inference_wall_time_x_gpu_hour_rate",
                "gpu_hour_usd": self.gpu_hour_usd,
            }
        )
        return BackendOutput(
            raw_text=output.raw_text,
            label_log_likelihoods=output.label_log_likelihoods,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=output.latency_ms,
            cost_usd=cost_usd,
            runtime_metadata=metadata,
        )
