from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SupportingMetric:
    name: str
    value: float
    metadata: dict[str, object]


class RagasSupportingAdapter:
    """Namespaced adapter for externally computed RAGAS/supporting scores.

    These values are deliberately kept outside the deterministic primary metric namespace.
    Phase 08 can supply real RAGAS outputs without changing the common evaluator contract.
    """

    def __init__(self, *, evaluator_version: str) -> None:
        if not evaluator_version:
            raise ValueError("supporting evaluator version is required")
        self.evaluator_version = evaluator_version

    def adapt(self, scores: Mapping[str, float]) -> tuple[SupportingMetric, ...]:
        metrics: list[SupportingMetric] = []
        for name in sorted(scores):
            value = float(scores[name])
            if not math.isfinite(value):
                raise ValueError("supporting metric values must be finite")
            metrics.append(
                SupportingMetric(
                    name=f"supporting.ragas.{name}",
                    value=value,
                    metadata={
                        "role": "supporting_only",
                        "evaluator_version": self.evaluator_version,
                    },
                )
            )
        return tuple(metrics)
