from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TrackingEvidence:
    provider: str
    configured: bool
    run_reference: str | None = None
    artifact_reference: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "run_reference": self.run_reference,
            "artifact_reference": self.artifact_reference,
        }


class ExperimentTracker(Protocol):
    def log_baseline(
        self,
        *,
        run_id: str,
        protocol_payload: Mapping[str, object],
        summary: Mapping[str, object],
    ) -> TrackingEvidence: ...


class DisabledTracker:
    def log_baseline(
        self,
        *,
        run_id: str,
        protocol_payload: Mapping[str, object],
        summary: Mapping[str, object],
    ) -> TrackingEvidence:
        del run_id, protocol_payload, summary
        return TrackingEvidence(provider="wandb", configured=False)


class WandbTracker:
    """Optional W&B mirror; PostgreSQL remains the experiment system of record."""

    def __init__(self, *, project: str, entity: str | None = None) -> None:
        if not project.strip():
            raise ValueError("W&B project must be non-empty")
        self.project = project.strip()
        self.entity = entity.strip() if entity and entity.strip() else None

    def log_baseline(
        self,
        *,
        run_id: str,
        protocol_payload: Mapping[str, object],
        summary: Mapping[str, object],
    ) -> TrackingEvidence:
        try:
            wandb: Any = importlib.import_module("wandb")
        except ImportError as exc:
            message = "WANDB_PROJECT is configured but the wandb package is unavailable"
            raise RuntimeError(message) from exc

        run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_id,
            job_type="phase6-zero-shot-baseline",
            config=dict(protocol_payload),
            reinit=True,
        )
        try:
            raw_metrics = summary.get("metric_values", {})
            metrics: dict[str, float] = {}
            if isinstance(raw_metrics, Mapping):
                metrics = {
                    f"metrics/{str(name)}": float(value)
                    for name, value in raw_metrics.items()
                    if isinstance(value, int | float)
                }
            if metrics:
                run.log(metrics)
            artifact = wandb.Artifact(
                name=f"{run_id}-result",
                type="evaluation-result",
                metadata=dict(summary),
            )
            logged_artifact = run.log_artifact(artifact)
            run_reference = getattr(run, "url", None) or getattr(run, "id", None)
            artifact_reference = getattr(logged_artifact, "url", None) or getattr(
                logged_artifact, "name", None
            )
            return TrackingEvidence(
                provider="wandb",
                configured=True,
                run_reference=str(run_reference) if run_reference else None,
                artifact_reference=str(artifact_reference) if artifact_reference else None,
            )
        finally:
            run.finish()


def build_tracker_from_env() -> ExperimentTracker:
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not project:
        return DisabledTracker()
    return WandbTracker(project=project, entity=os.environ.get("WANDB_ENTITY"))
