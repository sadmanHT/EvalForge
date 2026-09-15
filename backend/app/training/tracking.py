from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.training.runtime import TrainingRunResult


@dataclass(frozen=True)
class TrainingTrackingEvidence:
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


class TrainingTracker(Protocol):
    def log_training(
        self,
        *,
        run_id: str,
        config: dict[str, object],
        reproducibility_metadata: dict[str, object],
        result: TrainingRunResult,
    ) -> TrainingTrackingEvidence: ...


class DisabledTrainingTracker:
    def log_training(
        self,
        *,
        run_id: str,
        config: dict[str, object],
        reproducibility_metadata: dict[str, object],
        result: TrainingRunResult,
    ) -> TrainingTrackingEvidence:
        del run_id, config, reproducibility_metadata, result
        return TrainingTrackingEvidence(provider="wandb", configured=False)


class WandbTrainingTracker:
    def __init__(self, *, project: str, entity: str | None = None) -> None:
        if not project.strip():
            raise ValueError("W&B project must be non-empty")
        self.project = project.strip()
        self.entity = entity.strip() if entity and entity.strip() else None

    def log_training(
        self,
        *,
        run_id: str,
        config: dict[str, object],
        reproducibility_metadata: dict[str, object],
        result: TrainingRunResult,
    ) -> TrainingTrackingEvidence:
        try:
            wandb: Any = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "WANDB_PROJECT is configured but the wandb package is unavailable"
            ) from exc

        run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_id,
            job_type="phase9-finetune",
            config=config,
            reinit=True,
        )
        try:
            for entry in result.log_history:
                step = entry.get("step")
                metrics = {
                    f"training/{key}": float(value)
                    for key, value in entry.items()
                    if key != "step" and isinstance(value, int | float)
                }
                if metrics:
                    if isinstance(step, int | float):
                        run.log(metrics, step=int(step))
                    else:
                        run.log(metrics)
            run.summary.update(reproducibility_metadata)
            run.summary.update(
                {
                    "adapter_sha256": result.adapter_sha256,
                    "selected_checkpoint": result.selected_checkpoint,
                    **{f"final/{key}": value for key, value in result.train_metrics.items()},
                }
            )
            artifact = wandb.Artifact(
                name=f"{run_id}-adapter",
                type="phase9-peft-adapter",
                metadata={
                    **reproducibility_metadata,
                    "adapter_sha256": result.adapter_sha256,
                    "selected_checkpoint": result.selected_checkpoint,
                    "checkpoint_paths": list(result.checkpoint_paths),
                },
            )
            artifact.add_dir(result.adapter_dir, name="adapter")
            if result.selected_checkpoint:
                selected = Path(result.selected_checkpoint)
                if selected.is_dir():
                    artifact.add_dir(str(selected), name="selected-checkpoint")
            logged = run.log_artifact(artifact)
            run_reference = getattr(run, "url", None) or getattr(run, "id", None)
            artifact_reference = getattr(logged, "url", None) or getattr(logged, "name", None)
            return TrainingTrackingEvidence(
                provider="wandb",
                configured=True,
                run_reference=str(run_reference) if run_reference else None,
                artifact_reference=str(artifact_reference) if artifact_reference else None,
            )
        finally:
            run.finish()


def build_training_tracker_from_env() -> TrainingTracker:
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not project:
        return DisabledTrainingTracker()
    return WandbTrainingTracker(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
    )
