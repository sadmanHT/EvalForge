from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.engine import Engine

from app.core.experiment_config import ExperimentConfig, PipelineType
from app.db import build_engine, session_scope
from app.inference.protocol import load_baseline_protocol
from app.models import AdapterVersion, Artifact, Job
from app.repositories import PersistenceRepository
from app.services.dataset_import import import_phase3_dataset
from app.training.config import TrainingConfigBundle, load_training_config
from app.training.formatter import PreparedDatasetManifest, prepare_training_dataset
from app.training.runtime import PeftTrainingRuntime
from app.training.smoke import SmokeAdapter, run_smoke_training
from app.training.tracking import build_training_tracker_from_env


@dataclass(frozen=True)
class TrainingArtifactResult:
    adapter_id: str
    adapter_revision: str
    adapter_uri: str
    adapter_sha256: str
    size_bytes: int | None
    selected_checkpoint: str | None
    scientific_adapter: bool
    metadata: dict[str, object]


class TrainingExecutor(Protocol):
    def execute(
        self,
        *,
        payload: dict[str, Any],
        prepared_dir: Path,
        bundle: TrainingConfigBundle,
    ) -> TrainingArtifactResult: ...


class ConnectedPeftTrainingExecutor:
    """Real CUDA/W&B Phase 09 executor used by the production worker."""

    def __init__(self, *, root: Path, output_root: Path) -> None:
        self.root = root
        self.output_root = output_root

    def execute(
        self,
        *,
        payload: dict[str, Any],
        prepared_dir: Path,
        bundle: TrainingConfigBundle,
    ) -> TrainingArtifactResult:
        if not os.environ.get("WANDB_PROJECT", "").strip():
            raise RuntimeError("real Phase 09 worker training requires WANDB_PROJECT")

        job_id = _required_text(payload, "job_id")
        run_id = str(payload.get("training_run_id") or job_id)
        output_dir = self.output_root / job_id
        raw_resume = payload.get("resume_from_checkpoint")
        if raw_resume is not None and (not isinstance(raw_resume, str) or not raw_resume.strip()):
            raise ValueError("resume_from_checkpoint must be a non-empty path when provided")
        resume = Path(raw_resume).resolve() if isinstance(raw_resume, str) else None

        prepared = PreparedDatasetManifest.model_validate_json(
            (prepared_dir / "manifest.json").read_text(encoding="utf-8")
        )
        result = PeftTrainingRuntime(bundle).train(
            train_path=prepared_dir / "train.jsonl",
            validation_path=prepared_dir / "validation.jsonl",
            output_dir=output_dir,
            resume_from_checkpoint=resume,
            report_to_wandb=False,
        )
        reproducibility_metadata: dict[str, object] = {
            "phase": 9,
            "training_config_hash": bundle.config_hash(),
            "dataset_version": prepared.dataset_version,
            "dataset_manifest_checksum": prepared.dataset_manifest_checksum,
            "dataset_content_checksum": prepared.dataset_content_checksum,
            "prepared_train_sha256": prepared.train_sha256,
            "prepared_validation_sha256": prepared.validation_sha256,
            "lineage_sha256": prepared.lineage_sha256,
            "base_model_id": bundle.lora.base_model_id,
            "base_model_revision": bundle.lora.base_model_revision,
            "seed": bundle.training.seed,
            "git_commit": _required_text(payload, "git_commit"),
            "hardware_runtime_descriptor": _required_text(payload, "hardware_runtime_descriptor"),
            "resumed_from_checkpoint": str(resume) if resume is not None else None,
        }
        tracking = build_training_tracker_from_env().log_training(
            run_id=run_id,
            config=bundle.canonical_payload(),
            reproducibility_metadata=reproducibility_metadata,
            result=result,
        )
        if (
            not tracking.configured
            or tracking.run_reference is None
            or tracking.artifact_reference is None
        ):
            raise RuntimeError("real Phase 09 worker training requires durable W&B references")

        adapter_dir = Path(result.adapter_dir).resolve()
        size_bytes = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())
        adapter_id = str(payload.get("adapter_id") or "evalforge/mistral-7b-phase9-qlora")
        revision = f"qlora-{result.adapter_sha256[:16]}"
        return TrainingArtifactResult(
            adapter_id=adapter_id,
            adapter_revision=revision,
            adapter_uri=adapter_dir.as_uri(),
            adapter_sha256=result.adapter_sha256,
            size_bytes=size_bytes,
            selected_checkpoint=result.selected_checkpoint,
            scientific_adapter=True,
            metadata={
                **reproducibility_metadata,
                "checkpoint_paths": list(result.checkpoint_paths),
                "train_metrics": result.train_metrics,
                "tracking": tracking.as_dict(),
            },
        )


class SmokeTrainingExecutor:
    """CPU-only worker smoke. Its artifact is explicitly non-scientific."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def execute(
        self,
        *,
        payload: dict[str, Any],
        prepared_dir: Path,
        bundle: TrainingConfigBundle,
    ) -> TrainingArtifactResult:
        del prepared_dir, bundle
        job_id = _required_text(payload, "job_id")
        output_dir = self.output_root / job_id
        partial = run_smoke_training(output_dir / "partial", total_steps=3)
        resumed = run_smoke_training(
            output_dir / "resumed",
            total_steps=6,
            resume_from_checkpoint=Path(partial.checkpoint_path),
        )
        reloaded = SmokeAdapter.load(Path(resumed.adapter_path))
        if reloaded.step != 6:
            raise RuntimeError("smoke adapter reload did not preserve resumed step")
        adapter_id = str(payload.get("adapter_id") or "evalforge/phase9-smoke-adapter")
        revision = f"smoke-{resumed.adapter_sha256[:16]}"
        adapter_path = Path(resumed.adapter_path)
        return TrainingArtifactResult(
            adapter_id=adapter_id,
            adapter_revision=revision,
            adapter_uri=adapter_path.resolve().as_uri(),
            adapter_sha256=resumed.adapter_sha256,
            size_bytes=adapter_path.stat().st_size,
            selected_checkpoint=resumed.checkpoint_path,
            scientific_adapter=False,
            metadata={
                "smoke_contract": "phase9-cpu-training-smoke-v1",
                "steps_completed": resumed.steps_completed,
                "resume_verified": True,
                "reload_verified": True,
            },
        )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Phase 09 training job requires non-empty {key}")
    return value.strip()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _dependency_lock_checksum(root: Path) -> str:
    return hashlib.sha256((root / "backend/requirements.full.lock").read_bytes()).hexdigest()


def _dataset_manifest(root: Path, dataset_version: str) -> dict[str, Any]:
    path = root / "datasets/incident_diagnosis/processed" / dataset_version / "manifest.json"
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("dataset manifest must be a JSON object")
    payload: dict[str, Any] = {str(key): value for key, value in raw_payload.items()}
    if payload.get("dataset_version") != dataset_version:
        raise ValueError("dataset manifest/version mismatch")
    return payload


def _registration_experiment_config(
    *,
    root: Path,
    payload: dict[str, Any],
    bundle: TrainingConfigBundle,
    prepared: PreparedDatasetManifest,
    artifact: TrainingArtifactResult,
) -> ExperimentConfig:
    protocol = load_baseline_protocol(root)
    dataset = _dataset_manifest(root, prepared.dataset_version)
    return ExperimentConfig(
        study_id=protocol.study_id,
        pipeline_type=PipelineType.FINETUNED,
        dataset_version=prepared.dataset_version,
        test_split_manifest_checksum=str(dataset["manifest_checksum"]),
        label_taxonomy_version=str(dataset["label_taxonomy_version"]),
        base_model_id=bundle.lora.base_model_id,
        base_model_revision=bundle.lora.base_model_revision,
        adapter_id=artifact.adapter_id,
        adapter_revision=artifact.adapter_revision,
        prompt_version=protocol.prompt_version,
        output_schema_version=protocol.output_schema_version,
        generation_config=protocol.generation_config.model_dump(mode="json"),
        temperature=protocol.generation_config.temperature,
        confidence_method=protocol.confidence_method,
        seed=bundle.training.seed,
        evaluator_version=protocol.evaluator_version,
        git_commit=_required_text(payload, "git_commit"),
        dependency_lock_checksum=_dependency_lock_checksum(root),
        hardware_runtime_descriptor=_required_text(payload, "hardware_runtime_descriptor"),
        cost_rate_snapshot_version=_required_text(payload, "cost_rate_snapshot_version"),
    )


class Phase9TrainingJobHandler:
    """Queue-worker boundary for real training jobs and explicit CPU smoke tests."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        engine: Engine | None = None,
        executor: TrainingExecutor | None = None,
        work_root: Path | None = None,
    ) -> None:
        self.root = (root or Path(os.environ.get("EVALFORGE_ROOT", "."))).resolve()
        self.engine = engine or build_engine()
        self.work_root = (work_root or self.root / ".phase9-work").resolve()
        self.executor = executor or ConnectedPeftTrainingExecutor(
            root=self.root,
            output_root=self.work_root / "training",
        )

    def __call__(self, payload: dict[str, Any]) -> dict[str, object]:
        job_id = _required_text(payload, "job_id")
        dataset_version = _required_text(payload, "dataset_version")
        bundle = load_training_config(self.root)
        prepared_dir = self.work_root / job_id / "prepared"
        prepared = prepare_training_dataset(
            root=self.root,
            output_dir=prepared_dir,
            dataset_version=dataset_version,
            include_reasoning=bundle.training.include_reasoning_target,
        )

        with session_scope(self.engine) as session:
            import_phase3_dataset(session, self.root, dataset_version)
            repo = PersistenceRepository(session)
            repo.create_job(
                job_id,
                "phase9_training",
                idempotency_key=str(payload.get("idempotency_key") or job_id),
                payload=payload,
            )

        try:
            with session_scope(self.engine) as session:
                job = session.get(Job, job_id)
                if job is None:
                    raise RuntimeError("Phase 09 job disappeared before execution")
                job.status = "running"

            artifact = self.executor.execute(
                payload=payload,
                prepared_dir=prepared_dir,
                bundle=bundle,
            )
            experiment_id = str(payload.get("experiment_id") or f"exp-{job_id}")
            with session_scope(self.engine) as session:
                repo = PersistenceRepository(session)
                model = repo.ensure_model_version(
                    bundle.lora.base_model_id,
                    bundle.lora.base_model_revision,
                )
                adapter = session.scalar(
                    select(AdapterVersion).where(
                        AdapterVersion.adapter_id == artifact.adapter_id,
                        AdapterVersion.revision == artifact.adapter_revision,
                    )
                )
                if adapter is None:
                    adapter = AdapterVersion(
                        adapter_version_id=_stable_id(
                            "adapter",
                            artifact.adapter_id,
                            artifact.adapter_revision,
                        ),
                        model_version_id=model.model_version_id,
                        adapter_id=artifact.adapter_id,
                        revision=artifact.adapter_revision,
                        metadata_json={
                            "phase": 9,
                            "training_config_hash": bundle.config_hash(),
                            "dataset_version": prepared.dataset_version,
                            "dataset_manifest_checksum": prepared.dataset_manifest_checksum,
                            "prepared_train_sha256": prepared.train_sha256,
                            "prepared_validation_sha256": prepared.validation_sha256,
                            "lineage_sha256": prepared.lineage_sha256,
                            "adapter_sha256": artifact.adapter_sha256,
                            "scientific_adapter": artifact.scientific_adapter,
                            **artifact.metadata,
                        },
                    )
                    session.add(adapter)
                    session.flush()
                config = _registration_experiment_config(
                    root=self.root,
                    payload=payload,
                    bundle=bundle,
                    prepared=prepared,
                    artifact=artifact,
                )
                repo.create_experiment(experiment_id, config, status="planned")
                artifact_id = _stable_id("artifact", experiment_id, artifact.adapter_sha256)
                stored_artifact = session.get(Artifact, artifact_id)
                if stored_artifact is None:
                    session.add(
                        Artifact(
                            artifact_id=artifact_id,
                            experiment_id=experiment_id,
                            kind="peft_adapter",
                            uri=artifact.adapter_uri,
                            sha256=artifact.adapter_sha256,
                            size_bytes=artifact.size_bytes,
                            artifact_metadata={
                                "phase": 9,
                                "adapter_id": artifact.adapter_id,
                                "adapter_revision": artifact.adapter_revision,
                                "base_model_id": bundle.lora.base_model_id,
                                "base_model_revision": bundle.lora.base_model_revision,
                                "training_config_hash": bundle.config_hash(),
                                "dataset_manifest_checksum": prepared.dataset_manifest_checksum,
                                "lineage_sha256": prepared.lineage_sha256,
                                "scientific_adapter": artifact.scientific_adapter,
                                **artifact.metadata,
                            },
                        )
                    )
                job = session.get(Job, job_id)
                if job is None:
                    raise RuntimeError("Phase 09 job disappeared before completion")
                job.experiment_id = experiment_id
                job.status = "completed"
                job.result_json = {
                    "adapter_id": artifact.adapter_id,
                    "adapter_revision": artifact.adapter_revision,
                    "adapter_sha256": artifact.adapter_sha256,
                    "artifact_id": artifact_id,
                    "experiment_id": experiment_id,
                    "training_config_hash": bundle.config_hash(),
                    "lineage_sha256": prepared.lineage_sha256,
                    "scientific_adapter": artifact.scientific_adapter,
                }
        except Exception as exc:
            with session_scope(self.engine) as session:
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
            raise

        return {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "adapter_id": artifact.adapter_id,
            "adapter_revision": artifact.adapter_revision,
            "adapter_sha256": artifact.adapter_sha256,
            "training_config_hash": bundle.config_hash(),
            "prepared_manifest": prepared.model_dump(mode="json"),
            "scientific_adapter": artifact.scientific_adapter,
            "executor_metadata": artifact.metadata,
        }
