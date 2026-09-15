from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.training.config import load_training_config
from app.training.evidence import sha256_file


class Phase9Stage(StrEnum):
    PRE_EXTERNAL_TRAINING = "pre_external_training"
    TRAINING_EVIDENCE_READY = "training_evidence_ready"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Phase9RepositoryStatus:
    stage: Phase9Stage
    training_config_hash: str
    adapter_sha256: str | None
    wandb_run_reference: str | None
    complete: bool
    blocker: str | None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return payload


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Phase 09 evidence requires non-empty {key}")
    return value.strip()


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 hex digest") from exc
    return value


def validate_training_run_evidence(
    payload: dict[str, Any],
    *,
    expected_config_hash: str,
    expected_model_id: str,
    expected_model_revision: str,
) -> tuple[str, str]:
    if payload.get("evidence_version") != "phase9-training-run-v1":
        raise ValueError("unexpected Phase 09 training evidence version")
    if payload.get("status") != "completed":
        raise ValueError("Phase 09 connected training did not complete")
    if payload.get("training_config_hash") != expected_config_hash:
        raise ValueError("training evidence config hash does not match repository config")
    if payload.get("base_model_id") != expected_model_id:
        raise ValueError("training evidence changed the frozen base model ID")
    if payload.get("base_model_revision") != expected_model_revision:
        raise ValueError("training evidence changed the frozen base model revision")
    _required_text(payload, "run_id")
    _required_text(payload, "dataset_version")
    _validate_sha256(payload.get("dataset_manifest_checksum"), "dataset_manifest_checksum")
    _validate_sha256(payload.get("dataset_content_checksum"), "dataset_content_checksum")
    _validate_sha256(payload.get("prepared_train_sha256"), "prepared_train_sha256")
    _validate_sha256(payload.get("prepared_validation_sha256"), "prepared_validation_sha256")
    _validate_sha256(payload.get("lineage_sha256"), "lineage_sha256")
    _required_text(payload, "git_commit")
    _required_text(payload, "hardware_runtime_descriptor")
    adapter_sha256 = _validate_sha256(payload.get("adapter_sha256"), "adapter_sha256")
    _required_text(payload, "selected_checkpoint")
    checkpoint_paths = payload.get("checkpoint_paths")
    if not isinstance(checkpoint_paths, list) or not checkpoint_paths:
        raise ValueError("Phase 09 completion evidence requires at least one saved checkpoint")
    for checkpoint in checkpoint_paths:
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("Phase 09 checkpoint_paths contains an invalid entry")

    environment = payload.get("gpu_environment")
    if not isinstance(environment, dict) or environment.get("cuda_available") is not True:
        raise ValueError("Phase 09 completion evidence requires a CUDA training runtime")
    _required_text(environment, "gpu_name")

    tracking = payload.get("tracking")
    if not isinstance(tracking, dict) or tracking.get("configured") is not True:
        raise ValueError("Phase 09 completion evidence requires configured W&B tracking")
    if tracking.get("provider") != "wandb":
        raise ValueError("Phase 09 completion evidence requires W&B as the tracking provider")
    run_reference = _required_text(tracking, "run_reference")
    _required_text(tracking, "artifact_reference")
    return adapter_sha256, run_reference


def _validate_pass_evidence(
    path: Path,
    *,
    evidence_version: str,
    adapter_sha256: str,
    training_config_hash: str,
    training_evidence_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("evidence_version") != evidence_version:
        raise ValueError(f"unexpected evidence version in {path.name}")
    if payload.get("status") != "pass":
        raise ValueError(f"{path.name} did not record PASS")
    if payload.get("adapter_sha256") != adapter_sha256:
        raise ValueError(f"{path.name} adapter checksum does not match training run")
    if payload.get("training_config_hash") != training_config_hash:
        raise ValueError(f"{path.name} training config hash does not match")
    if payload.get("training_evidence_sha256") != training_evidence_sha256:
        raise ValueError(f"{path.name} is not linked to the current training-run.json")
    return payload


def _validate_reload_evidence(
    payload: dict[str, Any],
    *,
    expected_model_id: str,
    expected_model_revision: str,
    expected_dataset_version: str,
) -> None:
    if payload.get("validation_split") != "validation":
        raise ValueError("adapter-reload.json must use validation-only inference")
    if payload.get("parse_status") != "OK":
        raise ValueError("adapter-reload.json did not record schema-compatible inference")
    if payload.get("pipeline_type") != "FINETUNED":
        raise ValueError("adapter-reload.json did not use the FINETUNED pipeline contract")
    if payload.get("base_model_id") != expected_model_id:
        raise ValueError("adapter-reload.json changed the frozen base model ID")
    if payload.get("base_model_revision") != expected_model_revision:
        raise ValueError("adapter-reload.json changed the frozen base model revision")
    if payload.get("dataset_version") != expected_dataset_version:
        raise ValueError("adapter-reload.json dataset version does not match training run")
    _required_text(payload, "validation_incident_id")
    _required_text(payload, "adapter_id")
    _required_text(payload, "adapter_revision")


def _validate_resume_evidence(
    payload: dict[str, Any],
    *,
    expected_source_run_id: str,
    expected_source_wandb_run_reference: str,
    expected_dataset_version: str,
    expected_dataset_manifest_checksum: str,
) -> None:
    _required_text(payload, "resume_from_checkpoint")
    if payload.get("source_run_id") != expected_source_run_id:
        raise ValueError("resume.json source run does not match training-run.json")
    if payload.get("source_wandb_run_reference") != expected_source_wandb_run_reference:
        raise ValueError("resume.json source W&B run does not match training-run.json")
    _required_text(payload, "resumed_run_id")
    _required_text(payload, "resumed_wandb_run_reference")
    _required_text(payload, "resumed_wandb_artifact_reference")
    _validate_sha256(payload.get("resumed_adapter_sha256"), "resumed_adapter_sha256")
    _validate_sha256(
        payload.get("resumed_training_evidence_sha256"),
        "resumed_training_evidence_sha256",
    )
    if payload.get("dataset_version") != expected_dataset_version:
        raise ValueError("resume.json dataset version does not match training run")
    if payload.get("dataset_manifest_checksum") != expected_dataset_manifest_checksum:
        raise ValueError("resume.json dataset manifest does not match training run")


def _validate_export_evidence(
    payload: dict[str, Any],
    *,
    adapter_sha256: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_dataset_manifest_checksum: str,
) -> None:
    if payload.get("exported_adapter_sha256") != adapter_sha256:
        raise ValueError("adapter-export.json copied adapter checksum does not match training run")
    _validate_sha256(payload.get("export_tree_sha256"), "export_tree_sha256")
    _required_text(payload, "destination")
    _required_text(payload, "manifest_path")
    if payload.get("dataset_manifest_checksum") != expected_dataset_manifest_checksum:
        raise ValueError("adapter-export.json dataset manifest does not match training run")
    if payload.get("base_model_id") != expected_model_id:
        raise ValueError("adapter-export.json changed the frozen base model ID")
    if payload.get("base_model_revision") != expected_model_revision:
        raise ValueError("adapter-export.json changed the frozen base model revision")
    if payload.get("hub_push_requested") is not False:
        raise ValueError("Phase 09 export evidence must not pull Phase 10 public release forward")


def _validate_resumed_training_evidence(
    evidence_dir: Path,
    *,
    resume_payload: dict[str, Any],
    training_payload: dict[str, Any],
    expected_config_hash: str,
    expected_model_id: str,
    expected_model_revision: str,
) -> None:
    resumed_path = evidence_dir / "resume-training-run.json"
    if not resumed_path.exists():
        raise ValueError("missing resume-training-run.json")
    recorded_sha = _validate_sha256(
        resume_payload.get("resumed_training_evidence_sha256"),
        "resumed_training_evidence_sha256",
    )
    if sha256_file(resumed_path) != recorded_sha:
        raise ValueError("resume-training-run.json does not match resume.json")
    resumed_payload = _read_json(resumed_path)
    resumed_adapter_sha256, resumed_run_reference = validate_training_run_evidence(
        resumed_payload,
        expected_config_hash=expected_config_hash,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
    )
    if resume_payload.get("resumed_adapter_sha256") != resumed_adapter_sha256:
        raise ValueError("resume.json resumed adapter checksum does not match resumed run")
    if resume_payload.get("resumed_wandb_run_reference") != resumed_run_reference:
        raise ValueError("resume.json resumed W&B run does not match resumed run evidence")
    if resumed_payload.get("run_id") != resume_payload.get("resumed_run_id"):
        raise ValueError("resume.json resumed run ID does not match resumed run evidence")
    if resumed_payload.get("dataset_version") != training_payload.get("dataset_version"):
        raise ValueError("resumed run dataset version does not match source training run")
    if resumed_payload.get("dataset_manifest_checksum") != training_payload.get(
        "dataset_manifest_checksum"
    ):
        raise ValueError("resumed run dataset manifest does not match source training run")
    resume_from_checkpoint = resumed_payload.get("resume_from_checkpoint")
    if not isinstance(resume_from_checkpoint, str) or not resume_from_checkpoint.strip():
        raise ValueError("resumed run did not record its recovery checkpoint")
    if resume_payload.get("resume_from_checkpoint") != resume_from_checkpoint:
        raise ValueError("resume.json checkpoint does not match resumed run evidence")


def evaluate_phase9_repository_state(root: Path) -> Phase9RepositoryStatus:
    bundle = load_training_config(root)
    config_hash = bundle.config_hash()
    evidence_dir = root / "evidence/phase-09"
    training_path = evidence_dir / "training-run.json"
    if not training_path.exists():
        return Phase9RepositoryStatus(
            stage=Phase9Stage.PRE_EXTERNAL_TRAINING,
            training_config_hash=config_hash,
            adapter_sha256=None,
            wandb_run_reference=None,
            complete=False,
            blocker="real connected GPU/W&B training evidence has not been preserved yet",
        )

    try:
        training_payload = _read_json(training_path)
        adapter_sha256, run_reference = validate_training_run_evidence(
            training_payload,
            expected_config_hash=config_hash,
            expected_model_id=bundle.lora.base_model_id,
            expected_model_revision=bundle.lora.base_model_revision,
        )
        training_evidence_sha256 = sha256_file(training_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return Phase9RepositoryStatus(
            stage=Phase9Stage.TRAINING_EVIDENCE_READY,
            training_config_hash=config_hash,
            adapter_sha256=None,
            wandb_run_reference=None,
            complete=False,
            blocker=f"invalid training evidence: {exc}",
        )

    required = (
        ("adapter-reload.json", "phase9-adapter-reload-v1"),
        ("resume.json", "phase9-resume-v1"),
        ("adapter-export.json", "phase9-adapter-export-evidence-v1"),
    )
    try:
        supporting: dict[str, dict[str, Any]] = {}
        for filename, version in required:
            path = evidence_dir / filename
            if not path.exists():
                raise ValueError(f"missing {filename}")
            supporting[filename] = _validate_pass_evidence(
                path,
                evidence_version=version,
                adapter_sha256=adapter_sha256,
                training_config_hash=config_hash,
                training_evidence_sha256=training_evidence_sha256,
            )
        _validate_reload_evidence(
            supporting["adapter-reload.json"],
            expected_model_id=bundle.lora.base_model_id,
            expected_model_revision=bundle.lora.base_model_revision,
            expected_dataset_version=_required_text(training_payload, "dataset_version"),
        )
        _validate_resume_evidence(
            supporting["resume.json"],
            expected_source_run_id=_required_text(training_payload, "run_id"),
            expected_source_wandb_run_reference=run_reference,
            expected_dataset_version=_required_text(training_payload, "dataset_version"),
            expected_dataset_manifest_checksum=_validate_sha256(
                training_payload.get("dataset_manifest_checksum"),
                "dataset_manifest_checksum",
            ),
        )
        _validate_resumed_training_evidence(
            evidence_dir,
            resume_payload=supporting["resume.json"],
            training_payload=training_payload,
            expected_config_hash=config_hash,
            expected_model_id=bundle.lora.base_model_id,
            expected_model_revision=bundle.lora.base_model_revision,
        )
        _validate_export_evidence(
            supporting["adapter-export.json"],
            adapter_sha256=adapter_sha256,
            expected_model_id=bundle.lora.base_model_id,
            expected_model_revision=bundle.lora.base_model_revision,
            expected_dataset_manifest_checksum=_validate_sha256(
                training_payload.get("dataset_manifest_checksum"),
                "dataset_manifest_checksum",
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return Phase9RepositoryStatus(
            stage=Phase9Stage.TRAINING_EVIDENCE_READY,
            training_config_hash=config_hash,
            adapter_sha256=adapter_sha256,
            wandb_run_reference=run_reference,
            complete=False,
            blocker=str(exc),
        )

    return Phase9RepositoryStatus(
        stage=Phase9Stage.COMPLETE,
        training_config_hash=config_hash,
        adapter_sha256=adapter_sha256,
        wandb_run_reference=run_reference,
        complete=True,
        blocker=None,
    )
