from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.training.config import load_training_config


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
) -> None:
    payload = _read_json(path)
    if payload.get("evidence_version") != evidence_version:
        raise ValueError(f"unexpected evidence version in {path.name}")
    if payload.get("status") != "pass":
        raise ValueError(f"{path.name} did not record PASS")
    if payload.get("adapter_sha256") != adapter_sha256:
        raise ValueError(f"{path.name} adapter checksum does not match training run")
    if payload.get("training_config_hash") != training_config_hash:
        raise ValueError(f"{path.name} training config hash does not match")


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
        for filename, version in required:
            path = evidence_dir / filename
            if not path.exists():
                raise ValueError(f"missing {filename}")
            _validate_pass_evidence(
                path,
                evidence_version=version,
                adapter_sha256=adapter_sha256,
                training_config_hash=config_hash,
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
