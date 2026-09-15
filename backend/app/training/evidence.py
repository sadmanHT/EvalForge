from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingEvidenceIdentity:
    training_config_hash: str
    adapter_sha256: str
    dataset_version: str
    dataset_manifest_checksum: str
    base_model_id: str
    base_model_revision: str
    run_id: str
    wandb_run_reference: str
    wandb_artifact_reference: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"adapter directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evidence requires non-empty {key}")
    return value.strip()


def _required_sha256(payload: Mapping[str, Any], key: str) -> str:
    value = _required_text(payload, key)
    if len(value) != 64:
        raise ValueError(f"{key} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a SHA-256 hex digest") from exc
    return value


def load_training_evidence_identity(path: Path) -> TrainingEvidenceIdentity:
    payload = read_json_object(path)
    if payload.get("evidence_version") != "phase9-training-run-v1":
        raise ValueError("unexpected Phase 09 training evidence version")
    if payload.get("status") != "completed":
        raise ValueError("Phase 09 training evidence is not completed")

    environment = payload.get("gpu_environment")
    if not isinstance(environment, dict) or environment.get("cuda_available") is not True:
        raise ValueError("Phase 09 evidence identity requires a CUDA training runtime")

    tracking = payload.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("Phase 09 evidence identity requires W&B tracking metadata")
    if tracking.get("provider") != "wandb" or tracking.get("configured") is not True:
        raise ValueError("Phase 09 evidence identity requires configured W&B tracking")

    return TrainingEvidenceIdentity(
        training_config_hash=_required_sha256(payload, "training_config_hash"),
        adapter_sha256=_required_sha256(payload, "adapter_sha256"),
        dataset_version=_required_text(payload, "dataset_version"),
        dataset_manifest_checksum=_required_sha256(payload, "dataset_manifest_checksum"),
        base_model_id=_required_text(payload, "base_model_id"),
        base_model_revision=_required_text(payload, "base_model_revision"),
        run_id=_required_text(payload, "run_id"),
        wandb_run_reference=_required_text(tracking, "run_reference"),
        wandb_artifact_reference=_required_text(tracking, "artifact_reference"),
    )


def write_supporting_evidence(
    path: Path,
    *,
    evidence_version: str,
    training_evidence_path: Path,
    identity: TrainingEvidenceIdentity,
    details: Mapping[str, object],
) -> dict[str, object]:
    protected = {
        "evidence_version",
        "status",
        "adapter_sha256",
        "training_config_hash",
        "training_evidence_sha256",
    }
    overlap = protected.intersection(details)
    if overlap:
        raise ValueError(f"supporting evidence details override protected keys: {sorted(overlap)}")
    payload: dict[str, object] = {
        "evidence_version": evidence_version,
        "status": "pass",
        "adapter_sha256": identity.adapter_sha256,
        "training_config_hash": identity.training_config_hash,
        "training_evidence_sha256": sha256_file(training_evidence_path),
        **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload
