from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.training.config import load_training_config
from app.training.evidence import sha256_file
from app.training.phase9_gate import Phase9Stage, evaluate_phase9_repository_state

ROOT = Path(__file__).resolve().parents[3]


def test_repository_state_stops_before_external_training_without_fabrication() -> None:
    status = evaluate_phase9_repository_state(ROOT)
    assert status.stage is Phase9Stage.PRE_EXTERNAL_TRAINING
    assert status.complete is False
    assert status.adapter_sha256 is None
    assert status.wandb_run_reference is None
    assert status.blocker is not None
    assert "GPU/W&B" in status.blocker
    assert status.training_config_hash == load_training_config(ROOT).config_hash()


def test_hard_exit_script_is_expected_to_fail_before_real_training() -> None:
    status = evaluate_phase9_repository_state(ROOT)
    if (ROOT / "evidence/phase-09/training-run.json").exists():
        pytest.skip("real Phase 09 evidence has been committed")
    assert status.complete is False


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_payload(
    *,
    config_hash: str,
    base_model_id: str,
    base_model_revision: str,
    run_id: str,
    adapter_sha256: str,
    wandb_run_reference: str,
    wandb_artifact_reference: str,
    selected_checkpoint: str,
    checkpoint_paths: list[str],
    resume_from_checkpoint: str | None = None,
) -> dict[str, object]:
    return {
        "evidence_version": "phase9-training-run-v1",
        "run_id": run_id,
        "status": "completed",
        "training_config_hash": config_hash,
        "dataset_version": "dataset-v1",
        "dataset_manifest_checksum": "a" * 64,
        "dataset_content_checksum": "b" * 64,
        "prepared_train_sha256": "c" * 64,
        "prepared_validation_sha256": "d" * 64,
        "lineage_sha256": "e" * 64,
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "git_commit": "abc123",
        "hardware_runtime_descriptor": "Fake CUDA runner",
        "resume_from_checkpoint": resume_from_checkpoint,
        "adapter_sha256": adapter_sha256,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_paths": checkpoint_paths,
        "gpu_environment": {"cuda_available": True, "gpu_name": "Fake GPU"},
        "tracking": {
            "provider": "wandb",
            "configured": True,
            "run_reference": wandb_run_reference,
            "artifact_reference": wandb_artifact_reference,
        },
    }


def _fake_complete_evidence_root(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "training/configs"
    config_dir.mkdir(parents=True)
    shutil.copy(ROOT / "training/configs/lora_config.yaml", config_dir / "lora_config.yaml")
    shutil.copy(ROOT / "training/configs/training_args.yaml", config_dir / "training_args.yaml")
    bundle = load_training_config(tmp_path)
    config_hash = bundle.config_hash()
    adapter_sha256 = "f" * 64
    evidence_dir = tmp_path / "evidence/phase-09"
    training_path = evidence_dir / "training-run.json"
    _write_json(
        training_path,
        _training_payload(
            config_hash=config_hash,
            base_model_id=bundle.lora.base_model_id,
            base_model_revision=bundle.lora.base_model_revision,
            run_id="phase9-gpu-run",
            adapter_sha256=adapter_sha256,
            wandb_run_reference="https://wandb.invalid/run/1",
            wandb_artifact_reference="https://wandb.invalid/artifact/1",
            selected_checkpoint="/tmp/checkpoint-2",
            checkpoint_paths=["/tmp/checkpoint-1", "/tmp/checkpoint-2"],
        ),
    )
    training_sha = sha256_file(training_path)
    common = {
        "status": "pass",
        "adapter_sha256": adapter_sha256,
        "training_config_hash": config_hash,
        "training_evidence_sha256": training_sha,
    }
    _write_json(
        evidence_dir / "adapter-reload.json",
        {
            **common,
            "evidence_version": "phase9-adapter-reload-v1",
            "validation_split": "validation",
            "validation_incident_id": "incident-validation-1",
            "parse_status": "OK",
            "pipeline_type": "FINETUNED",
            "adapter_id": "/tmp/adapter",
            "adapter_revision": "local",
            "base_model_id": bundle.lora.base_model_id,
            "base_model_revision": bundle.lora.base_model_revision,
            "dataset_version": "dataset-v1",
        },
    )

    resumed_training_path = evidence_dir / "resume-training-run.json"
    resumed_adapter_sha256 = "1" * 64
    _write_json(
        resumed_training_path,
        _training_payload(
            config_hash=config_hash,
            base_model_id=bundle.lora.base_model_id,
            base_model_revision=bundle.lora.base_model_revision,
            run_id="phase9-gpu-resume",
            adapter_sha256=resumed_adapter_sha256,
            wandb_run_reference="https://wandb.invalid/run/2",
            wandb_artifact_reference="https://wandb.invalid/artifact/2",
            selected_checkpoint="/tmp/resume-checkpoint-2",
            checkpoint_paths=["/tmp/resume-checkpoint-2"],
            resume_from_checkpoint="/tmp/checkpoint-1",
        ),
    )
    resumed_training_sha = sha256_file(resumed_training_path)
    _write_json(
        evidence_dir / "resume.json",
        {
            **common,
            "evidence_version": "phase9-resume-v1",
            "resume_from_checkpoint": "/tmp/checkpoint-1",
            "source_run_id": "phase9-gpu-run",
            "source_wandb_run_reference": "https://wandb.invalid/run/1",
            "resumed_run_id": "phase9-gpu-resume",
            "resumed_wandb_run_reference": "https://wandb.invalid/run/2",
            "resumed_wandb_artifact_reference": "https://wandb.invalid/artifact/2",
            "resumed_adapter_sha256": resumed_adapter_sha256,
            "resumed_training_evidence_sha256": resumed_training_sha,
            "dataset_version": "dataset-v1",
            "dataset_manifest_checksum": "a" * 64,
        },
    )
    _write_json(
        evidence_dir / "adapter-export.json",
        {
            **common,
            "evidence_version": "phase9-adapter-export-evidence-v1",
            "destination": "/tmp/export",
            "manifest_path": "/tmp/export/evalforge-adapter-manifest.json",
            "exported_adapter_sha256": adapter_sha256,
            "export_tree_sha256": "3" * 64,
            "dataset_manifest_checksum": "a" * 64,
            "base_model_id": bundle.lora.base_model_id,
            "base_model_revision": bundle.lora.base_model_revision,
            "hub_push_requested": False,
        },
    )
    return tmp_path, training_path


def test_hard_exit_accepts_complete_linked_evidence(tmp_path: Path) -> None:
    root, _training_path = _fake_complete_evidence_root(tmp_path)
    status = evaluate_phase9_repository_state(root)
    assert status.stage is Phase9Stage.COMPLETE
    assert status.complete is True
    assert status.blocker is None


def test_hard_exit_rejects_supporting_evidence_after_training_record_changes(
    tmp_path: Path,
) -> None:
    root, training_path = _fake_complete_evidence_root(tmp_path)
    training_path.write_text(training_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    status = evaluate_phase9_repository_state(root)
    assert status.stage is Phase9Stage.TRAINING_EVIDENCE_READY
    assert status.complete is False
    assert status.blocker is not None
    assert "not linked" in status.blocker
