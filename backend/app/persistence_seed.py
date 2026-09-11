from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from app.core.experiment_config import ExperimentConfig, PipelineType
from app.db import build_engine, session_scope
from app.repositories import PersistenceRepository
from app.services.dataset_import import import_phase3_dataset


def _dependency_lock_path(root: Path) -> Path:
    repository_path = root / "backend/requirements.full.lock"
    if repository_path.exists():
        return repository_path
    container_path = root / "requirements.full.lock"
    if container_path.exists():
        return container_path
    raise FileNotFoundError("backend dependency lock is unavailable")


def build_smoke_config(
    root: Path,
    dataset: dict[str, Any],
    model: dict[str, Any],
) -> ExperimentConfig:
    lock_checksum = hashlib.sha256(_dependency_lock_path(root).read_bytes()).hexdigest()
    return ExperimentConfig(
        study_id=model["study_id"],
        pipeline_type=PipelineType.ZERO_SHOT,
        dataset_version=dataset["dataset_version"],
        test_split_manifest_checksum=dataset["manifest_checksum"],
        label_taxonomy_version=dataset["label_taxonomy_version"],
        base_model_id=model["base_model_id"],
        base_model_revision=model["base_model_revision"],
        prompt_version="phase4-persistence-smoke-v1",
        output_schema_version="root-cause-prediction-v1",
        generation_config={"temperature": 0.0, "do_sample": False, "max_new_tokens": 64},
        temperature=0.0,
        confidence_method="normalized_label_sequence_log_likelihood",
        seed=20260908,
        evaluator_version="phase4-persistence-smoke-v1",
        git_commit=os.environ.get("GIT_COMMIT", "phase4-smoke"),
        dependency_lock_checksum=lock_checksum,
        hardware_runtime_descriptor="phase4-ci-smoke-no-model-inference",
        cost_rate_snapshot_version="phase4-smoke-rates-v1",
    )


def main() -> int:
    root = Path(os.environ.get("EVALFORGE_ROOT", ".")).resolve()
    model = json.loads((root / "configs/model.yaml").read_text(encoding="utf-8"))
    manifest_path = (
        root
        / "datasets/incident_diagnosis/processed"
        / "evalforge-incident-diagnosis-v0.1.0"
        / "manifest.json"
    )
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = build_engine()
    with session_scope(engine) as session:
        imported = import_phase3_dataset(session, root, dataset["dataset_version"])
        repo = PersistenceRepository(session)
        repo.ensure_model_version(
            model["base_model_id"],
            model["base_model_revision"],
            license_name=model.get("license"),
            metadata=model,
        )
        config = build_smoke_config(root, dataset, model)
        experiment = repo.create_experiment("exp-phase4-persistence-smoke", config)
        repo.create_job(
            "job-phase4-persistence-smoke",
            "persistence_smoke",
            "phase4-persistence-smoke-v1",
            {"dataset_version": imported.version, "config_hash": experiment.config_hash},
            experiment_id=experiment.experiment_id,
        )
    print("PHASE04_PERSISTENCE_SEED=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
