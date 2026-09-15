from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from redis import Redis
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import build_engine, sqlalchemy_database_url
from app.models import AdapterVersion, Artifact, Experiment, Job
from app.training.orchestration import Phase9TrainingJobHandler, SmokeTrainingExecutor
from app.worker.queue import JobState, enqueue, process_one

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.set_main_option(
        "sqlalchemy.url",
        sqlalchemy_database_url(Settings.from_env().database_url),
    )
    return config


@pytest.fixture(scope="module")
def engine() -> Engine:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    yield engine
    engine.dispose()


def test_worker_smoke_persists_job_adapter_artifact_and_candidate_experiment(
    engine: Engine,
    tmp_path: Path,
) -> None:
    settings = Settings.from_env()
    redis_client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.flushdb()
    handler = Phase9TrainingJobHandler(
        root=ROOT,
        engine=engine,
        work_root=tmp_path / "phase9-work",
        executor=SmokeTrainingExecutor(tmp_path / "phase9-smoke"),
    )
    payload = {
        "job_id": "phase9-worker-smoke-job",
        "idempotency_key": "phase9-worker-smoke-job-v1",
        "dataset_version": DATASET_VERSION,
        "git_commit": "phase9-worker-smoke",
        "hardware_runtime_descriptor": "ci-cpu-contract-no-real-model",
        "cost_rate_snapshot_version": "phase9-ci-fixture-rates-v1",
        "experiment_id": "exp-phase9-worker-smoke",
    }
    enqueue(
        redis_client,
        "phase9_training",
        payload,
        job_id="phase9-worker-smoke-redis-job",
    )
    processed = process_one(
        redis_client,
        timeout=1,
        task_handlers={"phase9_training": handler},
    )
    assert processed is not None
    assert processed.state is JobState.SUCCEEDED
    assert isinstance(processed.result, dict)
    assert processed.result["scientific_adapter"] is False
    assert processed.result["prepared_manifest"]["train_record_count"] == 6
    assert processed.result["prepared_manifest"]["validation_record_count"] == 6

    with Session(engine) as session:
        job = session.get(Job, payload["job_id"])
        assert job is not None
        assert job.status == "completed"
        assert job.experiment_id == payload["experiment_id"]
        assert job.result_json is not None
        assert job.result_json["scientific_adapter"] is False

        experiment = session.get(Experiment, payload["experiment_id"])
        assert experiment is not None
        assert experiment.pipeline_type == "FINETUNED"
        assert experiment.status == "planned"
        assert experiment.kb_version is None
        assert experiment.adapter_version_id is not None

        adapter = session.get(AdapterVersion, experiment.adapter_version_id)
        assert adapter is not None
        assert adapter.metadata_json["scientific_adapter"] is False
        assert adapter.metadata_json["training_config_hash"] == processed.result["training_config_hash"]

        artifact = session.scalar(
            select(Artifact).where(Artifact.experiment_id == payload["experiment_id"])
        )
        assert artifact is not None
        assert artifact.kind == "peft_adapter"
        assert artifact.sha256 == processed.result["adapter_sha256"]
        assert artifact.artifact_metadata["scientific_adapter"] is False
        assert artifact.artifact_metadata["base_model_id"] == "mistralai/Mistral-7B-Instruct-v0.3"
