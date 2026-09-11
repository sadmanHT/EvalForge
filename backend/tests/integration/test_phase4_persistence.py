from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.experiment_config import ExperimentConfig, PipelineType
from app.db import build_engine, sqlalchemy_database_url
from app.models import (
    DatasetVersion,
    Incident,
    IncidentFamily,
    ModelVersion,
    Prediction,
    Run,
)
from app.repositories import PersistenceRepository
from app.services.dataset_import import import_phase3_dataset

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


def _manifest() -> dict[str, Any]:
    path = ROOT / "datasets/incident_diagnosis/processed" / DATASET_VERSION / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config() -> dict[str, Any]:
    return json.loads((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))


def _experiment_config(*, prompt_version: str) -> ExperimentConfig:
    manifest = _manifest()
    model = _model_config()
    lock_checksum = hashlib.sha256((BACKEND / "requirements.full.lock").read_bytes()).hexdigest()
    return ExperimentConfig(
        study_id=model["study_id"],
        pipeline_type=PipelineType.ZERO_SHOT,
        dataset_version=DATASET_VERSION,
        test_split_manifest_checksum=manifest["manifest_checksum"],
        label_taxonomy_version=manifest["label_taxonomy_version"],
        base_model_id=model["base_model_id"],
        base_model_revision=model["base_model_revision"],
        prompt_version=prompt_version,
        output_schema_version="root-cause-prediction-v1",
        generation_config={"temperature": 0.0, "do_sample": False},
        temperature=0.0,
        confidence_method="normalized_label_sequence_log_likelihood",
        seed=20260908,
        evaluator_version="phase4-integration-v1",
        git_commit="phase4-integration",
        dependency_lock_checksum=lock_checksum,
        hardware_runtime_descriptor="github-actions-postgres-pgvector",
        cost_rate_snapshot_version="phase4-test-rates-v1",
    )


@pytest.fixture(scope="module")
def engine() -> Engine:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    yield engine
    engine.dispose()


def test_migration_from_empty_database_and_stepwise_upgrade(engine: Engine) -> None:
    config = _alembic_config()
    command.downgrade(config, "base")
    command.upgrade(config, "0001_phase02")
    with engine.connect() as connection:
        revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        assert revision == "0001_phase02"

    command.upgrade(config, "head")
    with engine.connect() as connection:
        revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        vector_extension = connection.scalar(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        )
        assert revision == "0002_phase04"
        assert vector_extension == 1


def test_phase3_dataset_import_round_trips_without_loss(engine: Engine) -> None:
    manifest = _manifest()
    with Session(engine) as session:
        first = import_phase3_dataset(session, ROOT, DATASET_VERSION)
        session.commit()
        second = import_phase3_dataset(session, ROOT, DATASET_VERSION)
        session.commit()
        assert first.version == second.version == DATASET_VERSION

    with Session(engine) as session:
        stored = session.get(DatasetVersion, DATASET_VERSION)
        assert stored is not None
        assert stored.manifest_checksum == manifest["manifest_checksum"]
        assert stored.content_checksum == manifest["content_checksum"]
        assert stored.record_count == manifest["record_count"] == 18
        assert stored.family_count == manifest["family_count"] == 18

        family_count = session.scalar(
            select(func.count())
            .select_from(IncidentFamily)
            .where(IncidentFamily.dataset_version == DATASET_VERSION)
        )
        incident_count = session.scalar(
            select(func.count())
            .select_from(Incident)
            .where(Incident.dataset_version == DATASET_VERSION)
        )
        split_rows = session.execute(
            select(Incident.split, func.count())
            .where(Incident.dataset_version == DATASET_VERSION)
            .group_by(Incident.split)
        ).all()
        assert family_count == 18
        assert incident_count == 18
        assert Counter(dict(split_rows)) == Counter({"train": 6, "validation": 6, "test": 6})


def test_family_split_mismatch_is_rejected_by_database(engine: Engine) -> None:
    with Session(engine) as session:
        family = session.scalar(
            select(IncidentFamily).where(IncidentFamily.dataset_version == DATASET_VERSION).limit(1)
        )
        assert family is not None
        wrong_split = "validation" if family.split != "validation" else "test"
        session.add(
            Incident(
                dataset_version=DATASET_VERSION,
                incident_id="incident-phase4-invalid-split",
                family_id=family.family_id,
                split=wrong_split,
                title="invalid split fixture",
                description="must be rejected",
                root_cause_code="no_fault",
                root_cause_category="no_fault",
                source_checksum="c" * 64,
                source_record_id="phase4:invalid-split",
                is_synthetic=False,
                incident_metadata={},
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


def test_prediction_cannot_cross_experiment_dataset_version(engine: Engine) -> None:
    with Session(engine) as session:
        other = DatasetVersion(
            version="phase4-test-other-dataset",
            schema_version="incident-schema-v1",
            label_taxonomy_version="1.0.0",
            manifest_checksum="c" * 64,
            content_checksum="d" * 64,
            split_seed=1,
            record_count=1,
            family_count=1,
            research_ready=False,
            manifest_json={"fixture": True},
        )
        session.add(other)
        session.add(
            IncidentFamily(
                dataset_version=other.version,
                family_id="family-other",
                split="test",
                source_family_key="phase4:other-family",
                root_cause_codes=["no_fault"],
                family_metadata={},
            )
        )
        session.add(
            Incident(
                dataset_version=other.version,
                incident_id="incident-other",
                family_id="family-other",
                split="test",
                title="other dataset incident",
                description="cross-version fixture",
                root_cause_code="no_fault",
                root_cause_category="no_fault",
                source_checksum="e" * 64,
                source_record_id="phase4:other-incident",
                is_synthetic=False,
                incident_metadata={},
            )
        )
        session.commit()

    with Session(engine) as session:
        repo = PersistenceRepository(session)
        experiment = repo.create_experiment(
            "exp-phase4-cross-dataset",
            _experiment_config(prompt_version="phase4-cross-dataset-v1"),
        )
        run = Run(
            run_id="run-phase4-cross-dataset",
            experiment_id=experiment.experiment_id,
            attempt=1,
            status="completed",
            runtime_metadata={},
        )
        session.add(run)
        session.commit()

        session.add(
            Prediction(
                prediction_id="prediction-phase4-cross-dataset",
                run_id=run.run_id,
                experiment_id=experiment.experiment_id,
                dataset_version="phase4-test-other-dataset",
                incident_id="incident-other",
                kb_version=None,
                predicted_root_cause_code="no_fault",
                confidence=0.5,
                output_json={"root_cause_code": "no_fault"},
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


def test_historical_dataset_version_is_database_immutable(engine: Engine) -> None:
    with Session(engine) as session:
        row = session.get(DatasetVersion, DATASET_VERSION)
        assert row is not None
        row.record_count = 19
        with pytest.raises(DBAPIError, match="historical version rows are immutable"):
            session.flush()
        session.rollback()


def test_completed_experiment_is_database_immutable(engine: Engine) -> None:
    with Session(engine) as session:
        repo = PersistenceRepository(session)
        row = repo.create_experiment(
            "exp-phase4-completed-immutable",
            _experiment_config(prompt_version="phase4-completed-immutable-v1"),
            status="completed",
        )
        session.commit()
        row.prompt_version = "mutated-prompt"
        with pytest.raises(DBAPIError, match="completed experiment rows are immutable"):
            session.flush()
        session.rollback()


def test_repository_transaction_rolls_back_partial_creation(engine: Engine) -> None:
    model_id = "phase4/rollback-model"
    revision = "rollback-revision"
    with Session(engine) as session:
        repo = PersistenceRepository(session)
        with (
            pytest.raises(RuntimeError, match="intentional rollback"),
            repo.transaction(),
        ):
            repo.ensure_model_version(model_id, revision)
            raise RuntimeError("intentional rollback")

    with Session(engine) as session:
        stored = session.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id,
                ModelVersion.revision == revision,
            )
        )
        assert stored is None


def test_worker_retry_job_creation_is_idempotent(engine: Engine) -> None:
    with Session(engine) as session:
        repo = PersistenceRepository(session)
        experiment = repo.create_experiment(
            "exp-phase4-idempotency",
            _experiment_config(prompt_version="phase4-idempotency-v1"),
        )
        payload = {"dataset_version": DATASET_VERSION, "attempt": "same-semantics"}
        first = repo.create_job(
            "job-phase4-idempotency",
            "evaluation",
            "phase4-idempotency-key",
            payload,
            experiment_id=experiment.experiment_id,
        )
        second = repo.create_job(
            "job-phase4-idempotency-retry",
            "evaluation",
            "phase4-idempotency-key",
            payload,
            experiment_id=experiment.experiment_id,
        )
        assert first.job_id == second.job_id

        with pytest.raises(ValueError, match="different job semantics"):
            repo.create_job(
                "job-phase4-idempotency-conflict",
                "evaluation",
                "phase4-idempotency-key",
                {"dataset_version": DATASET_VERSION, "attempt": "changed"},
                experiment_id=experiment.experiment_id,
            )
        session.commit()
