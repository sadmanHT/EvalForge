from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from redis import Redis
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import build_engine, sqlalchemy_database_url
from app.inference.base_model import BackendOutput, GenerationConfig, RuntimeConfig
from app.inference.orchestration import Phase6BaselineJobHandler
from app.inference.protocol import BaselineProtocol, load_taxonomy
from app.inference.tracking import DisabledTracker
from app.main import create_app
from app.readiness import ReadinessStatus
from app.services.dataset_import import import_phase3_dataset
from app.worker.queue import JobState, enqueue, process_one

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


class HealthyProbe:
    def check(self) -> ReadinessStatus:
        return ReadinessStatus(database=True, redis=True)


class FixtureBackend:
    def __init__(self, runtime_config: RuntimeConfig, labels: tuple[str, ...]) -> None:
        self._runtime_config = runtime_config
        self.labels = labels

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: tuple[str, ...],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        del prompt, generation_config
        assert allowed_labels == self.labels
        label = "no_fault"
        scores = {candidate: (-0.1 if candidate == label else -4.0) for candidate in self.labels}
        return BackendOutput(
            raw_text='{"root_cause_code":"no_fault","reasoning":"worker smoke"}',
            label_log_likelihoods=scores,
            input_tokens=90,
            output_tokens=10,
            latency_ms=12.0,
            cost_usd=0.001,
            runtime_metadata={"fixture": "phase6-worker-api"},
        )


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.set_main_option(
        "sqlalchemy.url", sqlalchemy_database_url(Settings.from_env().database_url)
    )
    return config


@pytest.fixture(scope="module")
def engine() -> Engine:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    with Session(engine) as session:
        import_phase3_dataset(session, ROOT, DATASET_VERSION)
        session.commit()
    yield engine
    engine.dispose()


def test_real_redis_worker_to_postgres_to_api_readback(engine: Engine) -> None:
    settings = Settings.from_env()
    redis_client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.flushdb()
    labels, _categories = load_taxonomy(ROOT)

    def backend_factory(protocol: BaselineProtocol) -> FixtureBackend:
        return FixtureBackend(protocol.runtime_config, labels)

    handler = Phase6BaselineJobHandler(
        root=ROOT,
        engine=engine,
        backend_factory=backend_factory,
        tracker=DisabledTracker(),
    )
    enqueue(
        redis_client,
        "phase6_baseline",
        {
            "split": "validation",
            "git_commit": "phase6-worker-smoke",
            "hardware_runtime_descriptor": "ci-worker-fixture-no-real-model",
            "cost_rate_snapshot_version": "phase6-ci-fixture-rates-v1",
            "experiment_id": "exp-phase6-worker-api",
            "run_id": "run-phase6-worker-api",
        },
        job_id="phase6-worker-api-job",
    )
    processed = process_one(
        redis_client,
        timeout=1,
        task_handlers={"phase6_baseline": handler},
    )
    assert processed is not None
    assert processed.state is JobState.SUCCEEDED
    assert isinstance(processed.result, dict)
    assert processed.result["prediction_count"] == 6
    assert processed.result["cost_record_count"] == 6
    assert processed.result["tracking"]["configured"] is False

    client = TestClient(create_app(readiness_probe=HealthyProbe(), engine=engine))
    response = client.get("/api/runs/run-phase6-worker-api")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stored_prediction_count"] == 6
    assert len(payload["predictions"]) == 6
    assert len({row["incident_id"] for row in payload["predictions"]}) == 6
    assert payload["stored_cost_record_count"] == 6
    assert len(payload["cost_records"]) == 6
    assert {row["cost_rate_snapshot_version"] for row in payload["cost_records"]} == {
        "phase6-ci-fixture-rates-v1"
    }
    assert payload["run"]["runtime_metadata"]["metric_recomputation_verified"] is True
    assert payload["run"]["runtime_metadata"]["experiment_tracking"]["configured"] is False
    assert "primary.exact_accuracy" in payload["metrics"]
