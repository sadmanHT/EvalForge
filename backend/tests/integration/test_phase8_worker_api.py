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
from app.inference.rag_orchestration import Phase8RAGJobHandler
from app.inference.rag_protocol import RAGProtocol, load_rag_protocol
from app.inference.tracking import DisabledTracker
from app.main import create_app
from app.readiness import ReadinessStatus
from app.retrieval.indexing import build_index_plan, load_kb_config, persist_index
from app.retrieval.phase8_variants import build_phase8_kb_variant_config
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
        assert "Retrieved context" in prompt
        del generation_config
        assert allowed_labels == self.labels
        label = "no_fault"
        scores = {candidate: (-0.1 if candidate == label else -4.0) for candidate in self.labels}
        return BackendOutput(
            raw_text='{"root_cause_code":"no_fault","reasoning":"phase8 worker smoke"}',
            label_log_likelihoods=scores,
            input_tokens=160,
            output_tokens=12,
            latency_ms=14.0,
            cost_usd=0.001,
            runtime_metadata={"fixture": "phase8-worker-api"},
        )


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.set_main_option(
        "sqlalchemy.url", sqlalchemy_database_url(Settings.from_env().database_url)
    )
    return config


def _index_baseline_rag_variant(session: Session) -> str:
    _protocol, suite = load_rag_protocol(ROOT)
    variant = next(item for item in suite.variants if item.variant_id == suite.baseline_variant_id)
    baseline = load_kb_config(ROOT / "configs/phase7-kb.json")
    config = build_phase8_kb_variant_config(
        baseline,
        kb_version=variant.knowledge_base_version,
        embedding_model_id=variant.embedding_model_id,
        embedding_model_revision=variant.embedding_model_revision,
        chunker_version=variant.chunker_version,
        chunk_size=variant.chunk_size,
        overlap=variant.overlap,
    )
    persist_index(session, build_index_plan(ROOT, config))
    return variant.variant_id


@pytest.fixture(scope="module")
def engine_and_variant() -> tuple[Engine, str]:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    with Session(engine) as session:
        import_phase3_dataset(session, ROOT, DATASET_VERSION)
        variant_id = _index_baseline_rag_variant(session)
        session.commit()
    yield engine, variant_id
    engine.dispose()


def test_real_redis_worker_rag_to_postgres_to_api_readback(
    engine_and_variant: tuple[Engine, str],
) -> None:
    engine, variant_id = engine_and_variant
    settings = Settings.from_env()
    redis_client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.flushdb()
    protocol, _suite = load_rag_protocol(ROOT)

    def backend_factory(loaded_protocol: RAGProtocol) -> FixtureBackend:
        assert loaded_protocol == protocol
        from app.inference.protocol import load_taxonomy

        labels, _categories = load_taxonomy(ROOT)
        return FixtureBackend(loaded_protocol.runtime_config, labels)

    handler = Phase8RAGJobHandler(
        root=ROOT,
        engine=engine,
        backend_factory=backend_factory,
        tracker=DisabledTracker(),
    )
    enqueue(
        redis_client,
        "phase8_rag",
        {
            "variant_id": variant_id,
            "split": "validation",
            "git_commit": "phase8-worker-smoke",
            "hardware_runtime_descriptor": "ci-worker-fixture-no-real-model",
            "cost_rate_snapshot_version": "phase8-ci-fixture-rates-v1",
            "experiment_id": "exp-phase8-worker-api",
            "run_id": "run-phase8-worker-api",
        },
        job_id="phase8-worker-api-job",
    )
    processed = process_one(
        redis_client,
        timeout=1,
        task_handlers={"phase8_rag": handler},
    )
    assert processed is not None
    assert processed.state is JobState.SUCCEEDED
    assert isinstance(processed.result, dict)
    assert processed.result["prediction_count"] == 6
    assert processed.result["retrieval_trace_count"] >= 6
    assert processed.result["cost_record_count"] == 6
    assert processed.result["tracking"]["configured"] is False

    client = TestClient(create_app(readiness_probe=HealthyProbe(), engine=engine))
    response = client.get("/api/runs/run-phase8-worker-api")
    assert response.status_code == 200
    payload = response.json()
    assert payload["experiment"]["pipeline_type"] == "RAG"
    assert payload["stored_prediction_count"] == 6
    assert payload["stored_cost_record_count"] == 6
    assert payload["stored_retrieval_trace_count"] == processed.result["retrieval_trace_count"]
    assert payload["stored_retrieval_trace_count"] >= 6
    assert len(payload["retrieval_traces"]) == payload["stored_retrieval_trace_count"]
    assert payload["run"]["runtime_metadata"]["metric_recomputation_verified"] is True
    assert payload["run"]["runtime_metadata"]["experiment_tracking"]["configured"] is False
    assert "primary.exact_accuracy" in payload["metrics"]

    for trace in payload["retrieval_traces"]:
        provenance = trace["metadata"]["provenance"]
        assert provenance.get("source_split") not in {"validation", "test"}
