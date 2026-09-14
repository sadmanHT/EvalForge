from __future__ import annotations

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import build_engine, sqlalchemy_database_url
from app.inference.base_model import BackendOutput, GenerationConfig, RuntimeConfig
from app.inference.rag_pipeline import ContextBudget, PgVectorRetriever, RAGPipeline
from app.inference.rag_protocol import load_rag_protocol
from app.inference.rag_runner import run_rag_experiment
from app.models import Experiment, Prediction, RetrievalTrace, Run
from app.retrieval.embeddings import HashEmbeddingAdapter
from app.retrieval.indexing import build_index_plan, load_kb_config, persist_index
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


class RunnerFakeBackend:
    def __init__(self, runtime_config: RuntimeConfig, winner: str) -> None:
        self._runtime_config = runtime_config
        self.winner = winner

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
        assert prompt
        assert self.winner in allowed_labels
        assert generation_config.temperature == 0.0
        scores = {label: (-0.1 if label == self.winner else -4.0) for label in allowed_labels}
        return BackendOutput(
            raw_text=json.dumps(
                {
                    "root_cause_code": self.winner,
                    "reasoning": "Deterministic Phase 08 runner fixture.",
                }
            ),
            label_log_likelihoods=scores,
            input_tokens=220,
            output_tokens=18,
            latency_ms=22.0,
            cost_usd=0.0,
            runtime_metadata={"fixture": "phase8-rag-runner"},
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


def test_rag_runner_is_reproducible_aligned_and_leakage_safe(engine: Engine) -> None:
    protocol, suite = load_rag_protocol(ROOT)
    variant = next(item for item in suite.variants if item.variant_id == suite.baseline_variant_id)
    kb_config = load_kb_config(ROOT / "configs/phase7-kb.json")
    plan = build_index_plan(ROOT, kb_config)

    with Session(engine) as session:
        import_phase3_dataset(session, ROOT, DATASET_VERSION)
        persist_index(session, plan)
        session.commit()

    taxonomy_payload = json.loads(
        (ROOT / "configs/label-taxonomy.yaml").read_text(encoding="utf-8")
    )
    labels = tuple(item["id"] for item in taxonomy_payload["labels"])
    backend = RunnerFakeBackend(protocol.runtime_config, labels[0])

    with Session(engine) as session:
        pipeline = RAGPipeline(
            backend=backend,
            retriever=PgVectorRetriever(
                session=session,
                adapter=HashEmbeddingAdapter(kb_config.embedding),
                kb_version=variant.knowledge_base_version,
            ),
            allowed_labels=labels,
            top_k=variant.top_k,
            generation_config=protocol.generation_config,
            context_budget=ContextBudget(
                max_context_tokens=variant.max_context_tokens,
                max_chunk_tokens=variant.max_chunk_tokens,
                policy_version=variant.context_policy_version,
            ),
        )
        summary = run_rag_experiment(
            session,
            root=ROOT,
            protocol=protocol,
            variant=variant,
            pipeline=pipeline,
            split="validation",
            git_commit="phase8-runner-integration",
            hardware_runtime_descriptor="phase8-postgres-pgvector-fake-model",
            cost_rate_snapshot_version="phase8-integration-zero-cost",
            experiment_id="exp-phase8-rag-runner",
            run_id="run-phase8-rag-runner",
        )
        session.commit()

        assert summary.prediction_count == 6
        assert summary.retrieval_trace_count > 0
        assert summary.cost_record_count == 6
        assert summary.recomputation_verified is True
        assert "retrieval.context_precision" in summary.metric_values
        assert "retrieval.context_recall" in summary.metric_values

        repeated = run_rag_experiment(
            session,
            root=ROOT,
            protocol=protocol,
            variant=variant,
            pipeline=pipeline,
            split="validation",
            git_commit="phase8-runner-integration",
            hardware_runtime_descriptor="phase8-postgres-pgvector-fake-model",
            cost_rate_snapshot_version="phase8-integration-zero-cost",
            experiment_id="exp-phase8-rag-runner",
            run_id="run-phase8-rag-runner",
        )
        assert repeated == summary

    baseline_validation = json.loads(
        (ROOT / "evidence/phase-06/validation-run.json").read_text(encoding="utf-8")
    )
    expected_ids = set(baseline_validation["expected_incident_ids"])

    with Session(engine) as session:
        experiment = session.get(Experiment, "exp-phase8-rag-runner")
        run = session.get(Run, "run-phase8-rag-runner")
        assert experiment is not None and experiment.pipeline_type == "RAG"
        assert experiment.status == "completed"
        assert run is not None and run.status == "completed"
        assert run.runtime_metadata["phase8_variant_id"] == variant.variant_id

        predictions = session.scalars(
            select(Prediction)
            .where(Prediction.run_id == "run-phase8-rag-runner")
            .order_by(Prediction.incident_id)
        ).all()
        assert {row.incident_id for row in predictions} == expected_ids

        traces = session.scalars(
            select(RetrievalTrace)
            .join(Prediction, RetrievalTrace.prediction_id == Prediction.prediction_id)
            .where(Prediction.run_id == "run-phase8-rag-runner")
            .order_by(Prediction.incident_id, RetrievalTrace.rank)
        ).all()
        assert len(traces) == summary.retrieval_trace_count
        for trace in traces:
            provenance = trace.trace_metadata["provenance"]
            assert provenance.get("research_eligible") is True
            assert provenance.get("source_split") not in {"validation", "test"}
            assert "root_cause_code" not in provenance


def test_rag_runner_refuses_locked_test_before_protocol_freeze(engine: Engine) -> None:
    protocol, suite = load_rag_protocol(ROOT)
    variant = next(item for item in suite.variants if item.variant_id == suite.baseline_variant_id)
    kb_config = load_kb_config(ROOT / "configs/phase7-kb.json")

    with Session(engine) as session:
        pipeline = RAGPipeline(
            backend=RunnerFakeBackend(protocol.runtime_config, "no_fault"),
            retriever=PgVectorRetriever(
                session=session,
                adapter=HashEmbeddingAdapter(kb_config.embedding),
                kb_version=variant.knowledge_base_version,
            ),
            allowed_labels=tuple(
                item["id"]
                for item in json.loads(
                    (ROOT / "configs/label-taxonomy.yaml").read_text(encoding="utf-8")
                )["labels"]
            ),
            top_k=variant.top_k,
            generation_config=protocol.generation_config,
            context_budget=ContextBudget(
                max_context_tokens=variant.max_context_tokens,
                max_chunk_tokens=variant.max_chunk_tokens,
                policy_version=variant.context_policy_version,
            ),
        )
        with pytest.raises(ValueError, match="locked test requires a frozen"):
            run_rag_experiment(
                session,
                root=ROOT,
                protocol=protocol,
                variant=variant,
                pipeline=pipeline,
                split="test",
                git_commit="must-not-run",
                hardware_runtime_descriptor="must-not-run",
                cost_rate_snapshot_version="must-not-run",
            )
