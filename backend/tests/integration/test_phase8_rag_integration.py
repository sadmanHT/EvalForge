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
from app.evaluation.contracts import EvaluationExample
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation
from app.inference.base_model import BackendOutput, GenerationConfig, IncidentInput, RuntimeConfig
from app.inference.rag_persistence import persist_rag_retrieval_traces
from app.inference.rag_pipeline import ContextBudget, PgVectorRetriever, RAGPipeline
from app.inference.rag_protocol import build_rag_experiment_config, load_rag_protocol
from app.models import Incident, RetrievalTrace
from app.repositories import PersistenceRepository
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


class DynamicFakeBackend:
    def __init__(self, allowed_labels: tuple[str, ...]) -> None:
        self.allowed_labels = allowed_labels
        self.winner = allowed_labels[0]
        self._runtime_config = RuntimeConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            revision="e8737b84b4470b28db3a0be719b362b1bd39a14d",
            seed=20260908,
        )

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
        assert allowed_labels == self.allowed_labels
        scores = {label: (-0.1 if label == self.winner else -4.0) for label in allowed_labels}
        return BackendOutput(
            raw_text=json.dumps(
                {
                    "root_cause_code": self.winner,
                    "reasoning": "Deterministic Phase 08 integration fixture.",
                }
            ),
            label_log_likelihoods=scores,
            input_tokens=200,
            output_tokens=20,
            latency_ms=20.0,
            cost_usd=0.0,
            runtime_metadata={"fixture": "phase8-rag-integration"},
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


def test_rag_pgvector_prompt_evaluation_and_trace_persistence(engine: Engine) -> None:
    protocol, suite = load_rag_protocol(ROOT)
    variant = next(item for item in suite.variants if item.variant_id == "rag-default-k5")
    kb_config = load_kb_config(ROOT / "configs/phase7-kb.json")
    plan = build_index_plan(ROOT, kb_config)

    with Session(engine) as session:
        import_phase3_dataset(session, ROOT, DATASET_VERSION)
        persist_index(session, plan)
        config = build_rag_experiment_config(
            root=ROOT,
            protocol=protocol,
            variant=variant,
            git_commit="phase8-integration",
            hardware_runtime_descriptor="phase8-postgres-pgvector-fake-model",
            cost_rate_snapshot_version="phase8-integration-zero-cost",
        )
        PersistenceRepository(session).create_experiment("exp-phase8-rag-integration", config)
        session.commit()

    taxonomy_payload = json.loads(
        (ROOT / "configs/label-taxonomy.yaml").read_text(encoding="utf-8")
    )
    labels = tuple(item["id"] for item in taxonomy_payload["labels"])
    categories = {
        item["id"]: item["category"]
        for item in taxonomy_payload["labels"]
    }
    relevant_document_by_label = {
        document.root_cause_code: document.document_id
        for document in plan.documents
        if document.source_type == "runbook" and document.root_cause_code is not None
    }
    relevant_chunks_by_label = {
        label: tuple(
            chunk.chunk_id
            for chunk in plan.chunks
            if chunk.document_id == relevant_document_by_label[label]
        )
        for label in labels
    }

    backend = DynamicFakeBackend(labels)
    adapter = HashEmbeddingAdapter(kb_config.embedding)
    predictions = []
    examples = []
    traces_by_incident = {}
    with Session(engine) as session:
        retriever = PgVectorRetriever(
            session=session,
            adapter=adapter,
            kb_version=variant.knowledge_base_version,
        )
        pipeline = RAGPipeline(
            backend=backend,
            retriever=retriever,
            allowed_labels=labels,
            top_k=variant.top_k,
            generation_config=protocol.generation_config,
            context_budget=ContextBudget(
                max_context_tokens=variant.max_context_tokens,
                max_chunk_tokens=variant.max_chunk_tokens,
                policy_version=variant.context_policy_version,
            ),
        )
        incidents = session.scalars(
            select(Incident)
            .where(Incident.dataset_version == DATASET_VERSION, Incident.split == "validation")
            .order_by(Incident.incident_id)
        ).all()
        assert len(incidents) == 6
        for incident in incidents:
            backend.winner = incident.root_cause_code
            rag_result = pipeline.predict(
                IncidentInput(
                    incident_id=incident.incident_id,
                    title=incident.title,
                    description=incident.description,
                ),
                query_family_id=incident.family_id,
            )
            predictions.append(rag_result.prediction)
            traces_by_incident[incident.incident_id] = rag_result.retrieval_traces
            examples.append(
                EvaluationExample(
                    incident_id=incident.incident_id,
                    split="validation",
                    root_cause_code=incident.root_cause_code,
                    root_cause_category=incident.root_cause_category,
                    relevant_chunk_ids=relevant_chunks_by_label[incident.root_cause_code],
                )
            )

        harness = EvaluationHarness(
            evaluator_version=protocol.evaluator_version,
            allowed_labels=labels,
            label_to_category=categories,
            ece_bins=protocol.ece_bins,
        )
        result = harness.evaluate(examples, predictions)
        assert result.metric_map()["primary.exact_accuracy"] == 1.0
        assert "retrieval.context_precision" in result.metric_map()
        assert "retrieval.context_recall" in result.metric_map()

        persist_evaluation(
            session,
            experiment_id="exp-phase8-rag-integration",
            run_id="run-phase8-rag-integration",
            dataset_version=DATASET_VERSION,
            predictions=predictions,
            result=result,
            split="validation",
            kb_version=variant.knowledge_base_version,
        )
        expected_trace_count = sum(len(items) for items in traces_by_incident.values())
        persisted_count = persist_rag_retrieval_traces(
            session,
            run_id="run-phase8-rag-integration",
            kb_version=variant.knowledge_base_version,
            traces_by_incident=traces_by_incident,
        )
        session.commit()
        assert persisted_count == expected_trace_count

    with Session(engine) as session:
        traces = session.scalars(
            select(RetrievalTrace).order_by(RetrievalTrace.prediction_id, RetrievalTrace.rank)
        ).all()
        assert len(traces) == expected_trace_count
        assert all(trace.rank >= 1 for trace in traces)
        assert all("raw_similarity_score" in trace.trace_metadata for trace in traces)
        assert all("provenance" in trace.trace_metadata for trace in traces)
        assert all(
            "root_cause_code" not in trace.trace_metadata["provenance"] for trace in traces
        )
        assert all(
            trace.trace_metadata["provenance"].get("source_split") not in {"validation", "test"}
            for trace in traces
        )
