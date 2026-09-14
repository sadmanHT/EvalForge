from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.contracts import ParseStatus
from app.inference.base_model import BackendOutput, GenerationConfig, IncidentInput, RuntimeConfig
from app.inference.rag_ablations import RAGAblationSuite, load_ablation_suite
from app.inference.rag_pipeline import ContextBudget, RAGPipeline, assemble_context
from app.inference.ragas import RagasFaithfulnessAdapter
from app.retrieval.reranker import TokenOverlapReranker
from app.retrieval.search import RetrievalResult

ROOT = Path(__file__).resolve().parents[3]
LABELS = (
    "n_plus_one_query",
    "database_connection_leak",
    "disk_exhaustion",
    "memory_leak",
    "broken_payment_configuration",
    "no_fault",
)


def _scores(winner: str) -> dict[str, float]:
    return {label: (-0.1 if label == winner else -4.0) for label in LABELS}


class FakeBackend:
    def __init__(self, winner: str = "broken_payment_configuration") -> None:
        self._runtime_config = RuntimeConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            revision="e8737b84b4470b28db3a0be719b362b1bd39a14d",
            seed=20260908,
        )
        self.winner = winner
        self.prompts: list[str] = []

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
        assert allowed_labels == LABELS
        self.prompts.append(prompt)
        return BackendOutput(
            raw_text=json.dumps(
                {
                    "root_cause_code": self.winner,
                    "reasoning": "The incident evidence best matches the selected cause.",
                }
            ),
            label_log_likelihoods=_scores(self.winner),
            input_tokens=180,
            output_tokens=20,
            latency_ms=31.5,
            cost_usd=0.0,
            runtime_metadata={"fixture": "phase8-ci"},
        )


class FakeRetriever:
    kb_version = "evalforge-kb-v0.1.0"

    def __init__(self, results: tuple[RetrievalResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(
        self,
        *,
        query_text: str,
        top_k: int,
        query_family_id: str | None,
    ) -> tuple[RetrievalResult, ...]:
        self.calls.append((query_text, top_k, query_family_id))
        return self.results[:top_k]


def _result(
    chunk_id: str,
    text: str,
    score: float,
    *,
    source_family_id: str | None = "family-train",
    metadata: dict[str, object] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        score=score,
        source_family_id=source_family_id,
        document_metadata=metadata
        or {
            "source_uri": f"evalforge://docs/{chunk_id}",
            "source_type": "runbook",
            "source_version": "v1",
            "source_split": "train",
            "research_eligible": True,
        },
    )


def test_context_budget_truncates_deterministically() -> None:
    assembly = assemble_context(
        (_result("chunk-a", "one two three four five six", 0.8),),
        budget=ContextBudget(max_context_tokens=5, max_chunk_tokens=4),
    )
    assert assembly.included_chunk_ids == ("chunk-a",)
    assert assembly.token_count == 5
    assert assembly.chunk_token_counts == (("chunk-a", 4),)
    assert assembly.truncated_chunk_ids == ("chunk-a",)
    assert "five" not in assembly.text


def test_rag_pipeline_builds_structured_prediction_and_complete_trace() -> None:
    results = (
        _result("chunk-db", "database connection pool leak timeout", 0.40),
        _result("chunk-pay", "payment configuration routing failure rollback", 0.35),
    )
    backend = FakeBackend()
    pipeline = RAGPipeline(
        backend=backend,
        retriever=FakeRetriever(results),
        allowed_labels=LABELS,
        top_k=2,
        reranker=TokenOverlapReranker(),
        context_budget=ContextBudget(max_context_tokens=32, max_chunk_tokens=16),
    )
    result = pipeline.predict(
        IncidentInput(
            incident_id="incident-validation",
            title="Payment failures after configuration rollout",
            description="Credit-card requests failed immediately after a routing change.",
        ),
        query_family_id="family-validation",
    )

    assert result.prediction.parse_status is ParseStatus.OK
    assert result.prediction.predicted_root_cause_code == "broken_payment_configuration"
    assert result.prediction.pipeline_metadata["pipeline_type"] == "RAG"
    assert result.prediction.pipeline_metadata["research_mode"] is True
    assert result.prediction.input_tokens == 180
    assert result.prediction.total_tokens == 200
    assert len(result.retrieval_traces) == 2
    assert [trace.rank for trace in result.retrieval_traces] == [1, 2]
    assert all(trace.raw_similarity_score is not None for trace in result.retrieval_traces)
    assert all(trace.reranker_score is not None for trace in result.retrieval_traces)
    assert set(result.prediction.retrieved_chunk_ids) == {"chunk-db", "chunk-pay"}
    assert result.prediction.evidence_citations
    assert all(trace.provenance["source_split"] == "train" for trace in result.retrieval_traces)


def test_forbidden_document_metadata_never_enters_rag_prompt() -> None:
    marker = "DO_NOT_PROMPT_HIDDEN_GROUND_TRUTH"
    metadata = {
        "source_uri": "evalforge://docs/safe",
        "source_type": "runbook",
        "source_version": "v1",
        "source_split": "train",
        "research_eligible": True,
        "root_cause_code": marker,
        "hidden_ground_truth": marker,
    }
    backend = FakeBackend()
    result = RAGPipeline(
        backend=backend,
        retriever=FakeRetriever(
            (
                _result(
                    "chunk-safe",
                    "configuration rollback restored traffic",
                    0.9,
                    metadata=metadata,
                ),
            )
        ),
        allowed_labels=LABELS,
        top_k=1,
    ).predict(
        IncidentInput(
            incident_id="incident-1",
            title="Checkout failures",
            description="Failures began after a configuration rollout.",
        ),
        query_family_id="heldout-family",
    )

    assert marker not in result.prompt
    assert marker not in backend.prompts[0]
    assert "root_cause_code" not in result.retrieval_traces[0].provenance
    assert "hidden_ground_truth" not in result.retrieval_traces[0].provenance


def test_no_context_fallback_remains_a_valid_rag_prediction() -> None:
    backend = FakeBackend(winner="no_fault")
    result = RAGPipeline(
        backend=backend,
        retriever=FakeRetriever(()),
        allowed_labels=LABELS,
        top_k=5,
    ).predict(
        IncidentInput(
            incident_id="incident-empty",
            title="Alert only",
            description="Independent service health is normal.",
        ),
        query_family_id="family-empty",
    )

    assert result.context.no_context_fallback is True
    assert result.retrieval_traces == ()
    assert result.prediction.retrieved_chunk_ids == ()
    assert result.prediction.parse_status is ParseStatus.OK
    assert result.prediction.pipeline_metadata["no_context_fallback"] is True
    assert "no eligible retrieval context" in result.prompt


class FakeFaithfulnessClient:
    def __init__(self, score: float) -> None:
        self.value = score
        self.calls = 0

    def score(self, *, question: str, answer: str, contexts: tuple[str, ...]) -> float:
        assert question
        assert answer
        assert contexts
        self.calls += 1
        return self.value


def test_ragas_adapter_records_versions_and_handles_no_context() -> None:
    client = FakeFaithfulnessClient(0.75)
    adapter = RagasFaithfulnessAdapter(
        client=client,
        evaluator_revision="ragas-0.x-pinned",
        model_id="judge-fixture",
        model_revision="judge-rev",
    )
    scored = adapter.evaluate(
        question="What caused the incident?",
        answer="A bad payment routing configuration.",
        contexts=("Rollback restored card traffic.",),
    )
    empty = adapter.evaluate(
        question="What caused the incident?",
        answer="No fault.",
        contexts=(),
    )

    assert scored.score == 0.75
    assert scored.status == "ok"
    assert scored.metadata()["model_revision"] == "judge-rev"
    assert client.calls == 1
    assert empty.score is None
    assert empty.status == "not_applicable_no_context"


def test_ablation_suite_changes_only_one_controlled_factor() -> None:
    suite = load_ablation_suite(ROOT / "configs/phase8-rag-ablations.json")
    assert suite.selection_split == "validation"
    assert suite.locked_test_selection_forbidden is True
    assert len(suite.config_hash()) == 64

    payload = suite.model_dump(mode="python")
    candidate = next(
        variant for variant in payload["variants"] if variant["factor"] == "top_k"
    )
    candidate["max_context_tokens"] += 1
    with pytest.raises(ValueError, match="uncontrolled fields"):
        RAGAblationSuite.model_validate(payload)
