from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from app.evaluation.calibration import normalize_label_sequence_log_likelihoods
from app.evaluation.contracts import ConfidenceSource, ParseStatus, Prediction, RankedLabel
from app.inference.base_model import (
    BaseModelBackend,
    GenerationConfig,
    IncidentInput,
    InferenceError,
)
from app.inference.prompts import RAG_PROMPT_VERSION, get_rag_prompt_template, parse_baseline_output
from app.retrieval.embeddings import HashEmbeddingAdapter
from app.retrieval.reranker import NoOpReranker, Reranker
from app.retrieval.search import RetrievalQuery, RetrievalResult, search_chunks

RAG_CONTEXT_POLICY_VERSION = "phase8-context-budget-v1"
_CONTEXT_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int = 384
    max_chunk_tokens: int = 128
    policy_version: str = RAG_CONTEXT_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if self.max_chunk_tokens > self.max_context_tokens:
            raise ValueError("max_chunk_tokens may not exceed max_context_tokens")
        if not self.policy_version:
            raise ValueError("context budget policy_version is required")


@dataclass(frozen=True)
class ContextAssembly:
    text: str
    included_chunk_ids: tuple[str, ...]
    token_count: int
    truncated_chunk_ids: tuple[str, ...]
    chunk_token_counts: tuple[tuple[str, int], ...]
    no_context_fallback: bool


@dataclass(frozen=True)
class RAGRetrievalTrace:
    chunk_id: str
    document_id: str
    rank: int
    raw_similarity_score: float
    reranker_score: float | None
    final_score: float
    included_in_prompt: bool
    prompt_citation: str | None
    context_token_count: int
    context_truncated: bool
    provenance: Mapping[str, object]

    def persistence_metadata(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "raw_similarity_score": self.raw_similarity_score,
            "reranker_score": self.reranker_score,
            "final_score": self.final_score,
            "included_in_prompt": self.included_in_prompt,
            "prompt_citation": self.prompt_citation,
            "context_token_count": self.context_token_count,
            "context_truncated": self.context_truncated,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class RAGPipelineResult:
    prediction: Prediction
    retrieval_traces: tuple[RAGRetrievalTrace, ...]
    prompt: str
    context: ContextAssembly


class Retriever(Protocol):
    kb_version: str

    def retrieve(
        self,
        *,
        query_text: str,
        top_k: int,
        query_family_id: str | None,
    ) -> tuple[RetrievalResult, ...]: ...


class PgVectorRetriever:
    def __init__(
        self,
        *,
        session: Session,
        adapter: HashEmbeddingAdapter,
        kb_version: str,
    ) -> None:
        if not kb_version:
            raise ValueError("kb_version is required")
        self.session = session
        self.adapter = adapter
        self.kb_version = kb_version

    def retrieve(
        self,
        *,
        query_text: str,
        top_k: int,
        query_family_id: str | None,
    ) -> tuple[RetrievalResult, ...]:
        return search_chunks(
            self.session,
            self.adapter,
            RetrievalQuery(
                kb_version=self.kb_version,
                text=query_text,
                top_k=top_k,
                research_mode=True,
                query_family_id=query_family_id,
            ),
        )


def _truncate_context_text(text: str, limit: int) -> tuple[str, int, bool]:
    matches = list(_CONTEXT_TOKEN_RE.finditer(text))
    if len(matches) <= limit:
        return text.strip(), len(matches), False
    end = matches[limit - 1].end()
    return text[:end].strip(), limit, True


def assemble_context(
    results: Sequence[RetrievalResult],
    *,
    budget: ContextBudget,
) -> ContextAssembly:
    if not results:
        return ContextAssembly(
            text="",
            included_chunk_ids=(),
            token_count=0,
            truncated_chunk_ids=(),
            chunk_token_counts=(),
            no_context_fallback=True,
        )

    blocks: list[str] = []
    included: list[str] = []
    truncated: list[str] = []
    chunk_token_counts: list[tuple[str, int]] = []
    used_tokens = 0
    for result in results:
        marker_tokens = 1
        remaining = budget.max_context_tokens - used_tokens - marker_tokens
        if remaining <= 0:
            break
        per_chunk_limit = min(budget.max_chunk_tokens, remaining)
        clipped, chunk_tokens, was_truncated = _truncate_context_text(
            result.text,
            per_chunk_limit,
        )
        if chunk_tokens == 0:
            continue
        citation = f"C{len(included) + 1}"
        blocks.append(f"[{citation}] chunk_id={result.chunk_id}\n{clipped}")
        included.append(result.chunk_id)
        chunk_token_counts.append((result.chunk_id, chunk_tokens))
        used_tokens += marker_tokens + chunk_tokens
        if was_truncated:
            truncated.append(result.chunk_id)

    if not included:
        return ContextAssembly(
            text="",
            included_chunk_ids=(),
            token_count=0,
            truncated_chunk_ids=(),
            chunk_token_counts=(),
            no_context_fallback=True,
        )
    return ContextAssembly(
        text="\n\n".join(blocks),
        included_chunk_ids=tuple(included),
        token_count=used_tokens,
        truncated_chunk_ids=tuple(truncated),
        chunk_token_counts=tuple(chunk_token_counts),
        no_context_fallback=False,
    )


def _safe_provenance(result: RetrievalResult) -> dict[str, object]:
    metadata = result.document_metadata
    safe_keys = (
        "source_uri",
        "source_type",
        "source_version",
        "source_timestamp",
        "source_split",
        "research_eligible",
    )
    provenance = {key: metadata[key] for key in safe_keys if key in metadata}
    provenance["source_family_id"] = result.source_family_id
    return provenance


class RAGPipeline:
    def __init__(
        self,
        *,
        backend: BaseModelBackend,
        retriever: Retriever,
        allowed_labels: Sequence[str],
        top_k: int,
        reranker: Reranker | None = None,
        prompt_version: str = RAG_PROMPT_VERSION,
        generation_config: GenerationConfig | None = None,
        context_budget: ContextBudget | None = None,
    ) -> None:
        labels = tuple(allowed_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.backend = backend
        self.retriever = retriever
        self.allowed_labels = labels
        self.top_k = top_k
        self.reranker: Reranker = reranker or NoOpReranker()
        self.prompt_template = get_rag_prompt_template(prompt_version)
        self.generation_config = generation_config or GenerationConfig()
        self.context_budget = context_budget or ContextBudget()

    def predict(
        self,
        incident: IncidentInput,
        *,
        query_family_id: str | None,
    ) -> RAGPipelineResult:
        query_text = f"{incident.title}\n\n{incident.description}"
        raw_results = self.retriever.retrieve(
            query_text=query_text,
            top_k=self.top_k,
            query_family_id=query_family_id,
        )
        raw_by_chunk = {result.chunk_id: result for result in raw_results}
        if len(raw_by_chunk) != len(raw_results):
            raise InferenceError("retriever returned duplicate chunk IDs")

        reranked = self.reranker.rerank(query_text, tuple(raw_results))
        reranked_ids = tuple(result.chunk_id for result in reranked)
        if len(reranked_ids) != len(set(reranked_ids)):
            raise InferenceError("reranker returned duplicate chunk IDs")
        if set(reranked_ids) != set(raw_by_chunk):
            raise InferenceError("reranker changed the retrieved candidate set")

        context = assemble_context(reranked, budget=self.context_budget)
        prompt = self.prompt_template.render(
            title=incident.title,
            description=incident.description,
            allowed_labels=self.allowed_labels,
            context=context.text,
        )
        output = self.backend.generate(
            prompt=prompt,
            allowed_labels=self.allowed_labels,
            generation_config=self.generation_config,
        )
        if set(output.label_log_likelihoods) != set(self.allowed_labels):
            raise InferenceError("backend label scores do not exactly cover the frozen label set")
        probabilities = normalize_label_sequence_log_likelihoods(
            output.label_log_likelihoods,
            allowed_labels=self.allowed_labels,
        )

        metadata: dict[str, object] = {
            "pipeline_type": "RAG",
            "prompt_version": self.prompt_template.version,
            "output_schema_version": self.prompt_template.output_schema_version,
            "generation_config": self.generation_config.model_dump(mode="json"),
            "label_log_likelihoods": dict(output.label_log_likelihoods),
            "runtime": dict(output.runtime_metadata or {}),
            "knowledge_base_version": self.retriever.kb_version,
            "research_mode": True,
            "query_family_id": query_family_id,
            "top_k": self.top_k,
            "reranker_id": self.reranker.reranker_id,
            "reranker_revision": self.reranker.revision,
            "context_policy_version": self.context_budget.policy_version,
            "max_context_tokens": self.context_budget.max_context_tokens,
            "max_chunk_tokens": self.context_budget.max_chunk_tokens,
            "prompt_context": context.text,
            "prompt_context_sha256": hashlib.sha256(context.text.encode("utf-8")).hexdigest(),
            "retrieval_trace_count": len(reranked),
            "included_context_count": len(context.included_chunk_ids),
            "no_context_fallback": context.no_context_fallback,
        }
        parsed = parse_baseline_output(
            incident_id=incident.incident_id,
            raw_output=output.raw_text,
            allowed_labels=self.allowed_labels,
            latency_ms=output.latency_ms,
            pipeline_metadata=metadata,
        )
        payload = parsed.model_dump(mode="python", exclude_none=False)
        payload.update(
            {
                "input_tokens": output.input_tokens,
                "output_tokens": output.output_tokens,
                "total_tokens": output.input_tokens + output.output_tokens,
                "cost_usd": output.cost_usd,
                "label_probabilities": probabilities,
                "retrieved_chunk_ids": reranked_ids,
                "evidence_citations": context.included_chunk_ids,
            }
        )
        if parsed.parse_status is ParseStatus.OK and parsed.predicted_root_cause_code is not None:
            predicted = parsed.predicted_root_cause_code
            payload["confidence_probability"] = probabilities[predicted]
            payload["confidence_source"] = ConfidenceSource.NORMALIZED_LABEL_SEQUENCE_LOG_LIKELIHOOD
            ranked_labels = sorted(
                self.allowed_labels,
                key=lambda label: (float(output.label_log_likelihoods[label]), label),
                reverse=True,
            )
            if ranked_labels[0] == predicted:
                payload["ranked_labels"] = tuple(
                    RankedLabel(
                        label=label,
                        score=float(output.label_log_likelihoods[label]),
                        probability=probabilities[label],
                    )
                    for label in ranked_labels
                )
            else:
                pipeline_metadata = dict(payload["pipeline_metadata"])
                pipeline_metadata["ranked_labels_omitted"] = (
                    "generated_label_disagrees_with_score_argmax"
                )
                payload["pipeline_metadata"] = pipeline_metadata
        prediction = Prediction.model_validate(payload)

        included_positions = {
            chunk_id: index + 1 for index, chunk_id in enumerate(context.included_chunk_ids)
        }
        included_token_counts = dict(context.chunk_token_counts)
        traces: list[RAGRetrievalTrace] = []
        for rank, reranked_result in enumerate(reranked, start=1):
            raw = raw_by_chunk[reranked_result.chunk_id]
            position = included_positions.get(reranked_result.chunk_id)
            traces.append(
                RAGRetrievalTrace(
                    chunk_id=reranked_result.chunk_id,
                    document_id=reranked_result.document_id,
                    rank=rank,
                    raw_similarity_score=raw.score,
                    reranker_score=(
                        None if self.reranker.reranker_id == "none" else reranked_result.score
                    ),
                    final_score=reranked_result.score,
                    included_in_prompt=position is not None,
                    prompt_citation=f"C{position}" if position is not None else None,
                    context_token_count=included_token_counts.get(reranked_result.chunk_id, 0),
                    context_truncated=reranked_result.chunk_id in context.truncated_chunk_ids,
                    provenance=_safe_provenance(reranked_result),
                )
            )

        return RAGPipelineResult(
            prediction=prediction,
            retrieval_traces=tuple(traces),
            prompt=prompt,
            context=context,
        )
