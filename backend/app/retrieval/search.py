from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import KnowledgeBaseVersion
from app.retrieval.embeddings import HashEmbeddingAdapter, validate_embedding_dimension


@dataclass(frozen=True)
class RetrievalQuery:
    kb_version: str
    text: str
    top_k: int
    research_mode: bool = True
    query_family_id: str | None = None
    source_type: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    text: str
    score: float
    source_family_id: str | None
    document_metadata: dict[str, Any]


def research_document_allowed(
    metadata: dict[str, Any],
    *,
    query_family_id: str | None,
) -> bool:
    if metadata.get("research_eligible") is not True:
        return False
    if metadata.get("source_split") in {"validation", "test"}:
        return False
    source_family_id = metadata.get("source_family_id")
    return not (query_family_id is not None and source_family_id == query_family_id)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".12g") for value in values) + "]"


def search_chunks(
    session: Session,
    adapter: HashEmbeddingAdapter,
    query: RetrievalQuery,
) -> tuple[RetrievalResult, ...]:
    if query.top_k <= 0:
        raise ValueError("top_k must be positive")
    version = session.get(KnowledgeBaseVersion, query.kb_version)
    if version is None:
        raise ValueError(f"unknown knowledge-base version: {query.kb_version}")

    metadata = version.metadata_json or {}
    expected_dimension = int(metadata.get("embedding_dimension", 0))
    if expected_dimension <= 0:
        raise ValueError("knowledge-base version is missing embedding dimension metadata")
    if version.embedding_model_id != adapter.config.model_id:
        raise ValueError("retrieval adapter model id disagrees with knowledge-base version")
    if version.embedding_model_revision != adapter.config.revision:
        raise ValueError("retrieval adapter revision disagrees with knowledge-base version")
    if adapter.dimension != expected_dimension:
        raise ValueError("retrieval adapter dimension disagrees with knowledge-base version")

    embedding = adapter.embed(query.text)
    validate_embedding_dimension(embedding, expected_dimension)

    clauses = [
        "c.kb_version = :kb_version",
        "c.embedding IS NOT NULL",
    ]
    params: dict[str, Any] = {
        "kb_version": query.kb_version,
        "query_embedding": _vector_literal(embedding),
        "top_k": query.top_k,
    }
    if query.research_mode:
        clauses.extend(
            [
                "COALESCE((d.document_metadata ->> 'research_eligible')::boolean, false) = true",
                "COALESCE(d.document_metadata ->> 'source_split', '') NOT IN ('validation','test')",
            ]
        )
        if query.query_family_id is not None:
            clauses.append("(c.source_family_id IS NULL OR c.source_family_id <> :query_family_id)")
            params["query_family_id"] = query.query_family_id
    if query.source_type is not None:
        clauses.append("d.document_metadata ->> 'source_type' = :source_type")
        params["source_type"] = query.source_type

    sql = text(
        f"""
        SELECT
            c.chunk_id,
            c.document_id,
            c.text,
            1 - (c.embedding <=> CAST(:query_embedding AS vector)) AS score,
            c.source_family_id,
            d.document_metadata
        FROM kb_chunks AS c
        JOIN kb_documents AS d
          ON d.kb_version = c.kb_version
         AND d.document_id = c.document_id
        WHERE {" AND ".join(clauses)}
        ORDER BY c.embedding <=> CAST(:query_embedding AS vector), c.chunk_id
        LIMIT :top_k
        """
    )
    rows = session.execute(sql, params).mappings().all()
    return tuple(
        RetrievalResult(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            text=str(row["text"]),
            score=float(row["score"]),
            source_family_id=row["source_family_id"],
            document_metadata=dict(row["document_metadata"]),
        )
        for row in rows
    )
