from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from app.models import KBDocument, KnowledgeBaseVersion
from app.retrieval.embeddings import (
    EmbeddingConfig,
    HashEmbeddingAdapter,
    validate_embedding_dimension,
)

_TOKEN_SPAN_RE = re.compile(r"\S+")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class ChunkingConfig:
    version: str
    size: int
    overlap: int


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    schema_version: str
    kb_version: str
    dataset_version: str
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    distance: str
    default_top_k: int
    research_policy_version: str
    reranker_id: str | None
    reranker_revision: str | None


@dataclass(frozen=True)
class DocumentInput:
    document_id: str
    source_uri: str
    source_checksum: str
    title: str
    content: str
    source_type: str
    source_version: str
    source_timestamp: str | None
    source_family_id: str | None
    source_split: str | None
    root_cause_code: str | None
    research_eligible: bool


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    text_checksum: str
    source_family_id: str | None
    embedding: list[float]


@dataclass(frozen=True)
class IndexPlan:
    config: KnowledgeBaseConfig
    documents: tuple[DocumentInput, ...]
    chunks: tuple[ChunkRecord, ...]
    manifest: dict[str, Any]
    manifest_checksum: str


def load_kb_config(path: Path) -> KnowledgeBaseConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    embedding = payload["embedding"]
    chunking = payload["chunking"]
    retrieval = payload["retrieval"]
    reranker = payload.get("reranker") or {}

    config = KnowledgeBaseConfig(
        schema_version=str(payload["schema_version"]),
        kb_version=str(payload["kb_version"]),
        dataset_version=str(payload["dataset_version"]),
        embedding=EmbeddingConfig(
            model_id=str(embedding["model_id"]),
            revision=str(embedding["revision"]),
            dimension=int(embedding["dimension"]),
            preprocessing=dict(embedding["preprocessing"]),
        ),
        chunking=ChunkingConfig(
            version=str(chunking["version"]),
            size=int(chunking["size"]),
            overlap=int(chunking["overlap"]),
        ),
        distance=str(retrieval["distance"]),
        default_top_k=int(retrieval["default_top_k"]),
        research_policy_version=str(retrieval["research_policy_version"]),
        reranker_id=reranker.get("id"),
        reranker_revision=reranker.get("revision"),
    )
    if config.chunking.size <= 0:
        raise ValueError("chunk size must be positive")
    if config.chunking.overlap < 0 or config.chunking.overlap >= config.chunking.size:
        raise ValueError("chunk overlap must satisfy 0 <= overlap < size")
    if config.distance != "cosine":
        raise ValueError("Phase 07 baseline supports cosine distance only")
    if config.default_top_k <= 0:
        raise ValueError("default_top_k must be positive")
    return config


def _document_id(source_uri: str, source_checksum: str) -> str:
    material = source_uri + "\0" + source_checksum
    return f"doc-{_sha256_text(material)[:32]}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_phase7_documents(root: Path, config: KnowledgeBaseConfig) -> tuple[DocumentInput, ...]:
    generic_path = root / "knowledge_base/phase7/generic-documents.jsonl"
    incident_path = (
        root
        / "datasets/incident_diagnosis/processed"
        / config.dataset_version
        / "incidents.jsonl"
    )

    documents: list[DocumentInput] = []
    for row in _load_jsonl(generic_path):
        content = str(row["content"]).strip()
        source_uri = str(row["source_uri"])
        checksum = _sha256_text(content)
        documents.append(
            DocumentInput(
                document_id=_document_id(source_uri, checksum),
                source_uri=source_uri,
                source_checksum=checksum,
                title=str(row["title"]),
                content=content,
                source_type=str(row["source_type"]),
                source_version=str(row["source_version"]),
                source_timestamp=row.get("source_timestamp"),
                source_family_id=None,
                source_split=None,
                root_cause_code=row.get("root_cause_code"),
                research_eligible=True,
            )
        )

    for row in _load_jsonl(incident_path):
        metadata = row.get("metadata") or {}
        provenance = row.get("source_provenance") or {}
        source_uri = str(
            metadata.get("original_url")
            or provenance.get("source_record_id")
            or f"dataset://{config.dataset_version}/{row['incident_id']}"
        )
        content = f"{row['title']}\n\n{row['description']}".strip()
        checksum = str(provenance.get("source_checksum") or _sha256_text(content))
        split = str(row["split"])
        documents.append(
            DocumentInput(
                document_id=_document_id(source_uri, checksum),
                source_uri=source_uri,
                source_checksum=checksum,
                title=str(row["title"]),
                content=content,
                source_type="historical_incident",
                source_version=checksum,
                source_timestamp=row.get("started_at"),
                source_family_id=str(row["incident_family_id"]),
                source_split=split,
                root_cause_code=str(row["root_cause_code"]),
                research_eligible=split == "train",
            )
        )

    return tuple(sorted(documents, key=lambda item: item.document_id))


def chunk_document(
    document: DocumentInput,
    chunking: ChunkingConfig,
    adapter: HashEmbeddingAdapter,
) -> tuple[ChunkRecord, ...]:
    matches = list(_TOKEN_SPAN_RE.finditer(document.content))
    if not matches:
        return ()
    step = chunking.size - chunking.overlap
    chunks: list[ChunkRecord] = []
    chunk_index = 0
    start_token = 0
    while start_token < len(matches):
        end_token = min(start_token + chunking.size, len(matches))
        start_offset = matches[start_token].start()
        end_offset = matches[end_token - 1].end()
        chunk_text = document.content[start_offset:end_offset]
        text_checksum = _sha256_text(chunk_text)
        chunk_id = "chunk-" + _sha256_text(
            f"{document.document_id}:{start_offset}:{end_offset}:{text_checksum}"
        )[:32]
        embedding = adapter.embed(chunk_text)
        validate_embedding_dimension(embedding, adapter.dimension)
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document.document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                start_offset=start_offset,
                end_offset=end_offset,
                text_checksum=text_checksum,
                source_family_id=document.source_family_id,
                embedding=embedding,
            )
        )
        if end_token == len(matches):
            break
        start_token += step
        chunk_index += 1
    return tuple(chunks)


def build_index_plan(root: Path, config: KnowledgeBaseConfig) -> IndexPlan:
    adapter = HashEmbeddingAdapter(config.embedding)
    documents = load_phase7_documents(root, config)
    chunks = tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, config.chunking, adapter)
    )
    manifest_body = {
        "manifest_version": "phase7-kb-manifest-v1",
        "kb_version": config.kb_version,
        "dataset_version": config.dataset_version,
        "embedding": {
            "model_id": config.embedding.model_id,
            "revision": config.embedding.revision,
            "dimension": config.embedding.dimension,
            "preprocessing": config.embedding.preprocessing,
        },
        "chunking": {
            "version": config.chunking.version,
            "size": config.chunking.size,
            "overlap": config.chunking.overlap,
        },
        "retrieval": {
            "distance": config.distance,
            "default_top_k": config.default_top_k,
            "research_policy_version": config.research_policy_version,
        },
        "reranker": {
            "id": config.reranker_id,
            "revision": config.reranker_revision,
        },
        "documents": [
            {
                "document_id": item.document_id,
                "source_uri": item.source_uri,
                "source_checksum": item.source_checksum,
                "source_type": item.source_type,
                "source_version": item.source_version,
                "source_timestamp": item.source_timestamp,
                "source_family_id": item.source_family_id,
                "source_split": item.source_split,
                "root_cause_code": item.root_cause_code,
                "research_eligible": item.research_eligible,
            }
            for item in documents
        ],
        "chunks": [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "chunk_index": item.chunk_index,
                "start_offset": item.start_offset,
                "end_offset": item.end_offset,
                "text_checksum": item.text_checksum,
                "source_family_id": item.source_family_id,
            }
            for item in chunks
        ],
    }
    checksum = _sha256_text(_canonical_json(manifest_body))
    manifest = {**manifest_body, "manifest_checksum": checksum}
    return IndexPlan(
        config=config,
        documents=documents,
        chunks=chunks,
        manifest=manifest,
        manifest_checksum=checksum,
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".12g") for value in values) + "]"


def persist_index(session: Session, plan: IndexPlan) -> tuple[int, int]:
    existing = session.get(KnowledgeBaseVersion, plan.config.kb_version)
    if existing is None:
        session.add(
            KnowledgeBaseVersion(
                kb_version=plan.config.kb_version,
                embedding_model_id=plan.config.embedding.model_id,
                embedding_model_revision=plan.config.embedding.revision,
                chunker_version=plan.config.chunking.version,
                chunk_size=plan.config.chunking.size,
                overlap=plan.config.chunking.overlap,
                reranker_id=plan.config.reranker_id,
                reranker_revision=plan.config.reranker_revision,
                manifest_checksum=plan.manifest_checksum,
                metadata_json={
                    "schema_version": plan.config.schema_version,
                    "dataset_version": plan.config.dataset_version,
                    "embedding_dimension": plan.config.embedding.dimension,
                    "embedding_preprocessing": plan.config.embedding.preprocessing,
                    "distance": plan.config.distance,
                    "default_top_k": plan.config.default_top_k,
                    "research_policy_version": plan.config.research_policy_version,
                },
            )
        )
        session.flush()
    else:
        if existing.manifest_checksum != plan.manifest_checksum:
            raise ValueError("knowledge-base version exists with a different manifest checksum")
        if existing.embedding_model_id != plan.config.embedding.model_id:
            raise ValueError("knowledge-base embedding model id disagrees with config")
        if existing.embedding_model_revision != plan.config.embedding.revision:
            raise ValueError("knowledge-base embedding model revision disagrees with config")

    session.execute(
        text("DELETE FROM kb_chunks WHERE kb_version = :kb_version"),
        {"kb_version": plan.config.kb_version},
    )
    session.execute(delete(KBDocument).where(KBDocument.kb_version == plan.config.kb_version))
    session.flush()

    documents_by_id = {item.document_id: item for item in plan.documents}
    for document in plan.documents:
        session.add(
            KBDocument(
                kb_version=plan.config.kb_version,
                document_id=document.document_id,
                source_uri=document.source_uri,
                source_checksum=document.source_checksum,
                title=document.title,
                document_metadata={
                    "source_uri": document.source_uri,
                    "source_type": document.source_type,
                    "source_version": document.source_version,
                    "source_timestamp": document.source_timestamp,
                    "source_family_id": document.source_family_id,
                    "source_split": document.source_split,
                    "root_cause_code": document.root_cause_code,
                    "research_eligible": document.research_eligible,
                },
            )
        )
    session.flush()

    insert_chunk = text(
        """
        INSERT INTO kb_chunks (
            kb_version, chunk_id, document_id, chunk_index, text, embedding,
            source_family_id, chunk_metadata
        ) VALUES (
            :kb_version, :chunk_id, :document_id, :chunk_index, :text,
            CAST(:embedding AS vector), :source_family_id, CAST(:chunk_metadata AS jsonb)
        )
        """
    )
    for chunk in plan.chunks:
        document = documents_by_id[chunk.document_id]
        session.execute(
            insert_chunk,
            {
                "kb_version": plan.config.kb_version,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": _vector_literal(chunk.embedding),
                "source_family_id": chunk.source_family_id,
                "chunk_metadata": _canonical_json(
                    {
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "text_checksum": chunk.text_checksum,
                        "source_type": document.source_type,
                        "source_split": document.source_split,
                        "research_eligible": document.research_eligible,
                    }
                ),
            },
        )
    session.flush()
    return len(plan.documents), len(plan.chunks)
