from __future__ import annotations

import math
from pathlib import Path

from app.retrieval.embeddings import EmbeddingConfig, HashEmbeddingAdapter
from app.retrieval.indexing import (
    ChunkingConfig,
    DocumentInput,
    build_index_plan,
    chunk_document,
    load_kb_config,
)
from app.retrieval.reranker import NoOpReranker, TokenOverlapReranker
from app.retrieval.search import RetrievalResult, research_document_allowed

ROOT = Path(__file__).resolve().parents[3]


def test_hash_embeddings_are_deterministic_normalized_and_dimensioned() -> None:
    config = EmbeddingConfig(
        model_id="evalforge/feature-hash-embedding",
        revision="phase7-feature-hash-v1",
        dimension=32,
        preprocessing={"lowercase": True, "include_bigrams": True, "l2_normalize": True},
    )
    adapter = HashEmbeddingAdapter(config)
    first, second = adapter.embed_batch(["Database pool leak", "Database pool leak"])

    assert first == second
    assert len(first) == 32
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_chunking_has_stable_ids_and_offsets() -> None:
    document = DocumentInput(
        document_id="doc-stable",
        source_uri="evalforge://fixture",
        source_checksum="a" * 64,
        title="fixture",
        content="one two three four five six seven eight",
        source_type="runbook",
        source_version="v1",
        source_timestamp=None,
        source_family_id=None,
        source_split=None,
        root_cause_code=None,
        research_eligible=True,
    )
    adapter = HashEmbeddingAdapter(
        EmbeddingConfig(
            model_id="fixture",
            revision="v1",
            dimension=16,
            preprocessing={"lowercase": True, "include_bigrams": True, "l2_normalize": True},
        )
    )
    chunking = ChunkingConfig(version="fixture-v1", size=4, overlap=1)
    first = chunk_document(document, chunking, adapter)
    second = chunk_document(document, chunking, adapter)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert [item.text for item in first] == [
        "one two three four",
        "four five six seven",
        "seven eight",
    ]
    assert all(item.start_offset < item.end_offset for item in first)


def test_repository_kb_manifest_is_deterministic_and_provenance_complete() -> None:
    config = load_kb_config(ROOT / "configs/phase7-kb.json")
    first = build_index_plan(ROOT, config)
    second = build_index_plan(ROOT, config)

    assert first.manifest_checksum == second.manifest_checksum
    assert [item.chunk_id for item in first.chunks] == [item.chunk_id for item in second.chunks]
    assert len(first.documents) == 24
    historical = [
        item for item in first.documents if item.source_type == "historical_incident"
    ]
    assert len(historical) == 18
    assert all(item.source_uri and item.source_version for item in first.documents)


def test_research_policy_rejects_heldout_and_same_family_documents() -> None:
    assert not research_document_allowed(
        {
            "research_eligible": False,
            "source_split": "test",
            "source_family_id": "family-test",
        },
        query_family_id="family-other",
    )
    assert not research_document_allowed(
        {
            "research_eligible": True,
            "source_split": "train",
            "source_family_id": "family-same",
        },
        query_family_id="family-same",
    )
    assert research_document_allowed(
        {
            "research_eligible": True,
            "source_split": "train",
            "source_family_id": "family-train",
        },
        query_family_id="family-other",
    )


def test_reranker_contract_handles_empty_and_score_ordering() -> None:
    low = RetrievalResult(
        chunk_id="chunk-low",
        document_id="doc-low",
        text="database connection pool",
        score=0.1,
        source_family_id=None,
        document_metadata={},
    )
    high = RetrievalResult(
        chunk_id="chunk-high",
        document_id="doc-high",
        text="memory pressure",
        score=0.9,
        source_family_id=None,
        document_metadata={},
    )

    assert NoOpReranker().rerank("query", ()) == ()
    ordered = NoOpReranker().rerank("query", (low, high))
    assert [item.chunk_id for item in ordered] == ["chunk-high", "chunk-low"]

    reranked = TokenOverlapReranker().rerank("database connection pool", (high, low))
    assert reranked[0].chunk_id == "chunk-low"
