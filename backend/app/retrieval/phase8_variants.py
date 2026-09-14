from __future__ import annotations

from dataclasses import replace

from app.retrieval.embeddings import EmbeddingConfig
from app.retrieval.indexing import ChunkingConfig, KnowledgeBaseConfig

PHASE8_KB_VARIANT_SCHEMA_VERSION = "phase8-kb-variant-v1"
UNIGRAM_EMBEDDING_MODEL_ID = "evalforge/feature-hash-unigram-embedding"
UNIGRAM_EMBEDDING_REVISION = "phase8-feature-hash-unigram-v1"


def build_phase8_kb_variant_config(
    baseline: KnowledgeBaseConfig,
    *,
    kb_version: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    chunker_version: str,
    chunk_size: int,
    overlap: int,
) -> KnowledgeBaseConfig:
    if not kb_version:
        raise ValueError("kb_version is required")
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid Phase 08 chunking variant")

    baseline_identity = (baseline.embedding.model_id, baseline.embedding.revision)
    requested_identity = (embedding_model_id, embedding_model_revision)
    if requested_identity == baseline_identity:
        embedding = baseline.embedding
    elif requested_identity == (UNIGRAM_EMBEDDING_MODEL_ID, UNIGRAM_EMBEDDING_REVISION):
        preprocessing = dict(baseline.embedding.preprocessing)
        preprocessing["include_bigrams"] = False
        embedding = EmbeddingConfig(
            model_id=UNIGRAM_EMBEDDING_MODEL_ID,
            revision=UNIGRAM_EMBEDDING_REVISION,
            dimension=baseline.embedding.dimension,
            preprocessing=preprocessing,
        )
    else:
        raise ValueError(
            "unsupported Phase 08 embedding identity; add an explicit reproducible adapter "
            "before adding it to the controlled ablation suite"
        )

    return replace(
        baseline,
        schema_version=PHASE8_KB_VARIANT_SCHEMA_VERSION,
        kb_version=kb_version,
        embedding=embedding,
        chunking=ChunkingConfig(
            version=chunker_version,
            size=chunk_size,
            overlap=overlap,
        ),
        reranker_id=None,
        reranker_revision=None,
    )
