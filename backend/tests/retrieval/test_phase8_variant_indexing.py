from __future__ import annotations

from pathlib import Path

import pytest

from app.inference.rag_ablations import load_ablation_suite
from app.retrieval.embeddings import HashEmbeddingAdapter
from app.retrieval.indexing import load_kb_config
from app.retrieval.phase8_variants import (
    UNIGRAM_EMBEDDING_MODEL_ID,
    UNIGRAM_EMBEDDING_REVISION,
    build_phase8_kb_variant_config,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase8_embedding_ablation_is_reproducible_and_dimension_compatible() -> None:
    baseline = load_kb_config(ROOT / "configs/phase7-kb.json")
    suite = load_ablation_suite(ROOT / "configs/phase8-rag-ablations.json")
    variant = next(item for item in suite.variants if item.factor == "embedding")

    assert variant.embedding_model_id == UNIGRAM_EMBEDDING_MODEL_ID
    assert variant.embedding_model_revision == UNIGRAM_EMBEDDING_REVISION
    config = build_phase8_kb_variant_config(
        baseline,
        kb_version=variant.knowledge_base_version,
        embedding_model_id=variant.embedding_model_id,
        embedding_model_revision=variant.embedding_model_revision,
        chunker_version=variant.chunker_version,
        chunk_size=variant.chunk_size,
        overlap=variant.overlap,
    )

    assert config.embedding.dimension == baseline.embedding.dimension == 1536
    assert config.embedding.preprocessing["include_bigrams"] is False
    assert HashEmbeddingAdapter(config.embedding).embed("payment routing failure") != (
        HashEmbeddingAdapter(baseline.embedding).embed("payment routing failure")
    )


def test_phase8_variant_factory_rejects_unimplemented_embedding_identity() -> None:
    baseline = load_kb_config(ROOT / "configs/phase7-kb.json")
    with pytest.raises(ValueError, match="explicit reproducible adapter"):
        build_phase8_kb_variant_config(
            baseline,
            kb_version="unsupported-kb",
            embedding_model_id="sentence-transformers/not-installed",
            embedding_model_revision="unsupported",
            chunker_version=baseline.chunking.version,
            chunk_size=baseline.chunking.size,
            overlap=baseline.chunking.overlap,
        )
