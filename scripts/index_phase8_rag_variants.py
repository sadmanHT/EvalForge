#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db import build_engine, session_scope  # noqa: E402
from app.inference.rag_ablations import RAGVariant  # noqa: E402
from app.inference.rag_protocol import load_rag_protocol  # noqa: E402
from app.retrieval.indexing import build_index_plan, load_kb_config, persist_index  # noqa: E402
from app.retrieval.phase8_variants import build_phase8_kb_variant_config  # noqa: E402


def _variant_index_identity(variant: RAGVariant) -> tuple[object, ...]:
    return (
        variant.knowledge_base_version,
        variant.embedding_model_id,
        variant.embedding_model_revision,
        variant.chunker_version,
        variant.chunk_size,
        variant.overlap,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evidence/phase-08/runtime",
    )
    args = parser.parse_args()

    protocol, suite = load_rag_protocol(ROOT)
    baseline = load_kb_config(ROOT / "configs/phase7-kb.json")
    index_variants: dict[str, RAGVariant] = {}
    for variant in suite.variants:
        existing = index_variants.get(variant.knowledge_base_version)
        if existing is not None:
            if _variant_index_identity(existing) != _variant_index_identity(variant):
                raise SystemExit(
                    "PHASE08_RAG_VARIANT_INDEX=FAIL shared KB version has "
                    "inconsistent index identity"
                )
            continue
        index_variants[variant.knowledge_base_version] = variant

    engine = build_engine()
    records: list[dict[str, Any]] = []
    try:
        for kb_version in sorted(index_variants):
            variant = index_variants[kb_version]
            config = build_phase8_kb_variant_config(
                baseline,
                kb_version=variant.knowledge_base_version,
                embedding_model_id=variant.embedding_model_id,
                embedding_model_revision=variant.embedding_model_revision,
                chunker_version=variant.chunker_version,
                chunk_size=variant.chunk_size,
                overlap=variant.overlap,
            )
            plan = build_index_plan(ROOT, config)
            with session_scope(engine) as session:
                document_count, chunk_count = persist_index(session, plan)
            records.append(
                {
                    "kb_version": kb_version,
                    "embedding_model_id": config.embedding.model_id,
                    "embedding_model_revision": config.embedding.revision,
                    "embedding_dimension": config.embedding.dimension,
                    "embedding_preprocessing": config.embedding.preprocessing,
                    "chunker_version": config.chunking.version,
                    "chunk_size": config.chunking.size,
                    "overlap": config.chunking.overlap,
                    "document_count": document_count,
                    "chunk_count": chunk_count,
                    "manifest_checksum": plan.manifest_checksum,
                }
            )
    finally:
        engine.dispose()

    payload = {
        "evidence_version": "phase8-rag-variant-index-v1",
        "protocol_version": protocol.protocol_version,
        "ablation_suite_hash": suite.config_hash(),
        "indexed_kb_count": len(records),
        "knowledge_bases": records,
    }
    output = args.output_dir / "rag-variant-index.json"
    _write_json(output, payload)

    print("PHASE08_RAG_VARIANT_INDEX=PASS")
    print(f"ABLATION_SUITE_HASH={suite.config_hash()}")
    print(f"INDEXED_KB_COUNT={len(records)}")
    for record in records:
        print(
            "KB="
            f"{record['kb_version']} documents={record['document_count']} "
            f"chunks={record['chunk_count']} manifest={record['manifest_checksum']}"
        )
    print(f"EVIDENCE_PATH={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
