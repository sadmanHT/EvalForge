#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.retrieval.indexing import build_index_plan, load_kb_config  # noqa: E402


def main() -> int:
    config = load_kb_config(ROOT / "configs/phase7-kb.json")
    first = build_index_plan(ROOT, config)
    second = build_index_plan(ROOT, config)

    if first.manifest_checksum != second.manifest_checksum:
        raise SystemExit("PHASE07_CONTRACT=FAIL KB manifest is not deterministic")
    if [item.chunk_id for item in first.chunks] != [item.chunk_id for item in second.chunks]:
        raise SystemExit("PHASE07_CONTRACT=FAIL logical chunk set is not deterministic")
    if config.embedding.dimension != 1536:
        raise SystemExit("PHASE07_CONTRACT=FAIL embedding dimension disagrees with schema")

    historical = [item for item in first.documents if item.source_type == "historical_incident"]
    heldout = [item for item in historical if item.source_split in {"validation", "test"}]
    if len(first.documents) != 24 or len(historical) != 18 or len(heldout) != 12:
        raise SystemExit(
            "PHASE07_CONTRACT=FAIL unexpected KB document counts "
            f"documents={len(first.documents)} historical={len(historical)} "
            f"heldout={len(heldout)}"
        )
    if any(item.research_eligible for item in heldout):
        raise SystemExit("PHASE07_CONTRACT=FAIL held-out incident marked research-eligible")
    if any(
        not item.source_uri or not item.source_version or not item.source_checksum
        for item in first.documents
    ):
        raise SystemExit("PHASE07_CONTRACT=FAIL document provenance is incomplete")
    if any(not item.text_checksum for item in first.chunks):
        raise SystemExit("PHASE07_CONTRACT=FAIL chunk checksum is missing")

    print("PHASE07_CONTRACT=PASS")
    print(f"KB_VERSION={config.kb_version}")
    print(f"KB_MANIFEST_CHECKSUM={first.manifest_checksum}")
    print(f"DOCUMENT_COUNT={len(first.documents)}")
    print(f"CHUNK_COUNT={len(first.chunks)}")
    print(f"HISTORICAL_INCIDENT_DOCUMENTS={len(historical)}")
    print(f"HELDOUT_INCIDENT_DOCUMENTS={len(heldout)}")
    print(f"EMBEDDING_MODEL={config.embedding.model_id}")
    print(f"EMBEDDING_REVISION={config.embedding.revision}")
    print(f"EMBEDDING_DIMENSION={config.embedding.dimension}")
    print(f"CHUNKER_VERSION={config.chunking.version}")
    print(f"CHUNK_SIZE={config.chunking.size}")
    print(f"CHUNK_OVERLAP={config.chunking.overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
