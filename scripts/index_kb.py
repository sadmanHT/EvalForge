#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sqlalchemy import select  # noqa: E402

from app.db import build_engine, session_scope  # noqa: E402
from app.models import KBChunk  # noqa: E402
from app.retrieval.indexing import build_index_plan, load_kb_config, persist_index  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    config = load_kb_config(ROOT / "configs/phase7-kb.json")
    plan = build_index_plan(ROOT, config)
    engine = build_engine()
    try:
        with session_scope(engine) as session:
            first_counts = persist_index(session, plan)
        with session_scope(engine) as session:
            first_chunk_ids = tuple(
                session.scalars(
                    select(KBChunk.chunk_id)
                    .where(KBChunk.kb_version == config.kb_version)
                    .order_by(KBChunk.chunk_id)
                )
            )
            second_counts = persist_index(session, plan)
        with session_scope(engine) as session:
            second_chunk_ids = tuple(
                session.scalars(
                    select(KBChunk.chunk_id)
                    .where(KBChunk.kb_version == config.kb_version)
                    .order_by(KBChunk.chunk_id)
                )
            )
    finally:
        engine.dispose()

    if first_counts != second_counts:
        raise SystemExit("PHASE07_INDEX=FAIL reindex changed document/chunk counts")
    if first_chunk_ids != second_chunk_ids:
        raise SystemExit("PHASE07_INDEX=FAIL reindex changed logical chunk IDs")
    if len(first_chunk_ids) != len(plan.chunks):
        raise SystemExit("PHASE07_INDEX=FAIL persisted chunk count disagrees with manifest")

    evidence_dir = ROOT / "evidence/phase-07"
    manifest_payload = dict(plan.manifest)
    manifest_payload["reindex_verified"] = True
    manifest_payload["persisted_document_count"] = first_counts[0]
    manifest_payload["persisted_chunk_count"] = first_counts[1]
    _write_json(evidence_dir / "kb-manifest.json", manifest_payload)

    chunk_ids_sha256 = hashlib.sha256(
        "\n".join(first_chunk_ids).encode("utf-8")
    ).hexdigest()
    _write_json(
        evidence_dir / "index-reindex.json",
        {
            "evidence_version": "phase7-index-reindex-v1",
            "kb_version": config.kb_version,
            "manifest_checksum": plan.manifest_checksum,
            "document_count": first_counts[0],
            "chunk_count": first_counts[1],
            "logical_chunk_ids_sha256": chunk_ids_sha256,
            "reindex_verified": True,
        },
    )

    print("PHASE07_INDEX=PASS")
    print(f"KB_VERSION={config.kb_version}")
    print(f"KB_MANIFEST_CHECKSUM={plan.manifest_checksum}")
    print(f"DOCUMENT_COUNT={first_counts[0]}")
    print(f"CHUNK_COUNT={first_counts[1]}")
    print(f"LOGICAL_CHUNK_IDS_SHA256={chunk_ids_sha256}")
    print("REINDEX_VERIFIED=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
