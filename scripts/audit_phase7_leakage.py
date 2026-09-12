#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db import build_engine, session_scope  # noqa: E402
from app.retrieval.embeddings import HashEmbeddingAdapter  # noqa: E402
from app.retrieval.indexing import load_kb_config  # noqa: E402
from app.retrieval.search import (  # noqa: E402
    RetrievalQuery,
    RetrievalResult,
    research_document_allowed,
    search_chunks,
)


def _load_incidents(dataset_version: str) -> list[dict[str, Any]]:
    path = ROOT / "datasets/incident_diagnosis/processed" / dataset_version / "incidents.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _result_payload(result: RetrievalResult) -> dict[str, Any]:
    metadata = result.document_metadata
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "score": result.score,
        "source_family_id": result.source_family_id,
        "source_uri": metadata.get("source_uri"),
        "source_type": metadata.get("source_type"),
        "source_split": metadata.get("source_split"),
        "root_cause_code": metadata.get("root_cause_code"),
        "research_eligible": metadata.get("research_eligible"),
        "text": result.text,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    config = load_kb_config(ROOT / "configs/phase7-kb.json")
    adapter = HashEmbeddingAdapter(config.embedding)
    incidents = _load_incidents(config.dataset_version)
    heldout_incidents = [
        item for item in incidents if item["split"] in {"validation", "test"}
    ]
    heldout_family_ids = {str(item["incident_family_id"]) for item in heldout_incidents}

    engine = build_engine()
    audit_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    try:
        with session_scope(engine) as session:
            for incident in heldout_incidents:
                family_id = str(incident["incident_family_id"])
                query_text = f"{incident['title']}\n\n{incident['description']}"
                results = search_chunks(
                    session,
                    adapter,
                    RetrievalQuery(
                        kb_version=config.kb_version,
                        text=query_text,
                        top_k=config.default_top_k,
                        research_mode=True,
                        query_family_id=family_id,
                    ),
                )
                result_payloads = [_result_payload(result) for result in results]
                audit_rows.append(
                    {
                        "incident_id": incident["incident_id"],
                        "incident_split": incident["split"],
                        "incident_family_id": family_id,
                        "result_count": len(results),
                        "results": result_payloads,
                    }
                )
                for result in results:
                    forbidden_family = result.source_family_id in heldout_family_ids
                    allowed = research_document_allowed(
                        result.document_metadata,
                        query_family_id=family_id,
                    )
                    if forbidden_family or not allowed:
                        violations.append(
                            {
                                "incident_id": incident["incident_id"],
                                "incident_family_id": family_id,
                                "returned_chunk_id": result.chunk_id,
                                "returned_family_id": result.source_family_id,
                                "returned_source_split": result.document_metadata.get(
                                    "source_split"
                                ),
                                "forbidden_family": forbidden_family,
                                "allowed_by_policy": allowed,
                            }
                        )

            sample_incidents = [
                next(item for item in incidents if item["split"] == "train"),
                next(item for item in incidents if item["split"] == "validation"),
            ]
            samples: list[dict[str, Any]] = []
            for incident in sample_incidents:
                family_id = str(incident["incident_family_id"])
                query_text = f"{incident['title']}\n\n{incident['description']}"
                results = search_chunks(
                    session,
                    adapter,
                    RetrievalQuery(
                        kb_version=config.kb_version,
                        text=query_text,
                        top_k=config.default_top_k,
                        research_mode=True,
                        query_family_id=family_id,
                    ),
                )
                samples.append(
                    {
                        "incident_id": incident["incident_id"],
                        "incident_split": incident["split"],
                        "incident_family_id": family_id,
                        "query_title": incident["title"],
                        "query_root_cause_code": incident["root_cause_code"],
                        "results": [_result_payload(result) for result in results],
                    }
                )
    finally:
        engine.dispose()

    evidence_dir = ROOT / "evidence/phase-07"
    _write_json(
        evidence_dir / "leakage-audit.json",
        {
            "evidence_version": "phase7-leakage-audit-v1",
            "kb_version": config.kb_version,
            "dataset_version": config.dataset_version,
            "research_policy_version": config.research_policy_version,
            "heldout_incident_count": len(heldout_incidents),
            "heldout_family_count": len(heldout_family_ids),
            "query_count": len(audit_rows),
            "violation_count": len(violations),
            "violations": violations,
            "queries": audit_rows,
        },
    )
    _write_json(
        evidence_dir / "sample-retrievals.json",
        {
            "evidence_version": "phase7-sample-retrievals-v1",
            "kb_version": config.kb_version,
            "research_policy_version": config.research_policy_version,
            "samples": samples,
        },
    )

    if violations:
        raise SystemExit(
            "PHASE07_LEAKAGE_AUDIT=FAIL "
            f"known held-out-family leakage violations={len(violations)}"
        )

    print("PHASE07_LEAKAGE_AUDIT=PASS")
    print(f"KB_VERSION={config.kb_version}")
    print(f"HELDOUT_INCIDENTS={len(heldout_incidents)}")
    print(f"HELDOUT_FAMILIES={len(heldout_family_ids)}")
    print(f"AUDITED_QUERIES={len(audit_rows)}")
    print("KNOWN_HELDOUT_FAMILY_LEAKAGE=0")
    print(f"SAMPLE_RETRIEVALS={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
