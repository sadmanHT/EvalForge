from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.retrieval.indexing import IndexPlan, build_index_plan, load_kb_config


class Phase7Stage(str, Enum):
    AWAITING_MANUAL_REVIEW = "awaiting_manual_review"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Phase7Status:
    stage: Phase7Stage
    blocker: str | None
    kb_version: str
    manifest_checksum: str
    document_count: int
    chunk_count: int
    leakage_query_count: int
    sample_incident_ids: tuple[str, ...]
    ci_run_id: int
    ci_head_sha: str
    reviewer: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _load_incidents(root: Path, dataset_version: str) -> list[dict[str, Any]]:
    path = root / "datasets/incident_diagnosis/processed" / dataset_version / "incidents.jsonl"
    incidents: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected incident object in {path}")
        incidents.append(payload)
    return incidents


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: actual={actual!r} expected={expected!r}")


def _expected_logical_chunk_hash(plan: IndexPlan) -> str:
    chunk_ids = sorted(chunk.chunk_id for chunk in plan.chunks)
    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()


def _validate_review_payload(
    review: dict[str, Any], expected_sample_incident_ids: tuple[str, ...]
) -> str:
    if review.get("review_version") != "phase7-retrieval-review-v1":
        raise ValueError("manual retrieval review version mismatch")
    if review.get("approved") is not True:
        raise ValueError("manual retrieval review is not approved")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("manual retrieval review is missing reviewer")
    reviewed_ids = review.get("reviewed_sample_incident_ids")
    if not isinstance(reviewed_ids, list) or any(
        not isinstance(item, str) for item in reviewed_ids
    ):
        raise ValueError("manual retrieval review sample IDs are invalid")
    if tuple(reviewed_ids) != expected_sample_incident_ids:
        raise ValueError("manual retrieval review sample IDs do not match preserved samples")
    for field in (
        "relevance_reviewed",
        "provenance_reviewed",
        "leakage_guard_reviewed",
        "limitations_acknowledged",
    ):
        if review.get(field) is not True:
            raise ValueError(f"manual retrieval review field {field!r} must be true")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ValueError("manual retrieval review is missing reviewed_at")
    notes = review.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("manual retrieval review is missing notes")
    return reviewer.strip()


def _validate_manifest(evidence_dir: Path, plan: IndexPlan) -> None:
    manifest = _load_json(evidence_dir / "kb-manifest.json")
    _require_equal(manifest.get("kb_version"), plan.config.kb_version, "KB version mismatch")
    _require_equal(
        manifest.get("manifest_checksum"),
        plan.manifest_checksum,
        "KB manifest checksum mismatch",
    )
    for key in ("dataset_version", "chunking", "embedding", "reranker", "retrieval"):
        _require_equal(manifest.get(key), plan.manifest.get(key), f"KB manifest {key} mismatch")
    _require_equal(
        manifest.get("documents"),
        plan.manifest.get("documents"),
        "document manifest mismatch",
    )
    _require_equal(manifest.get("chunks"), plan.manifest.get("chunks"), "chunk manifest mismatch")
    _require_equal(manifest.get("reindex_verified"), True, "reindex verification missing")
    _require_equal(
        manifest.get("persisted_document_count"),
        len(plan.documents),
        "persisted document count mismatch",
    )
    _require_equal(
        manifest.get("persisted_chunk_count"),
        len(plan.chunks),
        "persisted chunk count mismatch",
    )


def _validate_index_evidence(evidence_dir: Path, plan: IndexPlan) -> None:
    evidence = _load_json(evidence_dir / "index-reindex.json")
    _require_equal(
        evidence.get("evidence_version"),
        "phase7-index-reindex-v1",
        "index evidence version mismatch",
    )
    _require_equal(evidence.get("kb_version"), plan.config.kb_version, "index KB mismatch")
    _require_equal(
        evidence.get("manifest_checksum"),
        plan.manifest_checksum,
        "index evidence manifest mismatch",
    )
    _require_equal(
        evidence.get("document_count"),
        len(plan.documents),
        "index document count mismatch",
    )
    _require_equal(evidence.get("chunk_count"), len(plan.chunks), "index chunk count mismatch")
    _require_equal(evidence.get("reindex_verified"), True, "index reindex verification missing")
    _require_equal(
        evidence.get("logical_chunk_ids_sha256"),
        _expected_logical_chunk_hash(plan),
        "logical chunk ID checksum mismatch",
    )


def _validate_ci_evidence(evidence_dir: Path) -> tuple[int, str, dict[str, Any]]:
    ci_validation = _load_json(evidence_dir / "ci-validation.json")
    _require_equal(
        ci_validation.get("evidence_version"),
        "phase7-ci-validation-v1",
        "CI evidence version mismatch",
    )
    workflow = ci_validation.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("Phase 7 CI workflow evidence is missing")
    _require_equal(workflow.get("verify_all_conclusion"), "success", "verify-all is not green")
    _require_equal(
        workflow.get("clean_compose_conclusion"),
        "success",
        "clean Compose is not green",
    )
    run_id = workflow.get("run_id")
    head_sha = workflow.get("head_sha")
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("Phase 7 CI run ID is invalid")
    if not isinstance(head_sha, str) or len(head_sha) != 40:
        raise ValueError("Phase 7 CI head SHA is invalid")

    for artifact_field in ("retrieval_artifact", "clean_environment_artifact"):
        artifact = ci_validation.get(artifact_field)
        if not isinstance(artifact, dict):
            raise ValueError(f"Phase 7 {artifact_field} evidence is missing")
        if not isinstance(artifact.get("artifact_id"), int):
            raise ValueError(f"Phase 7 {artifact_field} ID is invalid")
        digest = artifact.get("artifact_digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError(f"Phase 7 {artifact_field} digest is invalid")

    clean_checks = ci_validation.get("clean_environment_checks")
    if not isinstance(clean_checks, dict):
        raise ValueError("Phase 7 clean-environment checks are missing")
    for field in (
        "compose_health",
        "phase04_exit",
        "phase05_exit",
        "phase06_exit",
        "phase07_contract",
        "phase07_index",
        "phase07_leakage_audit",
    ):
        _require_equal(clean_checks.get(field), "PASS", f"clean-environment {field} failed")
    services = clean_checks.get("services_healthy")
    _require_equal(
        set(services) if isinstance(services, list) else None,
        {"backend", "frontend", "postgres", "redis", "worker"},
        "clean-environment service health mismatch",
    )
    return run_id, head_sha, ci_validation


def _validate_leakage_evidence(
    evidence_dir: Path,
    incidents: list[dict[str, Any]],
    research_policy_version: str,
    ci_validation: dict[str, Any],
) -> int:
    heldout = [item for item in incidents if item.get("split") in {"validation", "test"}]
    heldout_by_id = {str(item["incident_id"]): item for item in heldout}
    heldout_family_ids = {str(item["incident_family_id"]) for item in heldout}

    leakage = _load_json(evidence_dir / "leakage-audit.json")
    _require_equal(
        leakage.get("evidence_version"),
        "phase7-leakage-audit-preserved-v1",
        "leakage evidence version mismatch",
    )
    _require_equal(
        leakage.get("source_evidence_version"),
        "phase7-leakage-audit-v1",
        "source leakage evidence version mismatch",
    )
    _require_equal(
        leakage.get("research_policy_version"),
        research_policy_version,
        "leakage audit policy mismatch",
    )
    _require_equal(
        leakage.get("heldout_incident_count"),
        len(heldout_by_id),
        "held-out incident count mismatch",
    )
    _require_equal(
        leakage.get("heldout_family_count"),
        len(heldout_family_ids),
        "held-out family count mismatch",
    )
    _require_equal(leakage.get("query_count"), len(heldout_by_id), "leakage query count mismatch")
    _require_equal(leakage.get("violation_count"), 0, "known held-out-family leakage detected")
    _require_equal(leakage.get("violations"), [], "leakage audit violations must be empty")

    query_rows = leakage.get("queries")
    if not isinstance(query_rows, list):
        raise ValueError("leakage audit queries are invalid")
    query_incident_ids = {str(item.get("incident_id")) for item in query_rows}
    _require_equal(
        query_incident_ids,
        set(heldout_by_id),
        "leakage audit incident coverage mismatch",
    )
    for query in query_rows:
        incident_id = str(query.get("incident_id"))
        incident = heldout_by_id[incident_id]
        _require_equal(
            query.get("incident_split"),
            incident["split"],
            "leakage query split mismatch",
        )
        _require_equal(
            query.get("incident_family_id"),
            str(incident["incident_family_id"]),
            "leakage query family mismatch",
        )
        if query.get("all_results_research_eligible") is not True:
            raise ValueError("leakage audit returned a research-ineligible result")
        result_count = query.get("result_count")
        if not isinstance(result_count, int) or result_count <= 0:
            raise ValueError("leakage audit result count is invalid")
        returned_splits = query.get("returned_source_splits")
        if not isinstance(returned_splits, list) or any(
            item != "train" for item in returned_splits
        ):
            raise ValueError("leakage audit returned a non-train incident split")
        returned_families = query.get("returned_source_families")
        if not isinstance(returned_families, list):
            raise ValueError("leakage audit returned source families are invalid")
        forbidden = heldout_family_ids.intersection(str(item) for item in returned_families)
        if forbidden:
            raise ValueError(f"leakage audit returned held-out families={sorted(forbidden)}")
        record_hash = query.get("query_record_sha256")
        if not isinstance(record_hash, str) or len(record_hash) != 64:
            raise ValueError("leakage audit query record hash is invalid")

    full_report = leakage.get("full_report")
    if not isinstance(full_report, dict):
        raise ValueError("leakage audit full-report reference is missing")
    workflow = ci_validation["workflow"]
    retrieval_artifact = ci_validation["retrieval_artifact"]
    _require_equal(
        full_report.get("workflow_run_id"),
        workflow["run_id"],
        "leakage artifact run mismatch",
    )
    _require_equal(
        full_report.get("workflow_head_sha"),
        workflow["head_sha"],
        "leakage artifact head mismatch",
    )
    _require_equal(
        full_report.get("artifact_id"),
        retrieval_artifact["artifact_id"],
        "leakage artifact ID mismatch",
    )
    _require_equal(
        full_report.get("artifact_digest"),
        retrieval_artifact["artifact_digest"],
        "leakage artifact digest mismatch",
    )
    return len(query_rows)


def _validate_sample_evidence(
    evidence_dir: Path,
    plan: IndexPlan,
    research_policy_version: str,
) -> tuple[str, ...]:
    samples = _load_json(evidence_dir / "sample-retrievals.json")
    _require_equal(
        samples.get("evidence_version"),
        "phase7-sample-retrievals-v1",
        "sample retrieval evidence version mismatch",
    )
    _require_equal(
        samples.get("kb_version"), plan.config.kb_version, "sample retrieval KB mismatch"
    )
    _require_equal(
        samples.get("research_policy_version"),
        research_policy_version,
        "sample retrieval policy mismatch",
    )
    sample_rows = samples.get("samples")
    if not isinstance(sample_rows, list) or len(sample_rows) < 2:
        raise ValueError("sample retrieval evidence must contain train and validation samples")
    sample_splits = {item.get("incident_split") for item in sample_rows}
    if "train" not in sample_splits or "validation" not in sample_splits or "test" in sample_splits:
        raise ValueError("sample retrieval evidence split coverage is invalid")

    known_chunk_ids = {chunk.chunk_id for chunk in plan.chunks}
    known_document_ids = {document.document_id for document in plan.documents}
    sample_ids = tuple(str(item["incident_id"]) for item in sample_rows)
    for sample in sample_rows:
        results = sample.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("sample retrieval evidence contains an empty result set")
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("sample retrieval result is invalid")
            if result.get("research_eligible") is not True:
                raise ValueError("sample retrieval contains a research-ineligible result")
            if result.get("source_split") in {"validation", "test"}:
                raise ValueError("sample retrieval contains a held-out split result")
            if result.get("chunk_id") not in known_chunk_ids:
                raise ValueError("sample retrieval contains an unknown chunk")
            if result.get("document_id") not in known_document_ids:
                raise ValueError("sample retrieval contains an unknown document")
            source_uri = result.get("source_uri")
            if not isinstance(source_uri, str) or not source_uri.strip():
                raise ValueError("sample retrieval result is missing source_uri")
    return sample_ids


def evaluate_phase7_repository_state(root: Path) -> Phase7Status:
    config = load_kb_config(root / "configs/phase7-kb.json")
    plan = build_index_plan(root, config)
    evidence_dir = root / "evidence/phase-07"

    _validate_manifest(evidence_dir, plan)
    _validate_index_evidence(evidence_dir, plan)
    run_id, head_sha, ci_validation = _validate_ci_evidence(evidence_dir)
    incidents = _load_incidents(root, config.dataset_version)
    leakage_query_count = _validate_leakage_evidence(
        evidence_dir,
        incidents,
        config.research_policy_version,
        ci_validation,
    )
    sample_ids = _validate_sample_evidence(evidence_dir, plan, config.research_policy_version)

    review_path = evidence_dir / "retrieval-review.json"
    if not review_path.exists():
        return Phase7Status(
            stage=Phase7Stage.AWAITING_MANUAL_REVIEW,
            blocker=(
                "real KB train/validation retrieval samples require manual "
                "relevance/provenance approval"
            ),
            kb_version=config.kb_version,
            manifest_checksum=plan.manifest_checksum,
            document_count=len(plan.documents),
            chunk_count=len(plan.chunks),
            leakage_query_count=leakage_query_count,
            sample_incident_ids=sample_ids,
            ci_run_id=run_id,
            ci_head_sha=head_sha,
        )

    reviewer = _validate_review_payload(_load_json(review_path), sample_ids)
    return Phase7Status(
        stage=Phase7Stage.COMPLETE,
        blocker=None,
        kb_version=config.kb_version,
        manifest_checksum=plan.manifest_checksum,
        document_count=len(plan.documents),
        chunk_count=len(plan.chunks),
        leakage_query_count=leakage_query_count,
        sample_incident_ids=sample_ids,
        ci_run_id=run_id,
        ci_head_sha=head_sha,
        reviewer=reviewer,
    )
