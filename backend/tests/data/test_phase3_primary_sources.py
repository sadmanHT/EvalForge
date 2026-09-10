from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.data.postmortems import PostmortemCandidate, load_candidate_index
from app.data.primary_sources import (
    PrimarySourceKind,
    PrimarySourceManifest,
    PrimarySourcePlan,
    PrimarySourcePlanEntry,
    PrimarySourceSnapshot,
    git_blob_sha,
    load_primary_source_manifest,
    load_primary_source_plan,
    validate_primary_source_preservation,
    validate_snapshot_payload,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_candidate_preservation_flag_requires_complete_metadata() -> None:
    root = repository_root()
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    candidate = next(
        item for item in index.candidates if not item.original_source_snapshot_preserved
    )
    payload = candidate.model_dump(mode="python")
    payload["original_source_snapshot_preserved"] = True
    with pytest.raises(ValidationError, match="requires path, checksum, and kind"):
        PostmortemCandidate.model_validate(payload)


def test_candidate_preservation_metadata_requires_flag() -> None:
    root = repository_root()
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    candidate = next(
        item for item in index.candidates if not item.original_source_snapshot_preserved
    )
    payload = candidate.model_dump(mode="python")
    payload["preserved_primary_source_path"] = "raw/source.json"
    payload["preserved_primary_source_sha256"] = "a" * 64
    payload["preserved_primary_source_kind"] = "github_issue"
    with pytest.raises(ValidationError, match="requires preserved-source flag"):
        PostmortemCandidate.model_validate(payload)


def test_primary_source_plan_rejects_incomplete_source_identity() -> None:
    with pytest.raises(ValidationError, match="git-blob primary sources require"):
        PrimarySourcePlanEntry(
            candidate_id="candidate",
            snapshot_path="snapshot.md",
            kind=PrimarySourceKind.GIT_BLOB,
            evidence_markers=["root cause"],
        )
    with pytest.raises(ValidationError, match="GitHub issue primary sources require"):
        PrimarySourcePlanEntry(
            candidate_id="candidate",
            snapshot_path="snapshot.json",
            kind=PrimarySourceKind.GITHUB_ISSUE,
            repository="example/repo",
            evidence_markers=["root cause"],
        )
    with pytest.raises(ValidationError, match="HTTP document primary sources require"):
        PrimarySourcePlanEntry(
            candidate_id="candidate",
            snapshot_path="snapshot.html",
            kind=PrimarySourceKind.HTTP_DOCUMENT,
            evidence_markers=["root cause"],
        )


def test_primary_source_plan_rejects_duplicate_candidates_and_paths() -> None:
    entry = PrimarySourcePlanEntry(
        candidate_id="candidate",
        snapshot_path="snapshot.json",
        kind=PrimarySourceKind.GITHUB_ISSUE,
        repository="example/repo",
        issue_number=7,
        evidence_markers=["root cause"],
    )
    with pytest.raises(ValidationError, match="candidate IDs must be unique"):
        PrimarySourcePlan(plan_version="v1", entries=[entry, entry])

    second = entry.model_copy(update={"candidate_id": "candidate-2"})
    with pytest.raises(ValidationError, match="snapshot paths must be unique"):
        PrimarySourcePlan(plan_version="v1", entries=[entry, second])


def test_git_blob_payload_validates_content_identity_and_evidence() -> None:
    payload = b"N+1 query problem with 6,000+ database calls and 4 seconds latency"
    entry = PrimarySourcePlanEntry(
        candidate_id="candidate",
        snapshot_path="snapshot.md",
        kind=PrimarySourceKind.GIT_BLOB,
        repository="example/repo",
        commit="a" * 40,
        source_path="post.md",
        expected_git_blob_sha=git_blob_sha(payload),
        evidence_markers=["N+1 query problem", "6,000+ database calls"],
    )
    validate_snapshot_payload(
        entry,
        original_url="https://example.invalid/post",
        payload=payload,
    )
    bad = entry.model_copy(update={"expected_git_blob_sha": "b" * 40})
    with pytest.raises(ValueError, match="git blob SHA mismatch"):
        validate_snapshot_payload(
            bad,
            original_url="https://example.invalid/post",
            payload=payload,
        )


def test_github_issue_payload_validates_identity_and_markers() -> None:
    entry = PrimarySourcePlanEntry(
        candidate_id="candidate",
        snapshot_path="snapshot.json",
        kind=PrimarySourceKind.GITHUB_ISSUE,
        repository="example/repo",
        issue_number=9,
        evidence_markers=["N+1 pattern", "production"],
    )
    payload = json.dumps(
        {
            "html_url": "https://github.com/example/repo/issues/9",
            "number": 9,
            "title": "Production regression",
            "body": "Observed an N+1 pattern in production.",
        }
    ).encode()
    validate_snapshot_payload(
        entry,
        original_url="https://github.com/example/repo/issues/9",
        payload=payload,
    )
    with pytest.raises(ValueError, match="evidence markers missing"):
        validate_snapshot_payload(
            entry.model_copy(update={"evidence_markers": ["not present"]}),
            original_url="https://github.com/example/repo/issues/9",
            payload=payload,
        )
    with pytest.raises(ValueError, match="candidate original URL mismatch"):
        validate_snapshot_payload(
            entry,
            original_url="https://github.com/example/repo/issues/10",
            payload=payload,
        )


def test_http_document_payload_validates_visible_text_and_url() -> None:
    entry = PrimarySourcePlanEntry(
        candidate_id="candidate",
        snapshot_path="snapshot.html",
        kind=PrimarySourceKind.HTTP_DOCUMENT,
        source_url="https://example.invalid/postmortem",
        evidence_markers=["slow memory leak", "service recovered"],
    )
    payload = (
        b"<html><body><h1>Incident</h1><p>A slow <strong>memory leak</strong> "
        b"caused errors.</p><script>not evidence</script><p>Service recovered.</p></body></html>"
    )
    validate_snapshot_payload(
        entry,
        original_url="https://example.invalid/postmortem",
        payload=payload,
    )
    with pytest.raises(ValueError, match="HTTP document original URL mismatch"):
        validate_snapshot_payload(
            entry,
            original_url="https://example.invalid/other",
            payload=payload,
        )


def test_committed_primary_sources_are_checksum_valid_and_research_admitted() -> None:
    root = repository_root()
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    plan = load_primary_source_plan(root / "configs/phase3-primary-sources.json")
    manifest = load_primary_source_manifest(
        root
        / "datasets/incident_diagnosis/raw/public_incidents/"
        / "phase3-primary-source-v1/manifest.json"
    )
    report = validate_primary_source_preservation(root, index, plan, manifest)
    assert report.preserved_candidate_count == 18
    assert report.preserved_supported_family_counts_by_root_cause_code == {
        "broken_payment_configuration": 3,
        "database_connection_leak": 3,
        "disk_exhaustion": 3,
        "memory_leak": 3,
        "n_plus_one_query": 3,
        "no_fault": 3,
    }
    assert report.preserved_supported_family_depth_sufficient_for_split

    plan_ids = {entry.candidate_id for entry in plan.entries}
    preserved = [item for item in index.candidates if item.candidate_id in plan_ids]
    assert all(item.original_source_snapshot_preserved for item in preserved)
    assert all(item.research_admitted for item in preserved)


def test_primary_source_verifier_rejects_manifest_plan_mismatch() -> None:
    root = repository_root()
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    plan = load_primary_source_plan(root / "configs/phase3-primary-sources.json")
    manifest = load_primary_source_manifest(
        root
        / "datasets/incident_diagnosis/raw/public_incidents/"
        / "phase3-primary-source-v1/manifest.json"
    )
    mismatched = manifest.model_copy(update={"plan_version": "different-plan"})
    with pytest.raises(ValueError, match="plan version mismatch"):
        validate_primary_source_preservation(root, index, plan, mismatched)


def test_primary_source_manifest_rejects_duplicate_snapshots() -> None:
    snapshot = PrimarySourceSnapshot(
        candidate_id="candidate",
        original_url="https://example.invalid",
        snapshot_path="snapshot.json",
        snapshot_sha256="a" * 64,
        snapshot_kind=PrimarySourceKind.GITHUB_ISSUE,
        source_identity="github_issue:example/repo#1",
        captured_at="2026-09-09T00:00:00Z",
        media_type="application/json",
        byte_length=1,
    )
    with pytest.raises(ValidationError, match="candidate IDs must be unique"):
        PrimarySourceManifest(
            manifest_version="v1",
            plan_version="v1",
            snapshots=[snapshot, snapshot],
        )
