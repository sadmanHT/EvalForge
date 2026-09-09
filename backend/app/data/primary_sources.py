from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from app.data.postmortems import CandidateDecision, PostmortemCandidateIndex
from app.data.schemas import StrictModel


class PrimarySourceKind(StrEnum):
    GIT_BLOB = "git_blob"
    GITHUB_ISSUE = "github_issue"


class PrimarySourcePlanEntry(StrictModel):
    candidate_id: str = Field(min_length=1)
    snapshot_path: str = Field(min_length=1)
    kind: PrimarySourceKind
    repository: str = Field(min_length=3)
    evidence_markers: list[str] = Field(min_length=1)
    commit: str | None = Field(default=None, min_length=40, max_length=40)
    source_path: str | None = Field(default=None, min_length=1)
    expected_git_blob_sha: str | None = Field(default=None, min_length=40, max_length=40)
    issue_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source_identity(self) -> PrimarySourcePlanEntry:
        if len(self.evidence_markers) != len(set(self.evidence_markers)):
            raise ValueError("primary-source evidence markers must be unique")
        if self.kind == PrimarySourceKind.GIT_BLOB:
            if not self.commit or not self.source_path or not self.expected_git_blob_sha:
                raise ValueError(
                    "git-blob primary sources require commit, source_path, and blob SHA"
                )
            if self.issue_number is not None:
                raise ValueError("git-blob primary sources must not define issue_number")
        elif self.kind == PrimarySourceKind.GITHUB_ISSUE:
            if self.issue_number is None:
                raise ValueError("GitHub issue primary sources require issue_number")
            if self.commit or self.source_path or self.expected_git_blob_sha:
                raise ValueError("GitHub issue primary sources must not define git-blob fields")
        return self


class PrimarySourcePlan(StrictModel):
    plan_version: str = Field(min_length=1)
    entries: list[PrimarySourcePlanEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> PrimarySourcePlan:
        candidate_ids = [item.candidate_id for item in self.entries]
        paths = [item.snapshot_path for item in self.entries]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("primary-source plan candidate IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("primary-source plan snapshot paths must be unique")
        return self


class PrimarySourceSnapshot(StrictModel):
    candidate_id: str = Field(min_length=1)
    original_url: str = Field(min_length=1)
    snapshot_path: str = Field(min_length=1)
    snapshot_sha256: str = Field(min_length=64, max_length=64)
    snapshot_kind: PrimarySourceKind
    source_identity: str = Field(min_length=1)
    captured_at: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    byte_length: int = Field(gt=0)


class PrimarySourceManifest(StrictModel):
    manifest_version: str = Field(min_length=1)
    plan_version: str = Field(min_length=1)
    snapshots: list[PrimarySourceSnapshot] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_snapshots(self) -> PrimarySourceManifest:
        candidate_ids = [item.candidate_id for item in self.snapshots]
        paths = [item.snapshot_path for item in self.snapshots]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("primary-source manifest candidate IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("primary-source manifest snapshot paths must be unique")
        return self


class PrimarySourcePreservationReport(StrictModel):
    plan_version: str
    manifest_version: str
    preserved_candidate_count: int = Field(ge=0)
    preserved_supported_family_counts_by_root_cause_code: dict[str, int]
    preserved_supported_family_depth_sufficient_for_split: bool
    candidate_ids: list[str]


def load_primary_source_plan(path: Path) -> PrimarySourcePlan:
    return PrimarySourcePlan.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_primary_source_manifest(path: Path) -> PrimarySourceManifest:
    return PrimarySourceManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    # Git object identity is defined by SHA-1 even though security hashes use SHA-256.
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def source_evidence_text(payload: bytes, kind: PrimarySourceKind) -> str:
    if kind == PrimarySourceKind.GIT_BLOB:
        return payload.decode("utf-8")
    issue = json.loads(payload)
    title = issue.get("title")
    body = issue.get("body")
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("GitHub issue snapshot must contain string title and body")
    return f"{title}\n{body}"


def validate_snapshot_payload(
    entry: PrimarySourcePlanEntry,
    *,
    original_url: str,
    payload: bytes,
) -> None:
    if not payload:
        raise ValueError(f"empty primary-source snapshot for {entry.candidate_id}")
    evidence_text = source_evidence_text(payload, entry.kind).casefold()
    missing = [
        marker for marker in entry.evidence_markers if marker.casefold() not in evidence_text
    ]
    if missing:
        raise ValueError(
            f"primary-source evidence markers missing for {entry.candidate_id}: {missing}"
        )

    if entry.kind == PrimarySourceKind.GIT_BLOB:
        actual_blob_sha = git_blob_sha(payload)
        if actual_blob_sha != entry.expected_git_blob_sha:
            raise ValueError(
                f"git blob SHA mismatch for {entry.candidate_id}: "
                f"expected={entry.expected_git_blob_sha} actual={actual_blob_sha}"
            )
    else:
        issue = json.loads(payload)
        expected_url = f"https://github.com/{entry.repository}/issues/{entry.issue_number}"
        if original_url != expected_url:
            raise ValueError(
                f"candidate original URL mismatch for {entry.candidate_id}: "
                f"expected={expected_url} actual={original_url}"
            )
        if issue.get("html_url") != expected_url:
            raise ValueError(f"GitHub issue snapshot URL mismatch for {entry.candidate_id}")
        if issue.get("number") != entry.issue_number:
            raise ValueError(f"GitHub issue snapshot number mismatch for {entry.candidate_id}")


def validate_primary_source_preservation(
    root: Path,
    index: PostmortemCandidateIndex,
    plan: PrimarySourcePlan,
    manifest: PrimarySourceManifest,
) -> PrimarySourcePreservationReport:
    if manifest.plan_version != plan.plan_version:
        raise ValueError("primary-source manifest plan version mismatch")

    by_candidate = {item.candidate_id: item for item in index.candidates}
    plan_by_candidate = {item.candidate_id: item for item in plan.entries}
    snapshots_by_candidate = {item.candidate_id: item for item in manifest.snapshots}
    if set(plan_by_candidate) != set(snapshots_by_candidate):
        raise ValueError("primary-source manifest candidates must exactly match preservation plan")

    preserved_counts: Counter[str] = Counter()
    validated_ids: list[str] = []
    for candidate_id in sorted(plan_by_candidate):
        entry = plan_by_candidate[candidate_id]
        snapshot = snapshots_by_candidate[candidate_id]
        candidate = by_candidate.get(candidate_id)
        if candidate is None:
            raise ValueError(f"unknown primary-source candidate: {candidate_id}")
        if candidate.decision != CandidateDecision.SUPPORTED:
            raise ValueError(
                f"primary-source preservation requires supported candidate: {candidate_id}"
            )
        if not candidate.original_source_snapshot_preserved:
            raise ValueError(f"candidate not marked source-preserved: {candidate_id}")
        if candidate.preserved_primary_source_path != snapshot.snapshot_path:
            raise ValueError(f"candidate preserved-source path mismatch: {candidate_id}")
        if candidate.preserved_primary_source_sha256 != snapshot.snapshot_sha256:
            raise ValueError(f"candidate preserved-source checksum mismatch: {candidate_id}")
        if candidate.preserved_primary_source_kind != snapshot.snapshot_kind.value:
            raise ValueError(f"candidate preserved-source kind mismatch: {candidate_id}")
        if snapshot.original_url != candidate.original_url:
            raise ValueError(f"primary-source original URL mismatch: {candidate_id}")
        if snapshot.snapshot_path != entry.snapshot_path:
            raise ValueError(f"primary-source planned path mismatch: {candidate_id}")
        if snapshot.snapshot_kind != entry.kind:
            raise ValueError(f"primary-source kind mismatch: {candidate_id}")

        snapshot_path = root / snapshot.snapshot_path
        payload = snapshot_path.read_bytes()
        if len(payload) != snapshot.byte_length:
            raise ValueError(f"primary-source byte length mismatch: {candidate_id}")
        actual_sha = sha256_bytes(payload)
        if actual_sha != snapshot.snapshot_sha256:
            raise ValueError(
                f"primary-source SHA-256 mismatch for {candidate_id}: "
                f"expected={snapshot.snapshot_sha256} actual={actual_sha}"
            )
        validate_snapshot_payload(entry, original_url=candidate.original_url, payload=payload)
        preserved_counts[candidate.proposed_root_cause_code] += 1
        validated_ids.append(candidate_id)

    taxonomy_codes = {
        item.proposed_root_cause_code
        for item in index.candidates
        if item.decision == CandidateDecision.SUPPORTED
    }
    counts_by_code = {code: preserved_counts.get(code, 0) for code in sorted(taxonomy_codes)}
    return PrimarySourcePreservationReport(
        plan_version=plan.plan_version,
        manifest_version=manifest.manifest_version,
        preserved_candidate_count=len(validated_ids),
        preserved_supported_family_counts_by_root_cause_code=counts_by_code,
        preserved_supported_family_depth_sufficient_for_split=all(
            count >= 3 for count in counts_by_code.values()
        ),
        candidate_ids=validated_ids,
    )
