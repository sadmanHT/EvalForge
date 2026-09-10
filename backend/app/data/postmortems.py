from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from app.data.schemas import StrictModel
from app.data.taxonomy import LabelTaxonomy


class CandidateDecision(StrEnum):
    SUPPORTED = "supported_candidate_mapping"
    REJECTED = "rejected_as_primary_mapping"
    AMBIGUOUS = "ambiguous_mapping"
    INSUFFICIENT = "insufficient_evidence"


class PostmortemCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    original_url: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_blob_sha: str = Field(min_length=40, max_length=40)
    evidence_repository: str | None = Field(default=None, min_length=1)
    evidence_commit: str | None = Field(default=None, min_length=40, max_length=40)
    proposed_root_cause_code: str = Field(min_length=1)
    decision: CandidateDecision
    mapping_basis: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    decision_reason: str = Field(min_length=1)
    original_source_snapshot_preserved: bool
    preserved_primary_source_path: str | None = Field(default=None, min_length=1)
    preserved_primary_source_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    preserved_primary_source_kind: str | None = Field(default=None, min_length=1)
    research_admitted: bool

    @model_validator(mode="after")
    def validate_admission(self) -> PostmortemCandidate:
        if (self.evidence_repository is None) != (self.evidence_commit is None):
            raise ValueError("candidate evidence repository and commit must be paired")
        if (
            self.decision == CandidateDecision.SUPPORTED
            and self.mapping_basis != "narrative_root_cause"
        ):
            raise ValueError("supported mappings require narrative_root_cause basis")

        preservation_fields = (
            self.preserved_primary_source_path,
            self.preserved_primary_source_sha256,
            self.preserved_primary_source_kind,
        )
        if self.original_source_snapshot_preserved:
            if any(value is None for value in preservation_fields):
                raise ValueError(
                    "preserved original-source evidence requires path, checksum, and kind"
                )
        elif any(value is not None for value in preservation_fields):
            raise ValueError("primary-source preservation metadata requires preserved-source flag")

        if self.research_admitted:
            if self.decision != CandidateDecision.SUPPORTED:
                raise ValueError("research admission requires a supported mapping")
            if not self.original_source_snapshot_preserved:
                raise ValueError("research admission requires preserved original-source evidence")
        return self


class PostmortemCandidateIndex(StrictModel):
    index_version: str = Field(min_length=1)
    source: dict[str, str]
    candidates: list[PostmortemCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> PostmortemCandidateIndex:
        ids = [item.candidate_id for item in self.candidates]
        paths = [item.source_path for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("postmortem candidate IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("postmortem source paths must be unique")
        return self


class PostmortemCoverageReport(StrictModel):
    source_repository: str
    source_commit: str
    candidate_count: int
    independent_candidate_family_count: int
    decision_counts: dict[str, int]
    supported_mapping_count: int
    supported_root_cause_codes: list[str]
    supported_family_counts_by_root_cause_code: dict[str, int]
    supported_family_depth_sufficient_for_split: bool
    preserved_original_source_count: int
    preserved_family_counts_by_root_cause_code: dict[str, int]
    preserved_family_depth_sufficient_for_split: bool
    missing_root_cause_codes: list[str]
    admitted_research_record_count: int
    taxonomy_coverage_sufficient_for_locked_holdout: bool
    blocker: str
    notes: list[str]


def load_candidate_index(path: Path) -> PostmortemCandidateIndex:
    return PostmortemCandidateIndex.model_validate(json.loads(path.read_text(encoding="utf-8")))


def inspect_candidate_coverage(
    index: PostmortemCandidateIndex,
    taxonomy: LabelTaxonomy,
) -> PostmortemCoverageReport:
    taxonomy_codes = set(taxonomy.label_to_category)
    for candidate in index.candidates:
        if candidate.proposed_root_cause_code not in taxonomy_codes:
            raise ValueError(
                f"unknown candidate root-cause code: {candidate.proposed_root_cause_code}"
            )

    supported = {
        item.proposed_root_cause_code
        for item in index.candidates
        if item.decision == CandidateDecision.SUPPORTED
    }
    admitted = [item for item in index.candidates if item.research_admitted]
    preserved = [
        item
        for item in index.candidates
        if item.decision == CandidateDecision.SUPPORTED and item.original_source_snapshot_preserved
    ]
    decision_counts = Counter(item.decision.value for item in index.candidates)
    supported_counts = Counter(
        item.proposed_root_cause_code
        for item in index.candidates
        if item.decision == CandidateDecision.SUPPORTED
    )
    preserved_counts = Counter(item.proposed_root_cause_code for item in preserved)
    supported_family_counts = {
        code: supported_counts.get(code, 0) for code in sorted(taxonomy_codes)
    }
    preserved_family_counts = {
        code: preserved_counts.get(code, 0) for code in sorted(taxonomy_codes)
    }
    supported_family_depth_sufficient = all(
        count >= 3 for count in supported_family_counts.values()
    )
    preserved_family_depth_sufficient = all(
        count >= 3 for count in preserved_family_counts.values()
    )
    missing = sorted(taxonomy_codes - supported)
    return PostmortemCoverageReport(
        source_repository=index.source["repository"],
        source_commit=index.source["commit"],
        candidate_count=len(index.candidates),
        independent_candidate_family_count=len(index.candidates),
        decision_counts=dict(sorted(decision_counts.items())),
        supported_mapping_count=sum(
            item.decision == CandidateDecision.SUPPORTED for item in index.candidates
        ),
        supported_root_cause_codes=sorted(supported),
        supported_family_counts_by_root_cause_code=supported_family_counts,
        supported_family_depth_sufficient_for_split=supported_family_depth_sufficient,
        preserved_original_source_count=len(preserved),
        preserved_family_counts_by_root_cause_code=preserved_family_counts,
        preserved_family_depth_sufficient_for_split=preserved_family_depth_sufficient,
        missing_root_cause_codes=missing,
        admitted_research_record_count=len(admitted),
        taxonomy_coverage_sufficient_for_locked_holdout=(
            not missing and len(admitted) >= len(taxonomy_codes) * 3
        ),
        blocker=(
            "original_source_preservation_and_research_admission_required"
            if not missing and supported_family_depth_sufficient
            else "independent_taxonomy_coverage_insufficient"
        ),
        notes=[
            (
                "postmortems.app is used as a pinned discovery/index source, "
                "not as primary-source ground truth"
            ),
            (
                "supported candidate mappings are not research-admitted until "
                "original incident evidence is preserved"
            ),
            (
                "preserved-source counts describe checksum-validated original-source "
                "artifacts and do not by themselves create canonical research records"
            ),
            "keyword or title matches alone are insufficient for label admission",
        ],
    )
