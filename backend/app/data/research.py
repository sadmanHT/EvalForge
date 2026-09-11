from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import Field, model_validator

from app.data.postmortems import (
    CandidateDecision,
    PostmortemCandidate,
    PostmortemCandidateIndex,
)
from app.data.schemas import (
    DifficultyTier,
    IncidentFamily,
    IncidentRecord,
    SourceProvenance,
    Split,
    StrictModel,
)
from app.data.split import FamilySeed, family_stratified_split
from app.data.taxonomy import LabelTaxonomy


class ResearchAdmissionPolicy(StrictModel):
    independence_unit: str = Field(min_length=1)
    require_supported_mapping: bool
    require_preserved_primary_source: bool
    require_exactly_three_families_per_label: bool
    difficulty_policy: str = Field(min_length=1)
    synthetic_rows_in_locked_base_dataset: int = Field(ge=0)


class ResearchAdmissionPlan(StrictModel):
    admission_version: str = Field(min_length=1)
    candidate_index_version_before_admission: str = Field(min_length=1)
    candidate_index_version_after_admission: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    split_seed: int
    candidate_ids: list[str] = Field(min_length=1)
    policy: ResearchAdmissionPolicy

    @model_validator(mode="after")
    def validate_plan(self) -> ResearchAdmissionPlan:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("research admission candidate IDs must be unique")
        if self.policy.independence_unit != "incident_family":
            raise ValueError("research admission independence unit must be incident_family")
        if not self.policy.require_supported_mapping:
            raise ValueError("research admission must require supported mappings")
        if not self.policy.require_preserved_primary_source:
            raise ValueError("research admission must require preserved primary sources")
        if not self.policy.require_exactly_three_families_per_label:
            raise ValueError("Phase 03 lock requires exactly three families per label")
        if self.policy.difficulty_policy != "unknown_unless_source_supported":
            raise ValueError("research admission difficulty policy changed unexpectedly")
        if self.policy.synthetic_rows_in_locked_base_dataset != 0:
            raise ValueError("locked base research dataset must not contain synthetic rows")
        return self


def load_research_admission_plan(path: Path) -> ResearchAdmissionPlan:
    return ResearchAdmissionPlan.model_validate(json.loads(path.read_text(encoding="utf-8")))


def supported_preserved_candidates(index: PostmortemCandidateIndex) -> list[PostmortemCandidate]:
    return [
        candidate
        for candidate in index.candidates
        if candidate.decision == CandidateDecision.SUPPORTED
        and candidate.original_source_snapshot_preserved
    ]


def validate_research_admission_plan(
    index: PostmortemCandidateIndex,
    plan: ResearchAdmissionPlan,
    taxonomy: LabelTaxonomy,
    *,
    require_applied: bool,
) -> list[PostmortemCandidate]:
    by_id = {candidate.candidate_id: candidate for candidate in index.candidates}
    planned = set(plan.candidate_ids)
    eligible = {candidate.candidate_id for candidate in supported_preserved_candidates(index)}
    if planned != eligible:
        missing = sorted(eligible - planned)
        extra = sorted(planned - eligible)
        raise ValueError(
            "research admission plan must exactly match supported preserved candidates: "
            f"missing={missing} extra={extra}"
        )

    expected_count = len(taxonomy.label_to_category) * 3
    if len(planned) != expected_count:
        raise ValueError(
            f"research admission requires exactly {expected_count} independent families, "
            f"got {len(planned)}"
        )

    candidates = [by_id[candidate_id] for candidate_id in sorted(planned)]
    counts = Counter(candidate.proposed_root_cause_code for candidate in candidates)
    expected_labels = set(taxonomy.label_to_category)
    if set(counts) != expected_labels or set(counts.values()) != {3}:
        raise ValueError(
            "research admission requires exactly three independent families for every frozen label"
        )

    admitted = {
        candidate.candidate_id for candidate in index.candidates if candidate.research_admitted
    }
    if require_applied and admitted != planned:
        raise ValueError(
            "candidate index research admission flags do not match admission plan: "
            f"missing={sorted(planned - admitted)} extra={sorted(admitted - planned)}"
        )
    if not require_applied and admitted and admitted != planned:
        raise ValueError("candidate index contains partial or out-of-plan research admissions")
    return candidates


def stable_research_id(prefix: str, candidate_id: str) -> str:
    digest = hashlib.sha256(f"{prefix}:{candidate_id}".encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def build_research_records_and_families(
    index: PostmortemCandidateIndex,
    plan: ResearchAdmissionPlan,
    taxonomy: LabelTaxonomy,
) -> tuple[list[IncidentRecord], list[IncidentFamily]]:
    candidates = validate_research_admission_plan(
        index,
        plan,
        taxonomy,
        require_applied=True,
    )
    seeds = [
        FamilySeed(
            family_id=stable_research_id("family", candidate.candidate_id),
            stratify_label=candidate.proposed_root_cause_code,
        )
        for candidate in candidates
    ]
    assignments = family_stratified_split(seeds, seed=plan.split_seed)

    records: list[IncidentRecord] = []
    families: list[IncidentFamily] = []
    for candidate in candidates:
        family_id = stable_research_id("family", candidate.candidate_id)
        incident_id = stable_research_id("incident", candidate.candidate_id)
        split = assignments[family_id]
        assert candidate.preserved_primary_source_path is not None
        assert candidate.preserved_primary_source_sha256 is not None
        assert candidate.preserved_primary_source_kind is not None
        record = IncidentRecord(
            incident_id=incident_id,
            incident_family_id=family_id,
            title=candidate.title,
            description=candidate.evidence_summary,
            service=None,
            severity=None,
            started_at=None,
            root_cause_code=candidate.proposed_root_cause_code,
            root_cause_category=taxonomy.label_to_category[candidate.proposed_root_cause_code],
            evidence=[candidate.evidence_summary],
            difficulty_tier=DifficultyTier.UNKNOWN,
            source_provenance=SourceProvenance(
                source_name=candidate.company,
                source_kind="preserved_primary_source",
                source_record_id=candidate.candidate_id,
                source_checksum=candidate.preserved_primary_source_sha256,
                research_eligible=True,
                notes=(
                    f"original_url={candidate.original_url}; "
                    f"snapshot_path={candidate.preserved_primary_source_path}; "
                    f"snapshot_kind={candidate.preserved_primary_source_kind}"
                ),
            ),
            split=split,
            metadata={
                "candidate_id": candidate.candidate_id,
                "candidate_index_version": index.index_version,
                "mapping_basis": candidate.mapping_basis,
                "original_url": candidate.original_url,
                "preserved_primary_source_kind": candidate.preserved_primary_source_kind,
                "preserved_primary_source_path": candidate.preserved_primary_source_path,
                "preserved_primary_source_sha256": candidate.preserved_primary_source_sha256,
            },
        )
        family = IncidentFamily(
            family_id=family_id,
            split=split,
            source_family_key=candidate.candidate_id,
            member_incident_ids=[incident_id],
            root_cause_codes=[candidate.proposed_root_cause_code],
            metadata={
                "independence_basis": "one independently sourced public production incident",
                "source_candidate_id": candidate.candidate_id,
            },
        )
        records.append(record)
        families.append(family)

    split_label_counts = Counter((record.split.value, record.root_cause_code) for record in records)
    for split in Split:
        for label in taxonomy.label_to_category:
            if split_label_counts[(split.value, label)] != 1:
                raise ValueError(
                    "locked Phase 03 split requires exactly one family per label per split: "
                    f"split={split.value} label={label} "
                    f"count={split_label_counts[(split.value, label)]}"
                )
    if any(record.is_synthetic for record in records):
        raise ValueError("locked Phase 03 base research dataset must contain zero synthetic rows")
    return (
        sorted(records, key=lambda item: item.incident_id),
        sorted(families, key=lambda item: item.family_id),
    )
