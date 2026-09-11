from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DifficultyTier(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"
    COMPOUND = "compound"
    UNKNOWN = "unknown"


class SourceProvenance(StrictModel):
    source_name: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source_repository: str | None = None
    source_commit: str | None = None
    source_record_id: str = Field(min_length=1)
    source_checksum: str | None = None
    research_eligible: bool
    notes: str | None = None


class IncidentRecord(StrictModel):
    incident_id: str = Field(min_length=1, max_length=160)
    incident_family_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    description: str = Field(min_length=1)
    service: str | None = Field(default=None, max_length=120)
    severity: str | None = Field(default=None, max_length=40)
    started_at: datetime | None = None
    root_cause_code: str = Field(min_length=1, max_length=120)
    root_cause_category: str = Field(min_length=1, max_length=120)
    evidence: list[str] = Field(default_factory=list)
    difficulty_tier: DifficultyTier
    source_provenance: SourceProvenance
    split: Split
    is_synthetic: bool = False
    parent_incident_id: str | None = None
    generator_model_id: str | None = None
    generator_model_revision: str | None = None
    generator_prompt_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_synthetic_lineage(self) -> IncidentRecord:
        generator_fields = (
            self.generator_model_id,
            self.generator_model_revision,
            self.generator_prompt_version,
        )
        if self.is_synthetic:
            if not self.parent_incident_id:
                raise ValueError("synthetic incidents require parent_incident_id")
            if any(not value for value in generator_fields):
                raise ValueError(
                    "synthetic incidents require generator model, revision, and prompt version"
                )
        elif self.parent_incident_id or any(generator_fields):
            raise ValueError("non-synthetic incidents must not carry synthetic lineage fields")
        return self


class IncidentFamily(StrictModel):
    family_id: str = Field(min_length=1, max_length=160)
    split: Split
    source_family_key: str = Field(min_length=1)
    member_incident_ids: list[str] = Field(min_length=1)
    root_cause_codes: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_members(self) -> IncidentFamily:
        if len(self.member_incident_ids) != len(set(self.member_incident_ids)):
            raise ValueError("family member incident IDs must be unique")
        if len(self.root_cause_codes) != len(set(self.root_cause_codes)):
            raise ValueError("family root-cause codes must be unique")
        return self


class SourceSnapshot(StrictModel):
    source_repository: str
    source_commit: str
    snapshot_path: str
    snapshot_sha256: str
    snapshot_identity_kind: str | None = None
    accepted_transport_sha256s: list[str] = Field(default_factory=list)
    source_generated: bool
    research_eligible_as_independent_heldout_evidence: bool
    source_name: str | None = None
    source_uri: str | None = None
    source_license: str | None = None
    notes: str | None = None


class DatasetManifest(StrictModel):
    dataset_version: str
    schema_version: str
    label_taxonomy_version: str
    source_snapshots: list[SourceSnapshot]
    split_seed: int
    generated_at: datetime
    content_checksum: str
    manifest_checksum: str
    record_count: int = Field(ge=0)
    family_count: int = Field(ge=0)
    source_candidate_count: int = Field(default=0, ge=0)
    auxiliary_source_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int]
    research_ready: bool
    limitations: list[str] = Field(default_factory=list)
