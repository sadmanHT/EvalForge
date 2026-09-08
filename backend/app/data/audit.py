from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.data.schemas import IncidentFamily, IncidentRecord, Split, StrictModel
from app.data.taxonomy import LabelTaxonomy


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class AuditFinding(StrictModel):
    code: str
    severity: FindingSeverity
    message: str
    incident_ids: list[str] = Field(default_factory=list)
    family_ids: list[str] = Field(default_factory=list)


class DatasetAuditReport(StrictModel):
    passed: bool
    record_count: int
    family_count: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    findings: list[AuditFinding]


def _normalize_text(record: IncidentRecord) -> str:
    text = f"{record.title} {record.description}".casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _training_search_text(record: IncidentRecord) -> str:
    return " ".join(
        [
            record.title,
            record.description,
            *record.evidence,
            json.dumps(record.metadata, sort_keys=True),
            json.dumps(record.source_provenance.model_dump(mode="json"), sort_keys=True),
        ]
    ).casefold()


def audit_dataset(
    records: list[IncidentRecord],
    families: list[IncidentFamily],
    taxonomy: LabelTaxonomy,
    *,
    research_mode: bool,
    near_duplicate_threshold: float = 0.92,
) -> DatasetAuditReport:
    findings: list[AuditFinding] = []
    by_id: dict[str, IncidentRecord] = {}
    duplicate_ids: list[str] = []
    for record in records:
        if record.incident_id in by_id:
            duplicate_ids.append(record.incident_id)
        by_id[record.incident_id] = record
    if duplicate_ids:
        findings.append(
            AuditFinding(
                code="DUPLICATE_INCIDENT_ID",
                severity=FindingSeverity.ERROR,
                message="incident IDs must be unique",
                incident_ids=sorted(set(duplicate_ids)),
            )
        )

    family_splits: dict[str, set[Split]] = defaultdict(set)
    family_members: dict[str, set[str]] = defaultdict(set)
    for record in records:
        family_splits[record.incident_family_id].add(record.split)
        family_members[record.incident_family_id].add(record.incident_id)
        expected_category = taxonomy.label_to_category.get(record.root_cause_code)
        if expected_category is None:
            findings.append(
                AuditFinding(
                    code="INVALID_ROOT_CAUSE_LABEL",
                    severity=FindingSeverity.ERROR,
                    message=f"unknown label {record.root_cause_code}",
                    incident_ids=[record.incident_id],
                )
            )
        elif expected_category != record.root_cause_category:
            findings.append(
                AuditFinding(
                    code="ROOT_CAUSE_CATEGORY_MISMATCH",
                    severity=FindingSeverity.ERROR,
                    message=(
                        f"{record.root_cause_code} belongs to {expected_category}, "
                        f"not {record.root_cause_category}"
                    ),
                    incident_ids=[record.incident_id],
                )
            )

        if record.is_synthetic:
            parent = by_id.get(record.parent_incident_id or "")
            if record.split != Split.TRAIN:
                findings.append(
                    AuditFinding(
                        code="SYNTHETIC_OUTSIDE_TRAIN",
                        severity=FindingSeverity.ERROR,
                        message="synthetic descendants are allowed only in train",
                        incident_ids=[record.incident_id],
                    )
                )
            if parent is None:
                findings.append(
                    AuditFinding(
                        code="SYNTHETIC_PARENT_MISSING",
                        severity=FindingSeverity.ERROR,
                        message="synthetic descendant parent is not present",
                        incident_ids=[record.incident_id],
                    )
                )
            else:
                if parent.split != Split.TRAIN:
                    findings.append(
                        AuditFinding(
                            code="SYNTHETIC_PARENT_NOT_TRAIN",
                            severity=FindingSeverity.ERROR,
                            message="synthetic parent must be in train",
                            incident_ids=[record.incident_id, parent.incident_id],
                        )
                    )
                if parent.incident_family_id != record.incident_family_id:
                    findings.append(
                        AuditFinding(
                            code="SYNTHETIC_FAMILY_MISMATCH",
                            severity=FindingSeverity.ERROR,
                            message="synthetic child must inherit the parent family",
                            incident_ids=[record.incident_id, parent.incident_id],
                        )
                    )

        if (
            research_mode
            and record.split in {Split.VALIDATION, Split.TEST}
            and not record.source_provenance.research_eligible
        ):
            findings.append(
                AuditFinding(
                    code="INELIGIBLE_HELDOUT_SOURCE",
                    severity=FindingSeverity.ERROR,
                    message="held-out research evidence must be independently source-eligible",
                    incident_ids=[record.incident_id],
                )
            )

    overlapping = {
        family_id: splits for family_id, splits in family_splits.items() if len(splits) > 1
    }
    for family_id, splits in sorted(overlapping.items()):
        findings.append(
            AuditFinding(
                code="CROSS_SPLIT_FAMILY",
                severity=FindingSeverity.ERROR,
                message=(
                    "family occurs in multiple splits: "
                    f"{sorted(item.value for item in splits)}"
                ),
                incident_ids=sorted(family_members[family_id]),
                family_ids=[family_id],
            )
        )

    declared_families = {family.family_id: family for family in families}
    for family_id, member_ids in sorted(family_members.items()):
        declared = declared_families.get(family_id)
        if declared is None:
            findings.append(
                AuditFinding(
                    code="FAMILY_METADATA_MISSING",
                    severity=FindingSeverity.ERROR,
                    message="record family has no IncidentFamily metadata",
                    incident_ids=sorted(member_ids),
                    family_ids=[family_id],
                )
            )
            continue
        actual_splits = family_splits[family_id]
        if len(actual_splits) == 1 and declared.split not in actual_splits:
            findings.append(
                AuditFinding(
                    code="FAMILY_SPLIT_METADATA_MISMATCH",
                    severity=FindingSeverity.ERROR,
                    message="IncidentFamily split disagrees with member records",
                    incident_ids=sorted(member_ids),
                    family_ids=[family_id],
                )
            )
        if set(declared.member_incident_ids) != member_ids:
            findings.append(
                AuditFinding(
                    code="FAMILY_MEMBERSHIP_MISMATCH",
                    severity=FindingSeverity.ERROR,
                    message="IncidentFamily member list disagrees with records",
                    incident_ids=sorted(member_ids | set(declared.member_incident_ids)),
                    family_ids=[family_id],
                )
            )

    normalized: dict[str, list[IncidentRecord]] = defaultdict(list)
    for record in records:
        normalized[_normalize_text(record)].append(record)
    for same_text in normalized.values():
        splits = {record.split for record in same_text}
        if len(splits) > 1:
            findings.append(
                AuditFinding(
                    code="CROSS_SPLIT_EXACT_DUPLICATE",
                    severity=FindingSeverity.ERROR,
                    message="exact normalized incident text occurs across splits",
                    incident_ids=sorted(record.incident_id for record in same_text),
                )
            )

    distinct = list(records)
    for left_index, left in enumerate(distinct):
        left_text = _normalize_text(left)
        for right in distinct[left_index + 1 :]:
            if left.split == right.split:
                continue
            right_text = _normalize_text(right)
            if left_text == right_text:
                continue
            ratio = SequenceMatcher(None, left_text, right_text).ratio()
            if ratio >= near_duplicate_threshold:
                findings.append(
                    AuditFinding(
                        code="CROSS_SPLIT_NEAR_DUPLICATE",
                        severity=FindingSeverity.ERROR,
                        message=f"near-duplicate ratio {ratio:.3f} exceeds threshold",
                        incident_ids=sorted([left.incident_id, right.incident_id]),
                    )
                )

    heldout_tokens: set[str] = set()
    for record in records:
        if record.split in {Split.VALIDATION, Split.TEST}:
            for token in (record.incident_id, record.incident_family_id):
                if len(token) >= 6:
                    heldout_tokens.add(token.casefold())
    for record in records:
        if record.split != Split.TRAIN:
            continue
        haystack = _training_search_text(record)
        leaked = sorted(token for token in heldout_tokens if token and token in haystack)
        if leaked:
            findings.append(
                AuditFinding(
                    code="FORBIDDEN_HELDOUT_REFERENCE",
                    severity=FindingSeverity.ERROR,
                    message=f"training artifact references held-out identifiers: {leaked}",
                    incident_ids=[record.incident_id],
                )
            )

    class_counts = Counter(record.root_cause_code for record in records)
    difficulty_counts = Counter(record.difficulty_tier.value for record in records)
    split_counts = Counter(record.split.value for record in records)
    if class_counts:
        values = list(class_counts.values())
        if min(values) > 0 and max(values) / min(values) >= 4:
            findings.append(
                AuditFinding(
                    code="SEVERE_CLASS_IMBALANCE",
                    severity=FindingSeverity.WARNING,
                    message=f"class count range is {min(values)}..{max(values)}",
                )
            )
    expected_difficulties = {"easy", "medium", "hard"}
    missing = sorted(expected_difficulties - set(difficulty_counts))
    if missing:
        findings.append(
            AuditFinding(
                code="MISSING_DIFFICULTY_TIERS",
                severity=FindingSeverity.WARNING,
                message=f"dataset does not include baseline difficulty tiers: {missing}",
            )
        )

    passed = not any(item.severity == FindingSeverity.ERROR for item in findings)
    return DatasetAuditReport(
        passed=passed,
        record_count=len(records),
        family_count=len(family_members),
        split_counts=dict(sorted(split_counts.items())),
        class_counts=dict(sorted(class_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        findings=findings,
    )
