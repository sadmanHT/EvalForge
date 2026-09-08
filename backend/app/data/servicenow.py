from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

from pydantic import Field

from app.data.hashing import sha256_hex
from app.data.schemas import StrictModel


EXPECTED_MEMBER = "incident_event_log.csv"
EXPECTED_COLUMNS = (
    "number",
    "incident_state",
    "active",
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
    "made_sla",
    "caller_id",
    "opened_by",
    "opened_at",
    "sys_created_by",
    "sys_created_at",
    "sys_updated_by",
    "sys_updated_at",
    "contact_type",
    "location",
    "category",
    "subcategory",
    "u_symptom",
    "cmdb_ci",
    "impact",
    "urgency",
    "priority",
    "assignment_group",
    "assigned_to",
    "knowledge",
    "u_priority_confirmation",
    "notify",
    "problem_id",
    "rfc",
    "vendor",
    "caused_by",
    "closed_code",
    "resolved_by",
    "resolved_at",
    "closed_at",
)
MISSING_MARKERS = {"", "?", "NA", "nan", "None"}
INCIDENT_PRESENCE_FIELDS = (
    "category",
    "subcategory",
    "u_symptom",
    "problem_id",
    "rfc",
    "caused_by",
    "closed_code",
    "cmdb_ci",
)


class ServiceNowInspection(StrictModel):
    event_row_count: int
    incident_count: int
    column_count: int
    archive_sha256: str
    member_sha256: str
    member_name: str
    incident_field_presence_counts: dict[str, int]
    incident_state_counts: dict[str, int]
    source_generated: bool
    real_world_operational_source: bool
    has_human_readable_root_cause_text: bool
    research_eligible_as_independent_heldout_evidence: bool
    blockers: list[str] = Field(default_factory=list)


def _present(value: str | None) -> bool:
    return (value or "").strip() not in MISSING_MARKERS


def inspect_servicenow_archive(path: Path) -> ServiceNowInspection:
    archive_bytes = path.read_bytes()
    archive_sha = sha256_hex(archive_bytes)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as bundle:
        names = bundle.namelist()
        if names != [EXPECTED_MEMBER]:
            raise ValueError(
                f"ServiceNow UCI archive must contain only {EXPECTED_MEMBER}; got {names}"
            )
        member_bytes = bundle.read(EXPECTED_MEMBER)

    text = io.StringIO(member_bytes.decode("utf-8-sig"), newline="")
    reader = csv.DictReader(text)
    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
        raise ValueError("ServiceNow UCI schema does not match the pinned 36-column source")

    incident_presence: dict[str, dict[str, bool]] = {}
    incident_states: Counter[str] = Counter()
    row_count = 0
    for row in reader:
        row_count += 1
        incident_id = row["number"]
        state = row["incident_state"]
        incident_states[state] += 1
        flags = incident_presence.setdefault(
            incident_id,
            {field: False for field in INCIDENT_PRESENCE_FIELDS},
        )
        for field in INCIDENT_PRESENCE_FIELDS:
            if _present(row[field]):
                flags[field] = True

    presence_counts = {
        field: sum(flags[field] for flags in incident_presence.values())
        for field in INCIDENT_PRESENCE_FIELDS
    }
    blockers = [
        "The source is a real ServiceNow operational event log, but textual attributes were "
        "omitted/anonymized and it does not expose trustworthy human-readable root-cause labels "
        "for EvalForge taxonomy mapping.",
        "Anonymous category/symptom/closure codes must not be guessed into EvalForge root-cause "
        "codes; this source is auxiliary/robustness evidence, not the primary labeled holdout.",
    ]
    return ServiceNowInspection(
        event_row_count=row_count,
        incident_count=len(incident_presence),
        column_count=len(EXPECTED_COLUMNS),
        archive_sha256=archive_sha,
        member_sha256=sha256_hex(member_bytes),
        member_name=EXPECTED_MEMBER,
        incident_field_presence_counts=presence_counts,
        incident_state_counts=dict(sorted(incident_states.items())),
        source_generated=False,
        real_world_operational_source=True,
        has_human_readable_root_cause_text=False,
        research_eligible_as_independent_heldout_evidence=False,
        blockers=blockers,
    )
