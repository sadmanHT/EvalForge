from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.data.io import read_jsonl
from app.data.schemas import DatasetManifest, IncidentRecord
from app.data.schemas import IncidentFamily as DataFamily
from app.models import DatasetVersion, Incident, IncidentFamily


def import_phase3_dataset(session: Session, root: Path, dataset_version: str) -> DatasetVersion:
    dataset_dir = root / "datasets/incident_diagnosis/processed" / dataset_version
    manifest_payload = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(manifest_payload)
    records = read_jsonl(dataset_dir / "incidents.jsonl", IncidentRecord)
    families = read_jsonl(dataset_dir / "families.jsonl", DataFamily)
    if manifest.dataset_version != dataset_version:
        raise ValueError("dataset manifest version does not match import target")
    existing = session.get(DatasetVersion, dataset_version)
    if existing is not None:
        if (
            existing.manifest_checksum != manifest.manifest_checksum
            or existing.content_checksum != manifest.content_checksum
            or existing.record_count != len(records)
            or existing.family_count != len(families)
        ):
            raise ValueError("existing dataset version does not match immutable manifest")
        return existing
    row = DatasetVersion(
        version=dataset_version,
        schema_version=manifest.schema_version,
        label_taxonomy_version=manifest.label_taxonomy_version,
        manifest_checksum=manifest.manifest_checksum,
        content_checksum=manifest.content_checksum,
        split_seed=manifest.split_seed,
        record_count=len(records),
        family_count=len(families),
        research_ready=manifest.research_ready,
        manifest_json=manifest_payload,
    )
    session.add(row)
    for family in families:
        session.add(
            IncidentFamily(
                dataset_version=dataset_version,
                family_id=family.family_id,
                split=family.split.value,
                source_family_key=family.source_family_key,
                root_cause_codes=family.root_cause_codes,
                family_metadata=family.metadata,
            )
        )
    for record in records:
        session.add(
            Incident(
                dataset_version=dataset_version,
                incident_id=record.incident_id,
                family_id=record.incident_family_id,
                split=record.split.value,
                title=record.title,
                description=record.description,
                root_cause_code=record.root_cause_code,
                root_cause_category=record.root_cause_category,
                source_checksum=record.source_provenance.source_checksum,
                source_record_id=record.source_provenance.source_record_id,
                is_synthetic=record.is_synthetic,
                incident_metadata=record.metadata,
            )
        )
    session.flush()
    family_count = session.scalar(
        select(func.count())
        .select_from(IncidentFamily)
        .where(IncidentFamily.dataset_version == dataset_version)
    )
    record_count = session.scalar(
        select(func.count())
        .select_from(Incident)
        .where(Incident.dataset_version == dataset_version)
    )
    if family_count != manifest.family_count or record_count != manifest.record_count:
        raise ValueError("database round-trip counts disagree with dataset manifest")
    return row
