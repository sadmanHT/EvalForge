from __future__ import annotations

from datetime import datetime
from typing import Any

from app.data.hashing import dataset_content_checksum, manifest_checksum
from app.data.schemas import DatasetManifest, IncidentFamily, IncidentRecord, SourceSnapshot


def build_manifest(
    *,
    dataset_version: str,
    schema_version: str,
    label_taxonomy_version: str,
    source_snapshots: list[SourceSnapshot],
    split_seed: int,
    generated_at: datetime,
    records: list[IncidentRecord],
    families: list[IncidentFamily],
    source_candidate_count: int = 0,
    auxiliary_source_counts: dict[str, int] | None = None,
    research_ready: bool,
    limitations: list[str],
) -> DatasetManifest:
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        split_counts[record.split.value] += 1
    draft = DatasetManifest(
        dataset_version=dataset_version,
        schema_version=schema_version,
        label_taxonomy_version=label_taxonomy_version,
        source_snapshots=source_snapshots,
        split_seed=split_seed,
        generated_at=generated_at,
        content_checksum=dataset_content_checksum(records, families),
        manifest_checksum="pending",
        record_count=len(records),
        family_count=len(families),
        source_candidate_count=source_candidate_count,
        auxiliary_source_counts=auxiliary_source_counts or {},
        split_counts=split_counts,
        research_ready=research_ready,
        limitations=limitations,
    )
    payload: dict[str, Any] = draft.model_dump(mode="json")
    payload["manifest_checksum"] = manifest_checksum(payload)
    return DatasetManifest.model_validate(payload)
