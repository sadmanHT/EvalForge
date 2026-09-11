#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.data.audit import audit_dataset
from app.data.io import write_jsonl
from app.data.manifest import build_manifest
from app.data.postmortems import inspect_candidate_coverage, load_candidate_index
from app.data.primary_sources import (
    load_primary_source_manifest,
    load_primary_source_plan,
    validate_primary_source_preservation,
)
from app.data.research import (
    build_research_records_and_families,
    load_research_admission_plan,
)
from app.data.schemas import SourceSnapshot, Split
from app.data.taxonomy import load_taxonomy


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_research_outputs(root: Path) -> dict[str, str]:
    config_path = root / "configs/dataset.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    admission_plan_path = root / "configs/phase3-research-admissions.json"
    index_path = root / "configs/postmortem-candidates.json"
    taxonomy_path = root / "configs/label-taxonomy.yaml"
    primary_plan_path = root / "configs/phase3-primary-sources.json"
    primary_manifest_path = (
        root
        / "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/manifest.json"
    )

    admission_plan = load_research_admission_plan(admission_plan_path)
    index = load_candidate_index(index_path)
    taxonomy = load_taxonomy(taxonomy_path)
    primary_plan = load_primary_source_plan(primary_plan_path)
    primary_manifest = load_primary_source_manifest(primary_manifest_path)
    preservation = validate_primary_source_preservation(
        root,
        index,
        primary_plan,
        primary_manifest,
    )
    coverage = inspect_candidate_coverage(index, taxonomy)

    if config["research_dataset_version"] != admission_plan.dataset_version:
        raise ValueError("research dataset version disagrees with admission plan")
    if config["split_seed"] != admission_plan.split_seed:
        raise ValueError("research split seed disagrees with admission plan")
    if preservation.preserved_candidate_count != len(admission_plan.candidate_ids):
        raise ValueError("research admission does not cover every preserved supported family")
    if not preservation.preserved_supported_family_depth_sufficient_for_split:
        raise ValueError("preserved primary-source family depth is insufficient")
    if coverage.admitted_research_record_count != len(admission_plan.candidate_ids):
        raise ValueError("candidate coverage does not reflect the explicit admission plan")
    if not coverage.taxonomy_coverage_sufficient_for_locked_holdout:
        raise ValueError("candidate coverage is not sufficient for a locked holdout")

    records, families = build_research_records_and_families(
        index,
        admission_plan,
        taxonomy,
    )
    audit = audit_dataset(records, families, taxonomy, research_mode=True)
    if not audit.passed:
        raise ValueError(
            "research dataset leakage audit failed: "
            + json.dumps(audit.model_dump(mode="json"), sort_keys=True)
        )

    snapshots_by_candidate = {
        snapshot.candidate_id: snapshot for snapshot in primary_manifest.snapshots
    }
    candidate_by_id = {candidate.candidate_id: candidate for candidate in index.candidates}
    source_snapshots: list[SourceSnapshot] = []
    for candidate_id in sorted(admission_plan.candidate_ids):
        candidate = candidate_by_id[candidate_id]
        snapshot = snapshots_by_candidate.get(candidate_id)
        if snapshot is None:
            raise ValueError(
                f"research candidate lacks primary-source manifest entry: {candidate_id}"
            )
        if snapshot.snapshot_sha256 != candidate.preserved_primary_source_sha256:
            raise ValueError(f"research source checksum mismatch: {candidate_id}")
        source_snapshots.append(
            SourceSnapshot(
                source_repository="public_primary_source",
                source_commit=f"sha256:{snapshot.snapshot_sha256}",
                snapshot_path=snapshot.snapshot_path,
                snapshot_sha256=snapshot.snapshot_sha256,
                snapshot_identity_kind=f"{snapshot.snapshot_kind.value}:sha256",
                source_generated=False,
                research_eligible_as_independent_heldout_evidence=True,
                source_name=candidate.company,
                source_uri=candidate.original_url,
                notes=(
                    "Independent public production-incident source admitted by explicit "
                    f"Phase 03 plan {admission_plan.admission_version}."
                ),
            )
        )

    planned_ids = set(admission_plan.candidate_ids)
    captured_times = [
        datetime.fromisoformat(snapshot.captured_at.replace("Z", "+00:00"))
        for snapshot in primary_manifest.snapshots
        if snapshot.candidate_id in planned_ids
    ]
    if len(captured_times) != len(admission_plan.candidate_ids):
        raise ValueError("primary-source capture timestamps are incomplete")
    generated_at = max(captured_times)

    version = config["research_dataset_version"]
    output = root / "datasets/incident_diagnosis/processed" / version
    output.mkdir(parents=True, exist_ok=True)
    incidents_path = output / "incidents.jsonl"
    families_path = output / "families.jsonl"
    write_jsonl(incidents_path, records, sort_key="incident_id")
    write_jsonl(families_path, families, sort_key="family_id")

    split_class_counts = {
        split.value: {
            label: sum(
                record.split == split and record.root_cause_code == label for record in records
            )
            for label in sorted(taxonomy.label_to_category)
        }
        for split in Split
    }
    split_family_counts = Counter(family.split.value for family in families)
    split_record_counts = Counter(record.split.value for record in records)
    class_counts = Counter(record.root_cause_code for record in records)
    difficulty_counts = Counter(record.difficulty_tier.value for record in records)
    summary = {
        "dataset_version": version,
        "admission_version": admission_plan.admission_version,
        "split_seed": admission_plan.split_seed,
        "record_count": len(records),
        "independent_family_count": len(families),
        "synthetic_record_count": sum(record.is_synthetic for record in records),
        "split_record_counts": {
            split.value: split_record_counts.get(split.value, 0) for split in Split
        },
        "split_family_counts": {
            split.value: split_family_counts.get(split.value, 0) for split in Split
        },
        "class_counts": dict(sorted(class_counts.items())),
        "split_class_counts": split_class_counts,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "family_disjointness": "PASS",
        "research_audit_passed": True,
        "audit_warning_codes": sorted(
            finding.code for finding in audit.findings if finding.severity.value == "warning"
        ),
    }
    _write_json(output / "class-split-summary.json", summary)
    _write_json(output / "leakage-audit-report.json", audit.model_dump(mode="json"))

    record_by_candidate = {record.source_provenance.source_record_id: record for record in records}
    admission_summary = {
        "admission_version": admission_plan.admission_version,
        "candidate_index_version": index.index_version,
        "primary_source_manifest_version": primary_manifest.manifest_version,
        "candidate_count": len(admission_plan.candidate_ids),
        "candidates": [
            {
                "candidate_id": candidate_id,
                "root_cause_code": candidate_by_id[candidate_id].proposed_root_cause_code,
                "incident_id": record_by_candidate[candidate_id].incident_id,
                "family_id": record_by_candidate[candidate_id].incident_family_id,
                "split": record_by_candidate[candidate_id].split.value,
                "original_url": candidate_by_id[candidate_id].original_url,
                "primary_source_path": candidate_by_id[candidate_id].preserved_primary_source_path,
                "primary_source_sha256": candidate_by_id[
                    candidate_id
                ].preserved_primary_source_sha256,
                "primary_source_kind": candidate_by_id[candidate_id].preserved_primary_source_kind,
            }
            for candidate_id in sorted(admission_plan.candidate_ids)
        ],
    }
    _write_json(output / "research-admission-summary.json", admission_summary)

    limitations = [
        (
            "The locked corpus contains exactly three independent families per frozen label, "
            "which yields one independent family per label in each train/validation/test split; "
            "this is leakage-safe but statistically small."
        ),
        (
            "Difficulty tiers remain unknown because the reviewed primary sources do not provide "
            "a defensible common difficulty rubric; the audit reports this rather than "
            "inventing tiers."
        ),
        (
            "The immutable base research dataset contains zero synthetic rows. Any later synthetic "
            "augmentation must occur only after this family split and only from training families."
        ),
        (
            "OpsSentinel generated scenarios and the UCI/ServiceNow auxiliary corpus are excluded "
            "from independent research train/validation/test evidence."
        ),
    ]
    manifest = build_manifest(
        dataset_version=version,
        schema_version=config["schema_version"],
        label_taxonomy_version=config["label_taxonomy_version"],
        source_snapshots=source_snapshots,
        split_seed=admission_plan.split_seed,
        generated_at=generated_at,
        records=records,
        families=families,
        source_candidate_count=len(index.candidates),
        auxiliary_source_counts={
            "public_postmortem_supported_mappings": coverage.supported_mapping_count,
            "public_postmortem_preserved_primary_sources": coverage.preserved_original_source_count,
            "public_postmortem_research_admitted": coverage.admitted_research_record_count,
        },
        research_ready=True,
        limitations=limitations,
    )
    _write_json(output / "manifest.json", manifest.model_dump(mode="json"))

    return {
        "incidents_sha256": _sha256(incidents_path),
        "families_sha256": _sha256(families_path),
        "manifest_sha256": _sha256(output / "manifest.json"),
        "leakage_audit_sha256": _sha256(output / "leakage-audit-report.json"),
        "class_split_summary_sha256": _sha256(output / "class-split-summary.json"),
        "research_admission_summary_sha256": _sha256(output / "research-admission-summary.json"),
        "manifest_identity_checksum": manifest.manifest_checksum,
        "content_checksum": manifest.content_checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = build_research_outputs(args.root.resolve())
    print("PHASE03_RESEARCH_DATASET_BUILD=PASS")
    for key, value in sorted(result.items()):
        print(f"{key.upper()}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
