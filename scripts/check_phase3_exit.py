#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.data.audit import FindingSeverity, audit_dataset
from app.data.hashing import dataset_content_checksum, manifest_checksum
from app.data.io import read_jsonl
from app.data.schemas import DatasetManifest, IncidentFamily, IncidentRecord, Split
from app.data.taxonomy import load_taxonomy


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/dataset.yaml").read_text(encoding="utf-8"))
    version = config["research_dataset_version"]
    dataset_dir = root / "datasets/incident_diagnosis/processed" / version
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print("PHASE03_EXIT_GATE=BLOCKED")
        print("BLOCKER=locked_research_dataset_not_built")
        return 2

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(manifest_payload)
    records = read_jsonl(dataset_dir / "incidents.jsonl", IncidentRecord)
    families = read_jsonl(dataset_dir / "families.jsonl", IncidentFamily)
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    audit = audit_dataset(records, families, taxonomy, research_mode=True)

    if not manifest.research_ready:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL research manifest not ready")
    if manifest.dataset_version != version:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL research dataset version mismatch")
    if manifest.schema_version != config["schema_version"]:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL schema version mismatch")
    if manifest.label_taxonomy_version != config["label_taxonomy_version"]:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL taxonomy version mismatch")
    if manifest.split_seed != config["split_seed"]:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL split seed mismatch")
    if manifest.record_count != len(records) or manifest.family_count != len(families):
        raise SystemExit("PHASE03_EXIT_GATE=FAIL manifest count mismatch")
    if len(records) != 18 or len(families) != 18:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL expected 18 independent research families")

    expected_split_counts = {"train": 6, "validation": 6, "test": 6}
    actual_split_counts = Counter(record.split.value for record in records)
    actual_family_split_counts = Counter(family.split.value for family in families)
    if dict(actual_split_counts) != expected_split_counts:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL research record split counts")
    if dict(actual_family_split_counts) != expected_split_counts:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL research family split counts")
    if manifest.split_counts != expected_split_counts:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL manifest split counts")

    for split in Split:
        labels = Counter(
            record.root_cause_code for record in records if record.split == split
        )
        expected = {label: 1 for label in taxonomy.label_to_category}
        if labels != expected:
            raise SystemExit(
                "PHASE03_EXIT_GATE=FAIL per-label family stratification "
                f"split={split.value}"
            )

    if any(record.is_synthetic for record in records):
        raise SystemExit("PHASE03_EXIT_GATE=FAIL locked base dataset contains synthetic rows")
    if not audit.passed:
        error_codes = sorted(
            finding.code
            for finding in audit.findings
            if finding.severity == FindingSeverity.ERROR
        )
        raise SystemExit(f"PHASE03_EXIT_GATE=FAIL leakage audit errors={error_codes}")

    expected_content_checksum = dataset_content_checksum(records, families)
    if manifest.content_checksum != expected_content_checksum:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL content checksum mismatch")
    if manifest.manifest_checksum != manifest_checksum(manifest_payload):
        raise SystemExit("PHASE03_EXIT_GATE=FAIL manifest checksum mismatch")
    if len(manifest.source_snapshots) != 18:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL source snapshot count")
    if not all(
        snapshot.research_eligible_as_independent_heldout_evidence
        and not snapshot.source_generated
        for snapshot in manifest.source_snapshots
    ):
        raise SystemExit("PHASE03_EXIT_GATE=FAIL ineligible research source snapshot")

    print("PHASE03_EXIT_GATE=PASS")
    print(f"RESEARCH_DATASET_VERSION={version}")
    print(f"RESEARCH_RECORDS={len(records)}")
    print(f"RESEARCH_FAMILIES={len(families)}")
    print("RESEARCH_TRAIN=6")
    print("RESEARCH_VALIDATION=6")
    print("RESEARCH_TEST=6")
    print("SYNTHETIC_BASE_RECORDS=0")
    print("LEAKAGE_ERRORS=0")
    print(f"CONTENT_CHECKSUM={manifest.content_checksum}")
    print(f"MANIFEST_CHECKSUM={manifest.manifest_checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
