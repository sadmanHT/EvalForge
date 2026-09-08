#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from app.data.audit import audit_dataset
from app.data.io import read_jsonl
from app.data.postmortems import inspect_candidate_coverage, load_candidate_index
from app.data.servicenow import inspect_servicenow_archive
from app.data.schemas import IncidentFamily, IncidentRecord
from app.data.taxonomy import load_taxonomy


REQUIRED = (
    "configs/dataset.yaml",
    "configs/postmortem-candidates.json",
    "datasets/incident_diagnosis/raw/README.md",
    (
        "datasets/incident_diagnosis/raw/opssentinel/"
        "fae661fc1634aad6a3855a1dec8dcddb16a890dd/"
        "release-catalog.snapshot.json.gz.b64"
    ),
    "datasets/incident_diagnosis/processed/README.md",
    "backend/app/data/schemas.py",
    "backend/app/data/split.py",
    "backend/app/data/audit.py",
    "backend/app/data/opssentinel.py",
    "backend/app/data/postmortems.py",
    "backend/app/data/servicenow.py",
    "training/dataset_prep.py",
    "scripts/audit_dataset.py",
    "scripts/check_phase3_reproducibility.py",
    "scripts/fetch_phase3_sources.py",
    "scripts/check_phase3_exit.py",
    "docs/data-card.md",
    "docs/phase-03-handoff.md",
    "datasets/incident_diagnosis/fixtures/ci_smoke/incidents.jsonl",
    "datasets/incident_diagnosis/fixtures/ci_smoke/families.jsonl",
    "datasets/incident_diagnosis/raw/servicenow_uci/uci-498/SOURCE.json",
    "datasets/incident_diagnosis/raw/servicenow_uci/uci-498/SHA256SUMS.txt",
    (
        "datasets/incident_diagnosis/raw/servicenow_uci/uci-498/"
        "incident-management-process-enriched-event-log.zip"
    ),
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED if not (root / path).exists()]
    if missing:
        raise SystemExit(f"PHASE03_CONTRACT=FAIL missing={missing}")

    config = json.loads((root / "configs/dataset.yaml").read_text(encoding="utf-8"))
    snapshot_path = root / config["source_snapshot"]["path"]
    source_metadata_path = snapshot_path.parent / "SOURCE.json"
    sums_path = snapshot_path.parent / "SHA256SUMS.txt"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    transport_pack_path = root / config["source_snapshot"]["transport_pack_path"]
    transport_pack_sha = hashlib.sha256(transport_pack_path.read_bytes()).hexdigest()
    if transport_pack_sha != config["source_snapshot"]["transport_pack_sha256"]:
        raise SystemExit("PHASE03_CONTRACT=FAIL source_snapshot_transport_pack")
    source_metadata_sha = hashlib.sha256(source_metadata_path.read_bytes()).hexdigest()
    if source_metadata.get("snapshot_sha256") != snapshot_sha:
        raise SystemExit("PHASE03_CONTRACT=FAIL source_snapshot_metadata_checksum")
    expected_sums = {
        "release-catalog.snapshot.json": snapshot_sha,
        "SOURCE.json": source_metadata_sha,
    }
    parsed_sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        parsed_sums[name.strip()] = digest
    if parsed_sums != expected_sums:
        raise SystemExit("PHASE03_CONTRACT=FAIL source_snapshot_sha256sums")

    auxiliary_cfg = config["auxiliary_source_snapshots"][0]
    auxiliary_path = root / auxiliary_cfg["path"]
    auxiliary_metadata_path = auxiliary_path.parent / "SOURCE.json"
    auxiliary_sums_path = auxiliary_path.parent / "SHA256SUMS.txt"
    auxiliary_metadata = json.loads(auxiliary_metadata_path.read_text(encoding="utf-8"))
    auxiliary_metadata_sha = hashlib.sha256(auxiliary_metadata_path.read_bytes()).hexdigest()
    auxiliary_inspection = inspect_servicenow_archive(auxiliary_path)
    canonical_member_sha = auxiliary_cfg["canonical_member_sha256"]
    accepted_archive_shas = set(auxiliary_cfg["accepted_archive_sha256s"])
    metadata_archive_shas = {
        item["sha256"] for item in auxiliary_metadata["accepted_transport_archives"]
    }
    if auxiliary_inspection.member_sha256 != canonical_member_sha:
        raise SystemExit("PHASE03_CONTRACT=FAIL auxiliary_member_checksum")
    if auxiliary_inspection.archive_sha256 not in accepted_archive_shas:
        raise SystemExit("PHASE03_CONTRACT=FAIL auxiliary_transport_checksum")
    if metadata_archive_shas != accepted_archive_shas:
        raise SystemExit("PHASE03_CONTRACT=FAIL auxiliary_transport_metadata")
    if auxiliary_metadata.get("canonical_member_sha256") != canonical_member_sha:
        raise SystemExit("PHASE03_CONTRACT=FAIL auxiliary_member_metadata")
    expected_auxiliary_sums = {
        "incident_event_log.csv": canonical_member_sha,
        "SOURCE.json": auxiliary_metadata_sha,
    }
    parsed_auxiliary_sums = {}
    for line in auxiliary_sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        parsed_auxiliary_sums[name.strip()] = digest
    if parsed_auxiliary_sums != expected_auxiliary_sums:
        raise SystemExit("PHASE03_CONTRACT=FAIL auxiliary_snapshot_sha256sums")

    postmortem_cfg = config["public_postmortem_candidate_source"]
    postmortem_index = load_candidate_index(root / postmortem_cfg["candidate_index_path"])
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    coverage = inspect_candidate_coverage(postmortem_index, taxonomy)
    if coverage.source_repository != postmortem_cfg["repository"]:
        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_repository")
    if coverage.source_commit != postmortem_cfg["commit"]:
        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_commit")
    if coverage.admitted_research_record_count != 0:
        raise SystemExit("PHASE03_CONTRACT=FAIL premature_postmortem_research_admission")
    if set(coverage.supported_root_cause_codes) != {"disk_exhaustion", "memory_leak"}:
        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_supported_mapping_set")
    if "n_plus_one_query" not in coverage.missing_root_cause_codes:
        raise SystemExit("PHASE03_CONTRACT=FAIL n_plus_one_false_admission")

    fixture = root / "datasets/incident_diagnosis/fixtures/ci_smoke"
    records = read_jsonl(fixture / "incidents.jsonl", IncidentRecord)
    families = read_jsonl(fixture / "families.jsonl", IncidentFamily)
    report = audit_dataset(records, families, taxonomy, research_mode=False)
    if not report.passed:
        raise SystemExit(
            "PHASE03_CONTRACT=FAIL fixture_audit="
            + json.dumps(report.model_dump(mode="json"), sort_keys=True)
        )

    result = subprocess.run(
        [sys.executable, str(root / "scripts/check_phase3_reproducibility.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("PHASE03_CONTRACT=FAIL reproducibility")
    print(result.stdout.strip())
    print("PHASE03_CONTRACT=PASS")
    print(f"CI_SMOKE_RECORDS={len(records)}")
    print(f"CI_SMOKE_FAMILIES={len(families)}")
    print(f"PUBLIC_POSTMORTEM_CANDIDATES={coverage.candidate_count}")
    print(f"PUBLIC_POSTMORTEM_SUPPORTED_MAPPINGS={coverage.supported_mapping_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
