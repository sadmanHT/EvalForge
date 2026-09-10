#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from app.data.audit import FindingSeverity, audit_dataset
from app.data.io import read_jsonl
from app.data.postmortems import inspect_candidate_coverage, load_candidate_index
from app.data.primary_sources import (
    load_primary_source_manifest,
    load_primary_source_plan,
    validate_primary_source_preservation,
)
from app.data.research import (
    load_research_admission_plan,
    validate_research_admission_plan,
)
from app.data.schemas import DatasetManifest, IncidentFamily, IncidentRecord, Split
from app.data.servicenow import inspect_servicenow_archive
from app.data.taxonomy import load_taxonomy

RESEARCH_VERSION = "evalforge-incident-diagnosis-v0.1.0"
REQUIRED = (
    "configs/dataset.yaml",
    "configs/postmortem-candidates.json",
    "configs/phase3-primary-sources.json",
    "configs/phase3-research-admissions.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/README.md",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/SHA256SUMS.txt",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/dispatcharr-1416.snapshot.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/google-payment-E18Caoo5X1m6dTa1PVr1.snapshot.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/google-no-fault-fLYHLzSGXGkLkAjc8MJG.snapshot.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/elevenlabs-2026-04-22-payment-configuration.snapshot.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/manifest.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/medoc-n-plus-one.primary.md",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/openlibrary-12432.primary.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/dify-40036.primary.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/dispatcharr-1416.primary.json",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/aws-2012-memory-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/honeycomb-2019-memory-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/dnsimple-2015-memory-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/tarsnap-2016-disk-full.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/soundcloud-2011-binlog-disk-full.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/git-nrw-2025-wal-disk-full.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/coderden-2026-connection-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/altapay-2026-firewall-payment-config.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/atlassian-CRUC-8168-connection-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/fastly-2023-cloud-waf-false-alarm.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/gitlab-2025-version-skew-false-alarm.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/google-2021-payment-config.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/google-2024-mandiant-no-fault.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/elevenlabs-2026-billing-misconfiguration.primary.html",
    "datasets/incident_diagnosis/raw/README.md",
    (
        "datasets/incident_diagnosis/raw/opssentinel/"
        "fae661fc1634aad6a3855a1dec8dcddb16a890dd/"
        "release-catalog.snapshot.json.gz.b64"
    ),
    "datasets/incident_diagnosis/processed/README.md",
    f"datasets/incident_diagnosis/processed/{RESEARCH_VERSION}/incidents.jsonl",
    f"datasets/incident_diagnosis/processed/{RESEARCH_VERSION}/families.jsonl",
    f"datasets/incident_diagnosis/processed/{RESEARCH_VERSION}/manifest.json",
    f"datasets/incident_diagnosis/processed/{RESEARCH_VERSION}/leakage-audit-report.json",
    f"datasets/incident_diagnosis/processed/{RESEARCH_VERSION}/class-split-summary.json",
    f"datasets/incident_diagnosis/processed/{RESEARCH_VERSION}/research-admission-summary.json",
    "backend/app/data/schemas.py",
    "backend/app/data/split.py",
    "backend/app/data/audit.py",
    "backend/app/data/opssentinel.py",
    "backend/app/data/postmortems.py",
    "backend/app/data/primary_sources.py",
    "backend/app/data/research.py",
    "backend/app/data/servicenow.py",
    "training/dataset_prep.py",
    "scripts/audit_dataset.py",
    "scripts/admit_phase3_research_records.py",
    "scripts/build_phase3_research_dataset.py",
    "scripts/check_phase3_reproducibility.py",
    "scripts/check_phase3_research_reproducibility.py",
    "scripts/fetch_phase3_sources.py",
    "scripts/preserve_phase3_primary_sources.py",
    "scripts/sync_phase3_evidence.py",
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


def _run_check(root: Path, script: str, failure: str) -> None:
    result = subprocess.run(
        [sys.executable, str(root / script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"PHASE03_CONTRACT=FAIL {failure}")
    print(result.stdout.strip())


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED if not (root / path).exists()]
    if missing:
        raise SystemExit(f"PHASE03_CONTRACT=FAIL missing={missing}")

    config = json.loads((root / "configs/dataset.yaml").read_text(encoding="utf-8"))
    if config.get("dataset_config_version") != "phase3-dataset-config-v2":
        raise SystemExit("PHASE03_CONTRACT=FAIL dataset_config_version")
    if config.get("research_dataset_version") != RESEARCH_VERSION:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_dataset_version")
    if config.get("split_seed") != 20260908:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_split_seed")

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
    if coverage.candidate_count != 24 or coverage.supported_mapping_count != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_candidate_counts")
    if set(coverage.supported_family_counts_by_root_cause_code.values()) != {3}:
        raise SystemExit("PHASE03_CONTRACT=FAIL postmortem_family_depth_counts")
    if coverage.preserved_original_source_count != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_count")
    if set(coverage.preserved_family_counts_by_root_cause_code.values()) != {3}:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_distribution")
    if coverage.admitted_research_record_count != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_admission_count")
    if not coverage.taxonomy_coverage_sufficient_for_locked_holdout:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_taxonomy_coverage")
    if coverage.blocker != "none":
        raise SystemExit("PHASE03_CONTRACT=FAIL research_candidate_blocker")

    preservation_plan = load_primary_source_plan(root / "configs/phase3-primary-sources.json")
    preservation_manifest = load_primary_source_manifest(
        root
        / "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/manifest.json"
    )
    preservation = validate_primary_source_preservation(
        root,
        postmortem_index,
        preservation_plan,
        preservation_manifest,
    )
    if preservation.preserved_candidate_count != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_count")
    if not preservation.preserved_supported_family_depth_sufficient_for_split:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_depth")

    admission_plan = load_research_admission_plan(root / config["research_admission_config_path"])
    admitted = validate_research_admission_plan(
        postmortem_index,
        admission_plan,
        taxonomy,
        require_applied=True,
    )
    if len(admitted) != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL explicit_admission_plan")

    public_snapshot_dir = (
        root / "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1"
    )
    public_sums = {}
    for line in (public_snapshot_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        public_sums[name.strip()] = digest
    local_candidates = [
        item
        for item in postmortem_index.candidates
        if item.source_path.startswith(
            "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/"
        )
    ]
    if len(local_candidates) != 16:
        raise SystemExit("PHASE03_CONTRACT=FAIL public_snapshot_candidate_count")
    for candidate in local_candidates:
        path = root / candidate.source_path
        data = path.read_bytes()
        header = f"blob {len(data)}\0".encode()
        if hashlib.sha1(header + data).hexdigest() != candidate.source_blob_sha:
            raise SystemExit("PHASE03_CONTRACT=FAIL public_snapshot_git_blob")
        if public_sums.get(path.name) != hashlib.sha256(data).hexdigest():
            raise SystemExit("PHASE03_CONTRACT=FAIL public_snapshot_sha256")

    fixture = root / "datasets/incident_diagnosis/fixtures/ci_smoke"
    fixture_records = read_jsonl(fixture / "incidents.jsonl", IncidentRecord)
    fixture_families = read_jsonl(fixture / "families.jsonl", IncidentFamily)
    fixture_report = audit_dataset(
        fixture_records,
        fixture_families,
        taxonomy,
        research_mode=False,
    )
    if not fixture_report.passed:
        raise SystemExit(
            "PHASE03_CONTRACT=FAIL fixture_audit="
            + json.dumps(fixture_report.model_dump(mode="json"), sort_keys=True)
        )

    research_dir = root / "datasets/incident_diagnosis/processed" / RESEARCH_VERSION
    research_records = read_jsonl(research_dir / "incidents.jsonl", IncidentRecord)
    research_families = read_jsonl(research_dir / "families.jsonl", IncidentFamily)
    research_manifest = DatasetManifest.model_validate(
        json.loads((research_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    research_report = audit_dataset(
        research_records,
        research_families,
        taxonomy,
        research_mode=True,
    )
    if not research_report.passed:
        errors = [
            finding.code
            for finding in research_report.findings
            if finding.severity == FindingSeverity.ERROR
        ]
        raise SystemExit(f"PHASE03_CONTRACT=FAIL research_audit={errors}")
    if not research_manifest.research_ready:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_manifest_not_ready")
    if len(research_records) != 18 or len(research_families) != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_counts")
    if Counter(record.split for record in research_records) != {
        Split.TRAIN: 6,
        Split.VALIDATION: 6,
        Split.TEST: 6,
    }:
        raise SystemExit("PHASE03_CONTRACT=FAIL research_split_counts")
    for split in Split:
        if Counter(
            record.root_cause_code for record in research_records if record.split == split
        ) != {label: 1 for label in taxonomy.label_to_category}:
            raise SystemExit(f"PHASE03_CONTRACT=FAIL research_stratification={split.value}")
    if any(record.is_synthetic for record in research_records):
        raise SystemExit("PHASE03_CONTRACT=FAIL synthetic_base_record")

    _run_check(root, "scripts/check_phase3_reproducibility.py", "candidate_reproducibility")
    _run_check(
        root,
        "scripts/check_phase3_research_reproducibility.py",
        "research_reproducibility",
    )
    _run_check(root, "scripts/check_phase3_exit.py", "phase3_exit")

    print("PHASE03_CONTRACT=PASS")
    print(f"CI_SMOKE_RECORDS={len(fixture_records)}")
    print(f"CI_SMOKE_FAMILIES={len(fixture_families)}")
    print(f"PUBLIC_POSTMORTEM_CANDIDATES={coverage.candidate_count}")
    print(f"PUBLIC_POSTMORTEM_SUPPORTED_MAPPINGS={coverage.supported_mapping_count}")
    print(f"PRESERVED_PRIMARY_SOURCES={preservation.preserved_candidate_count}")
    print(f"RESEARCH_ADMITTED={coverage.admitted_research_record_count}")
    print(f"RESEARCH_RECORDS={len(research_records)}")
    print(f"RESEARCH_FAMILIES={len(research_families)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
