#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


PRESERVED_IDS_OLD = '''    preserved_ids = {
        "github:Nikhil-Gautam-dev:medoc-n-plus-one:2026-01-11",
        "github:Dispatcharr:issue-1416:2026-07-06",
        "openlibrary:issue-12432:2026-04-22:n-plus-one",
        "dify:issue-40036:2026-08-05:n-plus-one",
    }
'''
PRESERVED_IDS_NEW = '''    preserved_ids = {
        "github:Nikhil-Gautam-dev:medoc-n-plus-one:2026-01-11",
        "github:Dispatcharr:issue-1416:2026-07-06",
        "openlibrary:issue-12432:2026-04-22:n-plus-one",
        "dify:issue-40036:2026-08-05:n-plus-one",
        "postmortems-app:76f27cf3-b204-40e4-942e-19657614f658",
        "honeycomb:2019-11-06:running-dry-on-memory",
        "dnsimple:2015-05-09:san-jose-memory-leak",
        "postmortems-app:e7d7aa93-81f7-4338-9c0b-6e6c0dcefdcb",
        "soundcloud:2011-08-24:binlog-disk-full",
        "git-nrw:2025-05-09:wal-disk-full",
        "coderden:2026-02-19:database-connection-leak",
        "altapay:2026-07-15:shopify-3ds-firewall-config",
    }
'''

CONTRACT_DISTRIBUTION_OLD = '''    if coverage.preserved_original_source_count != 4:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_count")
    if coverage.preserved_family_counts_by_root_cause_code != {
        "broken_payment_configuration": 0,
        "database_connection_leak": 1,
        "disk_exhaustion": 0,
        "memory_leak": 0,
        "n_plus_one_query": 3,
        "no_fault": 0,
    }:
'''
CONTRACT_DISTRIBUTION_NEW = '''    if coverage.preserved_original_source_count != 12:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_count")
    if coverage.preserved_family_counts_by_root_cause_code != {
        "broken_payment_configuration": 1,
        "database_connection_leak": 2,
        "disk_exhaustion": 3,
        "memory_leak": 3,
        "n_plus_one_query": 3,
        "no_fault": 0,
    }:
'''

REQUIRED_ANCHOR = '''    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/dispatcharr-1416.primary.json",
'''
REQUIRED_EXPANDED = REQUIRED_ANCHOR + '''    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/aws-2012-memory-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/honeycomb-2019-memory-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/dnsimple-2015-memory-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/tarsnap-2016-disk-full.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/soundcloud-2011-binlog-disk-full.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/git-nrw-2025-wal-disk-full.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/coderden-2026-connection-leak.primary.html",
    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/altapay-2026-firewall-payment-config.primary.html",
'''

KIND_ASSERTION_OLD = (
    '        assert candidate.preserved_primary_source_kind in {"git_blob", "github_issue"}\n'
)
KIND_ASSERTION_NEW = '''        assert candidate.preserved_primary_source_kind in {
            "git_blob",
            "github_issue",
            "http_document",
        }
'''


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one migration anchor in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    test_path = root / "backend/tests/data/test_phase3_data.py"
    contract_path = root / "scripts/check_phase3_contract.py"

    replace_once(test_path, PRESERVED_IDS_OLD, PRESERVED_IDS_NEW)
    replace_once(test_path, KIND_ASSERTION_OLD, KIND_ASSERTION_NEW)
    replace_once(contract_path, REQUIRED_ANCHOR, REQUIRED_EXPANDED)
    replace_once(contract_path, CONTRACT_DISTRIBUTION_OLD, CONTRACT_DISTRIBUTION_NEW)
    replace_once(
        contract_path,
        "if preservation.preserved_candidate_count != 4:",
        "if preservation.preserved_candidate_count != 12:",
    )
    print("PHASE03_WAVE2_EXPECTATION_MIGRATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
