#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ELEVENLABS_ID = "elevenlabs:2026-04-22:billing-system-misconfiguration"
VISA_ID = "visa-cybersource:2026-04-28:payer-auth-misconfiguration"
ELEVENLABS_URL = "https://status.elevenlabs.io/incidents/01KPTTQSDT67Y0ZHAQRH7BNC65"
ELEVENLABS_SNAPSHOT = (
    "datasets/incident_diagnosis/raw/public_incidents/phase3-candidate-v1/"
    "elevenlabs-2026-04-22-payment-configuration.snapshot.json"
)

NEW_PRIMARY_ENTRIES = [
    {
        "candidate_id": "atlassian:CRUC-8168:bonecp-connection-leak",
        "snapshot_path": (
            "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/"
            "atlassian-CRUC-8168-connection-leak.primary.html"
        ),
        "kind": "http_document",
        "source_url": "https://jira.atlassian.com/browse/CRUC-8168",
        "evidence_markers": [
            "BoneCP connections are not returned to the pool",
            "unavailability of database connections in the pool",
            "Crucible becomes not responsive",
        ],
    },
    {
        "candidate_id": "fastly:2023-02-28:cloud-waf-false-alarm",
        "snapshot_path": (
            "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/"
            "fastly-2023-cloud-waf-false-alarm.primary.html"
        ),
        "kind": "http_document",
        "source_url": "https://fastly.status.page/incidents?componentId=510172",
        "evidence_markers": [
            "status update to be a false alarm",
            "there are currently no elevated errors on Cloud WAF services",
            "All other locations and services are unaffected",
        ],
    },
    {
        "candidate_id": "gitlab:INC-889:2025-05-12:false-alarm-version-skew",
        "snapshot_path": (
            "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/"
            "gitlab-2025-version-skew-false-alarm.primary.html"
        ),
        "kind": "http_document",
        "source_url": "https://gitlab.com/gitlab-com/gl-infra/production/-/issues/19803",
        "evidence_markers": [
            "This turned out to be a false alarm",
            "no traffic was reaching the lingering pod",
            "there was no customer impact",
        ],
    },
    {
        "candidate_id": "google-cloud-status:E18Caoo5X1m6dTa1PVr1",
        "snapshot_path": (
            "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/"
            "google-2021-payment-config.primary.html"
        ),
        "kind": "http_document",
        "source_url": "https://status.cloud.google.com/incidents/E18Caoo5X1m6dTa1PVr1",
        "evidence_markers": [
            "update to the payment configuration intended to support virtual credit cards",
            "no credit card payments were able to be processed in the U.S. region",
            "rollback of the configuration change",
        ],
    },
    {
        "candidate_id": "google-cloud-status:fLYHLzSGXGkLkAjc8MJG",
        "snapshot_path": (
            "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/"
            "google-2024-mandiant-no-fault.primary.html"
        ),
        "kind": "http_document",
        "source_url": "https://status.cloud.google.com/incidents/fLYHLzSGXGkLkAjc8MJG",
        "evidence_markers": [
            "there was no service degradation",
            "no supported alerts from CrowdStrike were missed",
            "service continues to operate as intended",
        ],
    },
    {
        "candidate_id": ELEVENLABS_ID,
        "snapshot_path": (
            "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/"
            "elevenlabs-2026-billing-misconfiguration.primary.html"
        ),
        "kind": "http_document",
        "source_url": ELEVENLABS_URL,
        "evidence_markers": [
            "issues related to payment-processing",
            "A misconfiguration in our billing system",
            "provisioning of subscriptions to be delayed",
        ],
    },
]

PRESERVED_IDS_APPEND = '''        "altapay:2026-07-15:shopify-3ds-firewall-config",
    }
'''
PRESERVED_IDS_NEW = '''        "altapay:2026-07-15:shopify-3ds-firewall-config",
        "atlassian:CRUC-8168:bonecp-connection-leak",
        "fastly:2023-02-28:cloud-waf-false-alarm",
        "gitlab:INC-889:2025-05-12:false-alarm-version-skew",
        "google-cloud-status:E18Caoo5X1m6dTa1PVr1",
        "google-cloud-status:fLYHLzSGXGkLkAjc8MJG",
        "elevenlabs:2026-04-22:billing-system-misconfiguration",
    }
'''

NORMALIZED_DICT_OLD = '''        "google-cloud-status:fLYHLzSGXGkLkAjc8MJG": "no_fault",
    }
'''
NORMALIZED_DICT_NEW = '''        "google-cloud-status:fLYHLzSGXGkLkAjc8MJG": "no_fault",
        "elevenlabs:2026-04-22:billing-system-misconfiguration": "broken_payment_configuration",
    }
'''

UNPRESERVED_LOOP = '''    for candidate_id in (
        "google-cloud-status:E18Caoo5X1m6dTa1PVr1",
        "google-cloud-status:fLYHLzSGXGkLkAjc8MJG",
    ):
        assert not by_id[candidate_id].original_source_snapshot_preserved
'''

CONTRACT_DISTRIBUTION_OLD = '''    if coverage.preserved_original_source_count != 12:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_count")
    if coverage.preserved_family_counts_by_root_cause_code != {
        "broken_payment_configuration": 1,
        "database_connection_leak": 2,
        "disk_exhaustion": 3,
        "memory_leak": 3,
        "n_plus_one_query": 3,
        "no_fault": 0,
    }:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_distribution")
    if coverage.preserved_family_depth_sufficient_for_split:
        raise SystemExit("PHASE03_CONTRACT=FAIL premature_primary_source_family_depth")
'''
CONTRACT_DISTRIBUTION_NEW = '''    if coverage.preserved_original_source_count != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_count")
    if set(coverage.preserved_family_counts_by_root_cause_code.values()) != {3}:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_preservation_distribution")
    if not coverage.preserved_family_depth_sufficient_for_split:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_family_depth")
'''

CONTRACT_MANIFEST_OLD = '''    if preservation.preserved_candidate_count != 12:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_count")
    if preservation.preserved_supported_family_counts_by_root_cause_code != (
        coverage.preserved_family_counts_by_root_cause_code
    ):
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_distribution")
    if preservation.preserved_supported_family_depth_sufficient_for_split:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_depth")
'''
CONTRACT_MANIFEST_NEW = '''    if preservation.preserved_candidate_count != 18:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_count")
    if preservation.preserved_supported_family_counts_by_root_cause_code != (
        coverage.preserved_family_counts_by_root_cause_code
    ):
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_distribution")
    if not preservation.preserved_supported_family_depth_sufficient_for_split:
        raise SystemExit("PHASE03_CONTRACT=FAIL primary_source_manifest_depth")
'''

PRIMARY_TEST_OLD = '''    assert report.preserved_candidate_count == 12
    assert report.preserved_supported_family_counts_by_root_cause_code == {
        "broken_payment_configuration": 1,
        "database_connection_leak": 2,
        "disk_exhaustion": 3,
        "memory_leak": 3,
        "n_plus_one_query": 3,
        "no_fault": 0,
    }
    assert not report.preserved_supported_family_depth_sufficient_for_split
'''
PRIMARY_TEST_NEW = '''    assert report.preserved_candidate_count == 18
    assert report.preserved_supported_family_counts_by_root_cause_code == {
        "broken_payment_configuration": 3,
        "database_connection_leak": 3,
        "disk_exhaustion": 3,
        "memory_leak": 3,
        "n_plus_one_query": 3,
        "no_fault": 3,
    }
    assert report.preserved_supported_family_depth_sufficient_for_split
'''


def git_blob_sha(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()  # noqa: S324


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one migration anchor in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_elevenlabs_snapshot(root: Path) -> tuple[Path, bytes]:
    path = root / ELEVENLABS_SNAPSHOT
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing candidate snapshot: {path}")
    payload = {
        "normalized_facts": [
            "ElevenLabs investigated payment-processing issues affecting ElevenAPI on April 22, 2026.",
            "The official incident update identified the root cause as a misconfiguration in the billing system.",
            "The misconfiguration delayed provisioning of subscriptions; error rates returned to baseline before full resolution was confirmed.",
        ],
        "observed_at": "2026-09-10T00:00:00Z",
        "provenance_limitations": [
            "This file is an EvalForge normalized semantic snapshot, not byte-for-byte original source content.",
            "It supports candidate review provenance only; research admission requires the separate preserved primary-source artifact.",
        ],
        "research_admission_eligible": False,
        "snapshot_format": "evalforge-normalized-public-incident-v1",
        "snapshot_kind": "normalized_semantic_snapshot",
        "source_publisher": "ElevenLabs",
        "source_title": "Payment Issues",
        "source_url": ELEVENLABS_URL,
    }
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(data)
    return path, data


def migrate_candidate_index(root: Path, snapshot_data: bytes) -> None:
    path = root / "configs/postmortem-candidates.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    if payload.get("index_version") != "phase3-public-postmortem-candidates-v6":
        raise SystemExit("unexpected candidate-index version before wave3")
    if len(candidates) != 23:
        raise SystemExit("unexpected candidate count before wave3")
    supported = [item for item in candidates if item["decision"] == "supported_candidate_mapping"]
    if len(supported) != 18:
        raise SystemExit("unexpected supported count before wave3")
    if any(item["candidate_id"] == ELEVENLABS_ID for item in candidates):
        raise SystemExit("ElevenLabs candidate already exists")

    visa = next((item for item in candidates if item["candidate_id"] == VISA_ID), None)
    if visa is None or visa["decision"] != "supported_candidate_mapping":
        raise SystemExit("expected supported Visa candidate before wave3")
    if visa["original_source_snapshot_preserved"] or visa["research_admitted"]:
        raise SystemExit("Visa candidate unexpectedly preserved/admitted")
    visa["decision"] = "rejected_as_primary_mapping"
    visa["mapping_basis"] = "public_republication_restricted"
    visa["decision_reason"] = (
        "The official PIR is publicly indexed but explicitly marked confidential and prohibits "
        "copying/distribution. EvalForge therefore keeps the review trace but rejects it as a "
        "primary mapping for a public immutable benchmark rather than republishing restricted text."
    )

    candidates.append(
        {
            "candidate_id": ELEVENLABS_ID,
            "company": "ElevenLabs",
            "decision": "supported_candidate_mapping",
            "decision_reason": (
                "Official first-party status incident explicitly identifies a billing-system "
                "misconfiguration as the root cause of payment-processing/subscription provisioning impact."
            ),
            "evidence_summary": (
                "ElevenLabs investigated payment-processing issues affecting ElevenAPI on April 22, "
                "2026. The official update identified a billing-system misconfiguration that delayed "
                "subscription provisioning; error rates returned to baseline and full resolution was confirmed."
            ),
            "mapping_basis": "narrative_root_cause",
            "original_source_snapshot_preserved": False,
            "original_url": ELEVENLABS_URL,
            "proposed_root_cause_code": "broken_payment_configuration",
            "research_admitted": False,
            "source_blob_sha": git_blob_sha(snapshot_data),
            "source_path": ELEVENLABS_SNAPSHOT,
            "title": "Payment Issues",
        }
    )
    candidates.sort(key=lambda item: item["candidate_id"])
    payload["index_version"] = "phase3-public-postmortem-candidates-v7"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def migrate_candidate_sums(root: Path, snapshot_path: Path, snapshot_data: bytes) -> None:
    sums_path = snapshot_path.parent / "SHA256SUMS.txt"
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    if snapshot_path.name in sums:
        raise SystemExit("ElevenLabs snapshot unexpectedly already checksummed")
    sums[snapshot_path.name] = hashlib.sha256(snapshot_data).hexdigest()
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )


def migrate_primary_plan(root: Path) -> None:
    path = root / "configs/phase3-primary-sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_version") != "phase3-primary-source-preservation-v2":
        raise SystemExit("unexpected preservation plan version before wave3")
    if len(payload.get("entries", [])) != 12:
        raise SystemExit("unexpected preservation plan depth before wave3")
    existing_ids = {item["candidate_id"] for item in payload["entries"]}
    new_ids = {item["candidate_id"] for item in NEW_PRIMARY_ENTRIES}
    if existing_ids & new_ids:
        raise SystemExit("wave3 primary-source entry already present")
    payload["entries"].extend(NEW_PRIMARY_ENTRIES)
    payload["plan_version"] = "phase3-primary-source-preservation-v3"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def migrate_tests_and_contract(root: Path) -> None:
    data_test = root / "backend/tests/data/test_phase3_data.py"
    primary_test = root / "backend/tests/data/test_phase3_primary_sources.py"
    contract = root / "scripts/check_phase3_contract.py"
    preserve_script = root / "scripts/preserve_phase3_primary_sources.py"

    replace_once(data_test, "assert coverage.candidate_count == 23", "assert coverage.candidate_count == 24")
    replace_once(data_test, PRESERVED_IDS_APPEND, PRESERVED_IDS_NEW)
    replace_once(data_test, NORMALIZED_DICT_OLD, NORMALIZED_DICT_NEW)
    replace_once(data_test, UNPRESERVED_LOOP, "")

    extra_test = '''\n\ndef test_restricted_visa_source_is_rejected_and_public_replacement_is_supported() -> None:\n    root = Path(__file__).resolve().parents[3]\n    index = load_candidate_index(root / "configs/postmortem-candidates.json")\n    by_id = {item.candidate_id: item for item in index.candidates}\n\n    visa = by_id["visa-cybersource:2026-04-28:payer-auth-misconfiguration"]\n    assert visa.decision == CandidateDecision.REJECTED\n    assert visa.mapping_basis == "public_republication_restricted"\n    assert not visa.original_source_snapshot_preserved\n    assert not visa.research_admitted\n\n    replacement = by_id["elevenlabs:2026-04-22:billing-system-misconfiguration"]\n    assert replacement.decision == CandidateDecision.SUPPORTED\n    assert replacement.proposed_root_cause_code == "broken_payment_configuration"\n    assert replacement.original_source_snapshot_preserved\n    assert not replacement.research_admitted\n'''
    marker = "\n\ndef test_supported_postmortem_primary_source_wave_remains_nonadmitted() -> None:\n"
    text = data_test.read_text(encoding="utf-8")
    if text.count(marker) != 1:
        raise SystemExit("expected wave test insertion marker exactly once")
    data_test.write_text(text.replace(marker, extra_test + marker, 1), encoding="utf-8")

    replace_once(primary_test, PRIMARY_TEST_OLD, PRIMARY_TEST_NEW)

    required_anchor = (
        '    "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/'
        'altapay-2026-firewall-payment-config.primary.html",\n'
    )
    required_additions = required_anchor + ''.join(
        f'    "{item["snapshot_path"]}",\n' for item in NEW_PRIMARY_ENTRIES
    ) + f'    "{ELEVENLABS_SNAPSHOT}",\n'
    replace_once(contract, required_anchor, required_additions)
    replace_once(
        contract,
        "if coverage.candidate_count != 23 or coverage.supported_mapping_count != 18:",
        "if coverage.candidate_count != 24 or coverage.supported_mapping_count != 18:",
    )
    replace_once(contract, CONTRACT_DISTRIBUTION_OLD, CONTRACT_DISTRIBUTION_NEW)
    replace_once(contract, CONTRACT_MANIFEST_OLD, CONTRACT_MANIFEST_NEW)
    replace_once(contract, "if len(local_candidates) != 15:", "if len(local_candidates) != 16:")
    replace_once(
        preserve_script,
        'manifest_version="phase3-primary-source-manifest-v2",',
        'manifest_version="phase3-primary-source-manifest-v3",',
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    snapshot_path, snapshot_data = write_elevenlabs_snapshot(root)
    migrate_candidate_index(root, snapshot_data)
    migrate_candidate_sums(root, snapshot_path, snapshot_data)
    migrate_primary_plan(root)
    migrate_tests_and_contract(root)
    print("PHASE03_WAVE3_EXPECTATION_MIGRATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
