#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from app.data.hashing import canonical_json_bytes, sha256_hex
from app.data.manifest import build_manifest
from app.data.opssentinel import inspect_opssentinel_catalog
from app.data.postmortems import inspect_candidate_coverage, load_candidate_index
from app.data.schemas import SourceSnapshot
from app.data.servicenow import inspect_servicenow_archive
from app.data.taxonomy import load_taxonomy


def build_outputs(root: Path) -> dict[str, str]:
    config = json.loads((root / "configs/dataset.yaml").read_text(encoding="utf-8"))
    source_cfg = config["source_snapshot"]
    source_path = root / source_cfg["path"]
    source_bytes = source_path.read_bytes()
    source_sha = sha256_hex(source_bytes)
    source_payload = json.loads(source_bytes)
    inspection = inspect_opssentinel_catalog(source_payload)

    auxiliary_cfg = config["auxiliary_source_snapshots"][0]
    auxiliary_path = root / auxiliary_cfg["path"]
    auxiliary_inspection = inspect_servicenow_archive(auxiliary_path)

    postmortem_cfg = config["public_postmortem_candidate_source"]
    postmortem_index_path = root / postmortem_cfg["candidate_index_path"]
    postmortem_index_sha = sha256_hex(postmortem_index_path.read_bytes())
    postmortem_index = load_candidate_index(postmortem_index_path)
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    postmortem_coverage = inspect_candidate_coverage(postmortem_index, taxonomy)
    if postmortem_coverage.source_repository != postmortem_cfg["repository"]:
        raise ValueError("postmortem candidate repository does not match dataset config")
    if postmortem_coverage.source_commit != postmortem_cfg["commit"]:
        raise ValueError("postmortem candidate commit does not match dataset config")
    expected_member_sha = auxiliary_cfg["canonical_member_sha256"]
    accepted_archive_shas = set(auxiliary_cfg["accepted_archive_sha256s"])
    if auxiliary_inspection.member_sha256 != expected_member_sha:
        raise ValueError("ServiceNow canonical CSV payload checksum does not match dataset config")
    if auxiliary_inspection.archive_sha256 not in accepted_archive_shas:
        raise ValueError("ServiceNow transport archive checksum is not an accepted equivalent")

    processed = root / "datasets/incident_diagnosis/processed" / config["candidate_dataset_version"]
    processed.mkdir(parents=True, exist_ok=True)

    source_summary = {
        "candidate_dataset_version": config["candidate_dataset_version"],
        "source_repository": source_cfg["repository"],
        "source_commit": source_cfg["commit"],
        "source_snapshot_sha256": source_sha,
        "inspection": inspection.model_dump(mode="json"),
    }
    source_summary_text = json.dumps(source_summary, indent=2, sort_keys=True) + "\n"
    (processed / "source-candidate-summary.json").write_text(source_summary_text, encoding="utf-8")

    source_snapshot = SourceSnapshot(
        source_repository=source_cfg["repository"],
        source_commit=source_cfg["commit"],
        snapshot_path=source_cfg["path"],
        snapshot_sha256=source_sha,
        source_generated=True,
        research_eligible_as_independent_heldout_evidence=False,
        source_name="OpsSentinel BenchmarkLab release catalog",
        notes="Generated controlled benchmark source; not independent held-out evidence.",
    )
    auxiliary_snapshot = SourceSnapshot(
        source_repository=auxiliary_cfg["source_uri"],
        source_commit=auxiliary_cfg["source_version"],
        snapshot_path=auxiliary_cfg["path"],
        snapshot_sha256=auxiliary_inspection.member_sha256,
        snapshot_identity_kind=("zip_member_sha256:" + auxiliary_cfg["canonical_member_filename"]),
        accepted_transport_sha256s=sorted(accepted_archive_shas),
        source_generated=False,
        research_eligible_as_independent_heldout_evidence=False,
        source_name=auxiliary_cfg["name"],
        source_uri=auxiliary_cfg["source_uri"],
        source_license=auxiliary_cfg["license"],
        notes=(
            "Real operational auxiliary corpus. Human-readable RCA text is unavailable, so "
            "anonymous categories must not be mapped to EvalForge root-cause labels."
        ),
    )
    postmortem_snapshot = SourceSnapshot(
        source_repository=postmortem_cfg["repository"],
        source_commit=postmortem_cfg["commit"],
        snapshot_path=postmortem_cfg["candidate_index_path"],
        snapshot_sha256=postmortem_index_sha,
        snapshot_identity_kind="evalforge_candidate_index_sha256",
        source_generated=False,
        research_eligible_as_independent_heldout_evidence=False,
        source_name=postmortem_cfg["name"],
        source_uri="https://postmortems.app/",
        source_license=postmortem_cfg["license"],
        notes=(
            "Pinned independent-public-postmortem discovery index. Candidate mappings are not "
            "research-admitted until original primary-source evidence is preserved and frozen "
            "taxonomy coverage is sufficient."
        ),
    )
    limitations = list(inspection.blockers) + list(auxiliary_inspection.blockers)
    limitations.extend(
        [
            (
                "Independent public incident candidates now provide the splitter's minimum "
                "three supported families per frozen taxonomy label, but no candidate is "
                "research-admitted until original primary-source evidence is preserved "
                "and canonical records are built."
            ),
            (
                "Supported public-postmortem candidate mappings are not research-admitted "
                "until original primary-source evidence is preserved."
            ),
            (
                "Keyword/title matches are not accepted as root-cause evidence; false "
                "friends and contributing factors are explicitly rejected."
            ),
        ]
    )
    manifest = build_manifest(
        dataset_version=config["candidate_dataset_version"],
        schema_version=config["schema_version"],
        label_taxonomy_version=config["label_taxonomy_version"],
        source_snapshots=[source_snapshot, auxiliary_snapshot, postmortem_snapshot],
        split_seed=config["split_seed"],
        generated_at=datetime.fromisoformat(config["generation_timestamp"].replace("Z", "+00:00")),
        records=[],
        families=[],
        source_candidate_count=inspection.scenario_count,
        auxiliary_source_counts={
            "servicenow_uci_incidents": auxiliary_inspection.incident_count,
            "servicenow_uci_event_rows": auxiliary_inspection.event_row_count,
            "public_postmortem_candidates": postmortem_coverage.candidate_count,
            "public_postmortem_supported_mappings": postmortem_coverage.supported_mapping_count,
            "public_postmortem_supported_taxonomy_labels": len(
                postmortem_coverage.supported_root_cause_codes
            ),
        },
        research_ready=False,
        limitations=limitations,
    )
    manifest_text = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    (processed / "manifest.json").write_text(manifest_text, encoding="utf-8")

    auxiliary_inspection_payload = auxiliary_inspection.model_dump(mode="json")
    auxiliary_inspection_payload.pop("archive_sha256")
    auxiliary_inspection_payload["canonical_member_sha256"] = expected_member_sha
    auxiliary_inspection_payload["accepted_transport_archive_sha256s"] = sorted(
        accepted_archive_shas
    )
    auxiliary_summary = {
        "name": auxiliary_cfg["name"],
        "source_uri": auxiliary_cfg["source_uri"],
        "source_version": auxiliary_cfg["source_version"],
        "license": auxiliary_cfg["license"],
        "role": auxiliary_cfg["role"],
        "snapshot_identity": auxiliary_cfg["snapshot_identity"],
        "inspection": auxiliary_inspection_payload,
    }
    auxiliary_text = json.dumps(auxiliary_summary, indent=2, sort_keys=True) + "\n"
    (processed / "auxiliary-servicenow-summary.json").write_text(auxiliary_text, encoding="utf-8")

    postmortem_payload = {
        "name": postmortem_cfg["name"],
        "repository": postmortem_cfg["repository"],
        "commit": postmortem_cfg["commit"],
        "license": postmortem_cfg["license"],
        "candidate_index_sha256": postmortem_index_sha,
        "coverage": postmortem_coverage.model_dump(mode="json"),
        "candidates": [item.model_dump(mode="json") for item in postmortem_index.candidates],
    }
    postmortem_text = json.dumps(postmortem_payload, indent=2, sort_keys=True) + "\n"
    (processed / "public-postmortem-candidate-coverage.json").write_text(
        postmortem_text, encoding="utf-8"
    )

    source_audit = {
        "status": "BLOCKED_BENCHMARK_EXPANSION_REQUIRED",
        "passed_leakage_safe_research_exit": False,
        "source_generated_count": inspection.source_generated_count,
        "source_split_counts": inspection.source_split_counts,
        "derived_generation_family_count": inspection.derived_generation_family_count,
        "cross_split_generation_families": inspection.cross_split_generation_families,
        "research_eligible_as_independent_heldout_evidence": False,
        "public_postmortem_candidate_count": postmortem_coverage.candidate_count,
        "public_postmortem_supported_mapping_count": postmortem_coverage.supported_mapping_count,
        "public_postmortem_supported_root_cause_codes": (
            postmortem_coverage.supported_root_cause_codes
        ),
        "public_postmortem_missing_root_cause_codes": postmortem_coverage.missing_root_cause_codes,
        "public_postmortem_research_admitted_count": (
            postmortem_coverage.admitted_research_record_count
        ),
        "blocker": postmortem_coverage.blocker,
        "blockers": inspection.blockers,
    }
    audit_text = json.dumps(source_audit, indent=2, sort_keys=True) + "\n"
    (processed / "source-audit-report.json").write_text(audit_text, encoding="utf-8")

    return {
        "manifest_sha256": sha256_hex(manifest_text.encode()),
        "source_summary_sha256": sha256_hex(source_summary_text.encode()),
        "source_audit_sha256": sha256_hex(audit_text.encode()),
        "source_snapshot_sha256": source_sha,
        "auxiliary_servicenow_sha256": sha256_hex(auxiliary_text.encode()),
        "auxiliary_archive_sha256": auxiliary_inspection.archive_sha256,
        "public_postmortem_coverage_sha256": sha256_hex(postmortem_text.encode()),
        "public_postmortem_candidate_index_sha256": postmortem_index_sha,
        "manifest_identity_checksum": manifest.manifest_checksum,
        "content_checksum": manifest.content_checksum,
        "source_summary_identity_sha256": sha256_hex(canonical_json_bytes(source_summary)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = build_outputs(args.root.resolve())
    for key, value in sorted(result.items()):
        print(f"{key.upper()}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
