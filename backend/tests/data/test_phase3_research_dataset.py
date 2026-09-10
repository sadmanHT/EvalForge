from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from app.data.audit import FindingSeverity, audit_dataset
from app.data.io import read_jsonl, write_jsonl
from app.data.postmortems import load_candidate_index
from app.data.primary_sources import load_primary_source_manifest
from app.data.research import (
    ResearchAdmissionPlan,
    build_research_records_and_families,
    load_research_admission_plan,
    stable_research_id,
    validate_research_admission_plan,
)
from app.data.schemas import IncidentFamily, IncidentRecord, Split
from app.data.taxonomy import load_taxonomy


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_research_admission_plan_matches_all_supported_preserved_candidates() -> None:
    root = _root()
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")

    candidates = validate_research_admission_plan(
        index,
        plan,
        taxonomy,
        require_applied=True,
    )
    assert len(candidates) == 18
    assert len({candidate.candidate_id for candidate in candidates}) == 18
    assert Counter(candidate.proposed_root_cause_code for candidate in candidates) == {
        label: 3 for label in taxonomy.label_to_category
    }
    assert all(candidate.original_source_snapshot_preserved for candidate in candidates)
    assert all(candidate.research_admitted for candidate in candidates)


def test_research_admission_rejects_partial_candidate_set() -> None:
    root = _root()
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    payload = plan.model_dump(mode="json")
    payload["candidate_ids"] = payload["candidate_ids"][:-1]
    partial = ResearchAdmissionPlan.model_validate(payload)

    with pytest.raises(ValueError, match="exactly match supported preserved candidates"):
        validate_research_admission_plan(
            index,
            partial,
            taxonomy,
            require_applied=True,
        )


def test_locked_research_corpus_is_family_stratified_and_deterministic() -> None:
    root = _root()
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")

    first_records, first_families = build_research_records_and_families(index, plan, taxonomy)
    second_records, second_families = build_research_records_and_families(index, plan, taxonomy)
    assert first_records == second_records
    assert first_families == second_families
    assert len(first_records) == 18
    assert len(first_families) == 18
    assert len({record.incident_family_id for record in first_records}) == 18
    assert Counter(record.split for record in first_records) == {
        Split.TRAIN: 6,
        Split.VALIDATION: 6,
        Split.TEST: 6,
    }
    assert Counter(family.split for family in first_families) == {
        Split.TRAIN: 6,
        Split.VALIDATION: 6,
        Split.TEST: 6,
    }
    for split in Split:
        split_records = [record for record in first_records if record.split == split]
        assert Counter(record.root_cause_code for record in split_records) == {
            label: 1 for label in taxonomy.label_to_category
        }
    assert all(not record.is_synthetic for record in first_records)
    assert all(record.difficulty_tier.value == "unknown" for record in first_records)


def test_locked_research_corpus_passes_research_leakage_audit() -> None:
    root = _root()
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    records, families = build_research_records_and_families(index, plan, taxonomy)

    report = audit_dataset(records, families, taxonomy, research_mode=True)
    assert report.passed
    assert not [finding for finding in report.findings if finding.severity == FindingSeverity.ERROR]
    assert report.split_counts == {"test": 6, "train": 6, "validation": 6}
    assert report.class_counts == {label: 3 for label in sorted(taxonomy.label_to_category)}
    assert {finding.code for finding in report.findings} == {"MISSING_DIFFICULTY_TIERS"}


def test_research_provenance_matches_primary_source_manifest() -> None:
    root = _root()
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    primary_manifest = load_primary_source_manifest(
        root
        / "datasets/incident_diagnosis/raw/public_incidents/phase3-primary-source-v1/manifest.json"
    )
    snapshots = {snapshot.candidate_id: snapshot for snapshot in primary_manifest.snapshots}
    records, _ = build_research_records_and_families(index, plan, taxonomy)

    for record in records:
        candidate_id = record.source_provenance.source_record_id
        snapshot = snapshots[candidate_id]
        assert record.source_provenance.research_eligible
        assert record.source_provenance.source_checksum == snapshot.snapshot_sha256
        assert record.metadata["preserved_primary_source_path"] == snapshot.snapshot_path
        assert record.metadata["preserved_primary_source_kind"] == snapshot.snapshot_kind.value


def test_research_jsonl_round_trip_preserves_identity_and_provenance(tmp_path: Path) -> None:
    root = _root()
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    records, families = build_research_records_and_families(index, plan, taxonomy)

    incident_path = tmp_path / "incidents.jsonl"
    family_path = tmp_path / "families.jsonl"
    write_jsonl(incident_path, records, sort_key="incident_id")
    write_jsonl(family_path, families, sort_key="family_id")
    assert read_jsonl(incident_path, IncidentRecord) == records
    assert read_jsonl(family_path, IncidentFamily) == families


def test_research_ids_are_stable_and_domain_separated() -> None:
    candidate_id = "example:incident:1"
    assert stable_research_id("incident", candidate_id) == stable_research_id(
        "incident", candidate_id
    )
    assert stable_research_id("incident", candidate_id) != stable_research_id(
        "family", candidate_id
    )
