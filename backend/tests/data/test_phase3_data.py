from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.data.audit import audit_dataset
from app.data.fixtures import build_ci_smoke_dataset
from app.data.hashing import dataset_content_checksum, manifest_checksum
from app.data.io import read_jsonl, write_jsonl
from app.data.manifest import build_manifest
from app.data.postmortems import CandidateDecision, inspect_candidate_coverage, load_candidate_index
from app.data.opssentinel import (
    SourceEligibilityError,
    inspect_opssentinel_catalog,
    require_research_eligible_catalog,
)
from app.data.servicenow import inspect_servicenow_archive
from app.data.schemas import (
    DifficultyTier,
    IncidentFamily,
    IncidentRecord,
    SourceProvenance,
    SourceSnapshot,
    Split,
)
from app.data.split import FamilySeed, family_stratified_split
from app.data.taxonomy import LabelTaxonomy, validate_record_taxonomy
from app.data.augmentation import make_synthetic_child


TAXONOMY = LabelTaxonomy(
    version="1.0.0",
    label_to_category={
        "n_plus_one_query": "database_behavior",
        "database_connection_leak": "database_behavior",
        "disk_exhaustion": "resource_exhaustion",
        "memory_leak": "resource_exhaustion",
        "broken_payment_configuration": "configuration",
        "no_fault": "control",
    },
)


def base_record(
    *,
    incident_id: str = "incident-a",
    family_id: str = "family-a",
    split: Split = Split.TRAIN,
    label: str = "n_plus_one_query",
    description: str | None = None,
    research_eligible: bool = True,
) -> IncidentRecord:
    return IncidentRecord(
        incident_id=incident_id,
        incident_family_id=family_id,
        title=f"Incident title {incident_id}",
        description=description or f"Distinct evidence narrative for {incident_id}.",
        service="checkout",
        severity="P2",
        root_cause_code=label,
        root_cause_category=TAXONOMY.label_to_category[label],
        evidence=[f"metric:{incident_id}"],
        difficulty_tier=DifficultyTier.EASY,
        source_provenance=SourceProvenance(
            source_name="test",
            source_kind="unit_fixture",
            source_record_id=incident_id,
            research_eligible=research_eligible,
        ),
        split=split,
    )


def families_for(records: list[IncidentRecord]) -> list[IncidentFamily]:
    grouped: dict[str, list[IncidentRecord]] = {}
    for record in records:
        grouped.setdefault(record.incident_family_id, []).append(record)
    return [
        IncidentFamily(
            family_id=family_id,
            split=members[0].split,
            source_family_key=family_id,
            member_incident_ids=[item.incident_id for item in members],
            root_cause_codes=sorted({item.root_cause_code for item in members}),
        )
        for family_id, members in sorted(grouped.items())
    ]


def finding_codes(report: object) -> set[str]:
    return {item.code for item in report.findings}  # type: ignore[attr-defined]


def test_synthetic_schema_requires_complete_provenance() -> None:
    payload = base_record().model_dump(mode="python")
    payload["is_synthetic"] = True
    payload["parent_incident_id"] = "parent"
    with pytest.raises(ValidationError):
        IncidentRecord.model_validate(payload)


def test_taxonomy_rejects_wrong_category() -> None:
    record = base_record()
    wrong = record.model_copy(update={"root_cause_category": "configuration"})
    with pytest.raises(ValueError, match="category mismatch"):
        validate_record_taxonomy(wrong, TAXONOMY)


def test_family_split_is_deterministic_disjoint_and_stratified() -> None:
    families = [
        FamilySeed(family_id=f"family-{label}-{index}", stratify_label=label)
        for label in ("a", "b")
        for index in range(8)
    ]
    first = family_stratified_split(families, seed=17)
    second = family_stratified_split(list(reversed(families)), seed=17)
    assert first == second
    assert set(first) == {item.family_id for item in families}
    for label in ("a", "b"):
        splits = {
            first[item.family_id]
            for item in families
            if item.stratify_label == label
        }
        assert splits == {Split.TRAIN, Split.VALIDATION, Split.TEST}


def test_manifest_hashing_is_stable_for_identical_inputs() -> None:
    records = [base_record()]
    families = families_for(records)
    snapshot = SourceSnapshot(
        source_repository="example/source",
        source_commit="abc",
        snapshot_path="raw.json",
        snapshot_sha256="123",
        source_generated=False,
        research_eligible_as_independent_heldout_evidence=True,
    )
    kwargs = dict(
        dataset_version="v1",
        schema_version="incident-schema-v1",
        label_taxonomy_version="1.0.0",
        source_snapshots=[snapshot],
        split_seed=7,
        generated_at=datetime(2026, 9, 8, tzinfo=UTC),
        records=records,
        families=families,
        research_ready=True,
        limitations=[],
    )
    first = build_manifest(**kwargs)
    second = build_manifest(**kwargs)
    assert first.content_checksum == second.content_checksum
    assert first.manifest_checksum == second.manifest_checksum
    assert first.content_checksum == dataset_content_checksum(records, families)
    payload = first.model_dump(mode="json")
    assert first.manifest_checksum == manifest_checksum(payload)


def test_synthetic_child_inherits_train_family_and_records_generator() -> None:
    parent = base_record()
    child = make_synthetic_child(
        parent,
        incident_id="synthetic-a",
        title="Augmented train incident",
        description="Paraphrased training-only fixture.",
        generator_model_id="model",
        generator_model_revision="rev",
        generator_prompt_version="prompt-v1",
    )
    assert child.split == Split.TRAIN
    assert child.incident_family_id == parent.incident_family_id
    assert child.parent_incident_id == parent.incident_id
    assert child.is_synthetic


def test_synthetic_child_rejects_nontraining_parent() -> None:
    parent = base_record(split=Split.VALIDATION)
    with pytest.raises(ValueError, match="training parents"):
        make_synthetic_child(
            parent,
            incident_id="synthetic-a",
            title="bad",
            description="bad",
            generator_model_id="model",
            generator_model_revision="rev",
            generator_prompt_version="prompt-v1",
        )


def test_audit_detects_cross_split_family_overlap() -> None:
    records = [
        base_record(incident_id="a", family_id="shared", split=Split.TRAIN),
        base_record(incident_id="b", family_id="shared", split=Split.TEST),
    ]
    report = audit_dataset(records, families_for(records), TAXONOMY, research_mode=False)
    assert not report.passed
    assert "CROSS_SPLIT_FAMILY" in finding_codes(report)


def test_audit_detects_exact_and_near_duplicate_text_across_splits() -> None:
    exact = "Checkout requests slow while query volume rises sharply."
    left = base_record(
        incident_id="a", family_id="fa", split=Split.TRAIN, description=exact
    ).model_copy(update={"title": "Shared duplicate title"})
    right = base_record(
        incident_id="b", family_id="fb", split=Split.TEST, description=exact
    ).model_copy(update={"title": "Shared duplicate title"})
    records = [left, right]
    report = audit_dataset(records, families_for(records), TAXONOMY, research_mode=False)
    assert "CROSS_SPLIT_EXACT_DUPLICATE" in finding_codes(report)

    near_records = [
        base_record(
            incident_id="c",
            family_id="fc",
            split=Split.TRAIN,
            description=(
                "Checkout requests slow while database query volume rises sharply after release."
            ),
        ),
        base_record(
            incident_id="d",
            family_id="fd",
            split=Split.TEST,
            description=(
                "Checkout requests slow while database query volume rises sharply after a release."
            ),
        ),
    ]
    near_report = audit_dataset(
        near_records,
        families_for(near_records),
        TAXONOMY,
        research_mode=False,
        near_duplicate_threshold=0.90,
    )
    assert "CROSS_SPLIT_NEAR_DUPLICATE" in finding_codes(near_report)


def test_audit_detects_heldout_identifier_reference_in_training() -> None:
    records = [
        base_record(
            incident_id="train",
            family_id="ftrain",
            split=Split.TRAIN,
            description="Training note mentions forbidden family-test-token explicitly.",
        ),
        base_record(
            incident_id="test",
            family_id="family-test-token",
            split=Split.TEST,
        ),
    ]
    report = audit_dataset(records, families_for(records), TAXONOMY, research_mode=False)
    assert "FORBIDDEN_HELDOUT_REFERENCE" in finding_codes(report)


def test_audit_rejects_noneligible_research_holdout() -> None:
    record = base_record(
        incident_id="heldout",
        family_id="heldout-family",
        split=Split.TEST,
        research_eligible=False,
    )
    report = audit_dataset([record], families_for([record]), TAXONOMY, research_mode=True)
    assert "INELIGIBLE_HELDOUT_SOURCE" in finding_codes(report)


def test_distribution_audit_reports_imbalance_and_missing_tiers() -> None:
    records = [
        base_record(incident_id=f"a-{i}", family_id=f"fa-{i}")
        for i in range(4)
    ]
    records.append(
        base_record(
            incident_id="b",
            family_id="fb",
            label="memory_leak",
        )
    )
    report = audit_dataset(records, families_for(records), TAXONOMY, research_mode=False)
    codes = finding_codes(report)
    assert "SEVERE_CLASS_IMBALANCE" in codes
    assert "MISSING_DIFFICULTY_TIERS" in codes


def test_round_trip_preserves_identity_labels_evidence_provenance_and_split(
    tmp_path: Path,
) -> None:
    records, families = build_ci_smoke_dataset()
    incidents_path = tmp_path / "incidents.jsonl"
    families_path = tmp_path / "families.jsonl"
    write_jsonl(incidents_path, records, sort_key="incident_id")
    write_jsonl(families_path, families, sort_key="family_id")
    loaded_records = read_jsonl(incidents_path, IncidentRecord)
    loaded_families = read_jsonl(families_path, IncidentFamily)
    assert [item.model_dump(mode="json") for item in loaded_records] == [
        item.model_dump(mode="json")
        for item in sorted(records, key=lambda item: item.incident_id)
    ]
    assert [item.model_dump(mode="json") for item in loaded_families] == [
        item.model_dump(mode="json")
        for item in sorted(families, key=lambda item: item.family_id)
    ]


def test_ci_smoke_fixture_is_leakage_clean_in_nonresearch_mode() -> None:
    records, families = build_ci_smoke_dataset()
    report = audit_dataset(records, families, TAXONOMY, research_mode=False)
    assert report.passed, report.model_dump(mode="json")


def test_opssentinel_snapshot_is_detected_as_generated_and_split_leaky() -> None:
    root = Path(__file__).resolve().parents[3]
    raw = (
        root
        / "datasets/incident_diagnosis/raw/opssentinel"
        / "fae661fc1634aad6a3855a1dec8dcddb16a890dd"
        / "release-catalog.snapshot.json"
    )
    payload = json.loads(raw.read_text(encoding="utf-8"))
    inspection = inspect_opssentinel_catalog(payload)
    assert inspection.scenario_count == 50
    assert inspection.source_split_counts == {"dev": 30, "hidden_test": 10, "validation": 10}
    assert inspection.derived_generation_family_count == 29
    assert len(inspection.cross_split_generation_families) == 3
    assert not inspection.research_eligible_as_independent_heldout_evidence
    with pytest.raises(SourceEligibilityError, match="benchmark expansion"):
        require_research_eligible_catalog(payload)


def test_servicenow_uci_snapshot_is_real_but_not_rca_holdout_eligible() -> None:
    root = Path(__file__).resolve().parents[3]
    raw = (
        root
        / "datasets/incident_diagnosis/raw/servicenow_uci/uci-498"
        / "incident-management-process-enriched-event-log.zip"
    )
    inspection = inspect_servicenow_archive(raw)
    assert inspection.event_row_count == 141712
    assert inspection.incident_count == 24918
    assert inspection.column_count == 36
    assert inspection.archive_sha256 in {
        "3ea92768cb2cfada908dd601b057d7066127fe75dc8fe19e4abaa1c6766a6c13",
        "6294e29a311647306bfdfc85783f7df66517c197b9cd49aa5ee36ba9c525d1d6",
    }
    assert inspection.member_sha256 == (
        "fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef"
    )
    assert inspection.incident_field_presence_counts["problem_id"] == 381
    assert inspection.incident_field_presence_counts["rfc"] == 179
    assert inspection.incident_field_presence_counts["caused_by"] == 3
    assert inspection.real_world_operational_source
    assert not inspection.source_generated
    assert not inspection.has_human_readable_root_cause_text
    assert not inspection.research_eligible_as_independent_heldout_evidence


def test_public_postmortem_candidate_coverage_is_conservative() -> None:
    root = Path(__file__).resolve().parents[3]
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    coverage = inspect_candidate_coverage(index, TAXONOMY)
    assert coverage.candidate_count == 7
    assert coverage.supported_mapping_count == 2
    assert coverage.supported_root_cause_codes == ["disk_exhaustion", "memory_leak"]
    assert coverage.missing_root_cause_codes == [
        "broken_payment_configuration",
        "database_connection_leak",
        "n_plus_one_query",
        "no_fault",
    ]
    assert coverage.admitted_research_record_count == 0
    assert not coverage.taxonomy_coverage_sufficient_for_locked_holdout
    assert coverage.blocker == "independent_taxonomy_coverage_insufficient"


def test_postmortem_keyword_false_friends_are_not_admitted() -> None:
    root = Path(__file__).resolve().parents[3]
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    by_id = {item.candidate_id: item for item in index.candidates}

    cloudflare = by_id[
        "postmortems-app:91f1928a-c9cd-4473-a761-dc0b07914b2e"
    ]
    assert cloudflare.proposed_root_cause_code == "memory_leak"
    assert cloudflare.decision == CandidateDecision.REJECTED
    assert cloudflare.mapping_basis == "terminology_false_friend"
    assert not cloudflare.research_admitted

    incident_io = by_id[
        "postmortems-app:eb95646f-90c2-4efe-b89e-060debafa0fc"
    ]
    assert incident_io.proposed_root_cause_code == "n_plus_one_query"
    assert incident_io.decision == CandidateDecision.REJECTED
    assert incident_io.mapping_basis == "contributing_factor"
    assert not incident_io.research_admitted


def test_supported_postmortem_candidates_still_require_primary_source_snapshot() -> None:
    root = Path(__file__).resolve().parents[3]
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    supported = [
        item for item in index.candidates if item.decision == CandidateDecision.SUPPORTED
    ]
    assert {item.company for item in supported} == {"Amazon", "Tarsnap"}
    assert all(not item.original_source_snapshot_preserved for item in supported)
    assert all(not item.research_admitted for item in supported)
