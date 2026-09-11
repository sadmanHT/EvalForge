from __future__ import annotations

from app.data.schemas import (
    DifficultyTier,
    IncidentFamily,
    IncidentRecord,
    SourceProvenance,
    Split,
)


def build_ci_smoke_dataset() -> tuple[list[IncidentRecord], list[IncidentFamily]]:
    taxonomy = {
        "n_plus_one_query": "database_behavior",
        "database_connection_leak": "database_behavior",
        "disk_exhaustion": "resource_exhaustion",
        "memory_leak": "resource_exhaustion",
        "broken_payment_configuration": "configuration",
        "no_fault": "control",
    }
    specs = [
        ("ci-train-n1", "family-train-n1", Split.TRAIN, "n_plus_one_query", "easy"),
        (
            "ci-train-conn",
            "family-train-conn",
            Split.TRAIN,
            "database_connection_leak",
            "medium",
        ),
        ("ci-val-disk", "family-val-disk", Split.VALIDATION, "disk_exhaustion", "hard"),
        ("ci-test-memory", "family-test-memory", Split.TEST, "memory_leak", "adversarial"),
        (
            "ci-test-config",
            "family-test-config",
            Split.TEST,
            "broken_payment_configuration",
            "compound",
        ),
        ("ci-val-control", "family-val-control", Split.VALIDATION, "no_fault", "easy"),
    ]
    records: list[IncidentRecord] = []
    families: list[IncidentFamily] = []
    for incident_id, family_id, split, label, difficulty in specs:
        record = IncidentRecord(
            incident_id=incident_id,
            incident_family_id=family_id,
            title=f"Fixture {incident_id}",
            description=f"Deterministic non-research fixture narrative unique to {incident_id}.",
            service="fixture-service",
            severity="P2",
            root_cause_code=label,
            root_cause_category=taxonomy[label],
            evidence=[f"fixture:evidence:{incident_id}"],
            difficulty_tier=DifficultyTier(difficulty),
            source_provenance=SourceProvenance(
                source_name="evalforge-ci",
                source_kind="ci_fixture",
                source_record_id=incident_id,
                research_eligible=False,
                notes="Non-research deterministic CI fixture.",
            ),
            split=split,
            is_synthetic=False,
            metadata={"fixture": True},
        )
        records.append(record)
        families.append(
            IncidentFamily(
                family_id=family_id,
                split=split,
                source_family_key=family_id,
                member_incident_ids=[incident_id],
                root_cause_codes=[label],
                metadata={"fixture": True},
            )
        )

    parent = records[0]
    synthetic = IncidentRecord(
        **{
            **parent.model_dump(mode="python"),
            "incident_id": "ci-train-n1-synthetic",
            "title": "Synthetic fixture descendant",
            "description": "Augmented training-only wording for fixture lineage validation.",
            "is_synthetic": True,
            "parent_incident_id": parent.incident_id,
            "generator_model_id": "fixture-generator",
            "generator_model_revision": "fixture-revision",
            "generator_prompt_version": "fixture-prompt-v1",
            "metadata": {"fixture": True, "augmentation_parent": parent.incident_id},
        }
    )
    records.append(synthetic)
    families[0] = IncidentFamily(
        family_id=families[0].family_id,
        split=Split.TRAIN,
        source_family_key=families[0].source_family_key,
        member_incident_ids=[parent.incident_id, synthetic.incident_id],
        root_cause_codes=[parent.root_cause_code],
        metadata={"fixture": True},
    )
    return records, families
