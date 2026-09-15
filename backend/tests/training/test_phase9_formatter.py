import json
from pathlib import Path

import pytest

from app.data.schemas import DifficultyTier, IncidentRecord, SourceProvenance, Split
from app.training.formatter import format_incident, prepare_training_dataset

ROOT = Path(__file__).resolve().parents[3]
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


def test_prepare_dataset_is_deterministic_and_excludes_test(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = prepare_training_dataset(
        root=ROOT,
        output_dir=first,
        dataset_version=DATASET_VERSION,
    )
    manifest_b = prepare_training_dataset(
        root=ROOT,
        output_dir=second,
        dataset_version=DATASET_VERSION,
    )
    assert manifest_a == manifest_b
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (first / "validation.jsonl").read_bytes() == (second / "validation.jsonl").read_bytes()
    train_rows = [
        json.loads(line)
        for line in (first / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (first / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train_rows and validation_rows
    assert {row["source_split"] for row in train_rows} == {"train"}
    assert {row["source_split"] for row in validation_rows} == {"validation"}
    assert set(manifest_a.train_family_ids).isdisjoint(manifest_a.validation_family_ids)


def test_formatter_refuses_locked_test_record() -> None:
    record = IncidentRecord(
        incident_id="i-test",
        incident_family_id="f-test",
        title="title",
        description="description",
        root_cause_code="memory_leak",
        root_cause_category="resource_exhaustion",
        difficulty_tier=DifficultyTier.UNKNOWN,
        source_provenance=SourceProvenance(
            source_name="fixture",
            source_kind="fixture",
            source_record_id="r",
            research_eligible=True,
        ),
        split=Split.TEST,
    )
    with pytest.raises(ValueError, match="locked test"):
        format_incident(
            record,
            allowed_labels=("memory_leak",),
            include_reasoning=True,
        )
