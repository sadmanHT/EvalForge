from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.data.schemas import DifficultyTier, IncidentRecord, SourceProvenance, Split
from app.inference.data_efficiency import (
    aggregate_efficiency_metric,
    build_data_efficiency_matrix,
    select_data_efficiency_subset,
)
from app.inference.efficiency_study import (
    build_efficiency_training_bundle,
    efficiency_condition_id,
    prepare_efficiency_condition,
)
from app.inference.finetuned_protocol import load_phase10_protocol


def _record(index: int, split: Split = Split.TRAIN) -> IncidentRecord:
    return IncidentRecord(
        incident_id=f"incident-{index}",
        incident_family_id=f"family-{index}",
        title=f"Incident {index}",
        description="fixture",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        root_cause_code=f"label-{index}",
        root_cause_category="fixture",
        difficulty_tier=DifficultyTier.EASY,
        source_provenance=SourceProvenance(
            source_name="fixture",
            source_kind="test",
            source_record_id=f"source-{index}",
            research_eligible=True,
        ),
        split=split,
    )


def test_fraction_matrix_is_family_nested_and_train_only() -> None:
    records = [_record(index) for index in range(6)]
    matrix = build_data_efficiency_matrix(
        records,
        fractions=(0.10, 0.25, 0.50, 1.00),
        seeds=(11, 12),
    )
    assert len(matrix) == 8
    seed_11 = [item for item in matrix if item.seed == 11]
    assert [item.selected_family_count for item in seed_11] == [1, 2, 3, 6]
    previous: set[str] = set()
    for item in seed_11:
        current = set(item.selected_family_ids)
        assert previous <= current
        previous = current


def test_data_efficiency_rejects_held_out_records() -> None:
    with pytest.raises(ValueError, match="training-split"):
        select_data_efficiency_subset([_record(1, Split.TEST)], fraction=1.0, seed=1)


def test_data_efficiency_selection_is_deterministic() -> None:
    records = [_record(index) for index in range(6)]
    first = select_data_efficiency_subset(records, fraction=0.5, seed=20260908)
    second = select_data_efficiency_subset(list(reversed(records)), fraction=0.5, seed=20260908)
    assert first == second


def test_statistical_aggregation_reports_population_variation() -> None:
    aggregate = aggregate_efficiency_metric((0.5, 1.0, 0.5))
    assert aggregate["count"] == 3
    assert aggregate["mean"] == pytest.approx(2 / 3)
    assert aggregate["variance"] == pytest.approx(1 / 18)
    assert aggregate["stddev"] == pytest.approx((1 / 18) ** 0.5)


ROOT = Path(__file__).resolve().parents[3]


def test_repository_efficiency_preparation_uses_train_only_and_full_validation(
    tmp_path: Path,
) -> None:
    protocol = load_phase10_protocol(ROOT)
    prepared = prepare_efficiency_condition(
        root=ROOT,
        output_dir=tmp_path / "condition",
        protocol=protocol,
        fraction=0.25,
        seed=20260908,
    )
    assert prepared.train_record_count == 2
    assert len(prepared.selected_family_ids) == 2
    assert prepared.validation_record_count == 6
    assert prepared.primary_adapter_selection_use is False
    assert prepared.locked_test_split_used is False
    assert set(prepared.selected_family_ids) <= set(prepared.population_train_family_ids)


def test_efficiency_training_seed_is_the_only_seed_override() -> None:
    base = build_efficiency_training_bundle(ROOT, seed=20260908)
    alternate = build_efficiency_training_bundle(ROOT, seed=20260909)
    assert base.lora.model_copy(update={"seed": 20260909}) == alternate.lora
    assert base.training.model_copy(update={"seed": 20260909}) == alternate.training
    assert base.lora.base_model_id == alternate.lora.base_model_id
    assert base.lora.base_model_revision == alternate.lora.base_model_revision


def test_efficiency_condition_id_is_stable() -> None:
    assert efficiency_condition_id(0.10, 20260908) == "phase10-efficiency-f0100-s20260908"
    assert efficiency_condition_id(0.25, 20260909) == "phase10-efficiency-f0250-s20260909"
    assert efficiency_condition_id(1.00, 20260910) == "phase10-efficiency-f1000-s20260910"
