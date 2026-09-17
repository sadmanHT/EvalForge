from datetime import UTC, datetime

import pytest

from app.data.schemas import DifficultyTier, IncidentRecord, SourceProvenance, Split
from app.inference.data_efficiency import (
    aggregate_efficiency_metric,
    build_data_efficiency_matrix,
    select_data_efficiency_subset,
)


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
