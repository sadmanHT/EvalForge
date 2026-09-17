from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.data.schemas import IncidentRecord, Split


@dataclass(frozen=True)
class DataEfficiencySubset:
    fraction: float
    seed: int
    population_family_count: int
    selected_family_count: int
    selected_family_ids: tuple[str, ...]
    selected_incident_ids: tuple[str, ...]
    root_cause_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "fraction": self.fraction,
            "seed": self.seed,
            "population_family_count": self.population_family_count,
            "selected_family_count": self.selected_family_count,
            "selected_family_ids": list(self.selected_family_ids),
            "selected_incident_ids": list(self.selected_incident_ids),
            "root_cause_codes": list(self.root_cause_codes),
        }


def _family_order_key(family_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()
    return digest, family_id


def select_data_efficiency_subset(
    records: Sequence[IncidentRecord],
    *,
    fraction: float,
    seed: int,
) -> DataEfficiencySubset:
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be finite and within (0, 1]")
    if not records:
        raise ValueError("data-efficiency selection requires training records")
    if any(record.split is not Split.TRAIN for record in records):
        raise ValueError("data-efficiency subsets may contain only training-split records")

    by_family: dict[str, list[IncidentRecord]] = {}
    for record in records:
        by_family.setdefault(record.incident_family_id, []).append(record)

    family_ids = tuple(sorted(by_family, key=lambda item: _family_order_key(item, seed)))
    selected_count = min(len(family_ids), max(1, math.ceil(fraction * len(family_ids))))
    selected_families = frozenset(family_ids[:selected_count])
    selected_records = sorted(
        (
            record
            for family_id in selected_families
            for record in by_family[family_id]
        ),
        key=lambda record: record.incident_id,
    )

    for record in selected_records:
        if record.is_synthetic and record.parent_incident_id is None:
            raise ValueError("synthetic training record is missing parent lineage")

    return DataEfficiencySubset(
        fraction=float(fraction),
        seed=int(seed),
        population_family_count=len(family_ids),
        selected_family_count=selected_count,
        selected_family_ids=tuple(sorted(selected_families)),
        selected_incident_ids=tuple(record.incident_id for record in selected_records),
        root_cause_codes=tuple(sorted({record.root_cause_code for record in selected_records})),
    )


def build_data_efficiency_matrix(
    records: Sequence[IncidentRecord],
    *,
    fractions: Sequence[float],
    seeds: Sequence[int],
) -> tuple[DataEfficiencySubset, ...]:
    if not fractions or not seeds:
        raise ValueError("data-efficiency matrix requires fractions and seeds")
    matrix = tuple(
        select_data_efficiency_subset(records, fraction=fraction, seed=seed)
        for seed in seeds
        for fraction in fractions
    )
    by_seed: dict[int, list[DataEfficiencySubset]] = {}
    for subset in matrix:
        by_seed.setdefault(subset.seed, []).append(subset)
    for seed, subsets in by_seed.items():
        ordered = sorted(subsets, key=lambda item: item.fraction)
        previous: set[str] = set()
        for subset in ordered:
            current = set(subset.selected_family_ids)
            if not previous.issubset(current):
                raise AssertionError(f"data-efficiency subsets are not nested for seed {seed}")
            previous = current
    return matrix


def aggregate_efficiency_metric(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("at least one metric value is required")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("data-efficiency metric values must be finite")
    mean = sum(normalized) / len(normalized)
    variance = sum((value - mean) ** 2 for value in normalized) / len(normalized)
    return {
        "count": len(normalized),
        "mean": mean,
        "variance": variance,
        "stddev": math.sqrt(variance),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }
