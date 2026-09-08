from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from app.data.schemas import Split


@dataclass(frozen=True)
class FamilySeed:
    family_id: str
    stratify_label: str


def _rank(seed: int, family_id: str) -> str:
    return hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()


def family_stratified_split(
    families: list[FamilySeed],
    *,
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Split]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation + test fractions must be < 1")
    ids = [item.family_id for item in families]
    if len(ids) != len(set(ids)):
        raise ValueError("family IDs must be unique")

    groups: dict[str, list[FamilySeed]] = defaultdict(list)
    for item in families:
        groups[item.stratify_label].append(item)

    assignments: dict[str, Split] = {}
    for label in sorted(groups):
        group = sorted(
            groups[label],
            key=lambda item: (_rank(seed, item.family_id), item.family_id),
        )
        n = len(group)
        if n >= 3:
            n_test = max(1, round(n * test_fraction)) if test_fraction else 0
            n_validation = max(1, round(n * validation_fraction)) if validation_fraction else 0
            while n_test + n_validation >= n:
                if n_validation >= n_test and n_validation > 0:
                    n_validation -= 1
                elif n_test > 0:
                    n_test -= 1
        else:
            n_test = 0
            n_validation = 0

        test_ids = {item.family_id for item in group[:n_test]}
        validation_ids = {item.family_id for item in group[n_test : n_test + n_validation]}
        for item in group:
            if item.family_id in test_ids:
                assignments[item.family_id] = Split.TEST
            elif item.family_id in validation_ids:
                assignments[item.family_id] = Split.VALIDATION
            else:
                assignments[item.family_id] = Split.TRAIN
    return assignments
