from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class StatisticsInputError(ValueError):
    pass


def _aligned_binary_outcomes(
    outcomes_a: Mapping[str, bool], outcomes_b: Mapping[str, bool]
) -> tuple[tuple[str, ...], list[int], list[int]]:
    if not outcomes_a or not outcomes_b:
        raise StatisticsInputError("paired statistics require non-empty outcomes")
    if set(outcomes_a) != set(outcomes_b):
        raise StatisticsInputError("paired experiments must cover exactly the same incident IDs")
    incident_ids = tuple(sorted(outcomes_a))
    a = [int(bool(outcomes_a[incident_id])) for incident_id in incident_ids]
    b = [int(bool(outcomes_b[incident_id])) for incident_id in incident_ids]
    return incident_ids, a, b


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise StatisticsInputError("quantile requires values")
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@dataclass(frozen=True)
class PairedBootstrapResult:
    observed_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    resamples: int
    seed: int


def paired_bootstrap_accuracy_difference(
    outcomes_a: Mapping[str, bool],
    outcomes_b: Mapping[str, bool],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260908,
) -> PairedBootstrapResult:
    """Bootstrap B-A accuracy difference using paired incident resampling."""

    _, a, b = _aligned_binary_outcomes(outcomes_a, outcomes_b)
    if resamples < 100:
        raise StatisticsInputError("paired bootstrap requires at least 100 resamples")
    if not 0.0 < confidence_level < 1.0:
        raise StatisticsInputError("confidence_level must be in (0,1)")
    n = len(a)
    observed = (sum(b) - sum(a)) / n
    generator = random.Random(seed)
    differences: list[float] = []
    for _ in range(resamples):
        difference_sum = 0
        for _ in range(n):
            index = generator.randrange(n)
            difference_sum += b[index] - a[index]
        differences.append(difference_sum / n)
    differences.sort()
    alpha = 1.0 - confidence_level
    return PairedBootstrapResult(
        observed_difference=observed,
        ci_lower=_quantile(differences, alpha / 2.0),
        ci_upper=_quantile(differences, 1.0 - alpha / 2.0),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )


@dataclass(frozen=True)
class McNemarResult:
    a_only_correct: int
    b_only_correct: int
    discordant_pairs: int
    p_value: float


def exact_mcnemar_test(
    outcomes_a: Mapping[str, bool], outcomes_b: Mapping[str, bool]
) -> McNemarResult:
    """Two-sided exact McNemar/binomial test for paired binary correctness."""

    incident_ids, _, _ = _aligned_binary_outcomes(outcomes_a, outcomes_b)
    a_only = 0
    b_only = 0
    for incident_id in incident_ids:
        a_correct = bool(outcomes_a[incident_id])
        b_correct = bool(outcomes_b[incident_id])
        if a_correct and not b_correct:
            a_only += 1
        elif b_correct and not a_correct:
            b_only += 1
    discordant = a_only + b_only
    if discordant == 0:
        return McNemarResult(
            a_only_correct=0,
            b_only_correct=0,
            discordant_pairs=0,
            p_value=1.0,
        )
    tail = min(a_only, b_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    p_value = min(1.0, 2.0 * probability)
    return McNemarResult(
        a_only_correct=a_only,
        b_only_correct=b_only,
        discordant_pairs=discordant,
        p_value=p_value,
    )


@dataclass(frozen=True)
class PairedAccuracyComparison:
    bootstrap: PairedBootstrapResult
    mcnemar: McNemarResult


def compare_paired_accuracy(
    outcomes_a: Mapping[str, bool],
    outcomes_b: Mapping[str, bool],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260908,
) -> PairedAccuracyComparison:
    return PairedAccuracyComparison(
        bootstrap=paired_bootstrap_accuracy_difference(
            outcomes_a,
            outcomes_b,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
        ),
        mcnemar=exact_mcnemar_test(outcomes_a, outcomes_b),
    )
