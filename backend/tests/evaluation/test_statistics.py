from __future__ import annotations

import pytest

from app.evaluation.statistics import (
    StatisticsInputError,
    compare_paired_accuracy,
    exact_mcnemar_test,
    paired_bootstrap_accuracy_difference,
)


def test_identical_experiments_have_zero_difference_and_p_one() -> None:
    outcomes = {f"i{index}": index % 2 == 0 for index in range(20)}
    bootstrap = paired_bootstrap_accuracy_difference(outcomes, outcomes, resamples=1000, seed=7)
    mcnemar = exact_mcnemar_test(outcomes, outcomes)
    assert bootstrap.observed_difference == 0.0
    assert bootstrap.ci_lower == 0.0
    assert bootstrap.ci_upper == 0.0
    assert mcnemar.discordant_pairs == 0
    assert mcnemar.p_value == 1.0


def test_intentionally_improved_predictions_show_positive_paired_difference() -> None:
    baseline = {f"i{index}": index < 4 for index in range(20)}
    improved = {f"i{index}": index < 16 for index in range(20)}
    result = compare_paired_accuracy(baseline, improved, resamples=2000, seed=11)
    assert result.bootstrap.observed_difference == pytest.approx(0.6)
    assert result.bootstrap.ci_lower > 0.0
    assert result.mcnemar.a_only_correct == 0
    assert result.mcnemar.b_only_correct == 12
    assert result.mcnemar.p_value < 0.001


def test_paired_statistics_require_identical_incident_sets() -> None:
    with pytest.raises(StatisticsInputError, match="same incident IDs"):
        exact_mcnemar_test({"a": True}, {"b": True})
