from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import Field

from app.data.schemas import StrictModel


class SourceEligibilityError(ValueError):
    pass


class UpstreamCatalogInspection(StrictModel):
    scenario_count: int
    source_generated_count: int
    source_split_counts: dict[str, int]
    label_counts: dict[str, int]
    difficulty_counts: dict[str, int]
    derived_generation_family_count: int
    cross_split_generation_families: dict[str, list[str]]
    research_eligible_as_independent_heldout_evidence: bool
    blockers: list[str] = Field(default_factory=list)


def derive_generation_family(scenario: dict[str, Any]) -> str:
    structure = scenario["structure"]
    counterfactual_family = structure.get("counterfactual_family")
    if counterfactual_family:
        return f"opssentinel:counterfactual:{counterfactual_family}"
    combination_family = structure.get("combination_family")
    if combination_family:
        return f"opssentinel:compound:{combination_family}"

    faults = scenario.get("faults") or []
    fault = faults[0]["fault"] if faults else "no_fault"
    failure_structure = structure["failure_structure"]
    if failure_structure.endswith("single_noisy_dependency"):
        generator_family = "single_noisy_dependency"
    elif failure_structure == "dev_single_strong_signal":
        generator_family = "single_strong_signal"
    elif failure_structure == "dev_delayed_multiservice_signal":
        generator_family = "delayed_multiservice_signal"
    elif failure_structure in {
        "validation_misleading_change_correlation",
        "hidden_cron_triggered_cause",
    }:
        generator_family = "misleading_change_temporal"
    else:
        generator_family = failure_structure
    return f"opssentinel:{generator_family}:{fault}"


def inspect_opssentinel_catalog(payload: dict[str, Any]) -> UpstreamCatalogInspection:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("OpsSentinel release catalog must contain scenarios")

    splits = Counter()
    labels = Counter()
    difficulties = Counter()
    family_splits: dict[str, set[str]] = defaultdict(set)
    for scenario in scenarios:
        splits[str(scenario["split"])] += 1
        labels[str(scenario["ground_truth"]["primary_root_cause_code"])] += 1
        difficulties[str(scenario["difficulty"])] += 1
        family_splits[derive_generation_family(scenario)].add(str(scenario["split"]))

    overlaps = {
        family_id: sorted(values)
        for family_id, values in sorted(family_splits.items())
        if len(values) > 1
    }
    blockers = [
        "The pinned OpsSentinel BenchmarkLab release catalog is programmatically generated, "
        "so its validation/hidden-test scenarios are not independent production incident evidence.",
        "EvalForge must acquire/version independent incident families before a research validation "
        "or test split can be locked.",
    ]
    if overlaps:
        blockers.append(
            "The upstream dev/validation/hidden-test assignments also split several inferred "
            "generation families across source splits; those assignments must not be reused."
        )
    return UpstreamCatalogInspection(
        scenario_count=len(scenarios),
        source_generated_count=len(scenarios),
        source_split_counts=dict(sorted(splits.items())),
        label_counts=dict(sorted(labels.items())),
        difficulty_counts=dict(sorted(difficulties.items())),
        derived_generation_family_count=len(family_splits),
        cross_split_generation_families=overlaps,
        research_eligible_as_independent_heldout_evidence=False,
        blockers=blockers,
    )


def require_research_eligible_catalog(payload: dict[str, Any]) -> None:
    inspection = inspect_opssentinel_catalog(payload)
    if not inspection.research_eligible_as_independent_heldout_evidence:
        raise SourceEligibilityError(
            "OpsSentinel release catalog is generated benchmark data, not independent "
            "research holdout evidence; benchmark expansion is required."
        )
