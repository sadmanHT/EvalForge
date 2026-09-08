from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

PIPELINES = {"ZERO_SHOT", "RAG", "FINETUNED", "COMBINED"}
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class ContractError(ValueError):
    """Raised when a Phase 01 scientific/reproducibility invariant is violated."""


def load_json_yaml(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"{path} must contain an object")
    return data


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def validate_label_taxonomy(taxonomy: dict[str, Any]) -> None:
    version = taxonomy.get("taxonomy_version")
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise ContractError("taxonomy_version must be semantic version X.Y.Z")

    change_policy = taxonomy.get("change_policy")
    if not isinstance(change_policy, str) or not change_policy.strip():
        raise ContractError("taxonomy change_policy is required")

    categories = taxonomy.get("categories")
    labels = taxonomy.get("labels")
    if not isinstance(categories, list) or not categories:
        raise ContractError("categories must be a non-empty list")
    if not isinstance(labels, list) or not labels:
        raise ContractError("labels must be a non-empty list")

    category_ids: list[str] = []
    for item in categories:
        if not isinstance(item, dict):
            raise ContractError("each category must be an object")
        category_id = item.get("id")
        if not isinstance(category_id, str) or not CANONICAL_ID.fullmatch(category_id):
            raise ContractError(f"invalid canonical category id: {category_id!r}")
        category_ids.append(category_id)
    if len(category_ids) != len(set(category_ids)):
        raise ContractError("duplicate category id")

    label_ids: list[str] = []
    for label in labels:
        if not isinstance(label, dict):
            raise ContractError("each label must be an object")
        label_id = label.get("id")
        category = label.get("category")
        if not isinstance(label_id, str) or not CANONICAL_ID.fullmatch(label_id):
            raise ContractError(f"invalid canonical label id: {label_id!r}")
        if category not in category_ids:
            raise ContractError(f"label {label_id} references unknown category {category!r}")
        label_ids.append(label_id)
    if len(label_ids) != len(set(label_ids)):
        raise ContractError("duplicate canonical label id")


def canonical_label_ids(taxonomy: dict[str, Any]) -> set[str]:
    validate_label_taxonomy(taxonomy)
    return {str(item["id"]) for item in taxonomy["labels"]}


def category_for_label(code: str, taxonomy: dict[str, Any]) -> str:
    validate_root_cause_code(code, taxonomy)
    for row in taxonomy["labels"]:
        if row["id"] == code:
            return str(row["category"])
    raise AssertionError("validated label disappeared")  # pragma: no cover - defensive unreachable after validation


def validate_root_cause_code(code: str, taxonomy: dict[str, Any]) -> None:
    if code not in canonical_label_ids(taxonomy):
        raise ContractError(f"unknown root_cause_code: {code}")


def validate_model_config(model: dict[str, Any]) -> None:
    required = (
        "study_id",
        "base_model_id",
        "base_model_revision",
        "license",
        "frozen",
        "selection_record",
        "smoke_test",
    )
    missing = [key for key in required if key not in model]
    if missing:
        raise ContractError(f"model config missing fields: {missing}")
    if model["frozen"] is not True:
        raise ContractError("primary model must be frozen")
    if not isinstance(model["base_model_id"], str) or "/" not in model["base_model_id"]:
        raise ContractError("base_model_id must be a Hub-style owner/model id")
    revision = model["base_model_revision"]
    if not isinstance(revision, str) or not GIT_SHA40.fullmatch(revision):
        raise ContractError("base_model_revision must be a full immutable 40-character git SHA")
    smoke = model["smoke_test"]
    if not isinstance(smoke, dict) or smoke.get("required_for_phase_complete") is not True:
        raise ContractError("model smoke must be required for phase completion")


def validate_study_config(study: dict[str, Any], model: dict[str, Any], taxonomy: dict[str, Any]) -> None:
    validate_model_config(model)
    validate_label_taxonomy(taxonomy)
    if study.get("study_id") != model.get("study_id"):
        raise ContractError("study/model study_id mismatch")
    if study.get("label_taxonomy_version") != taxonomy.get("taxonomy_version"):
        raise ContractError("study/taxonomy version mismatch")
    if set(study.get("pipelines", [])) != PIPELINES or len(study.get("pipelines", [])) != 4:
        raise ContractError("primary study must define exactly four canonical pipelines")
    if study.get("primary_metric") != "exact_root_cause_code_accuracy":
        raise ContractError("primary metric must be exact root-cause-code accuracy")
    if study.get("locked_test") is not True:
        raise ContractError("primary test must be locked")
    if study.get("split_unit") != "incident_family":
        raise ContractError("split unit must be incident_family")
    if not isinstance(study.get("hypotheses_record"), str):
        raise ContractError("study must link the pre-specified hypotheses record")

    fields = study.get("required_experiment_fields")
    non_null = study.get("required_non_null_run_fields")
    if not isinstance(fields, list) or not fields:
        raise ContractError("required_experiment_fields must be non-empty")
    if len(fields) != len(set(fields)):
        raise ContractError("required_experiment_fields contains duplicates")
    if not isinstance(non_null, list) or not non_null:
        raise ContractError("required_non_null_run_fields must be non-empty")
    if len(non_null) != len(set(non_null)):
        raise ContractError("required_non_null_run_fields contains duplicates")
    unknown_non_null = set(non_null) - set(fields)
    if unknown_non_null:
        raise ContractError(f"required_non_null_run_fields are not experiment fields: {sorted(unknown_non_null)}")

    for key in ("pipeline_required_fields", "pipeline_forbidden_non_null_fields"):
        rules = study.get(key)
        if not isinstance(rules, dict) or set(rules) != PIPELINES:
            raise ContractError(f"{key} must define all four canonical pipelines")
        for pipeline, names in rules.items():
            if not isinstance(names, list):
                raise ContractError(f"{key}.{pipeline} must be a list")
            unknown = set(names) - set(fields)
            if unknown:
                raise ContractError(f"{key}.{pipeline} contains unknown experiment fields: {sorted(unknown)}")


def validate_experiment_can_run(experiment: dict[str, Any], study: dict[str, Any], model: dict[str, Any]) -> None:
    required = study["required_experiment_fields"]
    missing = [key for key in required if key not in experiment]
    if missing:
        raise ContractError(f"experiment cannot RUN; missing fields: {missing}")

    required_non_null = study["required_non_null_run_fields"]
    blank = [key for key in required_non_null if _is_blank(experiment.get(key))]
    if blank:
        raise ContractError(f"experiment cannot RUN; blank required fields: {blank}")

    if experiment["study_id"] != study["study_id"]:
        raise ContractError("experiment study_id does not match primary study")
    pipeline = experiment["pipeline_type"]
    if pipeline not in PIPELINES:
        raise ContractError("invalid pipeline_type")
    if experiment["label_taxonomy_version"] != study["label_taxonomy_version"]:
        raise ContractError("experiment uses wrong label_taxonomy_version")
    if experiment["confidence_method"] != study["confidence_method"]:
        raise ContractError("experiment uses wrong confidence_method")
    if experiment["base_model_id"] != model["base_model_id"]:
        raise ContractError("primary experiment uses wrong base_model_id")
    if experiment["base_model_revision"] != model["base_model_revision"]:
        raise ContractError("primary experiment uses wrong base_model_revision")

    pipeline_required = study["pipeline_required_fields"][pipeline]
    blank_pipeline = [key for key in pipeline_required if _is_blank(experiment.get(key))]
    if blank_pipeline:
        raise ContractError(f"{pipeline} experiment missing pipeline-specific reproducibility fields: {blank_pipeline}")

    forbidden = study["pipeline_forbidden_non_null_fields"][pipeline]
    populated_forbidden = [key for key in forbidden if not _is_blank(experiment.get(key))]
    if populated_forbidden:
        raise ContractError(f"{pipeline} experiment has forbidden non-null fields: {populated_forbidden}")

    if _is_blank(experiment.get("reranker_id")) != _is_blank(experiment.get("reranker_revision")):
        raise ContractError("reranker_id and reranker_revision must be set or null together")


def validate_primary_comparison(experiments: Iterable[dict[str, Any]], study: dict[str, Any], model: dict[str, Any]) -> None:
    rows = list(experiments)
    if not rows:
        raise ContractError("comparison requires experiments")
    for row in rows:
        validate_experiment_can_run(row, study, model)

    frozen_fields = study["primary_comparison_frozen_fields"]
    baseline = rows[0]
    for row in rows[1:]:
        for field in frozen_fields:
            if row.get(field) != baseline.get(field):
                raise ContractError(
                    f"primary comparison mismatch in frozen field {field}: "
                    f"{baseline.get(field)!r} != {row.get(field)!r}"
                )
