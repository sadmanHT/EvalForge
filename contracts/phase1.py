from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PIPELINES = {"ZERO_SHOT", "RAG", "FINETUNED", "COMBINED"}


class ContractError(ValueError):
    """Raised when a Phase 01 research/configuration invariant is violated."""


def load_json_yaml(path: str | Path) -> dict[str, Any]:
    """Load Phase 01 JSON-compatible YAML without third-party dependencies."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"{path} must contain an object")
    return data


def validate_label_taxonomy(taxonomy: dict[str, Any]) -> None:
    version = taxonomy.get("taxonomy_version")
    if not isinstance(version, str) or not version.strip():
        raise ContractError("taxonomy_version is required")

    categories = taxonomy.get("categories")
    labels = taxonomy.get("labels")
    if not isinstance(categories, list) or not categories:
        raise ContractError("categories must be a non-empty list")
    if not isinstance(labels, list) or not labels:
        raise ContractError("labels must be a non-empty list")

    category_ids = [item.get("id") for item in categories if isinstance(item, dict)]
    if len(category_ids) != len(categories) or any(
        not isinstance(category_id, str) or not category_id for category_id in category_ids
    ):
        raise ContractError("every category requires a non-empty string id")
    if len(category_ids) != len(set(category_ids)):
        raise ContractError("duplicate category id")

    label_ids: list[str] = []
    for label in labels:
        if not isinstance(label, dict):
            raise ContractError("each label must be an object")
        label_id = label.get("id")
        category = label.get("category")
        if not isinstance(label_id, str) or not label_id:
            raise ContractError("every label requires a non-empty string id")
        if category not in category_ids:
            raise ContractError(f"label {label_id} references unknown category {category!r}")
        label_ids.append(label_id)
    if len(label_ids) != len(set(label_ids)):
        raise ContractError("duplicate canonical label id")


def canonical_label_ids(taxonomy: dict[str, Any]) -> set[str]:
    validate_label_taxonomy(taxonomy)
    return {str(item["id"]) for item in taxonomy["labels"]}


def validate_root_cause_code(code: str, taxonomy: dict[str, Any]) -> None:
    if code not in canonical_label_ids(taxonomy):
        raise ContractError(f"unknown root_cause_code: {code}")


def validate_model_config(model: dict[str, Any]) -> None:
    required = ("study_id", "base_model_id", "base_model_revision", "frozen")
    missing = [key for key in required if key not in model]
    if missing:
        raise ContractError(f"model config missing fields: {missing}")
    if model["frozen"] is not True:
        raise ContractError("primary model must be frozen")
    revision = model["base_model_revision"]
    if not isinstance(revision, str) or len(revision) < 12:
        raise ContractError("base_model_revision must be an exact immutable revision")


def validate_study_config(
    study: dict[str, Any],
    model: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    validate_model_config(model)
    validate_label_taxonomy(taxonomy)
    if study.get("study_id") != model.get("study_id"):
        raise ContractError("study/model study_id mismatch")
    if set(study.get("pipelines", [])) != PIPELINES:
        raise ContractError("primary study must define exactly four canonical pipelines")
    if study.get("primary_metric") != "exact_root_cause_code_accuracy":
        raise ContractError("primary metric must be exact root-cause-code accuracy")
    if study.get("locked_test") is not True:
        raise ContractError("primary test must be locked")
    fields = study.get("required_experiment_fields")
    if not isinstance(fields, list) or not fields:
        raise ContractError("required_experiment_fields must be non-empty")
    if len(fields) != len(set(fields)):
        raise ContractError("required_experiment_fields contains duplicates")


def validate_experiment_can_run(
    experiment: dict[str, Any],
    study: dict[str, Any],
    model: dict[str, Any],
) -> None:
    required = study["required_experiment_fields"]
    missing = [key for key in required if key not in experiment]
    if missing:
        raise ContractError(f"experiment cannot RUN; missing fields: {missing}")

    if experiment["study_id"] != study["study_id"]:
        raise ContractError("experiment study_id does not match primary study")
    if experiment["pipeline_type"] not in PIPELINES:
        raise ContractError("invalid pipeline_type")
    if experiment["base_model_id"] != model["base_model_id"]:
        raise ContractError("primary experiment uses wrong base_model_id")
    if experiment["base_model_revision"] != model["base_model_revision"]:
        raise ContractError("primary experiment uses wrong base_model_revision")


def validate_primary_comparison(
    experiments: Iterable[dict[str, Any]],
    study: dict[str, Any],
    model: dict[str, Any],
) -> None:
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
