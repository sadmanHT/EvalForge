from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.data.schemas import IncidentRecord


@dataclass(frozen=True)
class LabelTaxonomy:
    version: str
    label_to_category: dict[str, str]

    @property
    def labels(self) -> set[str]:
        return set(self.label_to_category)


def load_taxonomy(path: Path) -> LabelTaxonomy:
    # Phase 01 stores this YAML-compatible file as JSON, so no runtime YAML parser is needed.
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {item["id"]: item["category"] for item in payload["labels"]}
    if len(mapping) != len(payload["labels"]):
        raise ValueError("taxonomy contains duplicate label IDs")
    return LabelTaxonomy(version=payload["taxonomy_version"], label_to_category=mapping)


def validate_record_taxonomy(record: IncidentRecord, taxonomy: LabelTaxonomy) -> None:
    category = taxonomy.label_to_category.get(record.root_cause_code)
    if category is None:
        raise ValueError(f"unknown root_cause_code: {record.root_cause_code}")
    if category != record.root_cause_category:
        raise ValueError(
            f"category mismatch for {record.root_cause_code}: "
            f"expected {category}, got {record.root_cause_category}"
        )
