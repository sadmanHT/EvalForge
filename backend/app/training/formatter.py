from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.data.schemas import IncidentRecord, Split
from app.inference.prompts import (
    BASELINE_PROMPT_VERSION,
    OUTPUT_SCHEMA_VERSION,
    get_prompt_template,
)
from app.inference.protocol import load_taxonomy

FORMATTER_VERSION = "phase9-instruction-formatter-v1"
PREPARED_MANIFEST_VERSION = "phase9-prepared-dataset-manifest-v1"


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    formatter_version: str = FORMATTER_VERSION
    incident_id: str = Field(min_length=1)
    incident_family_id: str = Field(min_length=1)
    source_split: Literal["train", "validation"]
    is_synthetic: bool
    parent_incident_id: str | None = None
    root_cause_code: str = Field(min_length=1)
    messages: tuple[dict[str, str], dict[str, str]]


class PreparedDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = PREPARED_MANIFEST_VERSION
    formatter_version: str = FORMATTER_VERSION
    dataset_version: str
    dataset_manifest_checksum: str
    dataset_content_checksum: str
    prompt_version: str
    output_schema_version: str
    include_reasoning_target: bool
    train_record_count: int = Field(ge=0)
    validation_record_count: int = Field(ge=0)
    train_family_ids: tuple[str, ...]
    validation_family_ids: tuple[str, ...]
    train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _target_reasoning(record: IncidentRecord, include_reasoning: bool) -> str | None:
    if not include_reasoning:
        return None
    for item in record.evidence:
        normalized = item.strip()
        if normalized:
            return normalized
    return record.description.strip()


def format_incident(
    record: IncidentRecord,
    *,
    allowed_labels: tuple[str, ...],
    include_reasoning: bool,
) -> TrainingExample:
    if record.split not in {Split.TRAIN, Split.VALIDATION}:
        raise ValueError("locked test incidents may not be formatted for Phase 09 training")
    if record.root_cause_code not in allowed_labels:
        raise ValueError(f"unknown root_cause_code: {record.root_cause_code}")
    prompt = get_prompt_template(BASELINE_PROMPT_VERSION).render(
        title=record.title,
        description=record.description,
        allowed_labels=allowed_labels,
    )
    target: dict[str, str] = {"root_cause_code": record.root_cause_code}
    reasoning = _target_reasoning(record, include_reasoning)
    if reasoning is not None:
        target["reasoning"] = reasoning
    return TrainingExample(
        incident_id=record.incident_id,
        incident_family_id=record.incident_family_id,
        source_split=record.split.value,
        is_synthetic=record.is_synthetic,
        parent_incident_id=record.parent_incident_id,
        root_cause_code=record.root_cause_code,
        messages=(
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _canonical_json(target)},
        ),
    )


def _read_records(path: Path) -> tuple[IncidentRecord, ...]:
    records: list[IncidentRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            records.append(IncidentRecord.model_validate_json(raw_line))
        except Exception as exc:
            raise ValueError(f"invalid incident record at {path}:{line_number}") from exc
    return tuple(records)


def _audit_lineage(records: tuple[IncidentRecord, ...]) -> None:
    by_id = {record.incident_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("incident IDs must be unique before training preparation")
    train_families = {
        record.incident_family_id for record in records if record.split is Split.TRAIN
    }
    heldout_families = {
        record.incident_family_id
        for record in records
        if record.split in {Split.VALIDATION, Split.TEST}
    }
    overlap = train_families & heldout_families
    if overlap:
        raise ValueError(f"held-out family leaked into train split: {sorted(overlap)}")
    for record in records:
        if not record.is_synthetic:
            continue
        if record.split is not Split.TRAIN:
            raise ValueError("synthetic descendants are permitted only in the train split")
        parent = by_id.get(record.parent_incident_id or "")
        if parent is None:
            raise ValueError(
                f"synthetic record {record.incident_id} has an unavailable training parent"
            )
        if parent.split is not Split.TRAIN:
            raise ValueError(
                f"synthetic record {record.incident_id} descends from held-out parent"
            )
        if parent.incident_family_id != record.incident_family_id:
            raise ValueError("synthetic child must retain its parent incident family")


def _write_examples(path: Path, examples: tuple[TrainingExample, ...]) -> str:
    body = "".join(
        _canonical_json(example.model_dump(mode="json")) + "\n"
        for example in sorted(examples, key=lambda item: item.incident_id)
    )
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode()).hexdigest()


def prepare_training_dataset(
    *,
    root: Path,
    output_dir: Path,
    dataset_version: str,
    include_reasoning: bool = True,
) -> PreparedDatasetManifest:
    dataset_dir = root / "datasets/incident_diagnosis/processed" / dataset_version
    source_manifest_path = dataset_dir / "manifest.json"
    incident_path = dataset_dir / "incidents.jsonl"
    source_manifest: dict[str, Any] = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("dataset_version") != dataset_version:
        raise ValueError("dataset directory/version mismatch")
    records = _read_records(incident_path)
    _audit_lineage(records)
    allowed_labels, _categories = load_taxonomy(root)
    examples = tuple(
        format_incident(
            record,
            allowed_labels=allowed_labels,
            include_reasoning=include_reasoning,
        )
        for record in records
        if record.split in {Split.TRAIN, Split.VALIDATION}
    )
    train_examples = tuple(item for item in examples if item.source_split == "train")
    validation_examples = tuple(
        item for item in examples if item.source_split == "validation"
    )
    if not train_examples or not validation_examples:
        raise ValueError("Phase 09 preparation requires non-empty train and validation splits")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_sha = _write_examples(output_dir / "train.jsonl", train_examples)
    validation_sha = _write_examples(output_dir / "validation.jsonl", validation_examples)
    lineage_payload = [
        {
            "incident_id": record.incident_id,
            "incident_family_id": record.incident_family_id,
            "split": record.split.value,
            "is_synthetic": record.is_synthetic,
            "parent_incident_id": record.parent_incident_id,
        }
        for record in sorted(records, key=lambda item: item.incident_id)
    ]
    lineage_sha = hashlib.sha256(_canonical_json(lineage_payload).encode()).hexdigest()
    manifest = PreparedDatasetManifest(
        dataset_version=dataset_version,
        dataset_manifest_checksum=str(source_manifest["manifest_checksum"]),
        dataset_content_checksum=str(source_manifest["content_checksum"]),
        prompt_version=BASELINE_PROMPT_VERSION,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        include_reasoning_target=include_reasoning,
        train_record_count=len(train_examples),
        validation_record_count=len(validation_examples),
        train_family_ids=tuple(
            sorted({item.incident_family_id for item in train_examples})
        ),
        validation_family_ids=tuple(
            sorted({item.incident_family_id for item in validation_examples})
        ),
        train_sha256=train_sha,
        validation_sha256=validation_sha,
        lineage_sha256=lineage_sha,
    )
    (output_dir / "manifest.json").write_text(
        _canonical_json(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    return manifest
