from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.experiment_config import ExperimentConfig, PipelineType
from app.inference.base_model import GenerationConfig

BASELINE_PROTOCOL_PATH = Path("configs/phase6-baseline.json")


class ProtocolState(StrEnum):
    VALIDATION = "validation"
    FROZEN = "frozen"


class BaselineProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(min_length=1)
    state: ProtocolState
    study_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    test_split_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_taxonomy_version: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    generation_config: GenerationConfig
    confidence_method: str = Field(min_length=1)
    calibration_version: str | None = None
    seed: int
    evaluator_version: str = Field(min_length=1)
    ece_bins: int = Field(default=10, gt=0)
    locked_test_authorized: bool = False

    @model_validator(mode="after")
    def validate_freeze_state(self) -> BaselineProtocol:
        if self.locked_test_authorized and self.state is not ProtocolState.FROZEN:
            raise ValueError("locked test authorization requires a frozen protocol")
        return self

    def assert_split_allowed(self, split: str) -> Literal["validation", "test"]:
        if split == "validation":
            return "validation"
        if split == "test":
            if self.state is not ProtocolState.FROZEN or not self.locked_test_authorized:
                raise ValueError(
                    "locked test is unavailable until the Phase 06 protocol is frozen "
                    "and authorized"
                )
            return "test"
        raise ValueError("Phase 06 baseline runner supports only validation or test splits")

    def scientific_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "study_id": self.study_id,
            "dataset_version": self.dataset_version,
            "test_split_manifest_checksum": self.test_split_manifest_checksum,
            "label_taxonomy_version": self.label_taxonomy_version,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "prompt_version": self.prompt_version,
            "output_schema_version": self.output_schema_version,
            "generation_config": self.generation_config.model_dump(mode="json"),
            "confidence_method": self.confidence_method,
            "calibration_version": self.calibration_version,
            "seed": self.seed,
            "evaluator_version": self.evaluator_version,
            "ece_bins": self.ece_bins,
        }

    def scientific_config_hash(self) -> str:
        canonical = json.dumps(
            self.scientific_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_baseline_protocol(root: Path, path: Path = BASELINE_PROTOCOL_PATH) -> BaselineProtocol:
    payload = json.loads((root / path).read_text(encoding="utf-8"))
    return BaselineProtocol.model_validate(payload)


def dependency_lock_checksum(root: Path) -> str:
    candidates = (root / "backend/requirements.full.lock", root / "requirements.full.lock")
    for candidate in candidates:
        if candidate.exists():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    raise FileNotFoundError("backend dependency lock is unavailable")


def build_experiment_config(
    *,
    root: Path,
    protocol: BaselineProtocol,
    git_commit: str,
    hardware_runtime_descriptor: str,
    cost_rate_snapshot_version: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        study_id=protocol.study_id,
        pipeline_type=PipelineType.ZERO_SHOT,
        dataset_version=protocol.dataset_version,
        test_split_manifest_checksum=protocol.test_split_manifest_checksum,
        label_taxonomy_version=protocol.label_taxonomy_version,
        base_model_id=protocol.base_model_id,
        base_model_revision=protocol.base_model_revision,
        prompt_version=protocol.prompt_version,
        output_schema_version=protocol.output_schema_version,
        generation_config=protocol.generation_config.model_dump(mode="json"),
        temperature=protocol.generation_config.temperature,
        confidence_method=protocol.confidence_method,
        calibration_version=protocol.calibration_version,
        seed=protocol.seed,
        evaluator_version=protocol.evaluator_version,
        git_commit=git_commit,
        dependency_lock_checksum=dependency_lock_checksum(root),
        hardware_runtime_descriptor=hardware_runtime_descriptor,
        cost_rate_snapshot_version=cost_rate_snapshot_version,
    )


def load_taxonomy(root: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    payload: dict[str, Any] = json.loads(
        (root / "configs/label-taxonomy.yaml").read_text(encoding="utf-8")
    )
    labels = tuple(str(item["id"]) for item in payload["labels"])
    categories = {str(item["id"]): str(item["category"]) for item in payload["labels"]}
    return labels, categories
