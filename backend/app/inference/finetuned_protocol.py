from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.experiment_config import ExperimentConfig, PipelineType
from app.inference.base_model import GenerationConfig, RuntimeConfig
from app.inference.protocol import dependency_lock_checksum, load_baseline_protocol
from app.training.evidence import load_training_evidence_identity, sha256_file

PHASE10_PROTOCOL_PATH = Path("configs/phase10-finetuned.json")
PHASE9_TRAINING_EVIDENCE_PATH = Path("evidence/phase-09/training-run.json")


class Phase10State(StrEnum):
    VALIDATION = "validation"
    FROZEN = "frozen"


class DataEfficiencyProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fractions: tuple[float, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    subset_policy: Literal["nested_seeded_family_prefix_v1"]
    rounding: Literal["ceil"]
    selection_scope: Literal["train_families_only"]
    primary_adapter_selection_use: Literal[False]

    @model_validator(mode="after")
    def validate_matrix(self) -> DataEfficiencyProtocol:
        if tuple(sorted(set(self.fractions))) != self.fractions:
            raise ValueError("data-efficiency fractions must be unique and increasing")
        if any(fraction <= 0.0 or fraction > 1.0 for fraction in self.fractions):
            raise ValueError("data-efficiency fractions must be within (0, 1]")
        if 1.0 not in self.fractions:
            raise ValueError("data-efficiency matrix must include the 100% condition")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("data-efficiency seeds must be unique")
        return self


class Phase10Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["phase10-finetuned-evaluation-v1"]
    state: Phase10State
    study_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    test_split_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_taxonomy_version: str = Field(min_length=1)
    base_model_id: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    runtime_config: RuntimeConfig
    prompt_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    generation_config: GenerationConfig
    confidence_method: str = Field(min_length=1)
    calibration_version: str | None = None
    seed: int
    evaluator_version: str = Field(min_length=1)
    ece_bins: int = Field(gt=0)
    source_training_run_id: str = Field(min_length=1)
    source_training_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_training_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_adapter_artifact_reference: str = Field(min_length=1)
    candidate_selection_rule: Literal["phase09_validation_eval_loss_selected_checkpoint"]
    data_efficiency: DataEfficiencyProtocol
    locked_test_authorized: bool = False

    @model_validator(mode="after")
    def validate_state(self) -> Phase10Protocol:
        if self.locked_test_authorized and self.state is not Phase10State.FROZEN:
            raise ValueError("Phase 10 locked test authorization requires a frozen protocol")
        if self.runtime_config.model_id != self.base_model_id:
            raise ValueError("runtime model ID must match the frozen base model")
        if self.runtime_config.revision != self.base_model_revision:
            raise ValueError("runtime model revision must match the frozen base model")
        if self.runtime_config.seed != self.seed:
            raise ValueError("runtime seed must match the Phase 10 protocol seed")
        return self

    def assert_split_allowed(self, split: str) -> Literal["validation", "test"]:
        if split == "validation":
            return "validation"
        if split == "test":
            if self.state is not Phase10State.FROZEN or not self.locked_test_authorized:
                raise ValueError(
                    "Phase 10 locked test is unavailable until the primary adapter is "
                    "validation-selected, frozen, and explicitly authorized"
                )
            return "test"
        raise ValueError("Phase 10 fine-tuned runner supports only validation or test splits")

    def scientific_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "study_id": self.study_id,
            "dataset_version": self.dataset_version,
            "test_split_manifest_checksum": self.test_split_manifest_checksum,
            "label_taxonomy_version": self.label_taxonomy_version,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "runtime_config": self.runtime_config.model_dump(mode="json"),
            "prompt_version": self.prompt_version,
            "output_schema_version": self.output_schema_version,
            "generation_config": self.generation_config.model_dump(mode="json"),
            "confidence_method": self.confidence_method,
            "calibration_version": self.calibration_version,
            "seed": self.seed,
            "evaluator_version": self.evaluator_version,
            "ece_bins": self.ece_bins,
            "source_training_run_id": self.source_training_run_id,
            "source_training_evidence_sha256": self.source_training_evidence_sha256,
            "source_training_config_hash": self.source_training_config_hash,
            "candidate_adapter_sha256": self.candidate_adapter_sha256,
            "candidate_adapter_artifact_reference": self.candidate_adapter_artifact_reference,
            "candidate_selection_rule": self.candidate_selection_rule,
            "data_efficiency": self.data_efficiency.model_dump(mode="json"),
        }

    def scientific_config_hash(self) -> str:
        encoded = json.dumps(
            self.scientific_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_phase10_protocol(
    root: Path, path: Path = PHASE10_PROTOCOL_PATH
) -> Phase10Protocol:
    payload = json.loads((root / path).read_text(encoding="utf-8"))
    protocol = Phase10Protocol.model_validate(payload)

    baseline = load_baseline_protocol(root)
    parity_fields = (
        "study_id",
        "dataset_version",
        "test_split_manifest_checksum",
        "label_taxonomy_version",
        "base_model_id",
        "base_model_revision",
        "prompt_version",
        "output_schema_version",
        "confidence_method",
        "calibration_version",
        "seed",
        "evaluator_version",
        "ece_bins",
    )
    for field in parity_fields:
        if getattr(protocol, field) != getattr(baseline, field):
            raise ValueError(f"Phase 10 changed frozen baseline field: {field}")
    if protocol.runtime_config != baseline.runtime_config:
        raise ValueError("Phase 10 changed the frozen baseline runtime configuration")
    if protocol.generation_config != baseline.generation_config:
        raise ValueError("Phase 10 changed the frozen baseline generation configuration")

    evidence_path = root / PHASE9_TRAINING_EVIDENCE_PATH
    identity = load_training_evidence_identity(evidence_path)
    if sha256_file(evidence_path) != protocol.source_training_evidence_sha256:
        raise ValueError("Phase 10 source training evidence checksum is stale")
    if identity.run_id != protocol.source_training_run_id:
        raise ValueError("Phase 10 source training run ID disagrees with Phase 09 evidence")
    if identity.training_config_hash != protocol.source_training_config_hash:
        raise ValueError("Phase 10 source training config hash disagrees with Phase 09")
    if identity.adapter_sha256 != protocol.candidate_adapter_sha256:
        raise ValueError("Phase 10 candidate adapter hash disagrees with Phase 09")
    if identity.wandb_artifact_reference != protocol.candidate_adapter_artifact_reference:
        raise ValueError("Phase 10 candidate adapter artifact disagrees with Phase 09")
    if identity.dataset_version != protocol.dataset_version:
        raise ValueError("Phase 10 dataset version disagrees with Phase 09")
    if identity.dataset_manifest_checksum != protocol.test_split_manifest_checksum:
        raise ValueError("Phase 10 dataset manifest checksum disagrees with Phase 09")
    if identity.base_model_id != protocol.base_model_id:
        raise ValueError("Phase 10 base model ID disagrees with Phase 09")
    if identity.base_model_revision != protocol.base_model_revision:
        raise ValueError("Phase 10 base model revision disagrees with Phase 09")
    return protocol


def build_finetuned_experiment_config(
    *,
    root: Path,
    protocol: Phase10Protocol,
    git_commit: str,
    hardware_runtime_descriptor: str,
    cost_rate_snapshot_version: str,
) -> ExperimentConfig:
    return ExperimentConfig(
        study_id=protocol.study_id,
        pipeline_type=PipelineType.FINETUNED,
        dataset_version=protocol.dataset_version,
        test_split_manifest_checksum=protocol.test_split_manifest_checksum,
        label_taxonomy_version=protocol.label_taxonomy_version,
        base_model_id=protocol.base_model_id,
        base_model_revision=protocol.base_model_revision,
        adapter_id=protocol.candidate_adapter_artifact_reference,
        adapter_revision=protocol.candidate_adapter_sha256,
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
