from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.experiment_config import ExperimentConfig, PipelineType
from app.inference.base_model import GenerationConfig, RuntimeConfig
from app.inference.prompts import RAG_PROMPT_VERSION
from app.inference.protocol import (
    BaselineProtocol,
    dependency_lock_checksum,
    load_baseline_protocol,
)
from app.inference.rag_ablations import RAGAblationSuite, RAGVariant, load_ablation_suite

RAG_PROTOCOL_PATH = Path("configs/phase8-rag.json")


class RAGProtocolState(StrEnum):
    VALIDATION = "validation"
    FROZEN = "frozen"


class RAGProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = Field(min_length=1)
    state: RAGProtocolState
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
    ece_bins: int = Field(default=10, gt=0)
    ablation_suite_path: str = Field(min_length=1)
    ablation_suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_policy_version: str = Field(min_length=1)
    selected_variant_id: str | None = None
    supporting_evaluator_versions: dict[str, Any] | None = None
    locked_test_authorized: bool = False

    @model_validator(mode="after")
    def validate_state(self) -> RAGProtocol:
        if self.prompt_version != RAG_PROMPT_VERSION:
            raise ValueError("Phase 08 protocol must use the registered RAG prompt version")
        if self.state is RAGProtocolState.VALIDATION:
            if self.selected_variant_id is not None or self.locked_test_authorized:
                raise ValueError(
                    "validation protocol may not select or authorize the locked-test config"
                )
        else:
            if not self.selected_variant_id:
                raise ValueError("frozen Phase 08 protocol requires selected_variant_id")
            if not self.locked_test_authorized:
                raise ValueError("frozen Phase 08 protocol requires locked-test authorization")
        if self.runtime_config.model_id != self.base_model_id:
            raise ValueError("runtime model_id must match the frozen base model")
        if self.runtime_config.revision != self.base_model_revision:
            raise ValueError("runtime revision must match the frozen base model revision")
        if self.runtime_config.seed != self.seed:
            raise ValueError("runtime seed must match the Phase 08 protocol seed")
        return self

    def assert_split_allowed(self, split: str) -> Literal["validation", "test"]:
        if split == "validation":
            return "validation"
        if split == "test":
            if self.state is not RAGProtocolState.FROZEN or not self.locked_test_authorized:
                raise ValueError("Phase 08 locked test requires a frozen, authorized RAG protocol")
            return "test"
        raise ValueError("Phase 08 RAG runner supports only validation or test splits")

    def assert_variant_allowed(
        self,
        *,
        split: str,
        variant_id: str,
        suite: RAGAblationSuite,
    ) -> RAGVariant:
        evaluation_split = self.assert_split_allowed(split)
        by_id = {variant.variant_id: variant for variant in suite.variants}
        try:
            variant = by_id[variant_id]
        except KeyError as exc:
            raise ValueError(f"unknown Phase 08 RAG variant: {variant_id}") from exc
        if evaluation_split == "test" and variant_id != self.selected_variant_id:
            raise ValueError("locked test may run only the validation-selected frozen RAG variant")
        return variant

    def scientific_payload(self) -> dict[str, object]:
        """Scientific identity shared by validation and the post-selection locked test.

        The selected variant is deliberately excluded. Its complete retrieval identity lives in
        the ablation-suite hash plus each experiment config. Selection is an operational outcome
        of validation and must not retroactively change the scientific identity of those runs.
        """
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
            "ablation_suite_hash": self.ablation_suite_hash,
            "research_policy_version": self.research_policy_version,
            "supporting_evaluator_versions": self.supporting_evaluator_versions,
        }

    def scientific_config_hash(self) -> str:
        canonical = json.dumps(
            self.scientific_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_phase6_scientific_parity(
    rag: RAGProtocol,
    baseline: BaselineProtocol,
) -> None:
    common_fields = {
        "study_id": (rag.study_id, baseline.study_id),
        "dataset_version": (rag.dataset_version, baseline.dataset_version),
        "test_split_manifest_checksum": (
            rag.test_split_manifest_checksum,
            baseline.test_split_manifest_checksum,
        ),
        "label_taxonomy_version": (rag.label_taxonomy_version, baseline.label_taxonomy_version),
        "base_model_id": (rag.base_model_id, baseline.base_model_id),
        "base_model_revision": (rag.base_model_revision, baseline.base_model_revision),
        "runtime_config": (rag.runtime_config, baseline.runtime_config),
        "output_schema_version": (rag.output_schema_version, baseline.output_schema_version),
        "generation_config": (rag.generation_config, baseline.generation_config),
        "confidence_method": (rag.confidence_method, baseline.confidence_method),
        "calibration_version": (rag.calibration_version, baseline.calibration_version),
        "seed": (rag.seed, baseline.seed),
        "evaluator_version": (rag.evaluator_version, baseline.evaluator_version),
        "ece_bins": (rag.ece_bins, baseline.ece_bins),
    }
    mismatches = [name for name, (actual, expected) in common_fields.items() if actual != expected]
    if mismatches:
        raise ValueError(
            "Phase 08 RAG protocol must preserve the frozen Phase 06 scientific identity; "
            f"mismatches={mismatches}"
        )


def _load_and_validate_suite(root: Path, protocol: RAGProtocol) -> RAGAblationSuite:
    path = root / protocol.ablation_suite_path
    suite = load_ablation_suite(path)
    if suite.config_hash() != protocol.ablation_suite_hash:
        raise ValueError("Phase 08 ablation suite hash disagrees with the protocol")
    if any(variant.prompt_version != protocol.prompt_version for variant in suite.variants):
        raise ValueError("Phase 08 ablation variants disagree with the protocol prompt version")
    if protocol.selected_variant_id is not None and protocol.selected_variant_id not in {
        variant.variant_id for variant in suite.variants
    }:
        raise ValueError("selected Phase 08 RAG variant is absent from the ablation suite")
    return suite


def load_rag_protocol(
    root: Path,
    path: Path = RAG_PROTOCOL_PATH,
) -> tuple[RAGProtocol, RAGAblationSuite]:
    payload = json.loads((root / path).read_text(encoding="utf-8"))
    protocol = RAGProtocol.model_validate(payload)
    baseline = load_baseline_protocol(root)
    _assert_phase6_scientific_parity(protocol, baseline)
    suite = _load_and_validate_suite(root, protocol)
    return protocol, suite


def build_rag_experiment_config(
    *,
    root: Path,
    protocol: RAGProtocol,
    variant: RAGVariant,
    git_commit: str,
    hardware_runtime_descriptor: str,
    cost_rate_snapshot_version: str,
) -> ExperimentConfig:
    if variant.prompt_version != protocol.prompt_version:
        raise ValueError("RAG variant prompt version disagrees with the Phase 08 protocol")
    return ExperimentConfig(
        study_id=protocol.study_id,
        pipeline_type=PipelineType.RAG,
        dataset_version=protocol.dataset_version,
        test_split_manifest_checksum=protocol.test_split_manifest_checksum,
        label_taxonomy_version=protocol.label_taxonomy_version,
        base_model_id=protocol.base_model_id,
        base_model_revision=protocol.base_model_revision,
        knowledge_base_version=variant.knowledge_base_version,
        embedding_model_revision=variant.embedding_model_revision,
        chunker_version=variant.chunker_version,
        chunk_size=variant.chunk_size,
        overlap=variant.overlap,
        top_k=variant.top_k,
        reranker_id=variant.reranker_id,
        reranker_revision=variant.reranker_revision,
        retrieval_policy_version=protocol.research_policy_version,
        context_policy_version=variant.context_policy_version,
        max_context_tokens=variant.max_context_tokens,
        max_chunk_tokens=variant.max_chunk_tokens,
        prompt_version=protocol.prompt_version,
        output_schema_version=protocol.output_schema_version,
        generation_config=protocol.generation_config.model_dump(mode="json"),
        temperature=protocol.generation_config.temperature,
        confidence_method=protocol.confidence_method,
        calibration_version=protocol.calibration_version,
        seed=protocol.seed,
        evaluator_version=protocol.evaluator_version,
        ragas_or_judge_versions=protocol.supporting_evaluator_versions,
        git_commit=git_commit,
        dependency_lock_checksum=dependency_lock_checksum(root),
        hardware_runtime_descriptor=hardware_runtime_descriptor,
        cost_rate_snapshot_version=cost_rate_snapshot_version,
    )
