from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AblationFactor = Literal["default", "top_k", "chunking", "embedding", "reranker"]

_ALLOWED_CHANGES: dict[str, frozenset[str]] = {
    "top_k": frozenset({"top_k"}),
    "chunking": frozenset(
        {
            "knowledge_base_version",
            "chunker_version",
            "chunk_size",
            "overlap",
        }
    ),
    "embedding": frozenset(
        {
            "knowledge_base_version",
            "embedding_model_id",
            "embedding_model_revision",
        }
    ),
    "reranker": frozenset({"reranker_id", "reranker_revision"}),
}


class RAGVariant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(min_length=1)
    factor: AblationFactor
    knowledge_base_version: str = Field(min_length=1)
    embedding_model_id: str = Field(min_length=1)
    embedding_model_revision: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    reranker_id: str | None = None
    reranker_revision: str | None = None
    prompt_version: str = Field(min_length=1)
    context_policy_version: str = Field(min_length=1)
    max_context_tokens: int = Field(gt=0)
    max_chunk_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_variant(self) -> RAGVariant:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.max_chunk_tokens > self.max_context_tokens:
            raise ValueError("max_chunk_tokens may not exceed max_context_tokens")
        if (self.reranker_id is None) != (self.reranker_revision is None):
            raise ValueError("reranker id and revision must be present together")
        return self


class RAGAblationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_version: str = Field(min_length=1)
    selection_split: Literal["validation"]
    locked_test_selection_forbidden: bool
    baseline_variant_id: str = Field(min_length=1)
    variants: tuple[RAGVariant, ...]

    @model_validator(mode="after")
    def validate_controlled_design(self) -> RAGAblationSuite:
        if not self.locked_test_selection_forbidden:
            raise ValueError("locked test selection must be forbidden")
        if not self.variants:
            raise ValueError("at least one RAG variant is required")
        by_id = {variant.variant_id: variant for variant in self.variants}
        if len(by_id) != len(self.variants):
            raise ValueError("variant IDs must be unique")
        try:
            baseline = by_id[self.baseline_variant_id]
        except KeyError as exc:
            raise ValueError("baseline_variant_id is not present") from exc
        if baseline.factor != "default":
            raise ValueError("baseline variant must use factor='default'")

        baseline_payload = baseline.model_dump(mode="python")
        for variant in self.variants:
            if variant.variant_id == baseline.variant_id:
                continue
            if variant.factor == "default":
                raise ValueError("only the baseline may use factor='default'")
            payload = variant.model_dump(mode="python")
            changed = {
                key
                for key, value in payload.items()
                if key not in {"variant_id", "factor"} and value != baseline_payload[key]
            }
            allowed = _ALLOWED_CHANGES[variant.factor]
            if not changed:
                raise ValueError(f"variant {variant.variant_id} does not change its factor")
            if not changed <= allowed:
                raise ValueError(
                    f"variant {variant.variant_id} changes uncontrolled fields: "
                    f"{sorted(changed - allowed)}"
                )
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_ablation_suite(path: Path) -> RAGAblationSuite:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RAG ablation config must be a JSON object")
    return RAGAblationSuite.model_validate(payload)
