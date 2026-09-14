from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from app.db import build_engine, session_scope
from app.inference.base_model import BaseModelBackend, TransformersBackend
from app.inference.costing import HourlyRateCostBackend
from app.inference.protocol import load_taxonomy
from app.inference.rag_ablations import RAGVariant
from app.inference.rag_pipeline import ContextBudget, PgVectorRetriever, RAGPipeline
from app.inference.rag_protocol import RAGProtocol, load_rag_protocol
from app.inference.rag_runner import RAGRunSummary, run_rag_experiment
from app.inference.tracking import ExperimentTracker, build_tracker_from_env
from app.models import Run
from app.retrieval.embeddings import HashEmbeddingAdapter
from app.retrieval.indexing import load_kb_config
from app.retrieval.phase8_variants import build_phase8_kb_variant_config
from app.retrieval.reranker import NoOpReranker, TokenOverlapReranker
from app.services.dataset_import import import_phase3_dataset

BackendFactory = Callable[[RAGProtocol], BaseModelBackend]


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Phase 08 RAG job requires non-empty {key}")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string when provided")
    return value.strip()


def _optional_non_negative_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric when provided") from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{key} must be finite and non-negative")
    return converted


def _select_variant(protocol: RAGProtocol, payload: dict[str, Any]) -> RAGVariant:
    _loaded_protocol, suite = load_rag_protocol_from_protocol(protocol)
    variant_id = _required_text(payload, "variant_id")
    for variant in suite.variants:
        if variant.variant_id == variant_id:
            return variant
    raise ValueError(f"unknown Phase 08 RAG variant: {variant_id}")


def load_rag_protocol_from_protocol(protocol: RAGProtocol):  # type: ignore[no-untyped-def]
    root = Path(protocol.ablation_suite_path)
    del root
    raise AssertionError("internal helper must be replaced by handler-bound protocol loading")


def _build_reranker(variant: RAGVariant):  # type: ignore[no-untyped-def]
    if variant.reranker_id is None:
        return NoOpReranker()
    if (
        variant.reranker_id == TokenOverlapReranker.reranker_id
        and variant.reranker_revision == TokenOverlapReranker.revision
    ):
        return TokenOverlapReranker()
    raise ValueError(
        "unsupported Phase 08 reranker identity; add an explicit reproducible reranker adapter"
    )


def _summary_payload(summary: RAGRunSummary) -> dict[str, object]:
    payload = asdict(summary)
    return {str(key): value for key, value in payload.items()}


class Phase8RAGJobHandler:
    """Worker/CLI orchestration around the reusable Phase 08 RAG runner."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        engine: Engine | None = None,
        backend_factory: BackendFactory | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.root = (root or Path(os.environ.get("EVALFORGE_ROOT", "."))).resolve()
        self.engine = engine or build_engine()
        self.backend_factory = backend_factory
        self.tracker = tracker or build_tracker_from_env()

    def __call__(self, payload: dict[str, Any]) -> dict[str, object]:
        protocol, suite = load_rag_protocol(self.root)
        variant_id = _required_text(payload, "variant_id")
        variant_by_id = {variant.variant_id: variant for variant in suite.variants}
        try:
            variant = variant_by_id[variant_id]
        except KeyError as exc:
            raise ValueError(f"unknown Phase 08 RAG variant: {variant_id}") from exc

        split = _required_text(payload, "split")
        protocol.assert_split_allowed(split)
        git_commit = _required_text(payload, "git_commit")
        hardware_runtime_descriptor = _required_text(payload, "hardware_runtime_descriptor")
        cost_rate_snapshot_version = _required_text(payload, "cost_rate_snapshot_version")
        max_attempts = int(payload.get("max_attempts", 2))
        upfront_cost_usd = float(payload.get("upfront_cost_usd", 0.0))
        gpu_hour_usd = _optional_non_negative_float(payload, "gpu_hour_usd")

        if self.backend_factory is None:
            if gpu_hour_usd is None:
                raise ValueError("real Phase 08 RAG jobs require gpu_hour_usd")
            backend: BaseModelBackend = HourlyRateCostBackend(
                TransformersBackend(protocol.runtime_config),
                gpu_hour_usd=gpu_hour_usd,
            )
        else:
            backend = self.backend_factory(protocol)

        labels, _categories = load_taxonomy(self.root)
        baseline_kb = load_kb_config(self.root / "configs/phase7-kb.json")
        variant_kb = build_phase8_kb_variant_config(
            baseline_kb,
            kb_version=variant.knowledge_base_version,
            embedding_model_id=variant.embedding_model_id,
            embedding_model_revision=variant.embedding_model_revision,
            chunker_version=variant.chunker_version,
            chunk_size=variant.chunk_size,
            overlap=variant.overlap,
        )
        embedding_adapter = HashEmbeddingAdapter(variant_kb.embedding)

        with session_scope(self.engine) as session:
            import_phase3_dataset(session, self.root, protocol.dataset_version)
            retriever = PgVectorRetriever(
                session=session,
                adapter=embedding_adapter,
                kb_version=variant.knowledge_base_version,
            )
            pipeline = RAGPipeline(
                backend=backend,
                retriever=retriever,
                allowed_labels=labels,
                top_k=variant.top_k,
                reranker=_build_reranker(variant),
                prompt_version=protocol.prompt_version,
                generation_config=protocol.generation_config,
                context_budget=ContextBudget(
                    max_context_tokens=variant.max_context_tokens,
                    max_chunk_tokens=variant.max_chunk_tokens,
                    policy_version=variant.context_policy_version,
                ),
            )
            summary = run_rag_experiment(
                session,
                root=self.root,
                protocol=protocol,
                variant=variant,
                pipeline=pipeline,
                split=split,
                git_commit=git_commit,
                hardware_runtime_descriptor=hardware_runtime_descriptor,
                cost_rate_snapshot_version=cost_rate_snapshot_version,
                experiment_id=_optional_text(payload, "experiment_id"),
                run_id=_optional_text(payload, "run_id"),
                max_attempts=max_attempts,
                upfront_cost_usd=upfront_cost_usd,
            )

        summary_payload = _summary_payload(summary)
        tracking_payload = {
            **protocol.scientific_payload(),
            "phase8_variant": variant.model_dump(mode="json"),
        }
        tracking = self.tracker.log_rag(
            run_id=summary.run_id,
            protocol_payload=tracking_payload,
            summary=summary_payload,
        )
        with session_scope(self.engine) as session:
            run = session.get(Run, summary.run_id)
            if run is None:
                raise RuntimeError("completed Phase 08 RAG run disappeared before tracking update")
            run.runtime_metadata = {
                **run.runtime_metadata,
                "experiment_tracking": tracking.as_dict(),
            }

        return {**summary_payload, "tracking": tracking.as_dict()}
