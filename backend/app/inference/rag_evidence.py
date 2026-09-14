from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.contracts import ParseStatus, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import reload_evaluation, score_stored_run
from app.inference.evidence import (
    canonical_json_sha256,
    expected_split_incident_ids,
    seal_run_evidence,
)
from app.inference.protocol import load_taxonomy
from app.inference.rag_ablations import RAGAblationSuite, RAGVariant, load_ablation_suite
from app.inference.rag_protocol import RAGProtocol
from app.inference.rag_runner import RELEVANCE_REFERENCE_VERSION, _runbook_relevance_references
from app.models import Experiment, Incident, KnowledgeBaseVersion, Run
from app.models import Prediction as PredictionRow
from app.results import load_run_evidence

RAG_RUN_EVIDENCE_VERSION = "phase8-rag-run-evidence-v1"
_REQUIRED_METRICS = (
    "primary.exact_accuracy",
    "supporting.hierarchical_accuracy",
    "supporting.top3_accuracy",
    "quality.parse_failure_rate",
    "calibration.normalized_multiclass_brier",
    "latency.p50_ms",
    "latency.p95_ms",
    "cost.marginal_mean_usd",
    "cost.amortized_mean_usd",
    "retrieval.context_precision",
    "retrieval.context_recall",
)

_ECE_METRIC = "calibration.ece"


def _tracking_payload(run: Run) -> dict[str, object]:
    value = run.runtime_metadata.get("experiment_tracking")
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {
        "provider": "wandb",
        "configured": False,
        "run_reference": None,
        "artifact_reference": None,
    }


def _load_suite_for_evidence(root: Path, protocol: RAGProtocol) -> RAGAblationSuite:
    suite = load_ablation_suite(root / protocol.ablation_suite_path)
    if suite.config_hash() != protocol.ablation_suite_hash:
        raise ValueError("Phase 08 evidence ablation suite hash disagrees with protocol")
    return suite


def _prediction_incident_ids(predictions: object) -> tuple[str, ...]:
    if not isinstance(predictions, list):
        raise ValueError("RAG run evidence predictions must be a list")
    incident_ids: list[str] = []
    for raw in predictions:
        if not isinstance(raw, dict):
            raise ValueError("RAG run evidence predictions must contain objects")
        incident_ids.append(Prediction.model_validate(raw).incident_id)
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("RAG run evidence contains duplicate prediction incident IDs")
    return tuple(incident_ids)


def _validate_ece_contract(
    *,
    metrics: Mapping[str, object],
    predictions: list[dict[str, object]],
) -> None:
    """Allow canonical ECE omission only for the evaluator's parse-failure edge case.

    Phase 05's common evaluator intentionally omits ECE when any prediction lacks a
    confidence value. A non-OK parse result is a first-class prediction outcome and
    can legitimately have no generated-label confidence even though its full label
    probability distribution remains available for Brier scoring.

    Phase 08 selection treats such an undefined ECE conservatively as worse than any
    defined ECE at the already-declared ECE tie-break position. This validator keeps
    the exception narrow so unrelated metric loss cannot silently pass.
    """

    if _ECE_METRIC in metrics:
        value = metrics[_ECE_METRIC]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("Phase 08 RAG ECE metric must be numeric when present")
        return

    parse_failure_rate = metrics.get("quality.parse_failure_rate")
    if (
        isinstance(parse_failure_rate, bool)
        or not isinstance(parse_failure_rate, int | float)
        or float(parse_failure_rate) <= 0.0
    ):
        raise ValueError(
            "Phase 08 RAG evidence may omit calibration.ece only when parse failures occurred"
        )

    parsed = [Prediction.model_validate(item) for item in predictions]
    missing_confidence_parse_failures = [
        prediction
        for prediction in parsed
        if prediction.parse_status is not ParseStatus.OK
        and prediction.confidence_probability is None
    ]
    if not missing_confidence_parse_failures:
        raise ValueError(
            "Phase 08 RAG evidence omitted calibration.ece without a parse-failure "
            "prediction lacking confidence"
        )


def _trace_rank(trace: Mapping[str, object]) -> int:
    value = trace.get("rank")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("RAG retrieval trace rank must be an integer")
    return value


def _trace_included_in_prompt(trace: Mapping[str, object]) -> bool:
    metadata = trace.get("metadata")
    return isinstance(metadata, dict) and metadata.get("included_in_prompt") is True


def _validate_trace_contract(
    *,
    predictions: list[dict[str, object]],
    traces: list[dict[str, object]],
    prediction_id_by_incident: Mapping[str, str],
    family_id_by_incident: Mapping[str, str],
) -> None:
    if set(prediction_id_by_incident) != set(family_id_by_incident):
        raise ValueError("RAG evidence trace identities do not cover the same incident set")
    incident_by_prediction_id = {
        prediction_id: incident_id
        for incident_id, prediction_id in prediction_id_by_incident.items()
    }
    if len(incident_by_prediction_id) != len(prediction_id_by_incident):
        raise ValueError("RAG evidence reuses one prediction ID across incidents")
    traces_by_incident: dict[str, list[dict[str, object]]] = {
        incident_id: [] for incident_id in prediction_id_by_incident
    }

    for trace in traces:
        prediction_id = trace.get("prediction_id")
        incident_id = trace.get("incident_id")
        if not isinstance(prediction_id, str) or prediction_id not in incident_by_prediction_id:
            raise ValueError("RAG retrieval trace references an unknown prediction")
        expected_incident_id = incident_by_prediction_id[prediction_id]
        if incident_id != expected_incident_id:
            raise ValueError("RAG retrieval trace incident identity disagrees with prediction")
        traces_by_incident[expected_incident_id].append(trace)

        metadata = trace.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("RAG retrieval trace metadata must be an object")
        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("RAG retrieval trace is missing provenance")
        if provenance.get("research_eligible") is not True:
            raise ValueError("RAG retrieval trace is not research eligible")
        if provenance.get("source_split") in {"validation", "test"}:
            raise ValueError("RAG retrieval trace leaks validation or test source material")
        if provenance.get("source_family_id") == family_id_by_incident[expected_incident_id]:
            raise ValueError("RAG retrieval trace leaks the query incident family")

    prediction_by_incident: dict[str, Prediction] = {}
    for raw in predictions:
        prediction = Prediction.model_validate(raw)
        prediction_by_incident[prediction.incident_id] = prediction
    if set(prediction_by_incident) != set(prediction_id_by_incident):
        raise ValueError("RAG evidence predictions and persisted IDs cover different incidents")

    for incident_id, prediction in prediction_by_incident.items():
        incident_traces = sorted(
            traces_by_incident[incident_id],
            key=_trace_rank,
        )
        ranks = tuple(_trace_rank(item) for item in incident_traces)
        if ranks != tuple(range(1, len(incident_traces) + 1)):
            raise ValueError("RAG retrieval trace ranks must be contiguous and start at one")
        chunk_ids = tuple(str(item["chunk_id"]) for item in incident_traces)
        if chunk_ids != prediction.retrieved_chunk_ids:
            raise ValueError("RAG retrieval traces disagree with prediction retrieved_chunk_ids")
        included = tuple(
            str(item["chunk_id"]) for item in incident_traces if _trace_included_in_prompt(item)
        )
        if included != prediction.evidence_citations:
            raise ValueError("RAG prompt citations disagree with persisted retrieval traces")


def _family_ids_from_dataset(
    root: Path,
    *,
    dataset_version: str,
    incident_ids: tuple[str, ...],
) -> dict[str, str]:
    path = (
        root / "datasets" / "incident_diagnosis" / "processed" / dataset_version / "incidents.jsonl"
    )
    family_ids: dict[str, str] = {}
    wanted = set(incident_ids)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        incident_id = row.get("incident_id")
        family_id = row.get("incident_family_id")
        if incident_id in wanted and isinstance(incident_id, str) and isinstance(family_id, str):
            family_ids[incident_id] = family_id
    if set(family_ids) != wanted:
        raise ValueError("Phase 08 RAG evidence cannot resolve incident family identities")
    return family_ids


def export_phase8_rag_run_evidence(
    session: Session,
    *,
    root: Path,
    run_id: str,
    protocol: RAGProtocol,
    variant: RAGVariant,
) -> dict[str, Any]:
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"unknown Phase 08 RAG run: {run_id}")
    experiment = session.get(Experiment, run.experiment_id)
    if experiment is None:
        raise RuntimeError("Phase 08 RAG run references a missing experiment")
    if experiment.pipeline_type != "RAG":
        raise ValueError("Phase 08 RAG evidence requires a RAG experiment")
    if experiment.status != "completed" or run.status != "completed":
        raise ValueError("Phase 08 RAG evidence requires a completed experiment and run")

    readback = load_run_evidence(session, run_id=run_id)
    if readback is None:
        raise RuntimeError("completed Phase 08 RAG run has no readback evidence")
    raw_predictions = readback.get("predictions")
    raw_traces = readback.get("retrieval_traces")
    if not isinstance(raw_predictions, list) or not all(
        isinstance(item, dict) for item in raw_predictions
    ):
        raise RuntimeError("Phase 08 RAG readback predictions are malformed")
    if not isinstance(raw_traces, list) or not all(isinstance(item, dict) for item in raw_traces):
        raise RuntimeError("Phase 08 RAG readback retrieval traces are malformed")
    predictions = [dict(item) for item in raw_predictions]
    traces = [dict(item) for item in raw_traces]
    incident_ids = _prediction_incident_ids(predictions)

    incidents = session.scalars(
        select(Incident)
        .where(
            Incident.dataset_version == protocol.dataset_version,
            Incident.incident_id.in_(incident_ids),
        )
        .order_by(Incident.incident_id)
    ).all()
    incident_by_id = {incident.incident_id: incident for incident in incidents}
    if set(incident_by_id) != set(incident_ids):
        raise RuntimeError("Phase 08 RAG predictions reference missing dataset incidents")
    splits = {incident_by_id[incident_id].split for incident_id in incident_ids}
    if len(splits) != 1:
        raise RuntimeError("Phase 08 RAG run mixes dataset splits")
    split = next(iter(splits))
    if split not in {"validation", "test"}:
        raise RuntimeError(f"Phase 08 RAG evidence has unsupported split: {split}")

    suite = _load_suite_for_evidence(root, protocol)
    registered_variant = protocol.assert_variant_allowed(
        split=split,
        variant_id=variant.variant_id,
        suite=suite,
    )
    if registered_variant != variant:
        raise RuntimeError("Phase 08 RAG evidence variant is not the registered suite variant")

    expected_ids = expected_split_incident_ids(
        root,
        dataset_version=protocol.dataset_version,
        split=split,
    )
    if tuple(sorted(incident_ids)) != expected_ids:
        raise RuntimeError("Phase 08 RAG run does not exactly cover the expected dataset split")

    prediction_rows = session.scalars(
        select(PredictionRow)
        .where(PredictionRow.run_id == run_id)
        .order_by(PredictionRow.incident_id)
    ).all()
    prediction_id_by_incident = {row.incident_id: row.prediction_id for row in prediction_rows}
    if set(prediction_id_by_incident) != set(incident_ids):
        raise RuntimeError("Phase 08 RAG persisted prediction rows are incomplete")
    family_id_by_incident = {
        incident_id: incident_by_id[incident_id].family_id for incident_id in incident_ids
    }
    _validate_trace_contract(
        predictions=predictions,
        traces=traces,
        prediction_id_by_incident=prediction_id_by_incident,
        family_id_by_incident=family_id_by_incident,
    )

    scientific_hash = protocol.scientific_config_hash()
    if run.runtime_metadata.get("phase8_scientific_config_hash") != scientific_hash:
        raise RuntimeError("Phase 08 RAG run scientific identity disagrees with current protocol")
    if run.runtime_metadata.get("phase8_variant_id") != variant.variant_id:
        raise RuntimeError("Phase 08 RAG run variant identity disagrees with evidence request")
    if run.runtime_metadata.get("phase8_ablation_suite_hash") != protocol.ablation_suite_hash:
        raise RuntimeError("Phase 08 RAG run ablation suite identity disagrees with protocol")

    kb = session.get(KnowledgeBaseVersion, variant.knowledge_base_version)
    if kb is None:
        raise RuntimeError("Phase 08 RAG run references an unavailable knowledge-base version")
    if experiment.kb_version != variant.knowledge_base_version:
        raise RuntimeError("Phase 08 RAG experiment KB identity disagrees with selected variant")

    labels, categories = load_taxonomy(root)
    harness = EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )
    relevance_references = _runbook_relevance_references(
        session,
        kb_version=variant.knowledge_base_version,
        incidents=incidents,
    )
    upfront_cost_usd = float(run.runtime_metadata.get("upfront_cost_usd", 0.0))
    snapshot = reload_evaluation(session, run_id=run_id)
    rescored = score_stored_run(
        session,
        run_id=run_id,
        harness=harness,
        relevant_chunk_ids_by_incident=relevance_references,
        upfront_cost_usd=upfront_cost_usd,
    )
    if snapshot.result_hash != rescored.result_hash:
        raise RuntimeError("Phase 08 RAG result hash changed during evidence recomputation")
    if snapshot.metric_values != rescored.metric_map():
        raise RuntimeError("Phase 08 RAG metrics changed during evidence recomputation")

    payload: dict[str, Any] = {
        "evidence_version": RAG_RUN_EVIDENCE_VERSION,
        "split": split,
        "protocol_version": protocol.protocol_version,
        "protocol_state_at_export": protocol.state.value,
        "locked_test_authorized_at_export": protocol.locked_test_authorized,
        "scientific_config_hash": scientific_hash,
        "ablation_suite_hash": protocol.ablation_suite_hash,
        "variant_id": variant.variant_id,
        "variant": variant.model_dump(mode="json"),
        "experiment_id": experiment.experiment_id,
        "experiment_config_hash": experiment.config_hash,
        "run_id": run.run_id,
        "experiment_status": experiment.status,
        "run_status": run.status,
        "dataset_version": experiment.dataset_version,
        "test_split_manifest_checksum": experiment.test_split_manifest_checksum,
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "prompt_version": experiment.prompt_version,
        "output_schema_version": experiment.output_schema_version,
        "generation_config": dict(experiment.generation_config),
        "evaluator_version": experiment.evaluator_version,
        "relevance_reference_version": RELEVANCE_REFERENCE_VERSION,
        "knowledge_base": {
            "kb_version": kb.kb_version,
            "manifest_checksum": kb.manifest_checksum,
            "embedding_model_id": kb.embedding_model_id,
            "embedding_model_revision": kb.embedding_model_revision,
            "chunker_version": kb.chunker_version,
            "chunk_size": kb.chunk_size,
            "overlap": kb.overlap,
        },
        "git_commit": experiment.git_commit,
        "dependency_lock_checksum": experiment.dependency_lock_checksum,
        "hardware_runtime_descriptor": experiment.hardware_runtime_descriptor,
        "cost_rate_snapshot_version": experiment.cost_rate_snapshot_version,
        "upfront_cost_usd": upfront_cost_usd,
        "expected_incident_ids": list(expected_ids),
        "prediction_incident_ids": list(incident_ids),
        "prediction_ids_by_incident": dict(sorted(prediction_id_by_incident.items())),
        "stored_prediction_count": len(predictions),
        "stored_retrieval_trace_count": len(traces),
        "stored_cost_record_count": readback.get("stored_cost_record_count"),
        "no_context_prediction_count": int(
            run.runtime_metadata.get("no_context_prediction_count", 0)
        ),
        "inference_retry_count": int(run.runtime_metadata.get("inference_retry_count", 0)),
        "metric_recomputation_verified": True,
        "result_hash": snapshot.result_hash,
        "metrics": snapshot.metric_values,
        "failure_codes": [list(item) for item in snapshot.failure_codes],
        "tracking": _tracking_payload(run),
        "predictions": predictions,
        "retrieval_traces": traces,
        "cost_records": readback.get("cost_records"),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
    return seal_run_evidence(payload)


def validate_phase8_rag_run_evidence(
    payload: Mapping[str, Any],
    *,
    root: Path,
    protocol: RAGProtocol,
    expected_split: str,
    expected_variant: RAGVariant | None = None,
) -> None:
    if payload.get("evidence_version") != RAG_RUN_EVIDENCE_VERSION:
        raise ValueError("unsupported Phase 08 RAG run evidence version")
    sealed = dict(payload)
    evidence_sha = sealed.pop("evidence_sha256", None)
    if not isinstance(evidence_sha, str) or evidence_sha != canonical_json_sha256(sealed):
        raise ValueError("Phase 08 RAG run evidence seal is invalid")
    if payload.get("split") != expected_split:
        raise ValueError("Phase 08 RAG run evidence split disagrees with expectation")
    if payload.get("scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("Phase 08 RAG evidence scientific config hash disagrees with protocol")
    if payload.get("ablation_suite_hash") != protocol.ablation_suite_hash:
        raise ValueError("Phase 08 RAG evidence ablation suite hash disagrees with protocol")

    suite = _load_suite_for_evidence(root, protocol)
    variant_id = payload.get("variant_id")
    if not isinstance(variant_id, str):
        raise ValueError("Phase 08 RAG evidence is missing variant_id")
    variant = protocol.assert_variant_allowed(
        split=expected_split,
        variant_id=variant_id,
        suite=suite,
    )
    if expected_variant is not None and variant != expected_variant:
        raise ValueError("Phase 08 RAG evidence variant disagrees with expectation")
    if payload.get("variant") != variant.model_dump(mode="json"):
        raise ValueError("Phase 08 RAG evidence variant payload disagrees with registered suite")

    expected_ids = expected_split_incident_ids(
        root,
        dataset_version=protocol.dataset_version,
        split=expected_split,
    )
    if payload.get("expected_incident_ids") != list(expected_ids):
        raise ValueError("Phase 08 RAG evidence expected incident IDs disagree with dataset")
    if payload.get("prediction_incident_ids") != list(expected_ids):
        raise ValueError("Phase 08 RAG evidence prediction IDs do not exactly match the split")
    if payload.get("stored_prediction_count") != len(expected_ids):
        raise ValueError("Phase 08 RAG evidence prediction count is incomplete")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Phase 08 RAG evidence metrics must be an object")
    missing_metrics = [metric for metric in _REQUIRED_METRICS if metric not in metrics]
    if missing_metrics:
        raise ValueError(f"Phase 08 RAG evidence is missing required metrics: {missing_metrics}")
    if payload.get("metric_recomputation_verified") is not True:
        raise ValueError("Phase 08 RAG evidence lacks metric recomputation proof")

    predictions = payload.get("predictions")
    traces = payload.get("retrieval_traces")
    raw_prediction_ids = payload.get("prediction_ids_by_incident")
    if not isinstance(predictions, list) or not all(isinstance(item, dict) for item in predictions):
        raise ValueError("Phase 08 RAG evidence predictions are malformed")
    if not isinstance(traces, list) or not all(isinstance(item, dict) for item in traces):
        raise ValueError("Phase 08 RAG evidence retrieval traces are malformed")
    if not isinstance(raw_prediction_ids, dict):
        raise ValueError("Phase 08 RAG evidence prediction ID map is malformed")
    _validate_ece_contract(metrics=metrics, predictions=[dict(item) for item in predictions])
    prediction_ids = {
        str(incident_id): str(prediction_id)
        for incident_id, prediction_id in raw_prediction_ids.items()
    }
    if set(prediction_ids) != set(expected_ids):
        raise ValueError("Phase 08 RAG evidence prediction ID map is incomplete")
    if payload.get("stored_retrieval_trace_count") != len(traces):
        raise ValueError("Phase 08 RAG evidence retrieval trace count is inconsistent")

    family_ids = _family_ids_from_dataset(
        root,
        dataset_version=protocol.dataset_version,
        incident_ids=expected_ids,
    )
    _validate_trace_contract(
        predictions=[dict(item) for item in predictions],
        traces=[dict(item) for item in traces],
        prediction_id_by_incident=prediction_ids,
        family_id_by_incident=family_ids,
    )

    cost_records = payload.get("cost_records")
    if not isinstance(cost_records, list) or len(cost_records) != len(expected_ids):
        raise ValueError("Phase 08 RAG evidence requires one cost record per prediction")
    if payload.get("stored_cost_record_count") != len(cost_records):
        raise ValueError("Phase 08 RAG evidence cost-record count is inconsistent")
    if payload.get("experiment_status") != "completed" or payload.get("run_status") != "completed":
        raise ValueError("Phase 08 RAG evidence requires completed experiment and run status")
