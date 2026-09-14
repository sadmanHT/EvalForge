from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.inference.rag_pipeline import RAGRetrievalTrace
from app.models import Prediction as PredictionRow
from app.models import RetrievalTrace as RetrievalTraceRow


def _stable_trace_id(prediction_id: str, rank: int, chunk_id: str) -> str:
    material = f"{prediction_id}:{rank}:{chunk_id}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"retrieval-{digest}"


def persist_rag_retrieval_traces(
    session: Session,
    *,
    run_id: str,
    kb_version: str,
    traces_by_incident: Mapping[str, Sequence[RAGRetrievalTrace]],
) -> int:
    predictions = session.scalars(
        select(PredictionRow)
        .where(PredictionRow.run_id == run_id)
        .order_by(PredictionRow.incident_id)
    ).all()
    if not predictions:
        raise ValueError(f"no persisted predictions for RAG run: {run_id}")
    prediction_by_incident = {row.incident_id: row for row in predictions}
    if set(prediction_by_incident) != set(traces_by_incident):
        raise ValueError("retrieval trace incident IDs must exactly match persisted predictions")

    persisted = 0
    for incident_id in sorted(prediction_by_incident):
        prediction = prediction_by_incident[incident_id]
        if prediction.kb_version != kb_version:
            raise ValueError("retrieval trace KB version disagrees with persisted prediction")
        traces = tuple(traces_by_incident[incident_id])
        ranks = tuple(trace.rank for trace in traces)
        if ranks != tuple(range(1, len(traces) + 1)):
            raise ValueError("retrieval trace ranks must be contiguous and start at one")
        if len({trace.chunk_id for trace in traces}) != len(traces):
            raise ValueError("retrieval traces contain duplicate chunk IDs for one prediction")

        for trace in traces:
            trace_id = _stable_trace_id(prediction.prediction_id, trace.rank, trace.chunk_id)
            metadata = trace.persistence_metadata()
            row = session.get(RetrievalTraceRow, trace_id)
            if row is None:
                session.add(
                    RetrievalTraceRow(
                        retrieval_trace_id=trace_id,
                        prediction_id=prediction.prediction_id,
                        kb_version=kb_version,
                        chunk_id=trace.chunk_id,
                        rank=trace.rank,
                        score=trace.final_score,
                        trace_metadata=metadata,
                    )
                )
            elif (
                row.prediction_id != prediction.prediction_id
                or row.kb_version != kb_version
                or row.chunk_id != trace.chunk_id
                or row.rank != trace.rank
                or row.score != trace.final_score
                or row.trace_metadata != metadata
            ):
                raise ValueError("retrieval trace identity was reused with different evidence")
            persisted += 1
    session.flush()
    return persisted
