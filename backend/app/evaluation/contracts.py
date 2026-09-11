from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParseStatus(StrEnum):
    OK = "OK"
    INVALID_JSON = "INVALID_JSON"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    UNKNOWN_LABEL = "UNKNOWN_LABEL"
    PARTIAL_OUTPUT = "PARTIAL_OUTPUT"


class ConfidenceSource(StrEnum):
    NORMALIZED_LABEL_SEQUENCE_LOG_LIKELIHOOD = "normalized_label_sequence_log_likelihood"
    TEMPERATURE_SCALED_LABEL_SEQUENCE_LOG_LIKELIHOOD = (
        "temperature_scaled_label_sequence_log_likelihood"
    )


class RankedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    score: float
    probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_finite(self) -> RankedLabel:
        if not math.isfinite(self.score):
            raise ValueError("ranked-label score must be finite")
        if self.probability is not None and not math.isfinite(self.probability):
            raise ValueError("ranked-label probability must be finite")
        return self


class Prediction(BaseModel):
    """Canonical, pipeline-independent prediction contract.

    A non-OK parse status is a first-class prediction outcome. Evaluators count it as
    incorrect instead of crashing the run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(min_length=1)
    predicted_root_cause_code: str | None = None
    ranked_labels: tuple[RankedLabel, ...] = ()
    label_probabilities: dict[str, float] | None = None
    confidence_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource | None = None
    reasoning: str | None = None
    evidence_citations: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    raw_model_output: str = ""
    parse_status: ParseStatus = ParseStatus.OK
    parse_error: str | None = None
    pipeline_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> Prediction:
        if self.parse_status is ParseStatus.OK and not self.predicted_root_cause_code:
            raise ValueError("OK predictions require predicted_root_cause_code")
        if (self.confidence_probability is None) != (self.confidence_source is None):
            raise ValueError("confidence probability and source must be present together")
        if self.latency_ms is not None and not math.isfinite(self.latency_ms):
            raise ValueError("latency must be finite")
        if self.cost_usd is not None and not math.isfinite(self.cost_usd):
            raise ValueError("cost must be finite")
        if self.confidence_probability is not None and not math.isfinite(
            self.confidence_probability
        ):
            raise ValueError("confidence must be finite")
        labels = [entry.label for entry in self.ranked_labels]
        if len(labels) != len(set(labels)):
            raise ValueError("ranked labels must be unique")
        for previous, current in zip(self.ranked_labels, self.ranked_labels[1:], strict=False):
            if previous.score < current.score:
                raise ValueError("ranked labels must be ordered by descending score")
        if (
            self.parse_status is ParseStatus.OK
            and self.ranked_labels
            and self.ranked_labels[0].label != self.predicted_root_cause_code
        ):
            raise ValueError("top-ranked label must match predicted_root_cause_code")
        if self.label_probabilities is not None:
            if not self.label_probabilities:
                raise ValueError("label probability distribution may not be empty")
            total = 0.0
            for label, probability in self.label_probabilities.items():
                if not label:
                    raise ValueError("probability labels may not be empty")
                if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                    raise ValueError("label probabilities must be finite values in [0, 1]")
                total += probability
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("label probabilities must sum to one")
        if self.total_tokens is not None:
            known_parts = [part for part in (self.input_tokens, self.output_tokens) if part is not None]
            if len(known_parts) == 2 and self.total_tokens != sum(known_parts):
                raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self

    @classmethod
    def from_raw_output(
        cls,
        *,
        incident_id: str,
        raw_model_output: str,
        allowed_labels: set[str],
        latency_ms: float | None = None,
        pipeline_metadata: dict[str, Any] | None = None,
    ) -> Prediction:
        try:
            payload = json.loads(raw_model_output)
        except json.JSONDecodeError as exc:
            return cls(
                incident_id=incident_id,
                raw_model_output=raw_model_output,
                parse_status=ParseStatus.INVALID_JSON,
                parse_error=str(exc),
                latency_ms=latency_ms,
                pipeline_metadata=pipeline_metadata or {},
            )
        if not isinstance(payload, dict):
            return cls(
                incident_id=incident_id,
                raw_model_output=raw_model_output,
                parse_status=ParseStatus.PARTIAL_OUTPUT,
                parse_error="model output must be a JSON object",
                latency_ms=latency_ms,
                pipeline_metadata=pipeline_metadata or {},
            )
        code = payload.get("root_cause_code")
        if not isinstance(code, str) or not code:
            return cls(
                incident_id=incident_id,
                raw_model_output=raw_model_output,
                parse_status=ParseStatus.MISSING_REQUIRED_FIELD,
                parse_error="root_cause_code is required",
                latency_ms=latency_ms,
                pipeline_metadata=pipeline_metadata or {},
            )
        if code not in allowed_labels:
            return cls(
                incident_id=incident_id,
                predicted_root_cause_code=code,
                raw_model_output=raw_model_output,
                parse_status=ParseStatus.UNKNOWN_LABEL,
                parse_error=f"unknown root_cause_code: {code}",
                latency_ms=latency_ms,
                pipeline_metadata=pipeline_metadata or {},
            )

        ranked: list[RankedLabel] = []
        partial_error: str | None = None
        raw_ranked = payload.get("ranked_labels")
        if raw_ranked is not None:
            if not isinstance(raw_ranked, list):
                partial_error = "ranked_labels must be a list when present"
            else:
                try:
                    for item in raw_ranked:
                        if isinstance(item, str):
                            ranked.append(RankedLabel(label=item, score=0.0))
                        elif isinstance(item, dict):
                            ranked.append(RankedLabel.model_validate(item))
                        else:
                            raise ValueError("ranked label entries must be strings or objects")
                except (TypeError, ValueError) as exc:
                    ranked = []
                    partial_error = str(exc)
        if ranked and ranked[0].label != code:
            partial_error = "top-ranked label disagrees with root_cause_code"
            ranked = []
        if any(entry.label not in allowed_labels for entry in ranked):
            partial_error = "ranked_labels contains an unknown label"
            ranked = []

        confidence = payload.get("confidence_probability")
        confidence_source = payload.get("confidence_source")
        parsed_confidence: float | None = None
        parsed_source: ConfidenceSource | None = None
        if confidence is not None or confidence_source is not None:
            try:
                parsed_confidence = float(confidence)
                parsed_source = ConfidenceSource(str(confidence_source))
                if not math.isfinite(parsed_confidence) or not 0.0 <= parsed_confidence <= 1.0:
                    raise ValueError("confidence_probability must be finite and in [0,1]")
            except (TypeError, ValueError) as exc:
                parsed_confidence = None
                parsed_source = None
                partial_error = f"invalid confidence: {exc}"

        status = ParseStatus.PARTIAL_OUTPUT if partial_error else ParseStatus.OK
        return cls(
            incident_id=incident_id,
            predicted_root_cause_code=code,
            ranked_labels=tuple(ranked),
            confidence_probability=parsed_confidence,
            confidence_source=parsed_source,
            reasoning=payload.get("reasoning") if isinstance(payload.get("reasoning"), str) else None,
            evidence_citations=tuple(
                item for item in payload.get("evidence_citations", []) if isinstance(item, str)
            )
            if isinstance(payload.get("evidence_citations", []), list)
            else (),
            retrieved_chunk_ids=tuple(
                item for item in payload.get("retrieved_chunk_ids", []) if isinstance(item, str)
            )
            if isinstance(payload.get("retrieved_chunk_ids", []), list)
            else (),
            raw_model_output=raw_model_output,
            parse_status=status,
            parse_error=partial_error,
            latency_ms=latency_ms,
            pipeline_metadata=pipeline_metadata or {},
        )


class EvaluationExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    root_cause_code: str = Field(min_length=1)
    root_cause_category: str = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...] = ()


def canonical_prediction_payload(prediction: Prediction) -> dict[str, Any]:
    return prediction.model_dump(mode="json", exclude_none=False)
