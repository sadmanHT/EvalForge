from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.contracts import Prediction

PROMPT_REGISTRY_VERSION = "phase8-prompt-registry-v2"
BASELINE_PROMPT_VERSION = "zero-shot-baseline-v1"
RAG_PROMPT_VERSION = "rag-context-v1"
OUTPUT_SCHEMA_VERSION = "root-cause-prediction-v1"
RAG_NO_CONTEXT_SENTINEL = "(no eligible retrieval context was available)"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    output_schema_version: str
    template: str

    def render(
        self,
        *,
        title: str,
        description: str,
        allowed_labels: Sequence[str],
    ) -> str:
        labels = tuple(allowed_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        if not title.strip() or not description.strip():
            raise ValueError("incident title and description are required")
        return self.template.format(
            allowed_labels=json.dumps(labels, ensure_ascii=False, separators=(",", ":")),
            title=title.strip(),
            description=description.strip(),
        )


@dataclass(frozen=True)
class RAGPromptTemplate:
    version: str
    output_schema_version: str
    template: str

    def render(
        self,
        *,
        title: str,
        description: str,
        allowed_labels: Sequence[str],
        context: str,
    ) -> str:
        labels = tuple(allowed_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("allowed_labels must be a non-empty unique sequence")
        if not title.strip() or not description.strip():
            raise ValueError("incident title and description are required")
        normalized_context = context.strip() or RAG_NO_CONTEXT_SENTINEL
        return self.template.format(
            allowed_labels=json.dumps(labels, ensure_ascii=False, separators=(",", ":")),
            title=title.strip(),
            description=description.strip(),
            context=normalized_context,
        )


_BASELINE_TEMPLATE = PromptTemplate(
    version=BASELINE_PROMPT_VERSION,
    output_schema_version=OUTPUT_SCHEMA_VERSION,
    template=(
        "You are classifying the root cause of a production incident.\n"
        "Choose exactly one root_cause_code from this frozen label set: {allowed_labels}.\n"
        "Return exactly one JSON object with this schema: "
        '{{"root_cause_code":"<one allowed label>",'
        '"reasoning":"<brief evidence-based explanation>"}}.\n'
        "Do not add markdown fences, prose before the JSON, retrieval citations, "
        "or labels outside the set.\n\n"
        "Incident title: {title}\n"
        "Incident description: {description}\n"
    ),
)

_RAG_TEMPLATE = RAGPromptTemplate(
    version=RAG_PROMPT_VERSION,
    output_schema_version=OUTPUT_SCHEMA_VERSION,
    template=(
        "You are classifying the root cause of a production incident.\n"
        "Choose exactly one root_cause_code from this frozen label set: {allowed_labels}.\n"
        "Return exactly one JSON object with this schema: "
        '{{"root_cause_code":"<one allowed label>",'
        '"reasoning":"<brief evidence-based explanation>"}}.\n'
        "Do not add markdown fences, prose before the JSON, retrieval citations, "
        "or labels outside the set.\n"
        "Retrieved context is supporting evidence only. It may be incomplete or irrelevant; "
        "do not treat document identifiers or retrieval rank as ground truth.\n\n"
        "Retrieved context:\n{context}\n\n"
        "Incident title: {title}\n"
        "Incident description: {description}\n"
    ),
)

_PROMPTS = {_BASELINE_TEMPLATE.version: _BASELINE_TEMPLATE}
_RAG_PROMPTS = {_RAG_TEMPLATE.version: _RAG_TEMPLATE}


def get_prompt_template(version: str) -> PromptTemplate:
    try:
        return _PROMPTS[version]
    except KeyError as exc:
        raise KeyError(f"unknown prompt version: {version}") from exc


def get_rag_prompt_template(version: str) -> RAGPromptTemplate:
    try:
        return _RAG_PROMPTS[version]
    except KeyError as exc:
        raise KeyError(f"unknown RAG prompt version: {version}") from exc


def _unwrap_single_json_fence(raw_output: str) -> tuple[str, bool]:
    stripped = raw_output.strip()
    if not stripped.startswith("```"):
        return stripped, False
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped, False
    opener = lines[0].strip().lower()
    if opener not in {"```", "```json"}:
        return stripped, False
    return "\n".join(lines[1:-1]).strip(), True


def parse_baseline_output(
    *,
    incident_id: str,
    raw_output: str,
    allowed_labels: Sequence[str],
    latency_ms: float | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> Prediction:
    labels = tuple(allowed_labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("allowed_labels must be a non-empty unique sequence")
    normalized, unwrapped = _unwrap_single_json_fence(raw_output)
    metadata = dict(pipeline_metadata or {})
    if unwrapped:
        metadata["parser_normalization"] = "single_json_markdown_fence_removed"
    parsed = Prediction.from_raw_output(
        incident_id=incident_id,
        raw_model_output=normalized,
        allowed_labels=set(labels),
        latency_ms=latency_ms,
        pipeline_metadata=metadata,
    )
    if normalized == raw_output:
        return parsed
    payload = parsed.model_dump(mode="python", exclude_none=False)
    payload["raw_model_output"] = raw_output
    return Prediction.model_validate(payload)
