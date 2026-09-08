#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import load_json_yaml, validate_label_taxonomy, validate_model_config, validate_study_config

REQUIRED_ARTIFACTS = [
    "docs/research-protocol.md",
    "docs/model-selection.md",
    "docs/architecture.md",
    "docs/quality-gates.md",
    "datasets/schema.md",
    "configs/model.yaml",
    "configs/study.yaml",
    "configs/label-taxonomy.yaml",
    "scripts/model_smoke.py",
    "docs/phase-01-handoff.md",
]

REQUIRED_PROTOCOL_PHRASES = [
    "Pre-specified hypotheses",
    "Exact root-cause-code accuracy",
    "paired bootstrap",
    "normalized sequence log-likelihood",
    "amortized total cost/query",
    "Protocol-change rule",
]

REQUIRED_ADR_TOPICS = {
    "FastAPI": "ADR-001-fastapi-boundary.md",
    "React": "ADR-002-react-typescript.md",
    "pgvector": "ADR-003-postgres-pgvector-alembic.md",
    "Alembic": "ADR-003-postgres-pgvector-alembic.md",
    "Redis": "ADR-004-redis-worker.md",
    "Mistral": "ADR-005-primary-model.md",
    "RAGAS": "ADR-006-rag-evaluation-integrity.md",
    "Weights & Biases": "ADR-007-mlops-evidence.md",
    "Hugging Face": "ADR-007-mlops-evidence.md",
    "Langfuse": "ADR-008-observability.md",
    "OpenTelemetry": "ADR-008-observability.md",
    "Docker Compose": "ADR-009-containers-cicd-deployment.md",
    "GitHub Actions": "ADR-009-containers-cicd-deployment.md",
}


def fail(message: str) -> None:
    raise SystemExit(f"PHASE01_VALIDATION_FAILED: {message}")


def main() -> int:
    for relative in REQUIRED_ARTIFACTS:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing/empty artifact: {relative}")

    taxonomy = load_json_yaml(ROOT / "configs/label-taxonomy.yaml")
    model = load_json_yaml(ROOT / "configs/model.yaml")
    study = load_json_yaml(ROOT / "configs/study.yaml")
    validate_label_taxonomy(taxonomy)
    validate_model_config(model)
    validate_study_config(study, model, taxonomy)

    protocol = (ROOT / "docs/research-protocol.md").read_text(encoding="utf-8")
    for phrase in REQUIRED_PROTOCOL_PHRASES:
        if phrase.lower() not in protocol.lower():
            fail(f"research protocol missing required concept: {phrase}")

    adrs = ROOT / "docs/adrs"
    if len(list(adrs.glob("ADR-*.md"))) < 9:
        fail("expected at least 9 architecture decision records")
    for topic, filename in REQUIRED_ADR_TOPICS.items():
        text = (adrs / filename).read_text(encoding="utf-8")
        if topic.lower() not in text.lower():
            fail(f"{filename} does not cover required topic: {topic}")

    # Phase 01 scientific artifacts must not ship with unresolved placeholder claims.
    for relative in ["docs/research-protocol.md", "docs/model-selection.md", "docs/architecture.md", "docs/quality-gates.md", "datasets/schema.md"]:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in ("[TODO]", "[TBD]", "[X]%", "[Y]%", "PLACEHOLDER_RESULT"):
            if token in text:
                fail(f"unresolved placeholder {token!r} in {relative}")

    revision = model["base_model_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        fail("base_model_revision must be a full 40-character git SHA")

    if model.get("smoke_test", {}).get("required_for_phase_complete") is not True:
        fail("model smoke must be a hard phase-completion requirement")

    json.dumps(study, sort_keys=True)
    print("PHASE01_STATIC_VALIDATION=PASS")
    print(f"STUDY_ID={study['study_id']}")
    print(f"MODEL={model['base_model_id']}@{model['base_model_revision']}")
    print(f"TAXONOMY_VERSION={taxonomy['taxonomy_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
