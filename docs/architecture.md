# EvalForge Architecture — Phase 01 Contract

## Purpose

This document freezes architectural boundaries and decision ownership before implementation. Later phases may refine internals, but material changes require an ADR.

## Target system

```text
React + TypeScript UI
        | REST + SSE
        v
FastAPI application ---------------------- OpenTelemetry
        |                                      |
        | enqueue                              v
        v                                  service traces
Redis broker/progress <----> Worker(s) ---- Langfuse LLM traces
                                |
                                +--> Base inference
                                +--> RAG inference
                                +--> Fine-tuned inference
                                +--> Combined inference
                                +--> Evaluation harness
                                +--> Training orchestration
                                |
                                v
                         PostgreSQL + pgvector
                         - datasets / incident families
                         - model and adapter versions
                         - KB versions / documents / chunks
                         - experiments / runs / predictions
                         - retrieval traces / metrics / failures
                         - jobs / artifacts / cost ledger

External evidence systems:
- Weights & Biases: training/evaluation experiment tracking and artifacts
- Hugging Face Hub: released adapter/model card
- Deployment platform: public application/API and read-only experiment evidence
```

## Architectural invariants

- Python 3.11+ for backend/evaluation/retrieval/training code.
- React + TypeScript for the frontend.
- PostgreSQL + pgvector is the durable system of record.
- Redis plus a real worker process owns long-running evaluation/training jobs; HTTP request handlers never execute large loops synchronously.
- Database schema changes are migration-backed by Alembic and must migrate from an empty database.
- All four primary pipelines use the same frozen base-model ID/revision.
- Exact structured root-cause labels are primary ground truth.
- Every published metric is derived from stored experiment evidence; no hard-coded result values.
- RAGAS/LLM judges remain supporting evaluators only when deterministic ground truth exists.
- W&B, Langfuse, and OpenTelemetry have separate responsibilities and are not interchangeable.

## Component responsibility boundaries

### FastAPI
Owns REST/SSE contracts, validation, authentication/rate-limit boundaries when introduced, read/write orchestration, and job submission. It must not own long-running model loops.

### Worker
Owns training/evaluation/retrieval-heavy execution, progress events, retries, and durable result writes.

### PostgreSQL + pgvector
Owns canonical relational experiment state and vector retrieval data. Later phases define normalized entities and migration history.

### Redis
Owns queue/broker/progress/cache responsibilities only; it is not the durable source of research truth.

### Evaluation library
Owns common prediction/evaluation contracts and metrics. All four pipelines plug into the same evaluator.

### Observability
- W&B: ML training/evaluation run evidence and artifacts.
- Langfuse: LLM/prompt/retrieval/token/cost traces.
- OpenTelemetry: service/API/worker infrastructure traces.

## ADR index

- [ADR-001](adrs/ADR-001-fastapi-boundary.md) — FastAPI service boundary
- [ADR-002](adrs/ADR-002-react-typescript.md) — React + TypeScript frontend
- [ADR-003](adrs/ADR-003-postgres-pgvector-alembic.md) — PostgreSQL, pgvector, Alembic
- [ADR-004](adrs/ADR-004-redis-worker.md) — Redis and worker architecture
- [ADR-005](adrs/ADR-005-primary-model.md) — Primary base model
- [ADR-006](adrs/ADR-006-rag-evaluation-integrity.md) — RAG/evaluation integrity
- [ADR-007](adrs/ADR-007-mlops-evidence.md) — W&B and Hugging Face evidence
- [ADR-008](adrs/ADR-008-observability.md) — Langfuse and OpenTelemetry
- [ADR-009](adrs/ADR-009-containers-cicd-deployment.md) — Docker Compose, CI/CD, deployment

## Phase boundary

Phase 01 freezes these decisions. Repository scaffolding, service implementation, containers, Makefile targets, and CI workflows belong to Phase 02 unless a minimal file is required solely to validate a Phase 01 contract.
