# EvalForge Architecture — Phase 01 Baseline

## System intent

EvalForge is a reproducible research-and-product system. The benchmark and evaluation contract are the core assets; the product and MLOps layers make the evidence reproducible, inspectable, and usable.

## Target architecture

```text
React + TypeScript UI
        | REST + SSE
        v
FastAPI application -------------------- OpenTelemetry
        |
        | enqueue
        v
Redis broker/progress <----> Worker(s) ---- Langfuse LLM traces
                              |
                              +--> base inference
                              +--> RAG inference
                              +--> fine-tuned inference
                              +--> combined inference
                              +--> evaluation harness
                              +--> training orchestration
                              |
                              v
                      PostgreSQL + pgvector
                      - dataset/family versions
                      - model/adapter versions
                      - KB documents/chunks
                      - experiments/runs/predictions
                      - retrieval traces/metrics/failures
                      - jobs/artifacts/cost ledger

External evidence:
- Weights & Biases: ML/evaluation run metadata and artifacts
- Hugging Face Hub: released adapter/model card
- deployment platform: public read-only evidence and controlled execution
```

## Canonical boundaries

- **FastAPI** owns HTTP orchestration and validation, not long-running evaluation loops.
- **Workers** own long-running evaluation/training orchestration.
- **PostgreSQL + pgvector** is the system of record. W&B/Langfuse are complementary evidence systems.
- **Redis** is transient queue/progress infrastructure, not authoritative experiment storage.
- **Reusable backend libraries** own evaluation, retrieval, training, and inference logic. CLI scripts/notebooks call those libraries rather than duplicating logic.
- **Frontend** consumes versioned API contracts and never hard-codes research metrics.
- **Alembic** owns every database schema migration.
- **Model/data/KB/config versions** are explicit and immutable after completed primary experiments use them.

## Research boundary

OpsSentinel is an external upstream project. EvalForge may import/version benchmark material through a later adapter, but it must not rely on undocumented mutable state in that repository. Any imported upstream snapshot records source repository and commit/version. EvalForge development never writes to OpsSentinel.

## ADR index

- ADR-001: FastAPI and service boundary
- ADR-002: React + TypeScript frontend
- ADR-003: PostgreSQL + pgvector and Alembic
- ADR-004: Redis-backed asynchronous workers
- ADR-005: Frozen primary model
- ADR-006: RAG/evaluation integrity
- ADR-007: W&B and Hugging Face evidence
- ADR-008: Langfuse + OpenTelemetry observability
- ADR-009: Docker Compose, CI/CD, provider-independent deployment
