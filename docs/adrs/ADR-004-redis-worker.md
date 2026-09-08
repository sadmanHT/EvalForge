# ADR-004 — Redis + Real Worker Process

**Status:** Accepted

## Context
Evaluation/training jobs can exceed HTTP timeouts and must survive beyond request lifecycles.

## Decision
Use Redis for queue/broker/progress/cache responsibilities and one or more separate worker processes for long-running execution.

## Consequences
FastAPI submits and observes jobs; workers execute them and persist canonical outcomes to PostgreSQL. Later phases must prove API -> queue -> worker -> DB -> SSE integration.
