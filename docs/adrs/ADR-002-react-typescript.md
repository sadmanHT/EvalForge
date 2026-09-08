# ADR-002 — React + TypeScript Frontend

**Status:** Accepted

## Context
EvalForge requires a real comparison/dashboard product rather than notebook-only outputs.

## Decision
Use React with TypeScript. UI state consumes FastAPI REST/SSE contracts; published research values must come from persisted experiment evidence, never hard-coded benchmark numbers.

## Consequences
Typed client models reduce contract drift; frontend testing can later cover dashboard, comparison, failure analysis, Pareto, and live-run workflows.
