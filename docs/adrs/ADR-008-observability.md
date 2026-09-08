# ADR-008 — Langfuse + OpenTelemetry Observability Split

**Status:** Accepted

## Decision
Use Langfuse for LLM-centric traces (prompts, retrieval context, token/cost information) and OpenTelemetry for service/API/worker infrastructure traces.

## Consequences
The two systems have distinct ownership. Duplicate telemetry should be minimized, and neither system is the canonical metric database.
