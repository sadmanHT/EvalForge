# ADR-008 — Langfuse + OpenTelemetry

**Status:** Accepted

Use Langfuse for LLM/retrieval trace-level observability and OpenTelemetry for service/infrastructure traces. Trace IDs must correlate with experiment, run, and prediction IDs. Telemetry outages must not corrupt or falsely fail valid experiment execution; observability adapters degrade explicitly and record their own failure.
