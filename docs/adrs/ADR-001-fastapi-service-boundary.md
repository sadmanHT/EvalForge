# ADR-001 — FastAPI and Service Boundary

**Status:** Accepted

Use FastAPI for the HTTP/API layer. FastAPI validates requests, exposes REST/SSE endpoints, and orchestrates work. Long-running evaluation/training loops are never executed inside an HTTP request lifecycle; they are delegated to workers. This keeps API availability independent from experiment duration and enables retry, cancellation, and progress semantics.
