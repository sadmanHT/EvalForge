# ADR-001 — FastAPI Service Boundary

**Status:** Accepted  
**Phase:** 01

## Context
EvalForge needs typed REST endpoints and server-sent events while long-running model jobs execute outside request lifecycles.

## Decision
Use FastAPI for the Python application/API boundary. Request handlers validate/authorize/orchestrate work and enqueue long jobs; they do not run full evaluation or training loops synchronously.

## Consequences
- shared Python schemas can support API and experiment contracts;
- SSE can expose progress while workers execute jobs;
- worker/database boundaries remain explicit;
- async HTTP does not imply model compute belongs in-process.

## Alternatives considered
Flask/Django were viable but add either less native typed async ergonomics or broader framework surface than required for this focused API.
