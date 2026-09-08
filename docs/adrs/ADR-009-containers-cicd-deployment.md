# ADR-009 — Docker Compose, CI/CD, and Deployment Strategy

**Status:** Accepted

## Decision
Use Docker/Docker Compose for reproducible local multi-service development, GitHub Actions for CI gates/regression enforcement, and a managed deployment platform for the public application/API. Provider-specific deployment details are deferred until the deployment phase so the application remains portable through environment-based configuration.

## Consequences
Phase 02 establishes reproducible containers and CI. Phase 15 selects/finalizes production hosting and proves public read-only evidence. Local hidden state may not be required for a passing phase gate.
