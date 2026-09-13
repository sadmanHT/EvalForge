# EvalForge

EvalForge is a reproducible research-and-product system for production incident diagnosis and root-cause classification. The primary study compares zero-shot, RAG-only, fine-tuned-only, and fine-tuned+RAG pipelines under one frozen research contract.

## Current implementation status

- Phase 01 — research contract, benchmark rules, model freeze, taxonomy, architecture decisions: **complete**.
- Phase 02 — monorepo, local environments, containers, worker skeleton, frontend shell, and CI foundation: **complete**.
- Phase 03 — versioned research dataset, family-disjoint splits, provenance, source preservation, and leakage checks: **complete**.
- Phase 04 — PostgreSQL persistence, migrations, immutable versioning, experiment/run/prediction schemas, and readback contracts: **complete**.
- Phase 05 — deterministic evaluation harness, calibration/cost/latency metrics, persistence, and golden regression evidence: **complete**.
- Phase 06 — frozen zero-shot Mistral-7B baseline, real GPU validation, one authorized locked test, W&B tracking, and reproducibility evidence: **complete**.
- Phase 07 — versioned knowledge base, deterministic embedding/chunking baseline, pgvector retrieval, research leakage guards, reindex evidence, and manual retrieval review: **complete pending the final exact-head CI confirmation on `phase-07-knowledge-base-retrieval-foundation`**.

## Repository boundaries

- `backend/` — FastAPI application, persistence, evaluation, inference, retrieval, worker, and reusable backend libraries.
- `frontend/` — React + TypeScript application shell and API-client foundation.
- `training/` — training orchestration boundary for later fine-tuning phases.
- `evals/` — evaluation and baseline-run orchestration.
- `datasets/` — versioned benchmark contracts and research assets.
- `knowledge_base/` — versioned retrieval sources introduced in Phase 07.
- `configs/` — frozen/versioned study, taxonomy, model, baseline, and KB contracts.
- `contracts/` — executable scientific/reproducibility contracts.
- `infrastructure/` — local/container/CI foundation notes.
- `docs/` — research protocol, ADRs, architecture, phase handoffs, and runbooks.
- `evidence/` — preserved machine-checkable phase-gate evidence.

Reusable implementation belongs in `backend/app/*`; scripts and notebooks call canonical libraries rather than duplicating scientific or persistence logic.

## Local foundation

Copy the example environment and start the development stack:

```bash
cp .env.example .env
docker compose up --build
```

Services:

- backend: http://localhost:8000 (`/health`, `/ready`)
- frontend: http://localhost:5173
- PostgreSQL/pgvector: localhost:5432
- Redis: localhost:6379

Clean reset:

```bash
docker compose down -v --remove-orphans
docker compose build --no-cache
docker compose up -d
```

## Canonical verification targets

```bash
make format
make lint
make typecheck
make test
make test-integration
make test-e2e
make test-regression
make db-migrate
make smoke
make eval-smoke
make verify-all
```

`make verify-all` is the cumulative local gate once backend/frontend dependencies and local Postgres/Redis are available. As of Phase 07 it includes the completed Phase 06 exit plus the Phase 07 contract, index/reindex, leakage audit, and hard-exit gate. `make fresh-smoke` performs the clean no-cache Compose rebuild path and also exercises the Phase 07 hard exit.

For the current retrieval foundation, the focused checks are:

```bash
make test-phase7
make phase7-contract
make phase7-index
make phase7-leakage-audit
make phase7-exit
```

No benchmark result displayed by EvalForge may be hard-coded; published values must trace to stored experiment evidence. Retrieval research mode likewise excludes held-out validation/test sources and same-family historical evidence through the canonical retrieval layer.
