# EvalForge

EvalForge is a reproducible research-and-product system for production incident diagnosis and root-cause classification. The primary study compares zero-shot, RAG-only, fine-tuned-only, and fine-tuned+RAG pipelines under one frozen research contract.

## Current implementation status

- Phase 01 — research contract, benchmark rules, model freeze, taxonomy, architecture decisions: **complete** on `phase-01-research-contract`.
- Phase 02 — monorepo, local environments, containers, worker skeleton, frontend shell, and CI foundation: **in progress** on `phase-02-repository-foundation` until its clean Compose and CI hard gates are proven.

## Repository boundaries

- `backend/` — FastAPI application, reusable backend libraries, worker implementation, Alembic environment.
- `frontend/` — React + TypeScript application shell and API-client foundation.
- `training/` — training orchestration boundary; implementation intentionally deferred to later phases.
- `evals/` — evaluation orchestration boundary; benchmark implementation intentionally deferred.
- `datasets/` — dataset contracts and later versioned benchmark assets.
- `configs/` — frozen/versioned study, taxonomy, and model contracts.
- `contracts/` — executable scientific/reproducibility contracts.
- `infrastructure/` — local/container/CI foundation notes.
- `docs/` — research protocol, ADRs, architecture, phase handoffs.
- `evidence/` — preserved phase-gate evidence.

Reusable implementation belongs in `backend/app/*`; scripts and later notebooks must call canonical libraries rather than duplicate their logic.

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

`make verify-all` is the cumulative local gate once backend/frontend dependencies and local Postgres/Redis are available. `make fresh-smoke` performs the clean Compose rebuild path.

No benchmark result displayed by EvalForge may be hard-coded; later published values must trace to stored experiment evidence.