# Phase 04 persistence and reproducibility evidence

This directory preserves the committed evidence for EvalForge Phase 04: persistence, versioning, pgvector readiness, deterministic experiment identity, and database-backed reproducibility.

## Verified pre-merge head

- Branch: `phase-04-persistence-reproducibility-model`
- Exact verified SHA: `07316316b1550aea2ee47e3df611bb6ca68df4ab`
- GitHub Actions run: `34610303577`
- `verify-all`: PASS
- fresh `docker compose build --no-cache` + startup smoke: PASS
- Phase 03 hard exit during the cumulative gate: PASS
- Phase 04 hard exit: PASS

This is the final scientific/engineering validation head before the documentation-only closure commit. Phase 04 is not declared complete solely from this record; completion still requires the final cleaned documentation head to pass the same normal CI and fresh-Compose gates, merge into the Phase 03 target branch, and post-merge validation on the exact merge SHA.

## Phase 04 hard-exit facts

The hard-exit checker reported:

- `PHASE04_EXIT_GATE=PASS`
- Alembic head: `0002_phase04`
- persistence tables: `17`
- PostgreSQL extension `vector`: present and exercised by migration validation
- locked dataset version: `evalforge-incident-diagnosis-v0.1.0`
- persisted dataset records: `18`
- persisted incident families: `18`
- split counts: train `6`, validation `6`, test `6`
- dataset content checksum: `da9292d0e550cb141dbf1048e1a7c02f2fb4f6bd74b8642e781f06dd93acdbc5`
- dataset manifest checksum: `4a6767f0859eb93a2a9e7e8a8ea3caecb56b8a666c9ba9da97fee8e054c10c7d`
- deterministic experiment config SHA-256: `9b3bd8fb04a109b8b17f2b5252d0fe6a7c8900cb4dd7e8a3f62e3a9c66eef8dd`
- frozen base model: `mistralai/Mistral-7B-Instruct-v0.3`
- frozen base-model revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`

The same Phase 04 exit facts were reproduced from a fresh Compose volume after a no-cache image build.

## Test evidence from run 34610303577

- Ruff lint: PASS
- Ruff formatting check: PASS
- frontend Prettier/lint/typecheck: PASS
- mypy: `Success: no issues found in 34 source files`
- backend non-integration suite: `58 passed`, `13 deselected`
- Phase 03 focused data suite: `36 passed`; `app.data` coverage `90.96%`
- deterministic experiment-config suite: `16 passed`; module coverage `97.53%`
- real PostgreSQL integration suite: `13 passed`
- frontend Vitest shell suite: `2 passed`
- repository Python unittest suite: `51 tests`, PASS
- Phase 01 exit: PASS
- Phase 02 foundation and Compose health: PASS
- Phase 03 contract/reproducibility/hard exit: PASS
- secret scan: PASS
- production frontend build: PASS

The PostgreSQL integration tests intentionally provoke and assert rejected writes. Therefore the database logs contain expected FK-violation and immutability-trigger errors for negative tests; the integration suite itself completed with `13 passed`.

## Fresh-environment evidence

The `clean-compose` job in run `34610303577` performed:

1. `docker compose down -v --remove-orphans`
2. `docker compose build --no-cache`
3. `docker compose up -d`
4. worker smoke
5. backend readiness/health checks
6. frontend HTTP check
7. PostgreSQL and Redis checks
8. Phase 04 hard-exit verification against the freshly migrated and seeded database
9. clean teardown including volumes

Observed results included:

- `WORKER_SMOKE=PASS`
- `SUCCESS_STATE=SUCCEEDED`
- `FAILURE_STATE=FAILED`
- `PHASE02_COMPOSE_HEALTH=PASS`
- backend health HTTP `200`
- backend readiness HTTP `200`
- frontend HTTP `200`
- PostgreSQL `PASS`
- Redis `PASS`
- `PHASE04_EXIT_GATE=PASS`

GitHub Actions also preserved the generated `phase04-persistence-evidence` and `phase04-clean-environment` artifacts for that run.

## Committed evidence files

- `migration-history.txt` — canonical Phase 02 → Phase 04 Alembic upgrade path.
- `sample-experiment-config.canonical.json` — deterministic canonical experiment configuration sample.
- `sample-experiment-config.sha256` — expected SHA-256 for that canonical configuration.
- `docs/architecture.md` — relational persistence/versioning model and immutability contract.
- `docs/phase-04-handoff.md` — implementation and validation handoff.

## Persistence guarantees proved in Phase 04

- Historical dataset/model/adapter/knowledge-base version rows are database-immutable.
- Completed experiment rows cannot be mutated.
- Incidents cannot claim a split different from their persisted family split.
- Predictions cannot cross the dataset version bound to their experiment.
- Experiment creation validates that its test-manifest checksum and taxonomy version agree with the registered dataset version.
- Dataset import is lossless for the locked Phase 03 benchmark and is idempotent for the same immutable manifest.
- Job creation is idempotent for identical semantics and rejects reuse of an idempotency key with changed semantics.
- Repository transactions roll back partial writes on failure.
- Canonical experiment configuration rejects non-finite numeric values and produces a stable SHA-256 identity.

## Scope limits

Phase 04 establishes persistence and reproducibility infrastructure. It does not claim benchmark-model performance, calibrated accuracy, RAG quality, fine-tuning quality, or production-serving results. Knowledge-base, retrieval, prediction, metric, failure-annotation, artifact, cost, run, and job tables are schema capabilities for later phases; Phase 04 validates their relational contract rather than claiming completed later-phase experiments.
