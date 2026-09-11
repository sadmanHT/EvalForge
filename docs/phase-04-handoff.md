# Phase 04 handoff — Persistence, Versioning & Reproducibility Model

## Status

Phase 04 implementation is functionally complete and has passed its hard-exit criteria on the verified implementation head `07316316b1550aea2ee47e3df611bb6ca68df4ab` in GitHub Actions run `34610303577`.

This handoff is part of the documentation-only closure commit. Phase 04 must not be declared complete until this final cleaned head also passes normal CI plus fresh Compose, is merged into `phase-03-dataset-engineering`, and the exact target merge SHA passes the same post-merge validation. Phase 05 has not started.

## What Phase 04 adds

Phase 04 turns the Phase 03 research benchmark into an immutable, relationally versioned persistence substrate suitable for later evaluation, RAG, fine-tuning, analysis, and serving work.

The implementation adds:

- SQLAlchemy persistence models and a shared declarative base.
- Alembic migration `0002_phase04` on top of the Phase 02 foundation migration.
- PostgreSQL/pgvector schema creation and validation.
- immutable dataset, model, adapter, and knowledge-base version entities.
- experiment/run/prediction/retrieval/metric/failure/artifact/job/cost persistence entities.
- deterministic experiment configuration canonicalization and SHA-256 identity.
- a repository layer with transaction and idempotency semantics.
- deterministic, lossless import of the locked Phase 03 research dataset.
- persistence seeding used by cumulative CI and clean Compose startup.
- a Phase 04 hard-exit checker that validates the actual PostgreSQL state.
- real-PostgreSQL integration tests for migrations, invariants, rollback, idempotency, and immutable history.
- architecture documentation and committed reproducibility evidence.

## Canonical relational model

Phase 04 materializes 17 persistence tables:

1. `dataset_versions`
2. `incident_families`
3. `incidents`
4. `model_versions`
5. `adapter_versions`
6. `knowledge_base_versions`
7. `kb_documents`
8. `kb_chunks`
9. `experiments`
10. `runs`
11. `predictions`
12. `retrieval_traces`
13. `metrics`
14. `failure_annotations`
15. `artifacts`
16. `jobs`
17. `cost_records`

`kb_chunks.embedding` uses pgvector with `vector(1536)` in this schema revision. The schema is intentionally prepared for later RAG work, but Phase 04 does not claim that a production knowledge base or retrieval benchmark has been built.

## Migration contract

The canonical upgrade path is:

`base -> 0001_phase02 -> 0002_phase04`

The real-PostgreSQL integration suite explicitly downgrades to base, upgrades through `0001_phase02`, then upgrades to head and verifies:

- Alembic revision `0002_phase04`.
- pgvector extension availability.
- the expected schema can be created from an empty database.

The migration history is preserved in `evidence/phase-04/migration-history.txt`.

## Locked Phase 03 benchmark persistence

The database seed imports `evalforge-incident-diagnosis-v0.1.0` without changing the Phase 03 scientific record.

Verified persisted state:

- incidents: `18`
- incident families: `18`
- train: `6`
- validation: `6`
- test: `6`
- content checksum: `da9292d0e550cb141dbf1048e1a7c02f2fb4f6bd74b8642e781f06dd93acdbc5`
- manifest checksum: `4a6767f0859eb93a2a9e7e8a8ea3caecb56b8a666c9ba9da97fee8e054c10c7d`

Import is idempotent when the immutable manifest matches. Reusing a dataset version with conflicting manifest identity/counts is rejected.

The import transaction explicitly persists the dataset version, then incident families, then incidents. This preserves the strong family/split foreign-key invariant while avoiding ORM insertion-order ambiguity.

## Database invariants

Phase 04 enforces the following persistence rules:

- incident `(dataset_version, family_id, split)` must reference the exact persisted incident-family tuple.
- prediction `(experiment_id, dataset_version)` must match the dataset version of its experiment.
- predictions reference an incident from that same dataset version.
- retrieval traces bind predictions to chunks from the same knowledge-base version.
- adapter versions are tied to their base model version.
- completed experiment rows are frozen against mutation.
- historical dataset/model/adapter/knowledge-base version rows reject UPDATE and DELETE.
- confidence values are constrained to `[0,1]` when present.
- split fields are constrained to train/validation/test.
- retrieval/chunk configuration has positive and overlap bounds.
- RAG/fine-tuned experiment rows must carry the required retrieval/adapter identity, while non-RAG/non-fine-tuned variants may not carry incompatible identity fields.

At repository creation time, experiments also verify that the supplied test-manifest checksum and label-taxonomy version agree with the registered dataset version.

## Deterministic experiment identity

`ExperimentConfig` is frozen and rejects extra fields. Its canonical JSON representation is serialized with stable key ordering and separators and then SHA-256 hashed.

Non-finite floats are rejected recursively before JSON serialization, including nested generation/evaluator settings. This prevents semantically invalid NaN/Infinity values from producing unstable or implementation-dependent identities.

The committed canonical sample hashes to:

`9b3bd8fb04a109b8b17f2b5252d0fe6a7c8900cb4dd7e8a3f62e3a9c66eef8dd`

The Phase 04 hard-exit checker independently recomputes this hash and verifies the stored experiment row against it.

The seeded smoke experiment binds the frozen base model:

- model: `mistralai/Mistral-7B-Instruct-v0.3`
- revision: `e8737b84b4470b28db3a0be719b362b1bd39a14d`

## Transactions and retry safety

`PersistenceRepository` provides explicit transaction boundaries. Integration coverage proves a partial model-version write is rolled back when the transaction fails.

Jobs use a unique idempotency key plus a canonical payload hash. A retry with identical job semantics returns the existing job; reuse of the same key with different kind/payload/experiment semantics is rejected.

## Validation evidence

Exact verified implementation head:

- SHA: `07316316b1550aea2ee47e3df611bb6ca68df4ab`
- Actions run: `34610303577`
- `verify-all`: PASS
- `clean-compose`: PASS

Key results from `verify-all`:

- Ruff lint/format: PASS
- frontend formatting/lint/typecheck: PASS
- mypy: no issues in `34` source files
- backend non-integration tests: `58 passed`
- Phase 03 focused suite: `36 passed`, `90.96%` `app.data` coverage
- experiment-config suite: `16 passed`, `97.53%` module coverage
- real PostgreSQL integration suite: `13 passed`
- frontend tests: `2 passed`
- repository unittest discovery: `51` tests, PASS
- Phase 01 exit: PASS
- Phase 02 foundation: PASS
- Phase 03 contract/reproducibility/hard exit: PASS
- secret scan: PASS
- frontend production build: PASS
- Phase 04 persistence seed: PASS
- Phase 04 hard exit: PASS

Phase 04 hard-exit output:

- `ALEMBIC_HEAD=0002_phase04`
- `PERSISTENCE_TABLES=17`
- `DATASET_RECORDS=18`
- `DATASET_FAMILIES=18`
- `DATASET_TRAIN=6`
- `DATASET_VALIDATION=6`
- `DATASET_TEST=6`
- `EXPERIMENT_CONFIG_HASH=9b3bd8fb04a109b8b17f2b5252d0fe6a7c8900cb4dd7e8a3f62e3a9c66eef8dd`

The negative integration tests deliberately generate FK and immutability errors in PostgreSQL logs. Those errors are the expected proof that invalid cross-split/cross-dataset writes and historical mutations are rejected; the suite completed with `13 passed`.

## Fresh Compose proof

The same run destroyed existing Compose state and volumes, executed `docker compose build --no-cache`, started all services, then ran health and Phase 04 exit checks against that newly migrated database.

Verified clean-environment results:

- worker smoke: PASS
- successful task terminal state: `SUCCEEDED`
- intentionally failing task terminal state: `FAILED`
- backend health: HTTP `200`
- backend readiness: HTTP `200`
- frontend: HTTP `200`
- PostgreSQL: PASS
- Redis: PASS
- Phase 04 hard exit: PASS
- clean teardown including volumes: PASS

## Evidence locations

- `evidence/phase-04/README.md`
- `evidence/phase-04/migration-history.txt`
- `evidence/phase-04/sample-experiment-config.canonical.json`
- `evidence/phase-04/sample-experiment-config.sha256`
- `docs/architecture.md`
- `backend/migrations/versions/0002_phase04_persistence.py`
- `backend/tests/integration/test_phase4_persistence.py`
- `backend/tests/test_experiment_config.py`
- `scripts/check_phase4_exit.py`

GitHub Actions run `34610303577` also preserved `phase04-persistence-evidence`, `phase04-clean-environment`, and dependency-lock artifacts.

## Known scope boundaries

Phase 04 is a persistence/reproducibility phase, not a model-results phase. It does not claim:

- production benchmark performance,
- calibrated prediction quality,
- production RAG retrieval quality,
- completed fine-tuning,
- completed combined RAG + fine-tuning experiments,
- production deployment or serving performance.

Those later capabilities can now build on a persistence layer that has immutable version lineage, deterministic experiment identity, database-enforced cross-entity consistency, real migration evidence, and fresh-environment reproducibility.

No OpsSentinel source-project changes were made in Phase 04. Phase 03 transport material remains external/pinned as before and is not introduced into the Phase 04 image as raw research data.
