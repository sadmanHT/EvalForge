# ADR-003 — PostgreSQL + pgvector and Alembic

**Status:** Accepted

PostgreSQL is the system of record for dataset/model/KB versions, experiments, runs, predictions, metrics, jobs, artifacts, failures, and cost records. pgvector keeps retrieval provenance in the same versioned data system. Alembic owns every schema change, and migrations must work from an empty database.
