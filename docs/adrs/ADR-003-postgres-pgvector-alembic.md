# ADR-003 — PostgreSQL + pgvector + Alembic

**Status:** Accepted

## Context
EvalForge needs one durable system of record for experiments plus vector retrieval and version/provenance relationships.

## Decision
Use PostgreSQL as the durable data store, pgvector for vector columns/indexes, and Alembic for every relational schema change.

## Consequences
- experiments and retrieval provenance share transactional/versioned storage;
- all schema changes must have migrations;
- Phase 04 must prove migration from an empty database;
- Redis is never treated as durable research truth.
