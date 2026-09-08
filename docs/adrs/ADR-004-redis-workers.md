# ADR-004 — Redis-backed Asynchronous Workers

**Status:** Accepted

Use Redis as queue/broker/progress infrastructure with one or more real worker processes. Redis is not authoritative experiment storage. Job retries must be idempotent; final job/prediction state belongs in PostgreSQL. This architecture is required before large evaluation or training jobs are exposed through the API.
