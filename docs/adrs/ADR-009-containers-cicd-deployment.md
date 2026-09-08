# ADR-009 — Docker Compose, CI/CD, and Provider-Independent Deployment

**Status:** Accepted

Local integration uses Docker Compose. CI/CD will use GitHub Actions with clean database/service setup, regression tests, and Docker builds. Deployment remains provider-independent: the application consumes standard `DATABASE_URL`, Redis URL, object/artifact credentials, and API secrets rather than provider-specific database logic. Railway, Fly.io, or an equivalent provider may be selected later by deployment ADR without changing core persistence contracts.
