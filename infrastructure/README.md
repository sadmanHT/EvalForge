# Infrastructure foundation

Phase 02 local development uses root `docker-compose.yml` with PostgreSQL/pgvector, Redis, backend, worker, and frontend services. Persistent development state lives in named Compose volumes.

Clean reset:

```bash
docker compose down -v --remove-orphans
docker compose build --no-cache
docker compose up -d
python scripts/compose_health.py
```

CI must use clean runner state and may not depend on manually created files outside the repository.
