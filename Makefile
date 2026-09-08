SHELL := /bin/bash
PYTHON ?= python
NPM ?= npm

.PHONY: format lint typecheck test test-integration test-e2e test-regression db-migrate smoke eval-smoke verify-all frontend-build secrets worker-smoke fresh-smoke

format:
	cd backend && ruff check --fix app tests
	cd backend && ruff format app tests
	cd frontend && $(NPM) run format

lint:
	cd backend && ruff check app tests
	cd backend && ruff format --check app tests
	cd frontend && $(NPM) run format:check
	cd frontend && $(NPM) run lint

typecheck:
	cd backend && mypy app
	cd frontend && $(NPM) run typecheck

test:
	cd backend && pytest tests -m "not integration"
	cd frontend && $(NPM) test -- --run

test-integration:
	cd backend && pytest tests/integration -m integration

test-e2e:
	cd frontend && $(NPM) test -- --run src/App.test.tsx

test-regression:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) scripts/validate_phase1.py
	$(PYTHON) scripts/check_phase1_exit.py
	$(PYTHON) scripts/check_repo_foundation.py

db-migrate:
	cd backend && alembic upgrade head

smoke:
	cd backend && pytest tests/test_health.py tests/test_worker.py
	cd frontend && $(NPM) test -- --run src/App.test.tsx

eval-smoke:
	$(PYTHON) scripts/check_phase1_exit.py

frontend-build:
	cd frontend && $(NPM) run build

secrets:
	$(PYTHON) scripts/check_secrets.py

worker-smoke:
	$(PYTHON) scripts/worker_smoke.py

verify-all: lint typecheck test test-integration test-e2e test-regression db-migrate smoke eval-smoke frontend-build secrets

fresh-smoke:
	docker compose down -v --remove-orphans
	docker compose build --no-cache
	docker compose up -d
	$(PYTHON) scripts/compose_health.py
	docker compose ps
	docker compose down -v --remove-orphans
