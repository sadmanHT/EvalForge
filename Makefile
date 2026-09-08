SHELL := /bin/bash
PYTHON ?= python
NPM ?= npm

.PHONY: format lint typecheck test test-integration test-e2e test-regression db-migrate smoke eval-smoke verify-all frontend-build secrets worker-smoke fresh-smoke test-phase3 dataset-audit dataset-rebuild-check phase3-contract phase3-exit phase3-source-fetch

format:
	cd backend && ruff check --fix app tests
	cd backend && ruff format app tests
	cd frontend && $(NPM) run format

lint:
	cd backend && ruff check app tests ../scripts/audit_dataset.py ../scripts/build_phase3_candidate.py ../scripts/check_phase3_contract.py ../scripts/check_phase3_exit.py ../scripts/check_phase3_reproducibility.py ../scripts/fetch_phase3_sources.py ../training/dataset_prep.py
	cd backend && ruff format --check app tests ../scripts/audit_dataset.py ../scripts/build_phase3_candidate.py ../scripts/check_phase3_contract.py ../scripts/check_phase3_exit.py ../scripts/check_phase3_reproducibility.py ../scripts/fetch_phase3_sources.py ../training/dataset_prep.py
	cd frontend && $(NPM) run format:check
	cd frontend && $(NPM) run lint

typecheck:
	cd backend && mypy app
	cd frontend && $(NPM) run typecheck

test:
	cd backend && pytest tests -m "not integration"
	cd frontend && $(NPM) test -- --run

test-phase3:
	cd backend && pytest tests/data/test_phase3_data.py --cov=app.data --cov-report=term-missing --cov-fail-under=90

test-integration:
	cd backend && pytest tests/integration -m integration

test-e2e:
	cd frontend && $(NPM) test -- --run src/App.test.tsx

test-regression:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) scripts/validate_phase1.py
	$(PYTHON) scripts/check_phase1_exit.py
	$(PYTHON) scripts/check_repo_foundation.py
	$(PYTHON) scripts/check_phase3_contract.py

db-migrate:
	cd backend && alembic upgrade head

dataset-audit:
	$(PYTHON) scripts/audit_dataset.py datasets/incident_diagnosis/fixtures/ci_smoke

dataset-rebuild-check:
	$(PYTHON) scripts/check_phase3_reproducibility.py

phase3-contract:
	$(PYTHON) scripts/check_phase3_contract.py

phase3-source-fetch:
	$(PYTHON) scripts/fetch_phase3_sources.py

phase3-exit:
	$(PYTHON) scripts/check_phase3_exit.py

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

verify-all: lint typecheck test test-phase3 test-integration test-e2e test-regression db-migrate dataset-audit dataset-rebuild-check smoke eval-smoke frontend-build secrets

fresh-smoke:
	docker compose down -v --remove-orphans
	docker compose build --no-cache
	docker compose up -d
	$(PYTHON) scripts/compose_health.py
	docker compose ps
	docker compose down -v --remove-orphans
