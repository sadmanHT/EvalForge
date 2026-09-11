SHELL := /bin/bash
PYTHON ?= python
NPM ?= npm

.PHONY: format lint typecheck test test-integration test-e2e test-regression db-migrate smoke eval-smoke verify-all frontend-build secrets worker-smoke fresh-smoke test-phase3 test-phase4 test-phase5 test-phase6 dataset-audit dataset-rebuild-check phase3-contract phase3-exit phase3-source-fetch phase3-primary-source-check phase3-research-admission-check phase3-research-build phase3-research-rebuild-check phase4-seed phase4-exit phase5-smoke phase5-exit

PHASE_SCRIPTS := ../scripts/audit_dataset.py ../scripts/admit_phase3_research_records.py ../scripts/build_phase3_candidate.py ../scripts/build_phase3_research_dataset.py ../scripts/check_phase3_contract.py ../scripts/check_phase3_exit.py ../scripts/check_phase3_reproducibility.py ../scripts/check_phase3_research_reproducibility.py ../scripts/check_phase4_exit.py ../scripts/check_phase5_exit.py ../scripts/fetch_phase3_sources.py ../scripts/preserve_phase3_primary_sources.py ../scripts/run_phase5_smoke_evaluation.py ../scripts/sync_phase3_evidence.py ../training/dataset_prep.py ../evals/runner.py

format:
	cd backend && ruff check --fix app tests $(PHASE_SCRIPTS)
	cd backend && ruff format app tests $(PHASE_SCRIPTS)
	cd frontend && $(NPM) run format

lint:
	cd backend && ruff check app tests $(PHASE_SCRIPTS)
	cd backend && ruff format --check app tests $(PHASE_SCRIPTS)
	cd frontend && $(NPM) run format:check
	cd frontend && $(NPM) run lint

typecheck:
	cd backend && mypy app
	cd frontend && $(NPM) run typecheck

test:
	cd backend && pytest tests -m "not integration"
	cd frontend && $(NPM) test -- --run

test-phase3:
	cd backend && pytest tests/data/test_phase3_data.py tests/data/test_phase3_primary_sources.py tests/data/test_phase3_research_dataset.py --cov=app.data --cov-report=term-missing --cov-fail-under=90

test-phase4:
	cd backend && pytest tests/test_experiment_config.py --cov=app.core.experiment_config --cov-report=term-missing --cov-fail-under=90

test-phase5:
	cd backend && pytest tests/evaluation tests/integration/test_phase5_evaluation_persistence.py --cov=app.evaluation --cov-report=term-missing --cov-fail-under=90

test-phase6:
	cd backend && pytest tests/inference/test_phase6_base_model.py tests/inference/test_phase6_costing.py tests/inference/test_phase6_protocol.py tests/inference/test_phase6_tracking.py tests/integration/test_phase6_baseline_runner.py tests/integration/test_phase6_worker_api.py

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
	$(PYTHON) scripts/check_phase3_research_reproducibility.py

phase3-contract:
	$(PYTHON) scripts/check_phase3_contract.py

phase3-source-fetch:
	$(PYTHON) scripts/fetch_phase3_sources.py

phase3-primary-source-check:
	$(PYTHON) scripts/preserve_phase3_primary_sources.py

phase3-research-admission-check:
	$(PYTHON) scripts/admit_phase3_research_records.py

phase3-research-build:
	$(PYTHON) scripts/build_phase3_research_dataset.py

phase3-research-rebuild-check:
	$(PYTHON) scripts/check_phase3_research_reproducibility.py

phase3-exit:
	$(PYTHON) scripts/check_phase3_exit.py

phase4-seed:
	EVALFORGE_ROOT=. $(PYTHON) -m app.persistence_seed

phase4-exit:
	$(PYTHON) scripts/check_phase4_exit.py

phase5-smoke:
	$(PYTHON) scripts/run_phase5_smoke_evaluation.py

phase5-exit:
	$(PYTHON) scripts/check_phase5_exit.py

smoke:
	cd backend && pytest tests/test_health.py tests/test_worker.py
	cd frontend && $(NPM) test -- --run src/App.test.tsx

eval-smoke:
	$(PYTHON) scripts/check_phase1_exit.py
	$(PYTHON) scripts/run_phase5_smoke_evaluation.py

frontend-build:
	cd frontend && $(NPM) run build

secrets:
	$(PYTHON) scripts/check_secrets.py

worker-smoke:
	$(PYTHON) scripts/worker_smoke.py

verify-all: lint typecheck test test-phase3 test-phase4 test-phase5 test-phase6 test-integration test-e2e test-regression db-migrate dataset-audit dataset-rebuild-check phase3-primary-source-check phase3-research-admission-check phase3-exit phase4-seed phase4-exit smoke eval-smoke phase5-exit frontend-build secrets

fresh-smoke:
	docker compose down -v --remove-orphans
	docker compose build --no-cache
	docker compose up -d
	$(PYTHON) scripts/compose_health.py
	$(PYTHON) scripts/run_phase5_smoke_evaluation.py
	$(PYTHON) scripts/check_phase5_exit.py
	docker compose ps
	docker compose down -v --remove-orphans
