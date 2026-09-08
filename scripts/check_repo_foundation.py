from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = {
    "README.md",
    "Makefile",
    "docker-compose.yml",
    ".env.example",
    ".gitignore",
    ".pre-commit-config.yaml",
    "backend/pyproject.toml",
    "backend/Dockerfile",
    "backend/alembic.ini",
    "backend/app/main.py",
    "backend/app/worker/runner.py",
    "frontend/package.json",
    "frontend/Dockerfile",
    "training/README.md",
    "evals/README.md",
    "infrastructure/README.md",
}
REQUIRED_SERVICES = {"postgres", "redis", "backend", "worker", "frontend"}
REQUIRED_TARGETS = {
    "format",
    "lint",
    "typecheck",
    "test",
    "test-integration",
    "test-e2e",
    "test-regression",
    "db-migrate",
    "smoke",
    "eval-smoke",
    "verify-all",
}
REQUIRED_JOB_STATES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED"}


def fail(message: str) -> None:
    raise SystemExit(f"PHASE02_FOUNDATION=FAIL: {message}")


def main() -> None:
    missing = sorted(path for path in REQUIRED_PATHS if not (ROOT / path).exists())
    if missing:
        fail(f"missing required paths: {missing}")

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in REQUIRED_TARGETS:
        if re.search(rf"^{re.escape(target)}\s*:", makefile, flags=re.MULTILINE) is None:
            fail(f"Makefile target missing: {target}")

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = set(compose.get("services", {}))
    if services != REQUIRED_SERVICES:
        fail(f"compose services mismatch: {sorted(services)}")
    for service in REQUIRED_SERVICES:
        if "healthcheck" not in compose["services"][service]:
            fail(f"compose service has no healthcheck: {service}")

    queue_source = (ROOT / "backend/app/worker/queue.py").read_text(encoding="utf-8")
    for state in REQUIRED_JOB_STATES:
        if f'{state} = "{state}"' not in queue_source:
            fail(f"worker state missing: {state}")

    print("PHASE02_FOUNDATION=PASS")
    print("COMPOSE_SERVICES=" + ",".join(sorted(REQUIRED_SERVICES)))
    print("MAKEFILE_TARGETS=" + ",".join(sorted(REQUIRED_TARGETS)))


if __name__ == "__main__":
    main()
