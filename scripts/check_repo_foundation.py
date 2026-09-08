from __future__ import annotations

import hashlib
import json
import re
import tomllib
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
    ".github/workflows/ci.yml",
    "backend/pyproject.toml",
    "backend/requirements.full.lock",
    "backend/Dockerfile",
    "backend/alembic.ini",
    "backend/app/main.py",
    "backend/app/worker/runner.py",
    "frontend/package.json",
    "frontend/package-lock.json",
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
LOCK_LINE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s]+)$")
DIRECT_PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_.,-]+\])?==(?P<version>[^\s;]+)$"
)


def fail(message: str) -> None:
    raise SystemExit(f"PHASE02_FOUNDATION=FAIL: {message}")


def canonical_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_backend_lock() -> None:
    lock_path = ROOT / "backend/requirements.full.lock"
    lock_lines = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lock_lines:
        fail("backend dependency lock is empty")

    locked: dict[str, str] = {}
    for line in lock_lines:
        if "file://" in line or " @ " in line or line.startswith("-e "):
            fail(f"backend lock contains non-portable requirement: {line}")
        match = LOCK_LINE.fullmatch(line)
        if match is None:
            fail(f"backend lock entry is not an exact pin: {line}")
        name = canonical_package(match.group("name"))
        version = match.group("version")
        if name in locked:
            fail(f"backend lock contains duplicate package: {name}")
        locked[name] = version

    pyproject = tomllib.loads((ROOT / "backend/pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    direct_requirements = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        direct_requirements.extend(group)

    for requirement in direct_requirements:
        match = DIRECT_PIN.fullmatch(requirement)
        if match is None:
            fail(f"backend direct dependency is not exactly pinned: {requirement}")
        name = canonical_package(match.group("name"))
        expected = match.group("version")
        if locked.get(name) != expected:
            fail(f"backend lock mismatch for {name}: expected {expected}, got {locked.get(name)}")


def validate_frontend_lock() -> None:
    package = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "frontend/package-lock.json").read_text(encoding="utf-8"))
    if lock.get("lockfileVersion") != 3:
        fail("frontend package-lock.json must use lockfileVersion 3")
    root_package = lock.get("packages", {}).get("")
    if not isinstance(root_package, dict):
        fail("frontend package-lock.json is missing root package metadata")
    for field in ("name", "version", "dependencies", "devDependencies", "engines"):
        if root_package.get(field) != package.get(field):
            fail(f"frontend package-lock root metadata mismatch for {field}")


def validate_locked_consumption() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    backend_docker = (ROOT / "backend/Dockerfile").read_text(encoding="utf-8")
    frontend_docker = (ROOT / "frontend/Dockerfile").read_text(encoding="utf-8")

    workflow_markers = {
        "backend lock cache": "cache-dependency-path: backend/requirements.full.lock",
        "frontend lock cache": "cache-dependency-path: frontend/package-lock.json",
        "backend locked install": "python -m pip install -r backend/requirements.full.lock",
        "backend package no-deps install": "python -m pip install --no-deps -e ./backend",
        "frontend locked install": "npm ci --no-audit --no-fund",
        "backend lock verification": "diff -u backend/requirements.full.lock /tmp/backend.freeze",
    }
    for label, marker in workflow_markers.items():
        if marker not in workflow:
            fail(f"CI does not enforce {label}")

    backend_markers = {
        "backend lock copy": "COPY backend/requirements.full.lock ./requirements.full.lock",
        "backend locked Docker install": "pip install --no-cache-dir -r requirements.full.lock",
        "backend no-deps package install": "pip install --no-cache-dir --no-deps .",
    }
    for label, marker in backend_markers.items():
        if marker not in backend_docker:
            fail(f"backend Dockerfile does not enforce {label}")

    frontend_markers = {
        "frontend lock copy": "COPY frontend/package.json frontend/package-lock.json ./",
        "frontend locked Docker install": "RUN npm ci --no-audit --no-fund",
    }
    for label, marker in frontend_markers.items():
        if marker not in frontend_docker:
            fail(f"frontend Dockerfile does not enforce {label}")

    if "npm install --no-audit --no-fund" in workflow or "npm install --no-audit --no-fund" in frontend_docker:
        fail("floating frontend npm install is forbidden once package-lock.json is committed")
    if 'pip install -e "./backend[dev]"' in workflow:
        fail("floating backend editable dependency resolution is forbidden once the lock is committed")


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

    validate_backend_lock()
    validate_frontend_lock()
    validate_locked_consumption()

    print("PHASE02_FOUNDATION=PASS")
    print("COMPOSE_SERVICES=" + ",".join(sorted(REQUIRED_SERVICES)))
    print("MAKEFILE_TARGETS=" + ",".join(sorted(REQUIRED_TARGETS)))
    print("BACKEND_LOCK_SHA256=" + sha256(ROOT / "backend/requirements.full.lock"))
    print("FRONTEND_LOCK_SHA256=" + sha256(ROOT / "frontend/package-lock.json"))


if __name__ == "__main__":
    main()
